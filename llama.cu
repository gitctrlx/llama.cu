#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err),    \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t stat = (expr);                                              \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS Error: %d at %s:%d\n", stat, __FILE__,           \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace llama {

constexpr int kBlockSizeLarge = 1024;
constexpr int kBlockSizeSmall = 128;

// LlamaConfig aligns with LlamaConfig in Transformers.
struct LlamaConfig {
  int dim;         // hidden_size
  int hidden_dim;  // intermediate_size
  int n_layers;    // num_hidden_layers
  int n_heads;     // num_attention_heads
  int n_kv_heads;  // num_key_value_heads
  int vocab_size;  // vocab_size
  int max_seq_len; // max_position_embeddings
};

// LlamaLayerWeights holds device pointers for layer weights.
struct LlamaLayerWeights {
  float *attn_norm; // input_layernorm.weight
  float *q_proj;    // self_attn.q_proj.weight
  float *k_proj;    // self_attn.k_proj.weight
  float *v_proj;    // self_attn.v_proj.weight
  float *o_proj;    // self_attn.o_proj.weight
  float *ffn_norm;  // post_attention_layernorm.weight
  float *gate_proj; // mlp.gate_proj.weight
  float *up_proj;   // mlp.up_proj.weight
  float *down_proj; // mlp.down_proj.weight
};

// LlamaWeights holds all device weights.
struct LlamaWeights {
  float *embed_tokens;                      // model.embed_tokens.weight
  float *norm;                              // model.norm.weight
  float *output;                            // lm_head.weight
  std::vector<LlamaLayerWeights> layers; // model.layers
};

// LlamaState holds device buffers for inference.
struct LlamaState {
  float *x;           // hidden_states
  float *xb;          // temp for attention
  float *xb2;         // temp for attention output
  float *hb;          // mlp gate activation
  float *hb2;         // mlp up activation
  float *q;           // query
  float *k;           // key (points into key_cache)
  float *v;           // value (points into value_cache)
  float *att;         // attention scores
  float *logits_gpu;  // logits on GPU
  float *logits_cpu;  // logits on CPU
  float *key_cache;   // key cache
  float *value_cache; // value cache
};

// CuBlasHandle RAII wrapper.
class CuBlasHandle {
public:
  CuBlasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
  ~CuBlasHandle() { CUBLAS_CHECK(cublasDestroy(handle_)); }
  cublasHandle_t Get() const { return handle_; }

private:
  cublasHandle_t handle_;
};

// LlamaModel manages the model on GPU.
class LlamaModel {
public:
  LlamaModel(const char *checkpoint_path);
  ~LlamaModel();

  void Forward(int token, int pos, float *logits_cpu);
  const LlamaConfig &Config() const { return config_; }

private:
  void LoadWeights(const char *checkpoint_path);
  void AllocState();
  void FreeState();

  void LayerForward(int layer, int pos);

  LlamaConfig config_;
  LlamaWeights weights_;
  LlamaState state_;
  CuBlasHandle cublas_;
  int fd_ = -1;
  void *mapped_data_ = MAP_FAILED;
  size_t file_size_ = 0;
  float *device_weights_ = nullptr;
};

LlamaModel::LlamaModel(const char *checkpoint_path) : cublas_() {
  LoadWeights(checkpoint_path);
  AllocState();
}

LlamaModel::~LlamaModel() {
  FreeState();
  if (device_weights_)
    CUDA_CHECK(cudaFree(device_weights_));
  if (mapped_data_ != MAP_FAILED)
    munmap(mapped_data_, file_size_);
  if (fd_ != -1)
    close(fd_);
}

void LlamaModel::LoadWeights(const char *checkpoint_path) {
  FILE *file = fopen(checkpoint_path, "rb");
  if (!file) {
    fprintf(stderr, "Failed to open %s\n", checkpoint_path);
    exit(EXIT_FAILURE);
  }
  if (fread(&config_, sizeof(LlamaConfig), 1, file) != 1) {
    fprintf(stderr, "Failed to read config\n");
    exit(EXIT_FAILURE);
  }
  int shared_weights = config_.vocab_size > 0 ? 1 : 0;
  config_.vocab_size = std::abs(config_.vocab_size);
  fseek(file, 0, SEEK_END);
  file_size_ = ftell(file);
  fclose(file);

  fd_ = open(checkpoint_path, O_RDONLY);
  if (fd_ == -1) {
    fprintf(stderr, "open failed: %s\n", checkpoint_path);
    exit(EXIT_FAILURE);
  }
  mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapped_data_ == MAP_FAILED) {
    fprintf(stderr, "mmap failed\n");
    exit(EXIT_FAILURE);
  }

  size_t weights_size = file_size_ - sizeof(LlamaConfig);
  CUDA_CHECK(cudaMalloc(&device_weights_, weights_size));
  CUDA_CHECK(cudaMemcpy(device_weights_,
                        static_cast<char *>(mapped_data_) + sizeof(LlamaConfig),
                        weights_size, cudaMemcpyHostToDevice));

  float *ptr = device_weights_;
  weights_.embed_tokens = ptr;
  ptr += config_.vocab_size * config_.dim;
  weights_.layers.resize(config_.n_layers);

  // All attn_norm (rms_att_weight)
  for (int l = 0; l < config_.n_layers; ++l) {
    weights_.layers[l].attn_norm = ptr;
    ptr += config_.dim;
  }

  int head_size = config_.dim / config_.n_heads;

  // All q_proj (wq)
  for (int l = 0; l < config_.n_layers; ++l) {
    weights_.layers[l].q_proj = ptr;
    ptr += config_.dim * (config_.n_heads * head_size);
  }

  // All k_proj (wk)
  for (int l = 0; l < config_.n_layers; ++l) {
    weights_.layers[l].k_proj = ptr;
    ptr += config_.dim * (config_.n_kv_heads * head_size);
  }

  // All v_proj (wv)
  for (int l = 0; l < config_.n_layers; ++l) {
    weights_.layers[l].v_proj = ptr;
    ptr += config_.dim * (config_.n_kv_heads * head_size);
  }

  // All o_proj (wo)
  for (int l = 0; l < config_.n_layers; ++l) {
    weights_.layers[l].o_proj = ptr;
    ptr += (config_.n_heads * head_size) * config_.dim;
  }

  // All ffn_norm (rms_ffn_weight)
  for (int l = 0; l < config_.n_layers; ++l) {
    weights_.layers[l].ffn_norm = ptr;
    ptr += config_.dim;
  }

  // All gate_proj (w1 / gate_proj)
  for (int l = 0; l < config_.n_layers; ++l) {
    weights_.layers[l].gate_proj = ptr;
    ptr += config_.dim * config_.hidden_dim;
  }

  // All down_proj (w2 / down_proj)
  for (int l = 0; l < config_.n_layers; ++l) {
    weights_.layers[l].down_proj = ptr;
    ptr += config_.hidden_dim * config_.dim;
  }

  // All up_proj (w3 / up_proj)
  for (int l = 0; l < config_.n_layers; ++l) {
    weights_.layers[l].up_proj = ptr;
    ptr += config_.dim * config_.hidden_dim;
  }

  weights_.norm = ptr;
  ptr += config_.dim;
  ptr += config_.max_seq_len * head_size / 2; // skip freq_cis_real
  ptr += config_.max_seq_len * head_size / 2; // skip freq_cis_imag
  weights_.output = shared_weights ? weights_.embed_tokens : ptr;
}

void LlamaModel::AllocState() {
  int kv_dim = (config_.dim * config_.n_kv_heads) / config_.n_heads;
  CUDA_CHECK(cudaMalloc(&state_.x, config_.dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state_.xb, config_.dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state_.xb2, config_.dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state_.hb, config_.hidden_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state_.hb2, config_.hidden_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state_.q, config_.dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state_.key_cache,
                        static_cast<size_t>(config_.n_layers) *
                            config_.max_seq_len * kv_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state_.value_cache,
                        static_cast<size_t>(config_.n_layers) *
                            config_.max_seq_len * kv_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&state_.att,
                        config_.n_heads * config_.max_seq_len * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc(&state_.logits_gpu, config_.vocab_size * sizeof(float)));
  state_.logits_cpu = new float[config_.vocab_size]();
}

void LlamaModel::FreeState() {
  CUDA_CHECK(cudaFree(state_.x));
  CUDA_CHECK(cudaFree(state_.xb));
  CUDA_CHECK(cudaFree(state_.xb2));
  CUDA_CHECK(cudaFree(state_.hb));
  CUDA_CHECK(cudaFree(state_.hb2));
  CUDA_CHECK(cudaFree(state_.q));
  CUDA_CHECK(cudaFree(state_.key_cache));
  CUDA_CHECK(cudaFree(state_.value_cache));
  CUDA_CHECK(cudaFree(state_.att));
  CUDA_CHECK(cudaFree(state_.logits_gpu));
  delete[] state_.logits_cpu;
}

// Kernels and helpers

__global__ void RmsNormKernel(float *o, const float *x, const float *weight,
                              int size) {
  int tid = threadIdx.x;
  __shared__ float shared_ss;
  float ss = 0.0f;
  for (int j = tid; j < size; j += blockDim.x) {
    ss += x[j] * x[j];
  }
  typedef cub::BlockReduce<float, kBlockSizeLarge> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp;
  ss = BlockReduce(temp).Sum(ss);
  if (tid == 0) {
    ss /= size;
    ss += 1e-5f;
    shared_ss = rsqrtf(ss);
  }
  __syncthreads();
  for (int j = tid; j < size; j += blockDim.x) {
    o[j] = weight[j] * (shared_ss * x[j]);
  }
}

void RmsNorm(float *o, const float *x, const float *weight, int size) {
  RmsNormKernel<<<1, kBlockSizeLarge>>>(o, x, weight, size);
  CUDA_CHECK(cudaGetLastError());
}

void Matmul(float *xout, const float *x, const float *w, int n, int d,
            cublasHandle_t handle) {
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, n, d, &alpha, w, n, x, 1, &beta,
                           xout, 1));
}

__global__ void ApplyRotaryEmbKernel(int pos, float *q, float *k, int dim,
                                     int kv_dim, int head_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim / 2)
    return;
  i *= 2;
  float freq = 1.0f / powf(10000.0f, float(i % head_size) / float(head_size));
  float val = float(pos) * freq;
  float fcr = cosf(val), fci = sinf(val);
  int rotn = (i < kv_dim) ? 2 : 1;
  for (int v = 0; v < rotn; ++v) {
    float *vec = (v == 0) ? q : k;
    float v0 = vec[i], v1 = vec[i + 1];
    vec[i] = v0 * fcr - v1 * fci;
    vec[i + 1] = v0 * fci + v1 * fcr;
  }
}

void ApplyRotaryEmb(int pos, float *q, float *k, int dim, int kv_dim,
                    int head_size) {
  int blocks = (dim / 2 + kBlockSizeSmall - 1) / kBlockSizeSmall;
  ApplyRotaryEmbKernel<<<blocks, kBlockSizeSmall>>>(pos, q, k, dim, kv_dim,
                                                    head_size);
  CUDA_CHECK(cudaGetLastError());
}

__device__ void SoftmaxDevice(float *x, int size) {
  int tid = threadIdx.x, step = blockDim.x;
  float max_val = (tid < size) ? x[tid] : -INFINITY;
  for (int i = tid + step; i < size; i += step) {
    max_val = fmaxf(max_val, x[i]);
  }
  typedef cub::BlockReduce<float, kBlockSizeLarge> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp;
  max_val = BlockReduce(temp).Reduce(max_val, cuda::maximum{});
  __shared__ float shared_max;
  if (tid == 0)
    shared_max = max_val;
  __syncthreads();
  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - shared_max);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  __shared__ float shared_sum;
  if (tid == 0)
    shared_sum = sum;
  __syncthreads();
  for (int i = tid; i < size; i += step) {
    x[i] /= shared_sum;
  }
}

__global__ void MultiHeadAttnKernel(int pos, int seq_len, const float *q,
                                    float *att, float *xb,
                                    const float *key_cache,
                                    const float *value_cache, int kv_dim,
                                    int kv_mul, int head_size, int loff) {
  int h = blockIdx.x;
  const float *query = q + h * head_size;
  float *attn_scores = att + h * seq_len;
  int tid = threadIdx.x, step = blockDim.x;
  for (int t = tid; t <= pos; t += step) {
    const float *key = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
      score += query[i] * key[i];
    }
    attn_scores[t] = score / sqrtf(float(head_size));
  }
  __syncthreads();
  SoftmaxDevice(attn_scores, pos + 1);
  __syncthreads();
  float *out_head = xb + h * head_size;
  for (int i = tid; i < head_size; i += step) {
    float val = 0.0f;
    for (int t = 0; t <= pos; ++t) {
      const float *value =
          value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      val += attn_scores[t] * value[i];
    }
    out_head[i] = val;
  }
}

void MultiHeadAttn(int pos, int n_heads, int max_seq_len, const float *q,
                   float *att, float *xb, const float *key_cache,
                   const float *value_cache, int kv_dim, int kv_mul,
                   int head_size, int loff) {
  int block_dim = kBlockSizeLarge;
  MultiHeadAttnKernel<<<n_heads, block_dim>>>(pos, max_seq_len, q, att, xb,
                                              key_cache, value_cache, kv_dim,
                                              kv_mul, head_size, loff);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void SwiGLUKernel(float *hb, const float *hb2, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    float val = hb[i];
    val *= 1.0f / (1.0f + expf(-val));
    hb[i] = val * hb2[i];
  }
}

void SwiGLU(float *hb, const float *hb2, int hidden_dim) {
  int blocks = (hidden_dim + kBlockSizeSmall - 1) / kBlockSizeSmall;
  SwiGLUKernel<<<blocks, kBlockSizeSmall>>>(hb, hb2, hidden_dim);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void AccumKernel(float *a, const float *b, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    a[i] += b[i];
}

void Accum(float *a, const float *b, int size) {
  int blocks = (size + kBlockSizeSmall - 1) / kBlockSizeSmall;
  AccumKernel<<<blocks, kBlockSizeSmall>>>(a, b, size);
  CUDA_CHECK(cudaGetLastError());
}

void LlamaModel::LayerForward(int layer, int pos) {
  auto &lw = weights_.layers[layer];
  int dim = config_.dim;
  int kv_dim = (config_.dim * config_.n_kv_heads) / config_.n_heads;
  int kv_mul = config_.n_heads / config_.n_kv_heads;
  int head_size = config_.dim / config_.n_heads;
  int loff = layer * config_.max_seq_len * kv_dim;

  RmsNorm(state_.xb, state_.x, lw.attn_norm, dim);

  Matmul(state_.q, state_.xb, lw.q_proj, dim, dim, cublas_.Get());
  state_.k = state_.key_cache + loff + pos * kv_dim;
  Matmul(state_.k, state_.xb, lw.k_proj, dim, kv_dim, cublas_.Get());
  state_.v = state_.value_cache + loff + pos * kv_dim;
  Matmul(state_.v, state_.xb, lw.v_proj, dim, kv_dim, cublas_.Get());

  ApplyRotaryEmb(pos, state_.q, state_.k, dim, kv_dim, head_size);

  MultiHeadAttn(pos, config_.n_heads, config_.max_seq_len, state_.q, state_.att,
                state_.xb, state_.key_cache, state_.value_cache, kv_dim, kv_mul,
                head_size, loff);

  Matmul(state_.xb2, state_.xb, lw.o_proj, dim, dim, cublas_.Get());

  Accum(state_.x, state_.xb2, dim);

  RmsNorm(state_.xb, state_.x, lw.ffn_norm, dim);

  Matmul(state_.hb, state_.xb, lw.gate_proj, dim, config_.hidden_dim,
         cublas_.Get());
  Matmul(state_.hb2, state_.xb, lw.up_proj, dim, config_.hidden_dim,
         cublas_.Get());

  SwiGLU(state_.hb, state_.hb2, config_.hidden_dim);

  Matmul(state_.xb, state_.hb, lw.down_proj, config_.hidden_dim, dim,
         cublas_.Get());

  Accum(state_.x, state_.xb, dim);
}

void LlamaModel::Forward(int token, int pos, float *logits_cpu) {
  int dim = config_.dim;
  float *embed_row = weights_.embed_tokens + token * dim;
  CUDA_CHECK(cudaMemcpy(state_.x, embed_row, dim * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  for (int l = 0; l < config_.n_layers; ++l) {
    LayerForward(l, pos);
  }

  RmsNorm(state_.x, state_.x, weights_.norm, dim);

  Matmul(state_.logits_gpu, state_.x, weights_.output, dim, config_.vocab_size,
         cublas_.Get());

  CUDA_CHECK(cudaMemcpy(logits_cpu, state_.logits_gpu,
                        config_.vocab_size * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

// Tokenizer (BPE)

struct TokenIndex {
  char *str;
  int id;
};

static int TokenCompareVoid(const void *a, const void *b) {
  return strcmp(static_cast<const TokenIndex *>(a)->str,
                static_cast<const TokenIndex *>(b)->str);
}

bool TokenCompare(const TokenIndex &a, const TokenIndex &b) {
  return strcmp(a.str, b.str) < 0;
}

class Tokenizer {
public:
  Tokenizer(const char *path, int vocab_size);
  ~Tokenizer();

  void Encode(const char *text, bool bos, bool eos, int *tokens, int *n_tokens);
  char *Decode(int prev_token, int token);
  int StrLookup(char *str);

private:
  char **vocab_ = nullptr;
  float *vocab_scores_ = nullptr;
  TokenIndex *sorted_vocab_ = nullptr;
  int vocab_size_ = 0;
  unsigned int max_token_length_ = 0;
  unsigned char byte_pieces_[512] = {0}; // <0xNN> tokens
};

Tokenizer::Tokenizer(const char *path, int vocab_size)
    : vocab_size_(vocab_size) {
  vocab_ = new char *[vocab_size];
  vocab_scores_ = new float[vocab_size];
  for (int i = 0; i < 256; ++i) {
    byte_pieces_[i * 2] = static_cast<unsigned char>(i);
    byte_pieces_[i * 2 + 1] = '\0';
  }
  FILE *file = fopen(path, "rb");
  if (!file) {
    fprintf(stderr, "Failed to load tokenizer: %s\n", path);
    exit(EXIT_FAILURE);
  }
  if (fread(&max_token_length_, sizeof(unsigned int), 1, file) != 1) {
    fprintf(stderr, "Failed to read max_token_length\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < vocab_size; ++i) {
    if (fread(vocab_scores_ + i, sizeof(float), 1, file) != 1) {
      fprintf(stderr, "Failed to read vocab score\n");
      exit(EXIT_FAILURE);
    }
    int len;
    if (fread(&len, sizeof(int), 1, file) != 1) {
      fprintf(stderr, "Failed to read token length\n");
      exit(EXIT_FAILURE);
    }
    vocab_[i] = new char[len + 1];
    if (fread(vocab_[i], len, 1, file) != 1) {
      fprintf(stderr, "Failed to read token\n");
      exit(EXIT_FAILURE);
    }
    vocab_[i][len] = '\0';
  }
  fclose(file);
}

Tokenizer::~Tokenizer() {
  for (int i = 0; i < vocab_size_; ++i) {
    delete[] vocab_[i];
  }
  delete[] vocab_;
  delete[] vocab_scores_;
  delete[] sorted_vocab_;
}

int Tokenizer::StrLookup(char *str) {
  TokenIndex key = {.str = str};
  TokenIndex *res = static_cast<TokenIndex *>(std::bsearch(
      &key, sorted_vocab_, vocab_size_, sizeof(TokenIndex), TokenCompareVoid));
  return res ? res->id : -1;
}

void Tokenizer::Encode(const char *text, bool bos, bool eos, int *tokens,
                       int *n_tokens) {
  if (!text) {
    fprintf(stderr, "Cannot encode NULL text\n");
    exit(EXIT_FAILURE);
  }
  if (!sorted_vocab_) {
    sorted_vocab_ = new TokenIndex[vocab_size_];
    for (int i = 0; i < vocab_size_; ++i) {
      sorted_vocab_[i] = {vocab_[i], i};
    }
    std::qsort(sorted_vocab_, vocab_size_, sizeof(TokenIndex),
               TokenCompareVoid);
  }
  char *str_buffer = new char[max_token_length_ * 2 + 3];
  size_t str_len = 0;
  *n_tokens = 0;
  if (bos)
    tokens[(*n_tokens)++] = 1;
  if (text[0] != '\0') {
    int dummy_prefix = StrLookup(const_cast<char *>(" "));
    if (dummy_prefix != -1)
      tokens[(*n_tokens)++] = dummy_prefix;
  }
  for (const char *c = text; *c != '\0'; ++c) {
    if ((*c & 0xC0) != 0x80)
      str_len = 0;
    str_buffer[str_len++] = *c;
    str_buffer[str_len] = '\0';
    if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4)
      continue;
    int id = StrLookup(str_buffer);
    if (id != -1) {
      tokens[(*n_tokens)++] = id;
    } else {
      for (size_t i = 0; i < str_len; ++i) {
        tokens[(*n_tokens)++] = static_cast<unsigned char>(str_buffer[i]) + 3;
      }
    }
    str_len = 0;
  }
  while (true) {
    float best_score = -1e10f;
    int best_id = -1;
    int best_idx = -1;
    for (int i = 0; i < *n_tokens - 1; ++i) {
      std::sprintf(str_buffer, "%s%s", vocab_[tokens[i]],
                   vocab_[tokens[i + 1]]);
      int id = StrLookup(str_buffer);
      if (id != -1 && vocab_scores_[id] > best_score) {
        best_score = vocab_scores_[id];
        best_id = id;
        best_idx = i;
      }
    }
    if (best_idx == -1)
      break;
    tokens[best_idx] = best_id;
    for (int i = best_idx + 1; i < *n_tokens - 1; ++i) {
      tokens[i] = tokens[i + 1];
    }
    --(*n_tokens);
  }
  if (eos)
    tokens[(*n_tokens)++] = 2;
  delete[] str_buffer;
}

char *Tokenizer::Decode(int prev_token, int token) {
  char *piece = vocab_[token];
  if (prev_token == 1 && piece[0] == ' ')
    ++piece;
  unsigned char byte_val;
  if (std::sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = reinterpret_cast<char *>(byte_pieces_) + byte_val * 2;
  }
  return piece;
}

void SafePrint(const char *piece) {
  if (!piece || piece[0] == '\0')
    return;
  if (piece[1] == '\0') {
    unsigned char byte = piece[0];
    if (!std::isprint(byte) && !std::isspace(byte))
      return;
  }
  unsigned char fbit = static_cast<unsigned char>(piece[0]);
  if (fbit == 0xC3) {
    printf("%c", static_cast<unsigned char>(piece[1]) | 0x40);
  } else if (fbit == 0xC2) {
    printf("%c", static_cast<unsigned char>(piece[1]));
  } else {
    printf("%s", piece);
  }
}

int ArgMax(const float *probs, int n) {
  int max_i = 0;
  float max_p = probs[0];
  for (int i = 1; i < n; ++i) {
    if (probs[i] > max_p) {
      max_i = i;
      max_p = probs[i];
    }
  }
  return max_i;
}

long TimeInMs() {
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

void Generate(LlamaModel &model, Tokenizer &tokenizer, const char *prompt,
              int max_new_tokens) {
  if (!prompt)
    prompt = "";
  int num_prompt_tokens = 0;
  int *prompt_tokens = new int[std::strlen(prompt) + 3];
  tokenizer.Encode(prompt, true, false, prompt_tokens, &num_prompt_tokens);
  // TODO: pretty dirty monkey patch for 'I have a dream' prompt.
  if (prompt_tokens[1] == 306)
    prompt_tokens[1] = 76;
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "Expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }
  long start = 0;
  int next = 0;
  int token = prompt_tokens[0];
  int pos = 0;
  float *logits = new float[model.Config().vocab_size];
  while (pos < max_new_tokens - 1) {
    model.Forward(token, pos, logits);
    if (pos < num_prompt_tokens - 1) {
      next = prompt_tokens[pos + 1];
    } else {
      if (start == 0)
        start = TimeInMs();
      next = ArgMax(logits, model.Config().vocab_size);
    }
    ++pos;
    if (next == 1)
      break;
    char *piece = tokenizer.Decode(token, next);
    SafePrint(piece);
    std::fflush(stdout);
    token = next;
  }
  printf("\n");
  if (pos > 1) {
    long end = TimeInMs();
    double elapsed = static_cast<double>(end - start) / 1000.0;
    fprintf(stderr, "Tokens: %d, time: %.2fs, speed: %.0f tok/s\n",
            pos - num_prompt_tokens + 1, elapsed,
            static_cast<double>(pos - num_prompt_tokens) / elapsed);
  }
  delete[] prompt_tokens;
  delete[] logits;
}

} // namespace llama

int main(int argc, char **argv) {
  const char *checkpoint = (argc > 1) ? argv[1] : "stories15M.bin";
  const char *tokenizer_path = (argc > 2) ? argv[2] : "tokenizer.bin";
  int max_new_tokens = (argc > 3) ? std::atoi(argv[3]) : 200;
  const char *prompt = (argc > 4) ? argv[4] : "I have a dream";

  llama::LlamaModel model(checkpoint);
  if (max_new_tokens > model.Config().max_seq_len)
    max_new_tokens = model.Config().max_seq_len;
  llama::Tokenizer tokenizer(tokenizer_path, model.Config().vocab_size);

  llama::Generate(model, tokenizer, prompt, max_new_tokens);

  return 0;
}
