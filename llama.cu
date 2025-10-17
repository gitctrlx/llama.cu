#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>

#define CUDA_CHECK(val) { \
    cudaError_t err = val; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", err, cudaGetErrorString(err), __FILE__, __LINE__); \
        fflush(stderr); \
        exit(err); \
    } \
}

#define CUBLAS_CHECK(val) { \
    cublasStatus_t stat = val; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error %d. In file '%s' on line %d\n", stat, __FILE__, __LINE__); \
        fflush(stderr); \
        exit(stat); \
    } \
}

// Global cuBLAS handle
cublasHandle_t g_cublas_handle;

void create_cublas_handle() {
    CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
}

void destroy_cublas_handle() {
    CUBLAS_CHECK(cublasDestroy(g_cublas_handle));
}

// Transformer config
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
} Config;

// Weights on GPU
typedef struct {
    float* token_embedding;
    float* rms_att_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* rms_ffn_weight;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} TransformerWeights;

// Run state on GPU (except cpu logits)
typedef struct {
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    float* q;
    float* k;
    float* v;
    float* att;
    float* logits_gpu;
    float* logits; // CPU
    float* key_cache;
    float* value_cache;
} RunState;

// Transformer
typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

void malloc_run_state(RunState* s, const Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    CUDA_CHECK(cudaMalloc(&s->x, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->xb, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->xb2, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->hb, p->hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->hb2, p->hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->q, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->key_cache, (size_t)p->n_layers * p->max_seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->value_cache, (size_t)p->n_layers * p->max_seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->att, p->n_heads * p->max_seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->logits_gpu, p->vocab_size * sizeof(float)));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits) {
        fprintf(stderr, "cudaMalloc failed!\n");
        exit(EXIT_FAILURE);
    }
    s->k = NULL; // Set in forward
    s->v = NULL;
}

void free_run_state(RunState* s) {
    CUDA_CHECK(cudaFree(s->x));
    CUDA_CHECK(cudaFree(s->xb));
    CUDA_CHECK(cudaFree(s->xb2));
    CUDA_CHECK(cudaFree(s->hb));
    CUDA_CHECK(cudaFree(s->hb2));
    CUDA_CHECK(cudaFree(s->q));
    CUDA_CHECK(cudaFree(s->att));
    CUDA_CHECK(cudaFree(s->logits_gpu));
    free(s->logits);
    CUDA_CHECK(cudaFree(s->key_cache));
    CUDA_CHECK(cudaFree(s->value_cache));
}

void memory_map_weights(TransformerWeights* w, const Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    w->token_embedding = ptr;
    ptr += (size_t)p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->max_seq_len * head_size / 2; // skip freq_cis_real
    ptr += p->max_seq_len * head_size / 2; // skip freq_cis_imag
    w->wcls = shared_weights ? w->token_embedding : ptr;
}

void read_checkpoint(const char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE* file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    if (fread(config, sizeof(Config), 1, file) != 1) exit(EXIT_FAILURE);
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    float* weights_ptr;
    size_t weights_size = *file_size - sizeof(Config);
    CUDA_CHECK(cudaMalloc(&weights_ptr, weights_size));
    CUDA_CHECK(cudaMemcpy(weights_ptr, *data + sizeof(Config)/sizeof(float), weights_size, cudaMemcpyHostToDevice));
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer* t, const char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    if (t->data != MAP_FAILED) munmap(t->data, t->file_size);
    if (t->fd != -1) close(t->fd);
    CUDA_CHECK(cudaFree(t->weights.token_embedding));
    free_run_state(&t->state);
}

// Neural net blocks

int divUp(int a, int b) { return (a - 1) / b + 1; }

const int BLOCK_SIZE_LARGE = 256;
const int BLOCK_SIZE_SMALL = 128;

__global__ void rmsnorm_kernel(float* o, const float* x, const float* weight, int size) {
    int tid = threadIdx.x;
    typedef cub::BlockReduce<float, BLOCK_SIZE_LARGE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_ss;
    float ss = 0.0f;
    for (int j = tid; j < size; j += blockDim.x) {
        ss += x[j] * x[j];
    }
    ss = BlockReduce(temp).Sum(ss);
    if (tid == 0) {
        ss /= size;
        ss += 1e-5f;
        shared_ss = 1.0f / sqrtf(ss);
    }
    __syncthreads();
    ss = shared_ss;
    for (int j = tid; j < size; j += blockDim.x) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void rmsnorm(float* o, const float* x, const float* weight, int size) {
    rmsnorm_kernel<<<1, BLOCK_SIZE_LARGE, sizeof(float)>>>(o, x, weight, size);
    CUDA_CHECK(cudaGetLastError());
}

__device__ void softmax_gpu(float* x, int size) {
    int tid = threadIdx.x;
    int step = blockDim.x;
    typedef cub::BlockReduce<float, BLOCK_SIZE_LARGE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    float max_val = (tid < size) ? x[tid] : -INFINITY;
    for (int i = tid + step; i < size; i += step) {
        max_val = max(max_val, x[i]);
    }
    max_val = BlockReduce(temp).Reduce(max_val, cuda::maximum{});
    if (tid == 0) shared_val = max_val;
    __syncthreads();
    max_val = shared_val;
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    sum = BlockReduce(temp).Sum(sum);
    if (tid == 0) shared_val = sum;
    __syncthreads();
    sum = shared_val;
    for (int i = tid; i < size; i += step) {
        x[i] /= sum;
    }
}

// Matmul using cuBLAS (gemv)
void matmul(float* xout, const float* x, const float* w, int n, int d) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemv(g_cublas_handle, CUBLAS_OP_T, n, d, &alpha, w, n, x, 1, &beta, xout, 1));
}

__global__ void rope_rotation_kernel(int pos, float* sq, float* sk, int kv_dim, int head_size) {
    int i = threadIdx.x * 2;
    if (i >= head_size) return;
    float freq = 1.0f / powf(10000.0f, (float)(i % head_size) / head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = (i < kv_dim) ? 2 : 1;
    for (int v = 0; v < rotn; v++) {
        float* vec = (v == 0) ? sq : sk;
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

void rope_rotation(int pos, const RunState* s, int dim, int kv_dim, int head_size) {
    rope_rotation_kernel<<<1, (dim + 1) / 2>>>(pos, s->q, s->k, kv_dim, head_size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void multihead_attn_kernel(int pos, int seq_len, const float* sq, float* satt, float* sxb, const float* key_cache,
                                      const float* value_cache, int kv_dim, int kv_mul, int head_size, int loff) {
    int h = blockIdx.x;
    const float* q = sq + h * head_size;
    float* att = satt + h * seq_len;
    int tid = threadIdx.x;
    int step = blockDim.x;
    for (int t = tid; t <= pos; t += step) {
        const float* k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        att[t] = score / sqrtf((float)head_size);
    }
    __syncthreads();
    softmax_gpu(att, pos + 1);
    __syncthreads();
    float* xb = sxb + h * head_size;
    for (int i = tid; i < head_size; i += step) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float* v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            val += att[t] * v[i];
        }
        xb[i] = val;
    }
}

void multihead_attention(int pos, const Config* p, const RunState* s, int kv_dim, int kv_mul, int head_size, int loff) {
    int block_dim = 128;
    if (pos + 1 > 128) block_dim = 256;
    multihead_attn_kernel<<<p->n_heads, block_dim>>>(pos, p->max_seq_len, s->q, s->att, s->xb, s->key_cache, s->value_cache, kv_dim, kv_mul, head_size, loff);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void silu_mul_kernel(float* hb, const float* hb2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = hb[i];
        val *= 1.0f / (1.0f + expf(-val));
        hb[i] = val * hb2[i];
    }
}

void silu_mul(RunState* s, int hidden_dim) {
    silu_mul_kernel<<<divUp(hidden_dim, BLOCK_SIZE_SMALL), BLOCK_SIZE_SMALL>>>(s->hb, s->hb2, hidden_dim);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void accum_kernel(float* a, const float* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) a[i] += b[i];
}

void accum(float* a, const float* b, int size) {
    accum_kernel<<<divUp(size, BLOCK_SIZE_SMALL), BLOCK_SIZE_SMALL>>>(a, b, size);
    CUDA_CHECK(cudaGetLastError());
}

float* forward(Transformer* transformer, int token, int pos) {
    const Config* p = &transformer->config;
    const TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    float* content_row = w->token_embedding + (size_t)token * dim;
    CUDA_CHECK(cudaMemcpy(x, content_row, dim * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, w->rms_att_weight + (size_t)l * dim, dim);

        int loff = l * p->max_seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        matmul(s->q, s->xb, w->wq + (size_t)l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + (size_t)l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + (size_t)l * dim * kv_dim, dim, kv_dim);

        rope_rotation(pos, s, dim, kv_dim, head_size);

        multihead_attention(pos, p, s, kv_dim, kv_mul, head_size, loff);

        matmul(s->xb2, s->xb, w->wo + (size_t)l * dim * dim, dim, dim);

        accum(x, s->xb2, dim);

        rmsnorm(s->xb, x, w->rms_ffn_weight + (size_t)l * dim, dim);

        matmul(s->hb, s->xb, w->w1 + (size_t)l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + (size_t)l * dim * hidden_dim, dim, hidden_dim);

        silu_mul(s, hidden_dim);

        matmul(s->xb, s->hb, w->w2 + (size_t)l * hidden_dim * dim, hidden_dim, dim);

        accum(x, s->xb, dim);
    }

    rmsnorm(x, x, w->rms_final_weight, dim);

    matmul(s->logits_gpu, x, w->wcls, dim, p->vocab_size);

    CUDA_CHECK(cudaMemcpy(s->logits, s->logits_gpu, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    return s->logits;
}

// Tokenizer (kept similar, minor cleans)

typedef struct {
    char* str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void* a, const void* b) {
    return strcmp(((const TokenIndex*)a)->str, ((const TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc((size_t)vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc((size_t)vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE* file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&len, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        t->vocab[i] = (char*)malloc((size_t)len + 1);
        if (fread(t->vocab[i], (size_t)len, 1, file) != 1) exit(EXIT_FAILURE);
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(const char* piece) {
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    unsigned char fbit = (unsigned char)piece[0];
    unsigned char sbit = (unsigned char)piece[1];
    switch (fbit) {
        case 0xC3: printf("%c", sbit | 0x40); break;
        case 0xC2: printf("%c", sbit); break;
        default: printf("%s", piece); break;
    }
}

int str_lookup(const char* str, const TokenIndex* sorted_vocab, int vocab_size) {
    TokenIndex tok = {.str = (char*)str};
    const TokenIndex* res = (const TokenIndex*)bsearch(&tok, sorted_vocab, (size_t)vocab_size, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

void encode(Tokenizer* t, const char* text, int8_t bos, int8_t eos, int* tokens, int* n_tokens) {
    if (!text) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }
    if (!t->sorted_vocab) {
        t->sorted_vocab = (TokenIndex*)malloc((size_t)t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i] = {.str = t->vocab[i], .id = i};
        }
        qsort(t->sorted_vocab, (size_t)t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char* str_buffer = (char*)malloc((size_t)t->max_token_length*2 + 3);
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    for (const char* c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }
    while (true) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;
        for (int i = 0; i < *n_tokens - 1; i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < *n_tokens - 1; i++) tokens[i] = tokens[i+1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// Sampler
int argmax(const float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

// Time util
long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// Generate
void generate(Transformer* transformer, Tokenizer* tokenizer, const char* prompt, int max_new_tokens) {
    if (!prompt) prompt = "";
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, (char*)prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (prompt_tokens[1] == 306) prompt_tokens[1] = 76; // Monkey patch
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }
    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < max_new_tokens - 1) {
        float* logits = forward(transformer, token, pos);
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = argmax(logits, transformer->config.vocab_size);
        }
        pos++;
        if (next == 1) break;
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
        if (start == 0) start = time_in_ms();
    }
    printf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        double elapsed = (double)(end - start) / 1000.0;
        fprintf(stderr, "Token count: %d, elapsed: %.2fs, %.0f tokens/s\n",
                pos + 1, elapsed, (double)(pos - 1) / elapsed);
    }
    free(prompt_tokens);
}

int main(int argc, char** argv) {
    const char* checkpoint_path = "stories15M.bin";
    const char* tokenizer_path = "tokenizer.bin";
    int max_new_tokens = 50;
    const char* prompt = "I have a dream";
    if (argc >= 2) prompt = argv[1];
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (max_new_tokens > transformer.config.max_seq_len) max_new_tokens = transformer.config.max_seq_len;
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    create_cublas_handle();
    generate(&transformer, &tokenizer, prompt, max_new_tokens);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    destroy_cublas_handle();
    return 0;
}
