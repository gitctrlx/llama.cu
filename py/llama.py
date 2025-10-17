from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tokenizer import Tokenizer  # Assuming this is a custom module


@dataclass
class ModelArgs:
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    max_seq_len: int = 256
    max_new_tokens: int = 50
    norm_eps: float = 1e-6
    max_batch_size: int = 1


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1 / (1 + np.exp(-x)))


def compute_cos_sin_cache(
    head_dim: int, max_seq_len: int, base: int = 10000
) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (
        base
        ** (np.arange(0, head_dim, 2, dtype=np.float32)[: (head_dim // 2)] / head_dim)
    )
    t = np.arange(max_seq_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)
    return np.cos(freqs), np.sin(freqs)


def apply_rotary_emb(
    xq: np.ndarray,
    xk: np.ndarray,
    freqs_cos: np.ndarray,
    freqs_sin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Reshape for complex representation
    xq_ri = xq.reshape(*xq.shape[:-1], -1, 2)
    xk_ri = xk.reshape(*xk.shape[:-1], -1, 2)

    xq_r, xq_i = xq_ri[..., 0], xq_ri[..., 1]
    xk_r, xk_i = xk_ri[..., 0], xk_ri[..., 1]

    # Expand freqs for broadcasting
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))

    # Apply rotation
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Stack and reshape back
    xq_out = np.stack([xq_out_r, xq_out_i], axis=-1).reshape(*xq_out_r.shape[:-1], -1)
    xk_out = np.stack([xk_out_r, xk_out_i], axis=-1).reshape(*xk_out_r.shape[:-1], -1)

    return xq_out, xk_out


def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=2)


class FeedForward:
    def __init__(
        self,
        up_weight: np.ndarray,
        gate_weight: np.ndarray,
        down_weight: np.ndarray,
    ):
        self.up_weight = up_weight.T
        self.gate_weight = gate_weight.T
        self.down_weight = down_weight.T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        swish = silu(x @ self.gate_weight)
        x_v = x @ self.up_weight
        ff = swish * x_v
        return ff @ self.down_weight


class RMSNorm:
    def __init__(self, weight: np.ndarray, eps: float):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        variance = np.mean(x**2, axis=-1, keepdims=True) + self.eps
        normalized = x / np.sqrt(variance)
        return normalized * self.weight


class Attention:
    def __init__(
        self,
        q_weight: np.ndarray,
        k_weight: np.ndarray,
        v_weight: np.ndarray,
        o_weight: np.ndarray,
        args: ModelArgs,
    ):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_rep = args.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.q_weight = q_weight.T
        self.k_weight = k_weight.T
        self.v_weight = v_weight.T
        self.o_weight = o_weight.T

        self.cache_k = np.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )
        self.cache_v = np.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )

    def __call__(
        self,
        x: np.ndarray,
        start_pos: int,
        mask: Optional[np.ndarray],
        freqs_cos: np.ndarray,
        freqs_sin: np.ndarray,
    ) -> np.ndarray:
        bsz, seqlen, _ = x.shape

        # QKV projections
        xq = x @ self.q_weight
        xk = x @ self.k_weight
        xv = x @ self.v_weight

        # Reshape to [bsz, seqlen, n_heads, head_dim]
        xq = xq.reshape(bsz, seqlen, args.n_heads, self.head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Update KV cache
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # Retrieve from cache
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # Group query attention (repeat KV if necessary)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Transpose for attention: [bsz, n_heads, seqlen, head_dim]
        xq = xq.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Compute attention scores
        scores = (xq @ keys.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask[None, None, :, :]
        attn = softmax(scores)

        # Attention output
        output = attn @ values

        # Reshape back and project
        output = output.transpose(0, 2, 1, 3).reshape(bsz, seqlen, -1)
        return output @ self.o_weight


class TransformerBlock:
    def __init__(self, weights: dict[str, np.ndarray], layer_id: int, args: ModelArgs):
        prefix = f"model.layers.{layer_id}."
        self.attention = Attention(
            weights[f"{prefix}self_attn.q_proj.weight"],
            weights[f"{prefix}self_attn.k_proj.weight"],
            weights[f"{prefix}self_attn.v_proj.weight"],
            weights[f"{prefix}self_attn.o_proj.weight"],
            args,
        )
        self.feed_forward = FeedForward(
            weights[f"{prefix}mlp.up_proj.weight"],
            weights[f"{prefix}mlp.gate_proj.weight"],
            weights[f"{prefix}mlp.down_proj.weight"],
        )
        self.input_layernorm = RMSNorm(
            weights[f"{prefix}input_layernorm.weight"], eps=args.norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[f"{prefix}post_attention_layernorm.weight"], eps=args.norm_eps
        )

    def __call__(
        self,
        x: np.ndarray,
        start_pos: int,
        mask: Optional[np.ndarray],
        freqs_cos: np.ndarray,
        freqs_sin: np.ndarray,
    ) -> np.ndarray:
        norm_x = self.input_layernorm(x)
        attn_out = self.attention(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        h = x + attn_out

        norm_h = self.post_attention_layernorm(h)
        ff_out = self.feed_forward(norm_h)
        return h + ff_out


class Llama:
    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args
        weights = np.load(model_path)
        self.tok_embeddings = weights["model.embed_tokens.weight"]

        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(
            args.dim // args.n_heads, args.max_seq_len
        )

        self.layers = [
            TransformerBlock(weights, layer_id, args)
            for layer_id in range(args.n_layers)
        ]

        self.norm = RMSNorm(weights["model.norm.weight"], eps=args.norm_eps)
        self.output = weights["lm_head.weight"].T

        del weights

    def forward(self, tokens: np.ndarray, start_pos: int) -> np.ndarray:
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings[tokens]

        freqs_cos = self.freqs_cos[start_pos : start_pos + seqlen]
        freqs_sin = self.freqs_sin[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = np.full((seqlen, seqlen), float("-inf"))
            mask = np.triu(mask, k=1)
            mask = np.hstack([np.zeros((seqlen, start_pos)), mask])

        for layer in self.layers:
            h = layer(h, start_pos, mask, freqs_cos, freqs_sin)

        h = self.norm(h)
        return h[:, -1:, :] @ self.output  # Only last token's logits

    def generate(self, tokens: np.ndarray, max_new_tokens: int) -> np.ndarray:
        _, seqlen = tokens.shape
        for cur_pos in range(seqlen, max_new_tokens + seqlen):
            if cur_pos == seqlen:
                logits = self.forward(tokens, 0)
            else:
                logits = self.forward(next_token, cur_pos - 1)
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            yield next_token


if __name__ == "__main__":
    args = ModelArgs()
    tokenizer = Tokenizer("./tokenizer.np")
    model = Llama("./stories15M.npz", args)

    prompt = sys.argv[1] if len(sys.argv) > 1 else "I have a dream"
    print(f"\n{prompt}", end="", flush=True)

    input_tokens = np.array([tokenizer.encode(prompt)])
    start_time = time.time()
    generated_len = input_tokens.shape[1]

    for token_id in model.generate(input_tokens, args.max_new_tokens):
        generated_len += 1
        decoded = tokenizer.decode(token_id[0].tolist())
        if token_id[0, 0] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(decoded, end="", flush=True)

    elapsed = time.time() - start_time
    print(
        f"\n\nToken count: {generated_len}, elapsed: {elapsed:.2f}s, {generated_len / elapsed:.0f} tokens/s"
    )
