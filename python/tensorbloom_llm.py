"""LLM support for the TensorBloom sidecar.

Reads a large language model's architecture without downloading its weights:
from a Hugging Face model id (config.json plus the safetensors headers, fetched
with range requests), a local folder, a .safetensors file, or a .gguf file.
The result is a "spec": the handful of numbers that fix the architecture, which
the app draws as a graph and uses for memory estimates.

Also defines DecoderBlock, the Llama-style block the graph is drawn with, so an
imported architecture (or a small one built by hand) can be trained like any
other graph.

Only the standard library and torch are needed here. Loading weights and
fine-tuning (transformers, peft, bitsandbytes) live further down and import
those packages lazily, so the rest of the app works without them.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# DecoderBlock
# ---------------------------------------------------------------------------

# Source of the block, shared by the sidecar (exec'd below) and by generated
# code, so an exported model runs exactly what trained. app/src/lib/llm-block.ts
# holds a copy for the app's code view; test_llm.py checks the two match.
DECODER_BLOCK_SOURCE = '''class DecoderMLP(nn.Module):
    """Feed-forward part of a decoder block: gated (SwiGLU/GeGLU) or plain GELU."""

    def __init__(self, d_model, ffn_dim, mlp="swiglu", bias=False):
        super().__init__()
        self.gated = mlp in ("swiglu", "geglu")
        self.act = nn.SiLU() if mlp == "swiglu" else nn.GELU(approximate="tanh")
        if self.gated:
            self.gate_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=bias)

    def forward(self, x):
        h = self.act(self.gate_proj(x)) * self.up_proj(x) if self.gated else self.act(self.up_proj(x))
        return self.down_proj(h)


class DecoderBlock(nn.Module):
    """Pre-norm LLM decoder block, as in Llama, Mistral, Qwen and Gemma.

    x -> norm -> causal self-attention (rotary positions, grouped-query heads) -> + x
      -> norm -> MLP (gated, plain, or a mixture of experts)                  -> + x
    """

    def __init__(self, d_model=512, n_heads=8, n_kv_heads=8, head_dim=0, ffn_dim=1408,
                 mlp="swiglu", norm="rmsnorm", norm_eps=1e-5, rope_theta=10000.0,
                 bias="none", n_experts=0, experts_per_token=2):
        super().__init__()
        if n_heads % n_kv_heads:
            raise ValueError(f"n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = head_dim or d_model // n_heads
        self.rope_theta = rope_theta
        self.experts_per_token = experts_per_token
        norm_cls = nn.RMSNorm if norm == "rmsnorm" else nn.LayerNorm
        self.attn_norm = norm_cls(d_model, eps=norm_eps)
        self.mlp_norm = norm_cls(d_model, eps=norm_eps)
        qkv_bias = bias in ("qkv", "all")
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias == "all")
        if n_experts:
            self.router = nn.Linear(d_model, n_experts, bias=False)
            self.experts = nn.ModuleList(DecoderMLP(d_model, ffn_dim, mlp, bias == "all") for _ in range(n_experts))
        else:
            self.mlp = DecoderMLP(d_model, ffn_dim, mlp, bias == "all")

    def rotate(self, x):
        """Rotary position embedding: turn each pair of features by an angle set by the position."""
        t, d = x.size(-2), x.size(-1)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
        angles = torch.outer(torch.arange(t, device=x.device, dtype=torch.float32), inv_freq)
        angles = torch.cat([angles, angles], dim=-1)
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return x * angles.cos().to(x.dtype) + torch.cat([-x2, x1], dim=-1) * angles.sin().to(x.dtype)

    def attention(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.rope_theta:
            q, k = self.rotate(q), self.rotate(k)
        if self.n_kv_heads != self.n_heads:
            # Grouped-query attention: each key/value head serves several query heads
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))

    def feed_forward(self, x):
        if not hasattr(self, "router"):
            return self.mlp(x)
        # Mixture of experts: each token goes to its top-k experts, weighted by the router
        weights, chosen = self.router(x).softmax(dim=-1).topk(self.experts_per_token, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            hit = chosen == i
            tokens = hit.any(dim=-1)
            if tokens.any():
                w = (weights * hit).sum(dim=-1, keepdim=True)[tokens]
                out[tokens] = out[tokens] + expert(x[tokens]) * w
        return out

    def forward(self, x):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        x = x + self.attention(self.attn_norm(x))
        x = x + self.feed_forward(self.mlp_norm(x))
        return x.squeeze(0) if squeeze else x
'''

_block_namespace: dict[str, Any] = {"torch": torch, "nn": nn, "__name__": "tensorbloom_llm_blocks"}
exec(DECODER_BLOCK_SOURCE, _block_namespace)
DecoderBlock = _block_namespace["DecoderBlock"]
DecoderMLP = _block_namespace["DecoderMLP"]

DECODER_BLOCK_PARAMS = ("d_model", "n_heads", "n_kv_heads", "head_dim", "ffn_dim", "mlp", "norm",
                        "norm_eps", "rope_theta", "bias", "n_experts", "experts_per_token")


def build_decoder_block(params: dict) -> nn.Module:
    return DecoderBlock(**{k: v for k, v in params.items() if k in DECODER_BLOCK_PARAMS and v is not None})


# ---------------------------------------------------------------------------
# Spec: the numbers that fix an LLM's architecture
# ---------------------------------------------------------------------------

def count_params(spec: dict) -> int:
    """Parameter count implied by a spec (matches DecoderBlock plus embeddings and head)."""
    d, L, V = spec["d_model"], spec["n_layers"], spec["vocab_size"]
    H, KV = spec["n_heads"], spec["n_kv_heads"]
    hd = spec.get("head_dim") or d // H
    F = spec["ffn_dim"]
    bias = spec.get("bias", "none")
    gated = spec.get("mlp", "swiglu") in ("swiglu", "geglu")
    norm_params = d if spec.get("norm", "rmsnorm") == "rmsnorm" else 2 * d

    attn = d * H * hd + 2 * d * KV * hd + H * hd * d
    if bias in ("qkv", "all"):
        attn += H * hd + 2 * KV * hd
    if bias == "all":
        attn += d
    mlp = (3 if gated else 2) * d * F + ((2 * F if gated else F) + d if bias == "all" else 0)
    E = spec.get("n_experts") or 0
    if E:
        mlp = E * mlp + d * E
    block = attn + mlp + 2 * norm_params

    total = V * d + L * block + norm_params
    if not spec.get("tie_embeddings"):
        total += V * d
    if spec.get("learned_positions"):
        total += spec.get("max_context", 0) * d
    return total


def _first(cfg: dict, *keys, default=None):
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            return v
    return default


def spec_from_hf_config(cfg: dict) -> dict:
    """Spec from a Hugging Face config.json (Llama, Mistral, Qwen, Gemma, Phi, GPT-2, Mixtral, ...)."""
    warnings: list[str] = []
    top = cfg
    # Multimodal models keep the language model's settings in text_config
    if isinstance(cfg.get("text_config"), dict) and not cfg.get("hidden_size"):
        cfg = {**cfg["text_config"], "model_type": cfg["text_config"].get("model_type") or cfg.get("model_type")}
        warnings.append("Only the language model is shown; the vision or audio encoder is left out.")
    family = str(cfg.get("model_type") or "unknown")
    d = int(_first(cfg, "hidden_size", "n_embd", "d_model", "dim"))
    H = int(_first(cfg, "num_attention_heads", "n_head", "n_heads"))
    KV = _first(cfg, "num_key_value_heads", "n_kv_heads", "multi_query_group_num")
    if KV is None:
        KV = 1 if cfg.get("multi_query") else H
    gpt2_like = family in ("gpt2", "gpt_neox", "gptj", "gpt_bigcode", "phi", "opt", "bloom", "falcon")
    act = str(_first(cfg, "hidden_act", "hidden_activation", "activation_function", "activation", default="silu"))
    if gpt2_like:
        mlp = "gelu"
    elif family.startswith("gemma") or "gelu" in act:
        mlp = "geglu"
    else:
        mlp = "swiglu"
    F = _first(cfg, "intermediate_size", "n_inner", "ffn_dim", "ffn_hidden_size")
    if F is None:
        F = 4 * d
    # Mixture-of-experts models often give each expert its own (smaller) width
    E = int(_first(cfg, "num_local_experts", "num_experts", "n_routed_experts", default=0) or 0)
    if E:
        F = _first(cfg, "moe_intermediate_size", default=F)
        if cfg.get("n_shared_experts") or cfg.get("shared_expert_intermediate_size"):
            warnings.append("Shared experts aren't drawn, so the parameter count is a little low.")
    rope_theta = float(_first(cfg, "rope_theta", "rotary_emb_base", default=0.0 if gpt2_like else 10000.0))
    if family in ("gpt_neox", "phi") and cfg.get("rotary_pct", cfg.get("partial_rotary_factor", 1)) not in (1, 1.0):
        warnings.append("Rotary positions apply to only part of each head in this model; the graph rotates all of it.")
    learned_positions = family in ("gpt2", "opt", "gpt_bigcode") or (gpt2_like and rope_theta == 0)
    if family == "gpt2" or family in ("opt", "bloom", "falcon", "gpt_bigcode", "gptj", "phi", "gpt_neox"):
        bias = "all"
    elif family.startswith("qwen2") or cfg.get("attention_bias") or cfg.get("qkv_bias"):
        bias = "qkv" if not cfg.get("mlp_bias") else "all"
    else:
        bias = "none"
    norm = "layernorm" if gpt2_like or "layer_norm_epsilon" in cfg and "rms_norm_eps" not in cfg else "rmsnorm"
    window = cfg.get("sliding_window")
    context = _first(cfg, "max_position_embeddings", "n_positions", "seq_length", "max_seq_len", default=0) or 0
    if window and cfg.get("use_sliding_window", True) and window < context:
        warnings.append(f"Attention looks back at most {cfg['sliding_window']} tokens in this model; the graph attends to the whole context.")
    if family not in KNOWN_FAMILIES:
        warnings.append(f"'{family}' isn't a family TensorBloom knows. The graph is a best guess from its config.")
    spec = {
        "family": family,
        "name": top.get("_name_or_path") or "",
        "vocab_size": int(_first(cfg, "vocab_size", "padded_vocab_size")),
        "d_model": d,
        "n_layers": int(_first(cfg, "num_hidden_layers", "n_layer", "num_layers", "n_layers")),
        "n_heads": H,
        "n_kv_heads": int(KV),
        "head_dim": int(_first(cfg, "head_dim", "kv_channels", default=d // H)),
        "ffn_dim": int(F),
        "mlp": mlp,
        "norm": norm,
        "norm_eps": float(_first(cfg, "rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps", "norm_eps", default=1e-5)),
        "rope_theta": rope_theta,
        "learned_positions": learned_positions,
        "max_context": int(_first(cfg, "max_position_embeddings", "n_positions", "seq_length", "max_seq_len", default=2048)),
        "tie_embeddings": bool(_first(cfg, "tie_word_embeddings", default=family in ("gpt2", "gemma", "gemma2", "gemma3_text"))),
        "bias": bias,
        "n_experts": E,
        "experts_per_token": int(_first(cfg, "num_experts_per_tok", "moe_topk", default=2 if E else 0)),
        "torch_dtype": str(_first(cfg, "torch_dtype", "dtype", default="")) or None,
        "warnings": warnings,
    }
    q = top.get("quantization_config")
    if isinstance(q, dict):
        method = q.get("quant_method") or ("bitsandbytes" if q.get("load_in_4bit") or q.get("load_in_8bit") else "unknown")
        bits = q.get("bits") or (4 if q.get("load_in_4bit") else 8 if q.get("load_in_8bit") else None)
        spec["quantization"] = {"method": str(method), "bits": bits}
    spec["params"] = count_params(spec)
    spec["params_exact"] = False
    return spec


KNOWN_FAMILIES = {
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe", "gemma", "gemma2", "gemma3",
    "gemma3_text", "phi3", "phimoe", "gpt2", "gpt_neox", "olmo", "olmo2", "granite", "smollm3", "stablelm",
    "starcoder2", "deepseek", "internlm2", "cohere", "exaone", "falcon", "phi", "opt", "gptj", "gpt_bigcode",
}

# ---------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------

SAFETENSORS_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4, "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}


def parse_safetensors_header(raw: bytes) -> dict:
    """Tensor table from a safetensors header (the JSON after the 8-byte length)."""
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return header


def read_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 100_000_000:
            raise ValueError(f"{os.path.basename(path)} doesn't look like a safetensors file")
        return parse_safetensors_header(f.read(n))


def tensor_stats(header: dict) -> dict:
    """Parameter count, bytes and dtypes of a safetensors tensor table."""
    params = 0
    nbytes = 0
    dtypes: dict[str, int] = {}
    for info in header.values():
        n = 1
        for s in info["shape"]:
            n *= s
        params += n
        nbytes += n * SAFETENSORS_DTYPE_BYTES.get(info["dtype"], 2)
        dtypes[info["dtype"]] = dtypes.get(info["dtype"], 0) + n
    return {"params": params, "bytes": nbytes, "dtypes": dtypes}


def spec_from_tensors(header: dict, config: dict | None) -> dict:
    """Spec from a safetensors tensor table, using config.json when there is one."""
    names = list(header)
    if config:
        spec = spec_from_hf_config(config)
    else:
        spec = _spec_from_tensor_names(header)
    stats = tensor_stats(header)
    # Pre-quantized checkpoints (GPTQ, AWQ) pack several weights per element, so their count is not the model's
    if not spec.get("quantization") and not any(n.endswith((".qweight", ".scales")) for n in names):
        spec["params"] = stats["params"]
        spec["params_exact"] = True
    spec["stored_bytes"] = stats["bytes"]
    main = max(stats["dtypes"], key=stats["dtypes"].get) if stats["dtypes"] else None
    if main:
        spec["stored_dtype"] = main.lower()
    return spec


def _spec_from_tensor_names(header: dict) -> dict:
    """Best-effort spec for a Llama-style checkpoint with no config.json next to it."""
    def shape(*suffixes):
        for name, info in header.items():
            if name.endswith(suffixes):
                return info["shape"]
        return None
    layers = {int(p.split(".")[i + 1]) for p in header for i, part in enumerate(p.split(".")[:-1])
              if part in ("layers", "h", "blocks") and p.split(".")[i + 1].isdigit()}
    emb = shape("embed_tokens.weight", "wte.weight", "tok_embeddings.weight")
    q = shape("self_attn.q_proj.weight", "attention.wq.weight")
    k = shape("self_attn.k_proj.weight", "attention.wk.weight")
    up = shape("mlp.up_proj.weight", "feed_forward.w3.weight")
    if not (emb and q and k and up and layers):
        raise ValueError("Couldn't work out the architecture from the tensor names. Put the model's config.json next to the file and try again.")
    V, d = emb
    # Head size isn't stored in the tensors; 128 is by far the most common for models this shape
    hd = 128 if q[0] % 128 == 0 and q[0] >= 1024 else 64
    spec = {
        "family": "llama", "name": "", "vocab_size": V, "d_model": d, "n_layers": max(layers) + 1,
        "n_heads": q[0] // hd, "n_kv_heads": k[0] // hd, "head_dim": hd, "ffn_dim": up[0], "mlp": "swiglu",
        "norm": "rmsnorm", "norm_eps": 1e-5, "rope_theta": 10000.0, "learned_positions": False, "max_context": 4096,
        "tie_embeddings": shape("lm_head.weight", "output.weight") is None, "bias": "none", "n_experts": 0,
        "experts_per_token": 0, "torch_dtype": None,
        "warnings": ["No config.json was found, so head count, context length and rotary base are guesses."],
    }
    spec["params"] = count_params(spec)
    return spec


# GGUF value types
_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_STRING, _GGUF_ARRAY = 8, 9

GGUF_FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
    12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
    26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
}


class _Reader:
    def __init__(self, f):
        self.f = f

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) < size:
            raise ValueError("The GGUF file ended early; it may be incomplete")
        return struct.unpack(fmt, data)[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > 1 << 24:
            raise ValueError("Bad string length in GGUF header")
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int, keep_arrays: bool):
        if vtype in _GGUF_SCALARS:
            return self.unpack(_GGUF_SCALARS[vtype])
        if vtype == _GGUF_STRING:
            return self.string()
        if vtype == _GGUF_ARRAY:
            etype = self.unpack("<I")
            n = self.unpack("<Q")
            if not keep_arrays and etype in _GGUF_SCALARS:
                # Skip big arrays (token scores, merges) without decoding them
                self.f.seek(n * struct.calcsize(_GGUF_SCALARS[etype]), os.SEEK_CUR)
                return {"length": n}
            items = [self.value(etype, keep_arrays) for _ in range(n)]
            return items if keep_arrays else {"length": n}
        raise ValueError(f"Unknown GGUF value type {vtype}")


def read_gguf(path: str) -> dict:
    """Metadata and tensor shapes from a GGUF file's header (weights are not read)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{os.path.basename(path)} isn't a GGUF file")
        r = _Reader(f)
        version = r.unpack("<I")
        if version < 2:
            raise ValueError(f"GGUF version {version} is too old to read")
        n_tensors = r.unpack("<Q")
        n_kv = r.unpack("<Q")
        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.unpack("<I")
            # Per-layer arrays (e.g. head counts) are small and worth keeping; tokenizer arrays are not
            meta[key] = r.value(vtype, keep_arrays=not key.startswith("tokenizer."))
        tensors = {}
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.unpack("<I")
            dims = [r.unpack("<Q") for _ in range(n_dims)]
            ggml_type = r.unpack("<I")
            r.unpack("<Q")  # offset
            tensors[name] = {"shape": dims, "type": ggml_type}
    return {"version": version, "metadata": meta, "tensors": tensors}


def spec_from_gguf(gguf: dict, file_bytes: int) -> dict:
    meta, tensors = gguf["metadata"], gguf["tensors"]
    arch = str(meta.get("general.architecture", "llama"))

    def get(key, default=None):
        v = meta.get(f"{arch}.{key}", default)
        if isinstance(v, list):  # per-layer values: use the largest
            v = max(v) if v else default
        return v

    d = int(get("embedding_length"))
    H = int(get("attention.head_count"))
    emb = tensors.get("token_embd.weight")
    vocab = meta.get("tokenizer.ggml.tokens", {}).get("length") if isinstance(meta.get("tokenizer.ggml.tokens"), dict) else None
    vocab = int(get("vocab_size") or vocab or (emb["shape"][1] if emb else 32000))
    gpt2_like = arch in ("gpt2", "gptneox", "phi2", "falcon", "bloom", "starcoder")
    E = int(get("expert_count", 0) or 0)
    F = int(get("expert_feed_forward_length") or get("feed_forward_length") or 4 * d) if E else int(get("feed_forward_length") or 4 * d)
    spec = {
        "family": arch,
        "name": str(meta.get("general.name", "")),
        "vocab_size": vocab,
        "d_model": d,
        "n_layers": int(get("block_count")),
        "n_heads": H,
        "n_kv_heads": int(get("attention.head_count_kv", H)),
        "head_dim": int(get("attention.key_length", d // H)),
        "ffn_dim": F,
        "mlp": "gelu" if gpt2_like else "geglu" if arch.startswith("gemma") else "swiglu",
        "norm": "layernorm" if gpt2_like else "rmsnorm",
        # Stored as float32: round away the noise (9.99999997e-07 → 1e-06)
        "norm_eps": float(f'{float(get("attention.layer_norm_rms_epsilon") or get("attention.layer_norm_epsilon") or 1e-5):.4g}'),
        "rope_theta": float(get("rope.freq_base", 0.0 if gpt2_like else 10000.0)),
        "learned_positions": arch in ("gpt2", "starcoder"),
        "max_context": int(get("context_length", 2048)),
        "tie_embeddings": "output.weight" not in tensors,
        "bias": "all" if gpt2_like else "qkv" if "blk.0.attn_q.bias" in tensors else "none",
        "n_experts": E,
        "experts_per_token": int(get("expert_used_count", 0) or 0),
        "torch_dtype": None,
        "warnings": [],
    }
    params = 0
    for info in tensors.values():
        n = 1
        for s in info["shape"]:
            n *= s
        params += n
    spec["params"] = params or count_params(spec)
    spec["params_exact"] = bool(params)
    ftype = meta.get("general.file_type")
    quant = GGUF_FILE_TYPES.get(int(ftype), f"type {ftype}") if ftype is not None else None
    spec["stored_dtype"] = quant.lower() if quant else None
    spec["stored_bytes"] = file_bytes
    spec["quantization"] = {"method": "gguf", "type": quant, "bits": round(file_bytes * 8 / params, 2) if params else None}
    spec["warnings"].append("GGUF files run in llama.cpp, not PyTorch. TensorBloom can show and size this model; "
                            "to run or fine-tune it, import the original Hugging Face model.")
    return spec


# ---------------------------------------------------------------------------
# Hugging Face Hub (plain HTTPS so huggingface_hub isn't required)
# ---------------------------------------------------------------------------

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def hf_token(explicit: str | None = None) -> str | None:
    """Token from the dialog, the environment, or `huggingface-cli login`."""
    if explicit:
        return explicit.strip()
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    try:
        with open(os.path.join(home, "token")) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _hf_request(url: str, token: str | None, byte_range: tuple[int, int] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tensorbloom"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if byte_range:
        req.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(2 + 4 * attempt)
                continue
            raise


def _hf_get(model_id: str, filename: str, revision: str, token: str | None, byte_range: tuple[int, int] | None = None) -> bytes:
    url = f"{HF_ENDPOINT}/{model_id}/resolve/{revision}/{filename}"
    try:
        return _hf_request(url, token, byte_range)
    except urllib.error.HTTPError as e:
        raise _hf_error(e, model_id, token, filename) from e


def _hf_error(e: urllib.error.HTTPError, model_id: str, token: str | None, filename: str) -> Exception:
    if e.code in (401, 403):
        if token:
            return PermissionError(f"Hugging Face refused access to {model_id}. If it's a gated model, accept its license on huggingface.co with the account this token belongs to.")
        return PermissionError(f"{model_id} needs a Hugging Face token (it's private or gated). Paste a token, or run `huggingface-cli login`.")
    if e.code == 404:
        return FileNotFoundError(f"{filename} not found for {model_id}")
    if e.code == 429:
        return ConnectionError("Hugging Face is limiting how often this computer can ask. Wait a minute and try again, or add a token.")
    return ConnectionError(f"Hugging Face answered {e.code} for {model_id}")


def _hf_model_info(model_id: str, revision: str, token: str | None) -> dict | None:
    """The Hub's summary of a model: parameter counts per dtype and file sizes, in one request."""
    path = f"/api/models/{model_id}" + (f"/revision/{revision}" if revision != "main" else "") + "?blobs=true"
    try:
        return json.loads(_hf_request(HF_ENDPOINT + path, token))
    except (urllib.error.URLError, ValueError):
        return None


def _hf_safetensors_header(model_id: str, filename: str, revision: str, token: str | None) -> dict:
    (n,) = struct.unpack("<Q", _hf_get(model_id, filename, revision, token, (0, 7)))
    return parse_safetensors_header(_hf_get(model_id, filename, revision, token, (8, 8 + n - 1)))


def _weight_file(name: str) -> bool:
    return name.endswith(".safetensors") and not name.startswith(("original/", "consolidated"))


def inspect_hf(model_id: str, revision: str = "main", token: str | None = None) -> dict:
    model_id = model_id.strip().strip("/")
    if model_id.startswith(("http://", "https://")):
        model_id = "/".join(model_id.split("/")[3:5])
    token = hf_token(token)
    try:
        config = json.loads(_hf_get(model_id, "config.json", revision, token))
    except FileNotFoundError:
        raise FileNotFoundError(f"Couldn't find {model_id} on Hugging Face (no config.json). Check the name, e.g. Qwen/Qwen2.5-0.5B-Instruct.")
    spec = spec_from_hf_config(config)
    spec["name"] = model_id

    # Exact parameter count and download size, from the Hub's model info when it has them
    info = _hf_model_info(model_id, revision, token)
    st = (info or {}).get("safetensors") or {}
    files = [f for f in (info or {}).get("siblings", []) if _weight_file(f.get("rfilename", ""))]
    if files and all(f.get("size") for f in files):
        spec["stored_bytes"] = sum(f["size"] for f in files)
    if st.get("parameters") and not spec.get("quantization"):
        spec["params"] = int(st.get("total") or sum(st["parameters"].values()))
        spec["params_exact"] = True
        spec["stored_dtype"] = max(st["parameters"], key=st["parameters"].get).lower()
        return spec

    # Otherwise read the safetensors headers: two small range requests per shard, no weights downloaded
    shards = sorted({f["rfilename"] for f in files}) if files else []
    if not shards:
        try:
            index = json.loads(_hf_get(model_id, "model.safetensors.index.json", revision, token))
            shards = sorted(set(index.get("weight_map", {}).values()))
        except FileNotFoundError:
            shards = ["model.safetensors"]
    if len(shards) <= 64:
        try:
            header: dict = {}
            for s in shards:
                header.update(_hf_safetensors_header(model_id, s, revision, token))
            spec.update({k: v for k, v in spec_from_tensors(header, config).items() if k in ("params", "params_exact", "stored_bytes", "stored_dtype")})
        except (FileNotFoundError, ConnectionError, ValueError, urllib.error.URLError, struct.error):
            spec["warnings"].append("Couldn't read the weight files' headers, so the parameter count is computed from the config.")
    return spec


def inspect_path(path: str) -> dict:
    path = os.path.expanduser(path.strip())
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or folder: {path}")
    if os.path.isdir(path):
        config_path = os.path.join(path, "config.json")
        config = json.load(open(config_path)) if os.path.exists(config_path) else None
        gguf = sorted(f for f in os.listdir(path) if f.endswith(".gguf"))
        st = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        if st:
            header: dict = {}
            for f in st:
                header.update(read_safetensors_header(os.path.join(path, f)))
            spec = spec_from_tensors(header, config)
        elif config:
            spec = spec_from_hf_config(config)
        elif gguf:
            return inspect_path(os.path.join(path, gguf[0]))
        else:
            raise ValueError("This folder has no config.json, .safetensors or .gguf file")
        spec["name"] = spec.get("name") or os.path.basename(os.path.normpath(path))
        spec["local_path"] = path
        spec["loadable"] = bool(config)
        return spec
    if path.endswith(".gguf"):
        spec = spec_from_gguf(read_gguf(path), os.path.getsize(path))
        spec["name"] = spec.get("name") or os.path.basename(path)
        spec["loadable"] = False
        return spec
    if path.endswith(".safetensors"):
        folder = os.path.dirname(path)
        config_path = os.path.join(folder, "config.json")
        config = json.load(open(config_path)) if os.path.exists(config_path) else None
        spec = spec_from_tensors(read_safetensors_header(path), config)
        spec["name"] = spec.get("name") or os.path.basename(path)
        spec["local_path"] = folder
        spec["loadable"] = bool(config)
        return spec
    if os.path.basename(path) == "config.json":
        return inspect_path(os.path.dirname(path))
    raise ValueError("Pick a .gguf or .safetensors file, a config.json, or a model folder")


def llm_inspect(params: dict) -> dict:
    """RPC: read an LLM's architecture from a Hugging Face id or a local file, without loading weights."""
    if params.get("path"):
        spec = inspect_path(params["path"])
        spec["source"] = {"kind": "path", "path": params["path"]}
    else:
        model_id = (params.get("model_id") or "").strip()
        if not model_id:
            raise ValueError("Enter a Hugging Face model id or pick a file")
        spec = inspect_hf(model_id, params.get("revision") or "main", params.get("token"))
        spec["source"] = {"kind": "hf", "model_id": spec["name"], "revision": params.get("revision") or "main"}
        spec["loadable"] = "quantization" not in spec or spec["quantization"].get("method") == "bitsandbytes"
    spec["head_dim"] = spec.get("head_dim") or spec["d_model"] // spec["n_heads"]
    return spec


def system_memory() -> dict:
    """Total RAM, for memory estimates on CPU and Apple silicon (which shares RAM with the GPU)."""
    total = None
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if total is None and os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total = stat.ullTotalPhys
        except Exception:
            total = None
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return {"ram_total": total, "mps_available": mps}


def llm_device(params: dict) -> dict:
    """RPC: the memory an LLM can use here: the largest CUDA GPU, Apple silicon's shared memory, or RAM."""
    mem = system_memory()
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"name": torch.cuda.get_device_name(i), "memory_total": p.total_memory})
    try:
        import bitsandbytes  # noqa: F401
        has_bnb = True
    except Exception:
        has_bnb = False
    packages = {}
    for pkg in ("transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except Exception:
            packages[pkg] = None
    if gpus:
        best = max(gpus, key=lambda g: g["memory_total"])
        device = {"kind": "cuda", "name": best["name"], "memory": best["memory_total"], "gpu_count": len(gpus)}
    elif mem["mps_available"]:
        # Apple silicon: the GPU can use most, not all, of the shared memory
        device = {"kind": "mps", "name": "Apple silicon GPU", "memory": int(mem["ram_total"] * 0.75) if mem["ram_total"] else None}
    else:
        device = {"kind": "cpu", "name": "CPU", "memory": mem["ram_total"]}
    device["ram_total"] = mem["ram_total"]
    device["quantization_4bit"] = device["kind"] == "cuda" and has_bnb
    device["packages"] = packages
    return device


# ---------------------------------------------------------------------------
# Running a model: download, load (optionally 8- or 4-bit), and chat
# ---------------------------------------------------------------------------

# Set by the sidecar to its JSON-RPC notification sender
notify = lambda method, params=None: None  # noqa: E731

# What running and fine-tuning need beyond torch. bitsandbytes (8/4-bit) is added on NVIDIA GPUs only.
LLM_PACKAGES = ["transformers>=4.45", "accelerate>=0.34", "peft>=0.13", "huggingface_hub>=0.25", "datasets>=2.20"]
BNB_PACKAGE = "bitsandbytes>=0.43"


class _Runtime:
    """The loaded model and what the sidecar is doing with it. One model at a time."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.info: dict | None = None
        self.busy: str | None = None  # "installing" | "loading" | "generating" | "training"
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def claim(self, what: str):
        with self.lock:
            if self.busy:
                raise RuntimeError(f"Busy {self.busy}. Wait for it to finish or stop it first.")
            self.busy = what
            self.stop.clear()

    def release(self):
        with self.lock:
            self.busy = None


runtime = _Runtime()


def _has(pkg: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(pkg) is not None


def missing_packages(precision: str = "bf16") -> list[str]:
    missing = [p for p in ("transformers", "accelerate", "peft", "huggingface_hub", "datasets") if not _has(p)]
    if precision in ("int8", "nf4") and not _has("bitsandbytes"):
        missing.append("bitsandbytes")
    return missing


def _install_command(packages: list[str]) -> list[str]:
    import shutil
    if _has("pip"):
        return [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    uv = shutil.which("uv")
    if uv:
        # Environments made by uv have no pip of their own
        return [uv, "pip", "install", "--python", sys.executable, "--upgrade", *packages]
    raise RuntimeError("Can't install packages: this Python has no pip, and uv wasn't found.")


def llm_install(params: dict) -> dict:
    """RPC: install the packages for running and fine-tuning LLMs, reporting each line of output."""
    import subprocess
    packages = list(LLM_PACKAGES)
    if torch.cuda.is_available() and sys.platform != "darwin":
        packages.append(BNB_PACKAGE)
    cmd = _install_command(packages)
    runtime.claim("installing")

    def run():
        tail: list[str] = []
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    tail = (tail + [line])[-20:]
                    notify("llm.install", {"status": "running", "line": line})
            code = proc.wait()
            import importlib
            importlib.invalidate_caches()
            if code == 0:
                notify("llm.install", {"status": "done"})
            else:
                notify("llm.install", {"status": "error", "error": "\n".join(tail[-8:]) or f"pip exited with {code}"})
        except Exception as e:  # noqa: BLE001
            notify("llm.install", {"status": "error", "error": str(e)})
        finally:
            runtime.release()

    threading.Thread(target=run, daemon=True).start()
    return {"started": True, "packages": packages}


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _compute_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.bfloat16


def _folder_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _download(model_id: str, revision: str, token: str | None, expected_bytes: int | None) -> str:
    """Downloads the model's weights, config and tokenizer into the Hugging Face cache, reporting progress."""
    from huggingface_hub import snapshot_download, constants
    repo_dir = os.path.join(constants.HF_HUB_CACHE, "models--" + model_id.replace("/", "--"))
    # Downloads over Xet land in its chunk cache before the files are written, so count that too
    xet_dir = getattr(constants, "HF_XET_CACHE", os.path.join(constants.HF_HOME, "xet"))
    done = threading.Event()

    def size(path):
        return _folder_bytes(path) if os.path.isdir(path) else 0

    def watch():
        xet_start = size(xet_dir)
        while not done.wait(1.0):
            have = size(repo_dir) + max(0, size(xet_dir) - xet_start)
            notify("llm.progress", {"status": "downloading", "bytes": min(have, expected_bytes) if expected_bytes else have, "total": expected_bytes})

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        return snapshot_download(
            model_id, revision=revision, token=token,
            # Weights as safetensors only (never pickled .bin files), plus config and tokenizer
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.py"],
            ignore_patterns=["original/*", "consolidated*"],
        )
    finally:
        done.set()


def load_model(params: dict):
    """Loads a model for chat or fine-tuning. Blocking; reports progress as llm.progress notifications."""
    precision = params.get("precision", "bf16")
    missing = missing_packages(precision)
    if missing:
        raise RuntimeError(f"Running language models needs {', '.join(missing)}. Install them from the Language model tab.")
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _pick_device()
    if precision in ("int8", "nf4") and device != "cuda":
        raise RuntimeError("8-bit and 4-bit loading needs an NVIDIA GPU. Pick 16-bit instead.")
    source = params["source"]
    token = hf_token(params.get("token"))
    if source.get("kind") == "hf":
        path = _download(source["model_id"], source.get("revision", "main"), token, params.get("expected_bytes"))
    else:
        path = params.get("local_path") or source.get("path")
        if path and os.path.isfile(path):
            path = os.path.dirname(path)
        if not path or not os.path.exists(os.path.join(path, "config.json")):
            raise RuntimeError("Can't load this model: its folder needs a config.json next to the weights. GGUF files can't be loaded in PyTorch.")
    if runtime.stop.is_set():
        raise RuntimeError("Stopped")

    notify("llm.progress", {"status": "loading"})
    dtype = _compute_dtype(device)
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    kwargs["dtype" if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"] = (
        torch.float32 if precision == "fp32" else dtype)
    if precision in ("int8", "nf4"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True) if precision == "int8" else BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    if device == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    runtime.model, runtime.tokenizer = model, tokenizer
    used = torch.cuda.memory_allocated() if device == "cuda" else model.get_memory_footprint()
    runtime.info = {
        "name": source.get("model_id") or os.path.basename(os.path.normpath(path)),
        "path": path, "precision": precision, "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else ("Apple silicon GPU" if device == "mps" else "CPU"),
        "memory_used": used, "adapter": None,
    }
    return runtime.info


def _unload():
    runtime.model = runtime.tokenizer = runtime.info = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def llm_load(params: dict) -> dict:
    """RPC: download (if needed) and load a model in the background. Reports llm.progress, then loaded or error."""
    missing = missing_packages(params.get("precision", "bf16"))
    if missing:
        raise RuntimeError(f"Running language models needs {', '.join(missing)}. Install them first.")
    runtime.claim("loading")

    def run():
        try:
            _unload()
            info = load_model(params)
            notify("llm.progress", {"status": "loaded", "info": info})
        except Exception as e:  # noqa: BLE001
            _unload()
            notify("llm.progress", {"status": "error", "error": _friendly_error(e)})
        finally:
            runtime.release()

    threading.Thread(target=run, daemon=True).start()
    return {"started": True}


def _friendly_error(e: Exception) -> str:
    msg = str(e) or type(e).__name__
    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg.lower():
        return "Ran out of GPU memory. Try a smaller precision (8-bit or 4-bit), a shorter context, or a smaller model."
    if "401" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
        return "Hugging Face refused the download. This model is gated: accept its license on huggingface.co and add your token."
    return msg


def llm_unload(params: dict) -> dict:
    if runtime.busy in ("generating", "training"):
        raise RuntimeError(f"Can't unload while {runtime.busy}")
    _unload()
    return {"unloaded": True}


def llm_status(params: dict) -> dict:
    return {"loaded": runtime.info, "busy": runtime.busy, "missing": missing_packages("nf4" if torch.cuda.is_available() else "bf16")}


def _prompt_ids(messages: list[dict], max_context: int):
    tok = runtime.tokenizer
    if getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids, mask = enc["input_ids"], enc["attention_mask"]
    else:
        # A base model has no chat format: continue the text as written
        text = "\n\n".join(m["content"] for m in messages if m.get("content"))
        enc = tok(text, return_tensors="pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
    if ids.shape[1] > max_context:
        ids, mask = ids[:, -max_context:], mask[:, -max_context:]
    return ids, mask


def llm_generate(params: dict) -> dict:
    """RPC: reply to a conversation, streaming text as llm.token notifications and ending with llm.done."""
    if runtime.model is None:
        raise RuntimeError("Load the model first")
    from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
    runtime.claim("generating")
    rid = params.get("id")
    max_new = int(params.get("max_new_tokens", 512))
    temperature = float(params.get("temperature", 0.7))
    max_context = int(params.get("max_context") or getattr(runtime.model.config, "max_position_embeddings", 4096) or 4096)

    class _StopRequested(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return runtime.stop.is_set()

    def run():
        try:
            tok, model = runtime.tokenizer, runtime.model
            ids, mask = _prompt_ids(params["messages"], max(256, max_context - max_new))
            device = next(model.parameters()).device
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=600)
            failure: list[Exception] = []

            def generate():
                try:
                    with torch.no_grad():
                        model.generate(
                            input_ids=ids.to(device), attention_mask=mask.to(device), max_new_tokens=max_new,
                            do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                            top_p=float(params.get("top_p", 0.95)) if temperature > 0 else None,
                            streamer=streamer, stopping_criteria=StoppingCriteriaList([_StopRequested()]),
                            pad_token_id=tok.pad_token_id,
                        )
                except Exception as e:  # noqa: BLE001
                    failure.append(e)
                    streamer.end()

            start = time.time()
            first_token_at = None
            worker = threading.Thread(target=generate, daemon=True)
            worker.start()
            text = ""
            for piece in streamer:
                if piece:
                    if first_token_at is None:
                        first_token_at = time.time()
                    text += piece
                    notify("llm.token", {"id": rid, "text": piece})
            worker.join()
            if failure:
                raise failure[0]
            elapsed = time.time() - start
            n = len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0
            gen_time = time.time() - (first_token_at or start)
            notify("llm.done", {"id": rid, "text": text, "tokens": n, "seconds": round(elapsed, 2),
                                "tokens_per_second": round(n / gen_time, 1) if gen_time > 0 and n > 1 else None,
                                "prompt_tokens": int(ids.shape[1]), "stopped": runtime.stop.is_set()})
        except Exception as e:  # noqa: BLE001
            notify("llm.done", {"id": rid, "error": _friendly_error(e)})
        finally:
            runtime.release()

    threading.Thread(target=run, daemon=True).start()
    return {"started": True}


def llm_stop(params: dict) -> dict:
    """RPC: stop generating, loading or fine-tuning at the next chance."""
    runtime.stop.set()
    return {"stopping": runtime.busy}


# ---------------------------------------------------------------------------
# Fine-tuning with LoRA (QLoRA when the model is loaded in 4-bit)
# ---------------------------------------------------------------------------

def _row_text(row: dict, column: str | None, tok) -> str | None:
    """One training example as text: a chat (in the model's own format), a prompt and answer, or plain text."""
    def chat(messages):
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(messages, tokenize=False)
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)

    if column:
        v = row.get(column)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return chat([{"role": m.get("role") or m.get("from"), "content": m.get("content") or m.get("value", "")} for m in v])
        return str(v) if v is not None else None
    for key in ("messages", "conversations", "conversation"):
        v = row.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            roles = {"human": "user", "gpt": "assistant"}
            return chat([{"role": roles.get(m.get("role") or m.get("from"), m.get("role") or m.get("from")),
                          "content": m.get("content") or m.get("value", "")} for m in v])
    prompt = row.get("instruction") or row.get("prompt") or row.get("question")
    answer = row.get("output") or row.get("completion") or row.get("response") or row.get("answer")
    if prompt is not None and answer is not None:
        if row.get("input"):
            prompt = f"{prompt}\n\n{row['input']}"
        return chat([{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(answer)}])
    v = row.get("text") or row.get("content")
    return str(v) if v is not None else None


def load_texts(dataset: dict, tok, max_samples: int = 5000) -> list[str]:
    """Training texts from a Hugging Face dataset or a .txt/.jsonl/.json file."""
    kind = dataset.get("kind")
    column = dataset.get("column") or None
    rows: list[dict] = []
    if kind == "file":
        path = os.path.expanduser(dataset.get("path", ""))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No such file: {path}")
        if path.endswith(".txt") or path.endswith(".md"):
            text = open(path, encoding="utf-8", errors="replace").read()
            # Plain text is split into paragraphs-ish pieces and packed back together for training
            return [p for p in text.split("\n\n") if p.strip()] or [text]
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        elif path.endswith(".json"):
            data = json.load(open(path, encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("data", [])
        else:
            raise ValueError("Use a .txt, .md, .jsonl or .json file")
    elif kind == "hf":
        if not _has("datasets"):
            raise RuntimeError("Hugging Face datasets need the 'datasets' package. Install it from the Language model tab.")
        from datasets import load_dataset
        name = dataset.get("name", "").strip()
        if not name:
            raise ValueError("Enter a Hugging Face dataset name")
        split = dataset.get("split") or "train"
        ds = load_dataset(name, dataset.get("config") or None, split=f"{split}[:{max_samples}]")
        rows = list(ds)
    else:
        raise ValueError("Pick a dataset to fine-tune on")
    texts = [t for t in (_row_text(r, column, tok) for r in rows[:max_samples]) if t and t.strip()]
    if not texts:
        cols = ", ".join(rows[0].keys()) if rows else "none"
        raise ValueError(f"Couldn't find text in the dataset (columns: {cols}). Name the column to train on.")
    return texts


def pack_tokens(texts: list[str], tok, block: int) -> torch.Tensor:
    """Tokenizes the texts, joins them with end-of-text tokens, and cuts the stream into equal blocks."""
    eos = tok.eos_token_id
    stream: list[int] = []
    for t in texts:
        stream.extend(tok(t, add_special_tokens=False)["input_ids"])
        if eos is not None:
            stream.append(eos)
    n = len(stream) // block
    if n == 0:
        # Too little data for one full block: train on what there is
        return torch.tensor([stream], dtype=torch.long)
    return torch.tensor(stream[: n * block], dtype=torch.long).view(n, block)


def _adapter_dir(name: str) -> str:
    base = os.environ.get("TENSORBLOOM_DATA_DIR") or os.path.join(os.path.expanduser("~"), ".tensorbloom")
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in name.split("/")[-1])
    return os.path.join(base, "llm-adapters", f"{safe}-{time.strftime('%Y%m%d-%H%M%S')}")


def _finetune(params: dict):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    progress = lambda p: notify("train.progress", p)  # noqa: E731
    log = lambda text: notify("train.log", {"text": text, "stream": "stdout"})  # noqa: E731
    precision = params.get("precision", "bf16")
    if runtime.model is None or (runtime.info or {}).get("precision") != precision:
        progress({"status": "downloading", "dataset": params["source"].get("model_id") or "the model"})
        _unload()
        load_model(params)
    model, tok = runtime.model, runtime.tokenizer
    if isinstance(model, PeftModel):
        # Fine-tune the base model again rather than stacking adapters
        model = model.unload()
    device = next(model.parameters()).device
    seed = params.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))

    log(f"Loading dataset…")
    texts = load_texts(params["dataset"], tok, int(params.get("max_samples", 5000)))
    blocks = pack_tokens(texts, tok, int(params.get("max_length", 512)))
    perm = torch.randperm(len(blocks))
    n_val = max(1, int(len(blocks) * float(params.get("val_split", 0.05)))) if len(blocks) > 4 else 0
    val, train = blocks[perm[:n_val]], blocks[perm[n_val:]]
    log(f"{len(texts)} examples → {len(train)} training blocks and {len(val)} validation blocks of {blocks.shape[1]} tokens")

    quantized = precision in ("int8", "nf4")
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    lora = params.get("lora", {})
    model = get_peft_model(model, LoraConfig(
        r=int(lora.get("rank", 16)), lora_alpha=int(lora.get("alpha", 32)), lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules="all-linear", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"{'QLoRA' if precision == 'nf4' else 'LoRA'}: training {trainable:,} of {total:,} parameters ({100 * trainable / total:.2f}%)")

    batch_size = max(1, int(params.get("batch_size", 4)))
    accum = max(1, int(params.get("grad_accum", 1)))
    epochs = max(1, int(params.get("epochs", 1)))
    steps_per_epoch = max(1, (len(train) + batch_size - 1) // batch_size)
    total_updates = max(1, epochs * steps_per_epoch // accum)
    warmup = max(1, int(0.03 * total_updates))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(params.get("lr", 2e-4)),
                            weight_decay=float(params.get("weight_decay", 0.0)))
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, u / total_updates))))
    autocast = torch.autocast("cuda", dtype=_compute_dtype("cuda")) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    out_dir = _adapter_dir((runtime.info or {}).get("name", "model"))

    def evaluate() -> float | None:
        if len(val) == 0:
            return None
        model.eval()
        losses = []
        with torch.no_grad(), autocast:
            for i in range(0, min(len(val), 50 * batch_size), batch_size):
                b = val[i:i + batch_size].to(device)
                losses.append(model(input_ids=b, labels=b).loss.float().item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    update = 0
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train))
        running, count = 0.0, 0
        for step in range(steps_per_epoch):
            if runtime.stop.is_set():
                break
            b = train[order[step * batch_size:(step + 1) * batch_size]].to(device)
            with autocast:
                loss = model(input_ids=b, labels=b).loss
            (loss / accum).backward()
            running += loss.float().item()
            count += 1
            if (step + 1) % accum == 0 or step == steps_per_epoch - 1:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                update += 1
            if step % 5 == 0 or step == steps_per_epoch - 1:
                progress({"epoch": epoch, "step": step + 1, "total_steps": steps_per_epoch,
                          "train_loss": loss.float().item(), "lr": opt.param_groups[0]["lr"]})
        if count == 0:
            break
        val_loss = evaluate()
        gpu = None
        if device.type == "cuda":
            gpu = {"allocated_mb": torch.cuda.memory_allocated() / 2**20, "reserved_mb": torch.cuda.memory_reserved() / 2**20,
                   "max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
        log(f"Epoch {epoch}/{epochs}  train_loss={running / count:.4f}" + (f"  val_loss={val_loss:.4f}  perplexity={math.exp(min(val_loss, 50)):.1f}" if val_loss is not None else ""))
        progress({"epoch": epoch, "step": count, "train_loss": running / count, "val_loss": val_loss,
                  "lr": opt.param_groups[0]["lr"], "gpu": gpu})
        model.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        progress({"status": "checkpoint", "path": out_dir, "epoch": epoch})
        if runtime.stop.is_set():
            break

    model.eval()
    model.config.use_cache = True
    runtime.model = model
    if runtime.info is not None:
        runtime.info["adapter"] = out_dir if os.path.isdir(out_dir) else None
    notify("llm.progress", {"status": "loaded", "info": runtime.info})
    log(f"Adapter saved to {out_dir}")
    return out_dir


def llm_finetune_start(params: dict) -> dict:
    """RPC: LoRA fine-tune the model in the background, reporting train.progress like graph training does."""
    missing = missing_packages(params.get("precision", "bf16"))
    if params.get("dataset", {}).get("kind") == "hf" and not _has("datasets"):
        missing.append("datasets")
    if missing:
        raise RuntimeError(f"Fine-tuning needs {', '.join(missing)}. Install them from the Language model tab.")
    runtime.claim("training")

    def run():
        adapter = None
        try:
            adapter = _finetune(params)
        except Exception as e:  # noqa: BLE001
            import traceback
            notify("train.log", {"text": traceback.format_exc(), "stream": "stderr"})
            notify("train.progress", {"status": "error", "error": _friendly_error(e)})
        finally:
            if runtime.model is not None:
                runtime.model.config.use_cache = True
            runtime.release()
        # Only after the release, so a chat sent the moment training ends isn't refused as busy
        if adapter is not None:
            notify("train.progress", {"status": "complete", "adapter": adapter})

    threading.Thread(target=run, daemon=True).start()
    return {"started": True}


METHODS = {
    "llm.finetune.start": llm_finetune_start,
    "llm.inspect": llm_inspect,
    "llm.device": llm_device,
    "llm.install": llm_install,
    "llm.status": llm_status,
    "llm.load": llm_load,
    "llm.unload": llm_unload,
    "llm.generate": llm_generate,
    "llm.stop": llm_stop,
}
