"""Tests for LLM support: DecoderBlock, reading model files and configs, and the graph it's drawn as."""
import json
import os
import re
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

import torch
import tensorbloom_llm as llm
import tensorbloom_sidecar as sc

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


tmp = tempfile.mkdtemp(prefix="tensorbloom-llm-test-")

LLAMA_8B = {
    "model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
    "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256, "max_position_embeddings": 131072,
    "rope_theta": 500000.0, "rms_norm_eps": 1e-5, "tie_word_embeddings": False, "hidden_act": "silu", "torch_dtype": "bfloat16",
}
QWEN_05B = {
    "model_type": "qwen2", "hidden_size": 896, "num_hidden_layers": 24, "num_attention_heads": 14,
    "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936, "max_position_embeddings": 32768,
    "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": True, "hidden_act": "silu",
    "sliding_window": 32768, "use_sliding_window": False,
}
MIXTRAL = {**LLAMA_8B, "model_type": "mixtral", "vocab_size": 32000, "num_local_experts": 8, "num_experts_per_tok": 2,
           "max_position_embeddings": 32768, "rope_theta": 1e6}
GPT2 = {"model_type": "gpt2", "n_embd": 768, "n_layer": 12, "n_head": 12, "vocab_size": 50257, "n_positions": 1024,
        "layer_norm_epsilon": 1e-5, "activation_function": "gelu_new"}

# --- Specs from Hugging Face configs ---
print("Configs")
s = llm.spec_from_hf_config(LLAMA_8B)
check("Llama 3.1 8B parameter count", s["params"] == 8_030_261_248, s["params"])
check("Llama has grouped-query attention", (s["n_heads"], s["n_kv_heads"], s["head_dim"]) == (32, 8, 128))
s = llm.spec_from_hf_config(QWEN_05B)
check("Qwen2.5 0.5B parameter count (qkv bias, tied head)", s["params"] == 494_032_768, s["params"])
check("an unused sliding window isn't reported", not any("looks back" in w for w in s["warnings"]))
s = llm.spec_from_hf_config(MIXTRAL)
check("Mixtral parameter count", s["params"] == 46_702_792_704, s["params"])
s = llm.spec_from_hf_config(GPT2)
check("GPT-2 parameter count", s["params"] == 124_439_808, s["params"])
check("GPT-2 is LayerNorm, plain GELU, learned positions",
      (s["norm"], s["mlp"], s["rope_theta"], s["learned_positions"]) == ("layernorm", "gelu", 0.0, True))
gemma3 = llm.spec_from_hf_config({"model_type": "gemma3", "text_config": {
    "model_type": "gemma3_text", "hidden_size": 2560, "num_hidden_layers": 34, "num_attention_heads": 8,
    "num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 10240, "vocab_size": 262208,
    "hidden_activation": "gelu_pytorch_tanh"}})
check("multimodal configs read the text model", gemma3["d_model"] == 2560 and gemma3["mlp"] == "geglu" and gemma3["head_dim"] == 256)
check("…and say the encoder is left out", any("vision" in w for w in gemma3["warnings"]))

# --- DecoderBlock ---
print("DecoderBlock")
torch.manual_seed(0)
block = llm.DecoderBlock(d_model=64, n_heads=8, n_kv_heads=2, ffn_dim=96)
block.eval()
x = torch.randn(2, 10, 64)
with torch.no_grad():
    y = block(x)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 4, 64)
    y2 = block(x2)
check("keeps the shape", y.shape == x.shape)
check("is causal", torch.allclose(y[:, :6], y2[:, :6], atol=1e-5) and not torch.allclose(y[:, 6:], y2[:, 6:]))
check("rotary positions make order matter",
      not torch.allclose(block(x.flip(1)).flip(1), y, atol=1e-4))
check("takes an unbatched sequence", block(x[0]).shape == (10, 64))
moe = llm.DecoderBlock(d_model=32, n_heads=4, n_kv_heads=4, ffn_dim=48, n_experts=4, experts_per_token=2)
out = moe(torch.randn(3, 5, 32))
out.sum().backward()
check("mixture of experts runs and trains", out.shape == (3, 5, 32) and moe.router.weight.grad is not None)
try:
    llm.DecoderBlock(d_model=64, n_heads=8, n_kv_heads=3)
    check("rejects heads that don't group evenly", False)
except ValueError:
    check("rejects heads that don't group evenly", True)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


for name, cfg in [("Llama-style", dict(n_heads=8, n_kv_heads=2)), ("Qwen-style bias", dict(n_heads=8, n_kv_heads=4, bias="qkv")),
                  ("GPT-2-style", dict(n_heads=4, n_kv_heads=4, mlp="gelu", norm="layernorm", bias="all", rope_theta=0)),
                  ("experts", dict(n_heads=4, n_kv_heads=2, n_experts=3))]:
    spec = {"d_model": 64, "ffn_dim": 96, "vocab_size": 100, "n_layers": 1, "tie_embeddings": True, **cfg}
    m = llm.DecoderBlock(d_model=64, ffn_dim=96, **cfg)
    norm = 64 if cfg.get("norm", "rmsnorm") == "rmsnorm" else 128
    check(f"count_params matches the module ({name})", llm.count_params(spec) - 100 * 64 - norm == n_params(m),
          f"{llm.count_params(spec) - 100 * 64 - norm} vs {n_params(m)}")

ts_path = os.path.join(os.path.dirname(__file__), "..", "app", "src", "lib", "llm-block.ts")
ts_source = re.search(r"DECODER_BLOCK_SOURCE = `(.*)`;", open(ts_path).read(), re.S).group(1).replace("\\\\", "\\")
check("the app's copy of the block source matches", ts_source == llm.DECODER_BLOCK_SOURCE)

# --- In a graph: trains, and generated code matches ---
print("Graph")
graph = {
    "nodes": [
        {"id": "inp", "type": "placeholder", "moduleType": "Input", "label": "tokens", "params": {"shape": "[1, 16]", "dtype": "int64"}},
        {"id": "emb", "type": "call_module", "moduleType": "Embedding", "label": "embed_tokens", "params": {"num_embeddings": 50, "embedding_dim": 32}},
        {"id": "b0", "type": "call_module", "moduleType": "DecoderBlock", "label": "layer_0",
         "params": {"d_model": 32, "n_heads": 4, "n_kv_heads": 2, "head_dim": 0, "ffn_dim": 64, "mlp": "swiglu", "norm": "rmsnorm",
                    "norm_eps": 1e-5, "rope_theta": 10000, "bias": "none", "n_experts": 0, "experts_per_token": 2}},
        {"id": "norm", "type": "call_module", "moduleType": "RMSNorm", "label": "norm", "params": {"normalized_shape": "[32]", "eps": 1e-5}},
        {"id": "head", "type": "call_module", "moduleType": "Linear", "label": "lm_head", "params": {"in_features": 32, "out_features": 50, "bias": False}},
        {"id": "out", "type": "output", "moduleType": "Output", "label": "logits", "params": {}},
    ],
    "edges": [{"source": a, "target": b} for a, b in [("inp", "emb"), ("emb", "b0"), ("b0", "norm"), ("norm", "head"), ("head", "out")]],
}
model = sc._build_model_from_graph(graph)
tokens = torch.randint(0, 50, (2, 16))
logits = model(tokens)
check("a graph with DecoderBlock builds and runs", logits.shape == (2, 16, 50))
code = sc.decompile_graph(graph)
ns = {}
exec(code, ns)
gen = ns["Model"]()
gen.load_state_dict({k.replace("layer_0.", "layer_0."): v for k, v in model.state_dict().items()})
model.eval(), gen.eval()
with torch.no_grad():
    check("generated code defines DecoderBlock and matches", torch.allclose(gen(tokens), model(tokens), atol=1e-5))

# --- File formats ---
print("Files")


def write_safetensors(path, tensors, dtype="BF16"):
    header, offset = {}, 0
    for name, shape in tensors.items():
        n = 1
        for s_ in shape:
            n *= s_
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n * 2]}
        offset += n * 2
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.truncate(8 + len(raw) + offset)


tiny = {"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 128, "vocab_size": 100, "tie_word_embeddings": False}
tiny_tensors = {"model.embed_tokens.weight": [100, 64], "lm_head.weight": [100, 64], "model.norm.weight": [64]}
for i in range(2):
    p = f"model.layers.{i}."
    tiny_tensors.update({p + "self_attn.q_proj.weight": [64, 64], p + "self_attn.k_proj.weight": [32, 64],
                         p + "self_attn.v_proj.weight": [32, 64], p + "self_attn.o_proj.weight": [64, 64],
                         p + "mlp.gate_proj.weight": [128, 64], p + "mlp.up_proj.weight": [128, 64],
                         p + "mlp.down_proj.weight": [64, 128], p + "input_layernorm.weight": [64],
                         p + "post_attention_layernorm.weight": [64]})
folder = os.path.join(tmp, "tiny-llama")
os.makedirs(folder)
write_safetensors(os.path.join(folder, "model.safetensors"), tiny_tensors)
json.dump(tiny, open(os.path.join(folder, "config.json"), "w"))
s = llm.llm_inspect({"path": folder})
check("a model folder is read exactly", s["params_exact"] and s["params"] == llm.count_params(llm.spec_from_hf_config(tiny)), s["params"])
check("…and knows how it's stored", s["stored_dtype"] == "bf16" and s["loadable"])
os.remove(os.path.join(folder, "config.json"))
s = llm.llm_inspect({"path": os.path.join(folder, "model.safetensors")})
check("a lone safetensors file is guessed from tensor names", (s["n_layers"], s["d_model"], s["ffn_dim"]) == (2, 64, 128) and not s["loadable"])


def gguf_string(v):
    b = v.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path):
    kv = [
        ("general.architecture", 8, gguf_string("llama")),
        ("general.name", 8, gguf_string("Tiny Llama")),
        ("general.file_type", 4, struct.pack("<I", 15)),
        ("llama.block_count", 4, struct.pack("<I", 2)),
        ("llama.embedding_length", 4, struct.pack("<I", 64)),
        ("llama.feed_forward_length", 4, struct.pack("<I", 128)),
        ("llama.attention.head_count", 4, struct.pack("<I", 4)),
        ("llama.attention.head_count_kv", 4, struct.pack("<I", 2)),
        ("llama.context_length", 4, struct.pack("<I", 8192)),
        ("llama.rope.freq_base", 6, struct.pack("<f", 500000.0)),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 100) + b"".join(gguf_string(f"t{i}") for i in range(100))),
        ("tokenizer.ggml.scores", 9, struct.pack("<IQ", 6, 100) + struct.pack("<100f", *range(100))),
    ]
    tensors = [("token_embd.weight", [64, 100]), ("output.weight", [64, 100])]
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
        for k, t, v in kv:
            f.write(gguf_string(k) + struct.pack("<I", t) + v)
        for name, dims in tensors:
            f.write(gguf_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", 12, 0))
        f.write(b"\0" * 7000)


gguf_path = os.path.join(tmp, "tiny.Q4_K_M.gguf")
write_gguf(gguf_path)
s = llm.llm_inspect({"path": gguf_path})
check("GGUF metadata is read", (s["family"], s["n_layers"], s["n_heads"], s["n_kv_heads"], s["vocab_size"], s["rope_theta"])
      == ("llama", 2, 4, 2, 100, 500000.0), s)
check("GGUF quantization is named", s["stored_dtype"] == "q4_k_m" and s["quantization"]["method"] == "gguf")
check("GGUF isn't offered for loading in PyTorch", s["loadable"] is False)
try:
    llm.llm_inspect({"path": os.path.join(folder, "model.safetensors") + ".missing"})
    check("a missing file is a clear error", False)
except FileNotFoundError:
    check("a missing file is a clear error", True)

# --- Hugging Face, with the network replaced ---
print("Hugging Face")
shard_headers = {}
for i, shard in enumerate(["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]):
    part = {k: {"dtype": "BF16", "shape": v, "data_offsets": [0, 0]} for j, (k, v) in enumerate(tiny_tensors.items()) if j % 2 == i}
    shard_headers[shard] = json.dumps(part).encode()
requests = []


def fake_get(model_id, filename, revision, token, byte_range=None):
    requests.append((filename, byte_range, token))
    if filename == "config.json":
        return json.dumps(tiny).encode()
    if filename == "model.safetensors.index.json":
        return json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}).encode()
    raw = shard_headers[filename]
    if byte_range == (0, 7):
        return struct.pack("<Q", len(raw))
    return raw[byte_range[0] - 8: byte_range[1] - 8 + 1]


real_get, real_info = llm._hf_get, llm._hf_model_info
llm._hf_get = fake_get
llm._hf_model_info = lambda model_id, revision, token: None  # the Hub's summary isn't available
try:
    s = llm.llm_inspect({"model_id": "https://huggingface.co/someone/tiny-llama", "token": "hf_test"})
    llm_requests = list(requests)
    llm._hf_model_info = lambda model_id, revision, token: {
        "safetensors": {"parameters": {"BF16": 12345}, "total": 12345},
        "siblings": [{"rfilename": "model.safetensors", "size": 24690}, {"rfilename": "original/consolidated.00.pth", "size": 1}]}
    requests.clear()
    s_info = llm.llm_inspect({"model_id": "someone/tiny-llama"})
finally:
    llm._hf_get, llm._hf_model_info = real_get, real_info
check("a model URL is read as its id", s["name"] == "someone/tiny-llama" and s["source"]["model_id"] == "someone/tiny-llama")
check("shard headers give the exact count", s["params_exact"] and s["params"] == llm.count_params(llm.spec_from_hf_config(tiny)))
check("only headers are fetched", all(f.endswith(".json") or r is not None for f, r, _ in llm_requests))
check("the token is used", all(t == "hf_test" for _, _, t in llm_requests))
check("the Hub's summary gives the count and size in one request",
      (s_info["params"], s_info["stored_bytes"], s_info["stored_dtype"]) == (12345, 24690, "bf16") and [f for f, _, _ in requests] == ["config.json"])

# --- RPC wiring ---
print("RPC")
check("llm methods are registered", "llm.inspect" in sc.METHODS and "llm.device" in sc.METHODS)
dev = llm.llm_device({})
check("device memory is reported", dev["kind"] in ("cuda", "mps", "cpu") and "packages" in dev)

print()
if failures:
    print(f"{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)
print("All LLM tests passed")
