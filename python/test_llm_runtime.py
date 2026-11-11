"""End-to-end tests for running and fine-tuning a language model, on a tiny random Llama built locally.

Needs transformers, peft and accelerate (pip install -r python/requirements-llm.txt); skipped without them.
Nothing is downloaded: the model and its tokenizer are made here.
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
import tensorbloom_llm as llm

if llm.missing_packages("bf16"):
    print(f"Skipping LLM runtime tests: {', '.join(llm.missing_packages('bf16'))} not installed")
    sys.exit(0)

from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


tmp = tempfile.mkdtemp(prefix="tensorbloom-llm-runtime-")
os.environ["TENSORBLOOM_DATA_DIR"] = tmp
events: list[tuple[str, dict]] = []
llm.notify = lambda method, params=None: events.append((method, params or {}))


def wait_for(method: str, pred=lambda p: True, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        for m, p in list(events):
            if m == method and pred(p):
                return p
        time.sleep(0.05)
    raise TimeoutError(f"no {method} within {timeout}s")


# --- A tiny model and tokenizer ---
corpus = ["the cat sat on the mat", "the dog sat on the log", "a bird sang in the tree", "hello how are you", "i am fine thank you"] * 20
tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "<|user|>", "<|assistant|>"]))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
tok.chat_template = "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} </s> {% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
model_dir = os.path.join(tmp, "tiny-llama")
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=256, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                  tie_word_embeddings=False)
LlamaForCausalLM(cfg).save_pretrained(model_dir)
tok.save_pretrained(model_dir)

print("Inspect")
spec = llm.llm_inspect({"path": model_dir})
check("the saved model is read exactly", spec["params_exact"] and spec["n_kv_heads"] == 2 and spec["loadable"])

print("Load")
llm.llm_load({"source": spec["source"], "local_path": spec["local_path"], "precision": "bf16"})
p = wait_for("llm.progress", lambda p: p.get("status") in ("loaded", "error"))
check("loads", p["status"] == "loaded", p.get("error", ""))
check("reports where it loaded", p.get("info", {}).get("device") in ("cpu", "cuda", "mps"))
st = llm.llm_status({})
check("status says it's loaded and idle", st["loaded"] is not None and st["busy"] is None)

print("Chat")
llm.llm_generate({"id": 1, "messages": [{"role": "user", "content": "hello how are you"}], "max_new_tokens": 8, "temperature": 0})
done = wait_for("llm.done", lambda p: p.get("id") == 1)
streamed = "".join(p["text"] for m, p in events if m == "llm.token" and p.get("id") == 1)
check("replies", "error" not in done, done.get("error", ""))
check("streams the reply as it goes", streamed == done.get("text") and done.get("tokens", 0) > 0, f"{streamed!r} vs {done.get('text')!r}")
check("counts the prompt", done.get("prompt_tokens", 0) > 3)
try:
    llm.runtime.claim("generating")
    try:
        llm.llm_generate({"id": 2, "messages": [{"role": "user", "content": "hi"}]})
        check("one job at a time", False)
    except RuntimeError:
        check("one job at a time", True)
finally:
    llm.runtime.release()

print("Fine-tune")
data_path = os.path.join(tmp, "chats.jsonl")
with open(data_path, "w") as f:
    for _ in range(40):
        f.write(json.dumps({"messages": [{"role": "user", "content": "hello how are you"},
                                         {"role": "assistant", "content": "i am fine thank you"}]}) + "\n")
texts = llm.load_texts({"kind": "file", "path": data_path}, llm.runtime.tokenizer)
check("chats are written in the model's chat format", texts[0].startswith("<|user|> hello") and "<|assistant|> i am fine" in texts[0], texts[0])
blocks = llm.pack_tokens(texts, llm.runtime.tokenizer, 16)
check("text is packed into equal blocks", blocks.dim() == 2 and blocks.shape[1] == 16 and blocks.shape[0] > 4)
alpaca = llm._row_text({"instruction": "say hi", "input": "politely", "output": "hello"}, None, llm.runtime.tokenizer)
check("instruction rows become a chat", "say hi" in alpaca and "politely" in alpaca and "hello" in alpaca)

events.clear()
llm.llm_finetune_start({"source": spec["source"], "local_path": spec["local_path"], "precision": "bf16",
                        "dataset": {"kind": "file", "path": data_path}, "lora": {"rank": 8, "alpha": 16, "dropout": 0.0},
                        "max_length": 16, "epochs": 3, "batch_size": 4, "lr": 3e-3, "seed": 0})
end = wait_for("train.progress", lambda p: p.get("status") in ("complete", "error"), timeout=600)
check("fine-tunes", end["status"] == "complete", end.get("error", ""))
epochs = [p for m, p in events if m == "train.progress" and "val_loss" in p]
check("reports each epoch like graph training", len(epochs) == 3 and all(p["train_loss"] > 0 for p in epochs))
check("the loss goes down", epochs[-1]["train_loss"] < epochs[0]["train_loss"] * 0.8, [round(p["train_loss"], 3) for p in epochs])
steps = [p for m, p in events if m == "train.progress" and "total_steps" in p]
check("reports steps within an epoch", len(steps) > 0 and steps[0]["total_steps"] > 0)
adapter = end.get("adapter")
check("saves the adapter", adapter and os.path.exists(os.path.join(adapter, "adapter_config.json")), adapter)
check("chat uses the fine-tuned adapter", llm.runtime.info["adapter"] == adapter)
llm.llm_generate({"id": 3, "messages": [{"role": "user", "content": "hello how are you"}], "max_new_tokens": 6, "temperature": 0})
reply = wait_for("llm.done", lambda p: p.get("id") == 3)
check("the fine-tuned model learned its answer", reply.get("text", "").strip().startswith("i am fine"), reply)

print("Stop")
events.clear()
llm.llm_finetune_start({"source": spec["source"], "local_path": spec["local_path"], "precision": "bf16",
                        "dataset": {"kind": "file", "path": data_path}, "max_length": 16, "epochs": 50, "batch_size": 1, "lr": 1e-4})
wait_for("train.progress", lambda p: "total_steps" in p)
llm.llm_stop({})
end = wait_for("train.progress", lambda p: p.get("status") in ("complete", "error"), timeout=120)
check("stopping ends fine-tuning early", end["status"] == "complete" and llm.runtime.busy is None)
epochs_run = [p for m, p in events if m == "train.progress" and "val_loss" in p]
check("…without running every epoch", len(epochs_run) < 50)

print("Errors")
events.clear()
llm.llm_finetune_start({"source": spec["source"], "local_path": spec["local_path"], "precision": "bf16",
                        "dataset": {"kind": "file", "path": os.path.join(tmp, "missing.jsonl")}})
err = wait_for("train.progress", lambda p: p.get("status") == "error")
check("a missing dataset is reported", "No such file" in err["error"], err["error"])
llm.llm_unload({})
check("unloads", llm.runtime.model is None and llm.llm_status({})["loaded"] is None)

print()
if failures:
    print(f"{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)
print("All LLM runtime tests passed")
