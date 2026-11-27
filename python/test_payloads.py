"""Tests for the Try it payload layer: decoding, encoder choice, and wire shapes.

These tests need torch but no dataset download and no trained model. They call the
same describe_sample() the sidecar uses, so the payload contracts the UI depends on
are checked directly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import Subset, TensorDataset

import tensorbloom_payloads as tp

failures = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


WORD_VOCAB = {"<pad>": 0, "<unk>": 1, "the": 2, "cat": 3, "sat": 4}
DECODER = tp.VocabDecoder.from_vocab(WORD_VOCAB)

# --- Decoding -------------------------------------------------------------
print("Decoding")
decoded = DECODER.decode([2, 3, 4])
check("word ids decode to words", decoded.text == "the cat sat" and decoded.token_count == 3,
      str(decoded))

padded = DECODER.decode([2, 3, 0, 0])
check("trailing padding is removed", padded.text == "the cat" and padded.token_count == 2,
      str(padded))

unknown = DECODER.decode([1, 99, 2])
check("unknown and out-of-range ids stay visible",
      unknown.text == "<unk> <id 99> the" and unknown.token_count == 3, str(unknown))

empty = DECODER.decode([0, 0])
check("an all-pad sequence becomes empty text", empty.text == "" and empty.token_count == 0,
      str(empty))

char_decoder = tp.VocabDecoder.from_chars(["\n", "a", "b"])
chars = char_decoder.decode([0, 1, 2, 1])
check("character ids join without spaces and keep id 0",
      chars.text == "\naba" and chars.token_count == 4, str(chars))

big_vocab = {"<pad>": 0, "<unk>": 1, **{f"w{i}": i + 2 for i in range(600)}}
big = tp.VocabDecoder.from_vocab(big_vocab).decode(list(range(2, 600)))
check("long text is capped and marked truncated",
      big.truncated and big.token_count == 598 and len(big.text.split()) == tp.MAX_TEXT_TOKENS,
      str((big.token_count, big.truncated, len(big.text.split()))))

# --- Dataset semantics ----------------------------------------------------
print("Dataset semantics")
base = TensorDataset(torch.zeros(4, 1))
base.semantics = tp.DatasetSemantics(decoder=DECODER)
base.classes = ["neg", "pos"]
subset = Subset(base, [0, 1, 2])
semantics = tp.dataset_semantics(subset)
check("semantics are found through a Subset", semantics.decoder is DECODER, str(semantics))
check("class names fall back to the dataset's classes",
      semantics.class_names == ("neg", "pos"), str(semantics.class_names))
check("dataset_attr sees through wrappers", tp.dataset_attr(subset, "classes") == ["neg", "pos"])

plain = TensorDataset(torch.zeros(2, 1))
check("datasets without semantics stay empty", tp.dataset_semantics(plain) is tp.EMPTY_SEMANTICS)

# --- Image detection ------------------------------------------------------
print("Image detection")
check("a 1-channel image is an image", tp.is_image(torch.zeros(1, 8, 8)))
check("a 3-channel image is an image", tp.is_image(torch.zeros(3, 8, 8)))
check("four channels are not an image", not tp.is_image(torch.zeros(4, 8, 8)))
check("a 3x3 map is not an image", not tp.is_image(torch.zeros(1, 3, 3)))
check("a token sequence is not an image", not tp.is_image(torch.zeros(8, dtype=torch.long)))

# --- Input payloads -------------------------------------------------------
print("Inputs")
x_text = torch.tensor([2, 3, 4, 0, 0], dtype=torch.long)
out_classes = torch.tensor([0.1, 0.9, 0.2])
text_sample = tp.describe_sample(x_text, out_classes, 1,
                                 semantics=tp.DatasetSemantics(decoder=DECODER),
                                 is_reconstruction=False)
check("token ids become a text input",
      text_sample["input"] == {"kind": "text", "text": "the cat sat",
                               "token_count": 3, "truncated": False},
      str(text_sample["input"]))
check("a class output keeps probabilities and is marked correct",
      text_sample["output"]["kind"] == "classes"
      and text_sample["output"]["top"][0]["index"] == 1
      and text_sample["correct"] is True, str(text_sample))

numeric_sample = tp.describe_sample(torch.randn(8), out_classes, 1,
                                    semantics=tp.EMPTY_SEMANTICS, is_reconstruction=False)
check("a float input stays numeric", numeric_sample["input"]["kind"] == "values"
      and numeric_sample["input"]["total"] == 8, str(numeric_sample["input"]))

image_sample = tp.describe_sample(torch.rand(3, 8, 8), out_classes, 0,
                                  semantics=tp.EMPTY_SEMANTICS, is_reconstruction=False)
check("an image input becomes a PNG", image_sample["input"]["kind"] == "image"
      and image_sample["input"]["width"] == 8, str(image_sample["input"])[:120])

# --- Outputs and targets --------------------------------------------------
print("Outputs and targets")
x_seq = torch.tensor([2, 3, 4, 0, 0], dtype=torch.long)
logits = torch.full((5, 5), -10.0)
logits[:, 2] = 10.0  # argmax id 2 = "the" at every position
y_seq = torch.tensor([3, 4, 0, 0, 0], dtype=torch.long)
lm_sample = tp.describe_sample(x_seq, logits, y_seq,
                               semantics=tp.DatasetSemantics(decoder=DECODER),
                               is_reconstruction=False)
check("vocab-shaped logits become text",
      lm_sample["output"] == {"kind": "text", "text": "the the the the the",
                              "token_count": 5, "truncated": False},
      str(lm_sample["output"]))
check("a token target becomes text",
      lm_sample["target"] == {"kind": "text", "text": "cat sat",
                              "token_count": 2, "truncated": False},
      str(lm_sample["target"]))
check("text output leaves correctness unset", lm_sample["correct"] is None,
      str(lm_sample["correct"]))

wrong_size = torch.randn(5, 6)
guard = tp.describe_sample(x_seq, wrong_size, y_seq,
                           semantics=tp.DatasetSemantics(decoder=DECODER),
                           is_reconstruction=False)
check("a vocab-size mismatch falls back to numbers", guard["output"]["kind"] == "values",
      str(guard["output"])[:120])

wrong_rows = torch.randn(4, 5)
row_guard = tp.describe_sample(x_seq, wrong_rows, y_seq,
                               semantics=tp.DatasetSemantics(decoder=DECODER),
                               is_reconstruction=False)
check("a row-count mismatch falls back to numbers", row_guard["output"]["kind"] == "values",
      str(row_guard["output"])[:120])

regression = tp.describe_sample(torch.randn(8), torch.tensor([0.5]), torch.tensor(1.0),
                                semantics=tp.EMPTY_SEMANTICS, is_reconstruction=False)
check("a numeric target stays numeric", regression["output"]["kind"] == "values"
      and regression["target"]["kind"] == "values", str(regression))

# Reconstructions have never shown a target, whether the output is an image or numbers.
img = torch.rand(1, 8, 8)
img_recon = tp.describe_sample(img, img.clone(), torch.tensor(1),
                               semantics=tp.EMPTY_SEMANTICS, is_reconstruction=True)
check("image reconstruction has no target", img_recon["output"]["kind"] == "image"
      and img_recon["target"] is None, str(img_recon)[:160])

vec = torch.randn(8)
vec_recon = tp.describe_sample(vec, vec.clone(), torch.randn(8),
                               semantics=tp.EMPTY_SEMANTICS, is_reconstruction=True)
check("numeric reconstruction has no target", vec_recon["output"]["kind"] == "values"
      and vec_recon["target"] is None, str(vec_recon)[:160])

# --- Value payload and registry order ------------------------------------
print("Contracts")
values = tp.values_payload(torch.arange(20))
check("values keep the first 12, the shape, and the total",
      len(values["values"]) == 12 and values["shape"] == [20] and values["total"] == 20,
      str(values))

check("input encoders end with the numeric fallback", tp.INPUT_ENCODERS[-1].name == "values",
      str([e.name for e in tp.INPUT_ENCODERS]))
check("output encoders end with the numeric fallback", tp.OUTPUT_ENCODERS[-1].name == "values",
      str([e.name for e in tp.OUTPUT_ENCODERS]))
check("target encoders end with the numeric fallback", tp.TARGET_ENCODERS[-1].name == "values",
      str([e.name for e in tp.TARGET_ENCODERS]))

print()
if failures:
    print(f"{len(failures)} failed: {', '.join(failures)}")
    sys.exit(1)
print("All payload tests passed")
