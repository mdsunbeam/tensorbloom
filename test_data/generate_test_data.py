"""
Generate diverse test datasets for TensorFile loading.
Tests different formats, shapes, dtypes, naming conventions, and edge cases.
"""
import torch
import numpy as np
import os

OUT = os.path.dirname(__file__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. SIMPLE CLASSIFICATION — single .pt file, dict with "x" and "y" keys
#    (the happy path — keys match default handle names)
# ═══════════════════════════════════════════════════════════════════════════════
print("1. Simple classification (x/y keys, .pt)")
n = 500
x = torch.randn(n, 1, 28, 28)  # MNIST-like images
y = torch.randint(0, 10, (n,))  # 10 classes
torch.save({"x": x, "y": y}, os.path.join(OUT, "simple_classification.pt"))
print(f"   x: {x.shape} {x.dtype}, y: {y.shape} {y.dtype}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. NON-STANDARD KEYS — .pt file with "features"/"labels" instead of x/y
#    (tests that column mapping dropdown is needed)
# ═══════════════════════════════════════════════════════════════════════════════
print("2. Non-standard keys (features/labels, .pt)")
n = 300
features = torch.randn(n, 3, 32, 32)  # CIFAR-like
labels = torch.randint(0, 5, (n,)).long()
torch.save({"features": features, "labels": labels}, os.path.join(OUT, "custom_keys.pt"))
print(f"   features: {features.shape} {features.dtype}, labels: {labels.shape} {labels.dtype}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. REGRESSION — continuous targets, not classification
#    (tests float targets instead of long class indices)
# ═══════════════════════════════════════════════════════════════════════════════
print("3. Regression (tabular, float targets, .pt)")
n = 1000
inputs = torch.randn(n, 13)       # 13 features (like Boston Housing)
targets = torch.randn(n, 1)       # continuous target
torch.save({"inputs": inputs, "targets": targets}, os.path.join(OUT, "regression.pt"))
print(f"   inputs: {inputs.shape} {inputs.dtype}, targets: {targets.shape} {targets.dtype}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MULTI-TENSOR — 3+ tensors (features, labels, sample_weights)
#    (tests adding a third handle beyond x/y)
# ═══════════════════════════════════════════════════════════════════════════════
print("4. Multi-tensor (3 tensors, .pt)")
n = 200
images = torch.randn(n, 3, 64, 64)
classes = torch.randint(0, 20, (n,))
weights = torch.rand(n)  # per-sample weights
torch.save({"images": images, "classes": classes, "weights": weights},
           os.path.join(OUT, "multi_tensor.pt"))
print(f"   images: {images.shape}, classes: {classes.shape}, weights: {weights.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. NPZ FORMAT — numpy archive (common in scientific workflows)
#    (tests .npz loading path)
# ═══════════════════════════════════════════════════════════════════════════════
print("5. NumPy archive (.npz)")
n = 400
data = np.random.randn(n, 1, 28, 28).astype(np.float32)
target = np.random.randint(0, 10, n).astype(np.int64)
np.savez(os.path.join(OUT, "numpy_data.npz"), data=data, target=target)
print(f"   data: {data.shape} {data.dtype}, target: {target.shape} {target.dtype}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SEQUENCE DATA — 1D sequences for text/time-series models
#    (tests non-image tensor shapes)
# ═══════════════════════════════════════════════════════════════════════════════
print("6. Sequence data (1D, .pt)")
n = 600
seq_len = 128
tokens = torch.randint(0, 5000, (n, seq_len))  # token ids
sentiment = torch.randint(0, 2, (n,))           # binary
torch.save({"tokens": tokens, "sentiment": sentiment}, os.path.join(OUT, "sequence_data.pt"))
print(f"   tokens: {tokens.shape} {tokens.dtype}, sentiment: {sentiment.shape} {sentiment.dtype}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SINGLE TENSOR — just a tensor, not a dict (autoencoder/unsupervised)
#    (tests the non-dict .pt loading path)
# ═══════════════════════════════════════════════════════════════════════════════
print("7. Single tensor, no dict (.pt)")
n = 250
images = torch.randn(n, 1, 28, 28)
torch.save(images, os.path.join(OUT, "unsupervised.pt"))
print(f"   tensor: {images.shape} {images.dtype}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. MISMATCHED SAMPLE COUNTS — tensors with different batch sizes
#    (edge case: should produce an error during training)
# ═══════════════════════════════════════════════════════════════════════════════
print("8. Mismatched sample counts (.pt) — should error")
torch.save({
    "features": torch.randn(100, 10),
    "labels": torch.randint(0, 3, (150,)),  # deliberately wrong: 150 != 100
}, os.path.join(OUT, "mismatched_samples.pt"))
print(f"   features: [100, 10], labels: [150] (MISMATCH)")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. MANY TENSORS — dict with 6 tensors (embeddings, masks, etc.)
#    (tests the dropdown with many options)
# ═══════════════════════════════════════════════════════════════════════════════
print("9. Many tensors (6 keys, .pt)")
n = 200
torch.save({
    "input_ids": torch.randint(0, 30000, (n, 64)),
    "attention_mask": torch.ones(n, 64),
    "token_type_ids": torch.zeros(n, 64).long(),
    "position_ids": torch.arange(64).unsqueeze(0).expand(n, -1),
    "labels": torch.randint(0, 4, (n,)),
    "embeddings": torch.randn(n, 64, 768),
}, os.path.join(OUT, "many_tensors.pt"))
print(f"   6 tensors: input_ids, attention_mask, token_type_ids, position_ids, labels, embeddings")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. FLOAT64 + INT32 — unusual dtypes that need casting
#     (tests dtype coercion in sidecar)
# ═══════════════════════════════════════════════════════════════════════════════
print("10. Unusual dtypes (float64/int32, .pt)")
n = 300
torch.save({
    "x": torch.randn(n, 5, dtype=torch.float64),
    "y": torch.randint(0, 3, (n,), dtype=torch.int32),
}, os.path.join(OUT, "unusual_dtypes.pt"))
print(f"   x: float64, y: int32")

print(f"\nDone! {len(os.listdir(OUT)) - 1} test datasets in {OUT}/")
