# Custom Data Guide

TensorBloom supports loading your own datasets for training. This guide covers all supported data formats.

## Quick Start

1. Drag a **Data** node onto the canvas
2. In Properties, select your data source
3. Connect the Data node's **x** handle to your model's first layer
4. Connect the Data node's **y** handle to your Loss function's **target** input
5. Train

## Built-in Datasets

Select a preset from the dropdown — no configuration needed:

| Dataset | Domain | Input Shape | Classes | Description |
|---------|--------|-------------|---------|-------------|
| MNIST | Vision | [1, 28, 28] | 10 | Handwritten digits |
| Fashion-MNIST | Vision | [1, 28, 28] | 10 | Clothing items |
| CIFAR-10 | Vision | [3, 32, 32] | 10 | Natural images (10 categories) |
| CIFAR-100 | Vision | [3, 32, 32] | 100 | Natural images (100 categories) |
| TinyShakespeare | Text | [256] | 65 | Character-level Shakespeare |
| WikiText-2 | Text | [128] | 30,000 | Word-level Wikipedia |
| IMDB | Text | [100] | 2 | Movie review sentiment |
| AG News | Text | [100] | 4 | News classification |
| SpeechCommands | Audio | [80, 100] | 35 | Keyword spotting |

Datasets are downloaded automatically on first use to `./data/`.

## Custom Tensor Files

For your own data saved as PyTorch tensors, NumPy arrays, or SafeTensors.

### Supported Formats

| Format | Extension | How to Create |
|--------|-----------|---------------|
| PyTorch | `.pt` | `torch.save({"features": X, "labels": y}, "data.pt")` |
| NumPy | `.npz` | `np.savez("data.npz", features=X, labels=y)` |
| NumPy | `.npy` | `np.save("data.npy", X)` |
| SafeTensors | `.safetensors` | `save_file({"features": X}, "data.safetensors")` |

### Creating a Dataset

```python
import torch

# Classification
X = torch.randn(1000, 3, 32, 32)  # 1000 images, 3 channels, 32x32
y = torch.randint(0, 10, (1000,))  # 10 classes
torch.save({"images": X, "labels": y}, "my_dataset.pt")

# Regression
X = torch.randn(500, 13)           # 500 samples, 13 features
y = torch.randn(500, 1)            # continuous target
torch.save({"inputs": X, "targets": y}, "regression_data.pt")

# Autoencoder (no labels)
X = torch.randn(1000, 1, 28, 28)
torch.save({"data": X}, "autoencoder_data.pt")
```

### Loading in TensorBloom

1. Select **Custom Tensors** as the source
2. Browse for your file or type the path
3. Click **Scan Data**
4. The app discovers all tensors in your file and shows them with shapes, types, and statistics
5. Assign each tensor a role:
   - **Model Input** — feeds into your neural network
   - **Loss Target** — compared against the model's output by the loss function
   - **Not Used** — ignored during training

### Tensor Requirements

- All tensors must have the **same number of samples** (first dimension)
- The first dimension is the batch dimension (number of samples)
- Input tensors are cast to `float32` by default
- Target tensors for classification should be `int64`/`long` (class indices)
- Target tensors for regression should be `float32`

### Folder of Files

You can also point to a folder containing multiple tensor files. TensorBloom scans all `.pt`, `.npz`, `.npy`, and `.safetensors` files and merges their tensors.

## ImageFolder (Local Images)

For image classification with your own photos:

```
my_images/
├── train/
│   ├── cats/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── dogs/
│       ├── img003.jpg
│       └── img004.jpg
└── val/
    ├── cats/
    │   └── img005.jpg
    └── dogs/
        └── img006.jpg
```

1. Select **ImageFolder** source
2. Set the path to your folder
3. Configure image size and color mode (RGB/Grayscale)
4. Optionally enable data augmentation (random flip, crop, rotation, color jitter)

## HuggingFace Datasets

> **Note:** HuggingFace support is experimental in v0.1. Simple vision and text datasets work. Complex/nested datasets (multi-modal, structured data with nested dicts) require a custom preprocessing script.

Load datasets from the HuggingFace Hub:

1. Select **HuggingFace** source
2. Enter the dataset name (e.g., `mnist`, `imdb`, `ag_news`)
3. Click **Inspect Data** to discover columns, types, and splits
4. Configure column mapping (image/text/label columns)
5. For datasets with complex columns, provide a preprocessing script in the code editor

### What works

- **Vision datasets** with `Image` + `ClassLabel` columns (CIFAR-10, MNIST, etc.)
- **Text classification** with `string` + `ClassLabel` columns (IMDB, AG News, etc.)
- **Simple tabular** datasets with scalar columns

### What doesn't work yet

- Datasets with nested/dict columns (SQuAD, multi-modal datasets)
- Streaming datasets (large datasets that don't fit in memory)
- Datasets requiring custom tokenizers

For unsupported formats, export your data as a `.pt` file and use Custom Tensors instead.

Requires `pip install datasets`.

## CSV Files

For tabular data in CSV format:

1. Select **Custom CSV** source
2. Set the path to your CSV file
3. Configure delimiter, header, target column
4. Optionally select feature columns and normalization

The CSV should have feature columns followed by a target column. The first row can be a header.

## Validation Split

All data sources support a **Val Split** parameter (default 0.2). This splits your training data into training and validation sets. The validation loss is reported after each epoch to detect overfitting.

## Tips

- **Shape mismatch?** TensorBloom auto-fixes `in_features` and `in_channels` when you click Start Training
- **Wrong loss function?** If your target is class labels but you're using MSELoss, the preflight check will warn you
- **Large datasets?** Use mixed precision training (Advanced > Mixed Precision) to reduce VRAM usage
- **Reproducibility?** Set a random seed in Advanced > Random Seed
