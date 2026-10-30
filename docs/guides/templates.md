# Template Reference

TensorBloom includes 16 pre-built architecture templates. Each template creates a complete graph with a Data node, model layers, and loss function — ready to train with one click.

## Getting Started

| Template | Dataset | Description |
|----------|---------|-------------|
| **Simple MLP** | MNIST | 3-layer feedforward network. The "hello world" of deep learning. |

## Vision

| Template | Dataset | Description |
|----------|---------|-------------|
| **LeNet-5** | MNIST | The original CNN (LeCun 1998). Conv → Pool → Conv → Pool → FC. |
| **ResNet-18** | CIFAR-10 | Residual blocks with skip connections (He 2015). Demonstrates Add nodes for skip connections. |
| **Vision Transformer (ViT)** | CIFAR-10 | Patch embedding via Conv2d → Transformer encoder stack → classifier (Dosovitskiy 2020). |
| **MobileNetV2 Block** | CIFAR-10 | Inverted residual blocks: expand → depthwise → project + skip (Sandler 2018). |
| **U-Net** | ImageFolder | Encoder-decoder with skip connections for image segmentation (Ronneberger 2015). Uses Concatenate nodes. |

## Language

| Template | Dataset | Description |
|----------|---------|-------------|
| **Transformer** | WikiText-2 | Full encoder-decoder transformer with dual embedding inputs (Vaswani 2017). |
| **nanoGPT** | TinyShakespeare | Karpathy's char-level GPT. 6 decoder layers, 384 dim, 6 heads. |
| **BERT Base** | AG News | 12 encoder layers for text classification. Permute → Pool → Classify. |
| **LSTM Text Classifier** | IMDB | Bidirectional LSTM → Pool → Linear for sentiment analysis. |

## Audio

| Template | Dataset | Description |
|----------|---------|-------------|
| **Whisper Encoder** | SpeechCommands | Conv1d stem + Transformer encoder for keyword spotting (Radford 2022). |
| **WaveNet Block** | SpeechCommands | Dilated convolutions (d=1,2,4) with residual connections (van den Oord 2016). |

## Generative

| Template | Dataset | Description |
|----------|---------|-------------|
| **Autoencoder** | MNIST | Symmetric FC encoder-decoder with bottleneck. Unsupervised reconstruction. |
| **Conv Autoencoder** | MNIST | Convolutional encoder (Conv+Pool) → decoder (ConvTranspose). |
| **Embedding Classifier** | MNIST | Conv feature extractor with learned embedding and classification head. |

## Other

| Template | Dataset | Description |
|----------|---------|-------------|
| **DenseNet Block** | CIFAR-10 | Dense connectivity via Concatenate nodes. Each layer receives all previous features. |

## Using Templates

1. Go to **Insert > From Template**
2. Browse by category. Each card shows a preview of the model's layout, its layer count, dataset, and paper
3. Click a template to load it
4. The graph, training config, and dataset are all pre-configured
5. Press **Train** (or Ctrl+Enter) to train immediately, or modify the architecture first

## Modifying Templates

Templates are starting points. After loading:
- **Change layers** — Click any node to edit parameters, or delete/add layers
- **Swap datasets** — Click the Data node and select a different source
- **Adjust training** — Switch to the Training tab to change optimizer, learning rate, epochs
- **Auto-layout** — Press Ctrl+L after making structural changes
