# TensorBloom

**[tensorbloom.org](https://tensorbloom.org)** · [Docs](https://tensorbloom.org/docs) · [Download](https://github.com/mdsunbeam/tensorbloom/releases/latest) · [Roadmap](ROADMAP.md)

Build, train, and export PyTorch neural networks visually. Drag-and-drop layers, connect them, train on real datasets with GPU acceleration, and export to ONNX or TorchScript — all without writing boilerplate code.

![TensorBloom: ResNet-18 after a training run, with the run summary, loss and accuracy charts, and the model summary](docs/screenshot-hero.png)

## Why TensorBloom?

- **See your architecture** — No more debugging invisible dimension mismatches in code. The graph shows every layer, every connection, and every tensor shape.
- **Train instantly** — Built-in datasets (MNIST, CIFAR, IMDB, and more), one-click training with live loss curves, and GPU auto-detection.
- **See what changed** — Every run is kept, so you can compare runs side by side and try a trained model on samples it never saw.
- **Learn by doing** — 16 starter templates from Simple MLP to nanoGPT. Load one, modify it, train it, see what happens.
- **Real PyTorch underneath** — Every graph generates real `nn.Module` code. Export your model and use it anywhere.

## Features

**Graph Editor**
- 70+ PyTorch modules: Linear, Conv2d, Transformer, LSTM, BatchNorm, and more
- Drag-and-drop from the layer palette, Enter on a focused layer, or Ctrl+K spotlight search
- Eight layer color families, each with its own icon, so a layer's kind reads without relying on color
- Skip connections, concatenation, multi-branch architectures; skip connections run dotted beside the block they skip
- Group layers into named frames that collapse to a single card, without changing the model
- Reliable undo/redo, Save vs Save As, an unsaved-changes prompt, and auto-save you can restore or discard

**Shape Checking**
- Real-time shape inference on every edge and layer
- A shape error marks the edge *into* the layer that can't take it, shows "Gets [1, 512], expects [1, 500]" on the layer, and lists every problem in the toolbar
- One-click fixes such as "Set in_features to 512", from the problems list or the Properties panel

**Training**
- One-click training (Ctrl+Enter) with a live status in the toolbar: download, epoch progress, and the result
- Live loss/accuracy charts with a hover readout, and a summary card when the run ends
- Plain-language error panel with hints for shape mismatches, out-of-memory, bad labels, missing packages, and more
- Pre-flight fixes are listed for review (Fix and train, Train without changes, or Cancel) instead of being applied silently
- 10+ built-in datasets: MNIST, CIFAR-10/100, TinyShakespeare, WikiText-2, IMDB, AG News, SpeechCommands
- Custom tensor files (.pt, .npz, .safetensors), ImageFolder, HuggingFace datasets, CSV
- GPU acceleration with mixed precision support and VRAM monitoring
- Optimizers: SGD, Adam, AdamW, RMSprop | Schedulers: StepLR, Cosine, OneCycleLR, ReduceLROnPlateau

**Runs and Results**
- Every run is kept with its curves, settings, and model, saved in the project file
- The **Runs** tab overlays runs on shared charts and puts their results and settings side by side, highlighting what differs
- **Restore** brings back a run's graph and settings
- **Try it** runs a trained model on validation samples it didn't train on, shows its top guesses, filters to mistakes, or tries your own image

**Help Built In**
- A welcome card with starter models, a Getting Started guide, and a Help menu
- A "?" next to every layer parameter and training setting explains what it does
- Each layer links to its PyTorch documentation

**Code Generation & Export**
- Live PyTorch code generation — see the equivalent `nn.Module` as you build
- Import existing PyTorch code into the graph editor
- Export to ONNX (for deployment) or TorchScript (for PyTorch-native use)

<details>
<summary>Compare training runs</summary>

![The Runs tab comparing two runs' validation curves](docs/screenshot-runs.png)
</details>

<details>
<summary>Try a trained model</summary>

![A grid of validation digits with the model's top three guesses and whether each was right](docs/screenshot-try-model.png)
</details>

<details>
<summary>Shape error detection</summary>

![A shape mismatch marked on the edge into fc2, with the fix offered in the Properties panel](docs/screenshot-error-viz.png)
</details>

<details>
<summary>Layer groups</summary>

![One collapsed group and one open group on the canvas, with the group's panel](docs/screenshot-groups.png)
</details>

<details>
<summary>Training configuration and parameter help</summary>

![Training panel with optimizer, scheduler, and run settings](docs/screenshot-training.png)

![Help text for a Conv2d padding parameter](docs/screenshot-param-help.png)
</details>

<details>
<summary>Generated PyTorch code</summary>

![Live code generation showing the equivalent nn.Module](docs/screenshot-code.png)
</details>

<details>
<summary>Custom blocks</summary>

![Create custom PyTorch modules with inline code editor](docs/screenshot-custom-blocks.png)
</details>

<details>
<summary>Light theme</summary>

![TensorBloom in light theme](docs/screenshot-light.png)
</details>

## Getting Started

### Download

Grab the latest installer from the [releases page](https://github.com/mdsunbeam/tensorbloom/releases/latest):

- **Windows** — `.exe` installer or `.msi`
- **Linux** — `.deb`, `.rpm`, or `.AppImage`

**Requirement:** Python 3.9+ must be installed. The editor opens right away; if PyTorch isn't set up yet, a card on the canvas offers to install it and shows its progress while you keep building.

### Building from Source

For development or contributing:

```bash
git clone https://github.com/mdsunbeam/tensorbloom.git
cd tensorbloom/app
npm install
npx tauri dev
```

Requires Node.js 18+, [Rust](https://rustup.rs/), and Python 3.9+.
**Windows:** also needs [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
**Linux:** `sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file libssl-dev libayatana-appindicator3-dev librsvg2-dev`

### Your First Model in 30 Seconds

1. On the empty canvas, pick **Simple MLP** from the welcome card (or go to **Insert > From Template** for all 16)

![The welcome card on an empty canvas, with three starter models](docs/screenshot-welcome.png)

2. Press **Train** in the toolbar (or `Ctrl+Enter`) and watch the loss drop and accuracy climb in real time

![Simple MLP after training, with the run summary and charts](docs/screenshot-training.png)

3. Press **Try it** in the run summary to see the model's predictions on digits it didn't train on

From here: click any node to edit parameters, swap the dataset, add layers, or try a different template. New to TensorBloom? **Help > Getting Started** walks through the steps.

## Templates

16 pre-built architectures, ready to train:

| Category | Templates |
|----------|-----------|
| Getting Started | Simple MLP |
| Vision | LeNet-5, ResNet-18, ViT, MobileNetV2, U-Net |
| Language | Transformer, nanoGPT, BERT Base, LSTM Classifier |
| Audio | Whisper Encoder, WaveNet Block |
| Generative | Autoencoder, Conv Autoencoder, Embedding Classifier |
| Other | DenseNet Block |

Go to **Insert > From Template** to load any template. Each card shows a preview of the model's layout.

![The templates dialog, with a layout preview for each model](docs/screenshot-template-architectures.png)

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Start training |
| `Ctrl+K` | Add any layer (spotlight search) |
| `Ctrl+S` / `Ctrl+Shift+S` | Save / Save As |
| `Ctrl+O` | Open project |
| `Ctrl+N` | New project |
| `Ctrl+Z` / `Ctrl+Shift+Z` or `Ctrl+Y` | Undo / Redo |
| `Ctrl+C` / `Ctrl+V` | Copy / Paste nodes |
| `Ctrl+D` | Duplicate selected |
| `Ctrl+A` | Select all |
| `Ctrl+G` / `Ctrl+Shift+G` | Group selected layers / Ungroup |
| `Ctrl+L` | Auto-layout graph |
| `Ctrl+I` | Import PyTorch code |
| `Ctrl+E` | Export model |
| `Ctrl+B` / `Ctrl+.` | Toggle side panels |
| `Ctrl+M` | Toggle minimap |
| `Ctrl+/` | Show all shortcuts |
| `Delete` | Delete selected |
| `Escape` | Close the open dialog or menu |
| Scroll / `Ctrl+Scroll` | Pan / Zoom |
| Double-click | Rename node |

## Known Limitations

TensorBloom v0.1 focuses on standard feedforward, convolutional, recurrent, and transformer architectures. The following are not yet supported:

- **GANs and adversarial training** — no multi-model training loops
- **Multi-loss training** — only one loss node per graph
- **Attention masks** — Transformer nodes don't accept mask inputs
- **Distributed training** — single GPU only (multi-GPU DDP planned for v0.4)
- **Dynamic architectures** — model structure is static at graph definition time

See the [Roadmap](ROADMAP.md) for planned features.

## Documentation

Full docs at **[tensorbloom.org/docs](https://tensorbloom.org/docs)**, or browse locally:

- [Getting Started](docs/guides/getting-started.md) — Build and train your first model
- [Custom Data Guide](docs/guides/custom-data.md) — Load your own datasets
- [Code Blocks](docs/guides/code-blocks.md) — Write custom forward() logic
- [Training Guide](docs/guides/training.md) — Training settings, the run flow, comparing runs, and trying a model
- [Exporting Models](docs/guides/export.md) — ONNX, TorchScript, and Python export
- [Template Reference](docs/guides/templates.md) — All 16 built-in architectures
- [Architecture](docs/guides/architecture.md) — Technical internals for contributors
- [Roadmap](ROADMAP.md) — Planned features and development direction

## Building for Production

```bash
cd app
npx tauri build
```

Produces platform-specific installers in `app/src-tauri/target/release/bundle/`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[MIT](LICENSE)
