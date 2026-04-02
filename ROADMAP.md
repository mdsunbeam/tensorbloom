# Roadmap

TensorBloom's planned development path. Features are grouped by milestone, not ordered by priority within each milestone.

## v0.2 — Data Pipeline

- [ ] **Data store** — persistent dataset management: download once, reuse across projects, track dataset versions
- [ ] **Visual augmentation graph** — drag-and-drop data preprocessing: resize, normalize, random crop, color jitter as connectable nodes with live preview
- [ ] **Data preview** — sample visualizations in the property panel: thumbnail grid for images, text snippets for NLP, waveform plots for audio
- [ ] **Full HuggingFace support** — complex/nested columns, streaming datasets, preprocessing UI with code editor, tested dataset catalog

## v0.3 — Advanced Architectures

- [ ] **Multi-computation graphs** — separate graphs for encoder, decoder, discriminator that compose into a training pipeline (enables GANs, VAEs, teacher-student)
- [ ] **Multi-loss training** — multiple loss nodes with configurable weights (e.g., reconstruction loss + KL divergence for VAE)
- [ ] **Attention mask support** — pass masks through Transformer nodes for proper padding handling
- [ ] **Subgraph nodes** — collapse a group of nodes into a reusable, parameterized block that can be instantiated multiple times

## v0.4 — Training at Scale

- [ ] **Distributed training** — multi-GPU training with DDP, automatic device placement
- [ ] **Gradient accumulation** — effective batch size scaling without increasing VRAM
- [ ] **Learning rate warmup** — configurable warmup schedules (linear, cosine)
- [ ] **Experiment tracking** — Weights & Biases and TensorBoard integration
- [ ] **Checkpoint management** — visual checkpoint browser, resume from any epoch, compare checkpoints

## v0.5 — Deployment

- [ ] **ONNX optimization** — one-click quantization, operator fusion, graph optimization
- [ ] **Model serving** — local REST endpoint for inference testing
- [ ] **Edge export** — TensorFlow Lite, CoreML, and ONNX.js conversion
- [ ] **Model card generation** — automatic documentation with architecture diagram, training metrics, dataset info

## Future Ideas

- [ ] Collaborative editing — multiplayer graph editing with presence indicators
- [ ] Plugin system — custom node types, data loaders, and training hooks
- [ ] Model zoo — community-shared architectures and pretrained weights
- [ ] Automatic architecture search (NAS) — define search space visually, let the system find optimal architectures
- [ ] Python notebook integration — import/export `.ipynb` files
- [ ] Graph version control — diff and merge model architectures like code
- [ ] Hyperparameter sweep UI — grid/random/Bayesian search with parallel runs (currently experimental)

## Contributing

Want to work on any of these? Check the [issues](https://github.com/mdsunbeam/tensorbloom/issues) or open a new one to discuss your approach. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.
