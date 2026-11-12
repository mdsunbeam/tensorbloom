# Architecture

Technical overview of TensorBloom's internals for contributors and curious users.

## System Overview

```
┌────────────────────────────────────────────────────────────┐
│                       Tauri v2 Shell                       │
│  ┌────────────────────────┐    ┌────────────────────────┐  │
│  │  React Frontend        │    │  Python Sidecar        │  │
│  │  (Vite + TS)           │    │  (PyTorch)             │  │
│  │                        │    │                        │  │
│  │  Graph Editor          │◄──►│  Compile / Decompile   │  │
│  │  Property Panel        │JSON│  Train + layer stats   │  │
│  │  Code Generator        │-RPC│  Export / Predict      │  │
│  │  Shape Inference       │    │  Data Scan             │  │
│  │  Assistant, Lessons    │    │  LLM import/chat/LoRA  │  │
│  └────────────────────────┘    └────────────────────────┘  │
│  Rust: windows, sidecar IPC, Python setup, assistant API,  │
│        API key in the OS keychain, in-app updates          │
└────────────────────────────────────────────────────────────┘
```

## Frontend (`app/src/`)

### React + TypeScript + Vite

The frontend handles all UI rendering. Key frameworks:
- **React Flow** (xyflow) — Node graph rendering, edge routing, viewport controls
- **Zustand** — State management (graph, history, document, runs, layer stats, experiments, lessons, assistant, LLM, Python setup, updates, theme, training stores)
- **Tailwind CSS** — Styling

### Directory Structure

```
app/src/
├── components/          # React components
│   ├── ui/              # Shared kit: Button, Dialog, Tabs, Select, Segmented, HelpTip
│   ├── GraphEditor.tsx  # React Flow canvas, node/edge interaction
│   ├── PropertyPanel.tsx # Right panel — node properties, data config, training
│   ├── TrainingPanel.tsx # Training config form (optimizer, scheduler, etc.)
│   ├── NodePalette.tsx  # Left panel — layer browser
│   ├── Toolbar.tsx      # Menus, project name, Train/Stop and run status
│   ├── ProblemsButton.tsx # Toolbar list of shape problems with fixes
│   ├── WelcomeCard.tsx  # Empty-canvas starter models
│   ├── PythonSetupCard.tsx # Background Python/PyTorch setup with progress
│   ├── TemplatesDialog.tsx # Template cards with layout previews
│   ├── CodePanel.tsx    # Bottom panel — generated PyTorch code
│   ├── TrainingCharts.tsx # Live charts, run summary, error panel
│   ├── RunsPanel.tsx    # Run history: compare curves and settings
│   ├── TryModelDialog.tsx # Predictions from a trained model
│   ├── GroupPanel.tsx   # Properties for a layer group
│   ├── TrainingLogs.tsx # Training log output
│   ├── LayerOverlayControl.tsx # "Training on the graph" card (gradients/activations)
│   ├── LayerStatsCard.tsx # Per-layer hover card: gradient history, histogram, feature maps
│   ├── ExperimentBanner.tsx # "Change one thing" card
│   ├── RunChanges.tsx   # What changed between two runs, with the graph diff map
│   ├── LessonsDialog.tsx / LessonPanel.tsx # Guided lessons
│   ├── assistant/       # Assistant chat panel and the change-set review
│   ├── LLMPanel.tsx     # Language model tab: overview, chat, fine-tune
│   ├── ImportLLMDialog.tsx # Read an LLM's architecture from Hugging Face or a file
│   ├── DiagramExportDialog.tsx # 3D architecture diagram export
│   ├── UpdateBanner.tsx # In-app update offer
│   └── Toast.tsx        # Notification system
├── lib/                 # Core logic
│   ├── registry.ts      # 70+ PyTorch module definitions
│   ├── param-help.ts    # Help text for every parameter
│   ├── shape-inference.ts # Client-side dimension propagation
│   ├── shape-fixes.ts   # The fix for a shape problem (banner, list, pre-flight)
│   ├── decompiler.ts    # Graph → PyTorch code generation
│   ├── training.ts      # Training pipeline (preflight, graph → sidecar)
│   ├── training-errors.ts # Plain-language hints for training errors
│   ├── preflight-check.ts # Validation + suggested fixes before training
│   ├── groups.ts        # Layer groups (canvas-only frames)
│   ├── templates.ts     # 16 architecture templates
│   ├── project.ts       # Save/load .tbloom files
│   ├── document-actions.ts # New, open, save shared by menus and shortcuts
│   ├── auto-save.ts     # Recovery copy while there are unsaved changes
│   ├── auto-layout.ts   # Dagre-based graph layout
│   ├── layer-health.ts  # Dead units, vanishing/exploding gradient checks
│   ├── experiment-diff.ts # Graph and settings diff for "Change one thing"
│   ├── diagram-3d.ts    # 3D diagram layout and PDF/SVG/PNG rendering
│   ├── llm.ts / llm-runtime.ts # LLM config parsing, memory estimates, runtime calls
│   ├── assistant/       # Assistant agent loop, tools, prompt, change sets
│   ├── lessons/         # Lesson catalog, checks, and edits
│   ├── share.ts         # Share links (#m= fragment) for the web version
│   ├── platform.ts      # Desktop (Tauri) vs web
│   └── dataset-metadata.ts # Dataset definitions, shape resolution
├── nodes/
│   ├── TensorBloomNode.tsx # Custom React Flow node component
│   └── LayerGroupNode.tsx  # Group frame and collapsed card
└── store/
    ├── graph-store.ts   # Graph state (nodes, edges, groups, selection)
    ├── history-store.ts # Undo/redo (settled-state snapshots, with tests)
    ├── document-store.ts # File path and unsaved-changes tracking
    ├── run-store.ts     # Run history kept with the project
    ├── layer-stats-store.ts # Per-layer training stats for the graph overlay
    ├── experiment-store.ts # The "Change one thing" experiment in progress
    ├── lesson-store.ts  # The open lesson and its progress
    ├── assistant-store.ts # Assistant chat and pending changes
    ├── llm-store.ts     # Imported language model, chat and fine-tuning
    ├── update-store.ts  # In-app updates
    ├── python-store.ts  # Python environment status and setup progress
    ├── theme-store.ts   # Dark/light theme
    └── training-store.ts # Training state (status, metrics, config)
```

### State Flow

```
User action → Zustand store update → React re-render → React Flow update
                    ↓
            Shape inference runs → shapeMap updated → Nodes show shapes
                    ↓
            Code decompiler runs → Generated PyTorch code updates
```

## Rust Backend (`app/src-tauri/`)

The Rust layer is thin — it manages the window, the Python sidecar process, and anything that needs a secret:

- `sidecar_manager.rs` — Spawns Python process, pipes stdin/stdout, routes JSON-RPC messages
- `python_env.rs` — Installs Python and PyTorch with the bundled [uv](https://github.com/astral-sh/uv) binary (no Python needed first), adds the CUDA build on request, and streams `python-setup` progress events
- `assistant.rs` — Keeps the Anthropic API key in the OS keychain and sends the assistant's requests, so the interface never holds the key
- `lib.rs` — Tauri command handlers, GPU detection, and the updater and process plugins for in-app updates

## Python Sidecar (`python/`)

A JSON-RPC server (`tensorbloom_sidecar.py`) that handles all PyTorch operations, with language model support in `tensorbloom_llm.py` next to it:

### RPC Methods

| Method | Description |
|--------|-------------|
| `compile` | PyTorch code → graph JSON (via `torch.fx.symbolic_trace`) |
| `decompile` | Graph JSON → PyTorch code |
| `train.start` | Build model from graph, load dataset, train in background thread |
| `train.stop` | Signal training thread to stop |
| `data.scan` | Read tensor file metadata without loading all data |
| `data.probe` | Run one batch to discover output shape/dtype |
| `data.introspect` | Inspect HuggingFace dataset schema |
| `export.onnx` | Export to ONNX format |
| `export.torchscript` | Export to TorchScript |
| `model.predict` | Run a trained model on validation samples or an image file |
| `node.probe` | Run a code block on dummy tensors to find its output shapes |
| `system.gpu_info` | Return CUDA device info |
| `system.environment` | PyTorch, CUDA/MPS and memory details |
| `validate` | Check graph for structural errors |
| `llm.inspect` | Read an LLM's architecture from Hugging Face or a file, without the weights |
| `llm.install` / `llm.status` | Install and check the extra packages for language models |
| `llm.load` / `llm.unload` / `llm.generate` / `llm.stop` | Load a model, chat with it (streamed as `llm.token`) |
| `llm.finetune.start` | LoRA, QLoRA or full fine-tuning with progress notifications |

### Model Building

The sidecar builds a runnable `nn.Module` from graph JSON using `torch.fx.Graph`:

1. Topological sort of nodes
2. Create `nn.Module` submodules for each computation node
3. Build `torch.fx.Graph` with proper input/output routing
4. Handle special cases: Add, Concatenate, LSTM (output extraction), TransformerDecoderLayer (self-attention mode)
5. Wrap in `torch.fx.GraphModule` → ready to train

### Training Loop

1. Build model from graph
2. Detect loss function from graph (Loss node)
3. Load dataset (presets, custom tensors, HuggingFace, CSV, ImageFolder)
4. Train with gradient clipping, mixed precision, early stopping, checkpointing
5. Report progress via JSON-RPC notifications (`train.progress`, `train.log`), and per-layer gradients, output histograms and feature maps every 50 batches (`train.layers`)

## Module Registry

70+ PyTorch modules defined in `registry.ts`, organized by category:

**I/O:** Input, Output, Data
**Linear:** Linear, Identity
**Convolution:** Conv1d, Conv2d, Conv3d, ConvTranspose2d
**Pooling:** MaxPool2d, AvgPool2d, AdaptiveAvgPool1d/2d
**Normalization:** BatchNorm1d/2d, LayerNorm, RMSNorm, InstanceNorm
**Activation:** ReLU, GELU, Tanh, Sigmoid, Softmax, LeakyReLU, ELU, etc.
**Recurrent:** LSTM, GRU
**Transformer:** TransformerEncoderLayer, TransformerDecoderLayer, Transformer
**Dropout:** Dropout, Dropout2d
**Embedding:** Embedding
**Loss:** CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, L1Loss, and more
**Reshape:** Flatten, Unflatten, Permute, Squeeze, Unsqueeze, ZeroPad2d
**Math:** Add, Concatenate, MatMul

Each entry defines: type, label, category, description, parameters (with types, defaults, constraints), and input/output handles.
