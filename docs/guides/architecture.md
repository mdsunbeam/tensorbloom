# Architecture

Technical overview of TensorBloom's internals for contributors and curious users.

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                   Tauri v2 Shell                     │
│  ┌──────────────────────┐  ┌──────────────────────┐  │
│  │   React Frontend     │  │   Python Sidecar     │  │
│  │   (Vite + TS)        │  │   (PyTorch)          │  │
│  │                      │  │                      │  │
│  │  Graph Editor         │◄─JSON-RPC─►│  Compile     │  │
│  │  Property Panel      │  over stdio │  Train       │  │
│  │  Code Generator      │            │  Export      │  │
│  │  Shape Inference     │            │  Data Scan   │  │
│  └──────────────────────┘  └──────────────────────┘  │
│              Rust (window management, sidecar IPC)    │
└─────────────────────────────────────────────────────┘
```

## Frontend (`app/src/`)

### React + TypeScript + Vite

The frontend handles all UI rendering. Key frameworks:
- **React Flow** (xyflow) — Node graph rendering, edge routing, viewport controls
- **Zustand** — State management (graph, history, document, runs, Python setup, theme, training stores)
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
│   └── dataset-metadata.ts # Dataset definitions, shape resolution
├── nodes/
│   ├── TensorBloomNode.tsx # Custom React Flow node component
│   └── LayerGroupNode.tsx  # Group frame and collapsed card
└── store/
    ├── graph-store.ts   # Graph state (nodes, edges, groups, selection)
    ├── history-store.ts # Undo/redo (settled-state snapshots, with tests)
    ├── document-store.ts # File path and unsaved-changes tracking
    ├── run-store.ts     # Run history kept with the project
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

The Rust layer is thin — it manages the window and the Python sidecar process:

- `sidecar_manager.rs` — Spawns Python process, pipes stdin/stdout, routes JSON-RPC messages
- `python_env.rs` — Provisions a Python venv with PyTorch in the background, streaming `python-setup` progress events
- `lib.rs` — Tauri command handlers, GPU detection

## Python Sidecar (`python/`)

A single-file JSON-RPC server that handles all PyTorch operations:

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
| `system.gpu_info` | Return CUDA device info |
| `validate` | Check graph for structural errors |

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
5. Report progress via JSON-RPC notifications (`train.progress`, `train.log`)

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
