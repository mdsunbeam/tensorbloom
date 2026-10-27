# Contributing to TensorBloom

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

1. Install prerequisites: Node.js 18+, Rust (stable), Python 3.9+
2. Clone and install:
   ```bash
   git clone https://github.com/mdsunbeam/tensorbloom.git
   cd tensorbloom/app
   npm install
   npx tauri dev
   ```
3. The app auto-provisions a Python venv with PyTorch on first launch.

## Project Structure

- **`app/src/`** — React + TypeScript frontend (Vite, Tailwind, React Flow)
- **`app/src-tauri/`** — Rust backend (Tauri v2, sidecar management)
- **`python/`** — Python sidecar (PyTorch operations via JSON-RPC)

## Making Changes

1. Create a branch from `main`
2. Make your changes
3. Run `npm run build` in `app/` to verify TypeScript compiles
4. Run `python python/test_templates.py` to verify template models
5. Run `python python/test_sidecar.py` to verify training, checkpoints, and export
6. Open a pull request

## Guidelines

- **Keep PRs focused** — one feature or fix per PR
- **Test what you change** — if you modify templates or shape inference, run the Python tests
- **Match existing style** — no linter is configured yet, so follow the patterns in surrounding code
- **Don't add unnecessary dependencies** — the app is intentionally lightweight

## Areas for Contribution

- **New PyTorch modules** — add entries to `app/src/lib/registry.ts`
- **New templates** — add architectures to `app/src/lib/templates.ts`
- **Shape inference** — improve coverage in `app/src/lib/shape-inference.ts`
- **Dataset support** — add datasets in `python/tensorbloom_sidecar.py`
- **Bug fixes** — check the issues tab
- **Documentation** — README improvements, usage guides, screenshots

## Reporting Issues

Please include:
- What you did (steps to reproduce)
- What you expected
- What happened instead
- Your OS and Python/PyTorch versions
