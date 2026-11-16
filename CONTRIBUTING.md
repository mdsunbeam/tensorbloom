# Contributing to TensorBloom

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

1. Install prerequisites: Node.js 18+ and Rust (stable). Python 3.10+ is only needed to run the Python tests.
2. Clone and install:
   ```bash
   git clone https://github.com/mdsunbeam/tensorbloom.git
   cd tensorbloom/app
   npm install
   npx tauri dev
   ```
3. `tauri dev` downloads the bundled uv binary into `app/src-tauri/binaries/` (see `app/scripts/fetch-uv.mjs`). On first launch the app uses it to download a managed Python and install PyTorch into its own venv.

## Project Structure

- **`app/src/`** — React + TypeScript frontend (Vite, Tailwind, React Flow)
- **`app/src-tauri/`** — Rust backend (Tauri v2, sidecar management)
- **`python/`** — Python sidecar (PyTorch operations via JSON-RPC); language model support in `tensorbloom_llm.py`
- **`docs/guides/`** — The user guides, also published at [tensorbloom.org/docs](https://tensorbloom.org/docs)

See the [Architecture guide](docs/guides/architecture.md) for how the pieces fit together.

## Making Changes

1. Create a branch from `main`
2. Make your changes
3. Run `npm run build` in `app/` to verify TypeScript compiles
4. Run `npm test` in `app/` to run the frontend tests (shape inference, code generation, history, lessons, the assistant's tools, and more)
5. Run `python python/test_templates.py` to verify template models
6. Run `python python/test_sidecar.py` to verify training, checkpoints, and export
7. If you changed language model support, run `python python/test_llm.py` and `python python/test_llm_runtime.py`
8. If you changed what users see, update the guide in `docs/guides/` and any screenshot it shows
9. Open a pull request

## Guidelines

- **Keep PRs focused** — one feature or fix per PR
- **Test what you change** — if you modify templates or shape inference, run the Python tests
- **Match existing style** — no linter is configured yet, so follow the patterns in surrounding code
- **Don't add unnecessary dependencies** — the app is intentionally lightweight

## Areas for Contribution

- **New PyTorch modules** — add entries to `app/src/lib/registry.ts`, then run `npm run docs:modules` in `app/` to update the [module reference](docs/guides/modules.md)
- **New templates** — add architectures to `app/src/lib/templates.ts`
- **New lessons** — add them to `app/src/lib/lessons/catalog.ts`; see the [Lessons guide](docs/guides/lessons.md#adding-a-lesson)
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
