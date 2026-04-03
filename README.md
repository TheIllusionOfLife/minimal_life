# Minimal Life

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19351500.svg)](https://doi.org/10.5281/zenodo.19351500)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18780935.svg)](https://doi.org/10.5281/zenodo.18780935)

Minimal Life is an artificial life research codebase investigating the minimal diagnostic principles of life: finding 2–4 surrogate observables that predict life-likeness as well as the full seven-criteria system does.

The repository is a Rust workspace with optional Python bindings.

## Quick Start

### Prerequisites

- Rust stable toolchain
- `uv` for Python environment and packaging tasks
- `tectonic` for LaTeX paper compilation

### Build

```bash
cargo build --workspace
```

### Test and Lint

```bash
./scripts/check.sh
```

### Python Script Lint/Test

```bash
uv run ruff check scripts tests_python
uv run pytest tests_python
uv run python scripts/check_manuscript_consistency.py
```

### Config Compatibility Note

- Scheduled ablation targets are enum-backed (`ablation_targets`) and must be one of:
  `metabolism`, `boundary`, `homeostasis`, `response`, `reproduction`, `evolution`, `growth`.
- Unknown target values now fail during config deserialization instead of later runtime validation.

### Run the CLI

```bash
cargo run -p minimal-life-cli --release
```

### Build Python Extension (local)

```bash
uv run maturin develop --manifest-path crates/minimal-life-py/Cargo.toml
```

Then in Python:

```python
import minimal_life
print(minimal_life.version())
```

## Citation

If you use this software, please cite the accompanying paper. See `CITATION.cff` for details.

## Architecture (High-Level)

- `crates/minimal-life-core`: simulation core (world, metabolism, genome, NN, spatial systems)
- `crates/minimal-life-py`: PyO3 bindings exposing core functions to Python
- `crates/minimal-life-cli`: executable benchmark/feasibility experiment runner
- `python/minimal_life`: Python package surface for the extension module

## Development Workflow

- Create feature branches from `main`
- Keep commits focused and test-backed
- Open PRs against `main` with test evidence (`fmt`, `clippy`, `test`)

## Current Status

Software accompanying the paper "Minimal Diagnostic Principles of Life: Criterion-Ablation and Observable Surrogate Selection in a Digital Ecosystem" submitted to ALIFE 2026.
