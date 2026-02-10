# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

data-jet is a PyTorch-compatible DataLoader implemented in Rust with Python bindings. It implements its own sampling/batching/collation (inspired by PyTorch's DataLoader), and exposes results to Python as numpy arrays via **zero-copy ownership transfer** through PyO3.

## Build & Dev Commands

```bash
# Setup (first time)
uv venv .venv
source .venv/bin/activate
uv pip install torch numpy pytest maturin
# Build the Rust extension into the venv (must rebuild after any Rust change)
source .venv/bin/activate && maturin develop --release

# Run all tests
source .venv/bin/activate && python -m pytest tests/ -v

# Run a single test
source .venv/bin/activate && python -m pytest tests/test_dataloader.py::TestDataLoader::test_zero_copy -v

# Rust-only check (faster iteration on Rust code)
cargo check
```

## Architecture

### Two-layer design

1. **Rust core** (`src/lib.rs`) — PyO3 module `data_jet._core` compiled to a `.so`. Contains:
   - `TensorDataset` — implements the `Len` trait over `Vec<Vec<f64>>` features + `Vec<f64>` labels
   - `FlatCollate` — implements the `Collate` trait, produces `FlatBatch` (contiguous `Vec<f64>` + shape metadata) instead of ndarray types
   - `RustDataLoader` — `#[pyclass]` that stores dataset and config; `__iter__` uses samplers to build batches in Rust, returns `DataJetIterator`
   - `DataJetIterator` — `#[pyclass]` iterator; each `__next__` converts one Rust-owned `Vec<f64>` to `numpy.ndarray` via `PyArray1::from_vec` (zero-copy) then `.reshape()` (view, no copy)

2. **Python wrapper** (`python/data_jet/dataloader.py`) — `DataLoader` class that wraps `RustDataLoader`, converts numpy arrays to `torch.Tensor` via `torch.from_numpy` (zero-copy), handles device placement.

### Zero-copy data path

`Vec<f64>` (Rust) → `PyArray1::from_vec` (ownership transfer, no memcpy) → `.reshape()` (view) → `torch.from_numpy` (shared memory, no memcpy)

### Inlined sampler/collate code

The sampling and collation traits (`Len`, `Sampler`, `Collate`, `BatchSampler`, `SequentialSampler`, `RandomSampler`) live in `src/sampler.rs` and `src/collate.rs`. These were extracted from the `ai-dataloader` crate — only the subset actually used by data-jet was kept.

## Python API

```python
from data_jet import DataLoader  # PyTorch-compatible (returns torch.Tensor)
from data_jet import RustDataLoader  # Low-level (returns numpy.ndarray)
```

`DataLoader(features, labels, batch_size=32, shuffle=True, drop_last=False, device="cpu")`
