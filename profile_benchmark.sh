#!/bin/bash
set -e

# Setup environment
source ~/.cargo/env 2>/dev/null || true
source .venv/bin/activate

# Install py-spy if not already installed
pip install -q py-spy

# Build Rust extension with frame pointers for better profiling
echo "Building Rust extension with profiling symbols..."
RUSTFLAGS="-C force-frame-pointers=yes" maturin develop --release --features curvine

# Profile with py-spy and generate flamegraph directly as SVG
echo "Profiling benchmark with py-spy (this will take a moment)..."
py-spy record --native --rate 100 --format flamegraph -o flamegraph.svg -- python tests/bench_curvine.py

echo "✅ Flamegraph saved to flamegraph.svg"
echo "   Open it in your browser to view the profile!"
