#!/bin/bash

# Get the directory where this script is located
DIST_DIR="../distributed-cache/dist/"

# Set library paths for custom-built RDMA libraries under ./dist
export LD_LIBRARY_PATH="${DIST_DIR}/fabric/gdrcopy/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${DIST_DIR}/fabric/libfabric/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Run curvine CLI
export DATALOADER_TEST="python tests/bench_curvine.py"

# Run the CLI
${DATALOADER_TEST} "$@"