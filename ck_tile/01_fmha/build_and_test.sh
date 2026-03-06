#!/bin/bash
set -e

REPO_ROOT="/workspace/composable_kernel"
BUILD_DIR="${REPO_ROOT}/build"
BIN_DIR="${BUILD_DIR}/bin"

cd "${BUILD_DIR}"

echo "=== Building FMHA Forward Example (this may take several minutes) ==="
timeout 300 make -j4 tile_example_fmha_fwd || {
    echo "Build timed out or failed"
    exit 1
}

if [ ! -f "${BIN_DIR}/tile_example_fmha_fwd" ]; then
    echo "Binary not found at ${BIN_DIR}/tile_example_fmha_fwd"
    exit 1
fi

echo ""
echo "=== Running Correctness Test ==="
"${BIN_DIR}/tile_example_fmha_fwd" -mode=0 -b=1 -h=2 -s=128 -d=64 -v=1 -prec=fp16 -repeat=5

echo ""
echo "=== Running Benchmark Test ==="
"${BIN_DIR}/tile_example_fmha_fwd" -mode=0 -b=2 -h=8 -s=512 -d=128 -v=0 -prec=fp16 -repeat=20

echo ""
echo "=== ALL TESTS PASSED ==="
