#!/bin/bash
set -e

REPO_DIR="/workspace/GEAK/68_gemm_add"
cd "$REPO_DIR"

# Build
echo "Building gemm_add kernels..."
mkdir -p build
cd build
hipcc -I/opt/rocm/include -O3 --offload-arch=gfx942 \
  ../gemm_add_xdl_fp16.cpp -o example_gemm_add_xdl_fp16 \
  -L/opt/rocm/lib -lamdhip64 -lutility
cd ..

# Correctness test
echo ""
echo "Running correctness test..."
./build/example_gemm_add_xdl_fp16 1 1 0 1024 1024 1024 1024 1024 1024 1024
if [ $? -eq 0 ]; then
    echo "✓ Correctness PASSED"
else
    echo "✗ Correctness FAILED"
    exit 1
fi

# Benchmark test
echo ""
echo "Running benchmark..."
./build/example_gemm_add_xdl_fp16 1 1 1 2048 2048 2048 2048 2048 2048 2048

echo ""
echo "All tests completed successfully!"
