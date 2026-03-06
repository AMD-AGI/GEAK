#!/bin/bash
set -e

# Build directory
BUILD_DIR="build"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Compile each example
echo "Building gemm_add examples..."

# XDL FP16
hipcc -I/opt/rocm/include -O3 --offload-arch=gfx942 \
  ../gemm_add_xdl_fp16.cpp -o example_gemm_add_xdl_fp16 \
  -L/opt/rocm/lib -lamdhip64

# XDL BF16
hipcc -I/opt/rocm/include -O3 --offload-arch=gfx942 \
  ../gemm_add_xdl_bf16.cpp -o example_gemm_add_xdl_bf16 \
  -L/opt/rocm/lib -lamdhip64

# WMMA FP16
hipcc -I/opt/rocm/include -O3 --offload-arch=gfx942 \
  ../gemm_add_wmma_fp16.cpp -o example_gemm_add_wmma_fp16 \
  -L/opt/rocm/lib -lamdhip64

# WMMA BF16
hipcc -I/opt/rocm/include -O3 --offload-arch=gfx942 \
  ../gemm_add_wmma_bf16.cpp -o example_gemm_add_wmma_bf16 \
  -L/opt/rocm/lib -lamdhip64

echo "Build complete!"
cd ..

# Run correctness tests
echo ""
echo "Running correctness tests..."
./build/example_gemm_add_xdl_fp16 1 1 0
./build/example_gemm_add_wmma_fp16 1 1 0

# Run benchmarks
echo ""
echo "Running benchmarks..."
echo "=== XDL FP16 Benchmark ==="
./build/example_gemm_add_xdl_fp16 1 1 1

echo ""
echo "=== WMMA FP16 Benchmark ==="
./build/example_gemm_add_wmma_fp16 1 1 1

echo ""
echo "All tests completed successfully!"
