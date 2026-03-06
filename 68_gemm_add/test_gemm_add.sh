#!/bin/bash
set -e

# Absolute path to repository
REPO_DIR="/workspace/GEAK/68_gemm_add"
cd "$REPO_DIR"

# Build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "=========================================="
echo "Building gemm_add kernel variants..."
echo "=========================================="

# Build XDL FP16
echo "Building XDL FP16..."
hipcc -I/opt/rocm/include -O3 --offload-arch=gfx942 \
  ../gemm_add_xdl_fp16.cpp -o example_gemm_add_xdl_fp16 \
  -L/opt/rocm/lib -lamdhip64 -lutility

# Build XDL BF16
echo "Building XDL BF16..."
hipcc -I/opt/rocm/include -O3 --offload-arch=gfx942 \
  ../gemm_add_xdl_bf16.cpp -o example_gemm_add_xdl_bf16 \
  -L/opt/rocm/lib -lamdhip64 -lutility

# Try to build WMMA variants (may fail on some CK versions)
echo "Attempting to build WMMA FP16..."
if hipcc -I/opt/rocm/include -O3 --offload-arch=gfx942 \
  ../gemm_add_wmma_fp16.cpp -o example_gemm_add_wmma_fp16 \
  -L/opt/rocm/lib -lamdhip64 -lutility 2>/dev/null; then
    echo "WMMA FP16 build succeeded"
    WMMA_FP16_AVAILABLE=1
else
    echo "WMMA FP16 build failed (may not be supported in this CK version)"
    WMMA_FP16_AVAILABLE=0
fi

echo "Attempting to build WMMA BF16..."
if hipcc -I/opt/rocm/include -O3 --offload-arch=gfx942 \
  ../gemm_add_wmma_bf16.cpp -o example_gemm_add_wmma_bf16 \
  -L/opt/rocm/lib -lamdhip64 -lutility 2>/dev/null; then
    echo "WMMA BF16 build succeeded"
    WMMA_BF16_AVAILABLE=1
else
    echo "WMMA BF16 build failed (may not be supported in this CK version)"
    WMMA_BF16_AVAILABLE=0
fi

echo ""
echo "Build complete!"
echo ""

cd "$REPO_DIR"

# Run correctness tests with smaller sizes for faster execution
echo "=========================================="
echo "Running Correctness Tests..."
echo "=========================================="

echo ""
echo "--- XDL FP16 Correctness (M=1024, N=1024, K=1024) ---"
./build/example_gemm_add_xdl_fp16 1 1 0 1024 1024 1024 1024 1024 1024 1024
if [ $? -eq 0 ]; then
    echo "✓ XDL FP16 correctness PASSED"
else
    echo "✗ XDL FP16 correctness FAILED"
    exit 1
fi

echo ""
echo "--- XDL BF16 Correctness (M=1024, N=1024, K=1024) ---"
./build/example_gemm_add_xdl_bf16 1 1 0 1024 1024 1024 1024 1024 1024 1024
if [ $? -eq 0 ]; then
    echo "✓ XDL BF16 correctness PASSED"
else
    echo "✗ XDL BF16 correctness FAILED"
    exit 1
fi

if [ $WMMA_FP16_AVAILABLE -eq 1 ]; then
    echo ""
    echo "--- WMMA FP16 Correctness ---"
    ./build/example_gemm_add_wmma_fp16 1 1 0 1024 1024 1024 1024 1024 1024 1024
    if [ $? -eq 0 ]; then
        echo "✓ WMMA FP16 correctness PASSED"
    else
        echo "✗ WMMA FP16 correctness FAILED"
        exit 1
    fi
fi

if [ $WMMA_BF16_AVAILABLE -eq 1 ]; then
    echo ""
    echo "--- WMMA BF16 Correctness ---"
    ./build/example_gemm_add_wmma_bf16 1 1 0 1024 1024 1024 1024 1024 1024 1024
    if [ $? -eq 0 ]; then
        echo "✓ WMMA BF16 correctness PASSED"
    else
        echo "✗ WMMA BF16 correctness FAILED"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "All Correctness Tests PASSED!"
echo "=========================================="

# Run benchmarks with timing enabled
echo ""
echo "=========================================="
echo "Running Benchmarks..."
echo "=========================================="

echo ""
echo "--- XDL FP16 Benchmark (M=1024, N=1024, K=1024) ---"
./build/example_gemm_add_xdl_fp16 1 1 1 1024 1024 1024 1024 1024 1024 1024

echo ""
echo "--- XDL FP16 Benchmark (M=2048, N=2048, K=2048) ---"
./build/example_gemm_add_xdl_fp16 1 1 1 2048 2048 2048 2048 2048 2048 2048

echo ""
echo "--- XDL FP16 Benchmark (Default: M=3840, N=4096, K=4096) ---"
./build/example_gemm_add_xdl_fp16 1 1 1

echo ""
echo "--- XDL BF16 Benchmark (M=2048, N=2048, K=2048) ---"
./build/example_gemm_add_xdl_bf16 1 1 1 2048 2048 2048 2048 2048 2048 2048

if [ $WMMA_FP16_AVAILABLE -eq 1 ]; then
    echo ""
    echo "--- WMMA FP16 Benchmark (M=2048, N=2048, K=2048) ---"
    ./build/example_gemm_add_wmma_fp16 1 1 1 2048 2048 2048 2048 2048 2048 2048
fi

if [ $WMMA_BF16_AVAILABLE -eq 1 ]; then
    echo ""
    echo "--- WMMA BF16 Benchmark (M=2048, N=2048, K=2048) ---"
    ./build/example_gemm_add_wmma_bf16 1 1 1 2048 2048 2048 2048 2048 2048 2048
fi

echo ""
echo "=========================================="
echo "All Tests and Benchmarks Completed Successfully!"
echo "=========================================="
