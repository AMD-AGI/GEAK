---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "3"
category: "blas"
tags: ["cublas", "linear-algebra", "matrix-operations", "blas"]
cuda_version: "13.0+"
last_updated: 2025-11-17
---

# cuBLAS Usage Guide

*GPU-accelerated Basic Linear Algebra Subprograms (BLAS) library*

## Overview

cuBLAS provides GPU-accelerated implementations of standard BLAS operations for linear algebra computations.

**Official Documentation**: [cuBLAS Library Documentation](https://docs.nvidia.com/cuda/cublas/)

## Installation

cuBLAS is included with CUDA Toolkit:

```bash
# Verify installation
ls /usr/local/cuda/lib64/libcublas*

# Check version
cat /usr/local/cuda/include/cublas_v2.h | grep "CUBLAS_VER"
```

## Basic Usage

### Library Initialization

```cpp
#include <cublas_v2.h>

int main() {
    cublasHandle_t handle;
    
    // Create handle
    cublasCreate(&handle);
    
    // Perform operations...
    
    // Destroy handle
    cublasDestroy(handle);
    
    return 0;
}
```

**Compile:**
```bash
nvcc -lcublas program.cu -o program
```

## Common Operations

### Vector Operations (Level 1)

#### Vector Addition (AXPY): `y = alpha*x + y`

```cpp
float *d_x, *d_y;
int n = 1000;
float alpha = 2.0f;

// Allocate and initialize vectors
cudaMalloc(&d_x, n * sizeof(float));
cudaMalloc(&d_y, n * sizeof(float));

// Perform AXPY
cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);
```

#### Dot Product

```cpp
float result;
cublasSdot(handle, n, d_x, 1, d_y, 1, &result);
```

### Matrix-Vector Operations (Level 2)

#### General Matrix-Vector Multiplication (GEMV): `y = alpha*A*x + beta*y`

```cpp
int m = 1000, n = 500;  // A is m x n
float alpha = 1.0f, beta = 0.0f;
float *d_A, *d_x, *d_y;

// A stored in column-major order
cublasSgemv(handle, CUBLAS_OP_N, m, n,
            &alpha, d_A, m,  // Leading dimension = m
            d_x, 1,
            &beta, d_y, 1);
```

### Matrix-Matrix Operations (Level 3)

#### General Matrix Multiplication (GEMM): `C = alpha*A*B + beta*C`

```cpp
int m = 1024, n = 1024, k = 1024;  // C is m x n
float alpha = 1.0f, beta = 0.0f;
float *d_A, *d_B, *d_C;

// C = A * B
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            d_A, m,  // Leading dimension of A
            d_B, k,  // Leading dimension of B
            &beta,
            d_C, m);  // Leading dimension of C
```

## Data Types

### Precision Types

| Function Prefix | Precision | Example |
|----------------|-----------|---------|
| `S` | Single (FP32) | `cublasSgemm` |
| `D` | Double (FP64) | `cublasDgemm` |
| `C` | Complex single | `cublasCgemm` |
| `Z` | Complex double | `cublasZgemm` |
| `H` | Half (FP16) | `cublasHgemm` |

### Mixed Precision

```cpp
// FP16 input, FP32 compute (on Volta+)
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             m, n, k,
             &alpha,
             d_A, CUDA_R_16F, lda,  // FP16 input
             d_B, CUDA_R_16F, ldb,
             &beta,
             d_C, CUDA_R_32F, ldc,  // FP32 output
             CUBLAS_COMPUTE_32F,    // FP32 compute
             CUBLAS_GEMM_DEFAULT);
```

## Tensor Core Acceleration

### Enable Tensor Cores (Volta+)

```cpp
// Set math mode to use Tensor Cores
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// TF32 for FP32 inputs (Ampere+)
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

// Perform GEMM - automatically uses Tensor Cores
cublasSgemm(handle, ...);
```

### Mixed Precision with Tensor Cores

```cpp
__half *d_A, *d_B, *d_C;  // FP16 matrices
float alpha = 1.0f, beta = 0.0f;

// FP16 GEMM with Tensor Cores
cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             m, n, k,
             &alpha,
             d_A, CUDA_R_16F, lda,
             d_B, CUDA_R_16F, ldb,
             &beta,
             d_C, CUDA_R_16F, ldc,
             CUBLAS_COMPUTE_16F,
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

## Performance Optimization

### Batched Operations

Process multiple small matrices efficiently:

```cpp
// Batch of matrix multiplications
int batchCount = 100;
float **d_A_array, **d_B_array, **d_C_array;

cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   m, n, k,
                   &alpha,
                   d_A_array, lda,
                   d_B_array, ldb,
                   &beta,
                   d_C_array, ldc,
                   batchCount);
```

### Strided Batched Operations

For regularly spaced matrices:

```cpp
// Matrices are consecutive in memory
long long int strideA = m * k;
long long int strideB = k * n;
long long int strideC = m * n;

cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          m, n, k,
                          &alpha,
                          d_A, lda, strideA,
                          d_B, ldb, strideB,
                          &beta,
                          d_C, ldc, strideC,
                          batchCount);
```

### Asynchronous Execution

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// Set stream for cuBLAS operations
cublasSetStream(handle, stream);

// Operations execute asynchronously on stream
cublasSgemm(handle, ...);

// Wait for completion
cudaStreamSynchronize(stream);
```

## Complete Example

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

int main() {
    int m = 1024, n = 1024, k = 1024;
    float alpha = 1.0f, beta = 0.0f;
    
    // Allocate host memory
    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));
    
    // Initialize matrices
    for (int i = 0; i < m * k; i++) h_A[i] = 1.0f;
    for (int i = 0; i < k * n; i++) h_B[i] = 1.0f;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Enable Tensor Cores
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    
    // Perform matrix multiplication: C = A * B
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                d_A, m,
                d_B, k,
                &beta,
                d_C, m);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify result (should be k = 1024)
    printf("C[0] = %f (expected 1024.0)\n", h_C[0]);
    
    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

Compile and run:
```bash
nvcc -lcublas matrix_mul.cu -o matrix_mul
./matrix_mul
```

## Python Usage (via PyTorch/CuPy)

### PyTorch

```python
import torch

# cuBLAS automatically used for matmul on CUDA tensors
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')

# Uses cuBLAS GEMM internally
C = torch.matmul(A, B)

# Batch matrix multiplication
A_batch = torch.randn(32, 1024, 1024, device='cuda')
B_batch = torch.randn(32, 1024, 1024, device='cuda')
C_batch = torch.bmm(A_batch, B_batch)  # Batched GEMM
```

### CuPy

```python
import cupy as cp

# Direct cuBLAS calls via CuPy
A = cp.random.randn(1024, 1024, dtype=cp.float32)
B = cp.random.randn(1024, 1024, dtype=cp.float32)

# Matrix multiplication (uses cuBLAS)
C = cp.matmul(A, B)

# Or use @ operator
C = A @ B
```

## Common Issues

### Issue: Incorrect Results

**Cause**: Column-major vs row-major ordering

**Solution:**
```cpp
// cuBLAS uses column-major (Fortran) ordering
// For row-major C matrices, transpose:
// C^T = B^T * A^T
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,  // Swapped m and n
            &alpha,
            d_B, n,   // B comes first
            d_A, k,   // A comes second
            &beta,
            d_C, n);
```

### Issue: Poor Performance

**Cause**: Not using Tensor Cores

**Solution:**
```cpp
// Enable Tensor Core math mode
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// Or for Ampere+ with TF32
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
```

### Issue: Memory Alignment

**Cause**: Unaligned memory access

**Solution:**
```cpp
// Use aligned memory allocation
float *d_A;
cudaMalloc(&d_A, m * k * sizeof(float));  // Automatically aligned

// For manual allocation, ensure 256-byte alignment
```

## Benchmarking

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>

float benchmark_gemm(int m, int n, int k, int iterations) {
    // Setup...
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    cublasSgemm(handle, ...);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // GFLOPS = (2*m*n*k operations) / (time in seconds) / 1e9
    float gflops = (2.0f * m * n * k * iterations) / (milliseconds / 1000.0f) / 1e9;
    
    return gflops;
}
```

## Best Practices

1. **Use Tensor Cores**: Enable with `cublasSetMathMode()`
2. **Batch operations**: Use batched GEMM for multiple small matrices
3. **Async execution**: Use streams for concurrency
4. **Appropriate precision**: Use FP16/TF32 for training, FP32 for accuracy
5. **Reuse handles**: Create handle once, reuse for all operations

## External Resources

- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [cuBLAS API Reference](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm)
- [Matrix Multiplication Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/)

## Related Guides

- [CUDA Programming Basics](../../layer-2-compute-stack/cuda/cuda-basics.md)
- [Tensor Core Programming](../../layer-5-llm/05-advanced/custom-kernels/cuda-kernels.md)
- [PyTorch with CUDA](../../layer-4-frameworks/pytorch/pytorch-cuda-basics.md)

