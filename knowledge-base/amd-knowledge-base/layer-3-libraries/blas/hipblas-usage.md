---
layer: "3"
category: "blas"
subcategory: "portable-blas"
tags: ["hipblas", "blas", "portability", "rocblas"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
---

# hipBLAS Usage Guide

hipBLAS provides a portable BLAS interface that works on both AMD and NVIDIA GPUs.

## Installation

```bash
sudo apt install hipblas hipblas-dev
```

## Portable GEMM Example

```cpp
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

void portable_gemm() {
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    
    const int M = 1024, N = 1024, K = 1024;
    float *d_A, *d_B, *d_C;
    
    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));
    
    float alpha = 1.0f, beta = 0.0f;
    
    // hipBLAS API (works on AMD and NVIDIA)
    hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                 M, N, K, &alpha,
                 d_A, M, d_B, K, &beta, d_C, M);
    
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    hipblasDestroy(handle);
}
```

## Key Advantage: Portability

Write once, run on AMD or NVIDIA:
- On AMD GPUs: hipBLAS → rocBLAS
- On NVIDIA GPUs: hipBLAS → cuBLAS

## References

- [hipBLAS Documentation](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/)
