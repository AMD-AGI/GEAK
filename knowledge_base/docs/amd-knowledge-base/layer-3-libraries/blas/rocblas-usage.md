---
layer: "3"
category: "rocm-libraries"
subcategory: "blas"
tags: ["rocblas", "blas", "linear-algebra", "performance", "libraries"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
---

# rocBLAS Usage Guide

rocBLAS is AMD's optimized BLAS (Basic Linear Algebra Subprograms) library for ROCm.

## Installation

```bash
# Ubuntu/Debian
sudo apt install rocblas rocblas-dev

# Verify installation
ls /opt/rocm/lib/librocblas.so
```

## Basic Usage (C++)

```cpp
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize rocBLAS
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    // Matrix dimensions
    const int M = 1024, N = 1024, K = 1024;
    
    // Allocate host memory
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice);
    
    // GEMM parameters: C = alpha * A * B + beta * C
    float alpha = 1.0f, beta = 0.0f;
    
    // Call rocBLAS GEMM
    rocblas_sgemm(
        handle,
        rocblas_operation_none,  // No transpose on A
        rocblas_operation_none,  // No transpose on B
        M, N, K,
        &alpha,
        d_A, M,  // lda
        d_B, K,  // ldb
        &beta,
        d_C, M   // ldc
    );
    
    // Copy result back
    hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);
    
    std::cout << "Result sample: " << h_C[0] << std::endl;
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    rocblas_destroy_handle(handle);
    
    return 0;
}
```

Compile:
```bash
hipcc gemm.cpp -lrocblas -o gemm
```

## PyTorch Integration

rocBLAS is automatically used by PyTorch for ROCm:

```python
import torch

# Matrix multiplication uses rocBLAS
A = torch.randn(1024, 1024, device='cuda')
B = torch.randn(1024, 1024, device='cuda')
C = torch.matmul(A, B)  # Uses rocBLAS internally

# Verify rocBLAS is being used
print(torch.version.hip)  # Shows ROCm version
```

## Performance Optimization

### Batched Operations

```cpp
// Batched GEMM for multiple small matrices
const int batch_count = 100;
float *d_A_array[batch_count];
float *d_B_array[batch_count];
float *d_C_array[batch_count];

// Allocate batch
for (int i = 0; i < batch_count; i++) {
    hipMalloc(&d_A_array[i], M * K * sizeof(float));
    hipMalloc(&d_B_array[i], K * N * sizeof(float));
    hipMalloc(&d_C_array[i], M * N * sizeof(float));
}

// Copy array of pointers to device
float **d_A_ptrs, **d_B_ptrs, **d_C_ptrs;
hipMalloc(&d_A_ptrs, batch_count * sizeof(float*));
hipMalloc(&d_B_ptrs, batch_count * sizeof(float*));
hipMalloc(&d_C_ptrs, batch_count * sizeof(float*));
hipMemcpy(d_A_ptrs, d_A_array, batch_count * sizeof(float*), hipMemcpyHostToDevice);
// ... similar for B and C

// Batched GEMM
rocblas_sgemm_batched(
    handle,
    rocblas_operation_none,
    rocblas_operation_none,
    M, N, K,
    &alpha,
    d_A_ptrs, M,
    d_B_ptrs, K,
    &beta,
    d_C_ptrs, M,
    batch_count
);
```

### Strided Batched GEMM

```cpp
// For regularly strided batch of matrices
int stride_A = M * K;
int stride_B = K * N;
int stride_C = M * N;

rocblas_sgemm_strided_batched(
    handle,
    rocblas_operation_none,
    rocblas_operation_none,
    M, N, K,
    &alpha,
    d_A, M, stride_A,
    d_B, K, stride_B,
    &beta,
    d_C, M, stride_C,
    batch_count
);
```

## Mixed Precision

```cpp
// FP16 GEMM (uses matrix cores on CDNA2+)
rocblas_half *d_A_fp16, *d_B_fp16, *d_C_fp16;

rocblas_hgemm(
    handle,
    rocblas_operation_none,
    rocblas_operation_none,
    M, N, K,
    &alpha_fp16,
    d_A_fp16, M,
    d_B_fp16, K,
    &beta_fp16,
    d_C_fp16, M
);
```

## Other rocBLAS Functions

### Vector Operations

```cpp
// SAXPY: y = alpha * x + y
rocblas_saxpy(handle, N, &alpha, d_x, 1, d_y, 1);

// DOT product
float result;
rocblas_sdot(handle, N, d_x, 1, d_y, 1, &result);

// SCAL: x = alpha * x
rocblas_sscal(handle, N, &alpha, d_x, 1);
```

### Matrix-Vector Operations

```cpp
// GEMV: y = alpha * A * x + beta * y
rocblas_sgemv(
    handle,
    rocblas_operation_none,
    M, N,
    &alpha,
    d_A, M,
    d_x, 1,
    &beta,
    d_y, 1
);
```

## Benchmarking

```bash
# rocBLAS provides benchmark tools
/opt/rocm/bin/rocblas-bench -f gemm -m 1024 -n 1024 -k 1024

# With specific precision
/opt/rocm/bin/rocblas-bench -f gemm -m 4096 -n 4096 -k 4096 --precision f16_r
```

## References

- [rocBLAS Documentation](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/)
- [rocBLAS GitHub](https://github.com/ROCmSoftwarePlatform/rocBLAS)

