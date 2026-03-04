---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
layer: "3"
category: "solver"
subcategory: "linear-algebra"
tags: ["rocsolver", "linear-algebra", "lapack", "solver"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
---

# rocSOLVER Usage Guide

rocSOLVER provides LAPACK functionality for ROCm, enabling linear system solving and matrix factorizations.

## Installation

```bash
sudo apt install rocsolver rocsolver-dev
```

## Matrix Factorizations

### LU Factorization

```cpp
#include <rocsolver/rocsolver.h>
#include <hip/hip_runtime.h>

void lu_factorization() {
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    const int n = 3;  // 3x3 matrix
    float A[] = {1, 2, 3, 4, 5, 6, 7, 8, 10};  // Column-major
    
    float *d_A;
    int *d_ipiv;
    int *d_info;
    
    hipMalloc(&d_A, n * n * sizeof(float));
    hipMalloc(&d_ipiv, n * sizeof(int));
    hipMalloc(&d_info, sizeof(int));
    
    hipMemcpy(d_A, A, n * n * sizeof(float), hipMemcpyHostToDevice);
    
    // LU factorization
    rocsolver_sgetrf(handle, n, n, d_A, n, d_ipiv, d_info);
    
    // Check result
    int info;
    hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost);
    
    if (info == 0) {
        printf("LU factorization successful\\n");
    }
    
    hipFree(d_A);
    hipFree(d_ipiv);
    hipFree(d_info);
    rocblas_destroy_handle(handle);
}
```

### QR Factorization

```cpp
void qr_factorization() {
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    const int m = 4, n = 3;
    float *d_A, *d_tau;
    
    hipMalloc(&d_A, m * n * sizeof(float));
    hipMalloc(&d_tau, n * sizeof(float));
    
    // QR factorization
    rocsolver_sgeqrf(handle, m, n, d_A, m, d_tau);
    
    hipFree(d_A);
    hipFree(d_tau);
    rocblas_destroy_handle(handle);
}
```

## Solving Linear Systems

```cpp
void solve_system() {
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    
    const int n = 3;
    float *d_A, *d_B;
    int *d_ipiv, *d_info;
    
    hipMalloc(&d_A, n * n * sizeof(float));
    hipMalloc(&d_B, n * sizeof(float));
    hipMalloc(&d_ipiv, n * sizeof(int));
    hipMalloc(&d_info, sizeof(int));
    
    // Solve Ax = b
    // Step 1: LU factorization
    rocsolver_sgetrf(handle, n, n, d_A, n, d_ipiv, d_info);
    
    // Step 2: Solve using factorization
    rocsolver_sgetrs(handle, rocblas_operation_none, n, 1,
                     d_A, n, d_ipiv, d_B, n);
    
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_ipiv);
    hipFree(d_info);
    rocblas_destroy_handle(handle);
}
```

## PyTorch Integration

```python
import torch

# QR decomposition (uses rocSOLVER)
A = torch.randn(4, 3, device='cuda')
Q, R = torch.linalg.qr(A)

# Solve linear system
A = torch.randn(3, 3, device='cuda')
b = torch.randn(3, device='cuda')
x = torch.linalg.solve(A, b)  # Uses rocSOLVER

# Matrix inverse
A_inv = torch.linalg.inv(A)  # Uses rocSOLVER
```

## References

- [rocSOLVER Documentation](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/)
