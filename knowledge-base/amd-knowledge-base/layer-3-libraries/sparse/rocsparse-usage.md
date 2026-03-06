Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "3"
category: "sparse"
subcategory: "sparse-linear-algebra"
tags: ["rocsparse", "sparse-matrix", "iterative-solvers"]
rocm_version: "7.0+"
therock_included: true
last_updated: 2025-11-01
---

# rocSPARSE Usage Guide

rocSPARSE provides sparse linear algebra operations for ROCm.

## Installation

```bash
sudo apt install rocsparse rocsparse-dev
```

## Sparse Matrix-Vector Multiply (SpMV)

```cpp
#include <rocsparse/rocsparse.h>
#include <hip/hip_runtime.h>

void spmv_example() {
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);
    
    // CSR format: Compressed Sparse Row
    const int m = 4, n = 4, nnz = 6;
    
    // Matrix: [1 0 0 2]
    //         [0 3 0 0]
    //         [4 0 5 0]
    //         [0 0 0 6]
    
    int h_row_ptr[] = {0, 2, 3, 5, 6};
    int h_col_ind[] = {0, 3, 1, 0, 2, 3};
    float h_val[] = {1, 2, 3, 4, 5, 6};
    float h_x[] = {1, 2, 3, 4};
    float h_y[] = {0, 0, 0, 0};
    
    // Allocate device memory
    int *d_row_ptr, *d_col_ind;
    float *d_val, *d_x, *d_y;
    
    hipMalloc(&d_row_ptr, (m + 1) * sizeof(int));
    hipMalloc(&d_col_ind, nnz * sizeof(int));
    hipMalloc(&d_val, nnz * sizeof(float));
    hipMalloc(&d_x, n * sizeof(float));
    hipMalloc(&d_y, m * sizeof(float));
    
    // Copy to device
    hipMemcpy(d_row_ptr, h_row_ptr, (m + 1) * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_col_ind, h_col_ind, nnz * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_val, h_val, nnz * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_x, h_x, n * sizeof(float), hipMemcpyHostToDevice);
    
    // Create matrix descriptor
    rocsparse_mat_descr descr;
    rocsparse_create_mat_descr(&descr);
    
    // Perform SpMV: y = alpha * A * x + beta * y
    float alpha = 1.0f, beta = 0.0f;
    rocsparse_scsrmv(handle, rocsparse_operation_none,
                     m, n, nnz, &alpha, descr,
                     d_val, d_row_ptr, d_col_ind,
                     d_x, &beta, d_y);
    
    // Copy result back
    hipMemcpy(h_y, d_y, m * sizeof(float), hipMemcpyDeviceToHost);
    
    // Cleanup
    hipFree(d_row_ptr);
    hipFree(d_col_ind);
    hipFree(d_val);
    hipFree(d_x);
    hipFree(d_y);
    rocsparse_destroy_mat_descr(descr);
    rocsparse_destroy_handle(handle);
}
```

## PyTorch Sparse Operations

```python
import torch

# Create sparse tensor
indices = torch.tensor([[0, 1, 2], [0, 1, 2]], device='cuda')
values = torch.tensor([1.0, 2.0, 3.0], device='cuda')
sparse_tensor = torch.sparse_coo_tensor(indices, values, (3, 3))

# Sparse matrix multiplication (uses rocSPARSE)
dense = torch.randn(3, 4, device='cuda')
result = torch.sparse.mm(sparse_tensor, dense)
```

## References

- [rocSPARSE Documentation](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/)
