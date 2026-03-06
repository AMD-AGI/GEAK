Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.


# ROCm Libraries Reference

## Installation

Libraries listed below are pre-installed in the current environment under `/opt/rocm-*/` or `/opt/rocm`.
Use `ls /opt/rocm-*/include/` to discover installed headers and the exact ROCm version.

## Source Code

All library source code is in one monorepo. To obtain it:

```bash
# Clone to ~/.cache (shallow clone, latest commit only)
git clone --depth 1 https://github.com/ROCm/rocm-libraries.git ~/.cache/rocm-libraries
```

After cloning, each library's source code is at `~/.cache/rocm-libraries/projects/<library>/`.

## Core — Kernel Optimization

| Library | Local Path | Description |
|---------|-----------|-------------|
| **composablekernel** | `projects/composablekernel/` | Composable, high-performance GPU kernel template library. Provides building blocks (GEMM, convolution, reduction) for writing fused and tiled kernels. |
| **rocwmma** | `projects/rocwmma/` | Wave Matrix Multiply-Accumulate (WMMA) intrinsics wrapper. Enables efficient matrix-multiply operations using AMD matrix cores (MFMA instructions). |
| **rocprim** | `projects/rocprim/` | Parallel primitives (scan, reduce, sort, radix sort, etc.). Foundational building blocks for high-performance parallel algorithms on AMD GPUs. |
| **hipcub** | `projects/hipcub/` | Block-level and warp-level parallel primitives (HIP port of CUB). Provides fine-grained control over thread-block cooperative operations. |

## High Relevance — Math & BLAS

| Library | Local Path | Description |
|---------|-----------|-------------|
| **rocblas** | `projects/rocblas/` | BLAS (Basic Linear Algebra Subprograms) on AMD GPUs. Reference implementation for optimized GEMM, GEMV, and other matrix/vector operations. |
| **hipblaslt** | `projects/hipblaslt/` | Lightweight BLAS library optimized for GEMM. Supports mixed-precision matrix multiplication with flexible epilogue fusion. |

## Medium Relevance

| Library | Local Path | Description |
|---------|-----------|-------------|
| **rocthrust** | `projects/rocthrust/` | High-level parallel algorithms (ROCm port of Thrust). Provides sort, transform, reduce, scan with STL-like interface. |
| **hiptensor** | `projects/hiptensor/` | Tensor contraction and reduction operations on AMD GPUs. |
| **miopen** | `projects/miopen/` | Deep learning primitives (convolution, pooling, batch normalization, activation). Reference for DL kernel implementations. |

## Other Libraries

| Library | Local Path | Description |
|---------|-----------|-------------|
| hipblas | `projects/hipblas/` | HIP interface to BLAS |
| hipblas-common | `projects/hipblas-common/` | Common utilities shared by hipBLAS libraries |
| hipfft | `projects/hipfft/` | FFT (Fast Fourier Transform) |
| rocfft | `projects/rocfft/` | FFT on AMD GPUs |
| hiprand | `projects/hiprand/` | Random number generation (HIP interface) |
| rocrand | `projects/rocrand/` | Random number generation on AMD GPUs |
| hipsolver | `projects/hipsolver/` | LAPACK-like solvers (HIP interface) |
| rocsolver | `projects/rocsolver/` | LAPACK-like dense linear algebra solvers |
| hipsparse | `projects/hipsparse/` | Sparse matrix operations (HIP interface) |
| rocsparse | `projects/rocsparse/` | Sparse matrix operations on AMD GPUs |
| hipsparselt | `projects/hipsparselt/` | Structured sparsity matrix operations |
| hipdnn | `projects/hipdnn/` | DNN operations (HIP interface) |
