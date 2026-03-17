---
tags: ["optimization", "performance", "rocsparse", "design", "sparse-linear-algebra"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/conceptual/rocsparse-design.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# rocSPARSE Design Notes

## Overview

rocSPARSE is developed using the "Hourglass API" approach, providing a thin C89 API while maintaining C++ convenience. This design avoids binary compatibility issues and enables usage across programming languages. The library uses opaque types to conceal implementation details from users.

## Temporary Device Memory

Many rocSPARSE routines require temporary device storage buffers. Users manage allocation and deallocation themselves. These buffers can be reused across multiple API calls without regular deallocation. Routines offer dedicated functions to query required buffer sizes, such as `rocsparse_scsrsv_buffer_size()`.

## Library Source Code Organization

### library/include Directory

Contains all user-facing API declarations:

| File | Purpose |
|------|---------|
| `rocsparse.h` | Main header including all API files |
| `rocsparse-auxiliary.h` | Handle and descriptor management |
| `rocsparse-complex-types.h` | Complex number type definitions |
| `rocsparse-functions.h` | Sparse linear algebra subroutine declarations |
| `rocsparse-types.h` | Data type definitions |
| `rocsparse-version.h.in` | Version and configuration settings |

The `library/include/internal` directory organizes subroutines into categories: level1, level2, level3, extra, precond, conversion, reordering, generic, and utility.

### library/src Directory

Houses implementation files organized by subroutine class:

| File | Description |
|------|-------------|
| `handle.cpp` | Opaque handle structure implementation |
| `rocsparse_auxiliary.cpp` | Auxiliary function implementations |
| `status.cpp` | HIP error to rocSPARSE status conversion |
| `include/common.h` | Shared device functions |
| `include/definitions.h` | Status-flag macro definitions |

### Sparse Linear Algebra Subroutines

Each subroutine comprises three files:

- **rocsparse_<subroutine>.cpp**: C wrapper and API functionality for all precisions
- **rocsparse_<subroutine>.hpp**: Template-based API implementation
- **<subroutine>_device.h**: Device code for computation

All subroutines must return `rocsparse_status` and utilize the stream accessible through the library handle.

## Important Functions and Data Structures

### Commonly Shared Device Code

Essential device functions include:

- `rocsparse::clz()` - Compute leftmost significant bit position
- `rocsparse::one()` - Return pointer to 1 for specified precision
- `rocsparse::ldg()` - Load via cache wrapper
- `rocsparse::nontemporal_load()/store()` - Non-temporal memory operations
- `rocsparse::blockreduce_sum/max/min()` - Block-wide reductions
- `rocsparse::wfreduce_sum/max/min()` - DPP-based wavefront reductions

### Status-Flag Macros

| Macro | Function |
|-------|----------|
| `RETURN_IF_HIP_ERROR()` | Return on HIP error |
| `THROW_IF_HIP_ERROR()` | Throw exception on HIP error |
| `PRINT_IF_HIP_ERROR()` | Print message on HIP error |
| `RETURN_IF_ROCSPARSE_ERROR()` | Return on rocSPARSE error |

### rocsparse_mat_info Structure

Contains matrix metadata collected during analysis routines:

- `rocsparse_csrmv_info` - CSR matrix-vector multiplication metadata
- `rocsparse_csrtr_info` - Triangular matrix operation metadata
- `rocsparse_csrgemm_info` - CSR matrix-matrix multiplication metadata

#### Cross-Routine Data Sharing

Analysis data like dependency graphs can be shared between routines. For example, incomplete LU factorization analysis can be reused for subsequent triangular solves on the same matrix, controlled by the `rocsparse_analysis_policy` parameter.

## Clients

### Samples

Available examples demonstrating rocSPARSE functionality:

- `example_coomv` - COO format matrix-vector multiplication
- `example_csrmv` - CSR format matrix-vector multiplication
- `example_ellmv` - ELL format matrix-vector multiplication
- `example_handle` - Handle initialization and finalization
- `example_hybmv` - HYB format matrix-vector multiplication

### Unit Tests

GoogleTest-based tests covering all exposed routines with comprehensive floating-point precision testing and parameter validation.

### Benchmarks

The `rocsparse-bench` tool provides performance measurement across all API routines with configurable parameters including matrix dimensions, sparsity patterns, precision levels, and algorithms. Supports multiple matrix formats (MatrixMarket, rocALUTION) and output to JSON for analysis.

### Python Plotting Scripts

Two utility scripts generate performance visualizations:

- `rocsparse-bench-plot.py` - Creates performance plots (GB/s, GFLOPS/s, milliseconds)
- `rocsparse-bench-compare.py` - Compares multiple benchmark runs with ratio analysis

Both support linear and logarithmic y-axis scaling.

### Matrix Download Scripts

Helper scripts in `rocSPARSE/scripts/performance/matrices` download and convert matrices from the sparse suite collection to `.csr` format for benchmarking purposes.
