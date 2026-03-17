---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/level2.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# rocSPARSE Sparse Level 2 Functions Documentation

## Overview

The sparse level 2 functions module encompasses operations between matrices in sparse format and vectors in dense format. This documentation covers rocSPARSE version 4.1.0.

## Key Function Categories

### BSR Matrix-Vector Operations

**rocsparse_bsrmv_analysis()** - Performs preprocessing analysis for block sparse row (BSR) matrix-vector multiplication. This step should execute once per sparsity pattern and can improve subsequent computation performance.

**rocsparse_bsrmv()** - Executes the operation: `y := α·op(A)·x + β·y` where A is an m×n sparse matrix in BSR format. Supports execution with or without prior analysis.

**rocsparse_bsrxmv()** - Implements masked BSR matrix-vector multiplication, allowing selective row updates through a mask array.

### BSR Triangular Solve Operations

**rocsparse_bsrsv_analysis()** - Analyzes sparsity patterns for triangular solve operations. Can share metadata with related functions like bsrsm_analysis and bsrilu0_analysis.

**rocsparse_bsrsv_solve()** - Solves triangular systems: `op(A)·y = α·x` using BSR format, requiring prior analysis step.

**rocsparse_bsrsv_zero_pivot()** - Detects structural or numerical zero pivots during triangular solve operations.

### COO and CSR Operations

**rocsparse_coomv()** - Performs matrix-vector multiplication using coordinate (COO) sparse format.

**rocsparse_csrmv()** and **rocsparse_csrmv_analysis()** - Implements compressed sparse row (CSR) format matrix-vector operations with optional analysis preprocessing.

**rocsparse_csrsv_** family - Triangular solve operations in CSR format with analysis, zero-pivot detection, and buffer management functions.

### Specialized Formats

**rocsparse_ellmv()** - Ellpack format matrix-vector multiplication.

**rocsparse_hybmv()** - Hybrid format (COO/ELL blend) matrix-vector multiplication.

**rocsparse_gebsrmv()** - General block sparse row format operations.

**rocsparse_gemvi()** - Sparse matrix dense vector operations with buffer size management.

## Iterative Triangular Solve

**rocsparse_csritsv_** family - Iterative triangular solve using CSR format with multiple solution variants and extended functionality through solve_ex variants.

## Common Workflow Pattern

1. Create matrix descriptor and info structure
2. Call analysis function (optional but recommended for multiple operations)
3. Execute solve/multiply operation
4. Call zero_pivot to check for numerical issues
5. Clear resources when complete

## Data Type Support

Functions support single/double precision floating point and complex number types (float, double, rocsparse_float_complex, rocsparse_double_complex).
