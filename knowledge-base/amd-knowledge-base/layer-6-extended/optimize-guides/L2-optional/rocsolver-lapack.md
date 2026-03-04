---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# rocSOLVER LAPACK Functions Reference

This documentation covers the LAPACK function interfaces available in rocSOLVER 3.31.0, organized by functional category.

## Overview

rocSOLVER provides GPU-accelerated implementations of LAPACK routines across multiple precision types (single, double, and complex variants). The library supports batched and strided-batched operations for processing multiple problems simultaneously.

## Function Categories

### Triangular Factorizations

The library includes implementations of Cholesky and LU factorization routines:

- **Cholesky factorization** (`potf2`, `potrf`): Decomposes positive-definite matrices into triangular form
- **LU factorization** (`getf2`, `getrf`): Factors general matrices with partial pivoting
- **Symmetric indefinite factorization** (`sytf2`, `sytrf`): Handles symmetric matrices with pivoting

Each routine offers:
- Standard versions operating on single matrices
- Batched variants processing multiple independent matrices
- Strided-batched versions with flexible memory layouts

### Orthogonal Factorizations

QR-related decompositions and related transformations:

- **QR factorization** (`geqr2`, `geqrf`)
- **RQ factorization** (`gerq2`, `gerqf`)
- **QL factorization** (`geql2`, `geqlf`)
- **LQ factorization** (`gelq2`, `gelqf`)

### Matrix Reductions

Transformations for eigenvalue and SVD computations:

- **General bidiagonal reduction** (`gebd2`, `gebrd`)
- **Symmetric tridiagonal reduction** (`sytd2`, `sytrd`)
- **Hermitian tridiagonal reduction** (`hetd2`, `hetrd`)
- **Generalized symmetric/hermitian eigenvalue reduction** (`sygs2`, `sygst`, `hegs2`, `hegst`)

### Linear System Solvers

Direct solution methods for linear equations:

- **Triangular system solve** (`trtri`)
- **General matrix inversion** (`getri`)
- **LU-based solution** (`getrs`, `gesv`)
- **Cholesky-based solution** (`potri`, `potrs`, `posv`)

### Least-Squares Solvers

Overdetermined system solutions:

- **General least squares** (`gels`)

### Eigensolvers

Spectral decomposition routines:

- **Symmetric eigendecomposition** (`syev`)
- **Hermitian eigendecomposition** (`heev`)

## Precision Support

Functions support multiple data types through naming conventions where `<type>` represents:
- `s`: single precision (float)
- `d`: double precision (double)
- `c`: single-precision complex
- `z`: double-precision complex

## Variants

Most routines provide three implementation variants:

1. **Standard**: Single matrix operations
2. **Batched**: Multiple independent matrices with consistent leading dimensions
3. **Strided-batched**: Multiple matrices with flexible stride parameters

64-bit integer variants (suffix `_64`) are available for large-scale problems.
