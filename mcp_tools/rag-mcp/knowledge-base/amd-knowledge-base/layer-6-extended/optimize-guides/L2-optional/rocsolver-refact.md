---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/refact.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# rocSOLVER Refactorization and Direct Solvers

## Overview

These functions enable direct sparse solvers for systems where multiple coefficient matrices share an identical sparsity pattern. The API divides functionality into three categories:

- **Initialization and metadata**: Setup and teardown of refactorization data structures
- **Triangular refactorization**: Fast re-factorization for matrices with known sparsity patterns
- **Direct sparse solvers**: Solution computation based on triangular refactorization

## Initialization and Metadata Functions

### Create and Destroy rfinfo

`rocsolver_create_rfinfo()` initializes a metadata structure required by refactorization and solver routines. The corresponding `rocsolver_destroy_rfinfo()` deallocates this structure.

### Mode Management

`rocsolver_set_rfinfo_mode()` configures whether the structure operates in LU or Cholesky factorization mode. The default is LU for general matrices. Cholesky mode is activated for symmetric positive definite matrices. `rocsolver_get_rfinfo_mode()` retrieves the current mode setting.

### Analysis Phase

`rocsolver_csrrf_analysis()` (available in single and double precision variants) performs symbolic analysis on an initially factorized sparse matrix. It generates metadata describing the sparsity structure, pivot information, and reordering that subsequent refactorizations will leverage.

**Key parameters include**:
- Matrix dimensions and non-zero counts
- Sparse matrix data in CSR format (row pointers, column indices, values)
- Factorization results (L and U factors bundled as T)
- Permutation matrices P and Q
- Right-hand side matrix B (optional for refactorization-only workflows)

## Triangular Refactorization Functions

### SUMLU and SPLITLU

`rocsolver_csrrf_sumlu()` combines separate L and U sparse factors into a single bundled matrix T, discarding the implicit unit diagonal of L. The reverse operation, `rocsolver_csrrf_splitlu()`, extracts L and U from the bundled form.

### Fast LU Refactorization

`rocsolver_csrrf_refactlu()` numerically refactorizes a sparse matrix A that shares the sparsity pattern of a previously analyzed matrix M. It avoids expensive symbolic analysis by reusing the stored permutation and reordering information.

### Fast Cholesky Refactorization

`rocsolver_csrrf_refactchol()` performs analogous refactorization for symmetric positive definite systems using Cholesky decomposition. Only the lower triangular portion of input matrices is referenced; strictly upper triangular entries are ignored.

## Direct Sparse Solver

### CSRRF_SOLVE

`rocsolver_csrrf_solve()` solves the linear system AX = B using previously computed factors. It applies stored permutations, performs forward and backward substitution (or Cholesky-based solution), and outputs the solution matrix X, overwriting the input B.

The solver supports both LU and Cholesky modes, automatically selecting the appropriate algorithm based on rfinfo mode configuration.
