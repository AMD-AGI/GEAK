---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/level3.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Sparse Level 3 Functions

This documentation covers sparse level 3 routines in rocSPARSE 4.1.0, which describe operations between a sparse matrix and multiple vectors in dense format (treated as a dense matrix).

## Core Functions

### rocsparse_bsrmm()

Performs sparse matrix-dense matrix multiplication using BSR (Block Sparse Row) storage format:

**Operation:** `C := α·op(A)·op(B) + β·C`

Where:
- A is an m × k sparse matrix in BSR format
- B is a k × n dense column-oriented matrix
- C is an m × n dense result matrix
- m = block_dim × mb, k = block_dim × kb

**Variants:** rocsparse_sbsrmm, rocsparse_dbsrmm, rocsparse_cbsrmm, rocsparse_zbsrmm

**Limitations:**
- Currently only trans_A == rocsparse_operation_none is supported
- Matrix type must be rocsparse_matrix_type_general

### rocsparse_gebsrmm()

General BSR sparse matrix-dense matrix multiplication supporting rectangular blocks:

**Operation:** `C := α·op(A)·op(B) + β·C`

Where m = row_block_dim × mb and k = col_block_dim × kb

**Variants:** rocsparse_sgebsrmm, rocsparse_dgebsrmm, rocsparse_cgebsrmm, rocsparse_zgebsrmm

### rocsparse_csrmm()

Sparse matrix-dense matrix multiplication using CSR (Compressed Sparse Row) format:

**Operation:** `C := α·op(A)·op(B) + β·C`

Supports full transposition options for both matrices.

**Variants:** rocsparse_scsrmm, rocsparse_dcsrmm, rocsparse_ccsrmm, rocsparse_zcsrmm

**Note:** Results are non-deterministic when A is transposed.

## Sparse Triangular Solve Functions

### rocsparse_csrsm_*() Family

Solves triangular systems using CSR format matrices. Workflow consists of three steps:

1. **rocsparse_csrsm_buffer_size()** - Determines temporary storage requirements
2. **rocsparse_csrsm_analysis()** - Analyzes matrix structure (blocking operation)
3. **rocsparse_csrsm_solve()** - Performs the solve operation

**Utility:** rocsparse_csrsm_zero_pivot() detects singular matrices during solve.

**Variants:** Single/double precision, complex number support

### rocsparse_bsrsm_*() Family

Block sparse triangular solve functions with analogous three-step workflow.

### rocsparse_gemmi()

General matrix-sparse matrix multiplication (opposite operand order from bsrmm/csrmm):

**Operation:** `C := α·op(A)·op(B) + β·C`

Where A is dense and B is sparse.

## Key Parameters

- **handle:** rocSPARSE context queue
- **trans_A/trans_B:** Operation type (none, transpose, conjugate transpose)
- **alpha/beta:** Scalar coefficients
- **descr:** Matrix descriptor specifying properties
- **ldb/ldc:** Leading dimensions of dense matrices
- **policy:** Analysis/solve execution policy (reuse or force)

## Important Notes

- Functions are "non-blocking" and execute asynchronously relative to host code
- Most routines support hipGraph context execution
- Analysis phase can share metadata with related operations (ILU0, SV)
- Zero pivot detection requires blocking operation separate from solve
