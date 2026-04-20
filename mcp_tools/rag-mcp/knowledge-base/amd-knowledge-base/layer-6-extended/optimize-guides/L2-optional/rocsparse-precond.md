---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/precond.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Sparse Preconditioner Functions

## Overview

This rocSPARSE module provides sparse preconditioner functions for manipulating matrices in sparse format to obtain preconditioner matrices.

## Main Function Groups

### BSRIC0 - Block Sparse Row Incomplete Cholesky

**Purpose:** Computes incomplete Cholesky factorization (LL^T) with zero fill-ins for BSR format matrices.

**Key Functions:**
- `rocsparse_Xbsric0_buffer_size()` - Determines temporary storage requirements
- `rocsparse_Xbsric0_analysis()` - Analyzes sparsity pattern
- `rocsparse_Xbsric0()` - Performs the factorization
- `rocsparse_bsric0_zero_pivot()` - Detects numerical/structural zeros
- `rocsparse_bsric0_clear()` - Deallocates analysis metadata

**Data Types:** Single (s), double (d), complex float (c), complex double (z)

### BSRILU0 - Block Sparse Row Incomplete LU

**Purpose:** Computes incomplete LU factorization with zero fill-ins for BSR format matrices.

**Key Functions:**
- `rocsparse_Xbsrilu0_buffer_size()`
- `rocsparse_Xbsrilu0_analysis()`
- `rocsparse_Xbsrilu0()`
- `rocsparse_bsrilu0_zero_pivot()`
- `rocsparse_bsrilu0_numeric_boost()` - Enables numerical stabilization
- `rocsparse_bsrilu0_clear()`

### CSRIC0 - Compressed Sparse Row Incomplete Cholesky

Similar to BSRIC0 but operates on CSR format matrices with functions for tolerance management.

### CSRILU0 - Compressed Sparse Row Incomplete LU

Similar to BSRILU0 but for CSR format, including tolerance and boost options.

### CSRITILU0 - Iterative Refinement ILU

Advanced iterative ILU preconditioner with:
- `rocsparse_csritilu0_buffer_size()`
- `rocsparse_csritilu0_preprocess()`
- `rocsparse_csritilu0_compute()`
- History tracking for convergence monitoring

### Tridiagonal Solvers (GTSV/GPSV)

**GTSV Functions:** Handle general tridiagonal systems
- Buffer size queries
- Standard solve and no-pivot variants
- Strided batch processing
- Interleaved batch operations

**GPSV Functions:** Handle pentadiagonal systems with interleaved batch support

## Usage Pattern

For all preconditioner functions, the typical workflow is:

1. Call `*_buffer_size()` to determine storage needs
2. Allocate temporary buffer
3. Execute `*_analysis()` for sparsity pattern analysis
4. Call the main function (e.g., `rocsparse_Xbsric0()`) repeatedly if needed
5. Optionally call `*_clear()` to free analysis data

## Key Features

- **Analysis Reuse:** Metadata can be reused across operations via `rocsparse_analysis_policy_reuse`
- **Zero Pivot Detection:** Functions detect and report singular/structural zeros
- **Hipgraph Support:** Most computation functions support hipGraph contexts
- **Numeric Boosting:** Available for ILU methods to enhance stability
- **Batch Operations:** Support for strided and interleaved batch processing
