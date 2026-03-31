---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/conversion.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Sparse Conversion Functions - rocSPARSE 4.1.0

## Overview

The sparse conversion module contains routines for converting matrices between different sparse storage formats. These operations transform a matrix in one sparse format into another sparse format representation.

## Key Conversion Functions

### CSR ↔ COO Conversions

**rocsparse_csr2coo()** converts CSR row offsets into COO row indices. This function can also convert CSC column offsets to COO column indices.

**rocsparse_coo2csr()** performs the inverse operation, converting COO row indices into CSR row offsets. The input COO row index array must be sorted.

### CSR ↔ CSC Conversions

**rocsparse_csr2csc()** converts between CSR and CSC formats, effectively transposing the matrix. The conversion requires two steps:

1. Call `rocsparse_csr2csc_buffer_size()` to determine temporary buffer size
2. Allocate buffer and call the conversion function

The `copy_values` parameter controls whether values are copied (`rocsparse_action_numeric`) or only the sparsity pattern is determined (`rocsparse_action_symbolic`).

### General Block Sparse Conversions

**rocsparse_gebsr2gebsc()** converts between General BSR and General BSC formats with configurable row and column block dimensions. Like CSR↔CSC conversions, it requires a two-step process with buffer size calculation.

### Format-Specific Conversions

- **CSR ↔ ELL**: `rocsparse_csr2ell()` and `rocsparse_ell2csr()`
- **CSR ↔ HYB**: `rocsparse_csr2hyb()` and `rocsparse_hyb2csr()`
- **BSR ↔ CSR**: `rocsparse_bsr2csr()` and `rocsparse_csr2bsr()`
- **General BSR conversions**: Multiple GEneral BSR format transformations

### Dense Matrix Conversions

Functions for converting between dense and sparse formats:
- `rocsparse_dense2csr()`, `rocsparse_dense2csc()`, `rocsparse_dense2coo()`
- `rocsparse_csr2dense()`, `rocsparse_csc2dense()`, `rocsparse_coo2dense()`

### Specialized Operations

**rocsparse_csr2csr_compress()** removes small values below a threshold, reducing the number of stored elements.

**rocsparse_prune_*()** functions selectively remove matrix elements based on thresholds or percentages.

**rocsparse_bsrpad_value()** pads BSR blocks with specified values.

## Common Parameters

- `handle`: rocSPARSE library context
- `m, n`: Matrix dimensions
- `nnz`: Number of non-zero entries
- `*_row_ptr`, `*_col_ind`: Index arrays for sparse formats
- `*_val`: Value arrays
- `idx_base`: Index base (zero or one-based)
- `temp_buffer`: Temporary storage (when required)

## Notes

All conversion functions are non-blocking and execute asynchronously. They support execution within hipGraph contexts for GPU compute optimization.
