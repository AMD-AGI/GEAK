---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/level1.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Sparse Level 1 Functions

This section documents rocSPARSE level 1 operations, which perform computations between sparse and dense vectors.

## rocsparse_axpyi()

Scales a sparse vector by a scalar and adds it to a dense vector.

**Operation:** `y := y + α·x`

**Variants:**
- `rocsparse_saxpyi()` - single precision
- `rocsparse_daxpyi()` - double precision
- `rocsparse_caxpyi()` - single precision complex
- `rocsparse_zaxpyi()` - double precision complex

**Parameters:**
- `handle` - library context
- `nnz` - number of non-zeros
- `alpha` - scalar multiplier
- `x_val` - sparse values
- `x_ind` - sparse indices
- `y` - dense vector (modified in-place)
- `idx_base` - zero or one-based indexing

## rocsparse_doti()

Computes dot product of sparse vector with dense vector.

**Operation:** `result := y^T·x`

**Variants:** `rocsparse_sdoti()`, `rocsparse_ddoti()`, `rocsparse_cdoti()`, `rocsparse_zdoti()`

**Key Parameters:**
- `x_val`, `x_ind` - sparse vector components
- `y` - dense vector
- `result` - output (can be host or device memory)

## rocsparse_dotci()

Computes dot product using conjugate of complex sparse vector.

**Operation:** `result := conj(x)^H·y`

**Variants:** `rocsparse_cdotci()`, `rocsparse_zdotci()`

## rocsparse_gthr()

Gathers elements from dense vector at specified indices into sparse format.

**Operation:** `x_val[i] := y[x_ind[i]]`

**Variants:** `rocsparse_sgthr()`, `rocsparse_dgthr()`, `rocsparse_cgthr()`, `rocsparse_zgthr()`

## rocsparse_gthrz()

Gathers elements and zeroes them in the source dense vector.

**Operation:**
```
x_val[i] := y[x_ind[i]]
y[x_ind[i]] := 0
```

**Variants:** `rocsparse_sgthrz()`, `rocsparse_dgthrz()`, `rocsparse_cgthrz()`, `rocsparse_zgthrz()`

## rocsparse_roti()

Applies Givens rotation matrix to sparse and dense vectors.

**Rotation matrix:**
```
G = [c   s ]
    [-s  c ]
```

**Variants:** `rocsparse_sroti()`, `rocsparse_droti()`

## rocsparse_sctr()

Scatters sparse vector values into dense vector at specified indices.

**Operation:** `y[x_ind[i]] := x_val[i]`

**Variants:** `rocsparse_ssctr()`, `rocsparse_dsctr()`, `rocsparse_csctr()`, `rocsparse_zsctr()`

---

All functions are asynchronous and support hipGraph execution contexts. Return values indicate success or specific error conditions (invalid handles, invalid parameters, memory allocation failures).
