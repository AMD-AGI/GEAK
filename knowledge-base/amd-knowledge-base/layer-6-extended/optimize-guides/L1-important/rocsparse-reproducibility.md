---
tags: ["reference", "rocsparse", "reproducibility", "determinism"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/reproducibility.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Bitwise Reproducibility in rocSPARSE 4.1.0

## Overview

"Some routines do not produce deterministic results from run to run. This is typically the case when HIP atomics are used."

This documentation catalogs which rocSPARSE functions guarantee bitwise reproducibility across multiple executions.

## Sparse Level 1 Functions

All level 1 functions are reproducible, including:
- `rocsparse_Xaxpyi()`
- `rocsparse_Xdoti()`
- `rocsparse_Xdotci()`
- `rocsparse_Xgthr()` and `rocsparse_Xgthrz()`
- `rocsparse_Xroti()`
- `rocsparse_Xsctr()`

## Sparse Level 2 Functions

Most analysis and solve functions are reproducible. However, matrix-vector products show conditional reproducibility:

**Reproducible under specific conditions:**
- `rocsparse_Xbsrmv()`: reproducible without transpose (N/A for transpose)
- `rocsparse_Xbsrxmv()`: reproducible without transpose (N/A for transpose)
- `rocsparse_Xcoomv()`: non-reproducible for both transpose modes
- `rocsparse_Xcsrmv()`: non-reproducible for both transpose modes
- `rocsparse_Xellmv()`: non-reproducible for both transpose modes
- `rocsparse_Xhybmv()`: non-reproducible for both transpose modes
- `rocsparse_Xgebsrmv()`: reproducible without transpose (N/A for transpose)

## Sparse Level 3 Functions

Matrix multiply and solve functions vary:
- `rocsparse_Xcsrsm_*()` functions: reproducible
- `rocsparse_Xbsrsm_*()` functions: reproducible
- `rocsparse_Xgemmi()`: reproducible
- `rocsparse_Xbsrmm()`: reproducible without transpose (N/A for transpose)
- `rocsparse_Xgebsrmm()`: reproducible without transpose (N/A for transpose)
- `rocsparse_Xcsrmm()`: non-reproducible for both transpose modes

## Sparse Extra Functions

Functions like `rocsparse_Xbsrgeam()`, `rocsparse_Xcsrgeam()`, and various matrix multiplication variants are non-reproducible due to atomic operations.

## Preconditioner Functions

Incomplete LU and Cholesky factorizations (`rocsparse_Xcsric0()`, `rocsparse_Xcsrilu0()`, and BSR variants) are non-reproducible. Tridiagonal solvers are reproducible.

## Conversion Functions

All format conversion functions are reproducible, including CSR/CSC/COO/ELL/BSR transformations.

## Reordering Functions

- `rocsparse_Xcsrcolor()`: non-reproducible

## Utility Functions

All matrix validation functions are reproducible.

## Generic Functions

**Non-reproducible:**
- `rocsparse_spvv()`
- `rocsparse_spsv()` and `rocsparse_spsm()`: non-reproducible
- `rocsparse_spgemm()`: non-reproducible

**Reproducible under specific conditions:**
- `rocsparse_spmv()` and `rocsparse_v2_spmv()`: vary by algorithm and transpose mode
- `rocsparse_spmm()`: varies by algorithm and transpose mode

All CSR-based algorithms show non-reproducibility with transpose operations. ELL and BSR formats are reproducible without transpose (N/A applies to transpose).
