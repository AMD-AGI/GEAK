---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUSPARSE_API_supported_by_HIP.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# CUSPARSE API Supported by HIP

## Overview

This documentation provides comprehensive mapping tables for NVIDIA CUDA Sparse (cuSPARSE) APIs and their HIP equivalents in AMD's ROCm platform. The tables track API support across multiple categories with status indicators for Added (A), Deprecated (D), Changed (C), Removed (R), and Experimental (E) features.

## Documentation Structure

The reference is organized into 12 major sections:

### 1. **Types References**
Maps fundamental type definitions like `cusparseHandle_t` to `hipsparseHandle_t`, status enumerations, matrix descriptors, and format specifications. Many foundational types have been supported since HIP 1.9.2.

### 2. **Management Functions**
Core operations for creating/destroying handles, managing pointer modes, and version retrieval. Examples include `cusparseCreate`/`hipsparseCreate` and `cusparseGetVersion`/`hipsparseGetVersion`.

### 3. **Logging**
Logging functions introduced in CUDA 11.5, currently without direct HIP equivalents, including callback and file-based logging mechanisms.

### 4. **Helper Functions**
Matrix descriptor manipulation, info object creation/destruction, and attribute getters/setters for configuration management.

### 5. **Level 1 Functions**
Vector-sparse matrix operations including gather, scatter, and dot product variants for single (S), double (D), complex (C), and double-complex (Z) precision types.

### 6. **Level 2 Functions**
Sparse matrix-vector multiplication and triangular solve operations across multiple formats (BSR, CSR, HYB) with analysis and solve phases.

### 7. **Level 3 Functions**
Sparse matrix-matrix multiplication and solving, including block sparse formats and various algorithmic approaches.

### 8. **Extra Functions**
Matrix arithmetic operations like CSR addition and multiplication, along with pruning and format conversion utilities.

### 9. **Preconditioners**
Incomplete LU/Cholesky factorization routines with zero-pivot detection and tridiagonal system solvers.

### 10. **Reorderings**
Graph coloring functionality for sparse matrix reordering operations.

### 11. **Format Conversion**
Conversion utilities between COO, CSR, CSC, BSR, Dense, and HYB formats with accompanying buffer size calculators.

### 12. **Generic API**
Modern unified interface supporting sparse matrix/vector descriptors, dense matrix/vector operations, and high-level routines like SpGEMM, SpMM, and SpSV.

## Key Observations

Most fundamental operations map directly between CUDA and HIP with version alignment. Newer generic APIs (added in CUDA 10.1+) have corresponding HIP support, often with consistent versioning. Some advanced features like logging remain unimplemented in HIP, while specialized formats (BSR, blocked ELL) show varying adoption timelines.
