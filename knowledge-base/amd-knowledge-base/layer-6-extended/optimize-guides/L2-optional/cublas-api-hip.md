---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUBLAS_API_supported_by_HIP.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# CUBLAS API Supported by HIP

## Overview

This documentation presents comprehensive mapping tables between NVIDIA CUDA's CUBLAS library and AMD's HIP implementation. The tables track API equivalences across multiple categories with version information.

## Key Sections

### 1. CUBLAS Data Types
Maps fundamental CUBLAS type definitions to their HIP counterparts, including:
- Atomics modes and computation types
- Operation modes (transpose, conjugate)
- Pointer modes (host/device)
- Fill modes and diagonal types
- GEMM algorithm specifications

### 2. CUDA Library Data Types
Covers data type enumerations for floating-point, integer, and complex number representations across different precision levels.

### 3. CUBLASLt Data Types
Documents advanced matrix multiplication configuration types, including:
- Epilogue operations
- Matrix layout attributes
- Pointer mode masks
- Tile and cluster shape configurations
- Matmul descriptor attributes

### 4. Helper Functions
Initialization, memory management, and stream handling operations for CUBLAS contexts.

### 5-7. BLAS Operations
Organized by computational level:
- **Level-1**: Vector operations (axpy, dot, norm)
- **Level-2**: Matrix-vector operations (gemv, ger, trmv)
- **Level-3**: Matrix-matrix operations (gemm, trsm, symm)

### 8. BLAS-like Extensions
Extended functionality including batched operations, specialized formats, and mixed-precision compute.

### 9. BLASLt Functions
Lightweight API for tuning and heuristic-based algorithm selection in matrix multiplication.

## Documentation Structure

Each table follows consistent formatting with columns indicating:
- **A**: Added version
- **D**: Deprecated version
- **C**: Changed version
- **R**: Removed version
- **E**: Experimental status

This enables developers to track compatibility across ROCm releases.
