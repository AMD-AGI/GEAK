Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Device_API_supported_by_HIP.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# CUDA DEVICE API Supported by HIP

## Overview

This documentation table provides a comprehensive mapping of CUDA Device API functions and types to their HIP equivalents. The reference uses notation to indicate API status: **A** (Added), **D** (Deprecated), **C** (Changed), **R** (Removed), and **E** (Experimental).

## Device Functions

The Device Functions section contains extensive mappings of low-level intrinsic operations, including:

### Synchronization and Warp Operations
Functions like `__syncthreads`, `__threadfence`, and warp shuffle operations (`__shfl`, `__shfl_sync`, etc.) have direct HIP equivalents, with most available since HIP 1.6.0.

### Arithmetic and Type Conversions
Comprehensive support for floating-point operations across multiple precision levels:
- Half-precision (`__half`) conversions and operations
- BFloat16 (`__bfloat16`) operations (added in CUDA 11.0, HIP 5.7.0)
- Standard float and double manipulations
- Integer arithmetic with various rounding modes

### Memory Operations
Generic load operations like `__ldg` provide cached global memory access, with HIP support since version 1.6.0.

### Atomic Operations
All standard atomic operations are supported with both block and system-level variants (`atomicAdd`, `atomicCAS`, `atomicExch`, etc.).

### Mathematical Functions
Extensive support for trigonometric, exponential, logarithmic, and special functions (Bessel functions, error functions).

## Device Types

Device types include:

- **Half-precision types**: `__half`, `__half2`, `__half_raw`, `__half2_raw`
- **BFloat16 types**: `__hip_bfloat16`, `__hip_bfloat162` (and raw variants)
- **Low-precision formats**: FP4, FP6, FP8 variants with different exponent/mantissa configurations
- **Rounding modes**: `hipRoundNearest`, `hipRoundZero`, `hipRoundMinInf`, `hipRoundPosInf`
- **Constants**: Predefined values like `HIPRT_INF_FP16`, `HIPRT_NAN_FP16`

Most core functionality has been supported since HIP 1.6.0, with newer features progressively added in later releases.
