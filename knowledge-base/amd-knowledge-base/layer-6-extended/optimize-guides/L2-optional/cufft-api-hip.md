---
tags: ["optimization", "performance", "hip", "kernel"]
priority: "L2-optional"
source_url: "https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUFFT_API_supported_by_HIP.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# CUFFT API supported by HIP

## Overview

This documentation presents mapping tables between NVIDIA's CUDA Fast Fourier Transform (cuFFT) API and AMD's HIP equivalent functions, enabling code portability across platforms.

## Data Types

The first section catalogs approximately 70 CUFFT data types and their HIP counterparts. Notable mappings include:

- `cufftComplex` → `hipfftComplex`
- `cufftHandle` → `hipfftHandle`
- `cufftResult` → `hipfftResult`

Most data type mappings were introduced in HIP version 1.7.0, with newer callback-related types added in version 4.3.0 and distributed format support in version 11.8.

## API Functions

The second section covers approximately 60 CUFFT functions with HIP equivalents, organized by functionality:

**Plan Creation & Management:**
- `cufftCreate`, `cufftDestroy`, `cufftMakePlan1d/2d/3d`

**Execution Functions:**
- `cufftExecC2C`, `cufftExecR2C`, `cufftExecZ2Z` (plus variants)

**Size & Property Queries:**
- `cufftGetSize`, `cufftEstimate1d`, `cufftGetProperty`

**Extended (Xt) Operations:**
- `cufftXtMakePlanMany`, `cufftXtExecDescriptor`, callback management

Most core functions map to HIP 1.7.0, while newer extended features support added progressively through version 11.8.
