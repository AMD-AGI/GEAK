---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["reference", "hip", "deprecated", "api"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/reference/deprecated_api_list.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# HIP Deprecated Runtime API Functions

## Overview

The HIP documentation identifies multiple API functions that have been flagged for deprecation. As stated in the source material: "Using the following functions results in errors and unexpected results, so we encourage you to update your code accordingly."

## Deprecation Timeline

### ROCm 6.1.0
Two texture management functions were deprecated:
- `hipTexRefGetBorderColor()`
- `hipTexRefGetArray()`

### ROCm 5.7.0
One texture management function was deprecated:
- `hipBindTextureToMipmappedArray()`

### ROCm 5.3.0
Ten texture management functions were deprecated, including:
- `hipGetTextureReference()`
- `hipTexRefSetAddressMode()`
- `hipTexRefSetArray()`
- `hipTexRefSetFlags()`
- `hipTexRefSetFilterMode()`
- `hipTexRefSetFormat()`
- `hipTexRefSetMipmapFilterMode()`
- `hipTexRefSetMipmapLevelBias()`
- `hipTexRefSetMipmapLevelClamp()`
- `hipTexRefSetMipmappedArray()`

### ROCm 4.3.0
Fourteen texture management functions were deprecated, covering getter and setter operations for texture reference properties.

### ROCm 3.8.0
Five texture management functions and two memory management functions were deprecated:
- `hipBindTexture()`, `hipBindTexture2D()`, `hipBindTextureToArray()`
- `hipGetTextureAlignmentOffset()`, `hipUnbindTexture()`
- `hipMemcpyToArray()`, `hipMemcpyFromArray()`

### ROCm 3.1.0
Memory allocation functions were deprecated in favor of newer alternatives:
- `hipMallocHost()` → replaced by `hipHostAlloc()`
- `hipMemAllocHost()` → replaced by `hipHostAlloc()`

### ROCm 3.0.0
Profiler functions were deprecated: "Instead, you can use roctracer or rocTX for profiling which provide more flexibility and detailed profiling capabilities."
- `hipProfilerStart()`
- `hipProfilerStop()`

### ROCm 1.9.0
Context management functions were deprecated due to better alternate interfaces. The documentation explains that "HIP initially added limited support for context APIs in order to facilitate porting from existing driver codes." Recommended alternatives include `hipSetDevice` or the stream API. This includes 21 context-related functions.
