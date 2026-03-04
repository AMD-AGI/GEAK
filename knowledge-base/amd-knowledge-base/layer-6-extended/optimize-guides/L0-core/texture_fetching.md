---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "hip", "memory", "texture"]
priority: "L0-core"
source_url: "https://rocm.docs.amd.com/en/latest/how-to/hip_runtime_api/memory_management/device_memory/texture_fetching.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Texture Fetching

## Overview

Textures provide access to specialized GPU hardware optimized for graphics processing. They offer "a different way of accessing their underlying device memory" through "a special read-only texture cache, that is optimized for logical spatial locality, e.g. locality in 2D grids." This capability benefits GPGPU algorithms with similar access patterns.

Key advantages of textures include:
- Floating-point indexing capabilities (range 0 to size-1 or 0 to 1)
- Specialized interpolation modes between neighboring values
- Out-of-bounds access handling mechanisms

In HIP, texture objects are type `hipTextureObject_t` and created via `hipCreateTextureObject()`. See the [texture API reference](../../../../reference/hip_runtime_api/modules/memory_management/texture_management.html) for complete function details.

---

## Texture Filtering

Texture filtering interpolates values when indices are fractional. The filter modes are specified in `hipTextureFilterMode`.

### Nearest Point Filtering

**Mode:** `hipFilterModePoint`

**Formula:** `tex(x) = T[floor(x)]`

This method "doesn't interpolate between neighboring values, which results in a pixelated look." It simply returns the nearest texel value without smoothing.

### Linear Filtering

**Mode:** `hipFilterModeLinear`

This performs linear interpolation using the formula `(1-t)P1 + tP2`:

- **1D:** `tex(x) = (1-α)T[i] + αT[i+1]`
- **2D:** `tex(x,y) = (1-α)(1-β)T[i,j] + α(1-β)T[i+1,j] + (1-α)βT[i,j+1] + αβT[i+1,j+1]`
- **3D:** Eight-term interpolation across neighboring voxels

Where α, β, γ represent fractional positions calculated as: `i = floor(x')`, `α = frac(x')`, `x' = x - 0.5`.

---

## Texture Addressing

Texture addressing modes handle out-of-bounds accesses, specified in `hipTextureAddressMode`.

### Address Mode: Border

**Setting:** `hipAddressModeBorder`

Returns a preset border value for out-of-bounds accesses. The border color must be configured before texture fetching.

### Address Mode: Clamp

**Setting:** `hipAddressModeClamp` (default mode)

"Clamps the index between [0 to size-1]." Edge values repeat when accessing beyond bounds.

### Address Mode: Wrap

**Setting:** `hipAddressModeWrap`

*Only for normalized coordinates.* Uses `tex(frac(x))` to create "a repeating image effect" by utilizing only the fractional portion of the index.

### Address Mode: Mirror

**Setting:** `hipAddressModeMirror`

*Only for normalized coordinates.* Creates mirrored repetition:
- `tex(frac(x))` when `floor(x)` is even
- `tex(1 - frac(x))` when `floor(x)` is odd

---

**For implementation examples**, consult the [ROCm texture management example](https://github.com/ROCm/rocm-examples/blob/develop/HIP-Basic/texture_management/main.hip).
