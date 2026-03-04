---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "rocfft", "runtime-compilation", "caching"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocFFT/en/latest/how-to/runtime-compilation.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# Runtime Compilation

rocFFT includes many kernels for common FFT problems. Many plans require additional kernels aside from the ones built into the library. In these cases, rocFFT compiles optimized kernels for the plan when the plan is created.

## Kernel Storage and Reuse

Compiled kernels are stored in memory by default. They will be reused if they are required again for plans in the same process.

## Persistent Cache Configuration

If the `ROCFFT_RTC_CACHE_PATH` environment variable is set to a writable file location, rocFFT writes the compiled kernels to this location. rocFFT reads the kernels from this location for plans in other processes that need runtime-compiled kernels. rocFFT will create the specified file if it does not already exist.
