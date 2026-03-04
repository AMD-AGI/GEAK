---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "performance", "rocfft", "fft", "computation"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/rocFFT/en/latest/conceptual/fft-computation.html"
rocm_version: "7.0+"
last_updated: 2026-01-07
---

# FFT Computation

## Overview

rocFFT implements the Discrete Fourier Transform (DFT) by leveraging mathematical symmetries to optimize computational complexity from O(N²) down to O(N log N).

## How the Library Computes DFTs

### 1D Complex DFT

The transformation follows this formula:

**x̃ⱼ = Σ(k=0 to n-1) xₖ exp(±i·2πjk/n)** for j = 0,1,...,n-1

Where:
- xₖ represents the input complex data
- x̃ⱼ denotes the transformed output
- The ± sign indicates transform direction (negative for forward, positive for backward)

### 2D Complex DFT

**x̃ⱼₖ = Σ(q=0 to m-1) Σ(r=0 to n-1) xᵣₑ exp(±i·2πjr/n) exp(±i·2πkq/m)**

For j = 0,1,...,n-1 and k = 0,1,...,m-1

The variables follow similar conventions as the 1D case, with separate exponential factors for each dimension.

### 3D Complex DFT

**x̃ⱼₖₗ = Σ(s=0 to p-1) Σ(q=0 to m-1) Σ(r=0 to n-1) xᵣₑₛ exp(±i·2πjr/n) exp(±i·2πkq/m) exp(±i·2πls/p)**

For j = 0,1,...,n-1, k = 0,1,...,m-1, and l = 0,1,...,p-1

This extends the 2D pattern with an additional exponential term for the third dimension.
