Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
layer: "best-practices"
category: "performance"
subcategory: "kernels"
tags: ["kernels", "optimization", "performance", "wavefronts"]
rocm_version: "7.0+"
last_updated: 2025-11-01
---

# Kernel Optimization for AMD GPUs

## Occupancy Optimization

AMD GPUs: Wave size = 64 threads (vs NVIDIA's 32)

```cpp
// Calculate occupancy
__global__ void kernel() {
    // Resources per thread:
    // - Registers: 24 VGPRs
    // - Shared memory: 1KB per block
}

// Theoretical occupancy:
// MI250X: 40 waves per CU maximum
// Block size 256 = 4 waves per block
// If 4 blocks fit per CU = 16 waves = 40% occupancy

// Optimize: Use block size 128 or 256
// Minimize register and shared memory usage
```

## Vectorized Memory Access

```cpp
// BAD: Scalar loads
__global__ void scalar_load(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float a = data[idx];
    float b = data[idx + 1];
    float c = data[idx + 2];
    float d = data[idx + 3];
}

// GOOD: Vectorized loads
__global__ void vector_load(float4* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float4 val = data[idx];  // Single 128-bit load
    // val.x, val.y, val.z, val.w
}
```

## Loop Unrolling

```cpp
// Let compiler unroll for performance
#pragma unroll
for (int i = 0; i < 8; i++) {
    sum += data[i];
}

// Manual unrolling for critical loops
sum += data[0] + data[1] + data[2] + data[3];
sum += data[4] + data[5] + data[6] + data[7];
```

## Reduce Thread Divergence

```cpp
// BAD: Lots of divergence
if (threadIdx.x < 10) {
    // Only 10 threads active
    expensive_computation();
}

// GOOD: Minimize divergence
if (warpId < N) {  // All threads in wave take same path
    computation();
}
```

## Use Intrinsics

```cpp
// Fast math intrinsics
__global__ void fast_math(float* data) {
    float x = data[threadIdx.x];
    
    // Fast reciprocal square root
    float rsqrt = __frsqrt_rn(x);
    
    // Fast sin/cos
    float s, c;
    __sincosf(x, &s, &c);
}
```

## Optimization Checklist

- [ ] Block size multiple of 64 (wave size)
- [ ] Minimize register usage (target < 64 VGPRs)
- [ ] Coalesced memory access
- [ ] Use shared memory for reused data
- [ ] Minimize thread divergence
- [ ] Use vectorized loads (float2, float4)
- [ ] Loop unrolling where beneficial
- [ ] Reduce global memory transactions
- [ ] Profile with rocprof

## References

- [AMD Kernel Optimization Guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides.html)
