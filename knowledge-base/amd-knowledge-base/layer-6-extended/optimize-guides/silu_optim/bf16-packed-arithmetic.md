---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

---
tags: ["optimization", "bf16", "hip", "silu", "arithmetic"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html"
rocm_version: "6.0+"
last_updated: 2026-01-15
---

# BF16 Packed Arithmetic Operations

## Overview

Packed arithmetic operations allow processing two bfloat16 values simultaneously using the `__hip_bfloat162` data type, effectively doubling computational throughput for element-wise operations. For SiLU activation (`x * sigmoid(x)`), packed arithmetic is crucial because it enables computing two SiLU values in parallel within a single instruction, maximizing both memory bandwidth utilization and computational efficiency on AMD GPUs.

The `__hip_bfloat162` type, available in ROCm 6.0+ via `hip_bfloat16.h`, provides hardware-accelerated packed operations including addition (`__hadd2`), multiplication (`__hmul2`), fused multiply-add (`__hfma2`), and various mathematical functions. These operations are particularly valuable for SiLU because they eliminate the overhead of unpacking, computing, and repacking bf16 values, while maintaining the same accuracy as separate scalar operations.

## Technical Details

HIP's packed bf16 arithmetic operations are implemented as device-side intrinsics that map directly to hardware instructions on AMD GPUs. The key packed operations include:

1. **Basic Arithmetic**: `__hadd2`, `__hsub2`, `__hmul2`, `__hdiv2` perform element-wise operations on both halves of `__hip_bfloat162` simultaneously.

2. **Fused Operations**: `__hfma2(a, b, c)` computes `a*b + c` for both pairs in a single instruction, which is essential for efficient SiLU implementation where you need `x * sigmoid(x)`.

3. **Comparison Operations**: `__hbeq2`, `__hbge2`, `__hbgt2`, `__hble2`, `__hblt2`, `__hbne2` return packed boolean results for element-wise comparisons.

4. **Math Functions**: `h2exp`, `h2log`, `h2sqrt`, `h2floor`, `h2ceil` operate on both bf16 values simultaneously.

Performance characteristics:
- **Throughput**: Packed operations process 2 bf16 values per instruction, effectively doubling arithmetic throughput
- **Latency**: Same instruction latency as scalar bf16 operations (typically 4-8 cycles)
- **Register Efficiency**: Storing two values in one register reduces register pressure by 2x compared to separate storage
- **Pipeline Utilization**: Better instruction-level parallelism as fewer instructions are needed for the same work

For SiLU, the critical path is computing sigmoid activation followed by multiplication. Packed operations allow computing two complete SiLU activations with approximately half the instruction count, significantly improving performance on memory-bound workloads.

## Code Examples

### Example 1: Basic Packed Arithmetic for SiLU

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Helper function to create __hip_bfloat162 from two bf16 values
__device__ __forceinline__ __hip_bfloat162 make_bfloat162(
    __hip_bfloat16 low,
    __hip_bfloat16 high
) {
    __hip_bfloat162 result;
    result.x = low;
    result.y = high;
    return result;
}

// Packed sigmoid approximation using packed operations
__device__ __forceinline__ __hip_bfloat162 sigmoid_packed(
    __hip_bfloat162 x
) {
    // Constants for sigmoid approximation
    __hip_bfloat162 one = make_bfloat162(
        __float2bfloat16(1.0f),
        __float2bfloat16(1.0f)
    );
    __hip_bfloat162 half = make_bfloat162(
        __float2bfloat16(0.5f),
        __float2bfloat16(0.5f)
    );

    // Simple approximation: 0.5 + 0.5 * tanh(x)
    // For better accuracy, use: 1 / (1 + exp(-x))
    __hip_bfloat162 neg_x = __hneg2(x);  // Negate both elements
    // Note: h2exp not always available, may need scalar fallback
    __hip_bfloat162 exp_neg_x;
    exp_neg_x.x = hexp(__hneg(x.x));
    exp_neg_x.y = hexp(__hneg(x.y));

    __hip_bfloat162 denom = __hadd2(one, exp_neg_x);  // 1 + exp(-x)
    __hip_bfloat162 sigmoid_val = __hdiv2(one, denom);  // 1 / (1 + exp(-x))

    return sigmoid_val;
}

// SiLU using packed arithmetic
__global__ void silu_packed_kernel(
    const __hip_bfloat162* __restrict__ input,
    __hip_bfloat162* __restrict__ output,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pairs) {
        // Load 2 bf16 values as packed type
        __hip_bfloat162 x = input[idx];

        // Compute sigmoid for both values
        __hip_bfloat162 sigmoid_x = sigmoid_packed(x);

        // Multiply: silu(x) = x * sigmoid(x) for both values
        __hip_bfloat162 result = __hmul2(x, sigmoid_x);

        // Store result
        output[idx] = result;
    }
}
```

### Example 2: Optimized SiLU with Fused Multiply-Add

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Fast sigmoid using polynomial approximation with FMA
__device__ __forceinline__ __hip_bfloat162 sigmoid_fast_packed(
    __hip_bfloat162 x
) {
    // Clamp input to [-5, 5] for numerical stability
    __hip_bfloat162 neg_five = make_bfloat162(
        __float2bfloat16(-5.0f),
        __float2bfloat16(-5.0f)
    );
    __hip_bfloat162 five = make_bfloat162(
        __float2bfloat16(5.0f),
        __float2bfloat16(5.0f)
    );

    // Use packed comparison and selection
    x = __hmin2(x, five);
    x = __hmax2(x, neg_five);

    // Polynomial approximation: a0 + a1*x + a2*x^2 using FMA
    __hip_bfloat162 a0 = make_bfloat162(
        __float2bfloat16(0.5f),
        __float2bfloat16(0.5f)
    );
    __hip_bfloat162 a1 = make_bfloat162(
        __float2bfloat16(0.25f),
        __float2bfloat16(0.25f)
    );
    __hip_bfloat162 a2 = make_bfloat162(
        __float2bfloat16(-0.02f),
        __float2bfloat16(-0.02f)
    );

    // FMA chain: result = a0 + a1*x + a2*x*x
    __hip_bfloat162 x_sq = __hmul2(x, x);
    __hip_bfloat162 result = __hfma2(a2, x_sq, a0);  // a0 + a2*x^2
    result = __hfma2(a1, x, result);  // + a1*x

    return result;
}

// Optimized SiLU with minimal instruction count
__global__ void silu_fma_kernel(
    const __hip_bfloat162* __restrict__ input,
    __hip_bfloat162* __restrict__ output,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop for better occupancy
    for (int i = idx; i < num_pairs; i += stride) {
        __hip_bfloat162 x = input[i];

        // Compute sigmoid with fast approximation
        __hip_bfloat162 sigmoid_x = sigmoid_fast_packed(x);

        // Final SiLU: x * sigmoid(x)
        __hip_bfloat162 silu_x = __hmul2(x, sigmoid_x);

        output[i] = silu_x;
    }
}
```

### Example 3: Advanced Packed Operations with Swizzle

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Swizzle operations for cross-lane arithmetic
__device__ __forceinline__ __hip_bfloat162 swap_halves(
    __hip_bfloat162 x
) {
    // Swap low and high bf16 values
    __hip_bfloat162 result;
    result.x = x.y;
    result.y = x.x;
    return result;
}

// Horizontal reduction using packed ops
__device__ __forceinline__ __hip_bfloat16 horizontal_add(
    __hip_bfloat162 x
) {
    // Add low and high components
    return __hadd(x.x, x.y);
}

// Vectorized SiLU with reduction
__global__ void silu_with_reduction(
    const __hip_bfloat162* __restrict__ input,
    __hip_bfloat16* __restrict__ output,  // Note: scalar output
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pairs) {
        __hip_bfloat162 x = input[idx];

        // Compute sigmoid (using fast approximation from Example 2)
        __hip_bfloat162 sigmoid_x = sigmoid_fast_packed(x);

        // SiLU computation
        __hip_bfloat162 silu_x = __hmul2(x, sigmoid_x);

        // Horizontal reduction for some aggregation use case
        __hip_bfloat16 reduced = horizontal_add(silu_x);

        // Atomic add for global reduction (example use case)
        // Note: atomicAdd for bf16 may require conversion to float
        float reduced_f32 = __bfloat162float(reduced);
        atomicAdd((float*)&output[0], reduced_f32);
    }
}

// Mixed precision: compute in packed bf16, accumulate in fp32
__global__ void silu_mixed_precision(
    const __hip_bfloat162* __restrict__ input,
    float* __restrict__ output,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pairs) {
        __hip_bfloat162 x = input[idx];
        __hip_bfloat162 sigmoid_x = sigmoid_fast_packed(x);
        __hip_bfloat162 silu_x = __hmul2(x, sigmoid_x);

        // Convert to fp32 for higher precision output
        output[idx * 2] = __bfloat162float(silu_x.x);
        output[idx * 2 + 1] = __bfloat162float(silu_x.y);
    }
}
```

## Best Practices

**Use Native Packed Functions**: Always prefer `__hadd2`, `__hmul2`, `__hfma2` over manually unpacking, operating on scalars, and repacking. Packed operations provide 2x throughput and better register utilization, reducing instruction count by approximately 50% for element-wise operations.

**Leverage FMA Instructions**: Use `__hfma2(a, b, c)` for `a*b + c` computations instead of separate multiply and add. FMA provides higher precision (no intermediate rounding) and better performance (single instruction vs two), which is especially important for sigmoid and SiLU approximations.

**Handle Special Values**: Be aware that bf16 has limited precision (7.2 decimal digits vs 3-4 for fp16). For SiLU, clamp inputs to a reasonable range (e.g., [-10, 10]) before computing sigmoid to avoid numerical issues with extreme values.

**Optimize Math Functions**: When packed math functions like `h2exp` are not available or have poor performance, implement fast approximations using polynomial expansions with packed FMA operations. For SiLU, a 3rd-order polynomial approximation of sigmoid provides good accuracy with minimal instruction count.

**Memory Layout Considerations**: Ensure input data is properly interleaved for packed operations. If data comes as separate arrays of bf16 values, consider a preprocessing step to pack them into `__hip_bfloat162` format, or modify the kernel to pack during loading.

**Compiler Optimization**: Use `__forceinline__` for packed operation helper functions to ensure they are inlined and don't introduce function call overhead. Enable aggressive compiler optimization (-O3) to allow the compiler to optimize packed operation sequences.

**Common Pitfalls**:
- Mixing scalar and packed operations unnecessarily reduces performance
- Forgetting that packed operations process both halves independently (no cross-lane communication)
- Not considering numerical precision limitations of bf16 for sensitive operations
- Inefficient packing/unpacking patterns that negate performance benefits

## Performance Considerations

Compared to scalar bf16 operations, packed arithmetic provides:
- **2x arithmetic throughput**: Process 2 values per instruction
- **50% reduction in instruction count**: Fewer instructions for same work
- **Better register efficiency**: 2 values per register vs 1
- **Improved ILP**: More instruction-level parallelism for out-of-order execution

For a SiLU kernel processing 1M elements as 500K packed pairs:
- Scalar bf16: ~3.5M instructions (load, sigmoid, multiply, store per element)
- Packed bf16: ~1.75M instructions (~50% reduction)

Actual speedup depends on whether the kernel is compute-bound or memory-bound. For memory-bound SiLU, expect 10-30% speedup from packed ops; for compute-bound variations, expect 40-80% speedup.

## References

- AMD Official Documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
- HIP bfloat16 Header: https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/hip__bfloat16_8h_source.html
- HIP Performance Guidelines: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- Related APIs: `__hadd2`, `__hsub2`, `__hmul2`, `__hdiv2`, `__hfma2`, `__hneg2`, `__hmax2`, `__hmin2`, `h2exp`, `h2log`, `__hip_bfloat162`
