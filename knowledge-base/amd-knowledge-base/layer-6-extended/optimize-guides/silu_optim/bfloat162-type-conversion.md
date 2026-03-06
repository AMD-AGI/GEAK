Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

---
tags: ["optimization", "bf16", "hip", "silu", "type-conversion"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/hip__bfloat16_8h_source.html"
rocm_version: "6.0+"
last_updated: 2026-01-15
---

# Bfloat162 Type Definition and Conversion

## Overview

The `__hip_bfloat162` type is a packed data structure that holds two `__hip_bfloat16` values, enabling SIMD-style vectorized operations on AMD GPUs. Understanding type conversion between `__hip_bfloat162`, scalar bfloat16, and standard floating-point types is essential for implementing efficient element-wise kernels like SiLU, where data must flow seamlessly between different precision domains without performance penalties.

Type conversions in HIP bf16 programming serve several purposes: (1) loading data from standard fp32 arrays into bf16 format for computation, (2) packing individual bf16 values into vectorized `__hip_bfloat162` for maximum throughput, (3) extracting individual results from packed types for output or boundary conditions, and (4) converting back to fp32 when higher precision is required for accumulation or output. For SiLU optimization, efficient conversion is critical at kernel boundaries where input data arrives in fp32 format and must be processed in bf16 for performance.

This document covers the complete type hierarchy, conversion functions, and best practices for minimizing conversion overhead in production kernels.

## Technical Details

The HIP bfloat16 type system consists of three primary types defined in `hip_bfloat16.h`:

1. **`__hip_bfloat16`**: 16-bit scalar type with 1 sign bit, 8 exponent bits, 7 mantissa bits. Provides identical range to fp32 (±3.4×10^38) with reduced precision (~3 decimal digits).

2. **`hip_bfloat16`**: C-style struct wrapper containing a single `__hip_bfloat16` member, used for compatibility and debugging. Generally avoid in device code; use `__hip_bfloat16` instead.

3. **`__hip_bfloat162`**: Packed type containing two `__hip_bfloat16` values (4 bytes total). Defined as a structure with `.x` (low) and `.y` (high) members, or as array-like indexable components.

Conversion functions and their characteristics:

**Scalar Conversions**:
- `__float2bfloat16(float)`: Converts fp32 to bf16 with rounding to nearest even
- `__bfloat162float(__hip_bfloat16)`: Exact conversion from bf16 to fp32 (no precision loss in this direction)
- `__double2bfloat16(double)`: Converts fp64 to bf16
- `__int2bfloat16_rn(int)`: Converts int32 to bf16 with round-to-nearest

**Packed Conversions**:
- `__floats2bfloat162_rn(float, float)`: Pack two fp32 values into `__hip_bfloat162`
- `__bfloat1622float2(__hip_bfloat162)`: Unpack to float2 structure
- `__bfloat162bfloat162(__hip_bfloat16)`: Replicate single bf16 to both halves
- `__halves2bfloat162(__hip_bfloat16, __hip_bfloat16)`: Combine two bf16 into packed type

Performance considerations:
- Conversion latency: 1-2 cycles for scalar, 2-4 cycles for packed
- Bandwidth: Conversions are register-only operations (no memory traffic)
- Compiler optimization: Modern LLVM can often eliminate redundant conversions
- Precision loss: Only fp32→bf16 loses precision; all other directions preserve or gain precision

For SiLU kernels, the typical pattern is: load fp32 → convert to bf16 → compute in bf16 → convert to fp32 for output (if needed). Minimize conversions inside tight loops by keeping all intermediate values in bf16 format.

## Code Examples

### Example 1: Basic Type Conversions

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

__global__ void conversion_examples_kernel() {
    // Scalar conversions
    float fp32_val = 3.14159f;
    __hip_bfloat16 bf16_val = __float2bfloat16(fp32_val);
    float converted_back = __bfloat162float(bf16_val);
    // Note: converted_back ≈ 3.140625 (precision loss)

    // Integer to bf16
    int int_val = 42;
    __hip_bfloat16 bf16_from_int = __int2bfloat16_rn(int_val);

    // Packed conversion: two floats to bfloat162
    float val1 = 1.5f;
    float val2 = 2.5f;
    __hip_bfloat162 packed = __floats2bfloat162_rn(val1, val2);

    // Access individual components
    __hip_bfloat16 low = packed.x;   // 1.5 in bf16
    __hip_bfloat16 high = packed.y;  // 2.5 in bf16

    // Replicate single bf16 to both halves
    __hip_bfloat16 scalar = __float2bfloat16(7.0f);
    __hip_bfloat162 replicated = __bfloat162bfloat162(scalar);
    // Result: {7.0, 7.0} in bf16

    // Combine two bf16 values
    __hip_bfloat16 a = __float2bfloat16(1.0f);
    __hip_bfloat16 b = __float2bfloat16(2.0f);
    __hip_bfloat162 combined = __halves2bfloat162(a, b);
    // Result: {1.0, 2.0} in bf16
}
```

### Example 2: Efficient Batch Conversion for SiLU

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Convert FP32 input array to BF16 with vectorization
__global__ void fp32_to_bf16_vectorized(
    const float* __restrict__ fp32_input,
    __hip_bfloat162* __restrict__ bf16_output,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_pairs; i += stride) {
        // Load two fp32 values
        float val1 = fp32_input[i * 2];
        float val2 = fp32_input[i * 2 + 1];

        // Convert and pack in one operation
        bf16_output[i] = __floats2bfloat162_rn(val1, val2);
    }
}

// Convert BF16 output back to FP32
__global__ void bf16_to_fp32_vectorized(
    const __hip_bfloat162* __restrict__ bf16_input,
    float* __restrict__ fp32_output,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_pairs; i += stride) {
        __hip_bfloat162 packed = bf16_input[i];

        // Extract and convert
        fp32_output[i * 2] = __bfloat162float(packed.x);
        fp32_output[i * 2 + 1] = __bfloat162float(packed.y);
    }
}

// In-place conversion: FP32 array → BF16 computation → FP32 output
__global__ void silu_with_conversion(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pairs = num_elements / 2;

    if (idx < num_pairs) {
        // Load and convert to bf16
        float x1 = input[idx * 2];
        float x2 = input[idx * 2 + 1];
        __hip_bfloat162 x_bf16 = __floats2bfloat162_rn(x1, x2);

        // Compute SiLU in bf16 (simplified)
        // In practice, use sigmoid approximation from other docs
        float sig1 = 1.0f / (1.0f + expf(-x1));
        float sig2 = 1.0f / (1.0f + expf(-x2));
        __hip_bfloat162 sigmoid_bf16 = __floats2bfloat162_rn(sig1, sig2);

        __hip_bfloat162 result_bf16 = __hmul2(x_bf16, sigmoid_bf16);

        // Convert back to fp32 for output
        output[idx * 2] = __bfloat162float(result_bf16.x);
        output[idx * 2 + 1] = __bfloat162float(result_bf16.y);
    }
}
```

### Example 3: Zero-Copy Reinterpretation

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Reinterpret existing BF16 data as packed without explicit conversion
__global__ void reinterpret_bf16_array(
    const __hip_bfloat16* __restrict__ bf16_scalar_array,
    __hip_bfloat162* __restrict__ bf16_packed_array,
    int num_elements
) {
    // This kernel demonstrates reinterpretation, but in practice
    // you would directly cast pointers at kernel launch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pairs = num_elements / 2;

    if (idx < num_pairs) {
        // Manual packing from adjacent scalar values
        __hip_bfloat16 low = bf16_scalar_array[idx * 2];
        __hip_bfloat16 high = bf16_scalar_array[idx * 2 + 1];

        bf16_packed_array[idx] = __halves2bfloat162(low, high);
    }
}

// More efficient: direct pointer casting (host code)
void launch_silu_with_reinterpret(
    __hip_bfloat16* d_bf16_data,
    int num_elements
) {
    // Reinterpret scalar bf16 array as packed bf16 array
    __hip_bfloat162* d_bf16_packed =
        reinterpret_cast<__hip_bfloat162*>(d_bf16_data);

    int num_pairs = num_elements / 2;
    int threads = 256;
    int blocks = (num_pairs + threads - 1) / threads;

    // Launch kernel that operates on packed data
    // (assumes silu_bf16_kernel defined elsewhere)
    hipLaunchKernelGGL(silu_bf16_kernel, dim3(blocks), dim3(threads),
                      0, 0, d_bf16_packed, d_bf16_packed, num_pairs);
}
```

### Example 4: Mixed Precision Conversion Patterns

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Helper: create constant bf162 from float
__device__ __forceinline__ __hip_bfloat162 make_bf162_constant(float val) {
    __hip_bfloat16 bf16_val = __float2bfloat16(val);
    return __bfloat162bfloat162(bf16_val);
}

// Mixed precision SiLU: bf16 compute, fp32 accumulation
__global__ void silu_mixed_precision(
    const __hip_bfloat162* __restrict__ input_bf16,
    float* __restrict__ output_fp32,
    float* __restrict__ accumulator,  // Global accumulator
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pairs) {
        // Load bf16 data
        __hip_bfloat162 x = input_bf16[idx];

        // Compute SiLU in bf16 (simplified sigmoid)
        float x1_f = __bfloat162float(x.x);
        float x2_f = __bfloat162float(x.y);

        float sig1 = 1.0f / (1.0f + expf(-x1_f));
        float sig2 = 1.0f / (1.0f + expf(-x2_f));

        __hip_bfloat162 sigmoid_x = __floats2bfloat162_rn(sig1, sig2);
        __hip_bfloat162 silu_bf16 = __hmul2(x, sigmoid_x);

        // Convert to fp32 for accumulation (higher precision)
        float result1 = __bfloat162float(silu_bf16.x);
        float result2 = __bfloat162float(silu_bf16.y);

        // Store fp32 output
        output_fp32[idx * 2] = result1;
        output_fp32[idx * 2 + 1] = result2;

        // Accumulate in fp32 (for reduction, sum, etc.)
        atomicAdd(accumulator, result1 + result2);
    }
}

// Compile-time constant conversion
template<int N>
__device__ __hip_bfloat162 get_constant_bf162() {
    // Compile-time constant folding
    constexpr float value = static_cast<float>(N);
    return make_bf162_constant(value);
}
```

## Best Practices

**Minimize Conversion Frequency**: Perform conversions at kernel boundaries (input/output) rather than inside tight loops. Once data is in bf16 format, keep it that way throughout the computation pipeline. Each conversion consumes 1-4 instruction slots and prevents other optimizations.

**Use Packed Conversions**: Always prefer `__floats2bfloat162_rn()` over converting two values separately and packing them. The packed conversion is a single instruction on modern AMD GPUs, while separate conversions require 2 instructions plus packing overhead.

**Leverage Pointer Reinterpretation**: When you have a contiguous array of bf16 values and need to treat them as packed pairs, use `reinterpret_cast<__hip_bfloat162*>` rather than launching a conversion kernel. This is zero-cost and allows immediate vectorized access.

**Understand Precision Loss Direction**: Converting fp32→bf16 loses precision and should happen early in the pipeline. Converting bf16→fp32 is exact (no loss) and can be deferred until output. For SiLU, compute entirely in bf16 and only convert to fp32 if the output consumer requires it.

**Use Compile-Time Constants**: For frequently used constants (0, 1, 0.5, etc.), create them once at compile time using constexpr and template functions rather than converting at runtime. The compiler can fold these into immediate values.

**Handle Alignment**: `__hip_bfloat162` requires 4-byte alignment. When casting pointers, ensure the source address is properly aligned. Use `__align__(4)` attribute for local variables if needed.

**Consider Conversion Latency**: While conversions are fast (1-4 cycles), they add to the critical path. Profile with `rocprof` to verify conversions aren't becoming a bottleneck. If >10% of execution time is in conversions, restructure to keep more data in bf16 format.

**Common Pitfalls**:
- Converting back and forth unnecessarily between bf16 and fp32
- Using scalar conversions in vectorized code instead of packed versions
- Not checking alignment when reinterpreting pointers
- Assuming bf16↔fp32 conversion is free (it's cheap but not zero-cost)
- Forgetting that fp32→bf16 is lossy and can accumulate errors if done repeatedly

## Performance Impact

Conversion performance on AMD MI250X (typical):
- `__float2bfloat16`: 1-2 cycles latency, 1/cycle throughput
- `__floats2bfloat162_rn`: 2-3 cycles latency, 0.5/cycle throughput
- `__bfloat162float`: 1 cycle latency, 1/cycle throughput
- Pointer reinterpretation: 0 cycles (compile-time)

For a SiLU kernel processing 1M elements:
- With conversions at boundaries: ~5-10 μs overhead (<2% of total time)
- With conversions in inner loop: ~50-100 μs overhead (10-20% of total time)

Best practice: Keep conversion overhead below 5% of total execution time.

## References

- AMD Official Documentation: https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/hip__bfloat16_8h_source.html
- HIP Low Precision Types: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
- ROCm Precision Support: https://rocm.docs.amd.com/en/latest/reference/precision-support.html
- Related APIs: `__float2bfloat16`, `__bfloat162float`, `__floats2bfloat162_rn`, `__bfloat162bfloat162`, `__halves2bfloat162`, `__hip_bfloat16`, `__hip_bfloat162`
