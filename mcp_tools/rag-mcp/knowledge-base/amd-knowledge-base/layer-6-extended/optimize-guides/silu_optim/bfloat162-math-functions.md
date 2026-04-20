---
tags: ["optimization", "bf16", "hip", "silu", "math-functions"]
priority: "L1-important"
source_url: "https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html"
rocm_version: "6.0+"
last_updated: 2026-01-15
---

# Bfloat162 Mathematical Functions

## Overview

HIP provides a set of mathematical functions optimized for `__hip_bfloat162` types that compute transcendental and special functions on both halves of the packed value simultaneously. These functions are essential for implementing activation functions like SiLU, which requires computing `sigmoid(x) = 1/(1 + exp(-x))`. The exponential function `exp()` and its vectorized variant are critical components that directly impact both performance and numerical accuracy of element-wise kernels.

The bfloat162 math API includes vectorized versions of common mathematical operations such as exponential (`h2exp`), logarithm (`h2log`, `h2log2`, `h2log10`), square root (`h2sqrt`), trigonometric functions, and rounding operations (`h2floor`, `h2ceil`, `h2trunc`, `h2rint`). For SiLU optimization specifically, the exponential function is the most performance-critical component since sigmoid computation dominates the arithmetic workload.

Understanding the availability, performance characteristics, and numerical behavior of these functions is crucial for writing efficient HIP kernels. Some functions may not have native hardware implementations and instead fall back to scalar operations, potentially negating the benefits of vectorization.

## Technical Details

The HIP bfloat162 math function categories include:

1. **Exponential and Logarithmic Functions**:
   - `h2exp(x)`: Computes e^x for both halves, returns `{exp(x.x), exp(x.y)}`
   - `h2log(x)`: Natural logarithm for both halves
   - `h2log2(x)`: Base-2 logarithm
   - `h2log10(x)`: Base-10 logarithm
   - Note: Availability varies by ROCm version; may require scalar fallback

2. **Power and Root Functions**:
   - `h2sqrt(x)`: Square root for both halves
   - `h2rsqrt(x)`: Reciprocal square root (1/sqrt(x)), useful for normalization

3. **Rounding Functions**:
   - `h2floor(x)`: Floor (round toward -∞) for both halves
   - `h2ceil(x)`: Ceiling (round toward +∞)
   - `h2trunc(x)`: Truncate (round toward 0)
   - `h2rint(x)`: Round to nearest integer

4. **Trigonometric Functions** (less common in ML):
   - `h2sin(x)`, `h2cos(x)`: Sine and cosine
   - Primarily used in positional encodings, not in SiLU

Implementation characteristics:
- **Native Hardware Support**: Not all functions have dedicated hardware instructions; some are emulated using multiple operations
- **Precision**: BF16 math functions inherit reduced precision (~3 decimal digits), suitable for ML but not scientific computing
- **Performance**: Native operations (sqrt, floor) are fast (4-8 cycles); emulated operations (exp, log) may be slower (20-40 cycles)
- **Numerical Stability**: Range limitations due to bf16 format; exp(x) overflows for x > 88, underflows for x < -88

For SiLU implementation, the critical function is `exp(-x)` in the sigmoid computation. If `h2exp` is not available or performs poorly, alternative approaches include:
1. Convert to fp32, compute exp, convert back (hybrid precision)
2. Polynomial approximation of sigmoid (avoid exp entirely)
3. Lookup table with interpolation (trade memory for compute)

Performance comparison for sigmoid computation (1M elements):
- Native h2exp: ~100 μs (if available and optimized)
- Scalar exp with bf16 conversion: ~250 μs
- Polynomial approximation: ~80 μs (fastest, slight accuracy loss)

## Code Examples

### Example 1: Using h2exp for Sigmoid Computation

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Helper to create constant bf162
__device__ __forceinline__ __hip_bfloat162 make_bf162(float val) {
    return __bfloat162bfloat162(__float2bfloat16(val));
}

// Sigmoid using h2exp (if available in your ROCm version)
__device__ __hip_bfloat162 sigmoid_h2exp(
    __hip_bfloat162 x
) {
    // Note: h2exp may not be available in all ROCm versions
    // Check documentation for your specific version

    __hip_bfloat162 one = make_bf162(1.0f);

    // Negate input
    __hip_bfloat162 neg_x = __hneg2(x);

    // Compute exp(-x) for both halves (if h2exp available)
    // __hip_bfloat162 exp_neg_x = h2exp(neg_x);

    // Fallback: scalar exp
    float x1 = __bfloat162float(neg_x.x);
    float x2 = __bfloat162float(neg_x.y);
    __hip_bfloat162 exp_neg_x = __floats2bfloat162_rn(
        expf(x1),
        expf(x2)
    );

    // 1 + exp(-x)
    __hip_bfloat162 denom = __hadd2(one, exp_neg_x);

    // 1 / (1 + exp(-x))
    __hip_bfloat162 sigmoid = __hdiv2(one, denom);

    return sigmoid;
}

// Complete SiLU kernel using sigmoid_h2exp
__launch_bounds__(256, 4)
__global__ void silu_h2exp_kernel(
    const __hip_bfloat162* __restrict__ input,
    __hip_bfloat162* __restrict__ output,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_pairs; i += stride) {
        __hip_bfloat162 x = input[i];

        // Compute sigmoid
        __hip_bfloat162 sigmoid_x = sigmoid_h2exp(x);

        // SiLU: x * sigmoid(x)
        __hip_bfloat162 silu_x = __hmul2(x, sigmoid_x);

        output[i] = silu_x;
    }
}
```

### Example 2: Hybrid Precision Approach (BF16 + FP32 Math)

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Hybrid precision sigmoid: bf16 I/O, fp32 math
__device__ __hip_bfloat162 sigmoid_hybrid(
    __hip_bfloat162 x
) {
    // Convert to fp32 for math operations
    float x1 = __bfloat162float(x.x);
    float x2 = __bfloat162float(x.y);

    // Compute sigmoid in fp32 (more accurate)
    float sig1 = 1.0f / (1.0f + expf(-x1));
    float sig2 = 1.0f / (1.0f + expf(-x2));

    // Convert back to bf16
    return __floats2bfloat162_rn(sig1, sig2);
}

// SiLU kernel with hybrid precision sigmoid
__launch_bounds__(256, 4)
__global__ void silu_hybrid_kernel(
    const __hip_bfloat162* __restrict__ input,
    __hip_bfloat162* __restrict__ output,
    int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < num_pairs; i += stride) {
        __hip_bfloat162 x = input[i];
        __hip_bfloat162 sigmoid_x = sigmoid_hybrid(x);
        __hip_bfloat162 silu_x = __hmul2(x, sigmoid_x);
        output[i] = silu_x;
    }
}
```

### Example 3: Using h2sqrt for RMS Normalization

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// RMS normalization using h2sqrt
__device__ __hip_bfloat162 rms_normalize_bf162(
    __hip_bfloat162 x,
    __hip_bfloat162 mean_square  // Precomputed mean of squares
) {
    // Note: h2sqrt may not be available in all versions
    // Fallback implementation shown

    // Compute sqrt(mean_square) for both halves
    float ms1 = __bfloat162float(mean_square.x);
    float ms2 = __bfloat162float(mean_square.y);

    __hip_bfloat162 rms = __floats2bfloat162_rn(
        sqrtf(ms1),
        sqrtf(ms2)
    );

    // Normalize: x / rms
    return __hdiv2(x, rms);
}

// Example: h2rsqrt for fast inverse square root
__device__ __hip_bfloat162 fast_rms_normalize(
    __hip_bfloat162 x,
    __hip_bfloat162 mean_square
) {
    // Compute 1/sqrt(mean_square) directly
    // h2rsqrt is typically faster than h2sqrt + division

    float ms1 = __bfloat162float(mean_square.x);
    float ms2 = __bfloat162float(mean_square.y);

    __hip_bfloat162 inv_rms = __floats2bfloat162_rn(
        rsqrtf(ms1),  // 1/sqrt(ms1)
        rsqrtf(ms2)
    );

    // Normalize: x * (1/rms)
    return __hmul2(x, inv_rms);
}
```

### Example 4: Polynomial Approximation to Avoid Expensive Math

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Fast tanh approximation using polynomial (no exp needed)
__device__ __hip_bfloat162 tanh_approx_bf162(
    __hip_bfloat162 x
) {
    // Rational approximation: x * (27 + x^2) / (27 + 9*x^2)
    // Valid for |x| < 3, clamp otherwise

    __hip_bfloat162 abs_x = __hmul2(x, x);  // x^2 for magnitude check
    __hip_bfloat162 x2 = abs_x;

    __hip_bfloat162 c27 = make_bf162(27.0f);
    __hip_bfloat162 c9 = make_bf162(9.0f);
    __hip_bfloat162 c1 = make_bf162(1.0f);
    __hip_bfloat162 cneg1 = make_bf162(-1.0f);

    // Numerator: x * (27 + x^2)
    __hip_bfloat162 numer_inner = __hadd2(c27, x2);
    __hip_bfloat162 numer = __hmul2(x, numer_inner);

    // Denominator: 27 + 9*x^2
    __hip_bfloat162 denom = __hfma2(c9, x2, c27);

    // Result: numer / denom
    __hip_bfloat162 result = __hdiv2(numer, denom);

    // Clamp to [-1, 1]
    result = __hmax2(__hmin2(result, c1), cneg1);

    return result;
}

// GELU using tanh approximation (avoids erf)
__device__ __hip_bfloat162 gelu_fast_bf162(
    __hip_bfloat162 x
) {
    // GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    __hip_bfloat162 half = make_bf162(0.5f);
    __hip_bfloat162 one = make_bf162(1.0f);
    __hip_bfloat162 sqrt_2_pi = make_bf162(0.7978845608f);
    __hip_bfloat162 coeff = make_bf162(0.044715f);

    // Compute x^3
    __hip_bfloat162 x2 = __hmul2(x, x);
    __hip_bfloat162 x3 = __hmul2(x2, x);

    // Inner expression: x + 0.044715 * x^3
    __hip_bfloat162 inner = __hfma2(coeff, x3, x);

    // Scale by sqrt(2/π)
    inner = __hmul2(sqrt_2_pi, inner);

    // Apply tanh approximation
    __hip_bfloat162 tanh_val = tanh_approx_bf162(inner);

    // 1 + tanh(...)
    __hip_bfloat162 one_plus_tanh = __hadd2(one, tanh_val);

    // 0.5 * x * (1 + tanh(...))
    __hip_bfloat162 result = __hmul2(half, x);
    result = __hmul2(result, one_plus_tanh);

    return result;
}
```

## Best Practices

**Check Function Availability**: Before using `h2exp`, `h2log`, or other packed math functions, verify they are available in your ROCm version by checking the documentation and testing. Some functions may be defined but not optimized, resulting in worse performance than scalar fallbacks.

**Use Hybrid Precision for Transcendental Functions**: For functions like exp, log, sin, cos, computing in fp32 and converting results to bf16 often provides better performance and accuracy than pure bf16 math. The conversion overhead is small compared to the cost of transcendental operations.

**Prefer Polynomial Approximations**: For activation functions, polynomial approximations of sigmoid/tanh avoid expensive exponentials entirely. A 5th-order polynomial can approximate sigmoid on [-5, 5] with <1% error and 3-4x speedup over exp-based computation.

**Clamp Inputs Before Math Operations**: BF16's limited range makes overflow/underflow common for transcendental functions. Always clamp inputs: sigmoid to [-10, 10], tanh to [-3, 3], exp to [-88, 88] to ensure numerical stability.

**Use Reciprocal Square Root When Possible**: For normalization operations, `rsqrt(x)` is typically 2x faster than `1.0f / sqrt(x)`. If `h2rsqrt` is available, use it; otherwise, fp32 `rsqrtf` with conversion is still faster than divide.

**Batch Math Operations**: If you need to compute the same function (e.g., sqrt) on many values, ensure the kernel is vectorized and uses coalesced memory access. The math function itself is fast; memory bandwidth is usually the bottleneck.

**Profile to Identify Bottlenecks**: Use `rocprof` to determine if math functions or memory access dominate execution time. For SiLU, if sigmoid computation is >30% of runtime, consider faster approximations; if <10%, focus on memory optimization instead.

**Common Pitfalls**:
- Assuming all h2xxx functions are available and optimized (they're not always)
- Not clamping inputs before exponential/logarithmic operations
- Using bf16 math for accumulation (use fp32 for sums, products over many values)
- Ignoring fp32 fallback option for better accuracy when performance permits
- Over-optimizing math when memory bandwidth is the real bottleneck

## Performance Considerations

Approximate latencies for math functions on AMD MI250X (in cycles):
- **h2sqrt** (if native): 8-12 cycles
- **h2exp** (if native): 20-40 cycles; (fp32 fallback): 30-60 cycles
- **h2log** (if native): 20-40 cycles
- **Polynomial sigmoid** (5th order): 16-24 cycles
- **FP32 exp + conversion**: 30-50 cycles

For SiLU on 1M elements:
- Native h2exp sigmoid: ~100-150 μs (if optimized)
- Hybrid fp32 exp: ~200-250 μs
- Polynomial sigmoid: ~80-100 μs (fastest)

The optimal choice depends on accuracy requirements:
- **Research/inference**: Polynomial approximation (fastest, <1% error)
- **Training (forward pass)**: Hybrid fp32 (good speed/accuracy balance)
- **Training (gradient)**: Full fp32 (highest accuracy for backward pass)

## References

- HIP Low Precision Types: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
- ROCm Precision Support: https://rocm.docs.amd.com/en/latest/reference/precision-support.html
- HIP bfloat16 Header: https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/hip__bfloat16_8h_source.html
- Related APIs: `h2exp`, `h2log`, `h2log2`, `h2log10`, `h2sqrt`, `h2rsqrt`, `h2floor`, `h2ceil`, `h2trunc`, `h2rint`, `h2sin`, `h2cos`, polynomial approximations
