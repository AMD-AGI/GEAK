---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Paged Attention (FP8 KV Cache)

## Variant Context
- Input semantic type: Attention (Query-Key-Value computation with FP8 quantized KV cache)
- Datatype(s): fp8_e4m3, fp8_e5m2 (KV cache), fp16/bf16 (query/output)
- Data representation: FP8 quantized paged KV cache with scaling factors
- Target architecture: CUDA sm89+ (Ada/Hopper), HIP gfx90a+ (MI200/MI300)

## Functionality
FP8 Paged Attention extends the base PagedAttention kernel to support FP8-quantized KV caches, reducing memory bandwidth and storage by 2x compared to FP16. The kernel dequantizes FP8 values on-the-fly during attention computation using per-tensor or per-token scaling factors.

## Optimization 1: FP8 E5M2 KV Cache Support
- Commit ID: 9090bf02e
- Optimization type: Memory / Precision
- Summary: Initial FP8-E5M2 KV cache support for reduced memory footprint
- Detailed explanation:
  FP8-E5M2 format provides 5 exponent bits and 2 mantissa bits, offering larger dynamic range than E4M3 at the cost of precision. This is suitable for KV cache where the dynamic range of values can be large. The implementation adds a new cache data type and dequantization logic.
- Code excerpt:
    ```cpp
    // FP8 E5M2 support in attention kernel
    template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
              int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE, ...>
    __device__ void paged_attention_kernel(...) {
      // Load FP8 values from cache
      cache_t k_cache_val = k_cache[...];
      
      // Dequantize to compute precision
      // For E5M2: larger dynamic range, suitable for attention scores
      scalar_t k_val = fp8_e5m2_to_fp16(k_cache_val) * k_scale;
    }
    ```
- Evidence mapping:
  - "FP8 E5M2 format" → `Fp8KVCacheDataType::kFp8E5M2` enum value
  - "2x memory reduction" → `cache_t` is 8-bit vs 16-bit `scalar_t`
  - "Scaling factor" → `k_scale` parameter for dequantization

## Optimization 2: FP8 E4M3 with NVIDIA Float8 Support
- Commit ID: c83310174
- Optimization type: Memory / Precision
- Summary: Add FP8-E4M3 support using NVIDIA's native float8_e4m3fn type for better precision
- Detailed explanation:
  FP8-E4M3 provides 4 exponent bits and 3 mantissa bits, offering better precision than E5M2 at the cost of dynamic range. This format is preferred when values are well-scaled. The implementation uses NVIDIA's native FP8 types for efficient hardware conversion.
- Code excerpt:
    ```cpp
    // NVIDIA FP8 E4M3 support
    #include <cuda_fp8.h>
    
    // Efficient conversion using hardware support
    template <>
    __device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 val) {
      return __half2float(__nv_cvt_fp8_to_halfraw(val, __NV_E4M3));
    }
    
    // In kernel: dequantize with scaling
    float k_val = fp8_to_float(k_cache_val) * (*k_scale);
    ```
- Evidence mapping:
  - "Native FP8 type" → `__nv_fp8_e4m3` CUDA type
  - "Hardware conversion" → `__nv_cvt_fp8_to_halfraw` intrinsic
  - "Per-tensor scaling" → Single `k_scale` value for entire tensor

## Optimization 3: ROCm FP8 Support
- Commit ID: 2ff767b51
- Optimization type: Memory / Precision
- Summary: Enable FP8 KV cache on AMD GPUs (MI200/MI300 series)
- Detailed explanation:
  This optimization ports FP8 KV cache support to AMD ROCm platform, using HIP-compatible FP8 conversion routines. The implementation handles the differences in FP8 intrinsics between CUDA and HIP.
- Code excerpt:
    ```cpp
    #ifdef USE_ROCM
    #include <hip/hip_fp8.h>
    #include "../quantization/w8a8/fp8/amd/quant_utils.cuh"
    
    // AMD-specific FP8 conversion
    template <>
    __device__ __forceinline__ float fp8_to_float(hip_fp8_e4m3 val) {
      return hip_fp8_to_float(val);
    }
    #else
    #include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
    #endif
    ```
- Evidence mapping:
  - "ROCm compatibility" → `#ifdef USE_ROCM` conditional compilation
  - "AMD FP8 types" → `hip_fp8_e4m3` type
  - "Platform-specific utils" → Separate `amd/quant_utils.cuh` and `nvidia/quant_utils.cuh`

## Optimization 4: Dynamic KV Cache Scaling Factors
- Commit ID: e97f802b2
- Optimization type: Precision
- Summary: Compute KV cache scaling factors dynamically during inference for better accuracy
- Detailed explanation:
  Instead of using static scaling factors from calibration, this optimization computes scaling factors dynamically based on the actual KV values. This improves accuracy by adapting to the actual data distribution at runtime.
- Code excerpt:
    ```cpp
    // Dynamic scaling factor computation
    // Compute max absolute value for the current KV block
    float k_max = 0.0f;
    for (int i = 0; i < block_size; i++) {
      k_max = fmaxf(k_max, fabsf(k_values[i]));
    }
    
    // Compute scale to map to FP8 range
    float k_scale = k_max / FP8_E4M3_MAX;
    
    // Quantize with dynamic scale
    for (int i = 0; i < block_size; i++) {
      k_cache[i] = float_to_fp8(k_values[i] / k_scale);
    }
    ```
- Evidence mapping:
  - "Dynamic computation" → Scale computed from actual data max values
  - "Per-block scaling" → Each KV block can have different scale
  - "Adaptive precision" → Scale adjusts to data distribution

## Optimization 5: Separate K and V Scales
- Commit ID: 978aed530
- Optimization type: Precision
- Summary: Use separate scaling factors for K and V caches for improved accuracy
- Detailed explanation:
  Keys and values in attention can have different value distributions. Using separate scaling factors allows each to be quantized optimally, improving overall accuracy compared to a single shared scale.
- Code excerpt:
    ```cpp
    // Separate scales for K and V
    template <...>
    __device__ void paged_attention_kernel(...,
        const float* k_scale,  // Scale for key cache
        const float* v_scale,  // Scale for value cache
        ...) {
      
      // Dequantize keys with k_scale
      float k_val = fp8_to_float(k_cache_val) * (*k_scale);
      
      // Dequantize values with v_scale  
      float v_val = fp8_to_float(v_cache_val) * (*v_scale);
    }
    ```
- Evidence mapping:
  - "Separate scales" → Distinct `k_scale` and `v_scale` parameters
  - "Independent quantization" → K and V can have different dynamic ranges
  - "Improved accuracy" → Each cache optimally quantized for its distribution
