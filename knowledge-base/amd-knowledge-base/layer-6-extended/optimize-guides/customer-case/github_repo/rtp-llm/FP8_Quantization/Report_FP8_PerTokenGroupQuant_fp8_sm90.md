---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: FP8 Per-Token Group Quantization V2

## Variant Context
- Input semantic type: Quantization (per-token group-wise FP8 quantization)
- Datatype(s): FP8 (e4m3), INT8, with bf16/fp16 inputs
- Data representation: Group-wise quantization with per-group scales
- Target architecture: SM90+ (NVIDIA Hopper), with SM100 (Blackwell) optimizations

## Functionality
This kernel performs per-token group-wise quantization to FP8 format for efficient inference. It is used in:
- MoE (Mixture of Experts) models for expert GEMM inputs
- DeepGEMM and DeepEP integration
- FP8 block inference pipelines

Key features:
- Group-wise scale computation for better precision
- Optional SiLU activation fusion
- Support for UE8M0 scale format (unsigned 8-bit exponent-only)
- Efficient warp-level reductions for scale computation

## Optimization 1: UE8M0 Scale Format Support
- Commit ID: eb01e7eb2
- Optimization type: Precision / Memory
- Summary: Added support for UE8M0 (unsigned 8-bit exponent-only) scale format for DeepGEMM/DeepEP
- Detailed explanation:
  The UE8M0 format stores only the exponent of the scale factor as an unsigned 8-bit integer. This:
  - Reduces scale storage from 32 bits (float) to 8 bits
  - Enables faster scale application using bit manipulation
  - Is compatible with DeepGEMM and DeepEP requirements
  
  The scale is computed as a power of 2, allowing efficient bit-level extraction.

- Code excerpt:
    ```cpp
    // Fast power of 2 computation using bit manipulation
    __forceinline__ __device__ float fast_pow2(int x) {
        // We can ensure `-126 <= x and x <= 127`
        uint32_t bits_x = (x + 127) << 23;
        return *reinterpret_cast<float*>(&bits_x);
    }
    
    // Fast ceiling log2 using bit manipulation
    __forceinline__ __device__ int fast_log2_ceil(float x) {
        auto bits_x   = *reinterpret_cast<uint32_t*>(&x);
        auto exp_x    = (bits_x >> 23) & 0xff;
        auto man_bits = bits_x & ((1 << 23) - 1);
        return exp_x - 127 + (man_bits != 0);
    }
    
    // Extract UE8M0 format scale
    template<bool SCALE_UE8M0, typename OUT_DTYPE_T = std::conditional_t<SCALE_UE8M0, uint8_t, float>>
    __forceinline__ __device__ OUT_DTYPE_T extract_required_scale_format(float value) {
        if constexpr (SCALE_UE8M0) {
            return static_cast<uint8_t>((*reinterpret_cast<uint32_t*>(&value)) >> 23);
        } else {
            return value;
        }
    }
    ```
- Evidence mapping:
  - UE8M0 template parameter → `template<bool SCALE_UE8M0, ...>`
  - Bit manipulation for scale → `fast_pow2`, `fast_log2_ceil`, `extract_required_scale_format`
  - 8-bit output type → `uint8_t` when SCALE_UE8M0 is true

## Optimization 2: Power-of-2 Scale Rounding for Efficient Computation
- Commit ID: eb01e7eb2
- Optimization type: Compute
- Summary: Option to round scales to powers of 2 for faster scale application
- Detailed explanation:
  When `ROUND_SCALE` is enabled, scales are rounded to the nearest power of 2. This allows:
  - Scale application using bit shifts instead of multiplication
  - Reduced register pressure
  - Faster execution on tensor cores
  
  The trade-off is slightly reduced precision, but this is acceptable for many inference workloads.

- Code excerpt:
    ```cpp
    template<bool ROUND_SCALE, typename dtype_info>
    __forceinline__ __device__ void calculate_fp8_scales(float amax, float& scale, float& scale_inv) {
        constexpr float MAX_8BIT_INV = 1.0f / dtype_info::MAX;
        if constexpr (ROUND_SCALE) {
            // Round to power of 2
            auto exp_scale_inv = fast_log2_ceil(amax * MAX_8BIT_INV);
            scale              = fast_pow2(-exp_scale_inv);
            scale_inv          = fast_pow2(exp_scale_inv);
        } else {
            // Exact scale
            scale_inv = amax * MAX_8BIT_INV;
            scale     = dtype_info::MAX / amax;
        }
    }
    ```
- Evidence mapping:
  - Template parameter → `template<bool ROUND_SCALE, ...>`
  - Power-of-2 rounding → `fast_log2_ceil` and `fast_pow2` for rounded path
  - Exact computation → Direct division for non-rounded path

## Optimization 3: Warp-Level Reduction for Max Computation
- Commit ID: eb01e7eb2
- Optimization type: Compute
- Summary: Efficient warp-level reduction using shuffle instructions for finding group maximum
- Detailed explanation:
  The kernel uses warp shuffle instructions to compute the maximum absolute value within each group:
  - Uses `__shfl_xor_sync` for butterfly reduction pattern
  - Supports configurable subwarp sizes (1, 2, 4, 8, 16 threads)
  - Avoids shared memory for small group sizes
  
  This is critical for computing the quantization scale efficiently.

- Code excerpt:
    ```cpp
    template<int THREADS_PER_SUBWARP>
    __device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
        unsigned mask = 0xffffffff;
    
        static_assert((THREADS_PER_SUBWARP & (THREADS_PER_SUBWARP - 1)) == 0 
                      && THREADS_PER_SUBWARP <= 16 && THREADS_PER_SUBWARP >= 1,
                      "THREADS_PER_SUBWARP must be 1, 2, 4, 8, or 16");
    
        if constexpr (THREADS_PER_SUBWARP >= 16) {
            val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
        }
        if constexpr (THREADS_PER_SUBWARP >= 8) {
            val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
        }
        if constexpr (THREADS_PER_SUBWARP >= 4) {
            val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
        }
        if constexpr (THREADS_PER_SUBWARP >= 2) {
            val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
        }
        return val;
    }
    ```
- Evidence mapping:
  - Shuffle-based reduction → `__shfl_xor_sync` with butterfly pattern
  - Compile-time optimization → `if constexpr` for different subwarp sizes
  - No shared memory → Pure register-based reduction

## Optimization 4: Fused SiLU Activation with Quantization
- Commit ID: eb01e7eb2
- Optimization type: Fusion
- Summary: Option to fuse SiLU activation with quantization to reduce memory traffic
- Detailed explanation:
  For MoE models, the gate activation (SiLU) can be fused with the quantization kernel:
  - Reads input once, applies SiLU, then quantizes
  - Reduces memory bandwidth by 2x compared to separate kernels
  - Uses optimized SiLU implementation for SM100+ (Blackwell)

- Code excerpt:
    ```cpp
    __device__ __forceinline__ float silu(const float& val) {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
        // Optimized for Blackwell using tanh
        float half = 0.5f * val;
        float t    = __tanhf(half);
        return half * (1.0f + t);
    #else
        // Standard implementation
        return val / (1.0f + __expf(-val));
    #endif
    }
    
    template<bool FUSE_SILU_AND_MUL>
    __device__ __forceinline__ int compute_input_group_start_offset(...) {
        return expert_idx * num_tokens_per_expert * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1)
               + token_idx * hidden_size * (FUSE_SILU_AND_MUL ? 2 : 1) 
               + hidden_dim_group_idx * group_size;
    }
    ```
- Evidence mapping:
  - Fusion template → `template<bool FUSE_SILU_AND_MUL>`
  - Architecture-specific SiLU → `#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)`
  - Offset calculation → Accounts for 2x input size when fused

## Optimization 5: Non-Cached Global Memory Loads
- Commit ID: eb01e7eb2
- Optimization type: Memory
- Summary: Use non-cached loads for streaming input data
- Detailed explanation:
  The kernel uses PTX assembly for non-cached global memory loads:
  - `ld.global.nc` bypasses L1 cache for streaming data
  - Prevents cache pollution from one-time-use input data
  - Improves effective cache utilization for other data

- Code excerpt:
    ```cpp
    __device__ __forceinline__ int4 ld_global_nc(const int4* ptr) {
        int4 ret;
        asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                     : "l"(ptr));
        return ret;
    }
    
    __device__ __forceinline__ void st_global(const int4* ptr, const int4& value) {
        asm volatile(
            "st.global.v4.s32 [%0], {%1, %2, %3, %4};" 
            ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
    }
    ```
- Evidence mapping:
  - Non-cached load → `ld.global.nc` PTX instruction
  - Vectorized access → `v4.s32` for 128-bit loads/stores
  - Inline assembly → Direct PTX for optimal code generation
