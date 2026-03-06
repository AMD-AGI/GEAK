Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: rms_norm

## Variant Context
- Input semantic type: RMS Normalization for transformer attention (QK normalization)
- Datatype(s): bf16 (bfloat16) input/output, fp32 accumulation
- Data representation: Dense tensors with grouped normalization
- Target architecture: Generic HIP/AMD GPU (wave64)

## Functionality
This kernel performs fused Query-Key RMS (Root Mean Square) normalization for transformer models. It normalizes groups of elements by computing the RMS value, then scales by gamma and optionally adds bias. The kernel processes multiple groups (Q and K) in a single launch, with each warp handling one group. The normalization formula is: `output = (input / sqrt(mean(input^2) + eps)) * gamma + bias`.

## Optimization 1: Vectorized Memory Access with bf16x2 Packing
- Optimization type: Memory
- Summary: Use uint32_t loads to read two bf16 values at once, doubling memory throughput
- Detailed explanation: The baseline kernel processes one bf16 element at a time in a loop. The optimized version uses a union type `bf16x2_union` to load two bf16 values as a single 32-bit word. This halves the number of memory transactions and better utilizes memory bandwidth. Each thread now processes 2 elements instead of iterating through multiple elements.
- Code excerpt (baseline):
    ```cpp
    const int elements_per_thread = norm_size / (WARP * vec_size);
    
    // 1) sum of squares (accumulate in float)
    float square_sum = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < elements_per_thread; ++i) {
      const int elem_idx = i * WARP + threadIdx.x;
      T vT = group_start[elem_idx];
      float_packed_t v = cuda_cast<float_packed_t>(vT);
      square_sum += cuda_sum<float>(v * v);
    }
    ```
- Code excerpt (optimized):
    ```cpp
    // Union for vectorized bf16 access
    union bf16x2_union {
        uint32_t u32;
        hip_bfloat16 bf16[2];
    };
    
    // Cast to uint32_t for vectorized access (2 bf16 = 1 uint32_t)
    const uint32_t* input_u32 = reinterpret_cast<const uint32_t*>(group_start);
    uint32_t* output_u32 = reinterpret_cast<uint32_t*>(group_start);
    const uint32_t* gamma_u32 = reinterpret_cast<const uint32_t*>(gamma);

    // Load 2 bf16 elements as one uint32_t
    bf16x2_union val_packed;
    val_packed.u32 = input_u32[tid];
    
    // Convert bf16 to float
    float v0 = static_cast<float>(val_packed.bf16[0]);
    float v1 = static_cast<float>(val_packed.bf16[1]);

    // 1) sum of squares
    float square_sum = v0 * v0 + v1 * v1;
    ```
- Evidence mapping:
  - "Vectorized load" → `val_packed.u32 = input_u32[tid]` loads 2 bf16 as one uint32
  - "Union for type punning" → `bf16x2_union` allows accessing uint32 as two bf16 values
  - "Eliminated loop" → Direct computation instead of `for` loop with `elements_per_thread`

## Optimization 2: Fully Unrolled Warp Reduction
- Optimization type: Compute
- Summary: Replaced generic templated warp reduction with fully unrolled wave64-specific reduction
- Detailed explanation: The baseline uses a templated `warpReduceSum` function with a `#pragma unroll` loop. The optimized version provides a fully unrolled, wave64-specific reduction function `warpReduceSum64` with explicit shuffle operations for each reduction step. This eliminates loop overhead and ensures optimal instruction scheduling.
- Code excerpt (baseline):
    ```cpp
    template<typename T, int WARP=64>
    __device__ inline T warpReduceSum(T val) {
      #pragma unroll
      for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        val = add(val, __shfl_xor(val, offset, WARP));
      }
      return val;
    }
    ```
- Code excerpt (optimized):
    ```cpp
    // Fast warp reduction using butterfly pattern - unrolled for wave64
    __device__ __forceinline__ float warpReduceSum64(float val) {
      val += __shfl_xor(val, 32, 64);
      val += __shfl_xor(val, 16, 64);
      val += __shfl_xor(val, 8, 64);
      val += __shfl_xor(val, 4, 64);
      val += __shfl_xor(val, 2, 64);
      val += __shfl_xor(val, 1, 64);
      return val;
    }
    ```
- Evidence mapping:
  - "Fully unrolled" → 6 explicit `__shfl_xor` calls instead of loop
  - "Wave64 specific" → Hardcoded offsets 32, 16, 8, 4, 2, 1 for 64-thread warp
  - "Forceinline" → `__forceinline__` ensures inlining

## Optimization 3: Fast Reciprocal Square Root Intrinsic
- Optimization type: Compute / Precision
- Summary: Use `__frsqrt_rn` intrinsic instead of separate division and rsqrtf operations
- Detailed explanation: The baseline computes the scale factor using shared memory to store the result of `rsqrtf(variance + eps)`. The optimized version uses `__frsqrt_rn` (fast reciprocal square root with round-to-nearest) directly, combining the variance computation and reciprocal square root into a single expression without shared memory synchronization.
- Code excerpt (baseline):
    ```cpp
    __shared__ float smem_scale;
    
    float variance = warpReduceSum(square_sum) / static_cast<float>(norm_size);
    if (threadIdx.x == 0) smem_scale = rsqrtf(variance + eps);
    __syncthreads();
    ```
- Code excerpt (optimized):
    ```cpp
    // Warp reduction and compute scale
    float total_sq = warpReduceSum64(square_sum);
    
    // Compute scale: rsqrt((sum/n) + eps)
    float scale = __frsqrt_rn(total_sq * (1.0f / 128.0f) + eps);
    ```
- Evidence mapping:
  - "Fast intrinsic" → `__frsqrt_rn` instead of `rsqrtf`
  - "Eliminated shared memory" → No `__shared__ float smem_scale` needed
  - "Eliminated syncthreads" → No `__syncthreads()` after scale computation
  - "Compile-time constant" → `(1.0f / 128.0f)` computed at compile time instead of runtime division

## Optimization 4: Eliminated Shared Memory Synchronization
- Optimization type: Scheduling
- Summary: Removed shared memory usage and __syncthreads() by leveraging warp-synchronous execution
- Detailed explanation: The baseline kernel uses shared memory (`smem_scale`) to broadcast the computed scale from thread 0 to all threads, requiring a `__syncthreads()` barrier. The optimized version exploits the fact that all threads in a warp execute the same reduction and compute the same scale value, eliminating the need for shared memory and synchronization.
- Code excerpt (baseline):
    ```cpp
    __shared__ float smem_scale;
    
    float variance = warpReduceSum(square_sum) / static_cast<float>(norm_size);
    if (threadIdx.x == 0) smem_scale = rsqrtf(variance + eps);
    __syncthreads();
    
    // 2) normalize, scale, (optional) add bias
    #pragma unroll 1
    for (int i = 0; i < elements_per_thread; ++i) {
      // ... uses smem_scale
    }
    ```
- Code excerpt (optimized):
    ```cpp
    // Warp reduction and compute scale
    float total_sq = warpReduceSum64(square_sum);
    
    // Compute scale: rsqrt((sum/n) + eps) - all threads compute same value
    float scale = __frsqrt_rn(total_sq * (1.0f / 128.0f) + eps);
    
    // ... directly uses scale without shared memory
    ```
- Evidence mapping:
  - "No shared memory" → Removed `__shared__ float smem_scale`
  - "No synchronization" → Removed `__syncthreads()` call
  - "All threads compute same value" → After warp reduction, all threads have identical `total_sq`

## Optimization 5: Launch Bounds Specification
- Optimization type: Launch Configuration
- Summary: Added `__launch_bounds__(64)` to optimize register allocation for wave64 execution
- Detailed explanation: The optimized kernel adds `__launch_bounds__(64)` attribute to inform the compiler that exactly 64 threads will be launched per block. This allows the compiler to optimize register allocation knowing the exact occupancy requirements.
- Code excerpt (optimized):
    ```cpp
    template<typename T, bool IS_BIAS>
    __global__ __launch_bounds__(64)
    void fusedQkRmsNorm(T* __restrict__ input,
                        const T* __restrict__ q_gamma,
                        // ...
    ```
- Evidence mapping:
  - "Launch bounds" → `__launch_bounds__(64)` attribute on kernel
  - "Wave64 optimization" → 64 threads matches AMD wave size

## Optimization 6: Vectorized Gamma and Output Access
- Optimization type: Memory
- Summary: Apply the same vectorized access pattern to gamma loading and output storing
- Detailed explanation: The optimized kernel applies vectorized access not just to input data, but also to gamma coefficients and output writes. This ensures consistent memory access patterns and maximizes memory throughput for all data paths.
- Code excerpt (optimized):
    ```cpp
    // Load gamma values (vectorized)
    bf16x2_union gamma_packed;
    gamma_packed.u32 = gamma_u32[tid];
    float g0 = static_cast<float>(gamma_packed.bf16[0]);
    float g1 = static_cast<float>(gamma_packed.bf16[1]);

    // 2) normalize and scale
    float out0 = v0 * scale * g0;
    float out1 = v1 * scale * g1;

    // Store result (vectorized)
    bf16x2_union result;
    result.bf16[0] = static_cast<hip_bfloat16>(out0);
    result.bf16[1] = static_cast<hip_bfloat16>(out1);
    output_u32[tid] = result.u32;
    ```
- Evidence mapping:
  - "Vectorized gamma load" → `gamma_packed.u32 = gamma_u32[tid]`
  - "Vectorized output store" → `output_u32[tid] = result.u32`
  - "Consistent access pattern" → All memory accesses use uint32 for 2x bf16
