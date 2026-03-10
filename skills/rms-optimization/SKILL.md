---
name: rms-optimization
description: This skill should be used when optimizing rms kernel on AMD GPUs.
---

# rms Kernel Optimization

## Variant Context
- Input semantic type: RMS Normalization for transformer attention (QK normalization)
- Datatype(s): bf16 (bfloat16)
- Data representation: Dense tensor with group-wise normalization
- Target architecture: Generic HIP/AMD GPU (wave64)

## Functionality
This kernel performs fused Query-Key RMS normalization for transformer models. It computes the root mean square of input values within each group, then normalizes and scales the values with learnable gamma parameters (and optional bias). The kernel is optimized for wave64 AMD GPUs where each warp has 64 threads.

## Optimization 1: Vectorized Memory Access with bf16x2 Packing
- Commit ID: N/A (directory comparison)
- Optimization type: memory
- Summary: Use uint32_t to load/store 2 bf16 values at once, doubling memory bandwidth utilization
- Detailed explanation: The optimized kernel uses a union type `bf16x2_union` to reinterpret pairs of bf16 values as uint32_t. This allows loading 2 bf16 elements (4 bytes) in a single 32-bit memory transaction instead of two separate 16-bit loads. This effectively doubles the memory bandwidth utilization for bf16 data.
- Code excerpt:
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
    ```
- Evidence mapping:
  - "Vectorized access" → `reinterpret_cast<const uint32_t*>` for 32-bit loads
  - "2 bf16 values at once" → `bf16x2_union` with `bf16[2]` array
  - "Single memory transaction" → `val_packed.u32 = input_u32[tid]` loads 2 elements

## Optimization 2: Fully Unrolled Warp Reduction
- Commit ID: N/A (directory comparison)
- Optimization type: compute
- Summary: Replace generic templated warp reduction with fully unrolled wave64-specific reduction
- Detailed explanation: The baseline uses a templated warp reduction with a loop. The optimized version provides a fully unrolled reduction function specifically for wave64, eliminating loop overhead and ensuring optimal instruction scheduling. Each shuffle operation is explicit, allowing the compiler to schedule them optimally.
- Code excerpt:
    ```cpp
    // Baseline (generic with loop):
    // template<typename T, int WARP=64>
    // __device__ inline T warpReduceSum(T val) {
    //   #pragma unroll
    //   for (int offset = WARP / 2; offset > 0; offset >>= 1) {
    //     val = add(val, __shfl_xor(val, offset, WARP));
    //   }
    //   return val;
    // }
    
    // Optimized (fully unrolled for wave64):
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
  - "Fully unrolled" → Six explicit `__shfl_xor` operations instead of loop
  - "Wave64-specific" → Hardcoded offsets 32, 16, 8, 4, 2, 1 for 64-thread warp
  - "Forceinline" → `__forceinline__` ensures no function call overhead

## Optimization 3: Eliminated Shared Memory for Scale
- Commit ID: N/A (directory comparison)
- Optimization type: memory
- Summary: Remove shared memory usage for scale factor by leveraging warp-uniform values
- Detailed explanation: The baseline stores the computed scale in shared memory (`__shared__ float smem_scale`) and synchronizes. The optimized version computes the scale directly after warp reduction, recognizing that after a warp reduction, all threads in the warp have the same value. This eliminates shared memory access and the synchronization barrier.
- Code excerpt:
    ```cpp
    // Baseline:
    // __shared__ float smem_scale;
    // float variance = warpReduceSum(square_sum) / static_cast<float>(norm_size);
    // if (threadIdx.x == 0) smem_scale = rsqrtf(variance + eps);
    // __syncthreads();
    
    // Optimized:
    // Warp reduction and compute scale
    float total_sq = warpReduceSum64(square_sum);
    
    // Compute scale: rsqrt((sum/n) + eps)
    float scale = __frsqrt_rn(total_sq * (1.0f / 128.0f) + eps);
    ```
- Evidence mapping:
  - "Eliminated shared memory" → No `__shared__` declaration for scale
  - "No synchronization" → No `__syncthreads()` after scale computation
  - "Warp-uniform value" → All threads compute same scale after reduction

## Optimization 4: Fast Reciprocal Square Root Intrinsic
- Commit ID: N/A (directory comparison)
- Optimization type: compute
- Summary: Use hardware intrinsic __frsqrt_rn() instead of rsqrtf() for faster computation
- Detailed explanation: The optimized kernel uses `__frsqrt_rn()` (fast reciprocal square root with round-to-nearest) instead of the standard `rsqrtf()`. This maps directly to the GPU's special function unit (SFU) and provides faster execution with IEEE-compliant rounding.
- Code excerpt:
    ```cpp
    // Baseline:
    // if (threadIdx.x == 0) smem_scale = rsqrtf(variance + eps);
    
    // Optimized:
    float scale = __frsqrt_rn(total_sq * (1.0f / 128.0f) + eps);
    ```
- Evidence mapping:
  - "Hardware intrinsic" → `__frsqrt_rn()` function
  - "Fused multiply" → `total_sq * (1.0f / 128.0f)` instead of division

## Optimization 5: Simplified Kernel Structure with Launch Bounds
- Commit ID: N/A (directory comparison)
- Optimization type: launch
- Summary: Added __launch_bounds__(64) to help compiler optimize register allocation
- Detailed explanation: The optimized kernel adds `__launch_bounds__(64)` attribute, informing the compiler that exactly 64 threads will be launched per block. This allows the compiler to optimize register allocation knowing the exact occupancy requirements, potentially using more registers per thread for better performance.
- Code excerpt:
    ```cpp
    // Optimized kernel with launch bounds:
    template<typename T, bool IS_BIAS>
    __global__ __launch_bounds__(64)
    void fusedQkRmsNorm(T* __restrict__ input,
                        const T* __restrict__ q_gamma,
                        // ... parameters
    ```
- Evidence mapping:
  - "Launch bounds" → `__launch_bounds__(64)` attribute
  - "Register optimization" → Compiler knows exact thread count for allocation

## Optimization 6: Eliminated Loop for Fixed Norm Size
- Commit ID: N/A (directory comparison)
- Optimization type: compute
- Summary: Remove loops by assuming fixed norm_size=128 with 64 threads processing 2 elements each
- Detailed explanation: The baseline uses loops with `elements_per_thread` iterations. The optimized version assumes a fixed configuration (norm_size=128, 64 threads, 2 elements per thread via vectorization) and eliminates the loop entirely. Each thread processes exactly one uint32_t (2 bf16 values).
- Code excerpt:
    ```cpp
    // Baseline (with loop):
    // const int elements_per_thread = norm_size / (WARP * vec_size);
    // #pragma unroll 1
    // for (int i = 0; i < elements_per_thread; ++i) {
    //   const int elem_idx = i * WARP + threadIdx.x;
    //   // process element
    // }
    
    // Optimized (no loop, fixed 2 elements per thread):
    bf16x2_union val_packed;
    val_packed.u32 = input_u32[tid];
    float v0 = static_cast<float>(val_packed.bf16[0]);
    float v1 = static_cast<float>(val_packed.bf16[1]);
    float square_sum = v0 * v0 + v1 * v1;
    ```
- Evidence mapping:
  - "Eliminated loop" → Direct processing without for-loop
  - "Fixed 2 elements" → `val_packed.bf16[0]` and `val_packed.bf16[1]`
  - "Vectorized processing" → Single uint32_t load for 2 bf16 values
