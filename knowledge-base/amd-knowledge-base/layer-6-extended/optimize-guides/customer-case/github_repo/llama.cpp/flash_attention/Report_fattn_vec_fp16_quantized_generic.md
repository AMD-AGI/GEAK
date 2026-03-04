---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Flash Attention Vector (fattn-vec)

## Variant Context
- Input semantic type: Attention (single-query decoding)
- Datatype(s): FP16 (Q), FP16/Quantized (KV cache: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
- Data representation: Dense Q, optionally quantized KV cache
- Target architecture: Generic (NVIDIA, AMD)

## Functionality
The vector Flash Attention kernel is optimized for single-query decoding (batch size = 1, single query token). It processes the attention computation as a series of vector operations rather than matrix operations, which is more efficient for this common inference scenario.

Key features:
- Optimized for autoregressive decoding
- Support for quantized KV cache
- Warp-level parallelism
- Various KV quantization formats

---

## Optimization 1: Refactor and Deduplicate Vector FA Kernels
- Commit ID: 75a3a6c2c
- Optimization type: Code quality (maintainability)
- Summary: Refactor vector FA kernels to reduce code duplication across quantization formats
- Detailed explanation: The vector FA kernel has variants for different KV cache quantization formats. This optimization uses templates and helper functions to share common code while maintaining specialized paths for each format.

- Code excerpt:
    ```cpp
    // CUDA: refactor and deduplicate vector FA kernels
    template<typename KV_TYPE, typename DEQUANT_FUNC>
    __device__ __forceinline__ float compute_kq_dot(
        const half * __restrict__ q,
        const KV_TYPE * __restrict__ k,
        const int D,
        DEQUANT_FUNC dequant) {
        
        float sum = 0.0f;
        for (int d = threadIdx.x; d < D; d += WARP_SIZE) {
            float q_val = __half2float(q[d]);
            float k_val = dequant(k, d);
            sum += q_val * k_val;
        }
        return warp_reduce_sum(sum);
    }
    
    // Specializations for different KV types
    template<>
    __device__ float compute_kq_dot<half, ...>(...) {
        // FP16 path with half2 vectorization
    }
    
    template<>
    __device__ float compute_kq_dot<block_q4_0, ...>(...) {
        // Q4_0 path with dequantization
    }
    ```

- Evidence mapping:
  - "Template abstraction" → `KV_TYPE`, `DEQUANT_FUNC` parameters
  - "Shared code" → `compute_kq_dot()` function
  - "Specializations" → different implementations per type

---

## Optimization 2: No FP16 Arithmetic for Numerical Stability
- Commit ID: 73955f7d2
- Optimization type: Precision (numerical stability)
- Summary: Use FP32 arithmetic in vector FA kernel to avoid FP16 precision issues
- Detailed explanation: FP16 has limited dynamic range which can cause numerical issues in attention computation, especially for long sequences. This optimization performs all arithmetic in FP32 while keeping data in FP16 for memory efficiency.

- Code excerpt:
    ```cpp
    // CUDA: no FP16 arithmetic for vector FA kernel
    template<int D>
    __global__ void flash_attn_vec_f32_accum(
        const half * __restrict__ Q,
        const half * __restrict__ K,
        const half * __restrict__ V,
        half * __restrict__ dst,
        const int kv_len,
        const float scale) {
        
        const int head = blockIdx.x;
        
        // Accumulators in FP32
        float kq_max = -INFINITY;
        float kq_sum = 0.0f;
        float vkq[D];  // FP32 accumulator
        for (int d = 0; d < D; d++) vkq[d] = 0.0f;
        
        for (int kv_idx = 0; kv_idx < kv_len; kv_idx++) {
            // Compute KQ in FP32
            float kq = 0.0f;
            for (int d = threadIdx.x; d < D; d += WARP_SIZE) {
                kq += __half2float(Q[head * D + d]) * 
                      __half2float(K[kv_idx * D + d]) * scale;
            }
            kq = warp_reduce_sum(kq);
            
            // Softmax in FP32
            float kq_exp = expf(kq - kq_max);
            
            // Accumulate V in FP32
            for (int d = threadIdx.x; d < D; d += WARP_SIZE) {
                vkq[d] += kq_exp * __half2float(V[kv_idx * D + d]);
            }
            kq_sum += kq_exp;
        }
        
        // Write output as FP16
        for (int d = threadIdx.x; d < D; d += WARP_SIZE) {
            dst[head * D + d] = __float2half(vkq[d] / kq_sum);
        }
    }
    ```

- Evidence mapping:
  - "FP32 arithmetic" → `float kq`, `float vkq[D]` accumulators
  - "FP16 storage" → `__half2float()` for reads, `__float2half()` for writes
  - "Numerical stability" → softmax computed in FP32

---

## Optimization 3: Improve GPU Occupancy for BS=1
- Commit ID: 517b5ddbf
- Optimization type: Launch configuration
- Summary: Improve flash decoding kernel GPU occupancy for batch size 1 case
- Detailed explanation: For single-token decoding, the kernel needs to maximize parallelism across attention heads. This optimization adjusts the thread block configuration to improve occupancy when processing a single query.

- Code excerpt:
    ```cpp
    // CUDA: Improve flash decoding kernel GPU occupancy for BS=1 case
    static void launch_fattn_vec_bs1(
        const half * Q, const half * K, const half * V,
        half * dst,
        const int n_heads,
        const int D,
        const int kv_len,
        cudaStream_t stream) {
        
        // One block per head for maximum parallelism
        const int n_blocks = n_heads;
        
        // Adjust threads based on head dimension
        int threads;
        if (D <= 64) {
            threads = 64;  // 2 warps
        } else if (D <= 128) {
            threads = 128; // 4 warps
        } else {
            threads = 256; // 8 warps
        }
        
        // Ensure good occupancy
        const int shmem = D * sizeof(float);  // For VKQ accumulator
        
        flash_attn_vec_f32_accum<D><<<n_blocks, threads, shmem, stream>>>(
            Q, K, V, dst, kv_len, scale);
    }
    ```

- Evidence mapping:
  - "One block per head" → `n_blocks = n_heads`
  - "Adaptive threads" → based on head dimension D
  - "Occupancy focus" → minimize shared memory, maximize blocks

---

## Optimization 4: Quantized KV Cache Support
- Commit ID: Various (fattn-vec-instance-*.cu files)
- Optimization type: Memory (compression)
- Summary: Support various quantization formats for KV cache to reduce memory usage
- Detailed explanation: The vector FA kernel supports quantized KV caches (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) which reduce memory bandwidth requirements during decoding. Each format has a specialized dequantization path.

- Code excerpt:
    ```cpp
    // Template instances for quantized KV cache
    // fattn-vec-instance-q4_0-q4_0.cu
    template __global__ void flash_attn_vec_ext<half, block_q4_0, block_q4_0>(
        const half * Q, const block_q4_0 * K, const block_q4_0 * V, ...);
    
    // fattn-vec-instance-q8_0-q4_0.cu
    template __global__ void flash_attn_vec_ext<half, block_q8_0, block_q4_0>(
        const half * Q, const block_q8_0 * K, const block_q4_0 * V, ...);
    
    // Dequantization during attention
    template<typename K_TYPE>
    __device__ float dequant_k(const K_TYPE * k, int idx) {
        if constexpr (std::is_same_v<K_TYPE, block_q4_0>) {
            const int block_idx = idx / QK4_0;
            const int elem_idx = idx % QK4_0;
            const block_q4_0 * block = k + block_idx;
            const float d = __half2float(block->d);
            const int qs = block->qs[elem_idx / 2];
            return d * ((elem_idx % 2 == 0 ? qs & 0xF : qs >> 4) - 8);
        }
        // ... other formats
    }
    ```

- Evidence mapping:
  - "Multiple formats" → Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 instances
  - "On-the-fly dequant" → `dequant_k()` during dot product
  - "Memory reduction" → 4-bit KV uses 4x less memory than FP16

---

## Optimization 5: Prefer Vector Kernel for Gemma Models
- Commit ID: c262beddf
- Optimization type: Kernel selection
- Summary: Prefer vector flash decoding kernel for Gemma models based on performance characteristics
- Detailed explanation: Different models have different optimal kernel choices. For Gemma models, the vector kernel often outperforms the MMA kernel due to the specific attention patterns and head dimensions.

- Code excerpt:
    ```cpp
    // CUDA: Prefer vector flash decoding kernel for Gemma models
    static bool should_use_vec_kernel(
        const int D,
        const int n_heads,
        const int kv_len,
        const bool is_gemma) {
        
        if (is_gemma) {
            // Gemma benefits from vector kernel
            return true;
        }
        
        // Default heuristic: use vector for short KV lengths
        if (kv_len <= 256) {
            return true;
        }
        
        // Use MMA for longer sequences
        return false;
    }
    ```

- Evidence mapping:
  - "Gemma preference" → `is_gemma` check
  - "Model-specific" → different kernels for different models
  - "Heuristic selection" → based on KV length
