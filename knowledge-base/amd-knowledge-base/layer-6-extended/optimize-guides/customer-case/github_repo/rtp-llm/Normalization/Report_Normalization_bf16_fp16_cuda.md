---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: LayerNorm and RMSNorm Kernels

## Variant Context
- Input semantic type: Normalization (layer normalization and RMS normalization)
- Datatype(s): bf16, fp16, fp32
- Data representation: Token-wise normalization with optional bias and residual
- Target architecture: CUDA (SM70+), ROCm (gfx942)

## Functionality
These kernels implement normalization operations critical for transformer models:
- **LayerNorm**: Normalizes across hidden dimension with learnable scale and bias
- **RMSNorm**: Root Mean Square normalization (used in LLaMA, Qwen, etc.)
- **Fused QK RMSNorm**: Specialized RMSNorm for Q and K in attention

The kernels support:
- Fused residual addition
- Optional bias addition
- FP8 quantization output
- Various hidden dimensions

## Optimization 1: Warp-Level Reduction for Mean/Variance
- Commit ID: (core implementation)
- Optimization type: Compute
- Summary: Efficient warp-level reduction for computing mean and variance
- Detailed explanation:
  The normalization kernels use warp-level reductions to compute statistics:
  - Each warp computes partial sums
  - Warp shuffle instructions combine partial results
  - Avoids shared memory for small hidden dimensions

- Code excerpt:
    ```cpp
    // From layernorm_kernels.cu
    template<typename T>
    __inline__ __device__ T warpReduceSum(T val) {
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, mask, 32);
        }
        return val;
    }
    
    // Block-level reduction for larger hidden dims
    template<typename T>
    __inline__ __device__ T blockReduceSum(T val) {
        __shared__ T shared[32];
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        
        val = warpReduceSum(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();
        
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : (T)0.0f;
        if (wid == 0) val = warpReduceSum(val);
        return val;
    }
    ```
- Evidence mapping:
  - Warp reduction → `__shfl_xor_sync` with butterfly pattern
  - Block reduction → Two-level reduction with shared memory
  - Template for types → Works with fp16, bf16, fp32

## Optimization 2: Vectorized Load/Store Operations
- Commit ID: (core implementation)
- Optimization type: Memory
- Summary: Use vectorized memory operations for efficient bandwidth utilization
- Detailed explanation:
  The kernels use vectorized loads and stores:
  - float4/half4 for 128-bit transactions
  - Aligned memory access patterns
  - Reduces number of memory instructions

- Code excerpt:
    ```cpp
    // Vectorized RMSNorm kernel
    template<typename T, int VEC_SIZE>
    __global__ void rmsNormKernel(T* output, const T* input, 
                                   const T* gamma, float eps, int hidden_size) {
        using VecT = typename VecType<T, VEC_SIZE>::Type;
        
        int idx = blockIdx.x;
        int tid = threadIdx.x;
        
        // Vectorized load
        VecT* input_vec = reinterpret_cast<VecT*>(input + idx * hidden_size);
        VecT val = input_vec[tid];
        
        // Compute sum of squares
        float sum_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            float v = static_cast<float>(val.data[i]);
            sum_sq += v * v;
        }
        
        // Reduce and normalize
        sum_sq = blockReduceSum(sum_sq);
        float rms = rsqrtf(sum_sq / hidden_size + eps);
        
        // Vectorized store with gamma
        VecT* gamma_vec = reinterpret_cast<VecT*>(gamma);
        VecT g = gamma_vec[tid];
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            val.data[i] = static_cast<T>(static_cast<float>(val.data[i]) * rms 
                                         * static_cast<float>(g.data[i]));
        }
        
        VecT* output_vec = reinterpret_cast<VecT*>(output + idx * hidden_size);
        output_vec[tid] = val;
    }
    ```
- Evidence mapping:
  - Vector type → `VecType<T, VEC_SIZE>::Type` for different sizes
  - Vectorized access → `reinterpret_cast<VecT*>` for aligned loads
  - Unrolled computation → `#pragma unroll` for vector elements

## Optimization 3: Fused Residual and Bias Addition
- Commit ID: (core implementation)
- Optimization type: Fusion
- Summary: Fuse residual connection and bias addition with normalization
- Detailed explanation:
  The kernels support fusing multiple operations:
  - Add residual connection before normalization
  - Add bias after normalization
  - Reduces memory traffic by 2-3x

- Code excerpt:
    ```cpp
    // Fused LayerNorm with residual
    template<typename T>
    __global__ void layerNormWithResidualKernel(
        T* output, T* residual_out,
        const T* input, const T* residual,
        const T* gamma, const T* beta,
        float eps, int hidden_size) {
        
        int idx = blockIdx.x;
        int tid = threadIdx.x;
        
        // Fused residual addition
        float val = static_cast<float>(input[idx * hidden_size + tid]) 
                  + static_cast<float>(residual[idx * hidden_size + tid]);
        
        // Store residual output for next layer
        residual_out[idx * hidden_size + tid] = static_cast<T>(val);
        
        // Compute mean and variance
        float mean = blockReduceSum(val) / hidden_size;
        float diff = val - mean;
        float var = blockReduceSum(diff * diff) / hidden_size;
        
        // Normalize with gamma and beta
        float normalized = (val - mean) * rsqrtf(var + eps);
        output[idx * hidden_size + tid] = static_cast<T>(
            normalized * static_cast<float>(gamma[tid]) 
            + static_cast<float>(beta[tid]));
    }
    ```
- Evidence mapping:
  - Fused residual → `input + residual` in same kernel
  - Dual output → Both normalized and residual outputs
  - Fused scale/bias → `gamma` and `beta` applied in same pass

## Optimization 4: Fused QK RMSNorm for Attention
- Commit ID: 80eeede7f (ROCm optimizations)
- Optimization type: Fusion
- Summary: Specialized RMSNorm for Q and K tensors in attention
- Detailed explanation:
  For attention computation, Q and K often need separate RMSNorm:
  - Fuses both normalizations in single kernel
  - Reduces kernel launch overhead
  - Optimizes memory access patterns for QKV layout

- Code excerpt:
    ```cpp
    // From fused_qk_rmsnorm.cu
    template<typename T>
    __global__ void fusedQKRmsNormKernel(
        T* q_out, T* k_out,
        const T* q_in, const T* k_in,
        const T* q_weight, const T* k_weight,
        float eps, int head_dim, int num_heads) {
        
        int token_idx = blockIdx.x;
        int head_idx = blockIdx.y;
        int tid = threadIdx.x;
        
        // Process Q
        float q_val = static_cast<float>(q_in[...]);
        float q_sum_sq = blockReduceSum(q_val * q_val);
        float q_rms = rsqrtf(q_sum_sq / head_dim + eps);
        q_out[...] = static_cast<T>(q_val * q_rms * static_cast<float>(q_weight[tid]));
        
        // Process K (same pattern)
        float k_val = static_cast<float>(k_in[...]);
        float k_sum_sq = blockReduceSum(k_val * k_val);
        float k_rms = rsqrtf(k_sum_sq / head_dim + eps);
        k_out[...] = static_cast<T>(k_val * k_rms * static_cast<float>(k_weight[tid]));
    }
    ```
- Evidence mapping:
  - Dual normalization → Both Q and K processed in same kernel
  - Per-head processing → `blockIdx.y` for head parallelism
  - Separate weights → `q_weight` and `k_weight` for different scales
