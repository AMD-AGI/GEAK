# Kernel: Decoder Masked Multihead Attention (MMHA)

## Variant Context
- Input semantic type: Attention (decode phase single-token attention)
- Datatype(s): bf16, fp16, fp32
- Data representation: KV cache with masked attention
- Target architecture: CUDA (SM70+)

## Functionality
This kernel implements masked multi-head attention for the decode phase of LLM inference. During decode, each new token attends to all previous tokens in the KV cache. The kernel:
- Computes Q*K^T attention scores with causal masking
- Applies softmax normalization
- Computes weighted sum with V
- Supports various head dimensions and batch sizes

The implementation is based on FasterTransformer's optimized MMHA kernel.

## Optimization 1: Vectorized Memory Access
- Commit ID: (inherited from FasterTransformer)
- Optimization type: Memory
- Summary: Use vectorized loads/stores for efficient memory bandwidth utilization
- Detailed explanation:
  The kernel uses vectorized memory access patterns:
  - 128-bit loads for Q, K, V tensors
  - Coalesced memory access across threads in a warp
  - Shared memory for intermediate results

- Code excerpt:
    ```cpp
    // From decoder_masked_multihead_attention_template.h
    template<typename T, int Dh>
    struct Qk_vec_m_ {
        using Type = typename Qk_vec_m<T, Dh>::Type;
    };
    
    // Vectorized Q*K computation
    using Qk_vec = typename Qk_vec_m_<T, Dh>::Type;
    Qk_vec q_vec = *reinterpret_cast<const Qk_vec*>(&q[qk_offset]);
    Qk_vec k_vec = *reinterpret_cast<const Qk_vec*>(&k[qk_offset]);
    ```
- Evidence mapping:
  - Vectorized types → `Qk_vec_m_` template for different head dimensions
  - Reinterpret cast → Direct memory access as vector types

## Optimization 2: Warp-Level Reduction for Softmax
- Commit ID: (inherited from FasterTransformer)
- Optimization type: Compute
- Summary: Efficient warp-level reduction for softmax computation
- Detailed explanation:
  The softmax computation uses warp-level primitives:
  - `__shfl_xor_sync` for max reduction across warp
  - `__shfl_sync` for broadcasting max value
  - Fused exp and sum computation

- Code excerpt:
    ```cpp
    // From decoder_masked_multihead_attention_utils.h
    template<typename T, int NUM>
    __inline__ __device__ T warpReduceMax(T* val) {
        #pragma unroll
        for (int i = 0; i < NUM; i++) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                val[i] = fmaxf(val[i], __shfl_xor_sync(0xffffffff, val[i], mask, 32));
            }
        }
        return val[0];
    }
    ```
- Evidence mapping:
  - Warp reduction → `__shfl_xor_sync` with butterfly pattern
  - Template unrolling → `#pragma unroll` for compile-time optimization

## Optimization 3: Shared Memory for K/V Cache Access
- Commit ID: (inherited from FasterTransformer)
- Optimization type: Memory
- Summary: Use shared memory to reduce global memory traffic for K/V access
- Detailed explanation:
  The kernel loads K and V values into shared memory before computation:
  - Reduces redundant global memory accesses
  - Enables efficient data reuse across threads
  - Supports different cache layouts (blocked, linear)

- Code excerpt:
    ```cpp
    // Shared memory allocation for K/V
    extern __shared__ char smem_[];
    T* k_smem = reinterpret_cast<T*>(smem_);
    T* v_smem = k_smem + seq_len * head_dim;
    
    // Load K into shared memory
    for (int i = threadIdx.x; i < seq_len * head_dim; i += blockDim.x) {
        k_smem[i] = k_cache[i];
    }
    __syncthreads();
    ```
- Evidence mapping:
  - Shared memory declaration → `extern __shared__ char smem_[]`
  - Cooperative loading → Loop with `threadIdx.x` stride
  - Synchronization → `__syncthreads()` before computation

## Optimization 4: Template Specialization for Head Dimensions
- Commit ID: (inherited from FasterTransformer)
- Optimization type: Compute
- Summary: Compile-time specialization for common head dimensions (32, 64, 128, 256)
- Detailed explanation:
  The kernel uses template specialization for different head dimensions:
  - Enables compile-time loop unrolling
  - Optimizes register allocation
  - Reduces branch overhead

- Code excerpt:
    ```cpp
    // From decoder_masked_multihead_attention_launch.h
    template<typename T, int HEAD_SIZE>
    void launchDecoderMaskedMultiheadAttention(/* params */) {
        constexpr int THREADS_PER_BLOCK = 256;
        constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
        
        // Dispatch based on head size
        if constexpr (HEAD_SIZE == 64) {
            mmha_kernel<T, 64><<<grid, block, smem_size, stream>>>(params);
        } else if constexpr (HEAD_SIZE == 128) {
            mmha_kernel<T, 128><<<grid, block, smem_size, stream>>>(params);
        }
    }
    ```
- Evidence mapping:
  - Template parameter → `template<typename T, int HEAD_SIZE>`
  - Compile-time dispatch → `if constexpr` for head size selection
  - Constexpr configuration → `constexpr int THREADS_PER_BLOCK`
