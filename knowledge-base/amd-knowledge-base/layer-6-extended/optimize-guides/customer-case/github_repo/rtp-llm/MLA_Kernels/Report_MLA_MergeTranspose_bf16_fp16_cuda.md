---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: MLA (Multi-head Latent Attention) Merge and Transpose Kernels

## Variant Context
- Input semantic type: Attention (MLA-specific QKV processing for DeepSeek models)
- Datatype(s): bf16, fp16, fp32
- Data representation: Separate Q, K_nope, K_rope, V tensors merged into fused QKV
- Target architecture: CUDA (SM80+)

## Functionality
These kernels implement efficient data layout transformations for Multi-head Latent Attention (MLA), which is used in DeepSeek models. MLA differs from standard attention by:
- Separating K into K_nope (non-positional) and K_rope (rotary positional) components
- Using different head dimensions for different components
- Requiring efficient merging and broadcasting operations

The kernels fuse multiple operations:
1. Merge Q, K_nope, K_rope, V into a single QKV tensor
2. Apply RoPE transpose for rotary embeddings
3. Broadcast K_rope across all heads
4. Pad V to match the combined dimension

## Optimization 1: Fused QKV Merge with RoPE Transpose
- Commit ID: 281c590a2
- Optimization type: Fusion / Memory
- Summary: Fuse Q, K_nope, K_rope, V merge with RoPE dimension transpose in a single kernel
- Detailed explanation:
  The kernel performs multiple operations in a single pass:
  1. Copies Q_nope and K_nope directly
  2. Applies RoPE transpose: `[..., rope_dim/2, 2] => [..., 2, rope_dim/2]`
  3. Broadcasts K_rope from `[token_num, 1, rope_dim]` to all heads
  4. Pads V with zeros to match the combined dimension
  
  This reduces memory traffic by 4x compared to separate kernels.

- Code excerpt:
    ```cpp
    template<typename T>
    __global__ void mla_merge_transpose_kernel(T* q, T* k_nope, T* k_rope, T* v, T* qkv,
                                               int token_num, int head_num,
                                               int nope_head_dim, int rope_head_dim, int v_head_dim) {
        int nope_rope_dim = nope_head_dim + rope_head_dim;
        int tidx = threadIdx.x;
        int bs_idx = blockIdx.x;
        int head_idx = blockIdx.y;
        int rope_idx = tidx - nope_head_dim;
        int hidden_size = head_num * nope_rope_dim;
        
        int dst_base_offset = bs_idx * 3 * hidden_size + head_idx * nope_rope_dim + tidx;
        
        if (tidx < nope_head_dim) {
            // Direct copy for nope components
            qkv[dst_base_offset] = q[q_offset];
            qkv[dst_base_offset + hidden_size] = k_nope[k_nope_offset];
        } else {
            // RoPE transpose: [..., rope_dim/2, 2] => [..., 2, rope_dim/2]
            int trans_idx = rope_idx / 2;
            int trans_offset = trans_idx + (rope_idx % 2 ? 1 : 0) * rope_head_dim / 2 - tidx + nope_head_dim;
            int q_dst = dst_base_offset + trans_offset;
            int k_dst = q_dst + hidden_size;
            qkv[q_dst] = q[q_offset];
            qkv[k_dst] = k_rope[k_rope_offset];  // Broadcast from [token, 1, dim]
        }
        
        // Pad V with zeros
        if (tidx < v_head_dim) {
            qkv[dst_base_offset + 2 * hidden_size] = v[v_offset];
        } else {
            qkv[dst_base_offset + 2 * hidden_size] = 0;
        }
    }
    ```
- Evidence mapping:
  - Fused operations → Single kernel handles Q, K_nope, K_rope, V
  - RoPE transpose → `trans_idx` and `trans_offset` calculations
  - K_rope broadcast → `k_rope_offset = bs_idx * rope_head_dim + rope_idx` (no head_idx)
  - V padding → Conditional zero write for `tidx >= v_head_dim`

## Optimization 2: Warp-Level K Concatenation with Prefetching
- Commit ID: 281c590a2
- Optimization type: Memory / Compute
- Summary: Optimized K_nope and K_rope concatenation using warp-level operations and L2 prefetching
- Detailed explanation:
  The `concat_mla_k_kernel` is adapted from SGLang and uses:
  - Warp-level parallelism (32 threads per warp)
  - Vectorized loads/stores (int2 for nope, int for rope)
  - L2 cache prefetching for next iteration's data
  - Head chunking (16 heads per chunk) for better cache utilization

- Code excerpt:
    ```cpp
    template<typename T>
    __global__ void concat_mla_k_kernel(T* k, const T* k_nope, const T* k_rope, ...) {
        const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        const int token_id = flat_warp_id / NUM_HEAD_CHUNKS;
        const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
        const int lane_id = get_lane_id();
        
        using NopeVec = int2;  // 8B/thread, 32 threads = 256B/row
        using RopeVec = int;   // 4B/thread, 32 threads = 128B/row
        
        // Prefetch and load rope (broadcast across heads)
        const RopeVec rope_val = ld_na_global_v1(rope_base + lane_id);
        
        prefetch_L2(nope_src);
        NopeVec cur = ld_na_global_v2(nope_src);
        
        #pragma unroll
        for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
            NopeVec next;
            if (i + 1 < HEAD_CHUNK_SIZE) {
                prefetch_L2(next_src);  // Prefetch next iteration
                next = ld_na_global_v2(next_src);
            }
            
            st_na_global_v2(nope_dst, cur);
            st_na_global_v1(rope_dst, rope_val);  // Same rope for all heads
            
            // Update pointers
            nope_src += nope_src_stride_v;
            nope_dst += nope_dst_stride_v;
            rope_dst += rope_dst_stride_v;
            cur = next;
        }
    }
    ```
- Evidence mapping:
  - Warp-level parallelism → `flat_warp_id`, `lane_id` calculations
  - Vectorized access → `int2` for 256B/row, `int` for 128B/row
  - L2 prefetching → `prefetch_L2(nope_src)` and `prefetch_L2(next_src)`
  - Head chunking → `HEAD_CHUNK_SIZE = 16`, `NUM_HEAD_CHUNKS = 8`

## Optimization 3: Non-Temporal Memory Access
- Commit ID: 281c590a2
- Optimization type: Memory
- Summary: Use non-temporal (non-allocating) loads and stores for streaming data
- Detailed explanation:
  The kernel uses non-temporal memory access patterns:
  - `ld_na_global_v1/v2`: Non-allocating global loads (bypass L1)
  - `st_na_global_v1/v2`: Non-allocating global stores (bypass L1)
  
  This prevents cache pollution from streaming data that won't be reused.

- Code excerpt:
    ```cpp
    // Non-allocating vectorized load (int2 = 8 bytes)
    NopeVec cur = ld_na_global_v2(nope_src);
    
    // Non-allocating vectorized load (int = 4 bytes)
    const RopeVec rope_val = ld_na_global_v1(rope_base + lane_id);
    
    // Non-allocating vectorized stores
    st_na_global_v2(nope_dst, cur);
    st_na_global_v1(rope_dst, rope_val);
    ```
- Evidence mapping:
  - Non-allocating loads → `ld_na_global_v1`, `ld_na_global_v2` functions
  - Non-allocating stores → `st_na_global_v1`, `st_na_global_v2` functions
  - Vectorized access → `v1` for 4-byte, `v2` for 8-byte operations

## Optimization 4: Compile-Time Dimension Constants
- Commit ID: 281c590a2
- Optimization type: Compute
- Summary: Use compile-time constants for MLA-specific dimensions
- Detailed explanation:
  The kernel uses compile-time constants for DeepSeek's MLA dimensions:
  - `NUM_LOCAL_HEADS = 128`: Number of attention heads
  - `QK_NOPE_HEAD_DIM = 128`: Non-positional head dimension
  - `QK_ROPE_HEAD_DIM = 64`: Rotary positional head dimension
  - `HEAD_CHUNK_SIZE = 16`: Heads processed per warp
  
  This enables compiler optimizations like loop unrolling and constant folding.

- Code excerpt:
    ```cpp
    constexpr int NUM_LOCAL_HEADS  = 128;
    constexpr int QK_NOPE_HEAD_DIM = 128;
    constexpr int QK_ROPE_HEAD_DIM = 64;
    constexpr int HEAD_CHUNK_SIZE  = 16;
    constexpr int NUM_HEAD_CHUNKS  = NUM_LOCAL_HEADS / HEAD_CHUNK_SIZE;
    
    // Static assertions for vectorization correctness
    static_assert(sizeof(NopeVec) * 32 == QK_NOPE_HEAD_DIM * sizeof(nv_bfloat16), 
                  "nope vec mismatch");
    static_assert(sizeof(RopeVec) * 32 == QK_ROPE_HEAD_DIM * sizeof(nv_bfloat16), 
                  "rope vec mismatch");
    ```
- Evidence mapping:
  - Compile-time constants → `constexpr int` declarations
  - Static validation → `static_assert` for vectorization correctness
  - Loop unrolling → `#pragma unroll` with known iteration count
