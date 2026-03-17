# Kernel: KV Cache Kernels

## Variant Context
- Input semantic type: Cache management (KV cache operations for attention)
- Datatype(s): bf16, fp16, fp32, int8, fp8
- Data representation: Paged or contiguous KV cache
- Target architecture: CUDA (SM70+), ROCm (gfx942)

## Functionality
These kernels manage the Key-Value cache for efficient LLM inference:
- Write new K/V values to cache
- Reorder cache for beam search
- Copy cache between locations
- Support for paged attention cache layout

The KV cache is critical for:
- Avoiding recomputation of past K/V values
- Supporting long context lengths
- Enabling efficient batched inference

## Optimization 1: Fused KV Cache Write with RoPE
- Commit ID: 281c590a2
- Optimization type: Fusion
- Summary: Fuse RoPE application with KV cache write to reduce memory traffic
- Detailed explanation:
  The kernel combines multiple operations:
  - Apply rotary position embedding to K
  - Write K and V to cache in single pass
  - Support for different cache layouts (paged, contiguous)

- Code excerpt:
    ```cpp
    // From kv_cache_kernels.cu
    template<typename T, typename Tcache>
    __global__ void writeKVCacheWithRoPEKernel(
        Tcache* k_cache, Tcache* v_cache,
        const T* k, const T* v,
        const float* rope_cos, const float* rope_sin,
        const int* block_table, const int* seq_lens,
        int batch_size, int num_heads, int head_dim,
        int block_size, int max_blocks) {
        
        int batch_idx = blockIdx.x;
        int head_idx = blockIdx.y;
        int tid = threadIdx.x;
        
        int seq_len = seq_lens[batch_idx];
        int block_idx = seq_len / block_size;
        int block_offset = seq_len % block_size;
        int physical_block = block_table[batch_idx * max_blocks + block_idx];
        
        // Apply RoPE to K
        int rope_idx = tid % (head_dim / 2);
        float cos_val = rope_cos[seq_len * head_dim / 2 + rope_idx];
        float sin_val = rope_sin[seq_len * head_dim / 2 + rope_idx];
        
        float k_val = static_cast<float>(k[...]);
        float k_rotated;
        if (tid < head_dim / 2) {
            k_rotated = k_val * cos_val - k_val_pair * sin_val;
        } else {
            k_rotated = k_val * cos_val + k_val_pair * sin_val;
        }
        
        // Write to cache
        int cache_offset = physical_block * block_size * num_heads * head_dim
                         + block_offset * num_heads * head_dim
                         + head_idx * head_dim + tid;
        k_cache[cache_offset] = static_cast<Tcache>(k_rotated);
        v_cache[cache_offset] = static_cast<Tcache>(v[...]);
    }
    ```
- Evidence mapping:
  - Fused RoPE → `cos_val`, `sin_val` applied in same kernel
  - Paged layout → `block_table` for physical block mapping
  - Type conversion → `static_cast<Tcache>` for quantized cache

## Optimization 2: KV Cache Reordering for Beam Search
- Commit ID: 52f5085a1
- Optimization type: Memory
- Summary: Efficient cache reordering for beam search without full copy
- Detailed explanation:
  During beam search, cache entries need reordering based on beam indices:
  - Uses indirect indexing to avoid full copy
  - Supports in-place reordering
  - Optimized for common beam sizes (4, 8)

- Code excerpt:
    ```cpp
    // KV cache reorder kernel
    template<typename T>
    __global__ void reorderKVCacheKernel(
        T* k_cache, T* v_cache,
        const int* beam_indices,
        int batch_size, int beam_width,
        int num_heads, int seq_len, int head_dim) {
        
        int batch_idx = blockIdx.x / beam_width;
        int beam_idx = blockIdx.x % beam_width;
        int head_idx = blockIdx.y;
        int tid = threadIdx.x;
        
        // Get source beam index
        int src_beam = beam_indices[batch_idx * beam_width + beam_idx];
        
        // Copy from source to destination
        for (int s = 0; s < seq_len; s++) {
            int src_offset = (batch_idx * beam_width + src_beam) * num_heads * seq_len * head_dim
                           + head_idx * seq_len * head_dim + s * head_dim + tid;
            int dst_offset = (batch_idx * beam_width + beam_idx) * num_heads * seq_len * head_dim
                           + head_idx * seq_len * head_dim + s * head_dim + tid;
            
            k_cache[dst_offset] = k_cache[src_offset];
            v_cache[dst_offset] = v_cache[src_offset];
        }
    }
    ```
- Evidence mapping:
  - Beam indexing → `beam_indices[batch_idx * beam_width + beam_idx]`
  - Indirect copy → Source determined by `src_beam`
  - Full sequence copy → Loop over `seq_len`

## Optimization 3: Quantized KV Cache Support
- Commit ID: (core implementation)
- Optimization type: Memory
- Summary: Support for INT8 and FP8 quantized KV cache to reduce memory footprint
- Detailed explanation:
  The kernels support quantized cache formats:
  - INT8 with per-head or per-token scales
  - FP8 (e4m3) for Hopper GPUs
  - Automatic quantization during cache write

- Code excerpt:
    ```cpp
    // Quantized cache write
    template<typename T, typename Tcache>
    __global__ void writeQuantizedKVCacheKernel(
        Tcache* k_cache, Tcache* v_cache,
        float* k_scales, float* v_scales,
        const T* k, const T* v,
        int batch_size, int num_heads, int head_dim) {
        
        int batch_idx = blockIdx.x;
        int head_idx = blockIdx.y;
        int tid = threadIdx.x;
        
        // Compute scale for this head
        __shared__ float k_max, v_max;
        float k_val = fabsf(static_cast<float>(k[...]));
        float v_val = fabsf(static_cast<float>(v[...]));
        
        k_max = blockReduceMax(k_val);
        v_max = blockReduceMax(v_val);
        
        float k_scale = k_max / 127.0f;  // INT8 range
        float v_scale = v_max / 127.0f;
        
        if (tid == 0) {
            k_scales[batch_idx * num_heads + head_idx] = k_scale;
            v_scales[batch_idx * num_heads + head_idx] = v_scale;
        }
        
        // Quantize and store
        k_cache[...] = static_cast<Tcache>(static_cast<float>(k[...]) / k_scale);
        v_cache[...] = static_cast<Tcache>(static_cast<float>(v[...]) / v_scale);
    }
    ```
- Evidence mapping:
  - Scale computation → `blockReduceMax` for finding max value
  - Per-head scales → `k_scales[batch_idx * num_heads + head_idx]`
  - Quantization → Division by scale before cast

## Optimization 4: Block Table Management for Paged Attention
- Commit ID: bbb09350f
- Optimization type: Memory
- Summary: Efficient block table management for paged KV cache
- Detailed explanation:
  The paged attention system uses block tables:
  - Maps logical sequence positions to physical blocks
  - Supports dynamic memory allocation
  - Enables efficient memory reuse

- Code excerpt:
    ```cpp
    // Block table update kernel
    __global__ void updateBlockTableKernel(
        int* block_table,
        const int* new_blocks,
        const int* seq_lens,
        int batch_size, int block_size, int max_blocks) {
        
        int batch_idx = blockIdx.x;
        int seq_len = seq_lens[batch_idx];
        int num_blocks = (seq_len + block_size - 1) / block_size;
        
        // Append new block if needed
        if (seq_len % block_size == 1) {
            // New block needed
            int new_block_idx = num_blocks - 1;
            block_table[batch_idx * max_blocks + new_block_idx] = 
                new_blocks[batch_idx];
        }
    }
    
    // Convert sequence offset to block array data
    __device__ int getPhysicalBlock(
        const int* block_table, int batch_idx, 
        int seq_pos, int block_size, int max_blocks) {
        int block_idx = seq_pos / block_size;
        return block_table[batch_idx * max_blocks + block_idx];
    }
    ```
- Evidence mapping:
  - Block table → `block_table[batch_idx * max_blocks + block_idx]`
  - Dynamic allocation → `new_blocks[batch_idx]` for new physical blocks
  - Position mapping → `getPhysicalBlock` for logical to physical
