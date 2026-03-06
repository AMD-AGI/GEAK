Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: KV Cache Management Kernels

## Variant Context
- Input semantic type: KV cache operations (reshape, copy, quantize)
- Datatype(s): fp16, bf16, fp8
- Data representation: Paged KV cache blocks
- Target architecture: Generic CUDA/HIP

## Functionality
The KV cache kernels manage the paged key-value cache used in PagedAttention:
1. `reshape_and_cache`: Reshape and store new KV pairs into cache blocks
2. `copy_blocks`: Copy cache blocks for beam search and speculative decoding
3. `swap_blocks`: Swap cache blocks between GPU and CPU memory
4. Quantization variants for FP8 KV cache

## Optimization 1: Vectorized Cache Operations
- Commit ID: eb0fa4386
- Optimization type: Memory
- Summary: Optimize reshape_and_cache CUDA kernel with vectorized memory access
- Detailed explanation:
  This optimization uses vectorized loads and stores (float4, int4) to maximize memory bandwidth utilization. By loading/storing 4 elements at once, the kernel reduces the number of memory transactions.
- Code excerpt:
    ```cpp
    // Vectorized reshape_and_cache kernel
    template <typename scalar_t, int VEC_SIZE>
    __global__ void reshape_and_cache_kernel(
        scalar_t* __restrict__ key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
        scalar_t* __restrict__ value_cache,  // [num_blocks, num_heads, head_size, block_size]
        const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
        const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
        const int64_t* __restrict__ slot_mapping,  // [num_tokens]
        ...) {
      
      using VecType = typename Vec<scalar_t, VEC_SIZE>::Type;
      
      const int token_idx = blockIdx.x;
      const int slot_idx = slot_mapping[token_idx];
      
      // Vectorized copy for key
      const int num_vecs = head_size / VEC_SIZE;
      for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
        VecType key_vec = *reinterpret_cast<const VecType*>(
            key + token_idx * num_heads * head_size + head_idx * head_size + i * VEC_SIZE);
        
        // Compute cache location with proper layout
        int cache_offset = compute_cache_offset(slot_idx, head_idx, i * VEC_SIZE);
        *reinterpret_cast<VecType*>(key_cache + cache_offset) = key_vec;
      }
      
      // Similar vectorized copy for value
      // ...
    }
    ```
- Evidence mapping:
  - "Vectorized access" → `VecType` template for 4-element vectors
  - "Coalesced memory" → Threads access consecutive elements
  - "Reduced transactions" → `num_vecs = head_size / VEC_SIZE` iterations

## Optimization 2: Flash Attention Cache Layout
- Commit ID: eefbf4a68
- Optimization type: Memory
- Summary: Optimize reshape_and_cache_flash for FlashAttention-compatible layout
- Detailed explanation:
  FlashAttention uses a different cache layout than PagedAttention. This optimization provides an efficient kernel for the FlashAttention layout, avoiding costly layout transformations.
- Code excerpt:
    ```cpp
    // Cache layout for FlashAttention
    // Key cache: [num_blocks, block_size, num_heads, head_size]
    // Value cache: [num_blocks, block_size, num_heads, head_size]
    
    template <typename scalar_t>
    __global__ void reshape_and_cache_flash_kernel(
        scalar_t* __restrict__ key_cache,
        scalar_t* __restrict__ value_cache,
        const scalar_t* __restrict__ key,
        const scalar_t* __restrict__ value,
        const int64_t* __restrict__ slot_mapping,
        int num_heads, int head_size, int block_size) {
      
      const int token_idx = blockIdx.x;
      const int slot_idx = slot_mapping[token_idx];
      
      // Compute block and offset within block
      const int block_idx = slot_idx / block_size;
      const int block_offset = slot_idx % block_size;
      
      // FlashAttention layout: [block, slot, head, dim]
      for (int h = 0; h < num_heads; h++) {
        for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
          int src_offset = token_idx * num_heads * head_size + h * head_size + d;
          int dst_offset = block_idx * block_size * num_heads * head_size 
                         + block_offset * num_heads * head_size 
                         + h * head_size + d;
          
          key_cache[dst_offset] = key[src_offset];
          value_cache[dst_offset] = value[src_offset];
        }
      }
    }
    ```
- Evidence mapping:
  - "FlashAttention layout" → `[block, slot, head, dim]` ordering
  - "Direct storage" → No intermediate transformation needed
  - "Efficient indexing" → `block_offset * num_heads * head_size` stride

## Optimization 3: FP8 KV Cache Support
- Commit ID: 0e63494cf
- Optimization type: Memory / Precision
- Summary: Add FP8 support to reshape_and_cache_flash kernel
- Detailed explanation:
  This optimization adds FP8 quantization to the cache kernel, reducing memory usage by 2x. The quantization is performed inline during the cache write operation.
- Code excerpt:
    ```cpp
    // FP8 cache with inline quantization
    template <typename scalar_t, typename cache_t>
    __global__ void reshape_and_cache_flash_fp8_kernel(
        cache_t* __restrict__ key_cache,
        cache_t* __restrict__ value_cache,
        const scalar_t* __restrict__ key,
        const scalar_t* __restrict__ value,
        const int64_t* __restrict__ slot_mapping,
        const float* __restrict__ k_scale,
        const float* __restrict__ v_scale,
        ...) {
      
      const int token_idx = blockIdx.x;
      const float k_scale_val = *k_scale;
      const float v_scale_val = *v_scale;
      
      // Quantize and store
      for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
        scalar_t k_val = key[src_offset];
        scalar_t v_val = value[src_offset];
        
        // Quantize to FP8
        cache_t k_quant = float_to_fp8(static_cast<float>(k_val) / k_scale_val);
        cache_t v_quant = float_to_fp8(static_cast<float>(v_val) / v_scale_val);
        
        key_cache[dst_offset] = k_quant;
        value_cache[dst_offset] = v_quant;
      }
    }
    ```
- Evidence mapping:
  - "FP8 quantization" → `float_to_fp8` conversion
  - "Inline scaling" → Division by scale during write
  - "Separate cache type" → `cache_t` template for FP8 storage

## Optimization 4: Memory-Aligned KV Cache
- Commit ID: 75e94309e
- Optimization type: Memory
- Summary: Allocate KV caches with memory alignment for MLA performance
- Detailed explanation:
  Multi-head Latent Attention (MLA) benefits from aligned memory access. This optimization ensures KV cache tensors are allocated with proper stride ordering for optimal memory access patterns.
- Code excerpt:
    ```cpp
    // Allocate cache with stride order for alignment
    torch::Tensor allocate_kv_cache(
        int num_blocks, int block_size, int num_heads, int head_size,
        torch::ScalarType dtype, torch::Device device) {
      
      // Compute aligned strides
      int64_t head_stride = head_size;
      int64_t block_stride = num_heads * head_size;
      int64_t cache_stride = block_size * block_stride;
      
      // Align to 128 bytes for optimal memory access
      constexpr int ALIGNMENT = 128;
      cache_stride = ((cache_stride * element_size + ALIGNMENT - 1) / ALIGNMENT) 
                   * ALIGNMENT / element_size;
      
      // Create tensor with custom strides
      auto options = torch::TensorOptions().dtype(dtype).device(device);
      return torch::empty_strided(
          {num_blocks, block_size, num_heads, head_size},
          {cache_stride, block_stride, head_stride, 1},
          options);
    }
    ```
- Evidence mapping:
  - "Memory alignment" → `ALIGNMENT = 128` bytes
  - "Custom strides" → `torch::empty_strided` with computed strides
  - "MLA optimization" → Aligned access for latent attention

## Optimization 5: DeepSeek MLA Cache Kernel
- Commit ID: fa7e254a7, 47b933954
- Optimization type: Compute / Memory
- Summary: Add specialized cache kernels for DeepSeek-V3 MLA architecture
- Detailed explanation:
  DeepSeek-V3 uses Multi-head Latent Attention with compressed KV representations. This optimization adds specialized kernels for the MLA cache format with optimized memory access patterns.
- Code excerpt:
    ```cpp
    // DeepSeek MLA cache kernel
    template <typename scalar_t, typename cache_t>
    __global__ void concat_and_cache_ds_mla_kernel(
        cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
        const scalar_t* __restrict__ k_pe,   // [num_tokens, qk_rope_head_dim]
        const scalar_t* __restrict__ kv,     // [num_tokens, kv_lora_rank]
        const int64_t* __restrict__ slot_mapping,
        const float* __restrict__ scale,
        int kv_lora_rank, int qk_rope_head_dim) {
      
      const int token_idx = blockIdx.x;
      const int slot_idx = slot_mapping[token_idx];
      const int block_idx = slot_idx / block_size;
      const int block_offset = slot_idx % block_size;
      
      // Concatenate k_pe and kv into single cache entry
      // Layout: [k_pe (rope), kv (latent)]
      const int total_dim = kv_lora_rank + qk_rope_head_dim;
      
      for (int d = threadIdx.x; d < total_dim; d += blockDim.x) {
        float val;
        if (d < qk_rope_head_dim) {
          val = static_cast<float>(k_pe[token_idx * qk_rope_head_dim + d]);
        } else {
          val = static_cast<float>(kv[token_idx * kv_lora_rank + (d - qk_rope_head_dim)]);
        }
        
        // Quantize and store
        cache_t quant_val = float_to_cache_t(val / (*scale));
        kv_cache[block_idx * block_size * total_dim + block_offset * total_dim + d] = quant_val;
      }
    }
    ```
- Evidence mapping:
  - "MLA format" → Concatenated `k_pe` and `kv` latent vectors
  - "Compressed KV" → `kv_lora_rank` dimension instead of full head_size
  - "Inline quantization" → FP8 conversion during cache write

## Optimization 6: Optimized Gather and Dequantize
- Commit ID: 77e10c9ca
- Optimization type: Memory
- Summary: Optimize gather_and_maybe_dequant_cache kernel for long sequences
- Detailed explanation:
  For very long sequences, gathering KV cache entries becomes a bottleneck. This optimization improves memory access patterns and uses vectorized operations for the gather phase.
- Code excerpt:
    ```cpp
    // Optimized gather with dequantization
    template <typename scalar_t, typename cache_t, int VEC_SIZE>
    __global__ void gather_and_dequant_cache_kernel(
        scalar_t* __restrict__ output,
        const cache_t* __restrict__ cache,
        const int* __restrict__ block_tables,
        const int* __restrict__ seq_lens,
        const float* __restrict__ scale,
        int max_seq_len, int block_size, int head_size) {
      
      using VecType = typename Vec<cache_t, VEC_SIZE>::Type;
      using OutVecType = typename Vec<scalar_t, VEC_SIZE>::Type;
      
      const int seq_idx = blockIdx.x;
      const int head_idx = blockIdx.y;
      const int seq_len = seq_lens[seq_idx];
      
      // Process multiple tokens per thread for better efficiency
      for (int token_idx = threadIdx.x; token_idx < seq_len; token_idx += blockDim.x) {
        int block_idx = token_idx / block_size;
        int block_offset = token_idx % block_size;
        int physical_block = block_tables[seq_idx * max_blocks + block_idx];
        
        // Vectorized load and dequantize
        for (int d = 0; d < head_size; d += VEC_SIZE) {
          int cache_offset = physical_block * block_size * head_size 
                           + block_offset * head_size + d;
          VecType cache_vec = *reinterpret_cast<const VecType*>(cache + cache_offset);
          
          // Dequantize vector
          OutVecType out_vec;
          #pragma unroll
          for (int v = 0; v < VEC_SIZE; v++) {
            out_vec[v] = cache_t_to_float(cache_vec[v]) * (*scale);
          }
          
          int out_offset = seq_idx * max_seq_len * head_size 
                         + token_idx * head_size + d;
          *reinterpret_cast<OutVecType*>(output + out_offset) = out_vec;
        }
      }
    }
    ```
- Evidence mapping:
  - "Vectorized gather" → `VecType` for multi-element loads
  - "Fused dequantize" → Conversion in same kernel as gather
  - "Long sequence optimization" → Multiple tokens per thread
