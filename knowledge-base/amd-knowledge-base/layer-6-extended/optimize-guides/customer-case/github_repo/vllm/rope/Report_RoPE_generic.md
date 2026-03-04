---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Rotary Position Embedding (RoPE)

## Variant Context
- Input semantic type: Position encoding for attention
- Datatype(s): fp16, bf16, fp32
- Data representation: Query/Key tensors with rotary dimensions
- Target architecture: Generic CUDA/HIP, Triton

## Functionality
RoPE kernels apply rotary position embeddings to query and key tensors:
1. Standard RoPE: Rotate pairs of dimensions based on position
2. Multi-resolution RoPE (MRoPE): Different frequencies for different head groups
3. Fused RoPE with cache operations

## Optimization 1: Vectorized RoPE Application
- Commit ID: (csrc/pos_encoding_kernels.cu)
- Optimization type: Memory / Compute
- Summary: Apply RoPE with vectorized memory access and fused sin/cos computation
- Detailed explanation:
  RoPE requires computing sin and cos for each position and dimension, then applying rotations. This kernel vectorizes the computation and uses fast math intrinsics.
- Code excerpt:
    ```cpp
    template <typename scalar_t, int VEC_SIZE>
    __global__ void rotary_embedding_kernel(
        scalar_t* __restrict__ query,
        scalar_t* __restrict__ key,
        const int64_t* __restrict__ positions,
        const float* __restrict__ cos_cache,  // Precomputed cos values
        const float* __restrict__ sin_cache,  // Precomputed sin values
        float rope_theta,
        int num_heads, int num_kv_heads, int head_size, int rotary_dim) {
      
      const int token_idx = blockIdx.x;
      const int head_idx = blockIdx.y;
      const int64_t position = positions[token_idx];
      
      // Process rotary dimensions in pairs
      for (int d = threadIdx.x * 2; d < rotary_dim; d += blockDim.x * 2) {
        // Get precomputed sin/cos or compute on the fly
        float cos_val, sin_val;
        if (cos_cache != nullptr) {
          cos_val = cos_cache[position * (rotary_dim / 2) + d / 2];
          sin_val = sin_cache[position * (rotary_dim / 2) + d / 2];
        } else {
          float freq = 1.0f / powf(rope_theta, float(d) / rotary_dim);
          float angle = position * freq;
          cos_val = cosf(angle);
          sin_val = sinf(angle);
        }
        
        // Load query pair
        int q_offset = token_idx * num_heads * head_size + head_idx * head_size;
        float q0 = static_cast<float>(query[q_offset + d]);
        float q1 = static_cast<float>(query[q_offset + d + 1]);
        
        // Apply rotation: [cos, -sin; sin, cos] @ [q0, q1]
        float q0_rot = q0 * cos_val - q1 * sin_val;
        float q1_rot = q0 * sin_val + q1 * cos_val;
        
        query[q_offset + d] = static_cast<scalar_t>(q0_rot);
        query[q_offset + d + 1] = static_cast<scalar_t>(q1_rot);
        
        // Apply same rotation to key (if this head has a corresponding KV head)
        if (head_idx < num_kv_heads) {
          int k_offset = token_idx * num_kv_heads * head_size + head_idx * head_size;
          float k0 = static_cast<float>(key[k_offset + d]);
          float k1 = static_cast<float>(key[k_offset + d + 1]);
          
          key[k_offset + d] = static_cast<scalar_t>(k0 * cos_val - k1 * sin_val);
          key[k_offset + d + 1] = static_cast<scalar_t>(k0 * sin_val + k1 * cos_val);
        }
      }
    }
    ```
- Evidence mapping:
  - "Pair processing" → Dimensions processed in pairs `(d, d+1)`
  - "Precomputed cache" → `cos_cache`, `sin_cache` for efficiency
  - "Fused Q/K" → Both query and key rotated in same kernel

## Optimization 2: Multi-Resolution RoPE (MRoPE)
- Commit ID: (vllm/model_executor/layers/rotary_embedding/mrope.py)
- Optimization type: Compute
- Summary: Support different RoPE frequencies for different head groups
- Detailed explanation:
  Some models use different position encoding frequencies for different attention heads. MRoPE supports this by allowing per-head-group theta values.
- Code excerpt:
    ```python
    @triton.jit
    def mrope_kernel(
        query_ptr, key_ptr, positions_ptr,
        theta_ptr,  # Per-head-group theta values
        num_heads, num_kv_heads, head_size, rotary_dim,
        num_head_groups,  # Number of different theta groups
        BLOCK_SIZE: tl.constexpr
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        
        # Determine which theta group this head belongs to
        heads_per_group = num_heads // num_head_groups
        group_idx = head_idx // heads_per_group
        theta = tl.load(theta_ptr + group_idx)
        
        position = tl.load(positions_ptr + token_idx)
        
        # Apply RoPE with group-specific theta
        for d in range(0, rotary_dim, 2):
            freq = 1.0 / tl.pow(theta, d / rotary_dim)
            angle = position * freq
            cos_val = tl.cos(angle)
            sin_val = tl.sin(angle)
            
            # Rotate query
            q_offset = token_idx * num_heads * head_size + head_idx * head_size + d
            q0 = tl.load(query_ptr + q_offset)
            q1 = tl.load(query_ptr + q_offset + 1)
            
            tl.store(query_ptr + q_offset, q0 * cos_val - q1 * sin_val)
            tl.store(query_ptr + q_offset + 1, q0 * sin_val + q1 * cos_val)
    ```
- Evidence mapping:
  - "Multi-resolution" → `theta_ptr` with per-group values
  - "Group assignment" → `group_idx = head_idx // heads_per_group`
  - "Variable frequencies" → Different theta per head group

## Optimization 3: Fused RoPE with Cache Write
- Commit ID: (integrated in cache kernels)
- Optimization type: Fusion
- Summary: Fuse RoPE application with KV cache write operation
- Detailed explanation:
  When writing to KV cache, RoPE is applied to keys. This optimization fuses both operations to avoid an intermediate tensor.
- Code excerpt:
    ```cpp
    template <typename scalar_t, typename cache_t>
    __global__ void rope_and_cache_kernel(
        cache_t* __restrict__ key_cache,
        const scalar_t* __restrict__ key,
        const int64_t* __restrict__ positions,
        const int64_t* __restrict__ slot_mapping,
        float rope_theta, int rotary_dim,
        int num_kv_heads, int head_size, int block_size) {
      
      const int token_idx = blockIdx.x;
      const int head_idx = blockIdx.y;
      const int64_t position = positions[token_idx];
      const int64_t slot = slot_mapping[token_idx];
      
      // Compute cache location
      int block_idx = slot / block_size;
      int block_offset = slot % block_size;
      
      for (int d = threadIdx.x; d < head_size; d += blockDim.x) {
        float k_val = static_cast<float>(key[token_idx * num_kv_heads * head_size 
                                            + head_idx * head_size + d]);
        
        // Apply RoPE if in rotary dimensions
        if (d < rotary_dim) {
          int pair_d = (d % 2 == 0) ? d + 1 : d - 1;
          float k_pair = static_cast<float>(key[token_idx * num_kv_heads * head_size 
                                               + head_idx * head_size + pair_d]);
          
          float freq = 1.0f / powf(rope_theta, float(d / 2 * 2) / rotary_dim);
          float angle = position * freq;
          float cos_val = cosf(angle);
          float sin_val = sinf(angle);
          
          if (d % 2 == 0) {
            k_val = k_val * cos_val - k_pair * sin_val;
          } else {
            k_val = k_pair * sin_val + k_val * cos_val;
          }
        }
        
        // Write to cache (with optional quantization)
        int cache_offset = block_idx * num_kv_heads * head_size * block_size
                         + head_idx * head_size * block_size
                         + d * block_size + block_offset;
        key_cache[cache_offset] = convert_to_cache_type(k_val);
      }
    }
    ```
- Evidence mapping:
  - "Fused operations" → RoPE and cache write in single kernel
  - "No intermediate" → Key goes directly to cache after rotation
  - "Optional quantization" → `convert_to_cache_type` for FP8 cache
