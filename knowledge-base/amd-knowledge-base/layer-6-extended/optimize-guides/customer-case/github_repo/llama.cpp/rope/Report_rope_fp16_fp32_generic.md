# Kernel: Rotary Position Embedding (RoPE)

## Variant Context
- Input semantic type: Positional encoding for attention
- Datatype(s): FP16, FP32
- Data representation: Dense tensors (Q, K projections)
- Target architecture: Generic (NVIDIA, AMD, Moore Threads)

## Functionality
The RoPE kernel applies rotary position embeddings to query and key tensors in transformer attention. RoPE encodes position information by rotating pairs of dimensions using sinusoidal functions, enabling the model to learn relative position relationships.

Key features:
- Standard RoPE for LLaMA, Mistral, etc.
- Multi-modal RoPE (M-RoPE) for Qwen2-VL
- Vision RoPE for 2D positional encoding
- Partial rotation support (some dimensions unrotated)
- Neox-style and GPT-J style rotation patterns

---

## Optimization 1: Fused RoPE with Set Rows
- Commit ID: a90eb94ca
- Optimization type: Fusion (kernel fusion)
- Summary: Fuse RoPE application with set_rows operation to reduce memory traffic in KV cache updates
- Detailed explanation: When updating the KV cache, the rotated K values need to be written to specific rows. By fusing the RoPE computation with the set_rows operation, we avoid writing the rotated values to an intermediate buffer and then copying them to the cache.

- Code excerpt:
    ```cpp
    // CUDA: fuse rope + set_rows
    template<typename T, bool has_ff>
    __global__ void rope_norm_set_rows(
        const T * __restrict__ src,
        T * __restrict__ dst,
        const int32_t * __restrict__ pos,
        const float * __restrict__ freq_factors,
        const int64_t * __restrict__ dst_rows,  // Target rows in KV cache
        const int ne0,
        const int ne1,
        const float theta_scale,
        const float freq_scale,
        ...) {
        
        const int i0 = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
        const int row = blockIdx.y;
        const int dst_row = dst_rows[row];  // Fused: get target row
        
        // Compute RoPE rotation
        const float theta = pos[row] * powf(theta_scale, i0 / 2) * freq_scale;
        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);
        
        // Read pair of values
        const float x0 = (float)src[row * ne0 + i0];
        const float x1 = (float)src[row * ne0 + i0 + 1];
        
        // Rotate and write directly to destination row (fused set_rows)
        dst[dst_row * ne0 + i0]     = (T)(x0 * cos_theta - x1 * sin_theta);
        dst[dst_row * ne0 + i0 + 1] = (T)(x0 * sin_theta + x1 * cos_theta);
    }
    ```

- Evidence mapping:
  - "Fused operations" → RoPE + set_rows in single kernel
  - "Direct write to cache" → `dst[dst_row * ne0 + ...]`
  - "Reduced memory traffic" → no intermediate buffer

---

## Optimization 2: Multi-modal RoPE (M-RoPE) for Vision-Language Models
- Commit ID: ba1cb19cd
- Optimization type: Algorithm (new functionality)
- Summary: Add support for multi-modal RoPE used in Qwen2-VL for handling both text and image positions
- Detailed explanation: Vision-language models like Qwen2-VL use different positional encodings for text tokens and image patches. M-RoPE applies different rotation frequencies to different sections of the embedding dimension, allowing the model to encode temporal (text) and spatial (image) positions simultaneously.

- Code excerpt:
    ```cpp
    // Multi-modal RoPE for Qwen2-VL
    template<typename T>
    __global__ void rope_mrope(
        const T * __restrict__ src,
        T * __restrict__ dst,
        const int32_t * __restrict__ pos,  // [3, seq_len]: temporal, height, width
        const int * __restrict__ sections, // Dimension sections for each modality
        const int ne0,
        const float theta_base,
        ...) {
        
        const int i0 = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
        const int row = blockIdx.y;
        
        // Determine which section this dimension belongs to
        int section = 0;
        int section_start = 0;
        for (int s = 0; s < 3; s++) {
            if (i0 >= section_start && i0 < section_start + sections[s]) {
                section = s;
                break;
            }
            section_start += sections[s];
        }
        
        // Use position for this section (temporal, height, or width)
        const int position = pos[section * seq_len + row];
        const float theta = position * powf(theta_base, (i0 - section_start) / 2);
        
        // Apply rotation
        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);
        ...
    }
    ```

- Evidence mapping:
  - "Multi-modal positions" → `pos[3, seq_len]` for temporal, height, width
  - "Dimension sections" → `sections[]` array for modality boundaries
  - "Section-specific rotation" → different theta per section

---

## Optimization 3: Vision RoPE for 2D Positional Encoding
- Commit ID: ba1cb19cd
- Optimization type: Algorithm (2D positions)
- Summary: Add 2D rotary position encoding for vision transformer patches
- Detailed explanation: Vision transformers process image patches that have 2D spatial positions. Vision RoPE encodes both row and column positions by splitting the embedding dimensions and applying separate rotations for each spatial dimension.

- Code excerpt:
    ```cpp
    // Vision RoPE for 2D positions
    template<typename T>
    __global__ void rope_vision(
        const T * __restrict__ src,
        T * __restrict__ dst,
        const int32_t * __restrict__ pos,  // [2, seq_len]: row, col positions
        const int ne0,
        const float theta_base,
        ...) {
        
        const int i0 = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
        const int row = blockIdx.y;
        
        // First half of dimensions: encode row position
        // Second half: encode column position
        const int half_dim = ne0 / 2;
        const int spatial_dim = i0 < half_dim ? 0 : 1;  // 0=row, 1=col
        const int local_i = i0 < half_dim ? i0 : i0 - half_dim;
        
        const int position = pos[spatial_dim * seq_len + row];
        const float theta = position * powf(theta_base, local_i / 2);
        
        // Apply rotation
        ...
    }
    ```

- Evidence mapping:
  - "2D positions" → `pos[2, seq_len]` for row and column
  - "Split dimensions" → first half for rows, second for columns
  - "Spatial encoding" → `spatial_dim` selection based on dimension index

---

## Optimization 4: Coalesced Memory Writes
- Commit ID: a90eb94ca
- Optimization type: Memory (coalescing)
- Summary: Ensure coalesced writes to global memory in RoPE kernel
- Detailed explanation: GPU memory performance is maximized when adjacent threads write to adjacent memory locations. This optimization reorganizes the thread-to-data mapping to ensure coalesced writes, improving memory bandwidth utilization.

- Code excerpt:
    ```cpp
    // rope_norm: coalesced writes to global mem
    template<typename T>
    __global__ void rope_norm_coalesced(
        const T * __restrict__ src,
        T * __restrict__ dst,
        const int32_t * __restrict__ pos,
        const int ne0,
        ...) {
        
        // Thread mapping for coalesced access
        // Each thread handles one pair of adjacent elements
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int pair_idx = tid;  // Pair index
        const int row = pair_idx / (ne0 / 2);
        const int i0 = 2 * (pair_idx % (ne0 / 2));
        
        // Adjacent threads write to adjacent memory locations
        // Thread 0: dst[0], dst[1]
        // Thread 1: dst[2], dst[3]
        // ...
        dst[row * ne0 + i0]     = rotated_x0;
        dst[row * ne0 + i0 + 1] = rotated_x1;
    }
    ```

- Evidence mapping:
  - "Coalesced writes" → adjacent threads write adjacent pairs
  - "Thread mapping" → `pair_idx` ensures sequential access
  - "Memory efficiency" → maximizes memory bandwidth

---

## Optimization 5: Partial Rotation Support
- Commit ID: 4d0dcd4a0
- Optimization type: Algorithm (flexibility)
- Summary: Support partial rotation where only some dimensions are rotated
- Detailed explanation: Some models only apply RoPE to a subset of dimensions (e.g., first 64 of 128). This optimization handles partial rotation correctly, leaving unrotated dimensions unchanged while applying rotation to the specified range.

- Code excerpt:
    ```cpp
    // cuda: fix rope with partial rotation and non-cont src
    template<typename T>
    __global__ void rope_partial(
        const T * __restrict__ src,
        T * __restrict__ dst,
        const int32_t * __restrict__ pos,
        const int ne0,
        const int n_dims,  // Number of dimensions to rotate
        ...) {
        
        const int i0 = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
        const int row = blockIdx.y;
        
        if (i0 < n_dims) {
            // Apply rotation to first n_dims dimensions
            const float theta = pos[row] * powf(theta_base, i0 / 2);
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);
            
            const float x0 = (float)src[row * ne0 + i0];
            const float x1 = (float)src[row * ne0 + i0 + 1];
            
            dst[row * ne0 + i0]     = (T)(x0 * cos_theta - x1 * sin_theta);
            dst[row * ne0 + i0 + 1] = (T)(x0 * sin_theta + x1 * cos_theta);
        } else {
            // Copy remaining dimensions unchanged
            dst[row * ne0 + i0]     = src[row * ne0 + i0];
            dst[row * ne0 + i0 + 1] = src[row * ne0 + i0 + 1];
        }
    }
    ```

- Evidence mapping:
  - "Partial rotation" → `n_dims` parameter for rotation range
  - "Conditional rotation" → `if (i0 < n_dims)` check
  - "Unchanged dimensions" → direct copy for `i0 >= n_dims`
