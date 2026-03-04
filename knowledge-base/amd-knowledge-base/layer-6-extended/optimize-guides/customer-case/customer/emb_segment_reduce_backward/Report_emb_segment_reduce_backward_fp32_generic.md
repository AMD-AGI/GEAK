---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: segment_reduce_backward

## Variant Context
- Input semantic type: Embedding segment reduction (backward pass)
- Datatype(s): fp32
- Data representation: Sparse embedding gradients with segment offsets
- Target architecture: Generic HIP/AMD GPU

## Functionality
This kernel computes gradients for the segment reduction operation. It distributes output gradients back to input embeddings based on segment membership, using atomic operations for accumulation when multiple outputs map to the same input.

## Optimization 1: Vectorized Gradient Accumulation
- Commit ID: baseline → optimized
- Optimization type: Memory
- Summary: Use vector types for gradient load/store operations
- Detailed explanation: Similar to the forward pass, the backward kernel uses the Packer template for vectorized memory access, loading and storing gradients in float4 or float2 chunks for better memory bandwidth.

- Code excerpt (optimized):
    ```cpp
    using AP = Packer<scalar_t, PACK_SIZE>;
    typename AP::type grad_vec;
    AP::load(grad_output + offset, grad_vec);
    ```

- Evidence mapping:
  - "Vectorized gradient access" → Using Packer template for gradient tensors

## Optimization 2: Atomic Add with Custom Template
- Commit ID: baseline → optimized
- Optimization type: Compute
- Summary: Use templated atomic add for type-safe gradient accumulation
- Detailed explanation: The kernel uses a templated atomic_add_custom function that maps to the appropriate atomicAdd intrinsic for each data type, ensuring correct accumulation semantics.

- Code excerpt (optimized):
    ```cpp
    template <typename T>
    __device__ __forceinline__ void atomic_add_custom(T* address, const T val) {
      atomicAdd(address, val);
    }
    ```

- Evidence mapping:
  - "Templated atomic" → `atomic_add_custom<T>(address, val)`
