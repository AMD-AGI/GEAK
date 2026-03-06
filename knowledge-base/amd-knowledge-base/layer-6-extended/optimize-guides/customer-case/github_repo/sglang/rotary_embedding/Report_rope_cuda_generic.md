Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Rotary Position Embedding (RoPE) Kernel

## Variant Context
- Input semantic type: Position encoding for attention queries and keys
- Datatype(s): FP16, BF16, FP32
- Data representation: Q/K tensors with position IDs and cos/sin cache
- Target architecture: Generic CUDA

## Functionality
This kernel applies Rotary Position Embedding (RoPE) to query and key tensors. RoPE encodes position information by rotating pairs of dimensions based on their position in the sequence, enabling the model to understand relative positions between tokens.

## Optimization 1: Fused RoPE with KV Cache Write
- Commit ID: 9aea25552
- Optimization type: Fusion
- Summary: Fuses the RoPE application with writing K and V to the KV cache, eliminating a separate memory copy operation.

- Detailed explanation:
  Previously, RoPE and KV cache writing were separate operations:
  1. Apply RoPE to Q and K
  2. Write K to k_buffer at kv_cache_loc
  3. Write V to v_buffer at kv_cache_loc
  
  This optimization fuses all three operations into a single kernel, reducing memory traffic and kernel launch overhead.

- Code excerpt:
    ```cpp
    void apply_rope_pos_ids_cos_sin_cache(
        at::Tensor q,
        at::Tensor k,
        at::Tensor q_rope,
        at::Tensor k_rope,
        at::Tensor cos_sin_cache,
        at::Tensor pos_ids,
        bool interleave,
        int64_t cuda_stream,
        // New optional parameters for fused KV cache write
        const std::optional<at::Tensor>& v,
        const std::optional<at::Tensor>& k_buffer,
        const std::optional<at::Tensor>& v_buffer,
        const std::optional<at::Tensor>& kv_cache_loc
    ) {
        const bool save_kv_cache = v.has_value();
        
        if (save_kv_cache) {
            // Fused kernel: RoPE + KV cache write
            cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCacheEnhanced(
                q_ptr, k_ptr, v_ptr,
                q_rope_ptr, k_rope_ptr,
                k_buffer_ptr, v_buffer_ptr,
                cos_sin_cache_ptr, pos_ids_ptr,
                // ... strides and dimensions ...
                kv_cache_loc_ptr,
                interleave,
                save_kv_cache,
                stream
            );
        } else {
            // Original kernel: RoPE only
            cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCache(...);
        }
    }
    ```

- Evidence mapping:
  - "Fused operation" → `BatchQKApplyRotaryPosIdsCosSinCacheEnhanced` handles both RoPE and KV write
  - "Optional KV cache" → `const std::optional<at::Tensor>& v` and related parameters
  - "Conditional path" → `if (save_kv_cache)` selects fused vs original kernel

## Optimization 2: PDL (Programmatic Dependent Launch) Support
- Commit ID: 42c870456
- Optimization type: Scheduling / Latency
- Summary: Adds PDL support for overlapping RoPE kernel launch with previous operations.

- Detailed explanation:
  PDL (Programmatic Dependent Launch) allows kernels to be launched before their dependencies complete, with the GPU handling synchronization. This optimization enables the RoPE kernel to be launched earlier in the pipeline, hiding launch latency.

- Code excerpt:
    ```cpp
    // PDL-enabled kernel launch
    template <typename T>
    cudaError_t BatchQKApplyRotaryPosIdsCosSinCache_PDL(
        // ... parameters ...
        cudaStream_t stream,
        bool use_pdl
    ) {
        if (use_pdl) {
            // Launch with PDL attributes
            cudaLaunchAttribute attrs[1];
            attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attrs[0].val.programmaticStreamSerializationAllowed = 1;
            
            cudaLaunchConfig_t config = {0};
            config.attrs = attrs;
            config.numAttrs = 1;
            
            cudaLaunchKernelEx(&config, kernel, ...);
        } else {
            kernel<<<grid, block, 0, stream>>>(...);
        }
    }
    ```

- Evidence mapping:
  - "PDL support" → `cudaLaunchAttributeProgrammaticStreamSerialization`
  - "Conditional launch" → `if (use_pdl)` selects launch method
  - "Kernel config" → `cudaLaunchKernelEx` for PDL-enabled launch

## Optimization 3: Reduced Launch Overhead via C++ Stream Handling
- Commit ID: 20315697f
- Optimization type: Latency
- Summary: Moves stream acquisition from Python to C++ to reduce kernel launch overhead.

- Detailed explanation:
  Previously, the CUDA stream was obtained in Python and passed to the kernel. This optimization moves the stream handling to C++, reducing the Python-C++ boundary crossing overhead for each kernel launch.

- Code excerpt:
    ```cpp
    // Before: stream passed from Python
    void apply_rope(..., int64_t cuda_stream) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
        // ...
    }

    // After: stream obtained in C++
    void apply_rope(...) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        // ...
    }
    ```

- Evidence mapping:
  - "C++ stream handling" → `at::cuda::getCurrentCUDAStream()`
  - "Reduced overhead" → no Python int64_t to cudaStream_t conversion

## Optimization 4: FP32 Dtype Support
- Commit ID: 20e59f951
- Optimization type: Precision
- Summary: Adds FP32 datatype support for RoPE to enable higher precision position encoding.

- Detailed explanation:
  Some models or configurations require FP32 precision for position encoding to maintain numerical accuracy. This optimization extends the RoPE kernel to support FP32 inputs and outputs alongside FP16/BF16.

- Code excerpt:
    ```cpp
    // Extended dtype dispatch
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16_FP32(q.scalar_type(), c_type, [&] {
        cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCache<c_type>(
            // ...
        );
    });
    ```

- Evidence mapping:
  - "FP32 support" → `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16_FP32` macro
  - "Template instantiation" → `BatchQKApplyRotaryPosIdsCosSinCache<c_type>`

## Optimization 5: Optimized Memory Access Patterns
- Commit ID: cf0ccd406
- Optimization type: Memory
- Summary: Optimizes memory access patterns for better coalescing and cache utilization.

- Detailed explanation:
  The RoPE kernel accesses Q and K tensors with specific stride patterns. This optimization ensures:
  1. Coalesced memory access for the head dimension
  2. Efficient use of L2 cache for cos/sin values
  3. Vectorized loads where possible

- Code excerpt:
    ```cpp
    // Optimized memory access
    template <typename T, int VEC_SIZE>
    __global__ void rope_kernel_optimized(
        T* __restrict__ q,
        T* __restrict__ k,
        const float* __restrict__ cos_sin_cache,
        // ...
    ) {
        // Vectorized load
        using VecT = typename VecType<T, VEC_SIZE>::Type;
        VecT q_vec = *reinterpret_cast<VecT*>(q + offset);
        
        // Apply RoPE with cached cos/sin
        __shared__ float shared_cos_sin[ROTARY_DIM];
        if (threadIdx.x < ROTARY_DIM) {
            shared_cos_sin[threadIdx.x] = cos_sin_cache[pos * ROTARY_DIM + threadIdx.x];
        }
        __syncthreads();
        
        // Rotate using shared memory cos/sin
        // ...
    }
    ```

- Evidence mapping:
  - "Vectorized access" → `VecType<T, VEC_SIZE>::Type`
  - "Shared memory cache" → `__shared__ float shared_cos_sin[ROTARY_DIM]`
  - "Coalesced loads" → sequential thread access to consecutive elements
