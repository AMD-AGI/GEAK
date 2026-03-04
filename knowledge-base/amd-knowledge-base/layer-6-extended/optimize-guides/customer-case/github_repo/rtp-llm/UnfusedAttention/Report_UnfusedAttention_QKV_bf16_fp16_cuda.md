---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Unfused Attention QKV Bias Transpose with RoPE

## Variant Context
- Input semantic type: Attention (QKV processing for prefill phase)
- Datatype(s): bf16, fp16, fp32
- Data representation: Fused QKV tensor with optional bias and RoPE
- Target architecture: CUDA (SM80+, optimized for SM90)

## Functionality
This kernel performs the following operations in a fused manner:
1. Adds bias to the fused QKV tensor
2. Applies Rotary Position Embedding (RoPE) to Q and K
3. Transposes the output from [batch, seq, head, dim] to [batch, head, seq, dim]
4. Optionally writes to KV cache
5. Optionally quantizes output to FP8

The kernel is critical for the prefill phase of LLM inference where the entire input sequence is processed.

## Optimization 1: Increased Vector Size for Memory Access (4 → 8 elements)
- Commit ID: b722054e5
- Optimization type: Memory
- Summary: Doubled the vector size from 4 to 8 elements for more efficient memory transactions
- Detailed explanation:
  The optimization introduces a new `Vec_t2` type that uses 8-element vectors instead of 4-element vectors:
  - For fp16: Uses `uint4` (8 x fp16 = 128 bits)
  - For bf16: Uses `bf16_8_t` (8 x bf16 = 128 bits)
  - For fp32: Uses `Float8_` (8 x fp32 = 256 bits)
  
  This doubles the data loaded per memory transaction, improving memory bandwidth utilization and reducing the number of load/store instructions.

- Code excerpt:
    ```cpp
    // New Vec_t2 with 8-element vectors
    template<>
    struct Vec_t2<half> {
        using Type                = uint4;
        static constexpr int size = 8;
    #ifdef ENABLE_FP8
        using QuantizedType = fp8_8_t;
    #endif
    };
    
    template<>
    struct Vec_t2<__nv_bfloat16> {
        using Type                = bf16_8_t;
        static constexpr int size = 8;
    #ifdef ENABLE_FP8
        using QuantizedType = fp8_8_t;
    #endif
    };
    
    // Usage in kernel
    constexpr int vec_size = Vec_t2<T>::size;  // Now 8 instead of 4
    using vec_t2           = typename Vec_t2<T>::Type;
    ```
- Evidence mapping:
  - Vector size increase → `static constexpr int size = 8` in Vec_t2 vs `size = 4` in Vec_t
  - Wider load operations → `*reinterpret_cast<int4*>(&q[q_load_idx]) = *reinterpret_cast<int4*>(&QKV[src_q_idx])`

## Optimization 2: Vectorized RoPE Coefficient Loading
- Commit ID: b722054e5
- Optimization type: Memory
- Summary: Load 8 RoPE coefficients at once using float4 loads instead of float2
- Detailed explanation:
  The RoPE coefficients (cos/sin values) are now loaded using two float4 loads instead of one float2 load:
  - Before: Load 2 floats (cos, sin for one position)
  - After: Load 8 floats (cos, sin for 4 positions)
  
  This reduces the number of memory transactions for RoPE coefficient loading by 4x.

- Code excerpt:
    ```cpp
    // Before: float2 load
    // coef = *(reinterpret_cast<float2*>(
    //     const_cast<float*>(&rope_cache[position_id * rope_config.dim + lane_id * 2])));
    
    // After: Two float4 loads for 8 coefficients
    Float8_ coef;
    *reinterpret_cast<float4*>(&coef.x) =
        *(reinterpret_cast<const float4*>(&rope_cache[position_id * rope_config.dim + lane_id * 8]));
    *reinterpret_cast<float4*>(&coef.z) =
        *(reinterpret_cast<const float4*>(&rope_cache[position_id * rope_config.dim + lane_id * 8 + 4]));
    ```
- Evidence mapping:
  - Wider coefficient type → `Float8_ coef` instead of `float2 coef`
  - Vectorized loads → Two `float4` loads covering 8 coefficients

## Optimization 3: Shared Memory Alignment for Larger Vectors
- Commit ID: b722054e5
- Optimization type: Memory
- Summary: Aligned shared memory to Float8_ (256 bits) for optimal access
- Detailed explanation:
  The shared memory declaration is now aligned to `Float8_` (256 bits) instead of `float2` (64 bits). This ensures:
  - Optimal memory bank access patterns
  - Reduced bank conflicts for wider vector operations
  - Better coalescing for the 8-element vector operations

- Code excerpt:
    ```cpp
    // Before
    // extern __shared__ __align__(sizeof(float2)) char smem_[];
    
    // After
    extern __shared__ __align__(sizeof(Float8_)) char smem_[];
    ```
- Evidence mapping:
  - Alignment change → `__align__(sizeof(Float8_))` instead of `__align__(sizeof(float2))`

## Optimization 4: Multi-Head Block Processing
- Commit ID: 7d8dfb711
- Optimization type: Compute / Parallelism
- Summary: Process multiple heads per thread block using template parameters
- Detailed explanation:
  The kernel uses template parameters `HEAD_Q_BLOCK_NUM`, `HEAD_K_BLOCK_NUM`, and `HEAD_V_BLOCK_NUM` to process multiple heads per thread block. This:
  - Reduces kernel launch overhead
  - Improves data reuse within a thread block
  - Enables better instruction-level parallelism through loop unrolling

- Code excerpt:
    ```cpp
    template<typename T,
             typename Tcache,
             bool      PREFIX_PROMPT,
             bool      USE_PAGED_FMHA,
             RopeStyle ROPE_STYLE,
             int       HEAD_Q_BLOCK_NUM,  // Number of Q heads per block
             int       HEAD_K_BLOCK_NUM,  // Number of K heads per block
             int       HEAD_V_BLOCK_NUM>  // Number of V heads per block
    __global__ void add_fusedQKV_bias_transpose_non_int8_with_rope_cache_kernel(...) {
        // ...
        const int max_q_bidy = head_num / HEAD_Q_BLOCK_NUM;
        const int max_k_bidy = max_q_bidy + head_num_kv / HEAD_K_BLOCK_NUM;
        const int max_v_bidy = max_k_bidy + head_num_kv / HEAD_V_BLOCK_NUM;
        
        // Process multiple heads with unrolled loop
        #pragma unroll
        for (int h = 1; h < HEAD_Q_BLOCK_NUM; ++h) {
            // Process head h
        }
    }
    ```
- Evidence mapping:
  - Template parameters → `HEAD_Q_BLOCK_NUM`, `HEAD_K_BLOCK_NUM`, `HEAD_V_BLOCK_NUM`
  - Unrolled processing → `#pragma unroll` with loop over heads

## Optimization 5: Double Buffering for Q/K/V Processing
- Commit ID: 7d8dfb711
- Optimization type: Compute / Memory
- Summary: Use double buffering to overlap memory loads with computation
- Detailed explanation:
  The kernel uses a double-buffering scheme where:
  - While processing the current head's data (applying RoPE, bias), the next head's data is being loaded
  - This hides memory latency behind computation
  - Implemented using alternating buffer indices (`q_load_idx ^= q_idx_off`)

- Code excerpt:
    ```cpp
    vec_t2 q[2];  // Double buffer
    int    q_load_idx  = 0;
    int    q_store_idx = 0;
    int    q_idx_off   = 1;
    
    // Initial load
    q[q_load_idx] = *reinterpret_cast<const vec_t2*>(&QKV[src_q_idx]);
    
    #pragma unroll
    for (int h = 1; h < HEAD_Q_BLOCK_NUM; ++h) {
        q_load_idx ^= q_idx_off;  // Switch to other buffer
        
        // Load next head's data
        q[q_load_idx] = *reinterpret_cast<const vec_t2*>(&QKV[src_q_idx]);
        
        // Process previous head's data (overlapped with load)
        apply_rope_with_cache<vec_t2, T, Float8_, ROPE_STYLE>(
            q[q_store_idx], ...);
        
        // Store processed data
        *reinterpret_cast<vec_t2*>(&q_buf[dest_q_idx]) = q[q_store_idx];
        
        q_store_idx ^= q_idx_off;  // Switch store buffer
    }
    ```
- Evidence mapping:
  - Double buffer declaration → `vec_t2 q[2]`
  - Buffer switching → `q_load_idx ^= q_idx_off` and `q_store_idx ^= q_idx_off`
  - Overlapped operations → Load in current iteration, process previous iteration's data

## Optimization 6: Conditional Store Flags for Flexibility
- Commit ID: 7d8dfb711
- Optimization type: Compute / Flexibility
- Summary: Added boolean flags to control which outputs are stored
- Detailed explanation:
  The kernel accepts boolean flags to control which outputs are written:
  - `store_qkv`: Write back to original QKV buffer
  - `store_q_no_transpose`: Write Q without transpose
  - `store_q`: Write transposed Q
  - `store_kv`: Write K and V
  - `store_cache`: Write to KV cache
  
  This avoids unnecessary memory writes when certain outputs are not needed.

- Code excerpt:
    ```cpp
    __global__ void add_fusedQKV_bias_transpose_non_int8_with_rope_cache_kernel(
        // ... other params ...
        bool store_qkv,
        bool store_q_no_transpose,
        bool store_q,
        bool store_kv,
        bool store_cache) {
        // ...
        if (store_qkv) {
            *reinterpret_cast<vec_t2*>(&QKV[src_q_idx]) = q[q_store_idx];
        }
        if (store_q_no_transpose) {
            *reinterpret_cast<vec_t2*>(&q_no_transpose_buf[dest_q_no_transpose_idx]) = q[q_store_idx];
        }
        if (store_q) {
            *reinterpret_cast<vec_t2*>(&q_buf[dest_q_idx]) = q[q_store_idx];
        }
        // ...
    }
    ```
- Evidence mapping:
  - Boolean flags → `store_qkv`, `store_q_no_transpose`, `store_q`, `store_kv`, `store_cache`
  - Conditional writes → `if (store_qkv) { ... }`
