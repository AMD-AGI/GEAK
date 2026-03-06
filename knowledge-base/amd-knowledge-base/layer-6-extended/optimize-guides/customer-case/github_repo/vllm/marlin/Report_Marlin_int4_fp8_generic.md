Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Marlin Quantized GEMM

## Variant Context
- Input semantic type: Matrix multiplication with quantized weights
- Datatype(s): INT4 (weights), FP8 (weights), FP16/BF16 (activations)
- Data representation: Packed INT4/FP8 weights with group-wise scaling
- Target architecture: CUDA sm80+ (Ampere and later)

## Functionality
Marlin is a highly optimized GEMM kernel for quantized weight inference:
1. Supports INT4 and FP8 weight quantization
2. Uses specialized memory layouts for efficient Tensor Core utilization
3. Supports GPTQ and AWQ quantization formats
4. Provides near-FP16 performance with 4x memory reduction

## Optimization 1: Optimized Weight Layout for Tensor Cores
- Commit ID: (marlin.cu initial implementation)
- Optimization type: Memory / Compute
- Summary: Use specialized weight packing layout optimized for Tensor Core access patterns
- Detailed explanation:
  Marlin uses a custom weight layout that aligns with Tensor Core's matrix fragment requirements. Weights are pre-packed during model loading to enable efficient warp-level matrix operations without runtime unpacking overhead.
- Code excerpt:
    ```cpp
    // Marlin weight layout for INT4
    // Weights are packed as: [K/16, N/8, 16, 8] with INT4 values
    // This matches Tensor Core's 16x8 fragment size
    
    template <int THREADS, int STAGES>
    __global__ void marlin_gemm_kernel(
        const int4* __restrict__ A,      // [M, K] in FP16
        const int4* __restrict__ B,      // Packed INT4 weights
        const half* __restrict__ scales, // [K/group_size, N]
        int4* __restrict__ C,            // [M, N] output
        int M, int N, int K,
        int group_size) {
      
      // Each warp processes a 16x8 output tile
      // Weights are pre-arranged for direct Tensor Core consumption
      
      // Load weight fragment (already in correct layout)
      int4 weight_packed = B[weight_offset];
      
      // Unpack INT4 to FP16 for Tensor Core
      half2 weights[8];
      unpack_int4_to_half2(weight_packed, weights);
      
      // Apply group scale
      half scale = scales[group_idx * N + n_idx];
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        weights[i] = __hmul2(weights[i], __half2half2(scale));
      }
      
      // Tensor Core WMMA operation
      wmma::fragment<wmma::matrix_b, 16, 8, 16, half, wmma::row_major> b_frag;
      wmma::load_matrix_sync(b_frag, weights, 8);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    ```
- Evidence mapping:
  - "Optimized layout" → `[K/16, N/8, 16, 8]` matches Tensor Core fragments
  - "Pre-packed weights" → No runtime layout transformation
  - "WMMA operations" → Direct Tensor Core usage with `wmma::mma_sync`

## Optimization 2: Asynchronous Memory Pipeline
- Commit ID: (marlin.cu)
- Optimization type: Memory
- Summary: Use async memory copies with multi-stage software pipeline
- Detailed explanation:
  Marlin uses CUDA's async copy instructions (cp.async) with a multi-stage pipeline to hide memory latency. While one stage is computing, the next stage's data is being loaded.
- Code excerpt:
    ```cpp
    // Multi-stage async pipeline
    template <int STAGES>
    __device__ void marlin_mainloop(
        const int4* A, const int4* B, const half* scales,
        int4* C, int K) {
      
      // Shared memory for pipeline stages
      __shared__ half a_shared[STAGES][TILE_M][TILE_K];
      __shared__ int4 b_shared[STAGES][TILE_K/8][TILE_N/8];
      
      // Initialize pipeline
      #pragma unroll
      for (int s = 0; s < STAGES - 1; s++) {
        // Async load stage s
        cp_async_copy(a_shared[s], A + s * TILE_K);
        cp_async_copy(b_shared[s], B + s * TILE_K * TILE_N / 32);
        cp_async_commit();
      }
      
      int stage = 0;
      for (int k = 0; k < K; k += TILE_K) {
        // Wait for current stage
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        
        // Start loading next stage
        int next_stage = (stage + STAGES - 1) % STAGES;
        if (k + (STAGES - 1) * TILE_K < K) {
          cp_async_copy(a_shared[next_stage], A + (k + (STAGES-1) * TILE_K));
          cp_async_copy(b_shared[next_stage], B + ...);
          cp_async_commit();
        }
        
        // Compute current stage
        compute_tile(a_shared[stage], b_shared[stage], scales, c_frag);
        
        stage = (stage + 1) % STAGES;
      }
    }
    ```
- Evidence mapping:
  - "Async copy" → `cp_async_copy` for non-blocking loads
  - "Multi-stage" → `STAGES` buffers for pipeline depth
  - "Overlap" → Next stage loads while current computes

## Optimization 3: Efficient INT4 Unpacking
- Commit ID: (marlin.cu)
- Optimization type: Compute
- Summary: Optimized INT4 to FP16 unpacking using bit manipulation
- Detailed explanation:
  INT4 values are packed 8 per 32-bit word. Marlin uses efficient bit manipulation to unpack and convert to FP16 with minimal instructions.
- Code excerpt:
    ```cpp
    // Efficient INT4 unpacking
    __device__ __forceinline__ void unpack_int4_to_half2(
        int4 packed, half2* output) {
      
      // Each int4 contains 32 INT4 values (128 bits / 4 bits)
      uint32_t* vals = reinterpret_cast<uint32_t*>(&packed);
      
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        uint32_t v = vals[i];
        
        // Extract pairs of INT4 values
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          // Extract two 4-bit values
          int lo = (v >> (j * 8)) & 0xF;
          int hi = (v >> (j * 8 + 4)) & 0xF;
          
          // Convert to signed (-8 to 7 range for symmetric quant)
          lo = lo - 8;
          hi = hi - 8;
          
          // Convert to half2
          output[i * 4 + j] = __halves2half2(
              __int2half_rn(lo), __int2half_rn(hi));
        }
      }
    }
    
    // Alternative: use LUT for faster conversion
    __device__ __forceinline__ half2 int4_to_half2_lut(uint8_t packed) {
      // Pre-computed lookup table in shared memory
      extern __shared__ half2 lut[256];
      return lut[packed];
    }
    ```
- Evidence mapping:
  - "Bit extraction" → `(v >> offset) & 0xF` for INT4 extraction
  - "Symmetric quant" → `lo - 8` for signed range
  - "LUT option" → Lookup table for even faster conversion

## Optimization 4: GPTQ/AWQ Format Support
- Commit ID: (gptq_marlin_repack.cu, awq_marlin_repack.cu)
- Optimization type: Compatibility
- Summary: Add repacking kernels to convert GPTQ/AWQ formats to Marlin layout
- Detailed explanation:
  GPTQ and AWQ use different weight packing formats. These kernels convert weights to Marlin's optimized layout during model loading, enabling efficient inference with pre-quantized models.
- Code excerpt:
    ```cpp
    // Repack GPTQ weights to Marlin format
    __global__ void gptq_marlin_repack_kernel(
        const int32_t* __restrict__ gptq_weights,  // GPTQ packed format
        const half* __restrict__ gptq_scales,
        const int32_t* __restrict__ gptq_zeros,    // Zero points
        int4* __restrict__ marlin_weights,         // Output Marlin format
        half* __restrict__ marlin_scales,
        int K, int N, int group_size) {
      
      // GPTQ format: weights packed as [K, N/8] with 8 INT4 per int32
      // Marlin format: [K/16, N/8, 16, 8] for Tensor Core alignment
      
      const int k_block = blockIdx.x;
      const int n_block = blockIdx.y;
      
      // Load GPTQ weights for this tile
      int32_t gptq_tile[16][8];
      for (int k = 0; k < 16; k++) {
        for (int n = 0; n < 8; n++) {
          int gptq_k = k_block * 16 + k;
          int gptq_n = n_block * 8 + n;
          gptq_tile[k][n] = gptq_weights[gptq_k * (N/8) + gptq_n];
        }
      }
      
      // Repack to Marlin layout
      int4 marlin_tile;
      repack_tile(gptq_tile, &marlin_tile);
      
      // Apply zero point adjustment if needed
      if (gptq_zeros != nullptr) {
        apply_zero_points(marlin_tile, gptq_zeros, k_block, n_block, group_size);
      }
      
      // Store in Marlin format
      marlin_weights[k_block * (N/8) + n_block] = marlin_tile;
    }
    ```
- Evidence mapping:
  - "Format conversion" → GPTQ `[K, N/8]` to Marlin `[K/16, N/8, 16, 8]`
  - "Zero point handling" → `apply_zero_points` for asymmetric quantization
  - "One-time cost" → Repacking done during model loading

## Optimization 5: FP8 Weight Support
- Commit ID: (marlin_int4_fp8_preprocess.cu)
- Optimization type: Precision
- Summary: Add FP8 weight support to Marlin kernel
- Detailed explanation:
  FP8 provides better accuracy than INT4 while still offering significant memory savings. This optimization adds FP8 weight support with the same optimized memory layout.
- Code excerpt:
    ```cpp
    // Marlin with FP8 weights
    template <typename WeightType>  // int4 or fp8_e4m3
    __global__ void marlin_gemm_kernel(
        const half* __restrict__ A,
        const WeightType* __restrict__ B,
        const half* __restrict__ scales,
        half* __restrict__ C,
        int M, int N, int K) {
      
      // Load weights based on type
      if constexpr (std::is_same_v<WeightType, __nv_fp8_e4m3>) {
        // FP8 weights: direct conversion to FP16
        __nv_fp8_e4m3 w_fp8 = B[weight_idx];
        half w_fp16 = __nv_cvt_fp8_to_half(w_fp8);
        // Apply scale
        w_fp16 = __hmul(w_fp16, scale);
      } else {
        // INT4 weights: unpack and convert
        int4 w_packed = B[weight_idx];
        half2 w_fp16[8];
        unpack_int4_to_half2(w_packed, w_fp16);
        // Apply scale to each element
      }
      
      // Rest of GEMM computation...
    }
    ```
- Evidence mapping:
  - "FP8 support" → Template parameter for `WeightType`
  - "Direct conversion" → `__nv_cvt_fp8_to_half` for FP8
  - "Same layout" → Reuses Marlin's optimized memory access pattern

## Optimization 6: Fused Marlin MoE
- Commit ID: fc911880c, 8678a69ab
- Optimization type: Fusion
- Summary: Integrate Marlin kernel with fused MoE for quantized expert computation
- Detailed explanation:
  This optimization combines Marlin's efficient quantized GEMM with the fused MoE kernel, enabling efficient inference for quantized MoE models like Mixtral with GPTQ/AWQ weights.
- Code excerpt:
    ```cpp
    // Fused Marlin MoE kernel
    template <int EXPERTS, int TOP_K>
    __global__ void marlin_moe_kernel(
        const half* __restrict__ A,           // [num_tokens, K]
        const int4* __restrict__ expert_weights,  // [EXPERTS, K, N] packed
        const half* __restrict__ expert_scales,
        const int* __restrict__ sorted_token_ids,
        const int* __restrict__ expert_ids,
        const half* __restrict__ topk_weights,
        half* __restrict__ C,
        int num_tokens, int K, int N) {
      
      // Get expert assignment for this block
      int expert_id = expert_ids[blockIdx.x];
      
      // Get weight pointer for this expert
      const int4* weights = expert_weights + expert_id * (K * N / 32);
      const half* scales = expert_scales + expert_id * (K / group_size) * N;
      
      // Run Marlin GEMM for this expert's tokens
      marlin_gemm_tile(A, weights, scales, C, ...);
      
      // Apply routing weight
      half routing_weight = topk_weights[token_idx];
      C[output_idx] = __hmul(C[output_idx], routing_weight);
    }
    ```
- Evidence mapping:
  - "Expert selection" → `expert_ids[blockIdx.x]` for routing
  - "Per-expert weights" → Separate weight pointer per expert
  - "Fused routing" → Routing weight applied in same kernel
