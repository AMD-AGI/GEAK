Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Matrix Multiplication Floating-Point (MMF)

## Variant Context
- Input semantic type: Matrix multiplication (GEMM)
- Datatype(s): FP16, FP32, BF16
- Data representation: Dense floating-point tensors
- Target architecture: NVIDIA (Volta+), AMD (RDNA3+, CDNA)

## Functionality
The MMF kernel performs dense matrix multiplication for floating-point data types. It uses Tensor Core instructions when available (MMA on NVIDIA, WMMA on AMD) for high throughput. This kernel is used when weights are not quantized, such as during training or for models that don't benefit from quantization.

Key features:
- FP16, FP32, and BF16 support
- Tensor Core acceleration
- MoE (Mixture of Experts) support via mul_mat_id
- Optimized for various batch sizes

---

## Optimization 1: GEMM for Small Batch Sizes
- Commit ID: 1d72c8418
- Optimization type: Compute (Tensor Core)
- Summary: Add efficient GEMM implementation for FP32/FP16/BF16 with batch size <= 16
- Detailed explanation: For small batch sizes, cuBLAS overhead can dominate. This optimization provides a custom Tensor Core kernel that is more efficient for ne11 <= 16, avoiding cuBLAS launch overhead while still utilizing Tensor Cores.

- Code excerpt:
    ```cpp
    // CUDA: GEMM for FP32/FP16/BF16 and ne11 <= 16
    template<typename T, int TILE_M, int TILE_N, int TILE_K>
    __global__ void mmf_mma_small_batch(
        const T * __restrict__ A,
        const T * __restrict__ B,
        T * __restrict__ C,
        const int M, const int N, const int K) {
        
        // Use MMA for small batches
        using mma_a = mma_A_I16K8<T>;
        using mma_b = mma_B_J8K8<T>;
        using mma_c = mma_C_I16J8<float>;
        
        mma_c acc;
        acc.zero();
        
        for (int k = 0; k < K; k += TILE_K) {
            mma_a a_frag;
            mma_b b_frag;
            a_frag.load(A + ...);
            b_frag.load(B + ...);
            acc.mma(a_frag, b_frag);
        }
        
        acc.store(C + ...);
    }
    ```

- Evidence mapping:
  - "Small batch optimization" → `ne11 <= 16` condition
  - "MMA instructions" → `mma_A_I16K8`, `mma_B_J8K8` types
  - "Avoid cuBLAS" → custom kernel for small sizes

---

## Optimization 2: Volta Tensor Core Support
- Commit ID: 31c511a96
- Optimization type: Compute (architecture support)
- Summary: Add Volta Tensor Core support for floating-point matrix multiplication
- Detailed explanation: Volta GPUs have Tensor Cores but with a different instruction format than Turing/Ampere. This optimization adds Volta-specific MMA wrappers to enable Tensor Core acceleration on V100 and similar GPUs.

- Code excerpt:
    ```cpp
    // CUDA: Volta tensor core support for MMF
    #ifdef VOLTA_MMA_AVAILABLE
    template<typename T>
    __global__ void mmf_volta(
        const T * __restrict__ A,
        const T * __restrict__ B,
        T * __restrict__ C,
        const int M, const int N, const int K) {
        
        // Volta uses m8n8k4 MMA format
        using mma_a = mma_volta_A_I8K4<T>;
        using mma_b = mma_volta_B_J8K4<T>;
        using mma_c = mma_volta_C_I8J8<float>;
        
        // Process with Volta-specific tiles
        ...
    }
    #endif
    ```

- Evidence mapping:
  - "Volta support" → `VOLTA_MMA_AVAILABLE` macro
  - "Different format" → `m8n8k4` vs Turing's `m16n8k8`
  - "V100 acceleration" → enables Tensor Cores on Volta

---

## Optimization 3: MoE Support (mul_mat_id)
- Commit ID: a972faebe, c0bfc57af
- Optimization type: Algorithm (MoE)
- Summary: Add mul_mat_id support for MMF kernel to handle Mixture of Experts routing
- Detailed explanation: MoE models route different tokens to different expert weight matrices. This optimization adds support for batched matrix multiplication with expert IDs, allowing efficient processing of MoE layers without separate kernel launches per expert.

- Code excerpt:
    ```cpp
    // CUDA: Add mul_mat_id support for the mmf kernel
    template<typename T>
    __global__ void mmf_mul_mat_id(
        const T * __restrict__ weights,  // [n_experts, out_features, in_features]
        const T * __restrict__ input,    // [batch, in_features]
        T * __restrict__ output,         // [batch, out_features]
        const int32_t * __restrict__ expert_ids,  // [batch]
        const int n_experts,
        const int out_features,
        const int in_features) {
        
        const int batch_idx = blockIdx.z;
        const int expert_id = expert_ids[batch_idx];
        
        // Offset to correct expert weights
        const T * expert_weights = weights + expert_id * out_features * in_features;
        
        // Standard GEMM with expert-specific weights
        ...
    }
    
    // Optimized for different batch sizes
    // bs <= 64 for f16, bs <= 32 for f32
    ```

- Evidence mapping:
  - "Expert routing" → `expert_ids[batch_idx]` lookup
  - "Batched operation" → `blockIdx.z` for batch dimension
  - "Size-specific" → different thresholds for f16 vs f32

---

## Optimization 4: RDNA3/4 Tensor Core Support
- Commit ID: 028f93ef9, c33a58bce, 6bca76ff5
- Optimization type: Compute (AMD support)
- Summary: Enable WMMA-based matrix multiplication for AMD RDNA3 and RDNA4 GPUs
- Detailed explanation: RDNA3 and RDNA4 GPUs support WMMA instructions for matrix operations. This optimization enables the MMF kernel to use these instructions, providing Tensor Core-like acceleration on AMD consumer GPUs.

- Code excerpt:
    ```cpp
    // HIP: enable mmf for RDNA3
    // HIP: RDNA4 tensor core support for MMF
    #if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1200__)
    #define AMD_WMMA_MMF_AVAILABLE
    #endif
    
    #ifdef AMD_WMMA_MMF_AVAILABLE
    template<typename T>
    __global__ void mmf_wmma_rdna(
        const T * __restrict__ A,
        const T * __restrict__ B,
        T * __restrict__ C,
        const int M, const int N, const int K) {
        
        using namespace rocwmma;
        
        fragment<matrix_a, 16, 16, 16, T, row_major> a_frag;
        fragment<matrix_b, 16, 16, 16, T, col_major> b_frag;
        fragment<accumulator, 16, 16, 16, float> c_frag;
        
        fill_fragment(c_frag, 0.0f);
        
        for (int k = 0; k < K; k += 16) {
            load_matrix_sync(a_frag, A + ..., K);
            load_matrix_sync(b_frag, B + ..., N);
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        store_matrix_sync(C + ..., c_frag, N, mem_row_major);
    }
    #endif
    ```

- Evidence mapping:
  - "RDNA3/4 detection" → `__gfx1100__`, `__gfx1200__` macros
  - "WMMA instructions" → `rocwmma::mma_sync()`
  - "16x16x16 tiles" → fragment dimensions

---

## Optimization 5: CDNA MFMA Support
- Commit ID: f3dd7b8e6
- Optimization type: Compute (AMD datacenter)
- Summary: Add MMF support for AMD CDNA GPUs using MFMA instructions
- Detailed explanation: AMD CDNA GPUs (MI100, MI210, MI300) use MFMA (Matrix Fused Multiply-Add) instructions which are different from WMMA. This optimization enables high-performance matrix multiplication on AMD datacenter GPUs.

- Code excerpt:
    ```cpp
    // HIP: add mmf for CDNA
    #if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__)
    #define AMD_MFMA_MMF_AVAILABLE
    #endif
    
    #ifdef AMD_MFMA_MMF_AVAILABLE
    template<typename T>
    __global__ void mmf_mfma_cdna(
        const T * __restrict__ A,
        const T * __restrict__ B,
        T * __restrict__ C,
        const int M, const int N, const int K) {
        
        // MFMA uses 32x32 or 16x16 tiles
        // Different instruction format than WMMA
        using mfma_type = __builtin_amdgcn_mfma_f32_32x32x8f16;
        
        // Process with MFMA instructions
        ...
    }
    #endif
    ```

- Evidence mapping:
  - "CDNA detection" → `__gfx908__`, `__gfx90a__`, `__gfx942__`
  - "MFMA instructions" → `__builtin_amdgcn_mfma_*`
  - "Datacenter GPUs" → MI100, MI210, MI300 support

---

## Optimization 6: Micro-optimizations for mul_mat_id
- Commit ID: 106220562
- Optimization type: Compute (micro-optimization)
- Summary: Various micro-optimizations in MMF for mul_mat_id operations
- Detailed explanation: This optimization includes several small improvements: better register usage, reduced shared memory bank conflicts, and improved memory access patterns for the MoE use case.

- Code excerpt:
    ```cpp
    // CUDA: some micro-optimizations in mmf.cuh for mul_mat_id
    
    // 1. Precompute expert offset once
    const int64_t expert_offset = expert_id * weight_stride;
    
    // 2. Use __ldg for read-only data
    const T val = __ldg(&weights[expert_offset + idx]);
    
    // 3. Align shared memory access
    __shared__ __align__(16) T tile[TILE_SIZE];
    
    // 4. Unroll critical loops
    #pragma unroll 4
    for (int k = 0; k < K_TILE; k++) {
        ...
    }
    ```

- Evidence mapping:
  - "Precomputed offset" → avoid repeated multiplication
  - "Read-only cache" → `__ldg()` for texture cache
  - "Alignment" → `__align__(16)` for vectorized access
  - "Loop unrolling" → `#pragma unroll 4`
