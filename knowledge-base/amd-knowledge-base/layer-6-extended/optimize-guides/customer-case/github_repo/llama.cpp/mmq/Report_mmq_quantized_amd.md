Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Matrix Multiplication Quantized (MMQ) - AMD HIP

## Variant Context
- Input semantic type: Matrix multiplication (GEMM)
- Datatype(s): Quantized weights (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
- Data representation: Block-wise quantized with scales
- Target architecture: AMD (CDNA1 gfx908, CDNA2 gfx90a, CDNA3 gfx942, RDNA3 gfx1100+, RDNA4 gfx1200+)

## Functionality
The MMQ kernel for AMD GPUs performs quantized matrix multiplication using MFMA (Matrix Fused Multiply-Add) instructions on CDNA architectures and WMMA instructions on RDNA3+. The kernel handles the different wave sizes (64 for CDNA, 32 for RDNA) and instruction formats across AMD GPU generations.

---

## Optimization 1: MFMA Instructions for CDNA GPUs
- Commit ID: 66906cd82
- Optimization type: Compute (Matrix Core utilization)
- Summary: Enable MFMA (Matrix Fused Multiply-Add) instructions for MMQ on CDNA architectures
- Detailed explanation: This major optimization adds MFMA instruction support for AMD's CDNA GPUs (MI100, MI210, MI300). MFMA provides 32x32x8 or 16x16x16 matrix operations that significantly accelerate quantized GEMM. The implementation handles CDNA's 64-wide wavefronts and accumulator register architecture.

- Code excerpt:
    ```cpp
    // HIP: Enable Matrix cores for MMQ Kernels
    #if defined(AMD_MFMA_AVAILABLE)
    static int get_mmq_x_max_host(const int cc) {
        return (amd_mfma_available(cc) || new_mma_available(cc)) ? 128 : ...;
    }
    
    static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
        return mmq_x >= 128 ? 32 : 16;  // Larger granularity for MFMA
    }
    
    // MFMA tile configuration
    #define GGML_CUDA_CC_CDNA1 (GGML_CUDA_CC_OFFSET_AMD + 0x908)  // MI100
    #define GGML_CUDA_CC_CDNA2 (GGML_CUDA_CC_OFFSET_AMD + 0x910)  // MI210
    #define GGML_CUDA_CC_CDNA3 (GGML_CUDA_CC_OFFSET_AMD + 0x942)  // MI300
    ```

- Evidence mapping:
  - "MFMA support" → `AMD_MFMA_AVAILABLE` macro
  - "CDNA detection" → `GGML_CUDA_CC_CDNA1/2/3` constants
  - "Larger tiles" → `mmq_x >= 128` with granularity 32

---

## Optimization 2: Stream-K for CDNA3
- Commit ID: 66906cd82
- Optimization type: Scheduling (load balancing)
- Summary: Enable Stream-K work distribution specifically for CDNA3 (MI300) GPUs
- Detailed explanation: Stream-K is enabled only on CDNA3 because it consistently outperforms rocBLAS on MI300 due to issues in AMD's BLAS libraries. On older CDNA generations, the overhead of Stream-K doesn't always pay off, so it's disabled.

- Code excerpt:
    ```cpp
    // Stream-K only enabled on CDNA3
    static bool mmq_use_stream_k(const int cc) {
        return GGML_CUDA_CC_IS_CDNA3(cc);
    }
    
    // CDNA3 detection
    #define GGML_CUDA_CC_IS_CDNA3(cc) (cc >= GGML_CUDA_CC_CDNA3 && cc < GGML_CUDA_CC_RDNA1)
    
    // Stream-K provides better load balancing on MI300's many CUs
    if (mmq_use_stream_k(cc)) {
        launch_mul_mat_q_stream_k<...>(...);
    } else {
        launch_mul_mat_q<...>(...);
    }
    ```

- Evidence mapping:
  - "CDNA3 only" → `GGML_CUDA_CC_IS_CDNA3(cc)` check
  - "Stream-K selection" → `mmq_use_stream_k()` function
  - "MI300 optimization" → comment about rocBLAS issues

---

## Optimization 3: Wave Size Decoupling
- Commit ID: 66906cd82
- Optimization type: Compute (portability)
- Summary: Decouple shared memory tile sizes from WARP_SIZE to support different wave sizes
- Detailed explanation: AMD GPUs use 64-thread waves on CDNA and 32-thread waves on RDNA. This optimization introduces `MMQ_TILE_NE_K` constant to define tile sizes independently of wave size, enabling the same kernel to work efficiently on both architectures.

- Code excerpt:
    ```cpp
    // Decouple shared memory tile sizes from WARP_SIZE
    // The K dimension of tiles has 32 elements for quantized data
    #define MMQ_TILE_NE_K 32
    
    // Old: tied to WARP_SIZE
    // #define MMQ_DP4A_TXS_Q4_0 tile_x_sizes{mmq_y*WARP_SIZE + mmq_y, ...}
    
    // New: independent of wave size
    #define MMQ_DP4A_TXS_Q4_0 tile_x_sizes{mmq_y*MMQ_TILE_NE_K + mmq_y, ...}
    #define MMQ_DP4A_TXS_Q8_0 tile_x_sizes{mmq_y*MMQ_TILE_NE_K*2 + mmq_y, ...}
    
    // Use physical warp size for thread indexing
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_0);
    constexpr int nrows = warp_size / threads_per_row;
    ```

- Evidence mapping:
  - "Wave size independence" → `MMQ_TILE_NE_K` constant (32)
  - "Physical warp size" → `ggml_cuda_get_physical_warp_size()`
  - "Flexible indexing" → `nrows = warp_size / threads_per_row`

---

## Optimization 4: WMMA-MMQ for RDNA3/4
- Commit ID: 0543f928a, 668ed7657
- Optimization type: Compute (WMMA utilization)
- Summary: Enable WMMA-based MMQ kernels for RDNA3 and RDNA4 GPUs
- Detailed explanation: RDNA3 and RDNA4 support WMMA (Wave Matrix Multiply-Accumulate) instructions with 16x16 tiles. This optimization enables the WMMA code path for these architectures, providing Tensor Core-like acceleration for quantized GEMM.

- Code excerpt:
    ```cpp
    // HIP: WMMA-MMQ kernels for RDNA 4
    #if defined(__gfx1200__) || defined(__gfx1201__)
    #define AMD_WMMA_AVAILABLE
    #endif
    
    // HIP: enable WMMA-MMQ INT kernels for RDNA 3
    #if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    #define AMD_WMMA_AVAILABLE
    #endif
    
    #ifdef AMD_WMMA_AVAILABLE
    // Use WMMA for matrix multiply
    using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, int8_t>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, int8_t>;
    using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t>;
    
    rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    #endif
    ```

- Evidence mapping:
  - "RDNA3/4 detection" → `__gfx1100__`, `__gfx1200__` macros
  - "WMMA fragments" → `rocwmma::fragment` types
  - "INT8 WMMA" → `int8_t` element type for quantized data

---

## Optimization 5: MFMA for Specific Datatypes on CDNA1/2
- Commit ID: ad4a70011
- Optimization type: Compute (selective enablement)
- Summary: Enable MFMA MMQ on gfx908 and gfx90a for select datatypes and shapes where it outperforms rocBLAS
- Detailed explanation: On older CDNA generations (MI100, MI210), MFMA doesn't always outperform rocBLAS. This optimization selectively enables MFMA for specific quantization formats and matrix shapes where benchmarks show improvement.

- Code excerpt:
    ```cpp
    // HIP: enable mfma mmq on gfx908 and gfx90a for select datatypes and shapes
    static bool should_use_mfma_mmq(const int cc, ggml_type type, int ne11) {
        if (GGML_CUDA_CC_IS_CDNA1(cc) || GGML_CUDA_CC_IS_CDNA2(cc)) {
            // Only enable for specific types where MFMA wins
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q8_0:
                    return ne11 >= 32;  // Need sufficient batch size
                default:
                    return false;  // Use rocBLAS for other types
            }
        }
        return GGML_CUDA_CC_IS_CDNA3(cc);  // Always use on MI300
    }
    ```

- Evidence mapping:
  - "Selective enablement" → switch on `type` and `ne11` threshold
  - "CDNA1/2 specific" → `GGML_CUDA_CC_IS_CDNA1/2` checks
  - "Batch size requirement" → `ne11 >= 32` condition

---

## Optimization 6: MMQ/rocBLAS Switching for RDNA
- Commit ID: 968929528
- Optimization type: Launch configuration
- Summary: Tune the switching threshold between MMQ and rocBLAS for RDNA GPUs
- Detailed explanation: For RDNA GPUs, the optimal choice between custom MMQ kernels and rocBLAS depends on problem size. This optimization tunes the switching thresholds based on benchmarks to maximize performance across different workloads.

- Code excerpt:
    ```cpp
    // mmq.cu: tune mmq/rocblas switching for RDNA
    static bool should_use_mmq_rdna(const int cc, const int ne11, const int ne00) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // RDNA3: MMQ better for small batches
            if (ne11 <= 8) return true;
            if (ne11 <= 32 && ne00 <= 4096) return true;
            return false;  // Use rocBLAS for large problems
        }
        if (GGML_CUDA_CC_IS_RDNA2(cc)) {
            // RDNA2: Different thresholds
            return ne11 <= 4;
        }
        return false;
    }
    ```

- Evidence mapping:
  - "RDNA-specific tuning" → separate logic for RDNA2, RDNA3
  - "Size-based switching" → `ne11` and `ne00` thresholds
  - "Benchmark-driven" → different thresholds per architecture
