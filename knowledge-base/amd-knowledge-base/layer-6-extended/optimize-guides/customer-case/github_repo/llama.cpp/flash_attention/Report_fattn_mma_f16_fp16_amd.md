Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Flash Attention MMA (fattn-mma-f16) - AMD HIP

## Variant Context
- Input semantic type: Attention (Query-Key-Value dot product with softmax)
- Datatype(s): FP16 (half precision)
- Data representation: Dense FP16 tensors for Q, K, V
- Target architecture: AMD (RDNA3 gfx1100+, RDNA3.5 gfx1150+, RDNA4 gfx1200+)

## Functionality
This kernel implements Flash Attention for AMD GPUs using WMMA (Wave Matrix Multiply-Accumulate) instructions available on RDNA3 and newer architectures. It provides the same functionality as the NVIDIA variant but uses AMD-specific matrix instructions and optimizations.

---

## Optimization 1: RDNA4 WMMA Support for Flash Attention
- Commit ID: ea4a321f2
- Optimization type: Compute (architecture support)
- Summary: Added Flash Attention MMA support for AMD RDNA4 GPUs using WMMA instructions
- Detailed explanation: RDNA4 (gfx1200) introduces improved WMMA instructions. This commit enables the MMA-based Flash Attention kernel on RDNA4, leveraging the 16x16 wave matrix operations. The implementation uses rocWMMA library abstractions for portability across RDNA generations.

- Code excerpt:
    ```cpp
    // HIP: add fattn-mma-f16 for RDNA4
    #if defined(AMD_WMMA_AVAILABLE)
    static constexpr fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna(...) {
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
        GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);
        ...
    }
    #endif
    ```

- Evidence mapping:
  - "RDNA4 support" → `AMD_WMMA_AVAILABLE` macro check
  - "WMMA instructions" → configuration tuned for wave matrix operations
  - "16x16 tiles" → implicit in WMMA instruction format

---

## Optimization 2: rocWMMA Integration for CDNA and RDNA
- Commit ID: becade5de
- Optimization type: Compute (library integration)
- Summary: Implemented Flash Attention using rocWMMA library for both CDNA (MI series) and RDNA3+ GPUs
- Detailed explanation: This commit added Flash Attention support via the rocWMMA library, which provides a unified interface for matrix operations across AMD GPU architectures. CDNA uses MFMA instructions while RDNA3+ uses WMMA, but rocWMMA abstracts these differences.

- Code excerpt:
    ```cpp
    // HIP: implement FlashAttention via rocWMMA for CDNA and RDNA3+
    #include <rocwmma/rocwmma.hpp>
    
    using namespace rocwmma;
    
    // Fragment types for WMMA operations
    using FragA = fragment<matrix_a, 16, 16, 16, half, row_major>;
    using FragB = fragment<matrix_b, 16, 16, 16, half, col_major>;
    using FragC = fragment<accumulator, 16, 16, 16, float>;
    
    // Matrix multiply-accumulate
    FragC c_frag;
    fill_fragment(c_frag, 0.0f);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
    ```

- Evidence mapping:
  - "rocWMMA library" → `#include <rocwmma/rocwmma.hpp>`
  - "Unified interface" → same fragment types work on CDNA and RDNA
  - "16x16x16 tiles" → fragment dimensions in template parameters

---

## Optimization 3: RDNA3.5 Kernel Selection Tuning
- Commit ID: d2ff4e23a
- Optimization type: Launch configuration
- Summary: Adjusted kernel selection logic for RDNA3.5 (gfx1150) to optimize performance
- Detailed explanation: RDNA3.5 (found in AMD AI 370/395 laptop chips) has different performance characteristics than desktop RDNA3. This commit tunes the kernel selection thresholds to choose between MMQ and rocBLAS based on problem size and shape.

- Code excerpt:
    ```cpp
    // HIP: adjust RDNA3.5 MMQ kernel selection logic
    #define GGML_CUDA_CC_RDNA3_5    (GGML_CUDA_CC_OFFSET_AMD + 0x1150)
    
    #define GGML_CUDA_CC_IS_RDNA3_5(cc) (cc >= GGML_CUDA_CC_RDNA3_5 && cc < GGML_CUDA_CC_RDNA4)
    
    static bool should_use_mmq(const int cc, const int ne11, ...) {
        if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
            // Different thresholds for laptop chips
            return ne11 <= 32 || ...;
        }
        ...
    }
    ```

- Evidence mapping:
  - "RDNA3.5 detection" → `GGML_CUDA_CC_IS_RDNA3_5(cc)` macro
  - "Tuned thresholds" → different `ne11` threshold for laptop chips

---

## Optimization 4: Wave Size Handling for AMD GPUs
- Commit ID: 34c961b18
- Optimization type: Compute (architecture adaptation)
- Summary: Fixed Flash Attention vector kernels to handle AMD's 64-wide wavefronts correctly
- Detailed explanation: AMD GCN/CDNA GPUs use 64-thread wavefronts while RDNA uses 32-thread wavefronts. This optimization ensures the Flash Attention kernels work correctly regardless of wave size by using architecture-specific constants and avoiding hardcoded WARP_SIZE assumptions.

- Code excerpt:
    ```cpp
    // CUDA/HIP: Fix fattn-vec-* when device warp size is not 32
    static constexpr int get_physical_warp_size() {
    #if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
        #if defined(__gfx1100__) || defined(__gfx1101__) || ...
            return 32;  // RDNA uses 32-wide waves
        #else
            return 64;  // CDNA/GCN uses 64-wide waves
        #endif
    #else
        return 32;  // NVIDIA uses 32-wide warps
    #endif
    }
    
    // Use physical warp size instead of hardcoded 32
    constexpr int warp_size = get_physical_warp_size();
    ```

- Evidence mapping:
  - "Wave size detection" → `get_physical_warp_size()` function
  - "Architecture-specific" → `__gfx1100__` and similar macros
  - "64 vs 32 threads" → return values of 64 for CDNA, 32 for RDNA

---

## Optimization 5: FP16 Dot Product Instruction for AMD
- Commit ID: 17bc5a815
- Optimization type: Compute (instruction selection)
- Summary: Use AMD's `v_dot2_f32_f16` instruction for FP16 dot products in Flash Attention
- Detailed explanation: AMD GPUs have a specialized instruction `v_dot2_f32_f16` that computes a dot product of two FP16 pairs and accumulates to FP32. This is more efficient than separate multiply-add operations for the attention score computation.

- Code excerpt:
    ```cpp
    // HIP: use v_dot2_f32_f16 instruction for FA
    #if defined(__gfx90a__) || defined(__gfx942__) || ...
    static __device__ __forceinline__ float dot2_f16(half2 a, half2 b) {
        float result;
        asm volatile("v_dot2_f32_f16 %0, %1, %2, %3"
                     : "=v"(result)
                     : "v"(a), "v"(b), "v"(0.0f));
        return result;
    }
    #endif
    ```

- Evidence mapping:
  - "v_dot2_f32_f16" → inline assembly instruction
  - "FP16 pairs to FP32" → input half2, output float
  - "Architecture check" → `__gfx90a__`, `__gfx942__` guards
