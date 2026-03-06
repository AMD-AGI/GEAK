Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: CUTLASS MLA (Multi-head Latent Attention) Kernel

## Variant Context
- Input semantic type: Multi-head Latent Attention for DeepSeek V2/V3 models
- Datatype(s): FP16, BF16
- Data representation: Compressed KV cache with latent representations
- Target architecture: SM90 (Hopper), SM100 (Blackwell)

## Functionality
This kernel implements CUTLASS-based Multi-head Latent Attention (MLA) for DeepSeek models. MLA uses a compressed KV cache representation where K and V are projected to a lower-dimensional latent space, significantly reducing memory requirements for long-context inference.

## Optimization 1: Blackwell (SM100) Support
- Commit ID: f65b8d5c8
- Optimization type: Architecture Support
- Summary: Adds CUTLASS MLA kernel support for Blackwell (SM100) architecture with optimized tile configurations.

- Detailed explanation:
  Blackwell GPUs have enhanced tensor core capabilities and larger shared memory. This optimization adds SM100-specific kernel instantiations that leverage these improvements for MLA computation.

- Code excerpt:
    ```cpp
    // SM100 (Blackwell) specific configuration
    #if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
    template void cutlass_mla_decode_sm100<cutlass::bfloat16_t>(...);
    template void cutlass_mla_decode_sm100<cutlass::half_t>(...);
    #endif
    ```

- Evidence mapping:
  - "Blackwell support" → SM100 template instantiations
  - "CUDA version guard" → `CUDA_VERSION >= 12080`

## Optimization 2: Extended Head Count Support
- Commit ID: 18efb5e8e
- Optimization type: Flexibility
- Summary: Extends CUTLASS MLA decode to support models with fewer than 128 attention heads.

- Detailed explanation:
  The original implementation assumed 128 attention heads. This optimization adds support for smaller head counts, enabling MLA for a wider range of model configurations.

- Code excerpt:
    ```cpp
    // Support for variable head counts
    template <int NUM_HEADS>
    void cutlass_mla_decode_dispatch(
        // ... parameters
    ) {
        if constexpr (NUM_HEADS >= 128) {
            // Original path for large head counts
        } else {
            // Optimized path for smaller head counts
        }
    }
    ```

- Evidence mapping:
  - "Variable head support" → template parameter `NUM_HEADS`
  - "Conditional dispatch" → `if constexpr (NUM_HEADS >= 128)`

## Optimization 3: Removed Slow Concat Kernel
- Commit ID: aa46ed34d
- Optimization type: Latency
- Summary: Removes a 200μs slow concatenation kernel by fusing the operation into the MLA kernel.

- Detailed explanation:
  The MLA computation previously required a separate concatenation step for combining different attention components. This optimization fuses the concatenation into the main MLA kernel, eliminating significant latency overhead.

- Code excerpt:
    ```cpp
    // Fused concatenation within MLA kernel
    // Previously: separate concat kernel + MLA kernel
    // Now: single fused kernel handles both operations
    ```

- Evidence mapping:
  - "Removed concat" → eliminated separate kernel launch
  - "200μs savings" → significant latency reduction
