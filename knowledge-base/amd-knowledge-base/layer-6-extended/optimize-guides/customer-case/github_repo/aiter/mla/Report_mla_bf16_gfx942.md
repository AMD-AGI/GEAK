Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: mla (Multi-head Latent Attention)

## Variant Context
- Input semantic type: Multi-head Latent Attention (DeepSeek architecture)
- Datatype(s): BF16/FP16 (Q, K, V), FP8 (optional)
- Data representation: Latent compressed KV with RoPE integration
- Target architecture: gfx942 (MI300), gfx950 (MI350)

## Functionality
This kernel implements Multi-head Latent Attention (MLA) used in DeepSeek models. MLA compresses the KV cache using low-rank projections, significantly reducing memory requirements. The kernel handles:
1. Latent KV decompression
2. RoPE (Rotary Position Embedding) application
3. Scaled dot-product attention
4. Support for paged KV cache layouts

## Optimization 1: Persistent Mode for MLA Prefill (gfx950)
- Commit ID: 2d76e1812
- Optimization type: Scheduling / Compute
- Summary: Enabled persistent kernel mode for MLA prefill on gfx950, reducing kernel launch overhead.

- Detailed explanation:
  Persistent kernels keep thread blocks alive across multiple work items, avoiding the overhead of repeated kernel launches. For MLA prefill, this is particularly beneficial because:
  1. Reduces kernel launch latency
  2. Enables better workload balancing across CUs
  3. Improves cache utilization by keeping data in L2

- Code excerpt:
    ```cpp
    // feat(mla_prefill): enable asm mla_prefill with persistent mode for gfx950
    ```

- Evidence mapping:
  - Persistent mode → Commit title mentions "persistent mode"
  - gfx950 specific → Architecture-specific optimization

## Optimization 2: MLA Reduce Refactoring
- Commit ID: 70238cfe3
- Optimization type: Compute / Memory
- Summary: Refactored MLA reduce kernel for better performance and maintainability.

- Detailed explanation:
  The MLA reduce kernel combines partial attention results from multiple thread blocks. The refactoring improves:
  1. Memory access patterns for partial results
  2. Reduction algorithm efficiency
  3. Support for different head configurations (nhead4, nhead32, nhead64)

- Code excerpt:
    ```cpp
    // Refactor MLA Reduce
    // Add nhead4 & nhead32 case to support mla_reduce_v1 used in pa_persistent_fwd
    ```

- Evidence mapping:
  - Reduce refactoring → Commit title "Refactor MLA Reduce"
  - Multiple head configs → nhead4, nhead32, nhead64 support

## Optimization 3: Paged KV Cache with Page Size 64
- Commit ID: 0e0a37812
- Optimization type: Memory
- Summary: Added support for page size 64 and 3-buffer layout for DeepSeek 3.2 models.

- Detailed explanation:
  The optimization adds support for:
  1. Page size 64 (in addition to existing page sizes)
  2. 3-buffer layout: `page_size * nhead * dim` for efficient memory access
  3. Optimized metadata generation for paged attention

- Code excerpt:
    ```cpp
    // mla ps support paged 64 and 3buffer layout for ds3.2
    // support page_size=64 in metadata kernel
    // fix test tail block error and 3buffer layout to page_size * nhead * dim
    ```

- Evidence mapping:
  - Page size 64 → Commit title mentions "paged 64"
  - 3-buffer layout → `page_size * nhead * dim` layout
  - DeepSeek 3.2 → "ds3.2" in commit message

## Optimization 4: Multi-Head Configuration Support
- Commit ID: a83979399
- Optimization type: Compute
- Summary: Added support for nhead64 and nhead32 configurations for different model sizes.

- Detailed explanation:
  Different DeepSeek model sizes use different numbers of attention heads. The optimization adds specialized kernel paths for:
  1. nhead64: Larger models with 64 attention heads
  2. nhead32: Medium models with 32 attention heads
  3. Optimized register allocation for each configuration

- Code excerpt:
    ```cpp
    // [MLA] nhead64 and nhead32
    ```

- Evidence mapping:
  - nhead64 support → Commit title mentions "nhead64"
  - nhead32 support → Commit title mentions "nhead32"

## Optimization 5: FP8 V3 Data Synchronization Fix
- Commit ID: 3728dcedf
- Optimization type: Compute / Correctness
- Summary: Fixed data synchronization issues in FP8 MLA V3 implementation.

- Detailed explanation:
  The FP8 variant of MLA V3 had synchronization issues that could cause incorrect results. The fix ensures proper data dependencies between:
  1. FP8 dequantization
  2. Attention computation
  3. Output accumulation

- Code excerpt:
    ```cpp
    // fix mla f8 v3 data sync issue
    ```

- Evidence mapping:
  - FP8 variant → "f8" in commit message
  - V3 implementation → "v3" in commit message
  - Sync fix → "data sync issue" in commit message

## Optimization 6: Workgroup Calculation Fix
- Commit ID: 5d163d673
- Optimization type: Scheduling
- Summary: Fixed calculation of number of workgroups per batch and head for correct parallelization.

- Detailed explanation:
  The kernel launch configuration was incorrectly calculating the number of workgroups, leading to suboptimal GPU utilization. The fix ensures:
  1. Correct workgroup count for all batch/head combinations
  2. Proper load balancing across CUs
  3. Avoiding over-subscription or under-utilization

- Code excerpt:
    ```cpp
    // Fix calculation on number of workgroup per batch and head
    ```

- Evidence mapping:
  - Workgroup calculation → Commit title mentions "workgroup per batch and head"
  - Parallelization fix → Correct GPU utilization
