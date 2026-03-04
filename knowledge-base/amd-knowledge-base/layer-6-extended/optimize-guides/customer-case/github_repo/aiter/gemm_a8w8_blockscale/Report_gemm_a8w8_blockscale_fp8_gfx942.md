---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: gemm_a8w8_blockscale

## Variant Context
- Input semantic type: Matrix multiplication (GEMM)
- Datatype(s): FP8 (E4M3) activation, FP8 (E4M3) weight with block-wise scaling
- Data representation: Block-wise quantized with 128-element groups for A, 128x128 blocks for B
- Target architecture: gfx942 (MI300), gfx950 (MI350)

## Functionality
This kernel performs FP8 GEMM with block-wise scaling factors. The activation matrix A uses per-row 128-element group scaling, while the weight matrix B uses 128x128 block scaling. This enables efficient low-precision matrix multiplication while maintaining accuracy through fine-grained scaling.

## Optimization 1: CK-Tile Based Implementation
- Commit ID: da2948717
- Optimization type: Compute / Architecture
- Summary: Added CK-Tile (Composable Kernel Tile) based implementation for FP8 block-scale GEMM, providing better tile-level control and optimization opportunities.

- Detailed explanation: 
  The optimization introduces a new CK-Tile based implementation that provides:
  1. Configurable tile sizes (M_Tile, N_Tile, K_Tile) for different problem sizes
  2. Warp-level tiling (M_Warp, N_Warp, K_Warp) for efficient workload distribution
  3. Support for double shared memory buffering to hide memory latency
  4. Intrawave scheduling for better instruction-level parallelism
  5. Persistent kernel mode for reduced kernel launch overhead
  6. Automatic padding handling for non-aligned dimensions

- Code excerpt:
    ```cpp
    template <ck_tile::index_t M_Tile,
              ck_tile::index_t N_Tile,
              ck_tile::index_t K_Tile,
              ck_tile::index_t M_Warp,
              ck_tile::index_t N_Warp,
              ck_tile::index_t K_Warp,
              ck_tile::index_t M_Warp_Tile,
              ck_tile::index_t N_Warp_Tile,
              ck_tile::index_t K_Warp_Tile,
              bool TiledMMAPermuteN                    = false,
              bool TransposeC                          = false,
              bool DoubleSmemBuffer                    = false,
              bool UsePersistentKernel                 = false,
              ck_tile::GemmPipelineScheduler Scheduler = ck_tile::GemmPipelineScheduler::Intrawave,
              int BlockPerCu                           = 1>
    struct CreateTileGemmConfig
    {
        // Configuration parameters for tile-based GEMM
    };
    
    using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
    using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
    ```

- Evidence mapping:
  - Tile-level configuration → `M_Tile, N_Tile, K_Tile` template parameters
  - Warp-level tiling → `M_Warp, N_Warp, K_Warp` parameters
  - Double buffering → `DoubleSmemBuffer` parameter
  - Intrawave scheduling → `GemmPipelineScheduler::Intrawave`
  - Block-wise quantization → `AQuantGroupSize` and `BQuantGroupSize` with 128-element groups

## Optimization 2: Tail Handling for Non-Aligned K Dimensions
- Commit ID: da2948717
- Optimization type: Compute / Precision
- Summary: Implemented efficient tail handling for K dimensions that don't align with tile sizes.

- Detailed explanation:
  The kernel uses a tail handler mechanism to efficiently process the remaining elements when K dimension doesn't divide evenly by the tile size. This avoids wasted computation and ensures correct results for arbitrary matrix sizes.

- Code excerpt:
    ```cpp
    const ck_tile::index_t K_split =
        (args.K + GemmConfig::K_Tile_v - 1) / GemmConfig::K_Tile_v * GemmConfig::K_Tile_v;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    
    // Tail handling with padding support
    template <typename QDataType, typename OutDataType, typename GemmConfig>
    void TileGemmCompute(ck_tile::QuantGemmHostArgs& args)
    {
        const bool pad_n = (args.N % BQuantGroupSize::kN != 0);
        const bool pad_k = (args.K % AQuantGroupSize::kK != 0);
    
        if(pad_n && pad_k)
            TileGemmComputeImpl<QDataType, OutDataType, GemmConfig, true, true>(args);
        else if(pad_n && !pad_k)
            TileGemmComputeImpl<QDataType, OutDataType, GemmConfig, true, false>(args);
        // ... other combinations
    }
    ```

- Evidence mapping:
  - K dimension alignment → `K_split` calculation with ceiling division
  - Hot loop detection → `has_hot_loop` for optimized main loop
  - Padding specialization → Template parameters `PadN, PadK` for compile-time optimization

## Optimization 3: Quantized GEMM Pipeline with Scale Handling
- Commit ID: da2948717
- Optimization type: Precision / Memory
- Summary: Specialized pipeline for handling quantized inputs with separate scale tensors.

- Detailed explanation:
  The implementation uses a specialized quantized GEMM pipeline (`ABQuantGemmPipelineAgBgCrCompV3`) that efficiently handles:
  1. Loading quantized A and B matrices
  2. Loading corresponding scale tensors
  3. Dequantization during computation
  4. Accumulation in higher precision (FP32)

- Code excerpt:
    ```cpp
    using PipelineProblem = ck_tile::GemmABQuantPipelineProblem<ADataType,
                                                                QDataType, // AQDataType
                                                                BDataType,
                                                                QDataType, // BQDataType
                                                                AccDataType,
                                                                GemmShape,
                                                                GemmTraits,
                                                                AQuantGroupSize,
                                                                BQuantGroupSize,
                                                                GemmConfig::TransposeC_v,
                                                                ComputeDataType,
                                                                GemmConfig::Scheduler_v,
                                                                has_hot_loop_v,
                                                                tail_number_v>;

    using GemmPipeline = ck_tile::ABQuantGemmPipelineAgBgCrCompV3<PipelineProblem>;
    ```

- Evidence mapping:
  - Separate scale handling → `AQDataType, BQDataType` template parameters
  - FP32 accumulation → `AccDataType = TILE_FP32`
  - Quantization group sizes → `AQuantGroupSize, BQuantGroupSize` parameters
