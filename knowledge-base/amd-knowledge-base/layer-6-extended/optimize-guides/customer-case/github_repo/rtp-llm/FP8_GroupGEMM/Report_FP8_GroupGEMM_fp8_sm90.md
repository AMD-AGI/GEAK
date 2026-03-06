Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: FP8 Cutlass Group GEMM

## Variant Context
- Input semantic type: Matrix multiplication (MoE expert GEMM)
- Datatype(s): FP8 (e4m3) inputs and outputs
- Data representation: Grouped GEMM with per-group problem sizes
- Target architecture: SM90 (NVIDIA Hopper - H100/H20)

## Functionality
This kernel implements FP8 grouped GEMM using CUTLASS 3.x for NVIDIA Hopper architecture. It is designed for Mixture of Experts (MoE) models where multiple expert weight matrices need to be multiplied with different subsets of input tokens. The kernel leverages:
- CUTLASS 3.x collective operations
- Hopper's TMA (Tensor Memory Accelerator)
- Cluster-based execution for improved data sharing
- Warp-specialized mainloop for overlapping compute and memory

## Optimization 1: Comprehensive Heuristics Search for Tile and Cluster Configuration
- Commit ID: bcdf71549
- Optimization type: Launch configuration / Compute
- Summary: Introduced a comprehensive heuristics-based configuration search for optimal tile and cluster shapes
- Detailed explanation:
  The optimization adds a new heuristics search module that selects optimal tile and cluster configurations based on:
  - Problem dimensions (M, N, K)
  - Number of SMs on the target GPU (78 for H20, 132 for H100)
  - Occupancy calculations
  
  The heuristics cover a wide range of tile shapes from 64x16 to 256x128, enabling fine-grained optimization for different batch sizes.

- Code excerpt:
    ```cpp
    // From heuristics_search.hpp
    template<typename InType, typename OutType>
    tc::CutlassGemmConfig get_best_config_customized_sm90(int m, int n, int k, 
                                                           const int& num_groups, 
                                                           const int num_sms) {
        if (num_sms == 78) {  // H20
            auto best_cluster_shape = tc::ClusterShape::ClusterShape_1x1x1;
            auto best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x128x128B;
            if (m <= 2) {
                best_tile_shape = tc::CutlassTileConfigSM90::CtaShape128x16x128B;
            } else if (m > 2 && m <= 8) {
                best_tile_shape = tc::CutlassTileConfigSM90::CtaShape64x16x128B;
            } else if (m > 8 && m <= 16) {
                best_tile_shape = tc::CutlassTileConfigSM90::CtaShape128x16x256B;
            } else if (m > 32 && m <= 64) {
                best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x16x256B;
                best_cluster_shape = tc::ClusterShape::ClusterShape_2x1x1;
            } else if (m > 64 && m <= 4096) {
                best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape64x64x128B;
            } else {
                best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape64x128x128B;
                best_cluster_shape = tc::ClusterShape::ClusterShape_1x2x1;
            }
            // ...
        }
    }
    ```
- Evidence mapping:
  - SM-count aware selection → `if (num_sms == 78)` for H20-specific tuning
  - Fine-grained M thresholds → Multiple conditions for m <= 2, 2 < m <= 8, 8 < m <= 16, etc.
  - Combined tile and cluster optimization → Both `best_tile_shape` and `best_cluster_shape` selected together

## Optimization 2: Extended Tile Shape Coverage
- Commit ID: bcdf71549
- Optimization type: Compute
- Summary: Added support for a comprehensive set of tile shapes to cover all batch size ranges
- Detailed explanation:
  The optimization extends the supported tile shapes to include:
  - Very narrow tiles (64x16, 128x16) for small batch decode
  - Medium tiles (64x32, 64x64, 128x32, 128x64) for medium batches
  - Wide tiles (64x128, 64x256, 128x128, 128x256, 256x128) for large batches
  
  This comprehensive coverage ensures optimal performance across the full range of LLM inference scenarios.

- Code excerpt:
    ```cpp
    // From heuristics_search.hpp - Tile shape converter
    ShapeInfo get_tile_config_shape_info(tc::CutlassTileConfigSM90 config) {
        switch (config) {
            SHAPE_CASE(64, 16, 128)
            SHAPE_CASE(64, 32, 128)
            SHAPE_CASE(64, 64, 128)
            SHAPE_CASE(64, 128, 128)
            SHAPE_CASE(64, 256, 128)
            SHAPE_CASE(128, 16, 128)
            SHAPE_CASE(128, 32, 128)
            SHAPE_CASE(128, 64, 128)
            SHAPE_CASE(128, 128, 128)
            SHAPE_CASE(128, 256, 128)
            SHAPE_CASE(256, 128, 128)
            default:
                return ShapeInfo(128, 128, 128);
        }
    }
    ```
- Evidence mapping:
  - Comprehensive tile coverage → 11 different tile configurations from 64x16 to 256x128
  - K dimension fixed at 128 → Optimized for FP8 tensor core operations

## Optimization 3: Cluster Shape Optimization for Different Workloads
- Commit ID: bcdf71549
- Optimization type: Launch configuration
- Summary: Utilize Hopper's cluster execution model with workload-specific cluster shapes
- Detailed explanation:
  The optimization leverages Hopper's cluster-based execution with different cluster configurations:
  - 1x1x1: Default for most workloads
  - 2x1x1: For medium batch sizes (32-64) to improve M-dimension parallelism
  - 1x2x1: For large batches (>4096) to improve N-dimension parallelism
  - 2x2x1: For very large workloads
  - 8x1x1: For specific configurations requiring high M-parallelism

- Code excerpt:
    ```cpp
    // From heuristics_search.hpp - Cluster shape selection
    tc::ClusterShape cluster_shape_converter(int m, int n, int k) {
        if (m == 1 && n == 1 && k == 1) {
            return tc::ClusterShape::ClusterShape_1x1x1;
        } else if (m == 2 && n == 1 && k == 1) {
            return tc::ClusterShape::ClusterShape_2x1x1;
        } else if (m == 1 && n == 2 && k == 1) {
            return tc::ClusterShape::ClusterShape_1x2x1;
        } else if (m == 2 && n == 2 && k == 1) {
            return tc::ClusterShape::ClusterShape_2x2x1;
        } else if (m == 8 && n == 1 && k == 1) {
            return tc::ClusterShape::ClusterShape_8x1x1;
        }
        // ...
    }
    
    // Usage in heuristics
    if (m > 1 && m <= 2) {
        best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x32x128B;
        best_cluster_shape = tc::ClusterShape::ClusterShape_8x1x1;
    }
    ```
- Evidence mapping:
  - Multiple cluster shapes → 1x1x1, 2x1x1, 1x2x1, 2x2x1, 8x1x1 supported
  - Workload-specific selection → Different cluster shapes for different M ranges

## Optimization 4: K-Dimension Tile Variation for Memory Bandwidth Optimization
- Commit ID: bcdf71549
- Optimization type: Memory
- Summary: Support for K=256 tiles in addition to K=128 for better memory bandwidth utilization
- Detailed explanation:
  For certain configurations (particularly small M with medium batch sizes), using K=256 tiles instead of K=128 can improve memory bandwidth utilization by:
  - Reducing the number of memory transactions
  - Better amortizing TMA descriptor overhead
  - Improving L2 cache hit rates

- Code excerpt:
    ```cpp
    // From heuristics_search.hpp
    if (m > 8 && m <= 16) {
        best_tile_shape = tc::CutlassTileConfigSM90::CtaShape128x16x256B;  // K=256
    } else if (m > 32 && m <= 64) {
        best_tile_shape    = tc::CutlassTileConfigSM90::CtaShape128x16x256B;  // K=256
        best_cluster_shape = tc::ClusterShape::ClusterShape_2x1x1;
    }
    
    // Tile shape converter includes K=256 variant
    tc::CutlassTileConfigSM90 tile_shape_converter(int m, int n, int k) {
        // ...
        if (m == 128 && n == 16 && k == 256) {
            return tc::CutlassTileConfigSM90::CtaShape128x16x256B;
        }
        // ...
    }
    ```
- Evidence mapping:
  - K=256 tile support → CtaShape128x16x256B configuration
  - Selective usage → Only for specific M ranges (8-16, 32-64) where it provides benefit
