# Kernel: Cutlass W4A8 Group GEMM

## Variant Context
- Input semantic type: Matrix multiplication (MoE expert GEMM)
- Datatype(s): FP8 (e4m3) activations, INT4 weights, FP8 scales
- Data representation: Group-wise quantized weights with per-group scales
- Target architecture: SM90 (NVIDIA Hopper - H100/H20)

## Functionality
This kernel implements weight-only quantized group GEMM for Mixture of Experts (MoE) models on NVIDIA Hopper GPUs. It performs matrix multiplication with:
- FP8 (e4m3) input activations
- INT4 quantized weights with group-wise scaling
- BF16 output

The kernel is optimized for different batch sizes (M dimension) and GPU configurations (78 SMs for H20, 132 SMs for H100).

## Optimization 1: SM-Count Aware Tile Configuration
- Commit ID: ae01e3ae1
- Optimization type: Launch configuration / Compute
- Summary: Select optimal tile shapes based on GPU SM count (78 for H20, 132 for H100)
- Detailed explanation:
  The optimization introduces GPU-specific tile configuration selection. Different NVIDIA Hopper GPUs have different SM counts:
  - H20: 78 SMs
  - H100: 132 SMs
  
  The kernel selects different tile shapes to maximize occupancy and throughput on each GPU variant. For example, on H100 with more SMs, larger tile shapes can be used for larger batch sizes.

- Code excerpt:
    ```cpp
    static CutlassGemmConfig get_best_config_sm90(int m, int n, int k, const int& num_groups, const int num_sms) {
        if (num_sms == 78) {  // H20
            auto best_cluster_shape = ClusterShape::ClusterShape_1x1x1;
            auto best_tile_shape    = CutlassTileConfigSM90::CtaShape256x16x128B;
            if (m <= 8) {
                best_cluster_shape = ClusterShape::ClusterShape_2x1x1;
                best_tile_shape    = CutlassTileConfigSM90::CtaShape256x8x128B;
            } else if (m <= 16) {
                best_cluster_shape = ClusterShape::ClusterShape_1x1x1;
                best_tile_shape    = CutlassTileConfigSM90::CtaShape256x16x128B;
            } else if (m <= 32) {
                best_cluster_shape = ClusterShape::ClusterShape_1x1x1;
                best_tile_shape    = CutlassTileConfigSM90::CtaShape256x32x128B;
            } else {
                best_cluster_shape = ClusterShape::ClusterShape_1x1x1;
                best_tile_shape    = CutlassTileConfigSM90::CtaShape128x64x128B;
            }
            // ...
        } else if (num_sms == 132) {  // H100
            // Different configuration for H100
            if (m <= 8) {
                best_cluster_shape = ClusterShape::ClusterShape_2x1x1;
                best_tile_shape    = CutlassTileConfigSM90::CtaShape256x8x128B;
            } else if (m > 64) {
                best_cluster_shape = ClusterShape::ClusterShape_2x1x1;
                best_tile_shape    = CutlassTileConfigSM90::CtaShape256x128x128B;
            }
            // ...
        }
    }
    ```
- Evidence mapping:
  - SM-count detection → `if (num_sms == 78)` and `else if (num_sms == 132)` branches
  - Different tile shapes per GPU → CtaShape256x8x128B for small M on both, CtaShape256x128x128B for large M on H100

## Optimization 2: Cluster Shape Optimization for Small Batch Sizes
- Commit ID: ae01e3ae1
- Optimization type: Launch configuration
- Summary: Use 2x1x1 cluster shape for small batch sizes to improve SM utilization
- Detailed explanation:
  For small batch sizes (m <= 8), the kernel uses a 2x1x1 cluster shape instead of 1x1x1. This allows:
  - Better utilization of Hopper's cluster-based execution model
  - Improved data sharing between CTAs in the same cluster
  - Higher effective occupancy for memory-bound small-batch operations

- Code excerpt:
    ```cpp
    if (m <= 8) {
        best_cluster_shape = ClusterShape::ClusterShape_2x1x1;
        best_tile_shape    = CutlassTileConfigSM90::CtaShape256x8x128B;
    }
    
    // Enable 2x1x1 cluster shape dispatch
    switch (gemm_config.cluster_shape) {
        CLUSTER_SHAPE_CASE(1, 1, 1)
        CLUSTER_SHAPE_CASE(2, 1, 1)  // Newly enabled
    }
    ```
- Evidence mapping:
  - Cluster shape selection → `ClusterShape::ClusterShape_2x1x1` for m <= 8
  - Enabled dispatch → `CLUSTER_SHAPE_CASE(2, 1, 1)` uncommented in dispatch_cluster_shape

## Optimization 3: Batch-Size Adaptive Tile Shape Selection
- Commit ID: ae01e3ae1
- Optimization type: Compute / Launch configuration
- Summary: Select tile shapes based on batch size (M dimension) for optimal performance
- Detailed explanation:
  The kernel implements a heuristic-based tile shape selection that adapts to the batch size:
  - Very small M (≤8): Use narrow tiles (256x8) to avoid wasting compute
  - Small M (≤16): Use 256x16 tiles
  - Medium M (≤32): Use 256x32 tiles
  - Large M (>64): Use wider tiles (128x64 or 256x128) for better compute efficiency

- Code excerpt:
    ```cpp
    // For H100 (132 SMs)
    if (m <= 8) {
        best_tile_shape = CutlassTileConfigSM90::CtaShape256x8x128B;
    } else if (m <= 16) {
        best_tile_shape = CutlassTileConfigSM90::CtaShape256x16x128B;
    } else if (m <= 32) {
        best_tile_shape = CutlassTileConfigSM90::CtaShape256x32x128B;
    } else if (m <= 64) {
        best_tile_shape = CutlassTileConfigSM90::CtaShape128x64x128B;
    } else {
        best_tile_shape = CutlassTileConfigSM90::CtaShape256x128x128B;
    }
    ```
- Evidence mapping:
  - Batch-size thresholds → Explicit `if (m <= X)` conditions
  - Tile shape progression → 256x8 → 256x16 → 256x32 → 128x64 → 256x128 as M increases

## Optimization 4: Extended Tile Shape Support
- Commit ID: ae01e3ae1
- Optimization type: Compute
- Summary: Added support for additional tile shapes (128x64, 256x8) for better coverage
- Detailed explanation:
  The optimization adds new tile shape configurations to the dispatch table:
  - 128x64x128: Better for medium batch sizes where 256-wide tiles waste compute
  - 256x8x128: Optimal for very small batch sizes (decode phase with few tokens)
  - 128x128x128: Alternative for large batches

- Code excerpt:
    ```cpp
    switch (gemm_config.tile_config_sm90) {
        TILE_SHAPE_CASE(128, 64, 128, SWAP_AB)   // Newly added
        TILE_SHAPE_CASE(128, 128, 128, SWAP_AB)  // Newly added
        TILE_SHAPE_CASE(256, 8, 128, SWAP_AB)    // Newly added
        TILE_SHAPE_CASE(256, 16, 128, SWAP_AB)
        TILE_SHAPE_CASE(256, 32, 128, SWAP_AB)
        TILE_SHAPE_CASE(256, 64, 128, SWAP_AB)
        TILE_SHAPE_CASE(256, 128, 128, SWAP_AB)
    }
    ```
- Evidence mapping:
  - New tile shapes → TILE_SHAPE_CASE(128, 64, 128), TILE_SHAPE_CASE(256, 8, 128), TILE_SHAPE_CASE(128, 128, 128)
  - Template instantiation → Each case instantiates `cute::Shape<cute::_M, cute::_N, cute::_K>` for the tile
