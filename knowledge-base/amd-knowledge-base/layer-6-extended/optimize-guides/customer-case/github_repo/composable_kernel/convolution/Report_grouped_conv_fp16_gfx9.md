Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Grouped Convolution (Forward/Backward)

## Variant Context
- Input semantic type: Convolution operations for CNNs
- Datatype(s): FP16/BF16/FP32
- Data representation: NHWC/NCHW tensor layouts
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The grouped convolution kernels implement forward and backward passes for convolutional neural networks. They use implicit GEMM formulation where convolution is mapped to matrix multiplication, enabling efficient use of MFMA instructions.

## Optimization 1: Merge Multiple Conv Groups into Single GEMM Batch
- Commit ID: 66832861a
- Optimization type: compute / scheduling
- Summary: Merged multiple forward convolution groups into a single GEMM batch for improved GPU utilization.

- Detailed explanation:
  When processing multiple convolution groups (e.g., in grouped convolution or depthwise separable convolution), each group can be treated as a separate GEMM. By batching these GEMMs together, we improve GPU occupancy and reduce kernel launch overhead.

- Code excerpt:
    ```cpp
    // Merge convolution groups into batched GEMM
    template <typename Problem>
    struct GroupedConvAsBatchedGemm
    {
        // Map convolution groups to GEMM batch dimension
        static constexpr index_t kBatchSize = Problem::kNumGroups;
        
        CK_TILE_DEVICE auto make_batched_gemm_args(
            const ConvArgs& conv_args)
        {
            // Each group becomes a batch in GEMM
            // A: [G, N*Ho*Wo, C/G * Fy * Fx] - input im2col
            // B: [G, C/G * Fy * Fx, K/G] - weight
            // C: [G, N*Ho*Wo, K/G] - output
            
            return BatchedGemmArgs{
                .batch_count = kBatchSize,
                .M = conv_args.N * conv_args.Ho * conv_args.Wo,
                .N = conv_args.K / kBatchSize,
                .K = (conv_args.C / kBatchSize) * conv_args.Fy * conv_args.Fx,
                .batch_stride_A = conv_args.group_stride_input,
                .batch_stride_B = conv_args.group_stride_weight,
                .batch_stride_C = conv_args.group_stride_output
            };
        }
    };
    ```

- Evidence mapping:
  - "Multiple groups" → `kNumGroups` mapped to batch dimension
  - "Single GEMM batch" → `BatchedGemmArgs` with `batch_count`
  - "Improved utilization" → All groups processed in parallel

## Optimization 2: Split-K for Convolution
- Commit ID: fc22320d7
- Optimization type: compute / scheduling
- Summary: Added Split-K support with auto-deduction for convolution kernels.

- Detailed explanation:
  Split-K parallelizes the K dimension (reduction dimension) across multiple workgroups, then reduces partial results. The auto-deduction feature automatically determines the optimal split factor based on problem size and GPU resources.

- Code excerpt:
    ```cpp
    // Split-K auto-deduction for convolution
    template <typename Problem>
    struct ConvSplitKAutoDeduction
    {
        CK_TILE_HOST static index_t deduce_split_k(
            const ConvArgs& args,
            index_t num_cus)
        {
            // Compute number of output tiles
            index_t num_m_tiles = (args.N * args.Ho * args.Wo + kMPerBlock - 1) / kMPerBlock;
            index_t num_n_tiles = (args.K + kNPerBlock - 1) / kNPerBlock;
            index_t num_tiles = num_m_tiles * num_n_tiles;
            
            // Target: enough tiles to fill all CUs with good occupancy
            index_t target_tiles = num_cus * kTargetOccupancy;
            
            if(num_tiles >= target_tiles)
                return 1;  // No split needed
            
            // Compute split factor to reach target
            index_t k_tiles = (args.C * args.Fy * args.Fx + kKPerBlock - 1) / kKPerBlock;
            index_t split_k = (target_tiles + num_tiles - 1) / num_tiles;
            split_k = min(split_k, k_tiles);  // Can't split more than K tiles
            
            return split_k;
        }
    };
    ```

- Evidence mapping:
  - "Split-K" → Parallelization across K dimension
  - "Auto-deduction" → `deduce_split_k` function
  - "GPU resources" → `num_cus` and `kTargetOccupancy`

## Optimization 3: Index Optimizations for Backward Data
- Commit ID: 59265d5eb
- Optimization type: compute
- Summary: Added indexing optimizations for convolution backward data pass.

- Detailed explanation:
  The backward data pass requires complex index calculations to map output gradients back to input gradients. The optimization precomputes and reuses index calculations, reducing redundant arithmetic.

- Code excerpt:
    ```cpp
    // Optimized index calculation for conv backward data
    template <typename Problem>
    struct ConvBwdDataIndexOptimizer
    {
        // Precompute stride products
        static constexpr index_t kStrideHW = Problem::kStrideH * Problem::kStrideW;
        static constexpr index_t kDilationHW = Problem::kDilationH * Problem::kDilationW;
        
        CK_TILE_DEVICE auto compute_input_indices(
            index_t n, index_t hi, index_t wi, index_t c,
            index_t fy, index_t fx)
        {
            // Optimized: use precomputed strides
            index_t ho = (hi + Problem::kPadH - fy * Problem::kDilationH) / Problem::kStrideH;
            index_t wo = (wi + Problem::kPadW - fx * Problem::kDilationW) / Problem::kStrideW;
            
            // Check bounds with single comparison
            bool valid = (ho >= 0) && (ho < Problem::kHo) && 
                         (wo >= 0) && (wo < Problem::kWo) &&
                         ((hi + Problem::kPadH - fy * Problem::kDilationH) % Problem::kStrideH == 0) &&
                         ((wi + Problem::kPadW - fx * Problem::kDilationW) % Problem::kStrideW == 0);
            
            return std::make_tuple(ho, wo, valid);
        }
    };
    ```

- Evidence mapping:
  - "Precomputed strides" → `kStrideHW`, `kDilationHW` constexpr
  - "Reduced arithmetic" → Single formula for ho, wo
  - "Bounds checking" → Combined validity check

## Optimization 4: Explicit GEMM Formulation
- Commit ID: 00dfa2f2c
- Optimization type: compute
- Summary: Implemented explicit GEMM formulation for grouped convolution.

- Detailed explanation:
  The explicit GEMM approach directly maps convolution to GEMM without im2col transformation, computing the im2col indices on-the-fly. This saves memory bandwidth by avoiding the materialization of the im2col matrix.

- Code excerpt:
    ```cpp
    // Explicit GEMM: compute im2col indices on-the-fly
    template <typename Problem>
    struct ConvExplicitGemm
    {
        CK_TILE_DEVICE void load_a_tile(
            const InputType* input,
            ATile& a_tile,
            index_t m_offset,  // N*Ho*Wo dimension
            index_t k_offset)  // C*Fy*Fx dimension
        {
            // Decompose m_offset to n, ho, wo
            index_t n = m_offset / (Problem::kHo * Problem::kWo);
            index_t hw = m_offset % (Problem::kHo * Problem::kWo);
            index_t ho = hw / Problem::kWo;
            index_t wo = hw % Problem::kWo;
            
            // Decompose k_offset to c, fy, fx
            index_t c = k_offset / (Problem::kFy * Problem::kFx);
            index_t fyfx = k_offset % (Problem::kFy * Problem::kFx);
            index_t fy = fyfx / Problem::kFx;
            index_t fx = fyfx % Problem::kFx;
            
            // Compute input coordinates
            index_t hi = ho * Problem::kStrideH - Problem::kPadH + fy * Problem::kDilationH;
            index_t wi = wo * Problem::kStrideW - Problem::kPadW + fx * Problem::kDilationW;
            
            // Load with bounds checking
            if(hi >= 0 && hi < Problem::kHi && wi >= 0 && wi < Problem::kWi)
            {
                index_t input_offset = ((n * Problem::kHi + hi) * Problem::kWi + wi) * Problem::kC + c;
                a_tile = input[input_offset];
            }
            else
            {
                a_tile = 0;  // Zero padding
            }
        }
    };
    ```

- Evidence mapping:
  - "Explicit GEMM" → On-the-fly index computation
  - "No im2col" → Direct input access with computed indices
  - "Memory savings" → No intermediate im2col buffer
