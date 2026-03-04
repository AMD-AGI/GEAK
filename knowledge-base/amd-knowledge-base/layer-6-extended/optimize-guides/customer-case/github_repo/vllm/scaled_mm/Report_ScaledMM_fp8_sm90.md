---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: CUTLASS Scaled Matrix Multiplication (FP8)

## Variant Context
- Input semantic type: Matrix multiplication with FP8 quantized inputs
- Datatype(s): fp8_e4m3 (inputs), fp16/bf16 (output)
- Data representation: Per-tensor, per-token, or block-wise scaled FP8
- Target architecture: CUDA sm90 (Hopper)

## Functionality
The CUTLASS scaled matrix multiplication kernel performs efficient FP8 GEMM operations with various scaling strategies:
1. Per-tensor scaling: Single scale for entire tensor
2. Per-token scaling: Different scale per row (activation)
3. Per-channel scaling: Different scale per column (weight)
4. Block-wise scaling: Different scale per block (DeepSeek-V3 style)

## Optimization 1: Initial CUTLASS W8A8 Kernels
- Commit ID: 2060e9365
- Optimization type: Compute
- Summary: Add CUTLASS-based W8A8 (INT8/FP8) kernels for efficient quantized GEMM
- Detailed explanation:
  This optimization introduces CUTLASS 2.x and 3.x kernels for quantized matrix multiplication. CUTLASS provides highly optimized GEMM implementations that leverage Tensor Cores for maximum throughput.
- Code excerpt:
    ```cpp
    // CUTLASS 3.x kernel for FP8 GEMM on Hopper
    template <typename ElementA, typename ElementB, typename ElementD,
              typename AccumulatorType, typename ScaleType>
    void cutlass_scaled_mm_sm90(torch::Tensor& out, torch::Tensor const& a,
                                 torch::Tensor const& b, torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales) {
      using namespace cute;
      
      // Define GEMM configuration
      using TileShape = Shape<_128, _128, _64>;
      using ClusterShape = Shape<_1, _2, _1>;
      
      // Use TMA (Tensor Memory Accelerator) for efficient data movement
      using CollectiveMainloop = typename cutlass::gemm::collective::
          CollectiveBuilder<...>::CollectiveOp;
      
      // Custom epilogue for scaling
      using CollectiveEpilogue = cutlass::epilogue::collective::
          ScaledEpilogue<...>;
      
      // Launch kernel
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernel> gemm_op;
      gemm_op(args, stream);
    }
    ```
- Evidence mapping:
  - "CUTLASS 3.x" → Using `cutlass::gemm::collective` namespace
  - "Tensor Core utilization" → TileShape optimized for Tensor Cores
  - "TMA support" → Hopper's Tensor Memory Accelerator for async loads

## Optimization 2: Custom Epilogue for Scaled Output
- Commit ID: 85657b560
- Optimization type: Fusion
- Summary: Factor out and optimize epilogues for scaled GEMM operations
- Detailed explanation:
  The epilogue phase of GEMM applies scaling factors and converts to output type. This optimization creates custom epilogues that fuse scaling, bias addition, and type conversion into a single pass over the output tile.
- Code excerpt:
    ```cpp
    // Custom epilogue for scaled MM with bias
    template <typename ElementOutput, typename ElementCompute,
              typename ElementScale, typename ElementBias>
    struct ScaledEpilogueWithBias {
      
      struct Arguments {
        ElementScale const* a_scales;
        ElementScale const* b_scales;
        ElementBias const* bias;
      };
      
      CUTLASS_DEVICE
      void operator()(AccumulatorFragment& accum, 
                      OutputFragment& output,
                      int row, int col) {
        // Load scales
        ElementCompute a_scale = a_scales[row];
        ElementCompute b_scale = b_scales[col];
        
        // Apply scaling: output = accum * a_scale * b_scale + bias
        ElementCompute scaled = accum * a_scale * b_scale;
        if (bias != nullptr) {
          scaled += bias[col];
        }
        
        // Convert to output type
        output = cutlass::NumericConverter<ElementOutput, ElementCompute>{}(scaled);
      }
    };
    ```
- Evidence mapping:
  - "Fused epilogue" → Single operator applies scale, bias, and conversion
  - "Per-row/col scaling" → `a_scales[row]` and `b_scales[col]` indexing
  - "Optional bias" → Conditional bias addition in same pass

## Optimization 3: Per-Token and Per-Channel Quantization Fix
- Commit ID: c11de33da
- Optimization type: Precision
- Summary: Fix per-token/per-channel quantization for Hopper scaled MM
- Detailed explanation:
  Per-token (row-wise) and per-channel (column-wise) quantization require careful handling of scale broadcasting. This fix ensures correct scale application for asymmetric quantization scenarios.
- Code excerpt:
    ```cpp
    // Correct scale broadcasting for per-token/per-channel
    template <typename ScaleA, typename ScaleB>
    struct ScaleBroadcast {
      // Per-token: scale_a has shape [M, 1]
      // Per-channel: scale_b has shape [1, N]
      
      CUTLASS_DEVICE
      auto get_scale_a(int m, int n) {
        if constexpr (is_per_token<ScaleA>) {
          return scale_a_ptr[m];  // Broadcast across N
        } else {
          return scale_a_ptr[0];  // Single value
        }
      }
      
      CUTLASS_DEVICE
      auto get_scale_b(int m, int n) {
        if constexpr (is_per_channel<ScaleB>) {
          return scale_b_ptr[n];  // Broadcast across M
        } else {
          return scale_b_ptr[0];  // Single value
        }
      }
    };
    ```
- Evidence mapping:
  - "Per-token scaling" → `scale_a_ptr[m]` indexed by row
  - "Per-channel scaling" → `scale_b_ptr[n]` indexed by column
  - "Compile-time dispatch" → `if constexpr` for zero-overhead branching

## Optimization 4: Block-wise Quantization for DeepSeek-V3
- Commit ID: 9798b2fb0, eb5741ad4
- Optimization type: Precision
- Summary: Add 2D group (block-wise) scaling support for DeepSeek-V3 models
- Detailed explanation:
  DeepSeek-V3 uses block-wise quantization where the weight matrix is divided into blocks, each with its own scale. This provides better accuracy than per-tensor scaling while being more efficient than per-element scaling.
- Code excerpt:
    ```cpp
    // Block-wise scaling for DeepSeek-V3
    template <int BlockM, int BlockK>
    struct BlockwiseScale {
      // Scale tensor shape: [M/BlockM, K/BlockK]
      
      CUTLASS_DEVICE
      ElementScale get_scale(int m, int k) {
        int block_m = m / BlockM;
        int block_k = k / BlockK;
        return scales[block_m * num_blocks_k + block_k];
      }
      
      // Apply during GEMM mainloop
      CUTLASS_DEVICE
      void apply_scale(AccumulatorFragment& accum, int m_start, int k_start) {
        // Each accumulator tile may span multiple scale blocks
        for (int m = 0; m < TileM; m += BlockM) {
          for (int k = 0; k < TileK; k += BlockK) {
            ElementScale scale = get_scale(m_start + m, k_start + k);
            // Apply scale to corresponding accumulator elements
            apply_to_fragment(accum, m, k, scale);
          }
        }
      }
    };
    ```
- Evidence mapping:
  - "Block-wise indexing" → `block_m = m / BlockM`, `block_k = k / BlockK`
  - "2D scale tensor" → `scales[block_m * num_blocks_k + block_k]`
  - "Mainloop integration" → Scale applied during accumulation, not just epilogue

## Optimization 5: Hopper-Specific Optimizations (SM90)
- Commit ID: 8936316d5
- Optimization type: Compute / Memory
- Summary: Refactor CUTLASS 3.x kernels with Hopper-specific optimizations
- Detailed explanation:
  Hopper architecture introduces new features like TMA (Tensor Memory Accelerator) and warp-specialized execution. This optimization leverages these features for maximum performance.
- Code excerpt:
    ```cpp
    // Hopper-optimized GEMM with TMA and warp specialization
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<_128, _256, _64>,  // Tile shape optimized for Hopper
      cutlass::gemm::collective::CollectiveMma<
        // TMA-based mainloop
        cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<...>,
        // Warp-specialized epilogue
        cutlass::epilogue::collective::Sm90TmaWarpSpecialized<...>
      >
    >;
    
    // TMA descriptor setup
    auto tma_a = make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(a_ptr, make_layout(shape_a, stride_a)),
      SmemLayoutA{});
    ```
- Evidence mapping:
  - "TMA usage" → `SM90_TMA_LOAD` and `make_tma_copy`
  - "Warp specialization" → `MainloopSm90TmaGmmaWarpSpecialized`
  - "Optimized tile size" → `Shape<_128, _256, _64>` for Hopper Tensor Cores

## Optimization 6: Bias Epilogue Support
- Commit ID: 5bfd1bbc9
- Optimization type: Fusion
- Summary: Add bias addition support to CUTLASS scaled MM epilogue
- Detailed explanation:
  Many linear layers include a bias term. This optimization fuses bias addition into the GEMM epilogue, avoiding a separate kernel launch and memory round-trip.
- Code excerpt:
    ```cpp
    // Epilogue with optional bias
    template <typename ElementOutput, typename ElementBias>
    struct ScaledEpilogueWithOptionalBias {
      
      using Arguments = ScaledEpilogueArguments<ElementOutput, ElementBias>;
      
      CUTLASS_DEVICE
      ElementOutput operator()(ElementAccumulator accum, 
                               ElementScale a_scale,
                               ElementScale b_scale,
                               ElementBias const* bias,
                               int col) {
        ElementAccumulator result = accum * a_scale * b_scale;
        
        if (bias != nullptr) {
          result += static_cast<ElementAccumulator>(bias[col]);
        }
        
        return cutlass::NumericConverter<ElementOutput, ElementAccumulator>{}(result);
      }
    };
    
    // Entry point handles bias presence
    void cutlass_scaled_mm(out, a, b, a_scales, b_scales, 
                           c10::optional<torch::Tensor> bias) {
      if (bias.has_value()) {
        launch_kernel<WithBias>(out, a, b, a_scales, b_scales, bias.value());
      } else {
        launch_kernel<NoBias>(out, a, b, a_scales, b_scales);
      }
    }
    ```
- Evidence mapping:
  - "Optional bias" → `c10::optional<torch::Tensor> bias` parameter
  - "Fused addition" → `result += bias[col]` in epilogue
  - "Zero overhead when unused" → Separate kernel paths for with/without bias
