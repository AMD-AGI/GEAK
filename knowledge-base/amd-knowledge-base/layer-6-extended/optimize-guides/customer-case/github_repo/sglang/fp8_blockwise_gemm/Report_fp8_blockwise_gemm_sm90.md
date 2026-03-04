---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: FP8 Blockwise GEMM Kernel

## Variant Context
- Input semantic type: Matrix multiplication for linear layers in LLMs
- Datatype(s): FP8 (float8_e4m3fn for CUDA, float8_e4m3fnuz for ROCm), with FP32 accumulation
- Data representation: Blockwise quantized with per-block scales (typically 128x128 blocks)
- Target architecture: SM90 (Hopper), SM100 (Blackwell), SM120

## Functionality
This kernel implements high-performance FP8 matrix multiplication using CUTLASS 3.x with blockwise scaling. It's designed for quantized LLM inference where weights and activations are quantized to FP8 with per-block scale factors. The kernel supports both persistent and stream-K scheduling strategies.

## Optimization 1: CUTLASS 3.9 Scale Configuration API
- Commit ID: 5bb0accbc
- Optimization type: Compute / API
- Summary: Upgrades to CUTLASS 3.9's new blockwise scale configuration API, which provides better scale factor layout handling.

- Detailed explanation:
  CUTLASS 3.9 introduced a new API for handling blockwise scale factors through `sm90_trivial_blockwise_scale_config`. This optimization adopts the new API which:
  1. Automatically deduces optimal scale factor layouts (LayoutSFA, LayoutSFB)
  2. Provides better integration with the TMA (Tensor Memory Accelerator) pipeline
  3. Simplifies the kernel configuration by removing manual scale granularity parameters

- Code excerpt:
    ```cpp
    // New CUTLASS 3.9 scale configuration
    using ScaleTileShape = Shape<_1, _128, _128>;
    using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(ScaleTileShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    // Updated collective builder with scale layouts
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        cute::tuple<LayoutA, LayoutSFA>,  // Paired layout with scale
        AlignmentA,
        ElementB,
        cute::tuple<LayoutB, LayoutSFB>,  // Paired layout with scale
        AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<...>,
        KernelSchedule>::CollectiveOp;
    ```

    Previous version (before optimization):
    ```cpp
    // Old manual scale granularity specification
    template <typename ScaleGranularity>  // e.g., Shape<_1, _128, _128>
    void launch_sm90_fp8_blockwise_scaled_mm(...) {
      static constexpr int ScaleGranularityM = size<0>(ScaleGranularity{});
      static constexpr int ScaleGranularityN = size<1>(ScaleGranularity{});
      using KernelSchedule = 
          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum<
              ScaleGranularityM, ScaleGranularityN>;
      // ...
    }
    ```

- Evidence mapping:
  - "New scale config API" → `sm90_trivial_blockwise_scale_config(ScaleTileShape{})`
  - "Automatic layout deduction" → `ScaleConfig::deduce_layoutSFA()`
  - "Paired layouts" → `cute::tuple<LayoutA, LayoutSFA>`
  - "Simplified kernel schedule" → `KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum` without template params

## Optimization 2: Dynamic Scheduler Selection Based on Matrix Shape
- Commit ID: 5bb0accbc (refined from earlier commits)
- Optimization type: Scheduling
- Summary: Dynamically selects between StreamK and Persistent schedulers based on the K/N ratio of the matrix dimensions.

- Detailed explanation:
  Different GEMM shapes benefit from different scheduling strategies:
  - StreamK scheduler: Better for tall-skinny matrices (K >> N) as it provides better load balancing
  - Persistent scheduler: Better for square-ish matrices as it reduces scheduling overhead
  
  This optimization automatically selects the scheduler based on the K/N ratio, using StreamK when K > 3*N.

- Code excerpt:
    ```cpp
    template <typename OutType>
    void sm90_fp8_blockwise_dispatch_shape(
        torch::Tensor& out,
        const torch::Tensor& a,
        const torch::Tensor& b,
        const torch::Tensor& scales_a,
        const torch::Tensor& scales_b) {
      using TileShape = Shape<_128, _128, _128>;
      using ClusterShape = Shape<_1, _2, _1>;

      auto k = a.size(1);
      auto n = b.size(1);
      if (k > 3 * n) {
        // Tall-skinny: use StreamK for better load balancing
        launch_sm90_fp8_blockwise_scaled_mm<
            cutlass::gemm::StreamKScheduler, OutType, TileShape, ClusterShape>(
            out, a, b, scales_a, scales_b);
      } else {
        // Square-ish: use Persistent for lower overhead
        launch_sm90_fp8_blockwise_scaled_mm<
            cutlass::gemm::PersistentScheduler, OutType, TileShape, ClusterShape>(
            out, a, b, scales_a, scales_b);
      }
    }
    ```

- Evidence mapping:
  - "Shape-based selection" → `if (k > 3 * n)`
  - "StreamK for tall matrices" → `cutlass::gemm::StreamKScheduler` when K > 3*N
  - "Persistent for square" → `cutlass::gemm::PersistentScheduler` otherwise

## Optimization 3: SM100/SM120 (Blackwell) Support
- Commit ID: dd1e26893, 7a16db9bd
- Optimization type: Architecture Support
- Summary: Extends FP8 blockwise GEMM support to SM100 (Blackwell) and SM103, SM120 architectures.

- Detailed explanation:
  Blackwell GPUs (SM100+) have enhanced FP8 tensor core capabilities. This optimization adds architecture-specific kernel instantiations for SM100, SM103, and SM120, taking advantage of the improved hardware support for FP8 operations.

- Code excerpt:
    ```cpp
    // SM100/SM120 support
    #if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
    template void sm100_fp8_blockwise_scaled_mm<torch::kBFloat16>(...);
    template void sm100_fp8_blockwise_scaled_mm<torch::kFloat16>(...);
    #endif

    // Architecture dispatch
    void fp8_blockwise_scaled_mm(torch::Tensor& out, ...) {
      auto compute_capability = cyclic_get_sm_version();
      if (compute_capability >= 100) {
        sm100_fp8_blockwise_dispatch_shape<OutType>(...);
      } else if (compute_capability >= 90) {
        sm90_fp8_blockwise_dispatch_shape<OutType>(...);
      }
    }
    ```

- Evidence mapping:
  - "SM100+ detection" → `compute_capability >= 100`
  - "CUDA version guard" → `CUDA_VERSION >= 12080`
  - "Architecture-specific dispatch" → separate `sm100_fp8_blockwise_dispatch_shape`

## Optimization 4: Optimized Mainloop Arguments with Scale Layouts
- Commit ID: 5bb0accbc
- Optimization type: Memory Layout
- Summary: Uses CUTLASS's scale layout deduction to optimize memory access patterns for scale factors.

- Detailed explanation:
  The scale factors for blockwise quantization need to be accessed efficiently during the GEMM computation. This optimization uses CUTLASS's `tile_atom_to_shape_SFA/SFB` functions to compute optimal layouts for the scale tensors based on the problem shape, ensuring efficient memory access patterns.

- Code excerpt:
    ```cpp
    // Compute optimal scale layouts based on problem shape
    LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

    // Pass layouts to mainloop arguments
    typename GemmKernel::MainloopArguments mainloop_args{
        a_ptr, stride_a, 
        b_ptr, stride_b, 
        4,  // pipeline stages
        a_s_ptr, layout_sfa,  // A scales with layout
        b_s_ptr, layout_sfb   // B scales with layout
    };
    ```

- Evidence mapping:
  - "Layout computation" → `ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1))`
  - "Layout in arguments" → `a_s_ptr, layout_sfa` passed to mainloop
  - "Problem-shape dependent" → layouts computed based on `(m, n, k, 1)`

## Optimization 5: CUTLASS 4.0 Upgrade
- Commit ID: c23a7072b
- Optimization type: Framework / Performance
- Summary: Upgrades to CUTLASS 4.0 which includes various performance improvements and bug fixes for FP8 operations.

- Detailed explanation:
  CUTLASS 4.0 brings significant improvements to FP8 GEMM performance including:
  - Better TMA utilization
  - Improved warp-specialized cooperative kernels
  - Enhanced epilogue fusion capabilities
  - Bug fixes for edge cases in blockwise scaling

- Code excerpt:
    ```cpp
    // CUTLASS 4.0 features utilized
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
    
    // Enhanced epilogue with auto tile selection
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
    ```

- Evidence mapping:
  - "CUTLASS 4.0" → version upgrade in CMakeLists.txt/setup.py
  - "Warp-specialized kernel" → `KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum`
  - "Auto epilogue tile" → `EpilogueTileAuto` for optimal epilogue configuration
