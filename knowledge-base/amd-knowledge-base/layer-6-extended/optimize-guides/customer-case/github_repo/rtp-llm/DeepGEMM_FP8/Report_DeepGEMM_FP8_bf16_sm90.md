---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: DeepGEMM FP8 (Grouped Contiguous)

## Variant Context
- Input semantic type: Matrix multiplication (MoE expert GEMM)
- Datatype(s): FP8 (e4m3) inputs with per-token group quantization, BF16 output
- Data representation: Contiguous grouped layout with TMA-aligned scales
- Target architecture: SM90 (NVIDIA Hopper - H100/H20)

## Functionality
DeepGEMM is a high-performance FP8 GEMM kernel optimized for NVIDIA Hopper architecture. It leverages:
- Tensor Memory Accelerator (TMA) for efficient data movement
- Warp-specialized execution for overlapping compute and memory operations
- FP8 tensor cores for maximum throughput

The kernel is used primarily for MoE (Mixture of Experts) GEMM operations where multiple expert matrices are processed in a grouped/batched manner.

## Optimization 1: Dynamic Block M Selection Based on Padding Strategy
- Commit ID: 86756f2c3
- Optimization type: Launch configuration / Compute
- Summary: Dynamically select block M size (64 or 128) based on whether 64-padding is used
- Detailed explanation:
  The optimization introduces dynamic selection of the block M dimension based on the padding strategy:
  - When `use_64_padding=true`: Use block_m=64 for better efficiency with 64-aligned data
  - When `use_64_padding=false`: Use block_m=128 for larger tiles
  
  This is particularly important for MoE workloads where token counts per expert may not align well with larger block sizes. Using 64-padding with block_m=64 reduces wasted computation on padding tokens.

- Code excerpt:
    ```cpp
    // From ConfigUtils.h
    DeepGemmConfig getBestConfig(int          m,
                                 int          n,
                                 int          k,
                                 int          num_groups,
                                 int          num_sms,
                                 DeepGemmType gemm_type,
                                 int          expected_m     = -1,
                                 bool         use_64_padding = false) {
        // ...
        if (gemm_type == DeepGemmType::GroupedContiguous && use_64_padding) {
            block_m = 64;
        }
        // ...
    }
    ```
- Evidence mapping:
  - New parameter → `bool use_64_padding = false` added to getBestConfig signature
  - Conditional block size → `if (gemm_type == DeepGemmType::GroupedContiguous && use_64_padding) { block_m = 64; }`

## Optimization 2: Propagate Padding Strategy to GEMM Configuration
- Commit ID: 86756f2c3
- Optimization type: Compute / Memory
- Summary: Pass padding strategy from MoE layer to DeepGEMM for optimal configuration
- Detailed explanation:
  The optimization ensures that the padding strategy used in the MoE layer is propagated to the DeepGEMM kernel configuration. This allows:
  - Consistent alignment between data preparation and kernel execution
  - Optimal block size selection based on actual data layout
  - Reduced wasted computation on padding elements

- Code excerpt:
    ```cpp
    // From DeepGemmPlugin.cpp
    void DeepGemmPlugin::groupedGemmFp8Contiguous(const Buffer& lhs,
                                                   const Buffer& rhs,
                                                   Buffer&       output,
                                                   const Buffer& m_indices,
                                                   int           user_deep_gemm_num_sm,
                                                   bool          use_64_padding,  // New parameter
                                                   cudaStream_t  stream) {
        // ...
        auto best_config = getBestConfig(m, n, k, 1, num_sms, 
                                         DeepGemmType::GroupedContiguous, -1, use_64_padding);
        // ...
    }
    ```
- Evidence mapping:
  - New function parameter → `bool use_64_padding` added to groupedGemmFp8Contiguous
  - Configuration propagation → `use_64_padding` passed to getBestConfig call

## Optimization 3: MoE-Specific Block Size Heuristics
- Commit ID: 86756f2c3
- Optimization type: Launch configuration
- Summary: Implement MoE-aware heuristics for block size selection in FP8 MoE layer
- Detailed explanation:
  The CudaFP8Moe layer now includes logic to determine the optimal padding and block size strategy based on:
  - Number of tokens per expert
  - Total number of experts
  - GPU SM count
  
  This ensures that the DeepGEMM kernel receives optimal configuration for the specific MoE workload.

- Code excerpt:
    ```cpp
    // From CudaFP8Moe.cc - Determining padding strategy
    bool use_64_padding = false;
    // Heuristic: use 64-padding when average tokens per expert is small
    if (total_tokens / num_experts < 64) {
        use_64_padding = true;
    }
    
    // Pass to DeepGEMM
    deep_gemm_plugin_->groupedGemmFp8Contiguous(
        input, weights, output, m_indices, num_sms, use_64_padding, stream);
    ```
- Evidence mapping:
  - Padding decision logic → Heuristic based on tokens per expert ratio
  - Integration with DeepGEMM → `use_64_padding` passed through the call chain

## Optimization 4: Math Utility for Alignment Calculations
- Commit ID: 86756f2c3
- Optimization type: Utility / Memory
- Summary: Added utility functions for alignment calculations to support padding strategies
- Detailed explanation:
  New math utility functions were added to support the padding and alignment calculations needed for optimal block size selection.

- Code excerpt:
    ```cpp
    // From math_utils.h
    template<typename T>
    inline T align_up(T value, T alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }
    
    template<typename T>
    inline T align_down(T value, T alignment) {
        return value / alignment * alignment;
    }
    ```
- Evidence mapping:
  - Alignment utilities → `align_up` and `align_down` template functions
  - Used for padding calculations → Ensures data is properly aligned for chosen block size
