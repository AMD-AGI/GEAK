---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: LayerNorm / RMSNorm (Normalization Kernels)

## Variant Context
- Input semantic type: Normalization operations for neural networks
- Datatype(s): FP16/BF16/FP32
- Data representation: 2D tensors (batch × hidden_dim)
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The normalization kernels implement LayerNorm and RMSNorm operations commonly used in transformer models. LayerNorm normalizes across the hidden dimension with mean and variance, while RMSNorm uses only the root mean square. Both support fused operations like residual addition and quantization.

## Optimization 1: Tree Reduction for Cross-Warp Synchronization
- Commit ID: 5a27a9739
- Optimization type: compute
- Summary: Introduced tree reduction pattern for BlockReduce2dCrossWarpSync, reducing reduction steps from O(N) to O(log N).

- Detailed explanation:
  The normalization kernels require computing mean and variance across the hidden dimension, which spans multiple warps. The tree reduction optimization organizes the cross-warp reduction in a binary tree pattern, significantly reducing synchronization overhead for large hidden dimensions.

- Code excerpt:
    ```cpp
    // Tree reduction for cross-warp synchronization
    template <typename T, index_t NumWarps>
    struct BlockReduce2dCrossWarpSync
    {
        // Tree reduction: O(log N) steps instead of O(N)
        CK_TILE_DEVICE T reduce(T thread_data, T* smem)
        {
            const index_t warp_id = get_warp_id();
            const index_t lane_id = get_lane_id();
            
            // Intra-warp reduction first
            T warp_result = warp_reduce(thread_data);
            
            // Store warp results to shared memory
            if(lane_id == 0)
                smem[warp_id] = warp_result;
            
            __syncthreads();
            
            // Tree reduction across warps
            for(index_t stride = NumWarps / 2; stride > 0; stride /= 2)
            {
                if(warp_id < stride && lane_id == 0)
                {
                    smem[warp_id] += smem[warp_id + stride];
                }
                __syncthreads();
            }
            
            return smem[0];
        }
    };
    ```

- Evidence mapping:
  - "Tree reduction" → `stride /= 2` loop pattern
  - "O(log N) steps" → Halving stride each iteration
  - "Cross-warp sync" → `__syncthreads()` between levels

## Optimization 2: Two-Pass Pipeline for Improved Accuracy
- Commit ID: d49abdaa8
- Optimization type: precision / compute
- Summary: Implemented two-pass pipeline for RMS/Layer normalization with improved numerical accuracy.

- Detailed explanation:
  The two-pass approach first computes the mean (for LayerNorm) or sum of squares (for RMSNorm) in one pass, then applies the normalization in a second pass. This improves numerical stability compared to single-pass algorithms, especially for FP16.

- Code excerpt:
    ```cpp
    // Two-pass normalization pipeline
    template <typename Problem>
    struct Normalization2PassPipeline
    {
        CK_TILE_DEVICE void operator()(const InputType* input, OutputType* output,
                                        const ScaleType* gamma, const BiasType* beta)
        {
            // Pass 1: Compute statistics (mean, variance/rms)
            AccType sum = 0;
            AccType sum_sq = 0;
            
            for(index_t i = thread_start; i < thread_end; i += stride)
            {
                AccType val = static_cast<AccType>(input[i]);
                sum += val;
                sum_sq += val * val;
            }
            
            // Cross-thread reduction
            sum = block_reduce_sum(sum);
            sum_sq = block_reduce_sum(sum_sq);
            
            AccType mean = sum / hidden_dim;
            AccType var = sum_sq / hidden_dim - mean * mean;
            AccType inv_std = rsqrt(var + epsilon);
            
            // Pass 2: Apply normalization
            for(index_t i = thread_start; i < thread_end; i += stride)
            {
                AccType val = static_cast<AccType>(input[i]);
                AccType normalized = (val - mean) * inv_std;
                output[i] = static_cast<OutputType>(normalized * gamma[i] + beta[i]);
            }
        }
    };
    ```

- Evidence mapping:
  - "Two-pass" → Separate loops for statistics and normalization
  - "Numerical stability" → Computing mean before variance
  - "FP16 accuracy" → Using AccType (FP32) for intermediate calculations

## Optimization 3: Selectable Implementation for Accuracy
- Commit ID: 3499fe67f
- Optimization type: precision
- Summary: Added pipeline pass for selecting between different RMSNorm implementations based on accuracy requirements.

- Detailed explanation:
  Different applications have different accuracy requirements. This optimization allows runtime selection between:
  - Fast mode: Single-pass with potential accuracy loss
  - Accurate mode: Two-pass with better numerical stability
  - Model-sensitive mode: Extra precision for sensitive models

- Code excerpt:
    ```cpp
    enum class RMSNormMode
    {
        Fast,           // Single-pass, fastest
        Accurate,       // Two-pass, better accuracy
        ModelSensitive  // Extra precision for sensitive models
    };
    
    template <RMSNormMode Mode>
    struct RMSNormPipelineSelector
    {
        using type = std::conditional_t<
            Mode == RMSNormMode::Fast,
            RMSNormFastPipeline,
            std::conditional_t<
                Mode == RMSNormMode::Accurate,
                RMSNorm2PassPipeline,
                RMSNormModelSensitivePipeline
            >
        >;
    };
    ```

- Evidence mapping:
  - "Selectable implementation" → `RMSNormMode` enum
  - "Runtime selection" → Template-based pipeline selection
  - "Model-sensitive" → Special mode for accuracy-critical models

## Optimization 4: Fused Add + RMSNorm + Quantization
- Commit ID: 04dd31488
- Optimization type: fusion
- Summary: Added fused kernel combining residual addition, RMSNorm, and output quantization in a single pass.

- Detailed explanation:
  For transformer inference, the common pattern is: residual = input + residual; output = rmsnorm(residual). Fusing these operations reduces memory bandwidth by avoiding intermediate tensor materialization.

- Code excerpt:
    ```cpp
    // Fused Add + RMSNorm + Quantization kernel
    template <typename Problem>
    struct AddRMSNormRdQuantKernel
    {
        CK_TILE_DEVICE void operator()(
            const InputType* input,
            const ResidualType* residual,
            OutputType* output,
            QuantOutputType* quant_output,
            const ScaleType* gamma,
            float quant_scale)
        {
            // Fused computation: no intermediate tensor
            AccType sum_sq = 0;
            
            // Pass 1: Add residual and compute sum of squares
            for(index_t i = tid; i < hidden_dim; i += blockDim.x)
            {
                AccType val = static_cast<AccType>(input[i]) + 
                              static_cast<AccType>(residual[i]);
                temp_buffer[i] = val;  // Store in registers/LDS
                sum_sq += val * val;
            }
            
            AccType rms = rsqrt(sum_sq / hidden_dim + epsilon);
            
            // Pass 2: Normalize and quantize
            for(index_t i = tid; i < hidden_dim; i += blockDim.x)
            {
                AccType normalized = temp_buffer[i] * rms * gamma[i];
                output[i] = static_cast<OutputType>(normalized);
                quant_output[i] = quantize(normalized, quant_scale);
            }
        }
    };
    ```

- Evidence mapping:
  - "Fused operations" → Single kernel for add+norm+quant
  - "No intermediate tensor" → Direct computation without materialization
  - "Quantization output" → `quant_output` for INT8/FP8 inference
