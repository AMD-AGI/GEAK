Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: MoE Routing Kernels (TopK, Align, Permute)

## Variant Context
- Input semantic type: Expert routing and token permutation
- Datatype(s): fp16, bf16, fp32 (gating scores), int32 (indices)
- Data representation: Dense gating scores, sparse expert assignments
- Target architecture: Generic CUDA/HIP

## Functionality
MoE routing kernels handle the expert selection and token organization:
1. TopK Softmax: Select top-k experts per token with softmax normalization
2. Align Block Size: Pad token assignments for efficient GEMM tiling
3. Permute/Unpermute: Reorder tokens by expert for batched computation

## Optimization 1: Fused TopK with Softmax
- Commit ID: f0d4e1455
- Optimization type: Fusion
- Summary: Fuse top-k selection with softmax computation in single kernel
- Detailed explanation:
  Computing softmax over all experts and then selecting top-k requires two passes over the data. This optimization fuses both operations, computing softmax only for the selected experts.
- Code excerpt:
    ```cpp
    // Fused topk + softmax kernel
    template <typename T, int EXPERTS_PER_BLOCK>
    __global__ void topk_softmax_kernel(
        T* __restrict__ topk_weights,     // [num_tokens, top_k]
        int* __restrict__ topk_indices,   // [num_tokens, top_k]
        const T* __restrict__ gating,     // [num_tokens, num_experts]
        int num_tokens, int num_experts, int top_k) {
      
      const int token_idx = blockIdx.x;
      const T* token_gating = gating + token_idx * num_experts;
      
      // Load all expert scores into registers/shared memory
      __shared__ T scores[EXPERTS_PER_BLOCK];
      __shared__ int indices[EXPERTS_PER_BLOCK];
      
      // Initialize with expert indices
      for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        scores[e] = token_gating[e];
        indices[e] = e;
      }
      __syncthreads();
      
      // Parallel partial sort to find top-k
      // Using bitonic sort for small k
      for (int k = 0; k < top_k; k++) {
        // Find max in remaining elements
        int max_idx = k;
        T max_val = scores[k];
        for (int e = k + 1 + threadIdx.x; e < num_experts; e += blockDim.x) {
          if (scores[e] > max_val) {
            max_val = scores[e];
            max_idx = e;
          }
        }
        // Reduce to find global max
        // ... warp reduction ...
        
        // Swap max to position k
        if (threadIdx.x == 0 && max_idx != k) {
          T tmp_score = scores[k];
          int tmp_idx = indices[k];
          scores[k] = scores[max_idx];
          indices[k] = indices[max_idx];
          scores[max_idx] = tmp_score;
          indices[max_idx] = tmp_idx;
        }
        __syncthreads();
      }
      
      // Compute softmax only over top-k
      T max_score = scores[0];
      for (int k = 1; k < top_k; k++) {
        max_score = max(max_score, scores[k]);
      }
      
      T sum_exp = 0;
      for (int k = 0; k < top_k; k++) {
        scores[k] = exp(scores[k] - max_score);
        sum_exp += scores[k];
      }
      
      // Normalize and store
      for (int k = threadIdx.x; k < top_k; k += blockDim.x) {
        topk_weights[token_idx * top_k + k] = scores[k] / sum_exp;
        topk_indices[token_idx * top_k + k] = indices[k];
      }
    }
    ```
- Evidence mapping:
  - "Fused operations" → TopK and softmax in single kernel
  - "Partial sort" → Only sort top-k elements, not all experts
  - "Selective softmax" → Softmax computed only over selected experts

## Optimization 2: Grouped TopK for DeepSeek
- Commit ID: 8a3cd90af
- Optimization type: Compute
- Summary: Add fused grouped_topk kernel for DeepSeek-style expert grouping
- Detailed explanation:
  DeepSeek models use grouped expert selection where experts are divided into groups and top-k is selected within each group. This optimization provides a fused kernel for this pattern.
- Code excerpt:
    ```cpp
    // Grouped topk for DeepSeek
    template <typename T>
    __global__ void grouped_topk_kernel(
        T* __restrict__ topk_weights,
        int* __restrict__ topk_indices,
        const T* __restrict__ gating,
        int num_tokens, int num_experts, int num_groups,
        int top_k_per_group, int top_k_total) {
      
      const int token_idx = blockIdx.x;
      const int experts_per_group = num_experts / num_groups;
      
      // Select top-k from each group
      int selected_count = 0;
      T selected_weights[MAX_TOP_K];
      int selected_indices[MAX_TOP_K];
      
      for (int g = 0; g < num_groups; g++) {
        int group_start = g * experts_per_group;
        
        // Find top-k in this group
        for (int k = 0; k < top_k_per_group && selected_count < top_k_total; k++) {
          T max_val = -INFINITY;
          int max_idx = -1;
          
          for (int e = group_start; e < group_start + experts_per_group; e++) {
            T val = gating[token_idx * num_experts + e];
            // Skip already selected
            bool already_selected = false;
            for (int s = 0; s < selected_count; s++) {
              if (selected_indices[s] == e) {
                already_selected = true;
                break;
              }
            }
            if (!already_selected && val > max_val) {
              max_val = val;
              max_idx = e;
            }
          }
          
          if (max_idx >= 0) {
            selected_weights[selected_count] = max_val;
            selected_indices[selected_count] = max_idx;
            selected_count++;
          }
        }
      }
      
      // Softmax over selected experts
      // ... normalize selected_weights ...
      
      // Store results
      for (int k = 0; k < top_k_total; k++) {
        topk_weights[token_idx * top_k_total + k] = selected_weights[k];
        topk_indices[token_idx * top_k_total + k] = selected_indices[k];
      }
    }
    ```
- Evidence mapping:
  - "Grouped selection" → Loop over `num_groups`
  - "Per-group top-k" → `top_k_per_group` selected from each group
  - "DeepSeek pattern" → Matches DeepSeek-V2/V3 routing

## Optimization 3: Optimized moe_align_block_size
- Commit ID: 95460fc51, 2344192a5
- Optimization type: Memory / Compute
- Summary: Port and optimize moe_align_block_size kernel from SGLang
- Detailed explanation:
  The align_block_size operation pads token assignments so each expert processes a multiple of the GEMM block size. This optimization improves the kernel's efficiency for large numbers of experts (like DeepSeek's 256 experts).
- Code excerpt:
    ```cpp
    // Optimized moe_align_block_size
    __global__ void moe_align_block_size_kernel(
        int* __restrict__ sorted_token_ids,
        int* __restrict__ expert_ids,
        int* __restrict__ num_tokens_post_pad,
        const int* __restrict__ topk_ids,
        int num_tokens, int num_experts, int block_size, int top_k) {
      
      // Shared memory for per-expert token counts
      extern __shared__ int expert_counts[];
      
      // Initialize counts
      for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        expert_counts[e] = 0;
      }
      __syncthreads();
      
      // Count tokens per expert
      for (int t = threadIdx.x; t < num_tokens * top_k; t += blockDim.x) {
        int expert = topk_ids[t];
        atomicAdd(&expert_counts[expert], 1);
      }
      __syncthreads();
      
      // Compute padded counts and offsets
      __shared__ int expert_offsets[MAX_EXPERTS + 1];
      if (threadIdx.x == 0) {
        int offset = 0;
        for (int e = 0; e < num_experts; e++) {
          expert_offsets[e] = offset;
          // Pad to block_size
          int padded = ((expert_counts[e] + block_size - 1) / block_size) * block_size;
          offset += padded;
          
          // Fill expert_ids for this expert's blocks
          for (int b = 0; b < padded / block_size; b++) {
            expert_ids[expert_offsets[e] / block_size + b] = e;
          }
        }
        expert_offsets[num_experts] = offset;
        *num_tokens_post_pad = offset;
      }
      __syncthreads();
      
      // Scatter tokens to sorted positions
      // Reset counts for use as insertion indices
      for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        expert_counts[e] = 0;
      }
      __syncthreads();
      
      for (int t = threadIdx.x; t < num_tokens * top_k; t += blockDim.x) {
        int expert = topk_ids[t];
        int pos = atomicAdd(&expert_counts[expert], 1);
        sorted_token_ids[expert_offsets[expert] + pos] = t;
      }
    }
    ```
- Evidence mapping:
  - "Atomic counting" → `atomicAdd` for parallel token counting
  - "Block padding" → `((count + block_size - 1) / block_size) * block_size`
  - "Scatter operation" → Tokens placed at computed offsets

## Optimization 4: Sigmoid + Bias Fusion for Grouped TopK
- Commit ID: 085252764
- Optimization type: Fusion
- Summary: Fuse sigmoid activation and bias addition into grouped_topk kernel
- Detailed explanation:
  Some MoE models use sigmoid instead of softmax for gating, with an optional bias term. This optimization fuses these operations into the topk kernel.
- Code excerpt:
    ```cpp
    // Grouped topk with fused sigmoid and bias
    template <typename T, bool USE_SIGMOID, bool HAS_BIAS>
    __global__ void grouped_topk_sigmoid_bias_kernel(
        T* __restrict__ topk_weights,
        int* __restrict__ topk_indices,
        const T* __restrict__ gating,
        const T* __restrict__ bias,  // Optional bias [num_experts]
        int num_tokens, int num_experts, int top_k) {
      
      const int token_idx = blockIdx.x;
      
      // Load and transform gating scores
      T scores[MAX_EXPERTS];
      for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        T score = gating[token_idx * num_experts + e];
        
        // Apply bias if present
        if constexpr (HAS_BIAS) {
          score += bias[e];
        }
        
        // Apply sigmoid if requested (instead of softmax)
        if constexpr (USE_SIGMOID) {
          score = T(1.0) / (T(1.0) + exp(-score));
        }
        
        scores[e] = score;
      }
      
      // Select top-k
      // ... top-k selection ...
      
      // For sigmoid, optionally renormalize
      if constexpr (USE_SIGMOID) {
        T sum = 0;
        for (int k = 0; k < top_k; k++) {
          sum += selected_weights[k];
        }
        for (int k = 0; k < top_k; k++) {
          selected_weights[k] /= sum;
        }
      }
    }
    ```
- Evidence mapping:
  - "Fused sigmoid" → `USE_SIGMOID` template parameter
  - "Fused bias" → `HAS_BIAS` template parameter
  - "Compile-time dispatch" → `if constexpr` for zero overhead

## Optimization 5: Permute/Unpermute Kernels
- Commit ID: 3e887d2e0
- Optimization type: Memory
- Summary: Add optimized permute/unpermute kernels for MoE token reordering
- Detailed explanation:
  After expert selection, tokens need to be reordered so each expert's tokens are contiguous. After expert computation, tokens need to be unpermuted back to original order. These kernels optimize this reordering.
- Code excerpt:
    ```cpp
    // Optimized permute kernel
    template <typename T, int VEC_SIZE>
    __global__ void moe_permute_kernel(
        T* __restrict__ permuted_tokens,      // [num_tokens * top_k, hidden]
        const T* __restrict__ tokens,         // [num_tokens, hidden]
        const int* __restrict__ sorted_ids,   // [num_tokens * top_k]
        int num_tokens, int hidden_size, int top_k) {
      
      using VecType = typename Vec<T, VEC_SIZE>::Type;
      
      const int sorted_idx = blockIdx.x;
      const int original_idx = sorted_ids[sorted_idx] / top_k;  // Map back to original token
      
      // Vectorized copy
      const VecType* src = reinterpret_cast<const VecType*>(
          tokens + original_idx * hidden_size);
      VecType* dst = reinterpret_cast<VecType*>(
          permuted_tokens + sorted_idx * hidden_size);
      
      for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
        dst[i] = src[i];
      }
    }
    
    // Optimized unpermute with reduction
    template <typename T, int VEC_SIZE>
    __global__ void moe_unpermute_kernel(
        T* __restrict__ output,               // [num_tokens, hidden]
        const T* __restrict__ expert_output,  // [num_tokens * top_k, hidden]
        const int* __restrict__ sorted_ids,
        const T* __restrict__ topk_weights,
        int num_tokens, int hidden_size, int top_k) {
      
      const int token_idx = blockIdx.x;
      
      // Accumulate contributions from all top-k experts
      for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        T sum = 0;
        for (int k = 0; k < top_k; k++) {
          int sorted_idx = find_sorted_idx(sorted_ids, token_idx, k);
          T weight = topk_weights[token_idx * top_k + k];
          sum += expert_output[sorted_idx * hidden_size + d] * weight;
        }
        output[token_idx * hidden_size + d] = sum;
      }
    }
    ```
- Evidence mapping:
  - "Vectorized copy" → `VecType` for efficient memory access
  - "Index mapping" → `sorted_ids[sorted_idx] / top_k` for original token
  - "Weighted reduction" → Unpermute includes routing weight multiplication
