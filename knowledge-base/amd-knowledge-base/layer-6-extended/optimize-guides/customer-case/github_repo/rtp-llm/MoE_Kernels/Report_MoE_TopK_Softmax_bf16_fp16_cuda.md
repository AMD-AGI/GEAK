---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: MoE TopK Softmax and Routing Kernels

## Variant Context
- Input semantic type: MoE routing (expert selection and token dispatch)
- Datatype(s): bf16, fp16, fp32
- Data representation: Router logits to expert assignments
- Target architecture: CUDA (SM70+), ROCm (gfx942)

## Functionality
These kernels implement the routing logic for Mixture of Experts models:
- **TopK Softmax**: Select top-K experts for each token
- **Token Permutation**: Reorder tokens by expert assignment
- **Expert Load Balancing**: Compute auxiliary losses for training

The MoE routing is critical for:
- Sparse activation of experts
- Efficient expert parallelism
- Load balancing across experts

## Optimization 1: Fused TopK with Softmax
- Commit ID: 205df6f01
- Optimization type: Fusion
- Summary: Fuse top-K selection with softmax computation
- Detailed explanation:
  The kernel combines multiple operations:
  - Compute softmax over router logits
  - Select top-K experts per token
  - Compute routing weights
  - All in single kernel pass

- Code excerpt:
    ```cpp
    // From moe_topk_softmax_kernels.cu
    template<typename T, int K>
    __global__ void topKSoftmaxKernel(
        int* expert_ids, float* expert_weights,
        const T* router_logits,
        int batch_size, int num_experts) {
        
        int token_idx = blockIdx.x;
        int tid = threadIdx.x;
        
        // Load logits into shared memory
        extern __shared__ float smem[];
        float* logits = smem;
        
        for (int i = tid; i < num_experts; i += blockDim.x) {
            logits[i] = static_cast<float>(router_logits[token_idx * num_experts + i]);
        }
        __syncthreads();
        
        // Find top-K using partial sort
        float top_k_vals[K];
        int top_k_ids[K];
        
        #pragma unroll
        for (int k = 0; k < K; k++) {
            float max_val = -INFINITY;
            int max_idx = -1;
            
            for (int i = tid; i < num_experts; i += blockDim.x) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    max_idx = i;
                }
            }
            
            // Warp reduction to find global max
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                float other_val = __shfl_xor_sync(0xffffffff, max_val, mask);
                int other_idx = __shfl_xor_sync(0xffffffff, max_idx, mask);
                if (other_val > max_val) {
                    max_val = other_val;
                    max_idx = other_idx;
                }
            }
            
            if (tid == 0) {
                top_k_vals[k] = max_val;
                top_k_ids[k] = max_idx;
                logits[max_idx] = -INFINITY;  // Mask selected expert
            }
            __syncthreads();
        }
        
        // Compute softmax over top-K
        if (tid == 0) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < K; k++) {
                top_k_vals[k] = expf(top_k_vals[k]);
                sum += top_k_vals[k];
            }
            
            #pragma unroll
            for (int k = 0; k < K; k++) {
                expert_ids[token_idx * K + k] = top_k_ids[k];
                expert_weights[token_idx * K + k] = top_k_vals[k] / sum;
            }
        }
    }
    ```
- Evidence mapping:
  - Fused operations → TopK + softmax in single kernel
  - Warp reduction → `__shfl_xor_sync` for finding max
  - Template K → Compile-time unrolling for common K values

## Optimization 2: Token Permutation for Expert Batching
- Commit ID: 205df6f01
- Optimization type: Memory
- Summary: Efficient token reordering for batched expert computation
- Detailed explanation:
  After routing, tokens need to be grouped by expert:
  - Compute prefix sums for expert offsets
  - Permute tokens to contiguous expert groups
  - Support for variable tokens per expert

- Code excerpt:
    ```cpp
    // Token permutation kernel
    template<typename T>
    __global__ void permuteTokensKernel(
        T* permuted_tokens,
        int* permuted_indices,
        const T* tokens,
        const int* expert_ids,
        const int* expert_offsets,
        int batch_size, int hidden_size, int top_k) {
        
        int token_idx = blockIdx.x;
        int tid = threadIdx.x;
        
        // For each expert assignment
        for (int k = 0; k < top_k; k++) {
            int expert_id = expert_ids[token_idx * top_k + k];
            int offset = atomicAdd(&expert_offsets[expert_id], 1);
            
            // Copy token to permuted location
            for (int i = tid; i < hidden_size; i += blockDim.x) {
                permuted_tokens[offset * hidden_size + i] = 
                    tokens[token_idx * hidden_size + i];
            }
            
            if (tid == 0) {
                permuted_indices[offset] = token_idx * top_k + k;
            }
        }
    }
    ```
- Evidence mapping:
  - Atomic offset → `atomicAdd(&expert_offsets[expert_id], 1)`
  - Token copy → Vectorized copy to permuted location
  - Index tracking → `permuted_indices` for unpermutation

## Optimization 3: Expert Load Balancing Loss
- Commit ID: (core implementation)
- Optimization type: Compute
- Summary: Efficient computation of auxiliary load balancing loss
- Detailed explanation:
  MoE models use auxiliary losses for load balancing:
  - Compute fraction of tokens per expert
  - Compute average routing probability per expert
  - Multiply for load balancing loss

- Code excerpt:
    ```cpp
    // Load balancing loss kernel
    template<typename T>
    __global__ void loadBalancingLossKernel(
        float* loss,
        float* expert_counts,
        float* expert_probs,
        const int* expert_ids,
        const float* expert_weights,
        int batch_size, int num_experts, int top_k) {
        
        int expert_id = blockIdx.x;
        int tid = threadIdx.x;
        
        // Count tokens assigned to this expert
        int count = 0;
        float prob_sum = 0.0f;
        
        for (int i = tid; i < batch_size * top_k; i += blockDim.x) {
            if (expert_ids[i] == expert_id) {
                count++;
                prob_sum += expert_weights[i];
            }
        }
        
        // Reduce across threads
        count = blockReduceSum(count);
        prob_sum = blockReduceSum(prob_sum);
        
        if (tid == 0) {
            float f_i = static_cast<float>(count) / (batch_size * top_k);
            float p_i = prob_sum / batch_size;
            
            expert_counts[expert_id] = f_i;
            expert_probs[expert_id] = p_i;
            
            atomicAdd(loss, num_experts * f_i * p_i);
        }
    }
    ```
- Evidence mapping:
  - Token counting → Loop with expert_id comparison
  - Block reduction → `blockReduceSum` for aggregation
  - Loss computation → `num_experts * f_i * p_i` formula

## Optimization 4: Grouped GEMM Dispatch for Experts
- Commit ID: 32cf02375
- Optimization type: Compute
- Summary: Efficient dispatch of grouped GEMM for expert computation
- Detailed explanation:
  After permutation, expert computation uses grouped GEMM:
  - Each expert has different number of tokens
  - Grouped GEMM handles variable batch sizes
  - Supports FP8 and quantized weights

- Code excerpt:
    ```cpp
    // Expert GEMM dispatch
    void dispatchExpertGemm(
        void* output,
        const void* input,
        const void* weights,
        const int* expert_offsets,
        const int* problem_sizes,
        int num_experts, int hidden_size, int intermediate_size,
        cudaStream_t stream) {
        
        // Build problem descriptors
        std::vector<cutlass::gemm::GemmCoord> problems;
        for (int e = 0; e < num_experts; e++) {
            int m = problem_sizes[e];  // Tokens for this expert
            int n = intermediate_size;
            int k = hidden_size;
            problems.push_back({m, n, k});
        }
        
        // Launch grouped GEMM
        groupedGemm(output, input, weights, problems, stream);
    }
    ```
- Evidence mapping:
  - Variable batch sizes → `problem_sizes[e]` per expert
  - Grouped GEMM → Single kernel for all experts
  - Offset-based addressing → `expert_offsets` for data location
