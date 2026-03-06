Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: MoE TopK Routing Kernels

## Variant Context
- Input semantic type: Expert selection for Mixture of Experts routing
- Datatype(s): FP32 (routing logits), INT32 (expert indices)
- Data representation: Router logits, top-k indices and weights
- Target architecture: Generic CUDA

## Functionality
These kernels implement efficient top-k expert selection for MoE models, supporting both softmax and sigmoid-based routing mechanisms used in different MoE architectures.

## Optimization 1: Fused TopK with Softmax
- Commit ID: (moe_topk_softmax_kernels.cu)
- Optimization type: Fusion
- Summary: Fuses top-k selection with softmax normalization for efficient MoE routing.

- Detailed explanation:
  MoE routing requires:
  1. Computing softmax over expert logits
  2. Selecting top-k experts
  3. Renormalizing weights for selected experts
  
  This kernel fuses all operations, avoiding multiple passes over the data.

- Code excerpt:
    ```cpp
    __global__ void moe_topk_softmax_kernel(
        const float* __restrict__ logits,  // [batch, num_experts]
        float* __restrict__ weights,        // [batch, topk]
        int* __restrict__ indices,          // [batch, topk]
        int batch_size,
        int num_experts,
        int topk
    ) {
        int batch_idx = blockIdx.x;
        
        // Load logits to shared memory
        extern __shared__ float shared_logits[];
        for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
            shared_logits[i] = logits[batch_idx * num_experts + i];
        }
        __syncthreads();
        
        // Find top-k using parallel reduction
        float top_vals[MAX_TOPK];
        int top_idxs[MAX_TOPK];
        
        for (int k = 0; k < topk; k++) {
            // Find max (excluding already selected)
            float max_val = -INFINITY;
            int max_idx = -1;
            for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
                if (!is_selected(i, top_idxs, k)) {
                    if (shared_logits[i] > max_val) {
                        max_val = shared_logits[i];
                        max_idx = i;
                    }
                }
            }
            // Reduce to find global max
            // ...
            top_vals[k] = max_val;
            top_idxs[k] = max_idx;
        }
        
        // Compute softmax over selected experts
        float sum_exp = 0.0f;
        for (int k = 0; k < topk; k++) {
            top_vals[k] = expf(top_vals[k] - top_vals[0]);  // Subtract max for stability
            sum_exp += top_vals[k];
        }
        
        // Store normalized weights and indices
        for (int k = 0; k < topk; k++) {
            weights[batch_idx * topk + k] = top_vals[k] / sum_exp;
            indices[batch_idx * topk + k] = top_idxs[k];
        }
    }
    ```

- Evidence mapping:
  - "Fused operations" → top-k and softmax in single kernel
  - "Shared memory" → `extern __shared__ float shared_logits[]`
  - "Renormalization" → `top_vals[k] / sum_exp`

## Optimization 2: TopK with Sigmoid for Auxiliary Loss
- Commit ID: (moe_topk_sigmoid_kernels.cu)
- Optimization type: Compute
- Summary: Implements sigmoid-based routing with auxiliary load balancing loss computation.

- Detailed explanation:
  Some MoE architectures (like DeepSeek) use sigmoid-based routing with auxiliary losses for load balancing. This kernel computes both the routing weights and the auxiliary loss terms efficiently.

- Code excerpt:
    ```cpp
    __global__ void moe_topk_sigmoid_kernel(
        const float* __restrict__ logits,
        float* __restrict__ weights,
        int* __restrict__ indices,
        float* __restrict__ aux_loss,  // Auxiliary load balancing loss
        int batch_size,
        int num_experts,
        int topk
    ) {
        int batch_idx = blockIdx.x;
        
        // Apply sigmoid to logits
        float sigmoid_vals[MAX_EXPERTS];
        for (int i = 0; i < num_experts; i++) {
            sigmoid_vals[i] = 1.0f / (1.0f + expf(-logits[batch_idx * num_experts + i]));
        }
        
        // Find top-k by sigmoid value
        // ... top-k selection ...
        
        // Compute auxiliary loss (load balancing)
        float expert_load[MAX_EXPERTS] = {0};
        // Accumulate across batch using atomics
        for (int k = 0; k < topk; k++) {
            atomicAdd(&expert_load[top_idxs[k]], 1.0f);
        }
        
        // Compute variance-based load balancing loss
        float mean_load = batch_size * topk / num_experts;
        float loss = 0.0f;
        for (int i = 0; i < num_experts; i++) {
            float diff = expert_load[i] - mean_load;
            loss += diff * diff;
        }
        atomicAdd(aux_loss, loss);
    }
    ```

- Evidence mapping:
  - "Sigmoid routing" → `1.0f / (1.0f + expf(-logits[...]))`
  - "Auxiliary loss" → `aux_loss` output for load balancing
  - "Expert load tracking" → `atomicAdd(&expert_load[...], 1.0f)`

## Optimization 3: Fused Gate Computation
- Commit ID: (moe_fused_gate.cu)
- Optimization type: Fusion
- Summary: Fuses the entire gating computation including normalization and expert selection.

- Detailed explanation:
  This kernel fuses:
  1. Router linear projection (hidden_states @ router_weight)
  2. Normalization (softmax or sigmoid)
  3. Top-k selection
  4. Weight computation

- Code excerpt:
    ```cpp
    __global__ void moe_fused_gate_kernel(
        const half* __restrict__ hidden_states,  // [batch, hidden]
        const half* __restrict__ router_weight,  // [num_experts, hidden]
        float* __restrict__ weights,
        int* __restrict__ indices,
        int batch_size,
        int hidden_size,
        int num_experts,
        int topk
    ) {
        // Compute router logits: hidden @ router_weight.T
        float logits[MAX_EXPERTS];
        for (int e = 0; e < num_experts; e++) {
            float sum = 0.0f;
            for (int h = 0; h < hidden_size; h++) {
                sum += __half2float(hidden_states[...]) * 
                       __half2float(router_weight[e * hidden_size + h]);
            }
            logits[e] = sum;
        }
        
        // Top-k selection and normalization
        // ... (as above)
    }
    ```

- Evidence mapping:
  - "Fused projection" → router GEMM inside gating kernel
  - "End-to-end fusion" → hidden_states to final weights/indices
