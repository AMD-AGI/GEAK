Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Top-K Mixture of Experts (TopK-MoE)

## Variant Context
- Input semantic type: Expert routing for MoE models
- Datatype(s): FP32 (router logits), INT32 (expert indices)
- Data representation: Dense router scores, sparse expert selection
- Target architecture: Generic (NVIDIA, AMD)

## Functionality
The TopK-MoE kernel implements the expert routing mechanism for Mixture of Experts models. It computes softmax over router logits, selects the top-K experts for each token, and prepares the routing weights. This is a critical kernel for models like Mixtral, DeepSeek-MoE, and GPT-OSS.

Key features:
- Fused softmax and top-K selection
- Support for various K values (typically 2-8)
- Optional weight clamping and normalization
- Efficient register-based implementation

---

## Optimization 1: Fused Top-K MoE Kernel
- Commit ID: 077c94d0c
- Optimization type: Fusion (kernel fusion)
- Summary: Fuse softmax, top-K selection, and weight normalization into a single kernel
- Detailed explanation: The MoE routing involves multiple steps: softmax over expert logits, selecting top-K experts, and normalizing the selected weights. By fusing these operations, we avoid multiple kernel launches and intermediate memory writes.

- Code excerpt:
    ```cpp
    // CUDA: add a fused top-K MoE kernel
    template<int K, int N_EXPERTS>
    __global__ void topk_moe_fused(
        const float * __restrict__ router_logits,  // [batch, n_experts]
        float * __restrict__ weights,              // [batch, K]
        int32_t * __restrict__ indices,            // [batch, K]
        const int batch_size) {
        
        const int token = blockIdx.x;
        if (token >= batch_size) return;
        
        const float * logits = router_logits + token * N_EXPERTS;
        
        // Step 1: Find max for numerical stability
        float max_val = -INFINITY;
        for (int i = threadIdx.x; i < N_EXPERTS; i += blockDim.x) {
            max_val = fmaxf(max_val, logits[i]);
        }
        max_val = warp_reduce_max(max_val);
        
        // Step 2: Compute softmax and track top-K
        float top_vals[K];
        int top_ids[K];
        for (int k = 0; k < K; k++) {
            top_vals[k] = -INFINITY;
            top_ids[k] = -1;
        }
        
        float sum_exp = 0.0f;
        for (int i = threadIdx.x; i < N_EXPERTS; i += blockDim.x) {
            float val = expf(logits[i] - max_val);
            sum_exp += val;
            
            // Insert into top-K if larger than minimum
            if (val > top_vals[K-1]) {
                insert_topk(top_vals, top_ids, val, i, K);
            }
        }
        sum_exp = warp_reduce_sum(sum_exp);
        
        // Step 3: Normalize and output
        if (threadIdx.x == 0) {
            float weight_sum = 0.0f;
            for (int k = 0; k < K; k++) {
                top_vals[k] /= sum_exp;
                weight_sum += top_vals[k];
            }
            // Renormalize selected weights
            for (int k = 0; k < K; k++) {
                weights[token * K + k] = top_vals[k] / weight_sum;
                indices[token * K + k] = top_ids[k];
            }
        }
    }
    ```

- Evidence mapping:
  - "Fused operations" → softmax, top-K, normalize in one kernel
  - "Single pass" → track top-K during softmax computation
  - "No intermediate buffers" → direct output of weights and indices

---

## Optimization 2: Register-Based Top-K Tracking
- Commit ID: 38355c6c8
- Optimization type: Memory (register usage)
- Summary: Use registers instead of shared memory for top-K tracking
- Detailed explanation: For small K values (typically 2-8), the top-K candidates can be stored in registers rather than shared memory. This reduces shared memory pressure and avoids synchronization overhead.

- Code excerpt:
    ```cpp
    // CUDA: use registers instead of smem in topk-moe
    template<int K>
    __device__ __forceinline__ void insert_topk_reg(
        float * __restrict__ vals,  // In registers
        int * __restrict__ ids,     // In registers
        float new_val,
        int new_id) {
        
        // Insertion sort in registers
        #pragma unroll
        for (int k = K - 1; k >= 0; k--) {
            if (new_val > vals[k]) {
                if (k < K - 1) {
                    vals[k + 1] = vals[k];
                    ids[k + 1] = ids[k];
                }
                vals[k] = new_val;
                ids[k] = new_id;
                break;
            }
        }
    }
    
    // Usage in kernel
    float top_vals[K];  // Register array
    int top_ids[K];     // Register array
    
    for (int i = 0; i < N_EXPERTS; i++) {
        float val = softmax_val[i];
        insert_topk_reg<K>(top_vals, top_ids, val, i);
    }
    ```

- Evidence mapping:
  - "Register storage" → `float top_vals[K]` as local arrays
  - "No shared memory" → avoids `__shared__` declaration
  - "Unrolled insertion" → `#pragma unroll` for compile-time K

---

## Optimization 3: Weight Clamping Support
- Commit ID: 75d33b930
- Optimization type: Algorithm (model support)
- Summary: Add support for weight clamping in top-K normalization
- Detailed explanation: Some MoE models clamp the routing weights to prevent any single expert from dominating. This optimization adds optional weight clamping during the normalization step.

- Code excerpt:
    ```cpp
    // CUDA: support for weight clamp in top-k norm
    template<int K>
    __device__ void normalize_weights_clamped(
        float * __restrict__ weights,
        const float clamp_min,
        const float clamp_max) {
        
        float sum = 0.0f;
        
        // First pass: clamp and sum
        #pragma unroll
        for (int k = 0; k < K; k++) {
            weights[k] = fminf(fmaxf(weights[k], clamp_min), clamp_max);
            sum += weights[k];
        }
        
        // Second pass: normalize
        const float inv_sum = 1.0f / sum;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            weights[k] *= inv_sum;
        }
    }
    ```

- Evidence mapping:
  - "Weight clamping" → `fminf(fmaxf(...))` for bounds
  - "Renormalization" → divide by sum after clamping
  - "Model compatibility" → supports various MoE architectures

---

## Optimization 4: GPT-OSS Optional Parameters
- Commit ID: 03792ad93
- Optimization type: Algorithm (model support)
- Summary: Add optional parameters for GPT-OSS model's MoE routing
- Detailed explanation: GPT-OSS uses a slightly different MoE routing scheme with additional parameters. This optimization adds support for these optional parameters while maintaining backward compatibility.

- Code excerpt:
    ```cpp
    // CUDA: topk-moe: add optional parameter for gpt-oss
    template<int K, bool USE_EXP_PROBS>
    __global__ void topk_moe_gptoss(
        const float * __restrict__ router_logits,
        const float * __restrict__ exp_probs,  // Optional: pre-computed expert probs
        float * __restrict__ weights,
        int32_t * __restrict__ indices,
        const int batch_size,
        const int n_experts) {
        
        const int token = blockIdx.x;
        
        if constexpr (USE_EXP_PROBS) {
            // Use pre-computed probabilities
            const float * probs = exp_probs + token * n_experts;
            // Select top-K from probs directly
            ...
        } else {
            // Compute softmax from logits
            const float * logits = router_logits + token * n_experts;
            // Standard softmax + top-K
            ...
        }
    }
    ```

- Evidence mapping:
  - "Optional parameter" → `USE_EXP_PROBS` template flag
  - "GPT-OSS support" → `exp_probs` input for pre-computed probs
  - "Backward compatible" → works with or without exp_probs

---

## Optimization 5: Refactoring for More Models
- Commit ID: 3bcc99099
- Optimization type: Algorithm (generalization)
- Summary: Refactor topk-moe to support more models like GLM 4.7 and Nemotron
- Detailed explanation: Different MoE models have varying numbers of experts and routing schemes. This optimization generalizes the kernel to handle a wider range of configurations through template parameters and runtime dispatch.

- Code excerpt:
    ```cpp
    // CUDA: refactor topk-moe to enable more models
    // Support for GLM 4.7, Nemotron, etc.
    
    template<int K>
    void launch_topk_moe(
        const float * router_logits,
        float * weights,
        int32_t * indices,
        const int batch_size,
        const int n_experts,
        const float clamp_min,
        const float clamp_max,
        cudaStream_t stream) {
        
        // Dispatch based on number of experts
        const int block_size = min(n_experts, 256);
        
        switch (n_experts) {
            case 8:   topk_moe_fused<K, 8><<<batch_size, block_size, 0, stream>>>(...); break;
            case 16:  topk_moe_fused<K, 16><<<batch_size, block_size, 0, stream>>>(...); break;
            case 64:  topk_moe_fused<K, 64><<<batch_size, block_size, 0, stream>>>(...); break;
            case 128: topk_moe_fused<K, 128><<<batch_size, block_size, 0, stream>>>(...); break;
            default:  topk_moe_generic<K><<<batch_size, block_size, 0, stream>>>(...); break;
        }
    }
    
    // Instantiate for common K values
    template void launch_topk_moe<2>(...);  // Mixtral
    template void launch_topk_moe<4>(...);  // Some models
    template void launch_topk_moe<8>(...);  // DeepSeek
    ```

- Evidence mapping:
  - "Multiple expert counts" → switch on `n_experts`
  - "Template instantiation" → K=2,4,8 for different models
  - "Generic fallback" → `topk_moe_generic` for unusual configs
