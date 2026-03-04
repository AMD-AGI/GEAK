---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Token Sampling Kernels

## Variant Context
- Input semantic type: Token sampling from logits distribution
- Datatype(s): fp32 (logits), int32/int64 (token indices)
- Data representation: Dense logit tensors
- Target architecture: Generic CUDA/HIP, Triton

## Functionality
The sampling kernels implement various token sampling strategies for LLM inference:
1. Greedy sampling (argmax)
2. Top-k sampling
3. Top-p (nucleus) sampling
4. Temperature scaling
5. Repetition/presence penalties
6. Min-p sampling

## Optimization 1: Fused Top-K/Top-P Sampling
- Commit ID: (csrc/sampler.cu)
- Optimization type: Fusion
- Summary: Fuse temperature scaling, top-k filtering, top-p filtering, and sampling
- Detailed explanation:
  Instead of multiple kernel launches for each sampling step, this optimization fuses all operations into a single kernel, reducing memory traffic and launch overhead.
- Code excerpt:
    ```cpp
    template <typename scalar_t, int BLOCK_SIZE>
    __global__ void fused_sampling_kernel(
        int64_t* __restrict__ output_tokens,
        const scalar_t* __restrict__ logits,
        const float* __restrict__ temperatures,
        const int* __restrict__ top_ks,
        const float* __restrict__ top_ps,
        curandState* __restrict__ rand_states,
        int vocab_size, int batch_size) {
      
      const int batch_idx = blockIdx.x;
      const scalar_t* batch_logits = logits + batch_idx * vocab_size;
      
      // Apply temperature
      float temp = temperatures[batch_idx];
      __shared__ float scaled_logits[MAX_VOCAB_SIZE];
      for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        scaled_logits[i] = static_cast<float>(batch_logits[i]) / temp;
      }
      __syncthreads();
      
      // Find top-k
      int top_k = top_ks[batch_idx];
      float top_k_threshold = find_kth_largest(scaled_logits, vocab_size, top_k);
      
      // Apply top-k mask and compute softmax
      float max_val = -INFINITY;
      for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        if (scaled_logits[i] < top_k_threshold) {
          scaled_logits[i] = -INFINITY;
        }
        max_val = fmaxf(max_val, scaled_logits[i]);
      }
      max_val = block_reduce_max(max_val);
      
      // Softmax and top-p filtering
      float sum_exp = 0.0f;
      float top_p = top_ps[batch_idx];
      // ... compute cumulative probability and filter ...
      
      // Sample from filtered distribution
      float rand_val = curand_uniform(&rand_states[batch_idx]);
      int sampled_token = sample_from_distribution(scaled_logits, rand_val);
      
      if (threadIdx.x == 0) {
        output_tokens[batch_idx] = sampled_token;
      }
    }
    ```
- Evidence mapping:
  - "Fused operations" → Temperature, top-k, top-p, sampling in one kernel
  - "Shared memory" → `scaled_logits` avoids repeated global loads
  - "Per-batch parameters" → Different settings per sequence

## Optimization 2: Triton Min-P Sampling
- Commit ID: (vllm/v1/worker/gpu/sample/min_p.py)
- Optimization type: Compute
- Summary: Implement min-p sampling strategy in Triton
- Detailed explanation:
  Min-p sampling keeps tokens with probability >= min_p * max_probability. This provides adaptive filtering based on the confidence of the top prediction.
- Code excerpt:
    ```python
    @triton.jit
    def min_p_sampling_kernel(
        logits_ptr, output_ptr, min_p_ptr,
        vocab_size, batch_size,
        BLOCK_SIZE: tl.constexpr
    ):
        batch_idx = tl.program_id(0)
        
        # Load logits for this batch
        offs = tl.arange(0, BLOCK_SIZE)
        logits = tl.load(logits_ptr + batch_idx * vocab_size + offs,
                         mask=offs < vocab_size, other=-float('inf'))
        
        # Find max logit
        max_logit = tl.max(logits, axis=0)
        
        # Compute softmax
        exp_logits = tl.exp(logits - max_logit)
        sum_exp = tl.sum(exp_logits, axis=0)
        probs = exp_logits / sum_exp
        
        # Apply min-p threshold
        min_p = tl.load(min_p_ptr + batch_idx)
        max_prob = tl.max(probs, axis=0)
        threshold = min_p * max_prob
        
        # Mask tokens below threshold
        masked_probs = tl.where(probs >= threshold, probs, 0.0)
        
        # Renormalize and sample
        masked_sum = tl.sum(masked_probs, axis=0)
        normalized_probs = masked_probs / masked_sum
        
        # ... sampling logic ...
    ```
- Evidence mapping:
  - "Min-p threshold" → `threshold = min_p * max_prob`
  - "Adaptive filtering" → Threshold scales with confidence
  - "Renormalization" → Probabilities renormalized after filtering

## Optimization 3: Penalty Application Kernels
- Commit ID: (vllm/v1/worker/gpu/sample/penalties.py)
- Optimization type: Compute
- Summary: Efficient repetition and presence penalty application
- Detailed explanation:
  Repetition and presence penalties discourage the model from repeating tokens. This kernel efficiently applies these penalties by tracking token occurrences.
- Code excerpt:
    ```python
    @triton.jit
    def apply_penalties_kernel(
        logits_ptr, token_ids_ptr, token_counts_ptr,
        repetition_penalty, presence_penalty,
        vocab_size, seq_len,
        BLOCK_SIZE: tl.constexpr
    ):
        batch_idx = tl.program_id(0)
        
        # Load token occurrence counts
        for i in range(seq_len):
            token_id = tl.load(token_ids_ptr + batch_idx * seq_len + i)
            count = tl.load(token_counts_ptr + batch_idx * vocab_size + token_id)
            tl.store(token_counts_ptr + batch_idx * vocab_size + token_id, count + 1)
        
        # Apply penalties to logits
        offs = tl.arange(0, BLOCK_SIZE)
        for start in range(0, vocab_size, BLOCK_SIZE):
            idx = start + offs
            mask = idx < vocab_size
            
            logits = tl.load(logits_ptr + batch_idx * vocab_size + idx, mask=mask)
            counts = tl.load(token_counts_ptr + batch_idx * vocab_size + idx, mask=mask)
            
            # Apply repetition penalty (multiplicative)
            rep_penalty = tl.where(counts > 0, repetition_penalty, 1.0)
            logits = tl.where(logits > 0, logits / rep_penalty, logits * rep_penalty)
            
            # Apply presence penalty (additive)
            pres_penalty = tl.where(counts > 0, presence_penalty, 0.0)
            logits = logits - pres_penalty
            
            tl.store(logits_ptr + batch_idx * vocab_size + idx, logits, mask=mask)
    ```
- Evidence mapping:
  - "Token counting" → Track occurrences in `token_counts_ptr`
  - "Repetition penalty" → Multiplicative penalty based on count
  - "Presence penalty" → Additive penalty for any occurrence
