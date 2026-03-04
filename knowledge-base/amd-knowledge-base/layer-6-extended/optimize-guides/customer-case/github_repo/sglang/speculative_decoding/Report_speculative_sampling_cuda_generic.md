---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Speculative Sampling Kernel

## Variant Context
- Input semantic type: Token verification and acceptance for speculative decoding
- Datatype(s): FP32 (probabilities), INT32/INT64 (token IDs)
- Data representation: Draft and target probability distributions
- Target architecture: Generic CUDA

## Functionality
This kernel implements the speculative sampling algorithm for speculative decoding, where:
1. A smaller draft model proposes multiple tokens
2. The target model verifies these tokens in parallel
3. Tokens are accepted/rejected based on probability comparison
4. The kernel determines the acceptance length and samples the next token

## Optimization 1: Fused Acceptance and Sampling
- Commit ID: (Initial implementation in sgl-kernel)
- Optimization type: Fusion
- Summary: Fuses the token acceptance check and next-token sampling into a single kernel, avoiding multiple kernel launches.

- Detailed explanation:
  Speculative decoding requires:
  1. Computing acceptance probability: min(1, p_target / p_draft)
  2. Sampling from uniform distribution for acceptance decision
  3. Finding the first rejected position
  4. Sampling the next token from adjusted distribution
  
  This kernel fuses all these operations, reducing kernel launch overhead and memory traffic.

- Code excerpt:
    ```cpp
    __global__ void speculative_sampling_kernel(
        const float* target_probs,    // [batch, seq_len, vocab]
        const float* draft_probs,     // [batch, seq_len, vocab]
        const int* draft_tokens,      // [batch, seq_len]
        const float* uniform_samples, // [batch, seq_len]
        int* accepted_lengths,        // [batch]
        int* next_tokens,             // [batch]
        // ...
    ) {
        int batch_idx = blockIdx.x;
        
        // Find first rejection point
        int accept_len = 0;
        for (int i = 0; i < seq_len; i++) {
            int token = draft_tokens[batch_idx * seq_len + i];
            float p_target = target_probs[...];
            float p_draft = draft_probs[...];
            float accept_prob = min(1.0f, p_target / p_draft);
            
            if (uniform_samples[batch_idx * seq_len + i] < accept_prob) {
                accept_len++;
            } else {
                break;
            }
        }
        
        // Sample next token from adjusted distribution
        // p_adjusted = max(0, p_target - p_draft) / sum(max(0, p_target - p_draft))
        // ...
    }
    ```

- Evidence mapping:
  - "Fused operations" → acceptance check and sampling in same kernel
  - "Single pass" → sequential acceptance check with early termination
  - "Adjusted distribution" → `max(0, p_target - p_draft)` for rejection sampling
