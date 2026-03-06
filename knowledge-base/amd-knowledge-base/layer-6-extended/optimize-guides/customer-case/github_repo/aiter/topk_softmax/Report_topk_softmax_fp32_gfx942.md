Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: topk_softmax

## Variant Context
- Input semantic type: TopK selection with Softmax for MoE routing
- Datatype(s): FP32/FP16/BF16
- Data representation: Dense logits tensor
- Target architecture: gfx942 (MI300), gfx950 (MI350)

## Functionality
This kernel performs TopK selection followed by Softmax normalization for Mixture of Experts (MoE) routing. It identifies the top-K experts for each token and computes normalized routing weights. The kernel is critical for MoE model performance as it's called for every token.

## Optimization 1: Reduce One Round of TopK Reduction
- Commit ID: 070c532ea
- Optimization type: Compute
- Summary: Combined the first TopK selection with the max reduction for Softmax, eliminating one reduction round.

- Detailed explanation:
  The original implementation performed:
  1. Max reduction for Softmax normalization
  2. Separate TopK argmax reduction
  
  The optimization combines these by tracking both the max value AND its index during the initial max reduction. This eliminates one full reduction round for the first TopK element.

- Code excerpt:
    ```cpp
    // BEFORE: Separate max reduction and argmax
    float thread_max = row_chunk[0];
    for(int ii = 1; ii < VPT; ++ii)
    {
        thread_max = max(thread_max, row_chunk[ii]);
    }
    thread_max = multithread_reduce(thread_max, [](float a, float b) { return max(a, b); }, THREADS_PER_ROW);
    
    // AFTER: Combined max with argmax tracking
    float thread_max = row_chunk[0];
    int first_topk_expert = first_elt_read_by_thread;
    for(int ii = 1; ii < VPT; ++ii)
    {
        if (thread_max < row_chunk[ii])
        {
            thread_max = row_chunk[ii];
            first_topk_expert = first_elt_read_by_thread + ii;
        }
    }
    auto arg_max = [](const kvp& a, const kvp& b) {
        if(a.value > b.value || (a.value == b.value && a.key < b.key))
            return a;
        return b;
    };
    kvp thread_kvp = {first_topk_expert, thread_max};
    thread_kvp = multithread_reduce(thread_kvp, arg_max, THREADS_PER_ROW);
    ```

- Evidence mapping:
  - Combined reduction → `kvp` struct tracks both value and index
  - Eliminated round → First TopK expert found during max reduction
  - Commit message → "reduce one round of topK reduce"

## Optimization 2: Higher Occupancy with Optimized WARPS_PER_TB
- Commit ID: 957bb05ed
- Optimization type: Scheduling
- Summary: Optimized WARPS_PER_TB (warps per thread block) for higher GPU occupancy.

- Detailed explanation:
  The optimization tunes the number of warps per thread block to maximize GPU occupancy while maintaining efficient shared memory usage. This improves parallelism and hides memory latency.

- Code excerpt:
    ```cpp
    // Optimize topksoftmax WARPS_PER_TB for higher occupancy
    // and remove redundant precision conversion
    ```

- Evidence mapping:
  - Occupancy optimization → WARPS_PER_TB tuning
  - Precision optimization → Removed redundant conversions

## Optimization 3: 32-Byte Vector Loads
- Commit ID: 79a951a71
- Optimization type: Memory
- Summary: Implemented 32-byte (256-bit) vector loads for maximum memory bandwidth utilization.

- Detailed explanation:
  AMD GPUs support 256-bit memory transactions. Using 32-byte vector loads (e.g., float8 or 4x float2) maximizes memory bandwidth utilization and reduces the number of load instructions.

- Code excerpt:
    ```cpp
    // top-K-only softmax + 32B vector loads
    ```

- Evidence mapping:
  - 32-byte loads → Commit message mentions "32B vector loads"
  - Memory optimization → Maximizes memory bandwidth

## Optimization 4: BF16 Input Support
- Commit ID: bc362c1ec
- Optimization type: Precision
- Summary: Added ASM-optimized TopK Softmax kernel for BF16 input tensors.

- Detailed explanation:
  Added hand-optimized assembly kernel for BF16 inputs, which is the common precision for modern LLMs. This avoids precision conversion overhead and uses native BF16 operations.

- Code excerpt:
    ```cpp
    // add asm topksoftmax for bf16 input
    ```

- Evidence mapping:
  - BF16 support → Commit message mentions "bf16 input"
  - ASM optimization → Hand-optimized assembly kernel
