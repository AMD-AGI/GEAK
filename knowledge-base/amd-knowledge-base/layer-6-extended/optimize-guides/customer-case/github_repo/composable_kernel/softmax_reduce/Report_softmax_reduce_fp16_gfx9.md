---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: Softmax and Reduce Operations

## Variant Context
- Input semantic type: Reduction and normalization operations
- Datatype(s): FP16/BF16/FP32
- Data representation: Multi-dimensional tensors
- Target architecture: gfx9 family (gfx908, gfx90a, gfx942, gfx950)

## Functionality
The Softmax and Reduce kernels implement common reduction operations used throughout neural networks. Softmax computes exp(x)/sum(exp(x)) for classification and attention, while Reduce supports various reduction operations (sum, max, min, mean) across specified dimensions.

## Optimization 1: Multi-Reduce Improvements
- Commit ID: 91e32f305
- Optimization type: compute / memory
- Summary: Improved multi-reduce kernel to handle multiple reduction operations in a single kernel launch.

- Detailed explanation:
  When multiple reductions are needed (e.g., computing both sum and max for softmax), the multi-reduce kernel fuses them into a single pass over the data, reducing memory bandwidth requirements.

- Code excerpt:
    ```cpp
    // Multi-reduce kernel: compute multiple reductions in one pass
    template <typename... ReduceOps>
    struct MultiReduceKernel
    {
        static constexpr index_t kNumReduceOps = sizeof...(ReduceOps);
        
        CK_TILE_DEVICE void operator()(
            const InputType* input,
            std::tuple<OutputType*...> outputs,
            index_t reduce_dim)
        {
            // Initialize accumulators for each reduction
            std::tuple<typename ReduceOps::AccType...> accs = 
                std::make_tuple(ReduceOps::Init()...);
            
            // Single pass over data
            for(index_t i = tid; i < reduce_dim; i += blockDim.x)
            {
                InputType val = input[i];
                // Apply each reduction operation
                std::apply([&](auto&... acc) {
                    ((acc = ReduceOps::Reduce(acc, val)), ...);
                }, accs);
            }
            
            // Cross-thread reduction for each operation
            std::apply([&](auto&... acc) {
                ((acc = block_reduce<ReduceOps>(acc)), ...);
            }, accs);
            
            // Store results
            if(tid == 0)
            {
                std::apply([&](auto*... out) {
                    index_t idx = 0;
                    ((out[blockIdx.x] = std::get<idx++>(accs)), ...);
                }, outputs);
            }
        }
    };
    ```

- Evidence mapping:
  - "Multiple reductions" → `std::tuple<typename ReduceOps::AccType...>`
  - "Single pass" → One loop over data with multiple accumulators
  - "Fused operations" → Variadic template for arbitrary reduction count

## Optimization 2: Softmax Undefined Behavior Fix
- Commit ID: 565fea264
- Optimization type: correctness
- Summary: Fixed undefined behavior in softmax kernel for edge cases.

- Detailed explanation:
  The softmax kernel had undefined behavior when the input contained NaN or Inf values, or when the reduction dimension was zero. The fix adds proper handling for these edge cases.

- Code excerpt:
    ```cpp
    // Safe softmax with edge case handling
    CK_TILE_DEVICE void softmax_safe(
        const InputType* input,
        OutputType* output,
        index_t dim)
    {
        // Handle empty dimension
        if(dim == 0)
            return;
        
        // Find max with NaN handling
        AccType max_val = -std::numeric_limits<AccType>::infinity();
        for(index_t i = tid; i < dim; i += blockDim.x)
        {
            AccType val = static_cast<AccType>(input[i]);
            if(!isnan(val))
                max_val = max(max_val, val);
        }
        max_val = block_reduce_max(max_val);
        
        // Handle all-NaN case
        if(isinf(max_val) && max_val < 0)
        {
            // All values are NaN, output uniform distribution
            for(index_t i = tid; i < dim; i += blockDim.x)
                output[i] = static_cast<OutputType>(1.0f / dim);
            return;
        }
        
        // Standard softmax computation
        AccType sum_exp = 0;
        for(index_t i = tid; i < dim; i += blockDim.x)
        {
            AccType val = static_cast<AccType>(input[i]);
            sum_exp += exp(val - max_val);
        }
        sum_exp = block_reduce_sum(sum_exp);
        
        AccType inv_sum = 1.0f / sum_exp;
        for(index_t i = tid; i < dim; i += blockDim.x)
        {
            AccType val = static_cast<AccType>(input[i]);
            output[i] = static_cast<OutputType>(exp(val - max_val) * inv_sum);
        }
    }
    ```

- Evidence mapping:
  - "Edge case handling" → Checks for `dim == 0`, NaN, Inf
  - "NaN handling" → `isnan()` check in max computation
  - "Uniform fallback" → Output `1/dim` for all-NaN input

## Optimization 3: TopK + Softmax Fusion
- Commit ID: (topk_softmax kernel)
- Optimization type: fusion
- Summary: Fused TopK selection with Softmax computation for MoE routing.

- Detailed explanation:
  In MoE models, tokens are routed to top-K experts based on softmax scores. Fusing TopK selection with Softmax avoids materializing the full softmax output, only computing softmax for the selected top-K elements.

- Code excerpt:
    ```cpp
    // Fused TopK + Softmax kernel
    template <index_t K>
    struct TopKSoftmaxKernel
    {
        CK_TILE_DEVICE void operator()(
            const InputType* logits,
            OutputType* topk_weights,
            IndexType* topk_indices,
            index_t num_experts)
        {
            // Find top-K values and indices
            TopKHeap<K> heap;
            
            for(index_t i = tid; i < num_experts; i += blockDim.x)
            {
                AccType val = static_cast<AccType>(logits[i]);
                heap.push(val, i);
            }
            
            // Reduce heaps across threads
            heap = block_reduce_topk(heap);
            
            // Compute softmax only for top-K elements
            if(tid == 0)
            {
                AccType max_val = heap.top().value;
                AccType sum_exp = 0;
                
                for(index_t k = 0; k < K; ++k)
                {
                    sum_exp += exp(heap[k].value - max_val);
                }
                
                for(index_t k = 0; k < K; ++k)
                {
                    topk_weights[k] = exp(heap[k].value - max_val) / sum_exp;
                    topk_indices[k] = heap[k].index;
                }
            }
        }
    };
    ```

- Evidence mapping:
  - "Fused operation" → Single kernel for TopK + Softmax
  - "Partial softmax" → Only compute for K elements, not all
  - "MoE routing" → Output weights and indices for expert selection
