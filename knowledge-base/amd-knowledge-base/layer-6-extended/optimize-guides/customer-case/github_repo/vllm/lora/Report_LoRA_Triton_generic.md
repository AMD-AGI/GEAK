# Kernel: LoRA (Low-Rank Adaptation) Kernels

## Variant Context
- Input semantic type: Low-rank adapter computation
- Datatype(s): fp16, bf16
- Data representation: Low-rank matrices (A, B) for weight adaptation
- Target architecture: Triton (CUDA/HIP)

## Functionality
LoRA kernels implement efficient low-rank adaptation for fine-tuned models:
1. LoRA computation: output = base_output + (x @ A) @ B * scale
2. Shrink operation: x @ A (reduce dimension)
3. Expand operation: intermediate @ B (expand dimension)
4. Fused MoE LoRA for mixture-of-experts models

## Optimization 1: LoRA Shrink Kernel
- Commit ID: (vllm/lora/ops/triton_ops/lora_shrink_op.py)
- Optimization type: Compute
- Summary: Optimized kernel for the dimension-reducing LoRA projection
- Detailed explanation:
  The shrink operation projects input from model dimension to LoRA rank. This kernel handles multiple LoRA adapters efficiently by batching computations.
- Code excerpt:
    ```python
    @triton.jit
    def lora_shrink_kernel(
        input_ptr, lora_a_ptr, output_ptr,
        input_stride_batch, input_stride_dim,
        lora_a_stride_adapter, lora_a_stride_rank, lora_a_stride_dim,
        output_stride_batch, output_stride_rank,
        adapter_indices_ptr,  # Which adapter for each input
        model_dim, lora_rank, num_adapters,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        # Get batch and rank block indices
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Load adapter index for this batch
        batch_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        adapter_idx = tl.load(adapter_indices_ptr + batch_idx)
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Compute x @ A for each adapter
        for k in range(0, model_dim, BLOCK_K):
            # Load input block
            x = tl.load(input_ptr + batch_idx[:, None] * input_stride_batch 
                       + (k + tl.arange(0, BLOCK_K))[None, :] * input_stride_dim)
            
            # Load LoRA A weights (adapter-specific)
            a = tl.load(lora_a_ptr + adapter_idx[:, None, None] * lora_a_stride_adapter
                       + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :, None] * lora_a_stride_rank
                       + (k + tl.arange(0, BLOCK_K))[None, None, :] * lora_a_stride_dim)
            
            # Accumulate
            acc += tl.dot(x, tl.trans(a))
        
        # Store result
        tl.store(output_ptr + batch_idx[:, None] * output_stride_batch
                + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * output_stride_rank,
                acc.to(output_ptr.dtype.element_ty))
    ```
- Evidence mapping:
  - "Multi-adapter support" → `adapter_indices_ptr` selects per-input adapter
  - "Dimension reduction" → `model_dim` to `lora_rank`
  - "Batched computation" → Multiple inputs processed together

## Optimization 2: LoRA Expand Kernel
- Commit ID: (vllm/lora/ops/triton_ops/lora_expand_op.py)
- Optimization type: Compute
- Summary: Optimized kernel for the dimension-expanding LoRA projection
- Detailed explanation:
  The expand operation projects from LoRA rank back to model dimension. It includes scaling and optional addition to base output.
- Code excerpt:
    ```python
    @triton.jit
    def lora_expand_kernel(
        input_ptr, lora_b_ptr, output_ptr, base_output_ptr,
        lora_scales_ptr,  # Per-adapter scaling factors
        adapter_indices_ptr,
        lora_rank, model_dim,
        add_to_base: tl.constexpr,  # Whether to add to base output
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        batch_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        adapter_idx = tl.load(adapter_indices_ptr + batch_idx)
        
        # Load scaling factor for each adapter
        scale = tl.load(lora_scales_ptr + adapter_idx)
        
        # Compute intermediate @ B
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, lora_rank, BLOCK_K):
            # Load intermediate (shrink output)
            x = tl.load(input_ptr + ...)
            
            # Load LoRA B weights
            b = tl.load(lora_b_ptr + adapter_idx[:, None, None] * ...
                       + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :, None] * ...
                       + (k + tl.arange(0, BLOCK_K))[None, None, :] * ...)
            
            acc += tl.dot(x, tl.trans(b))
        
        # Apply scale
        acc = acc * scale[:, None]
        
        # Optionally add to base output
        if add_to_base:
            base = tl.load(base_output_ptr + ...)
            acc = acc + base
        
        tl.store(output_ptr + ..., acc.to(output_ptr.dtype.element_ty))
    ```
- Evidence mapping:
  - "Dimension expansion" → `lora_rank` to `model_dim`
  - "Per-adapter scaling" → `lora_scales_ptr` for different adapter strengths
  - "Fused addition" → `add_to_base` for combining with base model output

## Optimization 3: Fused MoE LoRA
- Commit ID: 5f6cbf60d
- Optimization type: Fusion
- Summary: Fuse LoRA computation with MoE expert computation
- Detailed explanation:
  For MoE models with LoRA adapters, this kernel fuses the expert GEMM with LoRA adaptation, avoiding intermediate tensor materialization.
- Code excerpt:
    ```python
    @triton.jit
    def fused_moe_lora_kernel(
        input_ptr, expert_weights_ptr, lora_a_ptr, lora_b_ptr,
        output_ptr,
        expert_ids_ptr, adapter_indices_ptr, lora_scales_ptr,
        sorted_token_ids_ptr, topk_weights_ptr,
        hidden_dim, expert_dim, lora_rank,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        # Get expert and adapter for this block
        pid_m = tl.program_id(0)
        expert_id = tl.load(expert_ids_ptr + pid_m)
        
        token_ids = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
        adapter_idx = tl.load(adapter_indices_ptr + token_ids)
        
        # Compute base expert output: x @ W_expert
        base_output = compute_expert_gemm(input_ptr, expert_weights_ptr, expert_id, ...)
        
        # Compute LoRA: (x @ A) @ B * scale
        # Shrink
        intermediate = tl.zeros((BLOCK_M, lora_rank), dtype=tl.float32)
        for k in range(0, hidden_dim, BLOCK_K):
            x = tl.load(input_ptr + ...)
            a = tl.load(lora_a_ptr + adapter_idx[:, None, None] * ... + expert_id * ...)
            intermediate += tl.dot(x, tl.trans(a))
        
        # Expand
        lora_output = tl.zeros((BLOCK_M, expert_dim), dtype=tl.float32)
        for k in range(0, lora_rank, BLOCK_K):
            b = tl.load(lora_b_ptr + adapter_idx[:, None, None] * ... + expert_id * ...)
            lora_output += tl.dot(intermediate[:, k:k+BLOCK_K], tl.trans(b))
        
        # Apply scale and combine
        scale = tl.load(lora_scales_ptr + adapter_idx)
        output = base_output + lora_output * scale[:, None]
        
        # Apply routing weight
        routing_weight = tl.load(topk_weights_ptr + token_ids)
        output = output * routing_weight[:, None]
        
        tl.store(output_ptr + ..., output)
    ```
- Evidence mapping:
  - "Fused MoE + LoRA" → Expert GEMM and LoRA in single kernel
  - "Per-expert LoRA" → Different LoRA weights per expert
  - "Multi-adapter" → Different adapters per token
