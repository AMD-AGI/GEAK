---
Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
---

# Kernel: LoRA (Low-Rank Adaptation) Kernels

## Variant Context
- Input semantic type: Low-rank adapter computation for fine-tuned models
- Datatype(s): FP16, BF16
- Data representation: LoRA A/B matrices with adapter indices
- Target architecture: Generic CUDA

## Functionality
These kernels implement efficient LoRA computation for serving multiple fine-tuned adapters simultaneously. LoRA adds low-rank updates to pretrained weights: W' = W + BA where B and A are low-rank matrices.

## Optimization 1: Chunked SGMV (Sparse Gather Matrix-Vector) for Multi-LoRA
- Commit ID: (chunked_sgmv_expand.py, chunked_sgmv_shrink.py)
- Optimization type: Batching / Memory
- Summary: Implements chunked sparse gather matrix multiplication for efficient multi-LoRA batching.

- Detailed explanation:
  When serving multiple LoRA adapters in a batch, different requests may use different adapters. SGMV efficiently handles this by:
  1. Gathering the appropriate LoRA weights for each request
  2. Performing batched matrix multiplication
  3. Chunking to handle variable adapter assignments

- Code excerpt:
    ```python
    @triton.jit
    def sgmv_expand_kernel(
        input_ptr,           # [batch, hidden]
        lora_a_ptr,          # [num_adapters, rank, hidden]
        output_ptr,          # [batch, rank]
        adapter_ids_ptr,     # [batch] - which adapter for each request
        batch_size,
        hidden_size,
        rank,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        
        # Get adapter ID for this request
        adapter_id = tl.load(adapter_ids_ptr + batch_idx)
        
        # Load input
        input_vec = tl.load(input_ptr + batch_idx * hidden_size + tl.arange(0, BLOCK_K))
        
        # Load LoRA A weights for this adapter
        lora_a = tl.load(lora_a_ptr + adapter_id * rank * hidden_size + ...)
        
        # Compute: output = input @ lora_a.T
        output = tl.dot(input_vec, lora_a)
        
        tl.store(output_ptr + batch_idx * rank + ..., output)
    ```

- Evidence mapping:
  - "Sparse gather" → `adapter_id = tl.load(adapter_ids_ptr + batch_idx)`
  - "Per-request adapter" → different adapters for different batch elements
  - "Efficient batching" → single kernel handles all requests

## Optimization 2: Fused LoRA for QKV Projection
- Commit ID: (qkv_lora_b.py)
- Optimization type: Fusion
- Summary: Fuses LoRA B projection with the base QKV computation.

- Detailed explanation:
  For attention layers, LoRA is applied to Q, K, V projections. This kernel fuses the LoRA B multiplication with the base weight computation, reducing memory traffic.

- Code excerpt:
    ```python
    @triton.jit
    def qkv_lora_b_kernel(
        base_output_ptr,     # Output from base W_qkv
        lora_intermediate_ptr,  # Output from LoRA A
        lora_b_ptr,          # LoRA B weights
        output_ptr,          # Final output
        adapter_ids_ptr,
        # ...
    ):
        # Load base output
        base_out = tl.load(base_output_ptr + ...)
        
        # Load LoRA intermediate (from A projection)
        lora_inter = tl.load(lora_intermediate_ptr + ...)
        
        # Get adapter-specific B weights
        adapter_id = tl.load(adapter_ids_ptr + batch_idx)
        lora_b = tl.load(lora_b_ptr + adapter_id * ...)
        
        # Fused: output = base_out + lora_inter @ lora_b
        lora_out = tl.dot(lora_inter, lora_b)
        output = base_out + lora_out
        
        tl.store(output_ptr + ..., output)
    ```

- Evidence mapping:
  - "Fused addition" → `output = base_out + lora_out`
  - "QKV specific" → handles Q, K, V projections together
  - "Multi-adapter" → adapter_id indexing

## Optimization 3: Gate-Up LoRA B Fusion
- Commit ID: (gate_up_lora_b.py)
- Optimization type: Fusion
- Summary: Fuses LoRA B for gate and up projections in MLP layers.

- Detailed explanation:
  MLP layers in transformers have gate and up projections that are often computed together. This kernel fuses the LoRA B computation for both projections.

- Code excerpt:
    ```python
    @triton.jit
    def gate_up_lora_b_kernel(
        gate_base_ptr,
        up_base_ptr,
        lora_inter_ptr,
        gate_lora_b_ptr,
        up_lora_b_ptr,
        gate_output_ptr,
        up_output_ptr,
        # ...
    ):
        # Compute gate LoRA
        gate_lora = tl.dot(lora_inter, gate_lora_b)
        gate_out = gate_base + gate_lora
        
        # Compute up LoRA
        up_lora = tl.dot(lora_inter, up_lora_b)
        up_out = up_base + up_lora
        
        tl.store(gate_output_ptr + ..., gate_out)
        tl.store(up_output_ptr + ..., up_out)
    ```

- Evidence mapping:
  - "Dual projection" → gate and up computed together
  - "Shared intermediate" → same lora_inter used for both
