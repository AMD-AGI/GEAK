Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
SPDX-License-Identifier: Apache-2.0

# Kernel: Mamba/SSM Kernels

## Variant Context
- Input semantic type: State Space Model (SSM) computation for Mamba architecture
- Datatype(s): FP16, BF16, FP32
- Data representation: Recurrent state, input sequences, SSM parameters
- Target architecture: Generic CUDA and ROCm

## Functionality
These kernels implement the Mamba State Space Model operations, including:
1. Causal Conv1D for input processing
2. Selective SSM scan for recurrent computation
3. State passing between chunks for long sequences

## Optimization 1: Triton Causal Conv1D
- Commit ID: (causal_conv1d_triton.py)
- Optimization type: Compute
- Summary: Implements causal 1D convolution in Triton for efficient Mamba input processing.

- Detailed explanation:
  Mamba uses a short causal convolution before the SSM. This Triton kernel implements the convolution efficiently with:
  1. Shared memory for filter weights
  2. Efficient boundary handling for causal masking
  3. Fused activation function

- Code excerpt:
    ```python
    @triton.jit
    def causal_conv1d_kernel(
        x_ptr,           # Input: [batch, dim, seq_len]
        weight_ptr,      # Filter: [dim, width]
        out_ptr,         # Output: [batch, dim, seq_len]
        seq_len,
        dim,
        width: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        seq_idx = tl.program_id(1)
        dim_offs = tl.arange(0, BLOCK_DIM)
        
        # Load filter weights to registers
        weights = tl.load(weight_ptr + dim_offs[:, None] * width + tl.arange(0, width)[None, :])
        
        # Causal convolution
        acc = tl.zeros([BLOCK_DIM], dtype=tl.float32)
        for w in range(width):
            if seq_idx - w >= 0:
                x = tl.load(x_ptr + batch_idx * dim * seq_len + dim_offs * seq_len + seq_idx - w)
                acc += x * weights[:, w]
        
        tl.store(out_ptr + ..., acc)
    ```

- Evidence mapping:
  - "Causal masking" → `if seq_idx - w >= 0` check
  - "Register weights" → filter loaded once per thread block
  - "Efficient scan" → single pass over sequence

## Optimization 2: Chunked SSM Scan
- Commit ID: (ssd_chunk_scan.py)
- Optimization type: Memory / Parallelism
- Summary: Implements chunked selective scan for parallel processing of long sequences.

- Detailed explanation:
  The selective SSM scan is inherently sequential. This optimization chunks the sequence and:
  1. Processes chunks in parallel
  2. Passes state between chunks
  3. Enables efficient GPU utilization for long sequences

- Code excerpt:
    ```python
    @triton.jit
    def ssd_chunk_scan_kernel(
        x_ptr,           # Input
        A_ptr,           # State transition
        B_ptr,           # Input projection
        C_ptr,           # Output projection
        state_ptr,       # Recurrent state
        out_ptr,
        chunk_size: tl.constexpr,
    ):
        chunk_idx = tl.program_id(0)
        
        # Load initial state for this chunk
        state = tl.load(state_ptr + chunk_idx * state_dim)
        
        # Scan within chunk
        for t in range(chunk_size):
            # SSM update: state = A * state + B * x
            x_t = tl.load(x_ptr + (chunk_idx * chunk_size + t) * dim)
            state = A * state + B * x_t
            out_t = C * state
            tl.store(out_ptr + ..., out_t)
        
        # Store final state for next chunk
        tl.store(state_ptr + (chunk_idx + 1) * state_dim, state)
    ```

- Evidence mapping:
  - "Chunked processing" → `chunk_idx = tl.program_id(0)`
  - "State passing" → state loaded/stored at chunk boundaries
  - "Parallel chunks" → independent chunk processing

## Optimization 3: Fused Gated MLP
- Commit ID: (layernorm_gated.py)
- Optimization type: Fusion
- Summary: Fuses layer normalization with gated activation for Mamba's MLP blocks.

- Detailed explanation:
  Mamba uses gated MLPs similar to SwiGLU. This kernel fuses:
  1. Layer normalization
  2. Linear projection
  3. Gated activation (SiLU gate * linear)

- Code excerpt:
    ```python
    @triton.jit
    def layernorm_gated_kernel(
        x_ptr,
        weight_ptr,
        gate_weight_ptr,
        out_ptr,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Compute layer norm
        mean = tl.sum(x) / BLOCK_SIZE
        var = tl.sum((x - mean) ** 2) / BLOCK_SIZE
        x_norm = (x - mean) / tl.sqrt(var + eps)
        
        # Gated activation
        gate = tl.sigmoid(x_norm * gate_weight)
        out = x_norm * weight * gate
        
        tl.store(out_ptr, out)
    ```

- Evidence mapping:
  - "Fused operations" → norm + gate in single kernel
  - "Gated activation" → `gate = tl.sigmoid(...)`
