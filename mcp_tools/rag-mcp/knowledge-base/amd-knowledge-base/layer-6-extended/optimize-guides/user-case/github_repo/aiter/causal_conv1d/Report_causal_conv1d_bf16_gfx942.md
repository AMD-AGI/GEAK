# Kernel: causal_conv1d

## Variant Context
- Input semantic type: 1D Causal Convolution (Mamba/State Space Models)
- Datatype(s): BF16/FP16
- Data representation: Sequence data with causal masking
- Target architecture: gfx942 (MI300), gfx950 (MI350)

## Functionality
This kernel implements causal 1D convolution used in Mamba-style state space models. It performs convolution where each output position only depends on current and previous input positions (causal constraint). The kernel supports:
1. Variable sequence lengths
2. Multiple channels
3. Efficient memory access patterns for AMD GPUs

## Optimization 1: Triton Implementation for AMD GPUs
- Commit ID: cbf2df776
- Optimization type: Compute / Memory
- Summary: Implemented efficient Triton-based causal conv1d kernel optimized for AMD GPU architecture.

- Detailed explanation:
  The kernel is implemented in Triton to leverage:
  1. Automatic memory coalescing
  2. Efficient shared memory usage
  3. AMD-specific optimizations through Triton's backend
  4. Support for SGLang integration

- Code excerpt:
    ```python
    # Causal conv1d triton implementation for SGLang
    # File: aiter/ops/triton/_triton_kernels/causal_conv1d.py (631 lines)
    # File: aiter/ops/triton/causal_conv1d.py (346 lines)
    ```

- Evidence mapping:
  - Triton implementation → New kernel files added
  - SGLang integration → Commit message mentions "sglang"
  - AMD optimization → Uses Triton's AMD backend
