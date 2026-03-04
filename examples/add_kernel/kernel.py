# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

#!/usr/bin/env python3
"""
Simple Vector Add Kernel - Baseline for GEAK Agent Testing

This is an UNOPTIMIZED baseline Triton kernel for testing the GEAK agent.
The agent should:
1. Discover this kernel
2. Create test cases automatically
3. Create benchmarks automatically
4. Generate benchmark/baseline/metrics.json
5. Run optimizer to improve performance

No tests, no benchmarks included - agent creates everything from scratch.

Usage:
    mini -m claude-sonnet-4.5 -t "Optimize this kernel" --yolo
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple unoptimized vector addition kernel.
    
    Baseline characteristics:
    - Fixed BLOCK_SIZE (no autotuning)
    - No warp/stage tuning
    - Basic memory access
    - No cache hints
    - Room for many optimizations!
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors using Triton kernel.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Sum of x and y
    """
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    
    x = x.contiguous()
    y = y.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    # Fixed BLOCK_SIZE - not optimized!
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation for correctness checking."""
    return x + y


# Exports for agent discovery
triton_op = triton_add
torch_op = torch_add


if __name__ == "__main__":
    size = 1_000_000
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)

    # Warm-up (compiles the kernel)
    output = triton_add(x, y)
    torch.cuda.synchronize()

    # Profiling-friendly run
    output = triton_add(x, y)
    torch.cuda.synchronize()

    expected = torch_add(x, y)
    assert torch.allclose(output, expected), "Correctness check failed!"
    print(f"add_kernel: {size} elements, output[0]={output[0].item():.4f}")
