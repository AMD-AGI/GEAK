# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Simple Vector Add Kernel -- AIG-Eval interface variant.

Same unoptimized Triton kernel as kernel.py, but implements the full
AIG-Eval interface required by OpenEvolve's auto-build mode:
  - triton_op / torch_op   (kernel wrappers)
  - EVAL_CONFIGS            (correctness check input sizes)
  - get_inputs(*config)     (tensor factory for correctness checks)
  - --profile CLI flag      (profiling-friendly execution)

Usage:
    python kernel_aig_eval.py              # run + correctness check
    python kernel_aig_eval.py --profile    # profiling-friendly run
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
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape

    x = x.contiguous()
    y = y.contiguous()
    output = torch.empty_like(x)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation for correctness checking."""
    return x + y


# ---------------------------------------------------------------------------
# AIG-Eval interface exports
# ---------------------------------------------------------------------------

triton_op = triton_add
torch_op = torch_add

EVAL_CONFIGS = [
    (1024,),
    (4096,),
    (65536,),
    (1_000_000,),
    (16_000_000,),
]


def get_inputs(n):
    """Create input tensors for a given vector size."""
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    return x, y


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vector add kernel")
    parser.add_argument("--profile", action="store_true",
                        help="Run in profiling mode (skip correctness check)")
    args = parser.parse_args()

    size = 1_000_000
    x, y = get_inputs(size)

    # Warm-up (compiles the kernel)
    output = triton_add(x, y)
    torch.cuda.synchronize()

    # Profiling-friendly run
    output = triton_add(x, y)
    torch.cuda.synchronize()

    if not args.profile:
        expected = torch_add(x, y)
        assert torch.allclose(output, expected), "Correctness check failed!"
        print(f"add_kernel: {size} elements, output[0]={output[0].item():.4f}")
