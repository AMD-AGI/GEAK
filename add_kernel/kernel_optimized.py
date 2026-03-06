#!/usr/bin/env python3
"""
Optimized Vector Add Kernel

Optimizations applied:
1. Autotuning for BLOCK_SIZE to find optimal configuration per GPU
2. Added num_warps tuning for better warp utilization
3. Added num_stages tuning for better instruction pipelining
4. Vectorized loads/stores with eviction policies for better memory bandwidth
5. Larger block sizes to reduce kernel launch overhead
6. Cache eviction hints for streaming access pattern
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel_optimized(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized vector addition kernel with autotuning.
    
    Optimizations:
    - Autotuned BLOCK_SIZE, num_warps, num_stages
    - Cache eviction hints for streaming pattern
    - Larger block sizes reduce launch overhead
    - Better warp utilization
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load with eviction policy for streaming access (evict first = don't cache)
    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy='evict_first')
    y = tl.load(y_ptr + offsets, mask=mask, eviction_policy='evict_first')
    output = x + y
    # Store with eviction policy
    tl.store(output_ptr + offsets, output, mask=mask, eviction_policy='evict_first')


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Add two tensors using optimized Triton kernel.
    
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
    
    # Grid function - autotuner will optimize BLOCK_SIZE
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel_optimized[grid](x, y, output, n_elements)
    
    return output


def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation for correctness checking."""
    return x + y


# Exports for agent discovery
triton_op = triton_add
torch_op = torch_add

# Evaluation configs for standard testing
EVAL_CONFIGS = [
    {"size": 1024, "dtype": "float32"},
    {"size": 4096, "dtype": "float32"},
    {"size": 1048576, "dtype": "float32"},
    {"size": 16777216, "dtype": "float32"},
]


def main():
    """Main entry point with --profile flag support."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Vector add kernel')
    parser.add_argument('--profile', action='store_true', help='Run for profiling')
    args = parser.parse_args()
    
    if args.profile:
        # Profile mode: run once with large input
        torch.manual_seed(42)
        size = 16 * 1024 * 1024  # 16M elements
        x = torch.randn(size, dtype=torch.float32, device='cpu').cuda()
        y = torch.randn(size, dtype=torch.float32, device='cpu').cuda()
        output = triton_add(x, y)
        torch.cuda.synchronize()
        sys.exit(0)
    else:
        # Default: run correctness tests
        print("Running correctness tests...")
        torch.manual_seed(42)
        for config in EVAL_CONFIGS:
            size = config["size"]
            dtype_str = config["dtype"]
            dtype = getattr(torch, dtype_str)
            
            x = torch.randn(size, dtype=dtype, device='cpu').cuda()
            y = torch.randn(size, dtype=dtype, device='cpu').cuda()
            
            triton_out = triton_add(x, y)
            torch_out = torch_add(x, y)
            
            torch.cuda.synchronize()
            
            if dtype == torch.float32:
                rtol, atol = 1e-5, 1e-6
            else:
                rtol, atol = 1e-3, 1e-3
            
            if torch.allclose(triton_out, torch_out, rtol=rtol, atol=atol):
                print(f"✓ PASS: size={size}, dtype={dtype_str}")
            else:
                max_diff = torch.max(torch.abs(triton_out - torch_out)).item()
                print(f"✗ FAIL: size={size}, dtype={dtype_str}, max_diff={max_diff}")
                sys.exit(1)
        
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
