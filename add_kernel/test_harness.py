#!/usr/bin/env python3
"""
Test harness for add_kernel with --profile, --correctness, and --benchmark modes.
"""

import torch
import sys
import argparse
import time

# Import the kernel
from kernel import triton_add, torch_add

def run_correctness():
    """Test kernel correctness against PyTorch reference."""
    torch.manual_seed(42)
    
    # Use fixed sizes for deterministic testing
    test_configs = [
        (1024, torch.float32),
        (4096, torch.float32),
        (1048576, torch.float32),
        (1024, torch.float16),
    ]
    
    for size, dtype in test_configs:
        # Generate on CPU then move to GPU to avoid RNG kernels in profile
        x = torch.randn(size, dtype=dtype, device='cpu').cuda()
        y = torch.randn(size, dtype=dtype, device='cpu').cuda()
        
        # PyTorch reference
        ref_output = torch_add(x, y)
        
        # Triton kernel output
        triton_output = triton_add(x, y)
        
        torch.cuda.synchronize()
        
        # Use appropriate tolerance for dtype
        if dtype == torch.float32:
            rtol, atol = 1e-5, 1e-6
        else:  # float16
            rtol, atol = 1e-3, 1e-3
        
        if not torch.allclose(triton_output, ref_output, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(triton_output - ref_output)).item()
            print(f"FAIL: size={size}, dtype={dtype}, max_diff={max_diff}")
            sys.exit(1)
    
    print("All correctness tests passed")
    return 0

def run_profile():
    """Run kernel once for profiling (large enough to saturate GPU)."""
    torch.manual_seed(42)
    
    # Use large fixed size for profiling (16M elements)
    size = 16 * 1024 * 1024
    dtype = torch.float32
    
    # Generate on CPU then move to GPU to avoid RNG kernels in profile trace
    x = torch.randn(size, dtype=dtype, device='cpu').cuda()
    y = torch.randn(size, dtype=dtype, device='cpu').cuda()
    
    # Run kernel once
    output = triton_add(x, y)
    torch.cuda.synchronize()
    
    return 0

def run_benchmark():
    """Benchmark kernel performance."""
    torch.manual_seed(42)
    
    # Benchmark configurations
    benchmark_configs = [
        (1048576, torch.float32),
        (4194304, torch.float32),
        (16777216, torch.float32),
    ]
    
    warmup_iters = 10
    benchmark_iters = 100
    
    print(f"{'Size':<12} {'Dtype':<10} {'Time (ms)':<12} {'Bandwidth (GB/s)':<15}")
    print("-" * 60)
    
    for size, dtype in benchmark_configs:
        # Generate inputs
        x = torch.randn(size, dtype=dtype, device='cpu').cuda()
        y = torch.randn(size, dtype=dtype, device='cpu').cuda()
        
        # Warmup
        for _ in range(warmup_iters):
            _ = triton_add(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(benchmark_iters):
            output = triton_add(x, y)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # Calculate metrics
        elapsed_ms = (end_time - start_time) * 1000 / benchmark_iters
        bytes_per_elem = x.element_size()
        total_bytes = size * bytes_per_elem * 3  # read x, read y, write output
        bandwidth_gb_s = (total_bytes / 1e9) / (elapsed_ms / 1000)
        
        print(f"{size:<12} {str(dtype).split('.')[-1]:<10} {elapsed_ms:<12.4f} {bandwidth_gb_s:<15.2f}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Test harness for add_kernel')
    parser.add_argument('--correctness', action='store_true', help='Run correctness tests')
    parser.add_argument('--profile', action='store_true', help='Run kernel once for profiling')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    if args.correctness:
        return run_correctness()
    elif args.profile:
        return run_profile()
    elif args.benchmark:
        return run_benchmark()
    else:
        print("Please specify --correctness, --profile, or --benchmark")
        return 1

if __name__ == "__main__":
    sys.exit(main())
