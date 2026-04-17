#!/usr/bin/env python3
"""
Test harness for add_kernel - Vector Addition Triton Kernel

Modes:
  --correctness    : Validate kernel output against PyTorch reference
  --profile        : Run kernel once for profiling (uses PROFILE_SHAPES)
  --benchmark      : Run benchmark on HARNESS_SHAPES (20-25 shapes)
  --full-benchmark : Run benchmark on ALL_SHAPES (all discovered shapes)
  --iterations N   : Override number of benchmark iterations (default: 20)
"""

import argparse
import os
import sys
import time
import statistics

import torch

# Import the kernel via package path
sys.path.insert(0, '/workspace/geak-merged/examples/add_kernel')
from kernel import triton_add, torch_add

# ============================================================================
# Shape definitions
# For elementwise kernels, use sizes that saturate the GPU (at least 16M elements)
# Sorted by total element count (small to large)
# ============================================================================

ALL_SHAPES = [
    # Small sizes for quick tests
    (1024,),
    (4096,),
    (16384,),
    (65536,),
    (262144,),
    # Medium sizes
    (1_000_000,),
    (2_000_000,),
    (4_000_000,),
    (8_000_000,),
    # Large sizes (saturate GPU)
    (16_000_000,),
    (32_000_000,),
    (64_000_000,),
    (128_000_000,),
]

# HARNESS_SHAPES: 20-25 shapes sampled from ALL_SHAPES
# Since ALL_SHAPES has <= 25 entries, HARNESS_SHAPES = ALL_SHAPES
HARNESS_SHAPES = ALL_SHAPES

# PROFILE_SHAPES: 5 evenly-spaced shapes from ALL_SHAPES
PROFILE_SHAPES = [
    ALL_SHAPES[0],                          # smallest
    ALL_SHAPES[len(ALL_SHAPES) // 4],       # 25%
    ALL_SHAPES[len(ALL_SHAPES) // 2],       # 50%
    ALL_SHAPES[3 * len(ALL_SHAPES) // 4],   # 75%
    ALL_SHAPES[-1],                         # largest
]


def generate_inputs(shape, dtype=torch.float32, device='cuda'):
    """Generate test inputs on CPU then move to GPU."""
    torch.manual_seed(42)
    n_elements = shape[0]
    x = torch.randn(n_elements, dtype=dtype, device='cpu').to(device)
    y = torch.randn(n_elements, dtype=dtype, device='cpu').to(device)
    return x, y


def run_correctness(shapes):
    """Run correctness tests on given shapes."""
    print(f"Running correctness tests on {len(shapes)} shapes...")
    
    for shape in shapes:
        n_elements = shape[0]
        x, y = generate_inputs(shape)
        
        # Run Triton kernel
        output = triton_add(x, y)
        torch.cuda.synchronize()
        
        # Run PyTorch reference
        expected = torch_add(x, y)
        torch.cuda.synchronize()
        
        # Compare results
        try:
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
            print(f"  Shape {shape}: PASS")
        except AssertionError as e:
            print(f"  Shape {shape}: FAIL")
            print(f"    Error: {e}")
            sys.exit(1)
    
    print("All correctness tests passed!")
    return True


def run_profile(shapes):
    """Run kernel once for profiling (minimal setup)."""
    # Use the middle shape from PROFILE_SHAPES
    shape = shapes[len(shapes) // 2]
    n_elements = shape[0]
    
    x, y = generate_inputs(shape)
    
    # Single kernel invocation for profiling
    output = triton_add(x, y)
    torch.cuda.synchronize()
    
    print(f"Profile run complete: {n_elements} elements")


def run_benchmark(shapes, iterations):
    """Run benchmark on given shapes."""
    print(f"Running benchmark on {len(shapes)} shapes with {iterations} iterations...")
    
    all_latencies = []
    
    for shape in shapes:
        n_elements = shape[0]
        x, y = generate_inputs(shape)
        
        # Warmup
        for _ in range(3):
            _ = triton_add(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = triton_add(x, y)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        median_latency = statistics.median(latencies)
        all_latencies.append(median_latency)
        
        # Calculate bandwidth (2 reads + 1 write, 4 bytes per element for float32)
        bytes_transferred = n_elements * 4 * 3  # x, y read + output write
        bandwidth_gb_s = (bytes_transferred / (median_latency / 1000)) / 1e9
        
        print(f"  Shape {shape}: median={median_latency:.4f}ms, bandwidth={bandwidth_gb_s:.2f} GB/s")
    
    # Overall median latency across all shapes
    overall_median = statistics.median(all_latencies)
    print(f"\nOverall median latency: {overall_median:.4f} ms")
    
    # CRITICAL: Last line must be exactly this format for evaluation pipeline
    print(f"GEAK_RESULT_LATENCY_MS={overall_median:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Test harness for add_kernel")
    parser.add_argument('--correctness', action='store_true', help='Run correctness tests')
    parser.add_argument('--profile', action='store_true', help='Run kernel once for profiling')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark on HARNESS_SHAPES')
    parser.add_argument('--full-benchmark', action='store_true', help='Run benchmark on ALL_SHAPES')
    parser.add_argument('--iterations', type=int, default=None, help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    # Determine iterations (CLI > env var > default)
    if args.iterations is not None:
        iterations = args.iterations
    else:
        iterations = int(os.environ.get('GEAK_BENCHMARK_ITERATIONS', 20))
    
    if args.correctness:
        run_correctness(HARNESS_SHAPES)
    elif args.profile:
        run_profile(PROFILE_SHAPES)
    elif args.benchmark:
        run_benchmark(HARNESS_SHAPES, iterations)
    elif args.full_benchmark:
        run_benchmark(ALL_SHAPES, iterations)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
