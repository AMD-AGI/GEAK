#!/usr/bin/env python3
"""
Test harness for matmul_kernel - Matrix Multiplication Triton Kernel

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
sys.path.insert(0, '/workspace/geak-merged/examples/matmul_kernel')
from kernel import triton_matmul, torch_matmul

# ============================================================================
# Shape definitions
# For matmul kernels, use square and rectangular matrices
# Format: (M, K, N) where A is MxK, B is KxN, C is MxN
# Sorted by total FLOPs (small to large)
# ============================================================================

ALL_SHAPES = [
    # Small sizes for quick tests
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    # Medium sizes
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    # Large sizes
    (8192, 8192, 8192),
    # Rectangular shapes
    (512, 1024, 512),
    (1024, 512, 1024),
    (2048, 1024, 2048),
    (4096, 2048, 4096),
    # Tall and wide matrices
    (8192, 1024, 1024),
    (1024, 1024, 8192),
    (4096, 512, 4096),
    (512, 4096, 512),
    # More varied shapes
    (1024, 2048, 512),
    (2048, 512, 1024),
    (3072, 3072, 3072),
    (5120, 5120, 5120),
    (6144, 6144, 6144),
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


def generate_inputs(shape, dtype=torch.float16, device='cuda'):
    """Generate test inputs on CPU then move to GPU."""
    torch.manual_seed(42)
    M, K, N = shape
    a = torch.randn((M, K), dtype=dtype, device='cpu').to(device)
    b = torch.randn((K, N), dtype=dtype, device='cpu').to(device)
    return a, b


def run_correctness(shapes):
    """Run correctness tests on given shapes."""
    print(f"Running correctness tests on {len(shapes)} shapes...")
    
    for shape in shapes:
        M, K, N = shape
        a, b = generate_inputs(shape)
        
        # Run Triton kernel
        output = triton_matmul(a, b)
        torch.cuda.synchronize()
        
        # Run PyTorch reference
        expected = torch_matmul(a, b)
        torch.cuda.synchronize()
        
        # Compare results
        try:
            torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
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
    M, K, N = shape
    
    a, b = generate_inputs(shape)
    
    # Single kernel invocation for profiling
    output = triton_matmul(a, b)
    torch.cuda.synchronize()
    
    print(f"Profile run complete: {M}x{K} @ {K}x{N}")


def run_benchmark(shapes, iterations):
    """Run benchmark on given shapes."""
    print(f"Running benchmark on {len(shapes)} shapes with {iterations} iterations...")
    
    all_latencies = []
    
    for shape in shapes:
        M, K, N = shape
        a, b = generate_inputs(shape)
        
        # Warmup
        for _ in range(3):
            _ = triton_matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = triton_matmul(a, b)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        median_latency = statistics.median(latencies)
        all_latencies.append(median_latency)
        
        # Calculate TFLOPS (2*M*N*K FLOPs for matmul)
        flops = 2 * M * N * K
        tflops = (flops / (median_latency / 1000)) / 1e12
        
        print(f"  Shape {shape}: median={median_latency:.4f}ms, TFLOPS={tflops:.2f}")
    
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
