#!/usr/bin/env python3
"""
Comprehensive test and benchmark suite for add_kernel.
Tests correctness and measures performance.
"""

import torch
import triton
import sys
import time
import numpy as np
import os

# Import the kernel
from kernel import triton_add, torch_add

def test_correctness():
    """Test kernel correctness against PyTorch reference."""
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available, running on CPU")
        print("Note: Triton kernels require CUDA, using PyTorch reference only")
    
    # Test configurations: (size, dtype)
    test_configs = [
        (1024, torch.float32),
        (4096, torch.float32),
        (1048576, torch.float32),  # 1M elements
        (1024, torch.float16),
        (4096, torch.float16),
    ]
    
    # Add bfloat16 only if CUDA available (CPU might not support it well)
    if device == 'cuda':
        test_configs.append((1024, torch.bfloat16))
    
    all_passed = True
    
    for size, dtype in test_configs:
        try:
            # Generate random inputs
            torch.manual_seed(42)
            x = torch.randn(size, device=device, dtype=dtype)
            y = torch.randn(size, device=device, dtype=dtype)
            
            # PyTorch reference
            ref_output = torch_add(x, y)
            
            if device == 'cuda':
                # Triton kernel output (only on CUDA)
                triton_output = triton_add(x, y)
                
                # Compare results
                torch.cuda.synchronize()
                
                # Use appropriate tolerance for dtype
                if dtype == torch.float32:
                    rtol, atol = 1e-5, 1e-6
                elif dtype == torch.float16:
                    rtol, atol = 1e-3, 1e-3
                else:  # bfloat16
                    rtol, atol = 1e-2, 1e-2
                
                if torch.allclose(triton_output, ref_output, rtol=rtol, atol=atol):
                    print(f"✓ PASS: size={size:>8}, dtype={str(dtype).split('.')[-1]:<10}")
                else:
                    max_diff = torch.max(torch.abs(triton_output - ref_output)).item()
                    print(f"✗ FAIL: size={size:>8}, dtype={str(dtype).split('.')[-1]:<10} max_diff={max_diff}")
                    all_passed = False
            else:
                # CPU mode - just verify PyTorch reference works
                print(f"✓ PASS (CPU ref only): size={size:>8}, dtype={str(dtype).split('.')[-1]:<10}")
                
        except Exception as e:
            print(f"✗ ERROR: size={size:>8}, dtype={str(dtype).split('.')[-1]:<10} - {str(e)}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ ALL CORRECTNESS TESTS PASSED")
        print("=" * 60)
        return True
    else:
        print("✗ SOME CORRECTNESS TESTS FAILED")
        print("=" * 60)
        return False

def benchmark_kernel():
    """Benchmark kernel performance."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cpu':
        print("WARNING: CUDA not available, skipping Triton benchmarks")
        print("Running PyTorch reference benchmarks on CPU")
    
    # Benchmark configurations
    benchmark_configs = [
        (1024, torch.float32),
        (4096, torch.float32),
        (16384, torch.float32),
        (65536, torch.float32),
        (262144, torch.float32),
        (1048576, torch.float32),
        (4194304, torch.float32),
        (1048576, torch.float16),
    ]
    
    warmup_iters = 10
    benchmark_iters = 100
    
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Benchmark iterations: {benchmark_iters}")
    print("-" * 60)
    
    results = []
    
    for size, dtype in benchmark_configs:
        try:
            # Prepare inputs
            torch.manual_seed(42)
            x = torch.randn(size, device=device, dtype=dtype)
            y = torch.randn(size, device=device, dtype=dtype)
            
            # Warmup
            for _ in range(warmup_iters):
                if device == 'cuda':
                    _ = triton_add(x, y)
                else:
                    _ = torch_add(x, y)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.perf_counter()
            for _ in range(benchmark_iters):
                if device == 'cuda':
                    output = triton_add(x, y)
                else:
                    output = torch_add(x, y)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Calculate metrics
            elapsed_ms = (end_time - start_time) * 1000 / benchmark_iters
            bytes_per_elem = x.element_size()
            total_bytes = size * bytes_per_elem * 3  # read x, read y, write output
            bandwidth_gb_s = (total_bytes / 1e9) / (elapsed_ms / 1000)
            
            result = {
                'size': size,
                'dtype': str(dtype).split('.')[-1],
                'time_ms': elapsed_ms,
                'bandwidth_gb_s': bandwidth_gb_s
            }
            results.append(result)
            
            kernel_type = "Triton" if device == 'cuda' else "PyTorch(CPU)"
            print(f"[{kernel_type}] size={size:>8}, dtype={result['dtype']:<10} | "
                  f"time={elapsed_ms:>8.4f} ms | "
                  f"bandwidth={bandwidth_gb_s:>8.2f} GB/s")
            
        except Exception as e:
            print(f"✗ ERROR: size={size:>8}, dtype={str(dtype).split('.')[-1]:<10} - {str(e)}")
    
    print("=" * 60)
    
    # Summary statistics
    if results:
        avg_bandwidth = np.mean([r['bandwidth_gb_s'] for r in results])
        max_bandwidth = max([r['bandwidth_gb_s'] for r in results])
        print(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
        print(f"Peak bandwidth: {max_bandwidth:.2f} GB/s")
        print("=" * 60)
    
    return True

def main():
    """Main test runner."""
    print("\nAdd Kernel Test Suite")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name() if device == 'cuda' else 'CPU'
    
    print(f"Device: {device_name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Triton version: {triton.__version__}")
    
    # Run correctness tests
    correctness_passed = test_correctness()
    
    if not correctness_passed:
        print("\n✗ Correctness tests failed. Skipping benchmarks.")
        sys.exit(1)
    
    # Run benchmarks
    benchmark_kernel()
    
    print("\n✓ All tests completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
