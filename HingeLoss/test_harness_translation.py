#!/usr/bin/env python3
"""
PyTorch-to-FlyDSL Translation Test Harness for HingeLoss kernel.

This harness supports TWO execution modes:
1. Baseline mode (no --flydsl-kernel): runs PyTorch reference only
2. Comparison mode (--flydsl-kernel <path>): compares PyTorch vs FlyDSL

CLI modes:
  --correctness: validate outputs (baseline: PyTorch sanity check, comparison: PyTorch vs FlyDSL)
  --profile: run for profiling (baseline: PyTorch, comparison: FlyDSL only)
  --benchmark: benchmark on HARNESS_SHAPES, print latency
  --full-benchmark: benchmark on ALL_SHAPES, print latency
  --iterations N: override number of benchmark iterations (default 20)
  --flydsl-kernel <path>: path to FlyDSL candidate module (optional)
"""

import argparse
import torch
import time
import sys
import os
import importlib.util
from pathlib import Path

# Add the HingeLoss directory to the path for PyTorch reference
sys.path.insert(0, '/workspace/GEAK/HingeLoss')

from hingeloss import Model as PyTorchModel, batch_size as default_batch_size, input_shape as default_input_shape

# Shape definitions: (batch_size, input_size)
# Based on the kernel's default: batch_size=32768, input_shape=(32768,)
# We create a range from small (128) to the default size (32768)
ALL_SHAPES = [
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
    (32768, 32768),
]

# HARNESS_SHAPES: uniformly sampled subset (5 shapes for this case)
HARNESS_SHAPES = [
    (128, 128),
    (2048, 2048),
    (8192, 8192),
    (16384, 16384),
    (32768, 32768),
]

# PROFILE_SHAPES: exactly 5 evenly-spaced shapes
PROFILE_SHAPES = [
    (128, 128),
    (2048, 2048),
    (8192, 8192),
    (16384, 16384),
    (32768, 32768),
]


def load_flydsl_kernel(kernel_path):
    """
    Dynamically load FlyDSL kernel from the given file path.
    Returns either a Model class or a forward function.
    """
    if not kernel_path or not Path(kernel_path).exists():
        return None
    
    try:
        spec = importlib.util.spec_from_file_location("flydsl_kernel", kernel_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Try to get Model class first, then fall back to forward function
        if hasattr(module, 'Model'):
            return module.Model
        elif hasattr(module, 'forward'):
            return module.forward
        else:
            print(f"WARNING: FlyDSL module at {kernel_path} has no Model class or forward function")
            return None
    except Exception as e:
        print(f"WARNING: Failed to load FlyDSL kernel from {kernel_path}: {e}")
        return None


def generate_inputs(batch_size, input_size, device='cpu'):
    """
    Generate deterministic inputs for testing.
    Inputs are generated on CPU, then moved to device.
    """
    torch.manual_seed(42)
    # predictions: (batch_size, input_size)
    predictions = torch.randn(batch_size, input_size, dtype=torch.float32, device='cpu')
    # targets: (batch_size,) with values in {-1, 1}
    targets = torch.randint(0, 2, (batch_size,), device='cpu').float() * 2 - 1
    
    # Move to device
    predictions = predictions.to(device)
    targets = targets.to(device)
    
    return predictions, targets


def test_correctness_baseline(shapes):
    """
    Baseline mode correctness: validate PyTorch reference runs without error.
    """
    print(f"[Baseline] Running correctness tests on {len(shapes)} shapes...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = PyTorchModel().to(device)
    
    for batch_size, input_size in shapes:
        predictions, targets = generate_inputs(batch_size, input_size, device)
        
        # Run PyTorch reference
        result = model(predictions, targets)
        
        # Sanity check: result should be a scalar tensor
        assert result.dim() == 0, f"Expected scalar output, got shape {result.shape}"
        assert result.item() >= 0, f"Hinge loss should be non-negative, got {result.item()}"
    
    print(f"✓ All {len(shapes)} baseline correctness tests passed")
    return True


def test_correctness_comparison(shapes, flydsl_model_or_fn):
    """
    Comparison mode correctness: compare PyTorch vs FlyDSL outputs.
    """
    print(f"[Comparison] Running correctness tests on {len(shapes)} shapes...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pytorch_model = PyTorchModel().to(device)
    
    # Determine if FlyDSL is a class or function
    if isinstance(flydsl_model_or_fn, type):
        flydsl_model = flydsl_model_or_fn().to(device)
        flydsl_forward = lambda *args: flydsl_model(*args)
    else:
        flydsl_forward = flydsl_model_or_fn
    
    for batch_size, input_size in shapes:
        predictions, targets = generate_inputs(batch_size, input_size, device)
        
        # Run PyTorch reference
        pytorch_result = pytorch_model(predictions, targets)
        
        # Run FlyDSL kernel
        flydsl_result = flydsl_forward(predictions, targets)
        
        # Compare outputs
        try:
            torch.testing.assert_close(flydsl_result, pytorch_result, rtol=1e-4, atol=1e-5)
        except AssertionError as e:
            print(f"FAILED: shape ({batch_size}, {input_size})")
            print(f"  PyTorch: {pytorch_result.item()}")
            print(f"  FlyDSL: {flydsl_result.item()}")
            print(f"  Error: {e}")
            return False
    
    print(f"✓ All {len(shapes)} comparison correctness tests passed")
    return True


def profile_baseline(shapes):
    """
    Baseline mode profile: run PyTorch reference for profiling.
    """
    print(f"[Baseline] Running profile mode on {len(shapes)} shapes...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = PyTorchModel().to(device)
    
    for batch_size, input_size in shapes:
        predictions, targets = generate_inputs(batch_size, input_size, device)
        
        # Warmup
        _ = model(predictions, targets)
        
        # Single run for profiling
        if device == 'cuda':
            torch.cuda.synchronize()
        result = model(predictions, targets)
        if device == 'cuda':
            torch.cuda.synchronize()
    
    print(f"✓ Baseline profile run completed on {len(shapes)} shapes")


def profile_comparison(shapes, flydsl_model_or_fn):
    """
    Comparison mode profile: run FlyDSL kernel only for profiling.
    """
    print(f"[Comparison] Running profile mode on {len(shapes)} shapes...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Determine if FlyDSL is a class or function
    if isinstance(flydsl_model_or_fn, type):
        flydsl_model = flydsl_model_or_fn().to(device)
        flydsl_forward = lambda *args: flydsl_model(*args)
    else:
        flydsl_forward = flydsl_model_or_fn
    
    for batch_size, input_size in shapes:
        predictions, targets = generate_inputs(batch_size, input_size, device)
        
        # Warmup
        _ = flydsl_forward(predictions, targets)
        
        # Single run for profiling
        if device == 'cuda':
            torch.cuda.synchronize()
        result = flydsl_forward(predictions, targets)
        if device == 'cuda':
            torch.cuda.synchronize()
    
    print(f"✓ Comparison profile run completed on {len(shapes)} shapes")


def benchmark_baseline(shapes, iterations=20):
    """
    Baseline mode benchmark: benchmark PyTorch reference only.
    """
    print(f"[Baseline] Running benchmark on {len(shapes)} shapes with {iterations} iterations...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = PyTorchModel().to(device)
    all_latencies = []
    
    for batch_size, input_size in shapes:
        predictions, targets = generate_inputs(batch_size, input_size, device)
        
        # Warmup
        for _ in range(5):
            _ = model(predictions, targets)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = model(predictions, targets)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        median_latency = sorted(latencies)[len(latencies) // 2]
        all_latencies.append(median_latency)
        print(f"  PyTorch shape ({batch_size:6d}, {input_size:6d}): {median_latency:.6f} ms")
    
    # Compute overall median
    overall_median = sorted(all_latencies)[len(all_latencies) // 2]
    print(f"\n[Baseline] PyTorch median latency: {overall_median:.6f} ms")
    print(f"GEAK_RESULT_LATENCY_MS={overall_median:.6f}")
    
    return overall_median


def benchmark_comparison(shapes, flydsl_model_or_fn, iterations=20):
    """
    Comparison mode benchmark: benchmark both PyTorch and FlyDSL, report speedup.
    """
    print(f"[Comparison] Running benchmark on {len(shapes)} shapes with {iterations} iterations...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pytorch_model = PyTorchModel().to(device)
    
    # Determine if FlyDSL is a class or function
    if isinstance(flydsl_model_or_fn, type):
        flydsl_model = flydsl_model_or_fn().to(device)
        flydsl_forward = lambda *args: flydsl_model(*args)
    else:
        flydsl_forward = flydsl_model_or_fn
    
    pytorch_latencies = []
    flydsl_latencies = []
    
    for batch_size, input_size in shapes:
        predictions, targets = generate_inputs(batch_size, input_size, device)
        
        # Benchmark PyTorch
        for _ in range(5):
            _ = pytorch_model(predictions, targets)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        pytorch_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = pytorch_model(predictions, targets)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            pytorch_times.append((end - start) * 1000)
        
        pytorch_median = sorted(pytorch_times)[len(pytorch_times) // 2]
        pytorch_latencies.append(pytorch_median)
        
        # Benchmark FlyDSL
        for _ in range(5):
            _ = flydsl_forward(predictions, targets)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        flydsl_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = flydsl_forward(predictions, targets)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            flydsl_times.append((end - start) * 1000)
        
        flydsl_median = sorted(flydsl_times)[len(flydsl_times) // 2]
        flydsl_latencies.append(flydsl_median)
        
        speedup = pytorch_median / flydsl_median if flydsl_median > 0 else 0
        print(f"  Shape ({batch_size:6d}, {input_size:6d}): PyTorch={pytorch_median:.6f} ms | FlyDSL={flydsl_median:.6f} ms | Speedup={speedup:.2f}x")
    
    # Compute overall medians
    pytorch_overall = sorted(pytorch_latencies)[len(pytorch_latencies) // 2]
    flydsl_overall = sorted(flydsl_latencies)[len(flydsl_latencies) // 2]
    overall_speedup = pytorch_overall / flydsl_overall if flydsl_overall > 0 else 0
    
    print(f"\n[Comparison] PyTorch: {pytorch_overall:.6f} ms | FlyDSL: {flydsl_overall:.6f} ms | Speedup: {overall_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={flydsl_overall:.6f}")
    
    return flydsl_overall


def main():
    parser = argparse.ArgumentParser(description='PyTorch-to-FlyDSL Translation Test Harness for HingeLoss')
    parser.add_argument('--correctness', action='store_true', help='Run correctness tests')
    parser.add_argument('--profile', action='store_true', help='Run profiling mode')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark on HARNESS_SHAPES')
    parser.add_argument('--full-benchmark', action='store_true', help='Run benchmark on ALL_SHAPES')
    parser.add_argument('--iterations', type=int, default=None, help='Number of benchmark iterations')
    parser.add_argument('--flydsl-kernel', type=str, default=None, help='Path to FlyDSL candidate module')
    
    args = parser.parse_args()
    
    # Get iterations from args or environment variable
    iterations = args.iterations
    if iterations is None:
        iterations = int(os.environ.get('GEAK_BENCHMARK_ITERATIONS', '20'))
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU")
    
    # Load FlyDSL kernel if provided
    flydsl_model_or_fn = None
    if args.flydsl_kernel:
        flydsl_model_or_fn = load_flydsl_kernel(args.flydsl_kernel)
        if flydsl_model_or_fn is None:
            print(f"WARNING: Could not load FlyDSL kernel from {args.flydsl_kernel}, falling back to baseline mode")
    
    # Determine execution mode
    is_comparison_mode = flydsl_model_or_fn is not None
    mode_str = "Comparison" if is_comparison_mode else "Baseline"
    print(f"=== Running in {mode_str} mode ===\n")
    
    # Execute based on CLI mode
    if args.correctness:
        if is_comparison_mode:
            success = test_correctness_comparison(HARNESS_SHAPES, flydsl_model_or_fn)
        else:
            success = test_correctness_baseline(HARNESS_SHAPES)
        sys.exit(0 if success else 1)
    
    elif args.profile:
        if is_comparison_mode:
            profile_comparison(PROFILE_SHAPES, flydsl_model_or_fn)
        else:
            profile_baseline(PROFILE_SHAPES)
        sys.exit(0)
    
    elif args.benchmark:
        if is_comparison_mode:
            benchmark_comparison(HARNESS_SHAPES, flydsl_model_or_fn, iterations)
        else:
            benchmark_baseline(HARNESS_SHAPES, iterations)
        sys.exit(0)
    
    elif args.full_benchmark:
        if is_comparison_mode:
            benchmark_comparison(ALL_SHAPES, flydsl_model_or_fn, iterations)
        else:
            benchmark_baseline(ALL_SHAPES, iterations)
        sys.exit(0)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
