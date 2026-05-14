#!/usr/bin/env python3
"""GEAK Test Harness Generator.

Generates a test harness for Triton or HIP kernels with support for
correctness, profile, benchmark, and full-benchmark modes.

Usage:
    python3 create_harness.py --kernel-path /path/to/kernel.py \
        --kernel-type triton --output /path/to/test_harness.py

The generated harness provides:
- --correctness: Validate kernel output against reference
- --profile: Single kernel run for profiling (minimal GPU allocations)
- --benchmark: Quick benchmark (fewer iterations)
- --full-benchmark: Authoritative benchmark (more iterations)

Output format: GEAK_RESULT_LATENCY_MS=<float>
"""

import argparse
import os
import re
import sys
from pathlib import Path


def detect_kernel_type(kernel_path: str) -> str:
    """Auto-detect kernel type from file extension and content."""
    path = Path(kernel_path)
    ext = path.suffix.lower()

    if ext in ('.hip', '.cu'):
        return 'hip'
    if ext in ('.cpp', '.hpp', '.h'):
        content = path.read_text()
        if '__global__' in content or 'hipLaunchKernelGGL' in content:
            return 'hip'
        return 'cpp'
    if ext == '.py':
        content = path.read_text()
        if '@triton.jit' in content or 'tl.' in content:
            return 'triton'
        return 'python'
    return 'unknown'


def extract_triton_kernels(kernel_path: str) -> list[dict]:
    """Extract @triton.jit kernel signatures from a Python file."""
    content = Path(kernel_path).read_text()
    kernels = []
    pattern = re.compile(
        r'@triton\.jit\s*\ndef\s+(\w+)\s*\(([^)]*)\)',
        re.MULTILINE
    )
    for match in pattern.finditer(content):
        name = match.group(1)
        params_str = match.group(2)
        params = [p.strip().split(':')[0].strip() for p in params_str.split(',') if p.strip()]
        kernels.append({'name': name, 'params': params})
    return kernels


def extract_hip_kernels(kernel_path: str) -> list[dict]:
    """Extract __global__ kernel signatures from HIP/CUDA files."""
    content = Path(kernel_path).read_text()
    kernels = []
    pattern = re.compile(
        r'__global__\s+void\s+(\w+)\s*\(([^)]*)\)',
        re.MULTILINE
    )
    for match in pattern.finditer(content):
        name = match.group(1)
        params_str = match.group(2)
        params = []
        for p in params_str.split(','):
            p = p.strip()
            if p:
                parts = p.rsplit(None, 1)
                if len(parts) >= 2:
                    params.append({'type': parts[0], 'name': parts[1].strip('*& ')})
                else:
                    params.append({'type': p, 'name': 'arg'})
        kernels.append({'name': name, 'params': params})
    return kernels


def generate_triton_harness(kernel_path: str, kernels: list[dict]) -> str:
    """Generate a test harness for Triton kernels."""
    kernel_dir = os.path.dirname(os.path.abspath(kernel_path))
    kernel_module = Path(kernel_path).stem

    harness = f'''#!/usr/bin/env python3
"""GEAK Test Harness for Triton kernel: {kernel_module}

Auto-generated. Supports --correctness, --profile, --benchmark, --full-benchmark modes.
"""

import argparse
import os
import sys
import time

import torch

# Resolve kernel directory
_KERNEL_DIR = os.environ.get("GEAK_WORK_DIR", "{kernel_dir}")
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)

import {kernel_module}

# Configuration
BENCHMARK_ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "30"))
FULL_BENCHMARK_ITERATIONS = int(os.environ.get("GEAK_FULL_BENCHMARK_ITERATIONS", "100"))
WARMUP_ITERATIONS = 10


def create_test_inputs(device="cpu"):
    """Create representative test inputs on CPU (moved to GPU later).

    TODO: Adjust shapes and dtypes to match your kernel's requirements.
    """
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(K, N, dtype=torch.float16, device=device)
    c = torch.empty(M, N, dtype=torch.float16, device=device)
    return {{"a": a, "b": b, "c": c, "M": M, "N": N, "K": K}}


def reference_impl(inputs):
    """Reference implementation for correctness checking.

    TODO: Implement the reference computation.
    """
    return torch.matmul(inputs["a"], inputs["b"])


def run_kernel(inputs):
    """Run the kernel under test.

    TODO: Call your kernel with the correct arguments.
    """
    # Example: {kernel_module}.kernel_name[grid](inputs["a"], inputs["b"], inputs["c"], ...)
    raise NotImplementedError(
        "TODO: Implement run_kernel() to call your Triton kernel. "
        "See the kernel source for the correct function signature and grid dimensions."
    )


def run_correctness():
    """Validate kernel output against reference."""
    inputs = create_test_inputs(device="cuda")
    ref_output = reference_impl(inputs)
    run_kernel(inputs)
    test_output = inputs.get("c", None)

    if test_output is None:
        print("ERROR: No output tensor to compare")
        sys.exit(1)

    if torch.allclose(ref_output, test_output, atol=1e-2, rtol=1e-2):
        print("CORRECTNESS: PASS")
    else:
        max_diff = (ref_output - test_output).abs().max().item()
        print(f"CORRECTNESS: FAIL (max diff: {{max_diff}})")
        sys.exit(1)


def run_profile():
    """Single kernel run for profiling (minimal GPU allocations)."""
    inputs = create_test_inputs(device="cpu")
    inputs = {{k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}}
    torch.cuda.synchronize()
    run_kernel(inputs)
    torch.cuda.synchronize()


def run_benchmark(iterations: int):
    """Benchmark the kernel and report latency."""
    inputs = create_test_inputs(device="cuda")

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        run_kernel(inputs)
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        start_events[i].record()
        run_kernel(inputs)
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times_ms.sort()
    # Use median to reduce noise
    median_ms = times_ms[len(times_ms) // 2]
    mean_ms = sum(times_ms) / len(times_ms)

    print(f"Iterations: {{iterations}}")
    print(f"Median latency: {{median_ms:.4f}} ms")
    print(f"Mean latency: {{mean_ms:.4f}} ms")
    print(f"Min latency: {{min(times_ms):.4f}} ms")
    print(f"Max latency: {{max(times_ms):.4f}} ms")
    print(f"GEAK_RESULT_LATENCY_MS={{median_ms:.6f}}")


def main():
    parser = argparse.ArgumentParser(description="GEAK Test Harness")
    parser.add_argument("--correctness", action="store_true", help="Run correctness tests")
    parser.add_argument("--profile", action="store_true", help="Single run for profiling")
    parser.add_argument("--benchmark", action="store_true", help="Quick benchmark")
    parser.add_argument("--full-benchmark", action="store_true", help="Full benchmark (authoritative)")
    args = parser.parse_args()

    if not any([args.correctness, args.profile, args.benchmark, args.full_benchmark]):
        parser.print_help()
        sys.exit(1)

    if args.correctness:
        run_correctness()
    if args.profile:
        run_profile()
    if args.benchmark:
        run_benchmark(BENCHMARK_ITERATIONS)
    if args.full_benchmark:
        run_benchmark(FULL_BENCHMARK_ITERATIONS)


if __name__ == "__main__":
    main()
'''
    return harness


def generate_hip_harness(kernel_path: str, kernels: list[dict]) -> str:
    """Generate a test harness for HIP kernels."""
    kernel_dir = os.path.dirname(os.path.abspath(kernel_path))
    kernel_name = Path(kernel_path).stem

    harness = f'''#!/usr/bin/env python3
"""GEAK Test Harness for HIP kernel: {kernel_name}

Auto-generated. Supports --correctness, --profile, --benchmark, --full-benchmark modes.
Wraps a HIP kernel via ctypes or subprocess.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Configuration
KERNEL_DIR = os.environ.get("GEAK_WORK_DIR", "{kernel_dir}")
KERNEL_SOURCE = os.path.join(KERNEL_DIR, "{Path(kernel_path).name}")
BENCHMARK_ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "30"))
FULL_BENCHMARK_ITERATIONS = int(os.environ.get("GEAK_FULL_BENCHMARK_ITERATIONS", "100"))


def build_kernel():
    """Build the HIP kernel if needed."""
    # Check for Makefile
    makefile = os.path.join(KERNEL_DIR, "Makefile")
    if os.path.exists(makefile):
        result = subprocess.run(
            ["make", "-C", KERNEL_DIR],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"Build failed:\\n{{result.stderr}}")
            sys.exit(1)
        return

    # Check for CMakeLists.txt
    cmake = os.path.join(KERNEL_DIR, "CMakeLists.txt")
    if os.path.exists(cmake):
        build_dir = os.path.join(KERNEL_DIR, "build")
        os.makedirs(build_dir, exist_ok=True)
        subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
        subprocess.run(["make", "-j"], cwd=build_dir, check=True)
        return

    print("WARNING: No build system found. Attempting hipcc compilation.")
    exe_path = os.path.join(KERNEL_DIR, kernel_name)
    subprocess.run(
        ["hipcc", "-O3", KERNEL_SOURCE, "-o", exe_path],
        check=True, timeout=120
    )


def find_executable():
    """Find the compiled executable."""
    candidates = [
        os.path.join(KERNEL_DIR, "{kernel_name}"),
        os.path.join(KERNEL_DIR, "build", "{kernel_name}"),
        os.path.join(KERNEL_DIR, "a.out"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    # Search for any executable in the directory
    for f in Path(KERNEL_DIR).iterdir():
        if f.is_file() and os.access(f, os.X_OK) and f.suffix == "":
            return str(f)

    return None


def run_executable(*args, timeout=300):
    """Run the kernel executable with given arguments."""
    exe = find_executable()
    if not exe:
        print("ERROR: No executable found. Run build first.")
        sys.exit(1)

    result = subprocess.run(
        [exe, *args],
        capture_output=True, text=True, timeout=timeout
    )
    return result


def run_correctness():
    """Run correctness validation."""
    build_kernel()
    result = run_executable("--correctness")
    print(result.stdout)
    if result.returncode != 0:
        print(f"CORRECTNESS: FAIL\\n{{result.stderr}}")
        sys.exit(1)
    print("CORRECTNESS: PASS")


def run_profile():
    """Single run for profiling."""
    build_kernel()
    run_executable("--profile")


def run_benchmark(iterations: int):
    """Run benchmark and report latency."""
    build_kernel()

    times_ms = []
    for i in range(iterations):
        start = time.perf_counter()
        result = run_executable("--benchmark")
        elapsed = (time.perf_counter() - start) * 1000

        # Try to extract timing from the executable output
        import re
        match = re.search(r"time[:\\s]+([\\d.]+)\\s*ms", result.stdout, re.IGNORECASE)
        if match:
            times_ms.append(float(match.group(1)))
        else:
            times_ms.append(elapsed)

    times_ms.sort()
    median_ms = times_ms[len(times_ms) // 2]
    mean_ms = sum(times_ms) / len(times_ms)

    print(f"Iterations: {{iterations}}")
    print(f"Median latency: {{median_ms:.4f}} ms")
    print(f"Mean latency: {{mean_ms:.4f}} ms")
    print(f"GEAK_RESULT_LATENCY_MS={{median_ms:.6f}}")


def main():
    parser = argparse.ArgumentParser(description="GEAK Test Harness (HIP)")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    if not any([args.correctness, args.profile, args.benchmark, args.full_benchmark]):
        parser.print_help()
        sys.exit(1)

    if args.correctness:
        run_correctness()
    if args.profile:
        run_profile()
    if args.benchmark:
        run_benchmark(BENCHMARK_ITERATIONS)
    if args.full_benchmark:
        run_benchmark(FULL_BENCHMARK_ITERATIONS)


if __name__ == "__main__":
    main()
'''
    return harness


def main():
    parser = argparse.ArgumentParser(description="GEAK Test Harness Generator")
    parser.add_argument("--kernel-path", required=True, help="Path to the kernel source file")
    parser.add_argument("--kernel-type", default=None,
                        choices=["triton", "hip", "python", "cpp"],
                        help="Kernel type (auto-detected if not specified)")
    parser.add_argument("--output", required=True, help="Output harness file path")
    args = parser.parse_args()

    kernel_path = os.path.abspath(args.kernel_path)
    if not os.path.exists(kernel_path):
        print(f"ERROR: Kernel file not found: {kernel_path}")
        sys.exit(1)

    kernel_type = args.kernel_type or detect_kernel_type(kernel_path)
    print(f"Kernel type: {kernel_type}")

    if kernel_type == "triton":
        kernels = extract_triton_kernels(kernel_path)
        print(f"Found {len(kernels)} Triton kernel(s): {[k['name'] for k in kernels]}")
        harness = generate_triton_harness(kernel_path, kernels)
    elif kernel_type in ("hip", "cpp"):
        kernels = extract_hip_kernels(kernel_path)
        print(f"Found {len(kernels)} HIP kernel(s): {[k['name'] for k in kernels]}")
        harness = generate_hip_harness(kernel_path, kernels)
    else:
        print(f"WARNING: Unsupported kernel type '{kernel_type}'. Generating Triton template.")
        harness = generate_triton_harness(kernel_path, [])

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(harness)

    print(f"Harness written to: {output_path}")
    print(f"NOTE: You must edit the harness to fill in TODO sections before use.")


if __name__ == "__main__":
    main()
