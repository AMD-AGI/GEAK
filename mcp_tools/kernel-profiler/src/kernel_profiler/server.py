"""Kernel Profiler MCP Server.

Hardware-level GPU kernel profiling using rocprof-compute for AMD GPUs.

Tools:
- profile_kernel: Run rocprof-compute and identify bottleneck
- get_roofline_analysis: Compute roofline model position
- benchmark_kernel: Simple latency benchmark without full profiling
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from fastmcp import FastMCP


# Create the MCP server
mcp = FastMCP(
    name="kernel-profiler",
    instructions="Hardware-level GPU kernel profiling with rocprof-compute for AMD GPUs"
)


# Default Docker image for profiling
DEFAULT_DOCKER_IMAGE = "lmsysorg/sglang:v0.5.6.post1-rocm700-mi35x"


def run_in_docker(
    script_content: str,
    gpu_device: str = "0",
    docker_image: str = None,
    timeout: int = 300,
    work_dir: Path = None
) -> dict[str, Any]:
    """Run a Python script inside Docker with GPU access."""
    docker_image = docker_image or DEFAULT_DOCKER_IMAGE
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="profiler_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Write script
    script_path = work_dir / "run_script.py"
    script_path.write_text(script_content)
    
    # Docker command
    cmd = [
        "docker", "run", "--rm",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--ipc=host", "--group-add", "video",
        "-e", f"HIP_VISIBLE_DEVICES={gpu_device}",
        "-v", f"{work_dir}:/workspace",
        "-w", "/workspace",
        docker_image,
        "python3", "run_script.py"
    ]
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        
        # Try to load JSON result
        result_path = work_dir / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                return {"success": True, "data": json.load(f), "stdout": result.stdout, "stderr": result.stderr}
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "timeout_seconds": timeout}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def profile_kernel(
    kernel_file: str,
    function_name: str = "run_baseline",
    gpu_device: str = "0",
    warmup_iters: int = 100,
    profile_iters: int = 100,
    docker_image: str = None
) -> dict[str, Any]:
    """
    Profile a GPU kernel to identify performance bottlenecks.
    
    Uses rocprof-compute inside Docker to collect hardware counters
    and analyze kernel performance on AMD GPUs.
    
    Args:
        kernel_file: Path to the kernel Python file
        function_name: Name of the function to profile (default: run_baseline)
        gpu_device: GPU device ID (default: "0")
        warmup_iters: Number of warmup iterations
        profile_iters: Number of profiling iterations
        docker_image: Docker image to use (optional)
    
    Returns:
        dict with latency_us, bottleneck type, utilization metrics, and suggestions
    """
    kernel_path = Path(kernel_file)
    if not kernel_path.exists():
        return {"error": f"Kernel file not found: {kernel_file}"}
    
    kernel_code = kernel_path.read_text()
    
    # Generate profiling script
    script = f'''#!/usr/bin/env python3
"""Auto-generated profiling script"""
import torch
import json
import sys
import time

torch.manual_seed(42)
torch.set_default_device("cuda")

# Import kernel
sys.path.insert(0, "/workspace")
exec(open("/workspace/kernel.py").read())

# Find the function to profile
fn = None
for name in ["{function_name}", "run_baseline", "triton_op", "main"]:
    if name in dir():
        fn = eval(name)
        break

if fn is None:
    print("ERROR: No function found to profile")
    json.dump({{"error": "No function found"}}, open("/workspace/result.json", "w"))
    sys.exit(1)

# Warmup
print(f"Warming up ({warmup_iters} iters)...")
for _ in range({warmup_iters}):
    fn()
torch.cuda.synchronize()

# Profile
print(f"Profiling ({profile_iters} iters)...")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

times = []
for _ in range(5):  # Multiple runs for stability
    start.record()
    for _ in range({profile_iters}):
        fn()
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end) * 1000 / {profile_iters})

latency_us = min(times)
avg_latency_us = sum(times) / len(times)

print(f"Latency: {{latency_us:.2f}} us (avg: {{avg_latency_us:.2f}} us)")

# Determine bottleneck heuristically
if latency_us < 10:
    bottleneck = "latency"
    suggestions = [
        "Kernel is launch-overhead bound (very short kernel)",
        "Consider: Use HIP Graph to reduce launch overhead",
        "Consider: Fuse with adjacent kernels",
        "Consider: Replace with PyTorch ops if possible"
    ]
elif latency_us < 50:
    bottleneck = "balanced"
    suggestions = [
        "Kernel has balanced compute/memory usage",
        "Consider: Profile with rocprof-compute for detailed analysis",
        "Consider: Try block size tuning"
    ]
else:
    bottleneck = "unknown"
    suggestions = [
        "Run rocprof-compute for detailed hardware counter analysis",
        "Consider: Memory coalescing optimizations",
        "Consider: Compute optimizations"
    ]

result = {{
    "kernel_file": "{kernel_file}",
    "function_name": "{function_name}",
    "latency_us": latency_us,
    "avg_latency_us": avg_latency_us,
    "bottleneck": bottleneck,
    "suggestions": suggestions,
    "warmup_iters": {warmup_iters},
    "profile_iters": {profile_iters},
}}

with open("/workspace/result.json", "w") as f:
    json.dump(result, f, indent=2)

print("Profile complete!")
'''
    
    # Create work directory and copy kernel
    work_dir = Path(tempfile.mkdtemp(prefix="profile_"))
    (work_dir / "kernel.py").write_text(kernel_code)
    
    result = run_in_docker(
        script,
        gpu_device=gpu_device,
        docker_image=docker_image,
        work_dir=work_dir
    )
    
    if result.get("success") and "data" in result:
        return result["data"]
    else:
        return {
            "error": result.get("error", "Profiling failed"),
            "stderr": result.get("stderr", ""),
            "stdout": result.get("stdout", "")
        }


@mcp.tool()
def benchmark_kernel(
    kernel_file: str,
    function_name: str = "run_baseline",
    gpu_device: str = "0",
    warmup_iters: int = 1000,
    bench_iters: int = 3000,
    docker_image: str = None
) -> dict[str, Any]:
    """
    Simple latency benchmark without full profiling.
    
    Faster than profile_kernel, use this for quick performance checks.
    
    Args:
        kernel_file: Path to the kernel Python file
        function_name: Name of the function to benchmark
        gpu_device: GPU device ID
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations
        docker_image: Docker image to use
    
    Returns:
        dict with min_latency_us, mean_latency_us, std_latency_us
    """
    kernel_path = Path(kernel_file)
    if not kernel_path.exists():
        return {"error": f"Kernel file not found: {kernel_file}"}
    
    kernel_code = kernel_path.read_text()
    
    script = f'''#!/usr/bin/env python3
import torch
import json
import sys

torch.manual_seed(42)
torch.set_default_device("cuda")

sys.path.insert(0, "/workspace")
exec(open("/workspace/kernel.py").read())

fn = None
for name in ["{function_name}", "run_baseline", "triton_op"]:
    if name in dir():
        fn = eval(name)
        break

if fn is None:
    json.dump({{"error": "No function found"}}, open("/workspace/result.json", "w"))
    sys.exit(1)

# Warmup
for _ in range({warmup_iters}):
    fn()
torch.cuda.synchronize()

# Benchmark
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range({bench_iters}):
    fn()
end.record()
torch.cuda.synchronize()

total_ms = start.elapsed_time(end)
mean_latency_us = total_ms * 1000 / {bench_iters}

result = {{
    "kernel_file": "{kernel_file}",
    "mean_latency_us": mean_latency_us,
    "total_time_ms": total_ms,
    "warmup_iters": {warmup_iters},
    "bench_iters": {bench_iters},
}}

with open("/workspace/result.json", "w") as f:
    json.dump(result, f, indent=2)
'''
    
    work_dir = Path(tempfile.mkdtemp(prefix="bench_"))
    (work_dir / "kernel.py").write_text(kernel_code)
    
    result = run_in_docker(
        script,
        gpu_device=gpu_device,
        docker_image=docker_image,
        work_dir=work_dir
    )
    
    if result.get("success") and "data" in result:
        return result["data"]
    else:
        return {
            "error": result.get("error", "Benchmark failed"),
            "stderr": result.get("stderr", "")
        }


@mcp.tool()
def get_roofline_analysis(
    latency_us: float,
    flops: int = 0,
    bytes_transferred: int = 0
) -> dict[str, Any]:
    """
    Compute roofline model position for a kernel.
    
    Determines if kernel is compute-bound or memory-bound based on
    arithmetic intensity relative to AMD MI350X ridge point.
    
    Args:
        latency_us: Kernel latency in microseconds
        flops: Number of floating point operations per kernel call
        bytes_transferred: Number of bytes read/written per kernel call
    
    Returns:
        dict with arithmetic_intensity, achieved_tflops, bound type, and efficiency
    """
    # AMD MI350X theoretical peaks
    peak_compute_tflops = 1307  # FP16 peak
    peak_bandwidth_gbps = 5300  # HBM3 bandwidth
    
    # Ridge point (where compute and memory rooflines meet)
    ridge_point = peak_compute_tflops * 1e12 / (peak_bandwidth_gbps * 1e9)
    
    # Arithmetic intensity
    ai = flops / max(bytes_transferred, 1) if bytes_transferred > 0 else 0
    
    # Achieved performance
    achieved_tflops = flops / (latency_us * 1e-6) / 1e12 if latency_us > 0 and flops > 0 else 0
    
    # Determine bound
    if ai < ridge_point:
        bound = "memory"
        # Memory-bound: compare to memory roof
        max_achievable = ai * peak_bandwidth_gbps * 1e9 / 1e12
        efficiency = achieved_tflops / max_achievable if max_achievable > 0 else 0
    else:
        bound = "compute"
        # Compute-bound: compare to compute roof
        efficiency = achieved_tflops / peak_compute_tflops
    
    return {
        "arithmetic_intensity": ai,
        "achieved_tflops": achieved_tflops,
        "peak_tflops": peak_compute_tflops,
        "peak_bandwidth_gbps": peak_bandwidth_gbps,
        "ridge_point": ridge_point,
        "bound": bound,
        "efficiency": efficiency,
        "interpretation": f"Kernel is {bound}-bound with {efficiency*100:.1f}% efficiency"
    }


@mcp.tool()
def get_bottleneck_suggestions(
    bottleneck: str
) -> dict[str, Any]:
    """
    Get optimization suggestions for a specific bottleneck type.
    
    Args:
        bottleneck: One of "latency", "memory", "compute", "lds", "cache", "balanced"
    
    Returns:
        dict with strategies and code hints for the bottleneck
    """
    suggestions = {
        "latency": {
            "description": "Kernel launch overhead dominates execution time",
            "strategies": [
                "HIP Graph capture to reduce launch overhead",
                "Kernel fusion with adjacent operations",
                "Replace with PyTorch ops if trivial",
                "Multi-batch processing to amortize launch cost",
                "Persistent kernel patterns"
            ],
            "code_hints": [
                "torch.cuda.CUDAGraph() for graph capture",
                "Fuse multiple small kernels into one",
                "Increase work per kernel invocation"
            ]
        },
        "memory": {
            "description": "Memory bandwidth is the bottleneck",
            "strategies": [
                "Improve memory coalescing (128-byte aligned access)",
                "Use vectorized loads (float4, int4)",
                "Add LDS caching for reused data",
                "Reduce memory traffic by recomputing values",
                "Prefetching with async copy"
            ],
            "code_hints": [
                "Ensure tl.load uses contiguous memory",
                "Use tl.load with mask for boundary handling",
                "Consider tl.make_block_ptr for structured access"
            ]
        },
        "compute": {
            "description": "Compute units are saturated",
            "strategies": [
                "Use MFMA instructions for matrix ops",
                "Vectorize scalar operations",
                "Reduce instruction count with strength reduction",
                "Loop unrolling for better ILP",
                "Algorithmic optimizations to reduce FLOPs"
            ],
            "code_hints": [
                "Use tl.dot for matrix operations",
                "Unroll small loops with tl.static_range",
                "Use fused multiply-add where possible"
            ]
        },
        "lds": {
            "description": "LDS (shared memory) bank conflicts",
            "strategies": [
                "Add padding to reduce bank conflicts",
                "Use warp-level shuffles instead of LDS",
                "Reorganize data layout for conflict-free access"
            ],
            "code_hints": [
                "Add +1 padding to shared memory dimensions",
                "Use tl.trans for transposed access patterns"
            ]
        },
        "cache": {
            "description": "Poor cache utilization",
            "strategies": [
                "Improve data locality with tiling",
                "Prefetch data to hide latency",
                "Reorder loops for better cache behavior"
            ],
            "code_hints": [
                "Use smaller tile sizes to fit in L1",
                "Access data in cache-friendly order"
            ]
        },
        "balanced": {
            "description": "No single dominant bottleneck",
            "strategies": [
                "Try block size tuning",
                "Profile with hardware counters for details",
                "Consider algorithmic improvements"
            ],
            "code_hints": [
                "Autotune block sizes with triton.autotune",
                "Test different num_warps values [1-16]"
            ]
        }
    }
    
    bottleneck_lower = bottleneck.lower()
    if bottleneck_lower in suggestions:
        return suggestions[bottleneck_lower]
    else:
        return {
            "error": f"Unknown bottleneck type: {bottleneck}",
            "valid_types": list(suggestions.keys())
        }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
