#!/usr/bin/env python3
"""
Example: Profile a GPU kernel using MetrixTool.

Usage:
    python examples/profile_kernel.py 'python3 kernel.py --profile'
    python examples/profile_kernel.py 'python3 kernel.py --profile' --gpu 3
    python examples/profile_kernel.py 'python3 kernel.py --profile' --filter '*topk*'
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mini_kernel.mcp_tools.metrix import MetrixTool

EXAMPLES = """
Examples:
  %(prog)s 'python3 /path/to/kernel.py --profile'
  %(prog)s 'python3 kernel.py --profile' --gpu-devices 3
  %(prog)s 'python3 kernel.py --profile' --filter '*topk*' --replays 5
  %(prog)s 'python3 kernel.py --profile' --auto-select  # Auto-select main kernel only
  %(prog)s 'python3 kernel.py --profile' --quick  # Fast profiling (3 metrics, 1 pass)
  %(prog)s 'python3 kernel.py --profile' --gpu-devices 0,1,2  # Profile on multiple GPUs
"""


def main():
    parser = argparse.ArgumentParser(
        description="Profile a GPU kernel using metrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument(
        "command", help='Command to profile (e.g., "python3 kernel.py --profile")'
    )
    parser.add_argument(
        "--gpu-devices",
        default="3",
        help='GPU device ID(s): single ("3") or multiple comma-separated ("0,1,2") (default: 3)',
    )
    parser.add_argument("--filter", help='Kernel name filter pattern (e.g., "*topk*")')
    parser.add_argument(
        "--replays",
        type=int,
        default=3,
        help="Number of profiling replays (default: 3)",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Automatically select main kernel (default: show all kernels)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick profile (3 metrics, 1 pass) for speed. Default: memory profile (14 metrics, 2 passes)",
    )

    args = parser.parse_args()

    # Parse GPU argument (support comma-separated list)
    gpu_devices = (
        args.gpu_devices.split(",") if "," in args.gpu_devices else args.gpu_devices
    )

    tool = MetrixTool(gpu_devices=gpu_devices)
    result = tool.profile(
        command=args.command,
        num_replays=args.replays,
        kernel_filter=args.filter,
        auto_select=args.auto_select,
        quick=args.quick,
    )

    # Display results (always a list, even for single GPU)
    for gpu_result in result["results"]:
        if len(result["results"]) > 1:
            # Show GPU separator for multiple GPUs
            device_id = gpu_result.get("device_id", "?")
            print(f"\n{'='*70}")
            print(f"GPU {device_id}")
            print(f"{'='*70}")
        _display_single_gpu_result(gpu_result, args.auto_select)


def _display_single_gpu_result(result, auto_select):
    """Display results for a single GPU."""
    # Display GPU info if detected
    if result.get("gpu_info", {}).get("detected"):
        gpu = result["gpu_info"]
        print(f"\nGPU: {gpu.get('vendor', 'Unknown')} {gpu.get('model', 'Unknown')}")
        print(f"Architecture: {gpu.get('architecture', 'Unknown')}")
        if "compute_units" in gpu:
            print(f"Compute Units: {gpu['compute_units']}")
        if "peak_hbm_bandwidth_gbs" in gpu:
            print(f"Peak HBM BW: {gpu['peak_hbm_bandwidth_gbs']:.1f} GB/s")
        if "fp32_tflops" in gpu:
            print(f"Peak FP32: {gpu['fp32_tflops']:.1f} TFLOPS")

    kernels = result["kernels"]

    # Print header for multi-kernel mode
    if not auto_select and len(kernels) > 1:
        print(f"\n{'='*70}")
        print(f"Found {len(kernels)} kernels")
        print(f"{'='*70}\n")

    # Display all kernels uniformly
    for i, kernel in enumerate(kernels):
        # Kernel name (with index if multiple kernels)
        if len(kernels) > 1:
            print(f"[{i}] {kernel['name']}")
        else:
            print(f"\nKernel: {kernel['name']}")

        indent = "  " if len(kernels) > 1 else ""

        # Bottleneck
        print(f"{indent}Bottleneck: {kernel['bottleneck']}")

        # Observations
        if kernel.get("observations"):
            print(f"{indent}Observations:")
            for obs in kernel["observations"]:
                print(f"{indent}  {obs}")

        # Metrics
        if kernel["metrics"]:
            metric_label = (
                f"Metrics ({len(kernel['metrics'])} total):"
                if not indent
                else "Metrics:"
            )
            print(f"{indent}{metric_label}")
            for name, value in sorted(kernel["metrics"].items()):
                print(f"{indent}  {name}: {value:.2f}")
        else:
            print(f"{indent}No metrics captured")
        print()

    # Print footer for multi-kernel mode
    if not auto_select and len(kernels) > 1:
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
