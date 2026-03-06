# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Example: Profile a GPU kernel with the unified profiler MCP.

Demonstrates programmatic usage of the profiler-mcp without running
the MCP server. Both metrix and rocprof-compute backends are supported.

Usage:
    # Metrix backend (default) -- structured JSON with bottleneck classification
    python examples/profile_kernel.py 'python3 kernel.py --profile'

    # rocprof-compute backend -- deep roofline and instruction analysis
    python examples/profile_kernel.py 'python3 kernel.py --profile' --backend rocprof-compute

    # Metrix quick mode (fewer metrics, faster)
    python examples/profile_kernel.py 'python3 kernel.py --profile' --quick

    # Specify working directory (for rocprof-compute)
    python examples/profile_kernel.py 'python3 kernel.py' --backend rocprof-compute --workdir /path/to/kernel
"""

import argparse
import json
import sys
from pathlib import Path

# Add profiler-mcp and metrix-mcp to path
_script_dir = Path(__file__).resolve().parent
_profiler_src = str(_script_dir.parent / "src")
_repo_root = _script_dir.parent.parent.parent
_metrix_src = str(_repo_root / "mcp_tools" / "metrix-mcp" / "src")
_agent_src = str(_repo_root / "src")

for _p in [_profiler_src, _metrix_src, _agent_src]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profiler_mcp.server import profile_kernel


def print_metrix_result(result):
    """Pretty-print metrix backend results."""
    for device in result.get("results", []):
        gpu_info = device.get("gpu_info", {})
        print(f"\nGPU {device['device_id']}: {gpu_info.get('model', 'unknown')} "
              f"({gpu_info.get('architecture', '?')})")
        print(f"  CUs: {gpu_info.get('compute_units', '?')}, "
              f"Peak HBM BW: {gpu_info.get('peak_hbm_bandwidth_gbs', '?')} GB/s")
        print()

        kernels = device.get("kernels", [])
        if not kernels:
            print("  No kernels found.")
            continue

        # Table header
        print(f"  {'Kernel':<50} {'Duration (us)':>14} {'Bottleneck':<15}")
        print(f"  {'-'*50} {'-'*14} {'-'*15}")

        for k in kernels:
            name = k["name"][:50]
            duration = f"{k['duration_us']:.2f}"
            bottleneck = k.get("bottleneck", "?")
            print(f"  {name:<50} {duration:>14} {bottleneck:<15}")

            # Show observations
            for obs in k.get("observations", []):
                print(f"    - {obs}")

            # Show key metrics
            metrics = k.get("metrics", {})
            if metrics:
                print(f"    Metrics: {json.dumps(metrics, indent=6)[:200]}...")
            print()


def print_rocprof_result(result):
    """Pretty-print rocprof-compute backend results."""
    print(f"\nProfiling type: {result.get('profiling_type', '?')}")
    print(f"Analysis ({len(result.get('analysis', ''))} chars):\n")
    print(result.get("analysis", "(no analysis)"))


def main():
    parser = argparse.ArgumentParser(
        description="Profile a GPU kernel using the unified profiler MCP."
    )
    parser.add_argument("command", help="Command to profile (e.g. 'python3 kernel.py --profile')")
    parser.add_argument(
        "--backend",
        choices=["metrix", "rocprof-compute"],
        default="metrix",
        help="Profiling backend (default: metrix)",
    )
    parser.add_argument("--workdir", default=None, help="Working directory (rocprof-compute only)")
    parser.add_argument(
        "--profiling-type",
        choices=["profiling", "roofline", "profiler_analyzer"],
        default="profiling",
        help="Profiling type for rocprof-compute (default: profiling)",
    )
    parser.add_argument("--quick", action="store_true", help="Quick profile (metrix only)")
    parser.add_argument("--gpu", default=None, help="GPU device ID(s)")
    args = parser.parse_args()

    print(f"Profiling with backend={args.backend}...")
    print(f"Command: {args.command}")

    result = profile_kernel.fn(
        command=args.command,
        backend=args.backend,
        workdir=args.workdir,
        profiling_type=args.profiling_type,
        quick=args.quick,
        gpu_devices=args.gpu,
    )

    if not result["success"]:
        print(f"\nERROR: {result.get('error', 'unknown error')}")
        sys.exit(1)

    if args.backend == "metrix":
        print_metrix_result(result)
    else:
        print_rocprof_result(result)

    print("\nDone.")


if __name__ == "__main__":
    main()
