# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""kernel-profile: Profile GPU kernels via profiler-mcp.

Thin CLI wrapper around ``profiler_mcp.server.profile_kernel()``.  Supports
two backends:
  - metrix (default): AMD Metrix Python API -- structured per-kernel metrics
  - rocprof-compute: rocprof-compute CLI -- deep roofline, instruction mix, cache analysis

Both backends produce the same top-level JSON structure (backend-neutral) so that
downstream tools like baseline-metrics work with either.

The heavy lifting is done by profiler-mcp; this module adds:
  * ``--from-discovery`` convenience (extract command from discovery.json)
  * Human-readable (non-JSON) display mode
  * rocprof-compute → backend-neutral conversion helpers (also used by profiler-mcp)
"""

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
for _sub in ("mcp_tools/profiler-mcp/src", "mcp_tools/metrix-mcp/src"):
    _p = str(_repo_root / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXAMPLES = """
Examples (metrix backend, default):
  %(prog)s 'python3 /path/to/kernel.py --profile'
  %(prog)s 'python3 kernel.py --profile' --gpu-devices 3
  %(prog)s 'python3 kernel.py --profile' --replays 5
  %(prog)s 'python3 kernel.py --profile' --quick

Examples (rocprof-compute backend):
  %(prog)s 'python3 kernel.py --profile' --backend rocprof-compute
  %(prog)s --backend rocprof-compute --workdir /path/to/repo \\
      --profiling-type roofline 'python3 kernel.py --profile'

Pipeline chaining (read test command from discovery output):
  %(prog)s --from-discovery discovery.json --json -o profile.json
  %(prog)s --from-discovery discovery.json --backend rocprof-compute --json -o profile.json
"""


def _extract_command_from_discovery(discovery_path: str) -> str:
    """Extract the profiling command from a discovery JSON file.

    Prefers focused_test.focused_command, falls back to tests[0].command.
    """
    data = json.loads(Path(discovery_path).read_text())

    focused = data.get("focused_test") or {}
    cmd = focused.get("focused_command")
    if cmd:
        return cmd

    tests = data.get("tests") or []
    if tests:
        cmd = tests[0].get("command")
        if cmd:
            return cmd

    raise ValueError(
        f"No profiling command found in {discovery_path}: need focused_test.focused_command or tests[0].command"
    )


# ---------------------------------------------------------------------------
# rocprof-compute -> backend-neutral JSON
# ---------------------------------------------------------------------------


def _classify_rocprof_bottleneck(metrics: dict) -> str:
    """Classify bottleneck from rocprof-compute native metrics."""
    mem_bw = metrics.get("max_mem_bw_pct")
    compute = metrics.get("compute_util_pct")
    l2_hit = metrics.get("l2_hit")

    if mem_bw is not None and mem_bw > 30:
        return "memory"
    if compute is not None and compute > 50:
        return "compute"
    if mem_bw is not None and mem_bw < 5:
        if l2_hit is not None and l2_hit > 80:
            return "compute"
        return "latency"
    return "balanced"


def _generate_rocprof_observations(metrics: dict, bottleneck: str) -> list[str]:
    """Generate factual observations from rocprof-compute metrics."""
    obs: list[str] = []

    mem_bw = metrics.get("max_mem_bw_pct")
    compute = metrics.get("compute_util_pct")

    if bottleneck == "memory" and mem_bw is not None:
        obs.append(f"Classified as memory-bound (HBM BW util: {mem_bw:.1f}%)")
        coal = metrics.get("l1_coalescing_pct")
        if coal is not None:
            desc = "poor" if coal < 50 else "good" if coal > 80 else "moderate"
            obs.append(f"Coalescing efficiency: {coal:.1f}% ({desc})")
    elif bottleneck == "compute" and compute is not None:
        obs.append(f"Classified as compute-bound (compute util: {compute:.1f}%)")
        ipc = metrics.get("ipc")
        if ipc is not None:
            obs.append(f"IPC: {ipc:.2f}")
    elif bottleneck == "latency":
        obs.append(
            f"Classified as latency-bound (HBM BW util: {mem_bw:.1f}%)"
            if mem_bw is not None
            else "Classified as latency-bound"
        )
    else:
        obs.append("Classified as balanced")

    occ = metrics.get("occupancy")
    if occ is not None:
        obs.append(f"Occupancy: {occ:.1f}%")

    valu = metrics.get("valu_ratio")
    vmem = metrics.get("vmem_ratio")
    if valu is not None and vmem is not None:
        obs.append(f"Instruction mix: VALU {valu * 100:.1f}%, VMEM {vmem * 100:.1f}%")

    if metrics.get("uses_mfma"):
        obs.append("MFMA instructions in use")

    dep = metrics.get("dependency_wait_pct")
    active = metrics.get("active_cycle_pct")
    if dep is not None and active is not None:
        obs.append(f"Cycle breakdown: active {active:.1f}%, dependency wait {dep:.1f}%")

    return obs


def _build_rocprof_result(structured: dict, gpu_device: str = "0") -> dict:
    """Convert ProfilingAnalyzer.profile_structured() output to backend-neutral JSON."""
    sys_info = structured.get("sys_info", {})
    sys_speed = structured.get("sys_speed", {})
    compute_units = structured.get("compute_units", {})
    l1_data = structured.get("l1_data", {})
    l2_data = structured.get("l2_data", {})
    wavefront = structured.get("wavefront", {})
    roofline_rates = structured.get("roofline_rates", {})
    roofline_ai = structured.get("roofline_ai", {})
    top_kernels = structured.get("top_kernels", [])

    gpu_info = {
        "detected": bool(sys_info.get("gpu model")),
        "device_id": gpu_device,
        "vendor": "AMD",
        "model": sys_info.get("gpu model", "Unknown"),
        "architecture": sys_info.get("gpu architecture", "Unknown"),
        "compute_units": sys_info.get("CU per gpu", "Unknown"),
        "l1_cache": sys_info.get("gpu L1", "Unknown"),
        "l2_cache": sys_info.get("gpu L2", "Unknown"),
    }

    # Flatten all sections into a single metrics dict with native keys
    metrics: dict = {}

    # sys_speed metrics
    for k, v in sys_speed.items():
        if v is not None:
            metrics[k] = v

    # compute_units metrics
    for k, v in compute_units.items():
        if v is not None:
            metrics[k] = v

    # l1_data (nested dicts -> flattened with prefix)
    for section_val in l1_data.values():
        if isinstance(section_val, dict):
            for k, v in section_val.items():
                if v is not None:
                    metrics[f"l1_{k}"] = v

    # l2_data (nested dicts -> flattened with prefix)
    for section_val in l2_data.values():
        if isinstance(section_val, dict):
            for k, v in section_val.items():
                if v is not None:
                    metrics[f"l2_{k}"] = v

    # wavefront metrics (each value is [float, status_string])
    total_cycles = 0
    for cycle_key in ("Dependency Wait Cycles", "Issue Wait Cycles", "Active Cycles"):
        vals = wavefront.get(cycle_key, [])
        if vals and vals[0] is not None:
            total_cycles += vals[0]

    wf_mapping = {
        "VGPRs": "vgpr_count",
        "SGPRs": "sgpr_count",
        "AGPRs": "agpr_count",
        "LDS Allocation": "lds_alloc",
        "Scratch Allocation": "scratch_alloc",
        "Wavefront Occupancy": "wavefront_occupancy",
        "Instructions per wavefront": "instructions_per_wavefront",
    }
    for wf_key, metric_key in wf_mapping.items():
        vals = wavefront.get(wf_key, [])
        if vals and vals[0] is not None:
            metrics[metric_key] = vals[0]

    if total_cycles > 0:
        for cycle_key, metric_key in (
            ("Dependency Wait Cycles", "dependency_wait_pct"),
            ("Issue Wait Cycles", "issue_wait_pct"),
            ("Active Cycles", "active_cycle_pct"),
        ):
            vals = wavefront.get(cycle_key, [])
            if vals and vals[0] is not None:
                metrics[metric_key] = round(vals[0] / total_cycles * 100, 2)

    kernel_time_vals = wavefront.get("Kernel Time", [])
    kernel_time_ns = kernel_time_vals[0] if kernel_time_vals and kernel_time_vals[0] is not None else None

    # roofline rates: (value, peak, unit) tuples
    for name, (value, peak, _unit) in roofline_rates.items():
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        metrics[f"roofline.{safe_name}"] = value
        metrics[f"roofline.{safe_name}_peak"] = peak

    # roofline AI: (value, unit) tuples
    for name, (value, _unit) in roofline_ai.items():
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        metrics[f"roofline.{safe_name}"] = value

    # duration_us from wavefront Kernel Time (nanoseconds -> microseconds)
    duration_us = kernel_time_ns / 1000.0 if kernel_time_ns is not None else 0.0
    metrics["duration_us"] = duration_us

    bottleneck = _classify_rocprof_bottleneck(metrics)
    observations = _generate_rocprof_observations(metrics, bottleneck)

    # Build one kernel entry per top kernel; detailed metrics go on the first (dominant)
    if not top_kernels:
        top_kernels = ["unknown"]

    kernels = []
    for i, kname in enumerate(top_kernels):
        kernels.append(
            {
                "name": kname,
                "duration_us": duration_us if i == 0 else 0.0,
                "bottleneck": bottleneck if i == 0 else "unknown",
                "observations": observations if i == 0 else [],
                "metrics": metrics if i == 0 else {},
            }
        )

    return {
        "backend": "rocprof-compute",
        "results": [
            {
                "device_id": gpu_device,
                "gpu_info": gpu_info,
                "kernels": kernels,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Metrix backend
# ---------------------------------------------------------------------------


def _profile_with_metrix(command: str, gpu_devices, replays: int, quick: bool) -> dict:
    """Profile using MetrixTool and return backend-neutral JSON."""
    from metrix_mcp.core import MetrixTool

    tool = MetrixTool(gpu_devices=gpu_devices)
    result = tool.profile(
        command=command,
        num_replays=replays,
        kernel_filter=None,
        auto_select=False,
        quick=quick,
    )
    result["backend"] = "metrix"
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Profile GPU kernels (Metrix or rocprof-compute backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        help='Command to profile (e.g., "python3 kernel.py --profile")',
    )
    parser.add_argument(
        "--from-discovery",
        default=None,
        metavar="FILE",
        help="Read discovery.json and extract the test command for profiling",
    )
    parser.add_argument(
        "--backend",
        choices=["metrix", "rocprof-compute"],
        default="metrix",
        help="Profiling backend (default: metrix)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output result as JSON (for piping to baseline-metrics)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help="Write output to file instead of stdout (implies --json)",
    )
    parser.add_argument(
        "--gpu-devices",
        default="3",
        help='GPU device ID(s): single ("3") or comma-separated ("0,1,2") (default: 3)',
    )

    # Metrix-specific options
    metrix_group = parser.add_argument_group("metrix backend options")
    metrix_group.add_argument(
        "--replays",
        type=int,
        default=3,
        help="Number of profiling replays (default: 3)",
    )
    metrix_group.add_argument(
        "--quick",
        action="store_true",
        help="Quick profile (3 metrics, 1 pass) instead of full memory profile (14 metrics, 2 passes)",
    )

    # rocprof-compute-specific options
    rocprof_group = parser.add_argument_group("rocprof-compute backend options")
    rocprof_group.add_argument(
        "--workdir",
        default=None,
        metavar="DIR",
        help="Working directory for rocprof-compute (default: cwd)",
    )
    rocprof_group.add_argument(
        "--profiling-type",
        choices=["profiling", "roofline"],
        default="profiling",
        help="rocprof-compute analysis type (default: profiling)",
    )

    args = parser.parse_args()

    # Resolve the command to profile
    command = args.command
    if args.from_discovery:
        try:
            discovery_cmd = _extract_command_from_discovery(args.from_discovery)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        if not command:
            command = discovery_cmd
        print(f"[kernel-profile] Using command from discovery: {command}", file=sys.stderr)

    if not command:
        parser.error("command is required (positional or via --from-discovery)")

    use_json = args.output_json or args.output is not None

    # Dispatch via profiler-mcp (single code path for both backends)
    from profiler_mcp.server import profile_kernel

    _profile_fn = getattr(profile_kernel, "fn", profile_kernel)
    result = _profile_fn(
        command=command,
        backend=args.backend,
        workdir=args.workdir,
        profiling_type=args.profiling_type,
        num_replays=args.replays,
        quick=args.quick,
        gpu_devices=args.gpu_devices,
    )

    if use_json:
        output_text = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output_text + "\n")
            print(f"Wrote {args.output}", file=sys.stderr)
        else:
            print(output_text)
    else:
        for gpu_result in result["results"]:
            if len(result["results"]) > 1:
                device_id = gpu_result.get("device_id", "?")
                print(f"\n{'=' * 70}")
                print(f"GPU {device_id}")
                print(f"{'=' * 70}")
            _display_single_gpu_result(gpu_result)


def _display_single_gpu_result(result):
    """Display results for a single GPU."""
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

    if len(kernels) > 1:
        print(f"\n{'=' * 70}")
        print(f"Found {len(kernels)} kernels")
        print(f"{'=' * 70}\n")

    for i, kernel in enumerate(kernels):
        if len(kernels) > 1:
            print(f"[{i}] {kernel['name']}")
        else:
            print(f"\nKernel: {kernel['name']}")

        indent = "  " if len(kernels) > 1 else ""
        print(f"{indent}Bottleneck: {kernel['bottleneck']}")

        if kernel.get("observations"):
            print(f"{indent}Observations:")
            for obs in kernel["observations"]:
                print(f"{indent}  {obs}")

        if kernel.get("metrics"):
            metric_label = f"Metrics ({len(kernel['metrics'])} total):" if not indent else "Metrics:"
            print(f"{indent}{metric_label}")
            for name, value in sorted(kernel["metrics"].items()):
                if isinstance(value, float):
                    print(f"{indent}  {name}: {value:.2f}")
                else:
                    print(f"{indent}  {name}: {value}")
        else:
            print(f"{indent}No metrics captured")
        print()

    if len(kernels) > 1:
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
