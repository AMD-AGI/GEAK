"""kernel-profile: Profile GPU kernels using MetrixTool (AMD ROCm).

CLI for hardware-level kernel profiling. Naming aligned with kernel-evolve and kernel-ercs.
All kernels are profiled and reported; the agent chooses which to use. --auto-select is not used.
"""
import argparse
import json
import sys
from pathlib import Path

# Repo root so MetrixTool is importable (src/minisweagent -> src -> repo root)
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from metrix_mcp.core import MetrixTool

EXAMPLES = """
Examples:
  %(prog)s 'python3 /path/to/kernel.py --profile'
  %(prog)s 'python3 kernel.py --profile' --gpu-devices 3
  %(prog)s 'python3 kernel.py --profile' --replays 5
  %(prog)s 'python3 kernel.py --profile' --quick
  %(prog)s 'python3 kernel.py --profile' --gpu-devices 0,1,2

Pipeline chaining (read test command from discovery output):
  %(prog)s --from-discovery discovery.json --json -o profile.json
  %(prog)s --from-discovery discovery.json --quick --json -o profile.json
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
        f"No profiling command found in {discovery_path}: "
        "need focused_test.focused_command or tests[0].command"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Profile a GPU kernel using MetrixTool (metrix)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument(
        "command", nargs="?", default=None,
        help='Command to profile (e.g., "python3 kernel.py --profile")',
    )
    parser.add_argument(
        "--from-discovery", default=None, metavar="FILE",
        help="Read discovery.json and extract the test command for profiling",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="Output raw MetrixTool result as JSON (for piping to baseline-metrics)",
    )
    parser.add_argument(
        "-o", "--output", default=None, metavar="FILE",
        help="Write output to file instead of stdout (implies --json)",
    )
    parser.add_argument(
        "--gpu-devices",
        default="3",
        help='GPU device ID(s): single ("3") or multiple comma-separated ("0,1,2") (default: 3)',
    )
    parser.add_argument(
        "--replays",
        type=int,
        default=3,
        help="Number of profiling replays (default: 3)",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="(Ignored: all kernels are always profiled and reported for the agent to use)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick profile (3 metrics, 1 pass) for speed. Default: memory profile (14 metrics, 2 passes)",
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

    # All kernels are profiled and reported; agent chooses which to use.
    args.auto_select = False

    # Parse GPU argument (support comma-separated list)
    gpu_devices = (
        args.gpu_devices.split(",") if "," in args.gpu_devices else args.gpu_devices
    )

    tool = MetrixTool(gpu_devices=gpu_devices)
    result = tool.profile(
        command=command,
        num_replays=args.replays,
        kernel_filter=None,
        auto_select=args.auto_select,
        quick=args.quick,
    )

    if use_json:
        output_text = json.dumps(result, indent=2)
        if args.output:
            Path(args.output).write_text(output_text + "\n")
            print(f"Wrote {args.output}", file=sys.stderr)
        else:
            print(output_text)
    else:
        # Display human-readable results (always a list, even for single GPU)
        for gpu_result in result["results"]:
            if len(result["results"]) > 1:
                device_id = gpu_result.get("device_id", "?")
                print(f"\n{'='*70}")
                print(f"GPU {device_id}")
                print(f"{'='*70}")
            _display_single_gpu_result(gpu_result, args.auto_select)


def _display_single_gpu_result(result, auto_select):
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

    if not auto_select and len(kernels) > 1:
        print(f"\n{'='*70}")
        print(f"Found {len(kernels)} kernels")
        print(f"{'='*70}\n")

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

    if not auto_select and len(kernels) > 1:
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
