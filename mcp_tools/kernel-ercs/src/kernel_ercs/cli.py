# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""CLI wrapper for kernel-ercs MCP tools.

Enables calling MCP tools via bash commands.

Usage:
    kernel-ercs evaluate <kernel_file> [--model <name>]
    kernel-ercs reflect <kernel_file> --output <test_output> [--speedup <x>] [--status <passed|failed>]
    kernel-ercs specs
    kernel-ercs compat <kernel_file>
"""

import argparse
import json
import sys
from pathlib import Path

from .server import (
    check_kernel_compatibility,
    evaluate_kernel_quality,
    get_amd_gpu_specs,
    reflect_on_kernel_result,
)


def _call_tool(tool, *args, **kwargs):
    """Call MCP tool, unwrapping FunctionTool if needed."""
    if hasattr(tool, "fn"):
        return tool.fn(*args, **kwargs)
    else:
        return tool(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        prog="kernel-ercs", description="Kernel ERCS: Evaluation, Reflection, Compatibility, Specs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate kernel quality")
    eval_parser.add_argument("kernel_file", help="Path to kernel file")
    eval_parser.add_argument("--model", "-m", default="claude-sonnet-4-6", help="LLM model")

    # reflect command
    ref_parser = subparsers.add_parser("reflect", help="Reflect on kernel test results")
    ref_parser.add_argument("kernel_file", help="Path to kernel file")
    ref_parser.add_argument("--output", "-o", required=True, help="Test output (string or @file)")
    ref_parser.add_argument("--speedup", "-s", type=float, default=0.0, help="Measured speedup")
    ref_parser.add_argument(
        "--status", default="unknown", choices=["passed", "failed", "unknown"], help="Correctness status"
    )
    ref_parser.add_argument("--history", help="Optimization history summary")
    ref_parser.add_argument("--tried", help="Comma-separated list of tried strategies")
    ref_parser.add_argument("--model", "-m", default="claude-sonnet-4-6", help="LLM model")

    # specs command
    subparsers.add_parser("specs", help="Get AMD MI350X GPU specifications")

    # compat command
    compat_parser = subparsers.add_parser("compat", help="Check AMD GPU compatibility")
    compat_parser.add_argument("kernel_file", help="Path to kernel file")

    args = parser.parse_args()

    if args.command == "evaluate":
        kernel_code = Path(args.kernel_file).read_text()
        result = _call_tool(evaluate_kernel_quality, kernel_code=kernel_code, model=args.model)
    elif args.command == "reflect":
        kernel_code = Path(args.kernel_file).read_text()
        # Handle @file syntax for output
        if args.output.startswith("@"):
            test_output = Path(args.output[1:]).read_text()
        else:
            test_output = args.output
        result = _call_tool(
            reflect_on_kernel_result,
            kernel_code=kernel_code,
            test_output=test_output,
            speedup=args.speedup,
            correctness_status=args.status,
            history=args.history or "",
            tried_strategies=args.tried or "",
            model=args.model,
        )
    elif args.command == "specs":
        result = _call_tool(get_amd_gpu_specs)
    elif args.command == "compat":
        kernel_code = Path(args.kernel_file).read_text()
        result = _call_tool(check_kernel_compatibility, kernel_code=kernel_code)
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
