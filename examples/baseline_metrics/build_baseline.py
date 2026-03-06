#!/usr/bin/env python3
"""Example: Build baseline_metrics.json from profiler output.

Usage:
    python examples/baseline_metrics/build_baseline.py profiler_output.json
    python examples/baseline_metrics/build_baseline.py profiler_output.json --kernels "rope_fwd,rope_bwd"
    python examples/baseline_metrics/build_baseline.py profiler_output.json --output baseline_metrics.json
"""

import argparse
import sys

from minisweagent.tools.baseline_metrics_tool import BaselineMetricsTool


def main():
    parser = argparse.ArgumentParser(description="Build baseline metrics from profiler output")
    parser.add_argument("profiler_json", help="Path to profiler JSON output file")
    parser.add_argument("--kernels", help="Comma-separated kernel names to include")
    parser.add_argument("--output", "-o", help="Output file path for baseline_metrics.json")
    args = parser.parse_args()

    from pathlib import Path

    profiler_output = Path(args.profiler_json).read_text()

    tool = BaselineMetricsTool()
    result = tool(
        profiler_output=profiler_output,
        kernel_names=args.kernels,
        output_path=args.output,
    )

    if result["returncode"] != 0:
        print(f"Error: {result['output']}", file=sys.stderr)
        sys.exit(1)

    print(result["output"])


if __name__ == "__main__":
    main()
