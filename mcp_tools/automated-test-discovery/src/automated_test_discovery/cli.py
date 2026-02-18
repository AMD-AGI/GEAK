"""CLI wrapper for automated-test-discovery MCP tool.

Enables calling MCP tools via bash commands.

Usage:
    test-discovery <kernel_path> [--max-tests <n>] [--max-benchmarks <n>]
    test-discovery --from-resolved resolved.json -o discovery.json

Examples:
    test-discovery /path/to/kernel.py
    test-discovery /path/to/repo/
    test-discovery /path/to/kernel.py --max-tests 10 --max-benchmarks 3
    test-discovery --from-resolved resolved.json -o discovery.json
"""

import argparse
import json
import sys
from pathlib import Path

from .server import discover


def main():
    parser = argparse.ArgumentParser(
        prog="test-discovery",
        description="Automated test and benchmark discovery for GPU kernels",
    )

    parser.add_argument(
        "kernel_path", nargs="?", default=None,
        help="Path to a kernel file (.py/.cu/.hip) or a repository directory",
    )
    parser.add_argument(
        "--from-resolved", default=None, metavar="FILE",
        help="Read resolved.json from resolve-kernel-url and extract kernel path",
    )
    parser.add_argument(
        "-o", "--output", default=None, metavar="FILE",
        help="Write JSON output to file instead of stdout",
    )
    parser.add_argument(
        "--max-tests", "-t", type=int, default=5, help="Max tests to return (default: 5)"
    )
    parser.add_argument(
        "--max-benchmarks", "-b", type=int, default=5, help="Max benchmarks to return (default: 5)"
    )

    args = parser.parse_args()

    # Resolve kernel_path from --from-resolved or positional arg
    kernel_path = args.kernel_path
    if args.from_resolved:
        resolved = json.loads(Path(args.from_resolved).read_text())
        kernel_path = resolved.get("local_file_path") or kernel_path
        if not kernel_path:
            print("ERROR: --from-resolved JSON has no local_file_path", file=sys.stderr)
            sys.exit(1)

    if not kernel_path:
        parser.error("kernel_path is required (positional or via --from-resolved)")

    result = discover(
        kernel_path=kernel_path,
        max_tests=args.max_tests,
        max_benchmarks=args.max_benchmarks,
    )

    output_text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output_text + "\n")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
