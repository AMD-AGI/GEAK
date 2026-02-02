"""CLI wrapper for automated-test-discovery MCP tool.

Enables calling MCP tools via bash commands.

Usage:
    test-discovery discover <kernel_path> [--max-tests <n>] [--max-benchmarks <n>]
    test-discovery <kernel_path>  # shorthand for discover
"""

import argparse
import json
import sys

from .server import discover


def main():
    parser = argparse.ArgumentParser(
        prog="test-discovery",
        description="Automated test and benchmark discovery for GPU kernels"
    )
    
    # Support both subcommand and direct usage
    parser.add_argument("kernel_path", nargs="?", help="Path to kernel file")
    parser.add_argument("--max-tests", "-t", type=int, default=5, help="Max tests to return")
    parser.add_argument("--max-benchmarks", "-b", type=int, default=5, help="Max benchmarks to return")
    
    # Also support subcommand style
    subparsers = parser.add_subparsers(dest="command")
    disc_parser = subparsers.add_parser("discover", help="Discover tests and benchmarks")
    disc_parser.add_argument("path", help="Path to kernel file")
    disc_parser.add_argument("--max-tests", "-t", type=int, default=5, help="Max tests to return")
    disc_parser.add_argument("--max-benchmarks", "-b", type=int, default=5, help="Max benchmarks to return")

    args = parser.parse_args()

    # Determine which path to use
    if args.command == "discover":
        kernel_path = args.path
        max_tests = args.max_tests
        max_benchmarks = args.max_benchmarks
    elif args.kernel_path:
        kernel_path = args.kernel_path
        max_tests = args.max_tests
        max_benchmarks = args.max_benchmarks
    else:
        parser.print_help()
        sys.exit(1)

    result = discover(
        kernel_path=kernel_path,
        max_tests=max_tests,
        max_benchmarks=max_benchmarks
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
