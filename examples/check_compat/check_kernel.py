#!/usr/bin/env python3
"""Example: Scan a kernel file for CUDA-only patterns.

Usage:
    python examples/check_compat/check_kernel.py /path/to/kernel.py
    python examples/check_compat/check_kernel.py --code "cudaMalloc(&ptr, size);"
"""

import argparse
import sys

from minisweagent.tools.check_compat import CheckKernelCompatibilityTool


def main():
    parser = argparse.ArgumentParser(description="Check kernel AMD compatibility")
    parser.add_argument("file", nargs="?", help="Path to kernel file")
    parser.add_argument("--code", help="Kernel code as string")
    args = parser.parse_args()

    if not args.file and not args.code:
        parser.error("Provide either a file path or --code")

    tool = CheckKernelCompatibilityTool()
    result = tool(kernel_code=args.code, file_path=args.file)
    print(result["output"])
    sys.exit(0 if result["returncode"] == 0 else 1)


if __name__ == "__main__":
    main()
