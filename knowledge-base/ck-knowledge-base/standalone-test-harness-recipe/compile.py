#!/usr/bin/env python3
"""Build kernel .so files with GPU architecture auto-detected from PyTorch.

Usage:
    ./compile.py                # build all
    ./compile.py optimized      # rebuild only optimized kernel
    ./compile.py clean          # clean build artifacts
    ./compile.py --arch gfx942  # override auto-detection
"""

import argparse
import subprocess
import sys


def detect_arch() -> str:
    """Detect GPU architecture from PyTorch."""
    try:
        import torch

        if not torch.cuda.is_available():
            sys.exit("Error: No GPU available (torch.cuda.is_available() = False)")
        props = torch.cuda.get_device_properties(0)
        arch = props.gcnArchName.split(":")[0]
        return arch
    except ImportError:
        sys.exit(
            "Error: PyTorch not found. Install it or use: make ARCH=gfx950 directly"
        )


def main():
    parser = argparse.ArgumentParser(description="Build GEAK kernel .so files")
    parser.add_argument(
        "targets", nargs="*", default=["all"], help="Make targets (default: all)"
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="GPU architecture (default: auto-detect from PyTorch)",
    )
    args = parser.parse_args()

    arch = args.arch or detect_arch()
    print(f"GPU architecture: {arch}")

    cmd = ["make", f"ARCH={arch}"] + args.targets
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()