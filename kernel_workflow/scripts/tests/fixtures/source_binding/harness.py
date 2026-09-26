# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic harness using the faulty source-link logic reported in #480.

The oracle expects 2*x. A candidate computing 3*x must fail even though it builds
and runs successfully. The source audit observes the bug, but deliberately does
not gate the verdict: that missing gate is part of the old harness's behavior.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE = "scale_kernel.cu"
OVERLAY = Path(".overlay/candidate/aiter_meta/csrc/kernels")


def _mirror(src_dir, dst_dir, override):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        target = override.get(child.name, child)
        link = dst_dir / child.name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(target)
    for name, target in override.items():
        link = dst_dir / name
        if not (link.exists() or link.is_symlink()):
            link.symlink_to(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    workspace = Path.cwd()
    candidate = workspace / SOURCE
    _mirror(args.vendor, workspace / OVERLAY, {SOURCE: candidate})
    compiler_input = workspace / OVERLAY / SOURCE
    build = workspace / "build"
    build.mkdir(exist_ok=True)
    (build / "compile_started").touch()
    report = {
        "candidate": str(candidate),
        "resolved_input": str(compiler_input.resolve()),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(compiler_input.read_bytes()).hexdigest(),
    }
    flags = (
        ["--offload-arch=gfx950"]
        if "hipcc" in Path(args.compiler).name
        else ["-x", "c++"]
    )
    executable = build / "kernel"
    # Always compile: this reproduces wrong source input, not a stale binary.
    subprocess.run(
        [
            args.compiler,
            *flags,
            "-std=c++17",
            "-O2",
            str(compiler_input),
            "-o",
            str(executable),
        ],
        check=True,
    )
    executed = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True
    )
    report.update(json.loads(executed.stdout))
    report["expected"] = [2.0, 4.0, -6.0, 1.0]
    report["correctness"] = "pass" if report["output"] == report["expected"] else "fail"
    report["benchmark_eligible"] = report["correctness"] == "pass"
    (build / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["correctness"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
