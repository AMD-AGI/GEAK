#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Generate a self-contained GEAK task dir for an ONNX-Runtime EP kernel.

Reads tools/ep_op_registry.yaml, vendors a COPY of the op's kernel .hip (the
editable optimization surface) plus the right oracle (fast hipcc harness or a
pointer to the numeric ONNX-vs-CPU framework), and emits a task dir GEAK can
drive via mode=optimize with zero JS changes.

Usage:
  python3 tools/gen_ep_task.py --op matmul_nbits --arch gfx1151 \
      --ep-repo /home/thpereir/projects/remote/ge/hip-ep-2bit-fpzp \
      --out examples/tasks/hip_ep_matmul_nbits

  python3 tools/gen_ep_task.py --op layer_norm --arch gfx1151 \
      --ep-repo <path> --ep-dll /path/to/amdgpu-ep.so \
      --out examples/tasks/hip_ep_layer_norm
"""
import argparse
import json
import os
import shutil
import sys

import yaml

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
GEAK_ROOT = os.path.dirname(TOOLS_DIR)
REGISTRY = os.path.join(TOOLS_DIR, "ep_op_registry.yaml")
RUNNER_SRC = os.path.join(TOOLS_DIR, "ep_task_runner.py")

# Named shape presets. (M, K, N, gs, no_zeros). K must be % 32 == 0 for the
# matmul_nbits fast path. Covers prefill (M>=16 -> WMMA) + decode (M=1 -> GEMV),
# with and without zero points.
SHAPE_PRESETS = {
    "matmul_nbits_default": [
        {"M": 128, "K": 2880, "N": 2880, "gs": 128, "no_zeros": False},   # prefill WMMA + zp
        {"M": 128, "K": 2880, "N": 5120, "gs": 128, "no_zeros": True},    # prefill WMMA no-zp
        {"M": 1,   "K": 2880, "N": 2880, "gs": 128, "no_zeros": False},   # decode GEMV + zp
        {"M": 1,   "K": 2880, "N": 5120, "gs": 128, "no_zeros": True},    # decode GEMV no-zp
    ],
    "numeric_default": [],  # numeric oracle drives its own shapes from the pytest file
    "layernorm_default": [
        {"M": 4096, "N": 2880, "iters": 50},   # prefill-ish batch
        {"M": 8192, "N": 4096, "iters": 50},   # larger batch
        {"M": 2048, "N": 8192, "iters": 50},   # wide rows
    ],
}

TEMPLATES_DIR = os.path.join(TOOLS_DIR, "selfcheck_templates")


def load_registry():
    with open(REGISTRY) as f:
        return yaml.safe_load(f)


def find_op(reg, op):
    for row in reg["ops"]:
        if row["op"] == op:
            return row
    sys.exit(f"op '{op}' not found in {REGISTRY}. Known: {[r['op'] for r in reg['ops']]}")


def ep_kernels_dir(ep_repo):
    return os.path.join(ep_repo, "hip-ep", "lib", "Runtime", "Kernels")


def gen_fast(row, arch, ep_repo, out):
    kdir = ep_kernels_dir(ep_repo)
    harness = row["fast_harness"]
    hdir = os.path.join(kdir, "test", "example", harness)

    # 1. vendor the editable kernel (the ONLY file GEAK edits)
    os.makedirs(os.path.join(out, "kernel_src"), exist_ok=True)
    shutil.copy2(os.path.join(kdir, "hip", row["kernel_hip"]),
                 os.path.join(out, "kernel_src", row["kernel_hip"]))

    # 2. vendor the ABI headers the kernel + test include
    inc_out = os.path.join(out, "include")
    os.makedirs(inc_out, exist_ok=True)
    for h in os.listdir(os.path.join(kdir, "include")):
        if h.endswith(".h"):
            shutil.copy2(os.path.join(kdir, "include", h), os.path.join(inc_out, h))

    # 3. vendor the harness (test driver + data gen), auto-discovering the file names
    hout = os.path.join(out, "harness")
    os.makedirs(hout, exist_ok=True)
    test_cpp = next(f for f in os.listdir(hdir)
                    if f.startswith("test_") and f.endswith(".cpp"))
    gen_data = next(f for f in os.listdir(hdir)
                    if f.startswith("gen_") and "data" in f and f.endswith(".py")
                    and "model" not in f)
    for f in (test_cpp, gen_data):
        shutil.copy2(os.path.join(hdir, f), os.path.join(hout, f))

    meta = {
        "op": row["op"], "oracle": "fast", "arch": arch,
        "kernel_file": row["kernel_hip"],
        "target_functions": row["entry_points"],
        "frozen_symbols": row["frozen_symbols"],
        "test_cpp": test_cpp, "gen_data_py": gen_data,
        "bits": row.get("bits", []),
        "shapes": SHAPE_PRESETS[row["shapes_preset"]],
    }
    return meta, os.path.join("kernel_src", row["kernel_hip"])


def gen_numeric(row, arch, ep_repo, out, ep_dll, ep_options):
    kdir = ep_kernels_dir(ep_repo)
    os.makedirs(os.path.join(out, "kernel_src"), exist_ok=True)
    shutil.copy2(os.path.join(kdir, "hip", row["kernel_hip"]),
                 os.path.join(out, "kernel_src", row["kernel_hip"]))
    inc_out = os.path.join(out, "include")
    os.makedirs(inc_out, exist_ok=True)
    for h in os.listdir(os.path.join(kdir, "include")):
        if h.endswith(".h"):
            shutil.copy2(os.path.join(kdir, "include", h), os.path.join(inc_out, h))

    meta = {
        "op": row["op"], "oracle": "numeric", "arch": arch,
        "kernel_file": row["kernel_hip"],
        "target_functions": row["entry_points"],
        "frozen_symbols": row["frozen_symbols"],
        "numeric_test": row["numeric_test"],
        "numeric": {
            "ep_repo": os.path.abspath(ep_repo),
            "backend": "ort_ep",
            "ep_name": "AMDGPUExecutionProvider",
            "ep_dll": ep_dll or "",
            "ep_options": ep_options,
            "build_cmd": "",  # fill in for in-loop EP rebuilds; empty => use prebuilt ep_dll
        },
        "shapes": [],
    }
    return meta, os.path.join("kernel_src", row["kernel_hip"])


def gen_selfcheck(row, arch, out):
    """Self-contained op: copy kernel + frozen harness + ABI header from the
    in-repo template (no EP repo involved)."""
    tdir = os.path.join(TEMPLATES_DIR, row["template"])
    if not os.path.isdir(tdir):
        sys.exit(f"selfcheck template dir not found: {tdir}")
    for sub in ("kernel_src", "harness", "include"):
        src = os.path.join(tdir, sub)
        dst = os.path.join(out, sub)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    meta = {
        "op": row["op"], "oracle": "selfcheck", "arch": arch,
        "kernel_file": row["kernel_hip"],
        "target_functions": row["entry_points"],
        "frozen_symbols": row["frozen_symbols"],
        "test_cpp": row["test_cpp"],
        "shapes": SHAPE_PRESETS[row["shapes_preset"]],
    }
    return meta, os.path.join("kernel_src", row["kernel_hip"])


def write_task(out, row, meta, source_rel):
    os.makedirs(os.path.join(out, "scripts"), exist_ok=True)
    shutil.copy2(RUNNER_SRC, os.path.join(out, "scripts", "task_runner.py"))
    json.dump(meta, open(os.path.join(out, "task_meta.json"), "w"), indent=2)

    config = {
        "source_file_path": [source_rel],
        "target_kernel_functions": row["entry_points"],
        "compile_command": ["python3 scripts/task_runner.py compile"],
        "correctness_command": ["python3 scripts/task_runner.py correctness"],
        "performance_command": ["python3 scripts/task_runner.py performance"],
        "task_type": "hip_ep2hip_ep",
        "prompt": {"source_code": None, "instructions": None, "cheatsheet": None},
    }
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

    frozen = "\n".join(f"  - `{s}`" for s in row["frozen_symbols"])
    targets = "\n".join(f"  - `{s}`" for s in row["entry_points"])
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(
            f"# GEAK task: {row['op']} ({meta['oracle']} oracle, {meta['arch']})\n\n"
            f"Optimizes the HIP EP `{row['op']}` kernel. GEAK edits ONLY "
            f"`{source_rel}`.\n\n"
            f"## Optimization targets (editable)\n{targets}\n\n"
            f"## FROZEN contract (must NOT change — the generated EP code calls these)\n"
            f"{frozen}\n\n"
            f"## Run\n```\npython3 scripts/task_runner.py compile\n"
            f"python3 scripts/task_runner.py correctness\n"
            f"python3 scripts/task_runner.py performance\n```\n"
        )


def main():
    ap = argparse.ArgumentParser(description="Generate a GEAK EP-kernel task dir")
    ap.add_argument("--op", required=True)
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--ep-repo", help="path to the hip-ep repo root "
                    "(required for fast/numeric oracles; unused for selfcheck)")
    ap.add_argument("--out", required=True, help="task dir to create")
    ap.add_argument("--oracle", choices=["fast", "numeric", "selfcheck"],
                    help="override the registry's oracle for this op")
    ap.add_argument("--ep-dll", help="(numeric) prebuilt EP shared lib path")
    ap.add_argument("--ep-option", action="append", default=[],
                    help="(numeric) repeatable KEY=VALUE for the EP")
    args = ap.parse_args()

    reg = load_registry()
    row = find_op(reg, args.op)
    oracle = args.oracle or row["oracle"]
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    if oracle == "selfcheck":
        meta, source_rel = gen_selfcheck(row, args.arch, out)
    elif oracle == "fast":
        if not args.ep_repo:
            sys.exit("--ep-repo is required for the fast oracle")
        meta, source_rel = gen_fast(row, args.arch, args.ep_repo, out)
    else:
        if not args.ep_repo:
            sys.exit("--ep-repo is required for the numeric oracle")
        meta, source_rel = gen_numeric(row, args.arch, args.ep_repo, out,
                                       args.ep_dll, args.ep_option)
    write_task(out, row, meta, source_rel)
    print(f"Generated {oracle} task for '{args.op}' at {out}")
    print(f"  source_file_path: {source_rel}")
    print(f"  targets: {', '.join(row['entry_points'])}")


if __name__ == "__main__":
    main()
