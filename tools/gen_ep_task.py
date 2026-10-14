#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Generate a self-contained GEAK task dir for an ONNX-Runtime EP kernel.

Reads tools/ep_op_registry.yaml, vendors a COPY of the op's kernel .hip (the
editable optimization surface) plus the oracle the registry row names (fast
hipcc harness, numeric ONNX-vs-CPU framework, or a selfcheck template), and
emits a task dir GEAK can drive via mode=optimize with zero JS changes.

Usage:
  python3 tools/gen_ep_task.py --op matmul_nbits --arch gfx1151 \
      --ep-repo <hip-ep checkout> \
      --out examples/tasks/hip_ep_matmul_nbits

  python3 tools/gen_ep_task.py --op layer_norm --arch gfx1151 \
      --ep-repo <hip-ep checkout> --ep-dll <install>/lib/libhipgpu.so \
      --ep-option config_file=<install>/bin/morphizen_config.json \
      --ep-kernel-lib <build>/lib/Runtime/Kernels/libcustom_kernels_gfx1151.so \
      --ep-build-cmd "cmake --build <build> --target custom_kernels_gfx1151" \
      --out examples/tasks/hip_ep_layer_norm

  python3 tools/gen_ep_task.py --op layernorm_selfcheck --arch gfx950 \
      --out /tmp/hip_ep_layernorm_selfcheck
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
    # Qwen3-Next shapes (hidden=5120, head_size=256, eps=1e-6). Block RMSNorm
    # normalizes over hidden; the narrow N=256 row is the per-head QK-norm regime.
    "rmsnorm_default": [
        {"M": 4096, "N": 5120, "iters": 50},   # block RMSNorm, prefill-ish batch
        {"M": 8192, "N": 5120, "iters": 50},   # block RMSNorm, larger batch
        {"M": 8192, "N": 256,  "iters": 50},   # QK-norm regime (per-head, narrow rows)
    ],
}

TEMPLATES_DIR = os.path.join(TOOLS_DIR, "selfcheck_templates")

# Registry fields each oracle needs, beyond the common ones.
ORACLE_FIELDS = {
    "fast": ("kernel_hip", "fast_harness", "shapes_preset"),
    "numeric": ("kernel_hip", "numeric_test"),
    "selfcheck": ("template", "kernel_hip", "test_cpp", "shapes_preset"),
}
COMMON_FIELDS = ("op", "oracle", "entry_points", "frozen_symbols", "arches")


def load_registry():
    with open(REGISTRY) as f:
        return yaml.safe_load(f)


def find_op(reg, op):
    for row in reg["ops"]:
        if row["op"] == op:
            return row
    sys.exit(f"op '{op}' not found in {REGISTRY}. Known: {[r['op'] for r in reg['ops']]}")


def validate_row(row):
    oracle = row.get("oracle")
    if oracle not in ORACLE_FIELDS:
        sys.exit(f"op '{row.get('op')}': unknown oracle '{oracle}' in {REGISTRY}; "
                 f"expected one of {sorted(ORACLE_FIELDS)}")
    missing = [k for k in COMMON_FIELDS + ORACLE_FIELDS[oracle] if k not in row]
    if missing:
        sys.exit(f"op '{row['op']}' ({oracle} oracle) is missing registry fields: {missing}")
    if "shapes_preset" in row and row["shapes_preset"] not in SHAPE_PRESETS:
        sys.exit(f"op '{row['op']}': unknown shapes_preset '{row['shapes_preset']}'")


def _require_dir(path, what):
    if not os.path.isdir(path):
        sys.exit(f"{what} not found: {path}")


def _require_file(path, what):
    if not os.path.isfile(path):
        sys.exit(f"{what} not found: {path}")


def ep_kernels_dir(ep_root):
    return os.path.join(ep_root, "lib", "Runtime", "Kernels")


def resolve_ep_root(ep_repo):
    """Return the hip-ep source root (the dir holding lib/ and test/). Accepts a
    ROCm/hip-ep checkout as-is, or a repo that nests it under hip-ep/ (e.g. the
    hip-ep-2bit-fpzp fork)."""
    _require_dir(ep_repo, "--ep-repo")
    for root in (ep_repo, os.path.join(ep_repo, "hip-ep")):
        if os.path.isdir(ep_kernels_dir(root)):
            return os.path.abspath(root)
    sys.exit(f"--ep-repo {ep_repo} has no lib/Runtime/Kernels (nor hip-ep/lib/Runtime/Kernels)")


def _vendor_ep_kernel(row, ep_repo, out):
    """Copy the editable kernel (the ONLY file GEAK edits) and the ABI headers it
    includes from the EP repo into the task. Returns the kernel's EP-tree path."""
    kdir = ep_kernels_dir(ep_repo)
    kernel = os.path.join(kdir, "hip", row["kernel_hip"])
    inc_dir = os.path.join(kdir, "include")
    _require_file(kernel, "EP kernel source")
    _require_dir(inc_dir, "EP kernel include dir")

    os.makedirs(os.path.join(out, "kernel_src"), exist_ok=True)
    shutil.copy2(kernel, os.path.join(out, "kernel_src", row["kernel_hip"]))
    inc_out = os.path.join(out, "include")
    os.makedirs(inc_out, exist_ok=True)
    for h in os.listdir(inc_dir):
        if h.endswith(".h"):
            shutil.copy2(os.path.join(inc_dir, h), os.path.join(inc_out, h))
    return kernel


def _find_one(hdir, pred, what):
    matches = sorted(f for f in os.listdir(hdir) if pred(f))
    if not matches:
        sys.exit(f"no {what} found in fast harness dir {hdir}")
    return matches[0]


def gen_fast(row, arch, ep_repo, out):
    hdir = os.path.join(ep_kernels_dir(ep_repo), "test", "example", row["fast_harness"])
    _require_dir(hdir, "fast harness dir")
    test_cpp = _find_one(hdir, lambda f: f.startswith("test_") and f.endswith(".cpp"),
                         "test_*.cpp driver")
    gen_data = _find_one(hdir, lambda f: f.startswith("gen_") and "data" in f
                         and f.endswith(".py") and "model" not in f,
                         "gen_*data*.py generator")

    _vendor_ep_kernel(row, ep_repo, out)
    hout = os.path.join(out, "harness")
    os.makedirs(hout, exist_ok=True)
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


def gen_numeric(row, arch, ep_repo, out, ep_dll, ep_build_cmd, ep_options,
                ep_kernel_lib=None):
    kernel_dst = _vendor_ep_kernel(row, ep_repo, out)

    meta = {
        "op": row["op"], "oracle": "numeric", "arch": arch,
        "kernel_file": row["kernel_hip"],
        "target_functions": row["entry_points"],
        "frozen_symbols": row["frozen_symbols"],
        "numeric_test": row["numeric_test"],
        "numeric": {
            "ep_repo": os.path.abspath(ep_repo),
            "kernel_dst": os.path.abspath(kernel_dst),
            "build_cmd": ep_build_cmd,
            "ep_dll": os.path.abspath(ep_dll),
            "kernel_lib": os.path.abspath(ep_kernel_lib or ep_dll),
            "backend": "ort_ep",
            "ep_name": "AMDGPUExecutionProvider",
            "ep_options": ep_options,
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
        _require_dir(src, f"selfcheck template {sub}/")
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


def _readme(row, meta, source_rel):
    oracle = meta["oracle"]
    frozen = "\n".join(f"  - `{s}`" for s in row["frozen_symbols"])
    targets = "\n".join(f"  - `{s}`" for s in row["entry_points"])
    if oracle == "selfcheck":
        title = f"# GEAK task: {row['op']} (standalone HIP microbenchmark, {meta['arch']})\n\n"
        intro = (f"Standalone HIP microbenchmark: GEAK edits ONLY `{source_rel}`, which is "
                 f"built with a frozen harness that checks it against an in-harness CPU "
                 f"reference. No ONNX Runtime or EP is involved, so a PASS here does NOT "
                 f"validate EP integration.\n\n")
        caller = "the frozen harness"
    else:
        title = f"# GEAK task: {row['op']} ({oracle} oracle, {meta['arch']})\n\n"
        intro = f"Optimizes the HIP EP `{row['op']}` kernel. GEAK edits ONLY `{source_rel}`.\n\n"
        caller = "the generated EP code"
    notes = ""
    if oracle == "numeric":
        notes = (
            "## Notes\n"
            "- `compile` overlays the candidate onto the EP tree, runs the configured "
            "EP build, checks the kernel library was rebuilt, copies the shared libraries "
            "next to the EP library plus the rebuilt kernel library into `build/ep/`, and "
            "restores the EP tree's original kernel. Correctness/performance load only "
            "that snapshot.\n"
            "- `performance` reports pytest `--durations` wall time per test (ORT session "
            "setup + graph execution), NOT kernel latency. Treat it as a coarse signal.\n\n"
        )
    return (
        title + intro +
        f"## Optimization targets (editable)\n{targets}\n\n"
        f"## FROZEN contract\n"
        f"These signatures (declared in `include/`) are what {caller} calls; they must "
        f"NOT change. Their implementations in `kernel_src/` remain editable.\n"
        f"{frozen}\n\n"
        + notes +
        "## Run\n```\npython3 scripts/task_runner.py compile\n"
        "python3 scripts/task_runner.py correctness\n"
        "python3 scripts/task_runner.py performance\n```\n"
    )


def write_task(out, row, meta, source_rel):
    os.makedirs(os.path.join(out, "scripts"), exist_ok=True)
    shutil.copy2(RUNNER_SRC, os.path.join(out, "scripts", "task_runner.py"))
    with open(os.path.join(out, "task_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

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

    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(_readme(row, meta, source_rel))


def main():
    ap = argparse.ArgumentParser(description="Generate a GEAK EP-kernel task dir")
    ap.add_argument("--op", required=True)
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--ep-repo", help="hip-ep source root (a ROCm/hip-ep checkout, or a "
                    "repo nesting it under hip-ep/); required for fast/numeric oracles, "
                    "unused for selfcheck")
    ap.add_argument("--out", required=True, help="task dir to create")
    ap.add_argument("--ep-dll", help="(numeric) EP shared lib registered with ORT; the "
                    "shared libs in its directory are snapshotted alongside it")
    ap.add_argument("--ep-build-cmd", help="(numeric) shell command, run in the hip-ep "
                    "source root, "
                    "that rebuilds the EP from the EP tree")
    ap.add_argument("--ep-kernel-lib", help="(numeric) link output of the library holding "
                    "the compiled kernel, which the EP loads from its own directory (HIP "
                    "EP: <build>/lib/Runtime/Kernels/libcustom_kernels_<arch>.so). Use the "
                    "link output, not a post-build copy: copies are skipped when the bytes "
                    "are unchanged. Defaults to --ep-dll")
    ap.add_argument("--ep-option", action="append", default=[],
                    help="(numeric) repeatable KEY=VALUE for the EP")
    args = ap.parse_args()

    row = find_op(load_registry(), args.op)
    validate_row(row)
    oracle = row["oracle"]
    if args.arch not in row["arches"]:
        sys.exit(f"--arch {args.arch} is not supported by '{args.op}'; "
                 f"registry arches: {row['arches']}")
    if oracle != "selfcheck":
        if not args.ep_repo:
            sys.exit(f"--ep-repo is required for the {oracle} oracle")
        args.ep_repo = resolve_ep_root(args.ep_repo)
    if oracle == "numeric":
        if not args.ep_build_cmd:
            sys.exit("--ep-build-cmd is required for the numeric oracle: without it the "
                     "candidate in kernel_src/ is never compiled into the EP")
        if not args.ep_dll:
            sys.exit("--ep-dll is required for the numeric oracle (the library "
                     "--ep-build-cmd produces)")

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    if oracle == "selfcheck":
        meta, source_rel = gen_selfcheck(row, args.arch, out)
    elif oracle == "fast":
        meta, source_rel = gen_fast(row, args.arch, args.ep_repo, out)
    else:
        meta, source_rel = gen_numeric(row, args.arch, args.ep_repo, out,
                                       args.ep_dll, args.ep_build_cmd, args.ep_option,
                                       args.ep_kernel_lib)
    write_task(out, row, meta, source_rel)
    print(f"Generated {oracle} task for '{args.op}' at {out}")
    print(f"  source_file_path: {source_rel}")
    print(f"  targets: {', '.join(row['entry_points'])}")


if __name__ == "__main__":
    main()
