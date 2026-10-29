#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Generate a self-contained GEAK author-HIP task for an EP-ready kernel.

Reads tools/author_hip_registry.yaml and emits a task GEAK drives with
`mode=author target_language=hip`. The authored kernel_src/<op>_kernel.hip must
implement the frozen `extern "C" <launch_symbol>` declared in the op's ABI header,
so it drops straight into the ONNX-Runtime HIP EP. Correctness + speedup are graded
by a GENERIC Python oracle (author_hip_oracle.py) that hipcc-compiles the authored
.hip, binds the symbol via ctypes, and compares against a torch reference.

Usage:
  python3 tools/gen_author_hip_task.py --op rmsnorm --arch gfx1151 \
      --out examples/tasks/author_rmsnorm_hip
"""
import argparse
import json
import os
import re
import shutil
import sys

import yaml

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
GEAK_ROOT = os.path.dirname(TOOLS_DIR)
REGISTRY = os.path.join(TOOLS_DIR, "author_hip_registry.yaml")
ORACLE_SRC = os.path.join(TOOLS_DIR, "author_hip_oracle.py")
FREEZE_SRC = os.path.join(TOOLS_DIR, "author_hip_freeze.py")
REFS_DIR = os.path.join(TOOLS_DIR, "author_hip_refs")
HARNESS_LIB = os.path.join(GEAK_ROOT, "e2e_workflow", "scripts", "harness_lib.py")

# Reuse the EP framework's shape presets so author + optimize tasks share shapes.
sys.path.insert(0, TOOLS_DIR)
from gen_ep_task import SHAPE_PRESETS  # noqa: E402

_CTYPE = {"float32": "float", "float16": "__half", "bfloat16": "__hip_bfloat16"}
_DTSIZE = {"float32": "sizeof(float)", "float16": "sizeof(__half)",
           "bfloat16": "sizeof(__hip_bfloat16)"}


def load_registry():
    with open(REGISTRY) as f:
        return yaml.safe_load(f)


def find_op(reg, op):
    for row in reg["ops"]:
        if row["op"] == op:
            return row
    sys.exit(f"op '{op}' not found in {REGISTRY}. Known: {[r['op'] for r in reg['ops']]}")


REQUIRED_FIELDS = ("op", "abi_header", "launch_symbol", "reference", "shapes_preset",
                   "tol", "arches", "args")
ARG_KINDS = ("in_tensor", "out_tensor", "dim", "scalar_f32", "scalar_i32")


def validate_row(row):
    """Exit with a clear reason (not a traceback) on a row the generator or the
    oracle could not use."""
    op = row.get("op")
    missing = [k for k in REQUIRED_FIELDS if k not in row]
    if missing:
        sys.exit(f"op '{op}' is missing registry fields: {missing}")
    if row["shapes_preset"] not in SHAPE_PRESETS:
        sys.exit(f"op '{op}': unknown shapes_preset '{row['shapes_preset']}'")
    for what, path in (("abi_header", os.path.join(TOOLS_DIR, row["abi_header"])),
                       ("reference", os.path.join(REFS_DIR, f"{row['reference']}.py"))):
        if not os.path.isfile(path):
            sys.exit(f"op '{op}': {what} not found: {path}")
    in_dims = set()
    for a in row["args"]:
        kind = a.get("kind")
        if kind not in ARG_KINDS:
            sys.exit(f"op '{op}': arg '{a.get('name')}' has unknown kind '{kind}'; "
                     f"expected one of {list(ARG_KINDS)}")
        if kind in ("in_tensor", "out_tensor"):
            if a.get("dtype") not in _CTYPE or "shape" not in a:
                sys.exit(f"op '{op}': tensor arg '{a.get('name')}' needs a shape and a "
                         f"dtype in {list(_CTYPE)}")
            if kind == "in_tensor":
                in_dims.update(a["shape"])
        if kind.startswith("scalar") and "value" not in a:
            sys.exit(f"op '{op}': scalar arg '{a.get('name')}' needs a value")
    if not any(a["kind"] == "out_tensor" for a in row["args"]):
        sys.exit(f"op '{op}': needs at least one out_tensor to grade")
    # The oracle reads every dim off an input tensor's shape, and the cases off
    # the preset, so each symbolic dim must come from both.
    used = {a["name"] for a in row["args"] if a["kind"] == "dim"}
    used.update(d for a in row["args"] if a["kind"] == "out_tensor" for d in a["shape"])
    if used - in_dims:
        sys.exit(f"op '{op}': dims {sorted(used - in_dims)} appear in no in_tensor shape")
    for case in SHAPE_PRESETS[row["shapes_preset"]]:
        if in_dims - set(case):
            sys.exit(f"op '{op}': shapes_preset '{row['shapes_preset']}' case {case} "
                     f"lacks dims {sorted(in_dims - set(case))}")


def parse_signature(header_path, launch_symbol):
    """Extract the exact `extern \"C\" ... <launch_symbol>(...)` declaration
    (names included) from the frozen header, minus the trailing semicolon."""
    text = open(header_path).read()
    m = re.search(r'extern\s+"C"\s+[A-Za-z_][\w\s\*]*?\b'
                  + re.escape(launch_symbol) + r'\s*\([^;{]*?\)\s*;', text, re.S)
    if not m:
        sys.exit(f"could not find `extern \"C\" ... {launch_symbol}(...)` in {header_path}")
    return m.group(0).rstrip().rstrip(";").rstrip()


def gen_stub(row, header_basename):
    """A safe stub: defines the frozen symbol, zeroes each out_tensor so it
    compiles, links, runs, and cleanly FAILs until the author fills it in."""
    sig = parse_signature(os.path.join(TOOLS_DIR, row["abi_header"]), row["launch_symbol"])
    zeros = []
    for a in row["args"]:
        if a["kind"] == "out_tensor":
            extent = " * ".join(str(d) for d in a["shape"])
            zeros.append(f"  hipMemset({a['name']}, 0, (size_t)({extent}) * "
                         f"{_DTSIZE[a['dtype']]});")
    body = "\n".join(zeros) or "  // (no out_tensor)"
    return (
        f"// AUTHOR MODE: implement {row['launch_symbol']} from scratch.\n"
        f"// The oracle (unittest.py) hipcc-compiles this file, binds the frozen\n"
        f"// `extern \"C\" {row['launch_symbol']}` via ctypes, and grades it against a\n"
        f"// torch reference. Keep the symbol name + signature byte-identical to\n"
        f"// include/{header_basename} so the kernel drops into the ONNX HIP EP.\n"
        f'#include "../include/{header_basename}"\n'
        f"#include <hip/hip_runtime.h>\n\n"
        f"{sig} {{\n"
        f"  // TODO(author): write the real kernel. Stub zeroes the output(s).\n"
        f"{body}\n"
        f"}}\n"
    )


def materialize_cases(preset):
    # Author oracle needs sig + the dim keys per case; carry a stable seed.
    out = []
    for i, c in enumerate(SHAPE_PRESETS[preset]):
        dims = {k: v for k, v in c.items() if k not in ("iters",)}
        sig = "_".join(f"{k}{v}" for k, v in dims.items())
        out.append({"sig": sig, "seed": 42 + i, **dims})
    return out


def write_task(out, row, arch):
    validate_row(row)
    for sub in ("include", "kernel_src", "baseline_src", "scripts"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    header_basename = os.path.basename(row["abi_header"])
    shutil.copy2(os.path.join(TOOLS_DIR, row["abi_header"]),
                 os.path.join(out, "include", header_basename))

    with open(os.path.join(out, "kernel_src", f"{row['op']}_kernel.hip"), "w") as f:
        f.write(gen_stub(row, header_basename))

    ref = row["reference"]
    shutil.copy2(os.path.join(REFS_DIR, f"{ref}.py"),
                 os.path.join(out, "baseline_src", f"{ref}_ref.py"))

    shutil.copy2(ORACLE_SRC, os.path.join(out, "unittest.py"))
    shutil.copy2(FREEZE_SRC, os.path.join(out, "freeze.py"))
    shutil.copy2(HARNESS_LIB, os.path.join(out, "harness_lib.py"))

    dtype = next(a["dtype"] for a in row["args"] if a["kind"] == "in_tensor")
    meta = {
        "op": row["op"],
        "op_kind": row.get("op_kind", "elementwise"),
        "dtype": dtype,
        "arch": arch,
        "candidate_backends": ["hip"],
        "build": True,
        "launch_symbol": row["launch_symbol"],
        "entry": f"kernel_src/{row['op']}_kernel.hip:{row['launch_symbol']}",
        "baseline_callable": f"{ref}_ref:{ref}",
        "args": row["args"],
        "cases": materialize_cases(row["shapes_preset"]),
        "tol": row["tol"],
        "random_draws": 5,
        "baseline_src_sha256": "",
        "harness_lib_sha256": "",
        "unittest_sha256": "",
        "notes": (f"Author-HIP task. GEAK (mode=author, target_language=hip) writes "
                  f"kernel_src/{row['op']}_kernel.hip implementing the frozen extern \"C\" "
                  f"{row['launch_symbol']} (include/{header_basename}); the generic oracle "
                  f"hipcc-compiles it + ctypes-binds the symbol, graded vs a torch reference. "
                  f"Same ABI the EP calls; a PASS does not validate it inside the EP."),
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)

    # README
    frozen = f"`extern \"C\" {row['launch_symbol']}` (see include/{header_basename})"
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(
            f"# GEAK author-HIP task: {row['op']} ({arch})\n\n"
            f"GEAK authors `kernel_src/{row['op']}_kernel.hip` from scratch, implementing the\n"
            f"FROZEN {frozen}. The oracle (`unittest.py`) hipcc-compiles it, binds the symbol\n"
            f"via ctypes, and grades correctness + speedup against a torch reference\n"
            f"(`baseline_src/{ref}_ref.py`). The authored kernel implements the same ABI the\n"
            f"ONNX-Runtime HIP EP calls, so it can be dropped into the EP unchanged; a PASS\n"
            f"here grades it against torch only, so validating it inside the EP still needs\n"
            f"an EP build.\n\n"
            f"## Validate\n```\nGEAK_SELFTEST=1 python3 unittest.py   # oracle wiring, no kernel\n"
            f"python3 unittest.py                   # grade the authored kernel (needs a GPU)\n```\n"
        )

    # freeze integrity anchors (run the copied freeze.py in-place)
    import subprocess
    subprocess.run([sys.executable, os.path.join(out, "freeze.py")], check=True)


def main():
    ap = argparse.ArgumentParser(description="Generate a GEAK author-HIP task dir")
    ap.add_argument("--op", required=True)
    ap.add_argument("--arch", default="gfx1151")
    ap.add_argument("--out", required=True, help="task dir to create")
    args = ap.parse_args()

    reg = load_registry()
    row = find_op(reg, args.op)
    validate_row(row)
    if args.arch not in row["arches"]:
        sys.exit(f"--arch {args.arch} is not supported by '{args.op}'; "
                 f"registry arches: {row['arches']}")
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    write_task(out, row, args.arch)
    print(f"Generated author-HIP task for '{args.op}' at {out}")
    print(f"  authored file: kernel_src/{row['op']}_kernel.hip")
    print(f"  frozen symbol: {row['launch_symbol']}  (include/{os.path.basename(row['abi_header'])})")


if __name__ == "__main__":
    main()
