#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Generate a GEAK kernel task for a hip-ep op described in OPS. The one op
described is the 4-bit MatMulNBits prefill (WMMA) GEMM; the task README's notes,
the runner's leaf command line (leaf_runner.py) and the second oracle (qmoe) are
written for it, so another op needs those as well as its OPS entry.

The task mirrors the hip-ep tree under src/, so the files GEAK edits map one to
one onto an overlay for build_kernels.sh. It carries:
  - the editable sources: hip/matmul_nbits_kernel.hip and the two WMMA tile
    cores in include/;
  - frozen: the other kernel headers, the autotune resolver and the LUT it reads,
    and two of the release's own unit-test leaves as oracles: matmul_nbits
    Xfp16_Wu4_Yfp16, and qmoe_ragged_mm Xfp16_Wu4_Yfp16, whose kernel also
    instantiates the WMMA core (GemmFp16U4Impl) and whose test also calls the
    matmul_nbits small-M grouped entry;
  all taken from the release tag as committed (git archive), plus --overlay;
  - task_meta.json with the release's compile command for each source (read from
    the configured build with `ninja -t commands`, so the harness compiles the
    kernels exactly as the shipped library does) and the model's shapes;
  - config.yaml in the hip_ep2hip_ep format tools/gen_ep_task.py writes, and
    scripts/task_runner.py.
The shapes come from the model's decoder graph (lut_refresh.extract_shapes). The
hip-ep checkout, build dir and TheRock dist are HIPEP_SRC, HIPEP_BUILD and
HIPEP_THEROCK.

    python3 make_task.py --op OP --arch GFX --model M --out T [--m 1024,2048] [--overlay DIR]
    python3 make_task.py --workload-from T/build/performance_report.json --out T
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hipep_env  # noqa: E402
import lut_refresh  # noqa: E402

K = hipep_env.K
GUARD_M = (1,)   # decode-sized guards: every model shape at M=1 must stay correct
PERF_REPS = 1    # one cold-weight run per shape; repeats spread under 1% on the shapes that weigh
COVERAGE = 3     # the leaves' fullest coverage tier

# The ops this generator writes a task for. An op names its kernel source and the headers GEAK may
# edit with it, the functions to optimize and the C ABI that stays frozen, its autotune (resolver,
# generated header, table), the release unit-test leaf that is its oracle, and the other kernel
# that compiles its editable sources, with that kernel's own leaf as a second oracle.
LEAF = "Xfp16_Wu4_Yfp16"
OPS = {
    "matmul_nbits_u4_prefill": {
        "kernel": f"{K}/hip/matmul_nbits_kernel.hip",
        "headers": [f"{K}/include/matmul_nbits_wmma_core.h", f"{K}/include/matmul_nbits_wmma_core_x.h"],
        # what compiles the editable sources (for the check that every includer has an oracle)
        "includes": r"matmul_nbits_wmma_core(_x)?\.h|GemmFp16U4Impl",
        "targets": ["MatMulNBitsWMMA_NoZP", "MatMulNBitsWMMA_ZP", "MatMulNBitsWMMAX_NoZP",
                    "MatMulNBitsWMMAX_ZP", "GemmFp16U4Impl", "GemmFp16U4ImplX"],
        "frozen": ["hip_matmul_nbits", "hip_matmul_nbits_u4_wmma_config_count",
                   "hip_matmul_nbits_u4_wmma_config_desc", "hip_matmul_nbits_u4_wmma_run_config",
                   "hip_matmul_nbits_u4_wmma_ext_device"],
        "autotune": "matmul_nbits",
        # what the task times, for its README and workload.json
        "title": "hip-ep matmul_nbits u4 prefill GEMM",
        "bits": 4, "dtypes": ["fp16", "u4"], "phase": "prefill",
        "kernel_name": "MatMulNBitsWMMA (u4 prefill)", "short_name": "matmul_nbits_wmma",
        "classification": "gemm",
        "generated_header": "matmul_nbits_autotune_generated.h",
        "leaf": LEAF,
        "test_cpp": f"{K}/unit_test/matmul_nbits/{LEAF}/test_matmul_nbits.cpp",
        # qmoe_ragged_mm instantiates the WMMA core too, so its leaf also checks every edit.
        "qmoe": {"test_cpp": f"{K}/unit_test/qmoe_ragged_mm/{LEAF}/test_qmoe_ragged_mm.cpp",
                 "sources": [f"{K}/hip/qmoe_ragged_mm_kernel.hip", f"{K}/hip/qmoe_kernel.hip"]},
    },
}


def editable(op):
    return [op["kernel"], *op["headers"]]


LINUX_COMPAT = """\
// Linux stand-ins for the two MSVC CRT calls the unit-test leaf makes. Force-
// included, so the leaf itself stays byte-identical to the release.
#pragma once
#ifndef _WIN32
#include <climits>
#include <cstdlib>
#include <unistd.h>
inline int _putenv_s(const char* name, const char* value) { return setenv(name, value, 1); }
inline int _get_pgmptr(char** out) {
  static char buf[PATH_MAX];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n < 0) return -1;
  buf[n] = '\\0';
  *out = buf;
  return 0;
}
#endif
"""


def read_shapes(path, ms, bits, max_n=65536):
    """extract_shapes.py csv -> [{M, K, N, gs, zp, calls}] for every M in `ms`."""
    by_kn = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["bits"]) != bits or int(r["N"]) >= max_n:
                continue
            key = (int(r["K"]), int(r["N"]), int(r["block_size"]), r["has_zp"] == "1")
            by_kn[key] = by_kn.get(key, 0) + int(r["node_count"])
    return [{"M": m, "K": k, "N": n, "gs": gs, "zp": zp, "calls": c}
            for m in ms for (k, n, gs, zp), c in sorted(by_kn.items())]


def _copy(src_root, rel, out):
    dst = os.path.join(out, "src", rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(os.path.join(src_root, rel), dst)


def _copy_tree(src, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _check_includers(hipep_src, op):
    """The oracles cover every kernel that compiles the op's editable sources. A release that adds
    another one needs its leaf in the op's description first."""
    out = subprocess.run(["git", "-C", hipep_src, "grep", "-l", "-E", op["includes"], "HEAD", "--", K],
                         capture_output=True, text=True, check=False).stdout
    found = {line.split(":", 1)[1] for line in out.splitlines()
             if line.endswith((".hip", ".h", ".hpp", ".cpp", ".cc", ".cu", ".cuh", ".inc"))}
    q = op["qmoe"]
    others = sorted(found - set(editable(op)) - set(q["sources"]) - {q["test_cpp"]})
    if others:
        sys.exit(f"these sources also compile the editable sources and no oracle covers them: {others}")


def _flags(build, rel):
    """The release build's (compiler, flags) for one source of the kernel library."""
    return hipep_env.compile_flags(hipep_env.product_command(build, os.path.relpath(rel, K) + ".o"))


def generate(name, arch, hipep_src, build, shapes_csv, out, ms, therock, overlay=None):
    op = OPS[name]
    at = hipep_env.autotune(op["autotune"])
    _check_includers(hipep_src, op)
    os.makedirs(out, exist_ok=True)
    for rel in (f"{K}/include", at):
        if os.path.isdir(os.path.join(out, "src", rel)):
            shutil.rmtree(os.path.join(out, "src", rel))
    hipep_env.export_tree(hipep_src, os.path.join(out, "src"),
                          [f"{K}/include", at, op["kernel"], op["test_cpp"], op["qmoe"]["test_cpp"],
                           *op["qmoe"]["sources"]])
    if overlay:
        for root, _, files in os.walk(overlay):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), overlay)
                _copy(overlay, rel, out)
    _copy_tree(os.path.join(build, "_deps", "flatbuffers-src", "include"),
               os.path.join(out, "deps", "flatbuffers", "include"))
    os.makedirs(os.path.join(out, "deps", "gen"), exist_ok=True)
    shutil.copy2(os.path.join(build, "schemas", op["generated_header"]),
                 os.path.join(out, "deps", "gen"))
    os.makedirs(os.path.join(out, "harness"), exist_ok=True)
    with open(os.path.join(out, "harness", "linux_compat.h"), "w") as f:
        f.write(LINUX_COMPAT)

    autotune_cpp = f"{at}/{op['autotune']}_autotune.cpp"
    compiler, kflags = _flags(build, op["kernel"])
    _, aflags = _flags(build, autotune_cpp)
    offload = next(f for f in kflags if f.startswith("--offload-arch="))
    shapes = read_shapes(shapes_csv, ms, op["bits"])
    if not shapes:
        sys.exit(f"no bits=4 shapes with N < 65536 in {shapes_csv}")
    guards = read_shapes(shapes_csv, GUARD_M, op["bits"])
    text = ""
    for e in editable(op):
        with open(os.path.join(out, "src", e)) as f:
            text += f.read()
    missing = [t for t in op["targets"] if t not in text]
    if missing:
        sys.exit(f"target functions not in the sources: {missing}")

    meta = {
        "op": name, "leaf": op["leaf"], "arch": arch, "frozen": op["frozen"],
        "hipcc": compiler, "therock": therock, "offload_arch": offload,
        "kernel_flags": kflags, "autotune_flags": aflags,
        "test_flags": ["-O3", "-std=c++17", "-Wno-c23-extensions", offload,
                       "-D__HIP_PLATFORM_AMD__=1", "-D__HIP_ROCclr__=1"],
        "kernel_file": f"src/{op['kernel']}", "autotune_cpp": f"src/{autotune_cpp}",
        "test_cpp": f"src/{op['test_cpp']}", "compat_header": "harness/linux_compat.h",
        "include_dirs": [f"src/{K}/include", f"src/{K}/hip"],
        "autotune_dir": f"src/{at}", "generated_dir": "deps/gen",
        "flatbuffers_include": "deps/flatbuffers/include",
        "lut_fb": f"src/{at}/lut/{arch}.fb",
        "shapes": shapes, "guard_shapes": guards, "perf_reps": PERF_REPS,
        "coverage": COVERAGE, "lock_path": hipep_env.serving_lock_path(),
        "editable": [f"src/{e}" for e in editable(op)],
        # The QMoE tiles are fixed, and the matmul_nbits leaf forces every extended WMMA
        # entry, so the QMoE leaf links hip-ep's no-table stub (as the autotune sweep does).
        "qmoe": {"test_cpp": f"src/{op['qmoe']['test_cpp']}",
                 "sources": [{"file": f"src/{q}", "flags": _flags(build, q)[1]} for q in op["qmoe"]["sources"]],
                 "lut_stub": f"src/{at}/tools/empty_lut_data.cpp"},
    }
    with open(os.path.join(out, "task_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    os.makedirs(os.path.join(out, "scripts"), exist_ok=True)
    shutil.copy2(os.path.join(HERE, "leaf_runner.py"), os.path.join(out, "scripts", "task_runner.py"))
    shutil.copy2(os.path.join(HERE, "hipep_env.py"), os.path.join(out, "scripts", "hipep_env.py"))
    config = {
        "source_file_path": [f"src/{e}" for e in editable(op)],
        "target_kernel_functions": op["targets"],
        "compile_command": ["python3 scripts/task_runner.py compile"],
        "correctness_command": ["python3 scripts/task_runner.py correctness"],
        "performance_command": ["python3 scripts/task_runner.py performance"],
        "task_type": "hip_ep2hip_ep",
        "prompt": {"source_code": None, "instructions": None, "cheatsheet": None},
    }
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(readme(meta))
    return meta


def readme(meta):
    shapes = "\n".join(f"| {s['M']} | {s['K']} | {s['N']} | {s['gs']} | {'yes' if s['zp'] else 'no'} "
                       f"| {s['calls']} |" for s in meta["shapes"])
    return f"""# GEAK task: {OPS[meta['op']]['title']} ({meta['arch']})

Optimize the WMMA prefill path of hip-ep's MatMulNBits (4-bit weights, fp16
activations) for the shapes one model runs. The sources mirror the hip-ep tree
under `src/`, and the edited files are rebuilt into hip-ep's kernel library as is.

## Editable
{chr(10).join(f'- `{e}`' for e in meta['editable'])}

## Frozen
- The C ABI the EP and the leaf call: {', '.join(f'`{s}`' for s in meta['frozen'])}.
- Everything else under `src/`: the other kernel headers, the autotune resolver
  and its table, and the unit-test leaf (the correctness oracle).

## How it is checked
- `correctness` checks every shape below plus M=1 guards against the leaf's CPU
  reference, then runs the leaf's full coverage sweep and every extended WMMA
  entry, then the qmoe_ragged_mm leaf, which instantiates the same WMMA core.
  A config the autotune may pick for another shape must stay correct.

## Not timed here
hip-ep pads B's row stride for some K (`shouldPadRow` in `matmul_nbits_kernel.hip`) and
runs a padded-B variant there. The leaf times the unpadded layout, so
an edit's effect on that variant shows only end to end, in the A/B and the parity gate.

## How it is measured
- Compiled with the release's own compile command for each source
  (`task_meta.json`: `kernel_flags`).
- `performance` times each shape below with `HIPDNN_MATMUL_AUTOTUNE_MODE=online`,
  so the kernel's own autotune picks the fastest config of the edited code;
  weights are rotated through cold copies, as a model streams them from DRAM.
## Shapes (M, K, N, group size, zero points, calls per forward)
| M | K | N | gs | zp | calls |
|---|---|---|---|---|---|
{shapes}

## Run
```
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```
"""


def workload(task_dir, perf_report):
    """workload-v1 from the task's shapes and a baseline performance report."""
    with open(os.path.join(task_dir, "task_meta.json")) as f:
        meta = json.load(f)
    op = OPS[meta["op"]]
    with open(perf_report) as f:
        ms = {c["test_case_id"]: c["execution_time_ms"] for c in json.load(f)["test_cases"]}
    cases = []
    for s in meta["shapes"]:
        cid = f"M{s['M']}_K{s['K']}_N{s['N']}_gs{s['gs']}{'' if s['zp'] else '_noz'}"
        lat = ms[cid]
        cases.append({"dims": [s["M"], s["K"], s["N"], s["gs"]], "dtypes": op["dtypes"],
                      "count": s["calls"], "baseline_latency_ms": lat,
                      "weight": round(s["calls"] * lat * 1000, 3), "weight_source": "model",
                      "regime": op["phase"], "test_case_id": cid})
    total = sum(c["weight"] for c in cases) or 1.0
    for c in cases:
        c["weight_norm"] = round(c["weight"] / total, 6)
    cases.sort(key=lambda c: -c["weight"])
    return {"schema": "workload-v1", "total_gpu_time_ms": round(total / 1000, 4),
            "num_kernels": 1,
            "kernels": [{"name": op["kernel_name"], "short_name": op["short_name"],
                         "classification": op["classification"], "backend_guess": "hip", "editable": True,
                         "calls": sum(c["count"] for c in cases), "total_ms": round(total / 1000, 4),
                         "pct_gpu_time": 100.0, "num_cases": len(cases), "cases": cases,
                         "phase": op["phase"], "served_regimes": [op["phase"]]}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--op", choices=sorted(OPS), help="the op to write a task for")
    ap.add_argument("--arch", help="the device's gfx name; the task reads that autotune table")
    ap.add_argument("--model", help="the OGA model dir whose shapes the task times")
    ap.add_argument("--m", default="1024,2048")
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--workload-from", default=None,
                    help="write <out>/workload.json from a baseline performance_report.json")
    a = ap.parse_args()
    if a.workload_from:
        w = workload(a.out, a.workload_from)
        with open(os.path.join(a.out, "workload.json"), "w") as f:
            json.dump(w, f, indent=2)
        print(f"wrote {a.out}/workload.json ({len(w['kernels'][0]['cases'])} cases)")
        return
    if not (a.model and a.op and a.arch):
        ap.error("--op, --arch and --model are required")
    src = hipep_env.require("HIPEP_SRC")
    shapes_dir = os.path.abspath(a.out) + ".shapes"
    os.makedirs(shapes_dir, exist_ok=True)
    shapes, _ = lut_refresh.extract_shapes(src, a.model, shapes_dir, os.path.join(shapes_dir, "log"))
    meta = generate(a.op, a.arch, src, hipep_env.require("HIPEP_BUILD"), shapes, a.out,
                    [int(x) for x in a.m.split(",")], hipep_env.require("HIPEP_THEROCK"),
                    overlay=a.overlay)
    print(f"task {a.out}: {len(meta['shapes'])} shapes, {len(meta['guard_shapes'])} guards, "
          f"compiler {meta['hipcc']}")


if __name__ == "__main__":
    main()
