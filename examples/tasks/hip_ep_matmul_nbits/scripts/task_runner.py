#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Generic task runner for GEAK ONNX-Runtime EP kernel tasks.

Copied verbatim into each generated task's scripts/ by tools/gen_ep_task.py.
It is data-driven: all op/oracle specifics live in the sibling task_meta.json
(one dir up), so this file is identical across every EP task.

Modes (GEAK's COMMANDMENT calls these):
  compile      build the vendored kernel against its oracle
  correctness  run the oracle; PASS iff every case matches the reference
  performance  run the oracle; emit `Perf: <ms> ms (<case_id>)` lines

Two oracle families:
  fast     standalone hipcc harness (kernel.hip + test.cpp), no ORT/CMake.
           Seconds per iteration. Parses the harness's Median/PASS output.
  numeric  the EP's ONNX-vs-CPU pytest framework (test/numeric). Needs a
           fully built EP shared lib. Heavier; the fallback for ops with no
           fast harness.
"""
import argparse
import json
import os
import re
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(TASK_DIR, "build")


def _meta():
    with open(os.path.join(TASK_DIR, "task_meta.json")) as f:
        return json.load(f)


def _detect_offload_arch(default_arch):
    """Prefer the on-box GPU arch (rocminfo); fall back to the task's arch."""
    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=30
        ).stdout
        m = re.search(r"gfx\d+[a-z]*", out)
        if m:
            return m.group(0)
    except Exception:
        pass
    return default_arch


def _hipcc():
    return os.environ.get("HIPCC", "hipcc")


class _MissingTool:
    """Stand-in result when the executable itself is absent, so callers see a
    clean non-zero exit instead of an uncaught FileNotFoundError."""
    def __init__(self, msg):
        self.returncode = 127
        self.stdout = ""
        self.stderr = msg


def _run(cmd, **kw):
    print("+ " + " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, **kw)
    except FileNotFoundError as e:
        msg = f"tool not found: {cmd[0]} ({e})"
        print(msg, flush=True)
        return _MissingTool(msg)


# ----------------------------------------------------------------------------
# FAST oracle: direct hipcc build of kernel_src + harness, parse harness stdout.
# ----------------------------------------------------------------------------
def _fast_paths(meta):
    kernel = os.path.join(TASK_DIR, "kernel_src", meta["kernel_file"])
    test_cpp = os.path.join(TASK_DIR, "harness", meta["test_cpp"])
    inc = os.path.join(TASK_DIR, "include")
    exe = os.path.join(BUILD_DIR, "test_direct")
    return kernel, test_cpp, inc, exe


def fast_compile(meta):
    kernel, test_cpp, inc, exe = _fast_paths(meta)
    arch = _detect_offload_arch(meta["arch"])
    offload = f"--offload-arch={arch}"
    kobj = os.path.join(BUILD_DIR, "kernel.o")
    tobj = os.path.join(BUILD_DIR, "test.o")
    kflags = ["-O3", "-std=c++17", "-x", "hip", "-ffp-contract=fast", offload, f"-I{inc}"]
    tflags = ["-O3", "-std=c++17", "-x", "hip", offload, f"-I{inc}"]
    steps = [
        [_hipcc(), "-c", *tflags, test_cpp, "-o", tobj],
        [_hipcc(), "-c", *kflags, kernel, "-o", kobj],
        [_hipcc(), offload, tobj, kobj, "-o", exe],
    ]
    for cmd in steps:
        r = _run(cmd)
        if r.returncode != 0:
            return False, f"build step failed: {' '.join(cmd)}"
    return True, None


def _fast_run_shape(meta, exe, shape, data_root):
    """Run one shape; return (raw_stdout, passed:bool)."""
    M, K, N = shape["M"], shape["K"], shape["N"]
    gs = shape.get("gs", 128)
    no_zeros = shape.get("no_zeros", False)
    data_dir = os.path.join(data_root, f"{M}x{K}x{N}_gs{gs}_{'nz' if no_zeros else 'z'}")
    os.makedirs(data_dir, exist_ok=True)

    gen = os.path.join(TASK_DIR, "harness", meta["gen_data_py"])
    gen_cmd = [sys.executable, gen, f"{M}x{K}x{N}", "--group-size", str(gs), "--dir", data_dir]
    if no_zeros:
        gen_cmd.append("--no-zeros")
    r = _run(gen_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return r.stdout + r.stderr, False

    run_cmd = [exe, f"{M}x{K}x{N}", str(gs), data_dir]
    if no_zeros:
        run_cmd.append("--no-zeros")
    r = _run(run_cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out, flush=True)
    passed = r.returncode == 0 and "SOME FAILED" not in out and "ALL PASSED" in out
    return out, passed


def fast_correctness(meta):
    _, _, _, exe = _fast_paths(meta)
    data_root = os.path.join(BUILD_DIR, "data")
    for i, shape in enumerate(meta["shapes"]):
        out, passed = _fast_run_shape(meta, exe, shape, data_root)
        if not passed:
            return False, f"case {i} {shape} failed correctness"
    return True, None


# Section markers like "  --- u2 ---" tag the Median line that follows.
_SECTION_RE = re.compile(r"---\s*(u\d)\s*---")
_MEDIAN_RE = re.compile(r"Median:\s*([0-9.]+)\s*ms")


def fast_performance(meta):
    _, _, _, exe = _fast_paths(meta)
    data_root = os.path.join(BUILD_DIR, "data")
    targeted = {f"u{b}" for b in meta.get("bits", [])}  # only bits we optimize
    cases = []
    for shape in meta["shapes"]:
        out, _ = _fast_run_shape(meta, exe, shape, data_root)
        label = None
        for line in out.splitlines():
            sm = _SECTION_RE.search(line)
            if sm:
                label = sm.group(1)
                continue
            mm = _MEDIAN_RE.search(line)
            if mm and label is not None:
                if targeted and label not in targeted:
                    continue
                sid = f"{shape['M']}x{shape['K']}x{shape['N']}_gs{shape.get('gs',128)}" \
                      f"_{'nz' if shape.get('no_zeros') else 'z'}_{label}"
                cases.append({"test_case_id": sid, "execution_time_ms": float(mm.group(1)),
                              "params": {**shape, "bits": label}})
    return cases


# ----------------------------------------------------------------------------
# NUMERIC oracle: ORT ONNX-vs-CPU pytest over the built EP shared lib.
# ----------------------------------------------------------------------------
def numeric_compile(meta):
    """Build the EP shared lib. Heavy (CMake/LLVM). The exact command is
    environment-specific, so it is taken from task_meta.numeric.build_cmd
    (a shell string run in the EP repo). If unset, we assume a prebuilt DLL
    at numeric.ep_dll and skip building."""
    nm = meta["numeric"]
    build_cmd = nm.get("build_cmd")
    if not build_cmd:
        dll = nm.get("ep_dll")
        if dll and os.path.exists(dll):
            return True, None
        return False, "numeric oracle: no build_cmd and no prebuilt ep_dll present"
    r = _run(["bash", "-lc", build_cmd], cwd=nm["ep_repo"])
    return (r.returncode == 0), (None if r.returncode == 0 else "EP build failed")


def _numeric_pytest(meta, extra):
    nm = meta["numeric"]
    numeric_dir = os.path.join(nm["ep_repo"], "hip-ep", "test", "numeric")
    cmd = [sys.executable, "-m", "pytest", "-q",
           "-k", meta["numeric_test"].replace(".py", ""),
           "--backend", nm.get("backend", "ort_ep")]
    if nm.get("backend", "ort_ep") == "ort_ep":
        cmd += ["--ep-name", nm.get("ep_name", "AMDGPUExecutionProvider"),
                "--ep-dll", nm["ep_dll"]]
        for opt in nm.get("ep_options", []):
            cmd += ["--ep-option", opt]
    cmd += extra
    r = _run(cmd, cwd=numeric_dir, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out, flush=True)
    return r.returncode, out


def numeric_correctness(meta):
    rc, out = _numeric_pytest(meta, [])
    if rc == 0:
        return True, None
    return False, "numeric pytest reported failures (see output)"


def numeric_performance(meta):
    # The numeric framework is correctness-first; perf comes from --durations.
    rc, out = _numeric_pytest(meta, ["--durations=0"])
    cases = []
    for m in re.finditer(r"([0-9.]+)s\s+call\s+.*::(\S+)", out):
        cases.append({"test_case_id": m.group(2),
                      "execution_time_ms": float(m.group(1)) * 1000.0, "params": {}})
    return cases


# ----------------------------------------------------------------------------
DISPATCH = {
    "fast": (fast_compile, fast_correctness, fast_performance),
    "numeric": (numeric_compile, numeric_correctness, numeric_performance),
}


def main():
    ap = argparse.ArgumentParser(description="GEAK EP-kernel task runner")
    ap.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = ap.parse_args()
    os.makedirs(BUILD_DIR, exist_ok=True)
    meta = _meta()
    compile_fn, correct_fn, perf_fn = DISPATCH[meta["oracle"]]

    if args.mode == "compile":
        ok, err = compile_fn(meta)
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(BUILD_DIR, "compile_report.json"), "w"), indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    if args.mode == "correctness":
        ok, err = correct_fn(meta)
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(BUILD_DIR, "correctness_report.json"), "w"), indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    cases = perf_fn(meta)
    json.dump({"test_cases": cases},
              open(os.path.join(BUILD_DIR, "performance_report.json"), "w"), indent=2)
    for c in cases:
        print(f"Perf: {c['execution_time_ms']:.4f} ms ({c['test_case_id']})")
    sys.exit(0)


if __name__ == "__main__":
    main()
