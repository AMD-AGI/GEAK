#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Generic task runner for GEAK ONNX-Runtime EP kernel tasks.

Copied verbatim into each generated task's scripts/ by tools/gen_ep_task.py, so
this file is identical across every EP task. Per-task data (kernel file, shapes,
oracle, EP paths) lives in the sibling task_meta.json (one dir up). The fast
oracle's shape fields (M/K/N/gs/no_zeros) and `--- u<bits> ---` section parsing
are specific to the matmul_nbits harness; a new fast op needs runner changes.

Modes (GEAK's COMMANDMENT calls these):
  compile      build the candidate kernel in kernel_src/ against its oracle
  correctness  run the oracle; PASS iff every case matches the reference
  performance  run the oracle; emit `Perf: <ms> ms (<case_id>)` lines. Exits
               non-zero (and writes a fail report) if any case fails, the oracle
               exits non-zero, or a case yields no timing.

Three oracle families:
  fast       standalone hipcc harness (kernel.hip + test.cpp) vendored from the
             EP's test/example dir, no ORT/CMake. Parses Median/PASS output.
  numeric    the EP's ONNX-vs-CPU pytest framework (test/numeric). `compile`
             overlays kernel_src/ into the EP tree, runs the configured EP
             build, and snapshots the EP library's directory (with the rebuilt
             kernel library) into build/ep/.
             Performance is pytest --durations wall time, not kernel latency.
  selfcheck  a self-contained editable-kernel + frozen-harness pair that carries
             its own CPU reference and GPU timing. No ORT, NumPy, or EP — a
             standalone HIP microbenchmark, not an EP-integration check.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(TASK_DIR, "build")


def _meta():
    with open(os.path.join(TASK_DIR, "task_meta.json")) as f:
        return json.load(f)


_ARCH_ENV_VARS = ("GPU_ARCHS", "PYTORCH_ROCM_ARCH")


def _resolve_offload_arch(task_arch):
    """Return (arch, err). The task's arch is authoritative; GPU_ARCHS /
    PYTORCH_ROCM_ARCH (exported by gpu_lock.sh for the locked GPU) must agree
    with it, otherwise we refuse rather than build for the wrong GPU."""
    for var in _ARCH_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if not val:
            continue
        env_archs = {a.split(":")[0] for a in re.split(r"[;,\s]+", val) if a}
        if task_arch not in env_archs:
            return None, (f"{var}={val} does not include the task arch {task_arch}; "
                          f"regenerate the task with --arch matching the locked GPU")
    return task_arch, None


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
# hipcc oracles (fast + selfcheck): kernel_src + frozen harness -> one exe.
# ----------------------------------------------------------------------------
def _paths(meta, exe_name):
    kernel = os.path.join(TASK_DIR, "kernel_src", meta["kernel_file"])
    test_cpp = os.path.join(TASK_DIR, "harness", meta["test_cpp"])
    inc = os.path.join(TASK_DIR, "include")
    exe = os.path.join(BUILD_DIR, exe_name)
    return kernel, test_cpp, inc, exe


def _hipcc_build(meta, exe_name, kernel_extra_flags=()):
    kernel, test_cpp, inc, exe = _paths(meta, exe_name)
    arch, err = _resolve_offload_arch(meta["arch"])
    if err:
        return False, err
    offload = f"--offload-arch={arch}"
    kobj = os.path.join(BUILD_DIR, "kernel.o")
    tobj = os.path.join(BUILD_DIR, "test.o")
    flags = ["-O3", "-std=c++17", "-x", "hip", offload, f"-I{inc}"]
    steps = [
        [_hipcc(), "-c", *flags, test_cpp, "-o", tobj],
        [_hipcc(), "-c", *flags, *kernel_extra_flags, kernel, "-o", kobj],
        [_hipcc(), offload, tobj, kobj, "-o", exe],
    ]
    for cmd in steps:
        r = _run(cmd)
        if r.returncode != 0:
            return False, f"build step failed: {' '.join(cmd)}"
    return True, None


# ----------------------------------------------------------------------------
# FAST oracle: direct hipcc build of kernel_src + harness, parse harness stdout.
# ----------------------------------------------------------------------------
_FAST_EXE = "test_direct"


def fast_compile(meta):
    # Matches KERNEL_FLAGS in the EP's test/example/gemm_fp16u*/Makefile, so the
    # vendored kernel is compiled (and rounds) the way the EP harness builds it.
    return _hipcc_build(meta, _FAST_EXE, kernel_extra_flags=["-ffp-contract=fast"])


def _fast_case_prefix(shape):
    return (f"{shape['M']}x{shape['K']}x{shape['N']}_gs{shape.get('gs', 128)}"
            f"_{'nz' if shape.get('no_zeros') else 'z'}")


def _fast_run_shape(meta, exe, shape, data_root):
    """Run one shape; return (raw_stdout, passed:bool)."""
    M, K, N = shape["M"], shape["K"], shape["N"]
    gs = shape.get("gs", 128)
    no_zeros = shape.get("no_zeros", False)
    data_dir = os.path.join(data_root, _fast_case_prefix(shape))
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
    _, _, _, exe = _paths(meta, _FAST_EXE)
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
    """Return (ok, cases, err)."""
    _, _, _, exe = _paths(meta, _FAST_EXE)
    data_root = os.path.join(BUILD_DIR, "data")
    targeted = {f"u{b}" for b in meta.get("bits", [])}  # only bits we optimize
    cases = []
    for i, shape in enumerate(meta["shapes"]):
        out, passed = _fast_run_shape(meta, exe, shape, data_root)
        if not passed:
            return False, cases, f"case {i} {shape} failed; its timing is not valid"
        shape_cases = []
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
                shape_cases.append({"test_case_id": f"{_fast_case_prefix(shape)}_{label}",
                                    "execution_time_ms": float(mm.group(1)),
                                    "params": {**shape, "bits": label}})
        if not shape_cases:
            return False, cases, f"case {i} {shape}: no Median timing parsed"
        cases += shape_cases
    return True, cases, None


# ----------------------------------------------------------------------------
# SELFCHECK oracle: standalone hipcc build of an editable kernel + a frozen
# harness that carries its own CPU reference and GPU timing. No ORT, no NumPy,
# no external harness dir — everything lives in the task. The harness's exit
# code is the verdict; it prints `GPU: <ms> ms` per shape.
# ----------------------------------------------------------------------------
_SELFCHECK_EXE = "test_selfcheck"


def selfcheck_compile(meta):
    return _hipcc_build(meta, _SELFCHECK_EXE)


def _selfcheck_run_shape(meta, exe, shape):
    """Run one shape; return (raw_stdout, passed:bool)."""
    label = f"{shape['M']}x{shape['N']}"
    iters = str(shape.get("iters", 50))
    r = _run([exe, label, iters], capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out, flush=True)
    return out, r.returncode == 0


def selfcheck_correctness(meta):
    _, _, _, exe = _paths(meta, _SELFCHECK_EXE)
    for i, shape in enumerate(meta["shapes"]):
        _, passed = _selfcheck_run_shape(meta, exe, shape)
        if not passed:
            return False, f"case {i} {shape} failed correctness"
    return True, None


_GPU_MS_RE = re.compile(r"^GPU:\s*([0-9.]+)\s*ms", re.MULTILINE)


def selfcheck_performance(meta):
    """Return (ok, cases, err)."""
    _, _, _, exe = _paths(meta, _SELFCHECK_EXE)
    cases = []
    for i, shape in enumerate(meta["shapes"]):
        out, passed = _selfcheck_run_shape(meta, exe, shape)
        if not passed:
            return False, cases, f"case {i} {shape} failed; its timing is not valid"
        m = _GPU_MS_RE.search(out)
        if not m:
            return False, cases, f"case {i} {shape}: no GPU timing parsed"
        cases.append({"test_case_id": f"{shape['M']}x{shape['N']}",
                      "execution_time_ms": float(m.group(1)),
                      "params": dict(shape)})
    return True, cases, None


# ----------------------------------------------------------------------------
# NUMERIC oracle: ORT ONNX-vs-CPU pytest over an EP shared lib rebuilt from the
# candidate in kernel_src/.
# ----------------------------------------------------------------------------
def _numeric_dll_snapshot(meta):
    return os.path.join(BUILD_DIR, "ep", os.path.basename(meta["numeric"]["ep_dll"]))


_SHARED_LIB_RE = re.compile(r"\.(so(\.\d+)*|dll)$")


def _snapshot_sources(ep_dll, kernel_lib):
    """basename -> source path for build/ep/: every shared lib co-located with
    ep_dll (the EP loads siblings such as its compiler and kernel libraries from
    its own directory), with the rebuilt kernel_lib replacing any same-named
    sibling."""
    ep_dir = os.path.dirname(ep_dll)
    libs = {f: os.path.join(ep_dir, f) for f in sorted(os.listdir(ep_dir))
            if _SHARED_LIB_RE.search(f) and os.path.isfile(os.path.join(ep_dir, f))}
    libs[os.path.basename(ep_dll)] = ep_dll
    libs[os.path.basename(kernel_lib)] = kernel_lib
    return libs


def _ep_build_lock_path(ep_repo):
    # Outside the EP tree so the lock never shows up as an untracked file there.
    key = hashlib.sha1(os.path.realpath(ep_repo).encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"geak_ep_build_{key}.lock")


def _mtime_ns(path):
    return os.stat(path).st_mtime_ns if os.path.exists(path) else None


def numeric_compile(meta):
    """Overlay kernel_src/<kernel_file> onto the EP tree, run build_cmd in the EP
    repo, check that kernel_lib (the library holding the compiled kernel) was
    rebuilt, and snapshot ep_dll's directory of shared libs into build/ep/ with
    the rebuilt kernel_lib in it, since the EP loads its kernel library from its
    own directory. The EP tree is shared
    by every workspace, so overlay+build+snapshot run under an exclusive lock and
    the original kernel source is restored afterwards; later modes load only this
    task's snapshot."""
    nm = meta["numeric"]
    build_cmd, dll = nm.get("build_cmd"), nm.get("ep_dll")
    if not build_cmd:
        return False, ("numeric oracle: no build_cmd configured, so the candidate in "
                       "kernel_src/ would never be compiled; regenerate with --ep-build-cmd")
    if not dll:
        return False, "numeric oracle: no ep_dll configured; regenerate with --ep-dll"
    kernel_lib = nm.get("kernel_lib") or dll

    src = os.path.join(TASK_DIR, "kernel_src", meta["kernel_file"])
    dst = nm["kernel_dst"]
    snap_dir = os.path.dirname(_numeric_dll_snapshot(meta))
    with open(_ep_build_lock_path(nm["ep_repo"]), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(dst, "rb") as f:
            original = f.read()
        try:
            # copyfile (not copy2) gives dst a fresh mtime, forcing a recompile.
            shutil.copyfile(src, dst)
            before = _mtime_ns(kernel_lib)
            r = _run(["bash", "-lc", build_cmd], cwd=nm["ep_repo"])
            if r.returncode != 0:
                return False, f"EP build failed (build_cmd exited {r.returncode})"
            after = _mtime_ns(kernel_lib)
            if after is None or after == before:
                return False, (f"build_cmd did not rebuild {kernel_lib}: it must be the "
                               f"library's link output (not a post-build copy, which is "
                               f"skipped when unchanged) and build_cmd must consume the "
                               f"overlaid {dst}")
            if not os.path.exists(dll):
                return False, f"EP library {dll} not found after build"
            shutil.rmtree(snap_dir, ignore_errors=True)
            os.makedirs(snap_dir)
            for name, lib in _snapshot_sources(dll, kernel_lib).items():
                shutil.copy2(lib, os.path.join(snap_dir, name))
        finally:
            with open(dst, "wb") as f:
                f.write(original)
    return True, None


def _numeric_pytest(meta, extra):
    nm = meta["numeric"]
    snapshot = _numeric_dll_snapshot(meta)
    if not os.path.exists(snapshot):
        return 1, f"no rebuilt EP library at {snapshot}; run `compile` first"
    numeric_dir = os.path.join(nm["ep_repo"], "test", "numeric")
    # Suite options must be single `--opt=value` tokens: pytest pre-parses argv
    # before loading conftest.py, so a separate absolute-path value is taken as a
    # test path, the suite's conftest is never loaded, and --backend is rejected.
    backend = nm.get("backend", "ort_ep")
    cmd = [sys.executable, "-m", "pytest", "-q",
           "-k", meta["numeric_test"].replace(".py", ""),
           f"--backend={backend}"]
    if backend == "ort_ep":
        cmd += [f"--ep-name={nm.get('ep_name', 'AMDGPUExecutionProvider')}", f"--ep-dll={snapshot}"]
        cmd += [f"--ep-option={opt}" for opt in nm.get("ep_options", [])]
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


_DURATION_RE = re.compile(r"([0-9.]+)s\s+call\s+.*::(\S+)")


def numeric_performance(meta):
    """Return (ok, cases, err). Timings are pytest --durations wall time per test
    (ORT session + graph execution), not kernel latency."""
    rc, out = _numeric_pytest(meta, ["--durations=0"])
    if rc != 0:
        return False, [], f"numeric pytest exited {rc}; timings are not valid"
    cases = [{"test_case_id": m.group(2),
              "execution_time_ms": float(m.group(1)) * 1000.0, "params": {}}
             for m in _DURATION_RE.finditer(out)]
    if not cases:
        return False, [], "no pytest --durations timings parsed"
    return True, cases, None


# ----------------------------------------------------------------------------
DISPATCH = {
    "fast": (fast_compile, fast_correctness, fast_performance),
    "numeric": (numeric_compile, numeric_correctness, numeric_performance),
    "selfcheck": (selfcheck_compile, selfcheck_correctness, selfcheck_performance),
}


def _write_report(name, report):
    with open(os.path.join(BUILD_DIR, name), "w") as f:
        json.dump(report, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="GEAK EP-kernel task runner")
    ap.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = ap.parse_args()
    os.makedirs(BUILD_DIR, exist_ok=True)
    meta = _meta()
    compile_fn, correct_fn, perf_fn = DISPATCH[meta["oracle"]]

    if args.mode in ("compile", "correctness"):
        fn = compile_fn if args.mode == "compile" else correct_fn
        ok, err = fn(meta)
        _write_report(f"{args.mode}_report.json",
                      {"status": "ok" if ok else "fail", "error": err})
        title = "Compilation" if args.mode == "compile" else "Correctness"
        print(f"{title}: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    ok, cases, err = perf_fn(meta)
    if not ok:
        _write_report("performance_report.json",
                      {"status": "fail", "error": err, "test_cases": []})
        print("Performance: FAIL")
        print(f"Error: {err}")
        sys.exit(1)
    _write_report("performance_report.json",
                  {"status": "ok", "error": None, "test_cases": cases})
    for c in cases:
        print(f"Perf: {c['execution_time_ms']:.4f} ms ({c['test_case_id']})")
    sys.exit(0)


if __name__ == "__main__":
    main()
