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
  export-patch (numeric only, not called by GEAK) write build/hip_ep.patch, the
               candidate as a git diff against the EP commit the task was
               generated from, and check that it applies there and upstream.

Three oracle families:
  fast       standalone hipcc harness (kernel.hip + test.cpp) vendored from the
             EP's test/example dir, no ORT/CMake. Parses Median/PASS output.
  numeric    the EP's ONNX-vs-CPU pytest framework (test/numeric). `compile`
             rejects candidates that change the frozen ABI or dispatch gates
             (scripts/ep_abi.py), overlays kernel_src/ into the EP tree, runs the
             configured EP build, and snapshots the EP library's directory (with
             the rebuilt kernel library and a manifest of what it was built from)
             into build/ep/. Correctness/performance refuse a snapshot that does
             not match kernel_src/, and run with HIPDNN_* pinned to the task's
             ep_env. With a `perf` config, performance is the EP's own per-op GPU
             time from harness/ep_perf_driver.py, whose cases correctness also
             checks against ORT's CPU EP; without one it is pytest --durations
             wall time, not kernel latency.
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
import signal
import subprocess
import sys
import tempfile
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(SCRIPTS_DIR)
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


_BUILD_STOP_GRACE_S = 10


def _run_build(build_cmd, cwd):
    """Run build_cmd in its own process group and return its exit code. If the
    runner is interrupted while it builds, the whole group is stopped before the
    caller restores the EP tree and releases the lock: killing only the shell
    would leave its children (a `docker run` client, say) writing the EP
    library after the restore."""
    cmd = ["bash", "-lc", build_cmd]
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
    try:
        return p.wait()
    except BaseException:
        _stop_process_group(p)
        raise


def _stop_process_group(p):
    """SIGTERM p's process group, SIGKILL what is left after the grace period,
    and return once the group is empty."""
    # The runner is exiting; a second SIGTERM must not cut the wait short.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + _BUILD_STOP_GRACE_S
        while time.monotonic() < deadline:
            p.poll()
            try:
                os.killpg(p.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)


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


_SNAPSHOT_MANIFEST = "manifest.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _kernel_src(meta):
    return os.path.join(TASK_DIR, "kernel_src", meta["kernel_file"])


def _check_snapshot(meta):
    """Error unless build/ep/ is what `compile` built from the current kernel_src/,
    so a stale or hand-edited snapshot is never measured as the candidate."""
    snap_dir = os.path.dirname(_numeric_dll_snapshot(meta))
    path = os.path.join(snap_dir, _SNAPSHOT_MANIFEST)
    if not os.path.exists(path):
        return f"no rebuilt EP snapshot in {snap_dir}; run `compile` first"
    with open(path) as f:
        man = json.load(f)
    if man.get("kernel_sha256") != _sha256(_kernel_src(meta)):
        return (f"kernel_src/{meta['kernel_file']} changed since the last `compile`; "
                f"run `compile` again so the EP snapshot is built from it")
    for name, digest in man.get("libs", {}).items():
        lib = os.path.join(snap_dir, name)
        if not os.path.exists(lib) or _sha256(lib) != digest:
            return f"EP snapshot file {name} changed after `compile`; run `compile` again"
    return None


def _ep_env(meta):
    """Runtime env for EP sessions: every inherited HIPDNN_* knob is dropped, so
    path selection, autotune and debug output depend only on the task's ep_env."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("HIPDNN_")}
    env.update({k: str(v) for k, v in meta["numeric"].get("ep_env", {}).items()})
    return env


def _ep_abi():
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    import ep_abi
    return ep_abi


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


def _kernel_backup_path(dst):
    # Holds the EP tree's own kernel while a candidate is overlaid on it: a compile
    # killed mid-build never reaches its `finally`, and the next one restores this.
    key = hashlib.sha1(os.path.realpath(dst).encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"geak_ep_kernel_{key}.orig")


def _mtime_ns(path):
    return os.stat(path).st_mtime_ns if os.path.exists(path) else None


def _abi_violations(meta, candidate):
    """Frozen-contract violations in the candidate source (empty when it only
    changes kernel bodies), or None for tasks generated without an ABI manifest."""
    if not meta.get("abi"):
        return None
    return _ep_abi().check_source(candidate.decode("utf-8", errors="replace"), meta["abi"])


def _export_violations(meta, kernel_lib):
    want = (meta.get("abi") or {}).get("exports")
    if want is None:
        return []
    abi = _ep_abi()
    try:
        got = abi.exported_c_symbols(kernel_lib, nm=os.environ.get("NM", "nm"))
    except (OSError, RuntimeError) as e:
        return [f"cannot list the exports of {kernel_lib}: {e}"]
    return abi.check_exports(got, want)


def numeric_compile(meta):
    """Reject candidates that change the frozen ABI or dispatch gates, overlay
    kernel_src/<kernel_file> onto the EP tree, run build_cmd in the EP repo, check
    that kernel_lib (the library holding the compiled kernel) was rebuilt and
    exports exactly the baseline's C symbols, and snapshot ep_dll's directory of
    shared libs into build/ep/ with the rebuilt kernel_lib in it, since the EP
    loads its kernel library from its own directory. The EP tree is shared by
    every workspace, so overlay+build+snapshot run under an exclusive lock and the
    original kernel source is restored afterwards; later modes load only this
    task's snapshot, after checking its manifest against kernel_src/."""
    nm = meta["numeric"]
    build_cmd, dll = nm.get("build_cmd"), nm.get("ep_dll")
    if not build_cmd:
        return False, ("numeric oracle: no build_cmd configured, so the candidate in "
                       "kernel_src/ would never be compiled; regenerate with --ep-build-cmd")
    if not dll:
        return False, "numeric oracle: no ep_dll configured; regenerate with --ep-dll"
    _, err = _resolve_offload_arch(meta["arch"])
    if err:
        return False, err
    kernel_lib = nm.get("kernel_lib") or dll

    with open(_kernel_src(meta), "rb") as f:
        candidate = f.read()
    violations = _abi_violations(meta, candidate)
    if violations:
        return False, ("candidate changes the frozen contract (only kernel bodies and "
                       "their device helpers are editable):\n  - " + "\n  - ".join(violations))
    dst = nm["kernel_dst"]
    snap_dir = os.path.dirname(_numeric_dll_snapshot(meta))
    backup = _kernel_backup_path(dst)
    with open(_ep_build_lock_path(nm["ep_repo"]), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(backup):
            with open(backup, "rb") as f:
                original = f.read()
        else:
            with open(dst, "rb") as f:
                original = f.read()
            with open(backup + ".tmp", "wb") as f:
                f.write(original)
            os.replace(backup + ".tmp", backup)
        try:
            # A fresh write gives dst a new mtime, forcing a recompile.
            with open(dst, "wb") as f:
                f.write(candidate)
            before = _mtime_ns(kernel_lib)
            rc = _run_build(build_cmd, cwd=nm["ep_repo"])
            if rc != 0:
                return False, f"EP build failed (build_cmd exited {rc})"
            after = _mtime_ns(kernel_lib)
            if after is None or after == before:
                return False, (f"build_cmd did not rebuild {kernel_lib}: it must be the "
                               f"library's link output (not a post-build copy, which is "
                               f"skipped when unchanged) and build_cmd must consume the "
                               f"overlaid {dst}")
            if not os.path.exists(dll):
                return False, f"EP library {dll} not found after build"
            violations = _export_violations(meta, kernel_lib)
            if violations:
                return False, ("rebuilt kernel library changes the frozen ABI:\n  - "
                               + "\n  - ".join(violations))
            shutil.rmtree(snap_dir, ignore_errors=True)
            os.makedirs(snap_dir)
            libs = {}
            for name, lib in _snapshot_sources(dll, kernel_lib).items():
                shutil.copy2(lib, os.path.join(snap_dir, name))
                libs[name] = _sha256(os.path.join(snap_dir, name))
            with open(os.path.join(snap_dir, _SNAPSHOT_MANIFEST), "w") as f:
                json.dump({"kernel_sha256": hashlib.sha256(candidate).hexdigest(),
                           "ep_base_sha": meta.get("ep_base_sha"), "libs": libs}, f, indent=2)
        finally:
            with open(dst, "wb") as f:
                f.write(original)
            os.remove(backup)
    return True, None


def _numeric_pytest(meta, extra):
    nm = meta["numeric"]
    snapshot = _numeric_dll_snapshot(meta)
    numeric_dir = os.path.join(nm["ep_repo"], "test", "numeric")
    # Suite options must be single `--opt=value` tokens: pytest pre-parses argv
    # before loading conftest.py, so a separate absolute-path value is taken as a
    # test path, the suite's conftest is never loaded, and --backend is rejected.
    backend = nm.get("backend", "ort_ep")
    cmd = [nm.get("python") or sys.executable, "-m", "pytest", "-q",
           "-k", meta["numeric_test"].replace(".py", ""),
           f"--backend={backend}"]
    if backend == "ort_ep":
        cmd += [f"--ep-name={nm.get('ep_name', 'AMDGPUExecutionProvider')}", f"--ep-dll={snapshot}"]
        cmd += [f"--ep-option={opt}" for opt in nm.get("ep_options", [])]
    cmd += extra
    r = _run(cmd, cwd=numeric_dir, env=_ep_env(meta), capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out, flush=True)
    return r.returncode, out


_DRIVER_LINE_RE = re.compile(r"^GEAK_EP_(?:PERF|CHECK) (\{.*\})\s*$", re.MULTILINE)


def _perf_driver(meta, check_only):
    """Run harness/ep_perf_driver.py over the task's perf cases on the snapshot.
    Return (ok, results in case order, err)."""
    nm = meta["numeric"]
    perf = nm["perf"]
    cmd = [nm.get("python") or sys.executable,
           os.path.join(TASK_DIR, "harness", "ep_perf_driver.py"),
           f"--ep-dll={_numeric_dll_snapshot(meta)}",
           f"--ep-name={nm.get('ep_name', 'AMDGPUExecutionProvider')}",
           *[f"--ep-option={opt}" for opt in nm.get("ep_options", [])],
           f"--op={perf.get('op', meta['op'])}",
           f"--cases={json.dumps(perf['cases'], separators=(',', ':'))}",
           f"--warmup={perf.get('warmup', 5)}", f"--iters={perf.get('iters', 20)}",
           f"--rotate-mib={perf.get('rotate_mib', 128)}"]
    if check_only:
        cmd.append("--check-only")
    r = _run(cmd, cwd=TASK_DIR, env=_ep_env(meta), capture_output=True, text=True)
    # The EP prints a [PERF SUMMARY] at library unload, after the driver exits main().
    stderr = "".join(line for line in r.stderr.splitlines(keepends=True)
                     if not line.startswith("[PERF SUMMARY]"))
    print(r.stdout + stderr, flush=True)
    by_id = {}
    for m in _DRIVER_LINE_RE.finditer(r.stdout):
        res = json.loads(m.group(1))
        by_id[res.get("id")] = res
    failed = [f"{i}: {res.get('error')}" for i, res in by_id.items() if res.get("status") != "ok"]
    if failed:
        return False, [], "perf-driver cases failed: " + "; ".join(failed)
    missing = [c["id"] for c in perf["cases"] if c["id"] not in by_id]
    if missing or r.returncode != 0:
        return False, [], f"perf driver exited {r.returncode} with no result for {missing}"
    return True, [by_id[c["id"]] for c in perf["cases"]], None


def numeric_correctness(meta):
    err = _check_snapshot(meta)
    if err:
        return False, err
    rc, _ = _numeric_pytest(meta, [])
    if rc != 0:
        return False, "numeric pytest reported failures (see output)"
    if meta["numeric"].get("perf"):
        ok, _, err = _perf_driver(meta, check_only=True)
        if not ok:
            return False, err
    return True, None


_DURATION_RE = re.compile(r"([0-9.]+)s\s+call\s+.*::(\S+)")
_STAT_KEYS = ("median_us", "min_us", "p10_us", "p90_us", "mean_us", "cv", "samples", "nodes")


def numeric_performance(meta):
    """Return (ok, cases, err). With a perf config, each case's time is the median
    per-op GPU time from the EP's trace; otherwise pytest --durations wall time
    per test (ORT session + graph execution), not kernel latency."""
    err = _check_snapshot(meta)
    if err:
        return False, [], err
    if meta["numeric"].get("perf"):
        ok, results, err = _perf_driver(meta, check_only=False)
        if not ok:
            return False, [], err
        return True, [{"test_case_id": res["id"],
                       "execution_time_ms": res["median_us"] / 1000.0,
                       "params": {**res["params"], "shape": res["shape"],
                                  **{k: res[k] for k in _STAT_KEYS if k in res}}}
                      for res in results], None
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
# export-patch: the candidate as a git patch against the EP commit the task was
# generated from. Uses a throwaway index and object store, so the shared EP
# checkout (and any compile holding its lock) is never touched.
# ----------------------------------------------------------------------------
def _git(top, args, env=None, **kw):
    return subprocess.run(["git", "-C", top, *args], env=env, capture_output=True,
                          text=True, check=False, **kw)


def numeric_export_patch(meta):
    """Return (ok, report, err) after writing build/hip_ep.patch."""
    nm = meta["numeric"]
    base = meta.get("ep_base_sha")
    if not base:
        return False, None, ("task has no ep_base_sha (generated from an EP tree that is "
                             "not a git checkout), so there is no commit to diff against")
    r = _git(nm["ep_repo"], ["rev-parse", "--show-toplevel"])
    if r.returncode != 0:
        return False, None, f"{nm['ep_repo']} is no longer a git checkout: {r.stderr.strip()}"
    top = r.stdout.strip()
    rel = os.path.relpath(os.path.realpath(nm["kernel_dst"]), os.path.realpath(top))
    objects = _git(top, ["rev-parse", "--path-format=absolute", "--git-common-dir"])
    if objects.returncode != 0:
        return False, None, f"cannot locate the EP object store: {objects.stderr.strip()}"
    ls = _git(top, ["ls-tree", base, "--", rel])
    if ls.returncode != 0 or not ls.stdout.strip():
        return False, None, f"{rel} not found in ep_base_sha {base}: {ls.stderr.strip()}"
    mode = ls.stdout.split()[0]
    patch_path = os.path.join(BUILD_DIR, "hip_ep.patch")
    with tempfile.TemporaryDirectory(prefix="geak_export_") as tmp:
        os.makedirs(os.path.join(tmp, "objects", "pack"))  # git rejects a missing object dir
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(tmp, "index"),
                   GIT_OBJECT_DIRECTORY=os.path.join(tmp, "objects"),
                   GIT_ALTERNATE_OBJECT_DIRECTORIES=os.path.join(objects.stdout.strip(), "objects"))
        steps = [["read-tree", base], ["hash-object", "-w", f"--path={rel}", "--", _kernel_src(meta)]]
        for step in steps:
            r = _git(top, step, env=env)
            if r.returncode != 0:
                return False, None, f"git {step[0]} failed: {r.stderr.strip()}"
        blob = r.stdout.strip()
        for step in (["update-index", "--cacheinfo", f"{mode},{blob},{rel}"],
                     ["diff", "--cached", "--binary", "--no-color", "--no-ext-diff",
                      "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/", base, "--", rel]):
            r = _git(top, step, env=env)
            if r.returncode != 0:
                return False, None, f"git {step[0]} failed: {r.stderr.strip()}"
        patch = r.stdout
        with open(patch_path, "w") as f:
            f.write(patch)
        applies = {}
        for ref in dict.fromkeys([base, nm.get("upstream_ref") or "origin/main"]):
            if _git(top, ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]).returncode:
                applies[ref] = "unresolved"
                continue
            if not patch:
                applies[ref] = "ok"
                continue
            check_env = dict(env, GIT_INDEX_FILE=os.path.join(tmp, "check_index"))
            r = _git(top, ["read-tree", ref], env=check_env)
            if r.returncode == 0:
                r = _git(top, ["apply", "--cached", "--check", patch_path], env=check_env)
            applies[ref] = "ok" if r.returncode == 0 else (r.stderr.strip() or "does not apply")
    report = {"patch": patch_path, "path": rel, "ep_base_sha": base, "empty": not patch,
              "kernel_sha256": _sha256(_kernel_src(meta)), "applies": applies}
    if applies[base] != "ok":
        return False, report, f"patch does not apply to its own base {base}: {applies[base]}"
    return True, report, None


# ----------------------------------------------------------------------------
DISPATCH = {
    "fast": (fast_compile, fast_correctness, fast_performance),
    "numeric": (numeric_compile, numeric_correctness, numeric_performance),
    "selfcheck": (selfcheck_compile, selfcheck_correctness, selfcheck_performance),
}


def _write_report(name, report):
    with open(os.path.join(BUILD_DIR, name), "w") as f:
        json.dump(report, f, indent=2)


def _exit_on_sigterm():
    # Agent tool timeouts send SIGTERM; exiting through SystemExit runs the
    # `finally` that puts the shared EP tree's kernel back.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))


def main():
    ap = argparse.ArgumentParser(description="GEAK EP-kernel task runner")
    ap.add_argument("mode", choices=["compile", "correctness", "performance", "export-patch"])
    args = ap.parse_args()
    _exit_on_sigterm()
    os.makedirs(BUILD_DIR, exist_ok=True)
    meta = _meta()
    compile_fn, correct_fn, perf_fn = DISPATCH[meta["oracle"]]

    if args.mode == "export-patch":
        if meta["oracle"] != "numeric":
            ok, report, err = False, None, "export-patch needs a numeric-oracle task (an EP checkout)"
        else:
            ok, report, err = numeric_export_patch(meta)
        _write_report("export_patch_report.json",
                      {"status": "ok" if ok else "fail", "error": err, **(report or {})})
        print(f"Export: {'PASS' if ok else 'FAIL'}")
        if report:
            print(f"Patch: {report['patch']}{' (empty: kernel_src matches the base)' if report['empty'] else ''}")
            for ref, res in report["applies"].items():
                print(f"Applies to {ref}: {res}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

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
