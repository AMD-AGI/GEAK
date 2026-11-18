#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for a GEAK task built from a hip-ep kernel unit-test leaf.

Copied into each generated task as scripts/task_runner.py by
tools/hip_ep_e2e/make_task.py. Per-task data lives in ../task_meta.json. The
task mirrors the hip-ep tree under src/, so the edited files map one to one onto
an overlay for tools/hip_ep_e2e/build_kernels.sh.

Modes (GEAK's COMMANDMENT calls these):
  compile      build the leaf's test executable against the editable kernel
               sources, with the compile flags the shipped kernel library uses
  correctness  every task shape and decode guard shape must PASS against the
               leaf's CPU reference, then the leaf's own full run (coverage
               sweep + every extended WMMA entry) must pass, then the
               qmoe_ragged_mm leaf, whose kernel instantiates the same WMMA core
  performance  per task shape, the median over `perf_reps` runs (default 1) of
               the leaf's cold-weight timing; prints `Perf: <ms> ms (<case_id>)`.
               One run of the largest shape can take about a minute, most of it the
               leaf's CPU reference; repeats spread under 1% on the shapes that
               carry the weight.

Timing runs with HIPDNN_MATMUL_AUTOTUNE_MODE=online: the kernel's own autotune
picks the fastest config of the edited code for each shape, as hip-ep does in
online mode, so a config choice alone never counts as a kernel change.

correctness and performance first rebuild the task when its kernel sources changed
since the last compile, so they never run an older build.

GPU runs hold the serving-GPU lock (the lock e2e runs use), with the locking code of
hipep_env.py, which make_task.py copies next to this runner.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hipep_env  # noqa: E402

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(TASK_DIR, "build")
EXE = os.path.join(BUILD_DIR, "test_direct")
QMOE_EXE = os.path.join(BUILD_DIR, "test_qmoe")

# "  M=1024 N=34816 K=5120 gs=32 zero=0 dtype=fp16  status=0  9.7012 ms  relL2=1.234e-03  PASS"
LEAF_RE = re.compile(r"M=(\d+) N=(\d+) K=(\d+) gs=(\d+) zero=(\d) dtype=(\w+)\s+status=(-?\d+)\s+"
                     r"([0-9.]+) ms\s+relL2=([0-9.eE+-]+)\s+(PASS|\*\*\* FAIL \*\*\*)")


def parse_leaf_line(text):
    """The case line of a single-shape leaf run, as a dict; None if absent."""
    for line in text.splitlines():
        m = LEAF_RE.search(line)
        if m:
            M, N, K, gs, zero, dtype, status, ms, rel, verdict = m.groups()
            return {"M": int(M), "N": int(N), "K": int(K), "gs": int(gs), "zero": zero == "1",
                    "dtype": dtype, "status": int(status), "ms": float(ms),
                    "relL2": float(rel), "pass": verdict == "PASS"}
    return None


def case_id(s):
    return f"M{s['M']}_K{s['K']}_N{s['N']}_gs{s['gs']}{'' if s.get('zp') else '_noz'}"


def _meta():
    with open(os.path.join(TASK_DIR, "task_meta.json")) as f:
        return json.load(f)


def _gpu_lock(meta):
    return hipep_env.hold_lock(meta["lock_path"], hipep_env.lock_wait())


def _run(cmd, env=None, timeout=1800):
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout,
                          cwd=TASK_DIR, check=False)


def _p(rel):
    return os.path.join(TASK_DIR, rel)


def _digest(flags, paths, dirs):
    """Hash of flags, files, and every header under dirs, by task-relative name, so it does
    not depend on where the task sits."""
    h = hashlib.sha256(json.dumps(flags).encode())
    paths = list(paths)
    for d in dirs:
        for root, _, files in os.walk(_p(d)):
            paths += [os.path.join(root, f) for f in files if f.endswith((".h", ".hpp", ".inc"))]
    for path in sorted({_p(x) for x in paths}):
        with open(path, "rb") as f:
            h.update(os.path.relpath(path, TASK_DIR).encode() + b"\0" + f.read())
    return h.hexdigest()


def _frozen_digest(meta):
    """Hash of everything the frozen objects (test.o, autotune.o) are built from."""
    return _digest([meta["test_flags"], meta["autotune_flags"]],
                   [meta["test_cpp"], meta["autotune_cpp"], meta["lut_fb"], meta["compat_header"]],
                   meta["include_dirs"] + [meta["autotune_dir"], meta["generated_dir"]])


def _qmoe_sources(meta):
    """(source, flags) of the QMoE leaf's own objects: its test, its kernels, the no-table stub."""
    q = meta["qmoe"]
    return [(q["test_cpp"], meta["test_flags"]), *((x["file"], x["flags"]) for x in q["sources"]),
            (q["lut_stub"], meta["test_flags"])]


def _qmoe_digest(meta):
    """Hash of everything the QMoE leaf's own objects are built from, the editable WMMA cores
    included."""
    srcs = _qmoe_sources(meta)
    return _digest([f for _, f in srcs], [s for s, _ in srcs], meta["include_dirs"])


def _kernel_digest(meta):
    """Hash of what kernel.o is built from that can change: the editable sources and the flags."""
    return _digest([meta["kernel_flags"]], meta["editable"], [])


def _includes(meta):
    return [f"-I{_p(d)}" for d in meta["include_dirs"]] + \
        ["-isystem", os.path.join(meta["therock"], "include"),
         f"-I{os.path.join(BUILD_DIR, 'gen')}", f"-I{_p(meta['generated_dir'])}",
         f"-I{_p(meta['flatbuffers_include'])}", f"-I{_p(meta['autotune_dir'])}"]


def _stamp_is(stamp, digest):
    if not os.path.exists(stamp):
        return False
    with open(stamp) as f:
        return f.read() == digest


def _build(steps):
    for cmd in steps:
        r = _run(cmd)
        if r.returncode != 0:
            sys.stdout.write(r.stdout[-4000:] + r.stderr[-8000:])
            return f"build step failed: {' '.join(cmd[:3])} ... {cmd[-3]}"
    return None


def compile_task(meta):
    os.makedirs(os.path.join(BUILD_DIR, "gen"), exist_ok=True)
    with open(os.path.join(BUILD_DIR, "gen", "lut_fb_path.h"), "w") as f:
        f.write(f'#define HIPDNN_LUT_FB "{_p(meta["lut_fb"])}"\n')
    hipcc = meta["hipcc"]
    inc = _includes(meta)
    objs = {"kernel": os.path.join(BUILD_DIR, "kernel.o"),
            "autotune": os.path.join(BUILD_DIR, "autotune.o"),
            "test": os.path.join(BUILD_DIR, "test.o")}
    # The test and autotune objects come from frozen files (headers included, since an editable
    # header could reach them), so they are rebuilt only when those change; the kernel always is.
    stamp = os.path.join(BUILD_DIR, "frozen.sha256")
    digest = _frozen_digest(meta)
    reuse = (_stamp_is(stamp, digest)
             and os.path.exists(objs["test"]) and os.path.exists(objs["autotune"]))
    steps = [
        [hipcc, "-c", *meta["kernel_flags"], "-x", "hip", *inc,
         _p(meta["kernel_file"]), "-o", objs["kernel"]],
    ]
    if not reuse:
        steps += [
            [hipcc, "-c", *meta["autotune_flags"], "-x", "hip", *inc,
             _p(meta["autotune_cpp"]), "-o", objs["autotune"]],
            [hipcc, "-c", *meta["test_flags"], "-DHIPDNN_LUT_LINKED_EXTERNALLY=1",
             "-include", _p(meta["compat_header"]), *inc, "-x", "hip",
             _p(meta["test_cpp"]), "-o", objs["test"]],
        ]
    steps.append([hipcc, "--hip-link", meta["offload_arch"], objs["test"], objs["kernel"],
                  objs["autotune"], "-o", EXE])
    if os.path.exists(stamp):
        os.remove(stamp)
    err = _build(steps)
    if err:
        return False, err
    with open(stamp, "w") as f:
        f.write(digest)
    with open(os.path.join(BUILD_DIR, "kernel.sha256"), "w") as f:
        f.write(_kernel_digest(meta))
    return True, None


def _current(meta):
    """Compile first when there is no build or the kernel sources changed since the last one."""
    if os.path.exists(EXE) and _stamp_is(os.path.join(BUILD_DIR, "kernel.sha256"), _kernel_digest(meta)):
        return None
    ok, err = compile_task(meta)
    return None if ok else err


def build_qmoe(meta):
    """The QMoE leaf, linked with this task's kernel and autotune objects: its test also calls
    the matmul_nbits small-M grouped entry. Its own objects are rebuilt only when their sources
    or the headers changed; the link is redone every time, for the current kernel object."""
    kernel_o, autotune_o = (os.path.join(BUILD_DIR, n) for n in ("kernel.o", "autotune.o"))
    hipcc, inc, srcs = meta["hipcc"], _includes(meta), _qmoe_sources(meta)
    objs = [os.path.join(BUILD_DIR, f"qmoe_{i}.o") for i in range(len(srcs))]
    stamp = os.path.join(BUILD_DIR, "qmoe.sha256")
    digest = _qmoe_digest(meta)
    steps = []
    if not (_stamp_is(stamp, digest) and all(os.path.exists(o) for o in objs)):
        steps = [[hipcc, "-c", *flags, "-x", "hip", *inc, _p(src), "-o", o]
                 for (src, flags), o in zip(srcs, objs)]
        if os.path.exists(stamp):
            os.remove(stamp)
    steps.append([hipcc, "--hip-link", meta["offload_arch"], *objs, kernel_o, autotune_o,
                  "-o", QMOE_EXE])
    err = _build(steps)
    if err:
        return f"QMoE leaf: {err}"
    with open(stamp, "w") as f:
        f.write(digest)
    return None


def qmoe_passed(returncode, output):
    """The QMoE leaf prints a PASS line per case and ends with ALL PASS."""
    return returncode == 0 and "ALL PASS" in output and "FAIL" not in output


def _leaf_env(meta, mode, csv=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("HIPDNN_")}
    env["HIPDNN_MATMUL_AUTOTUNE_MODE"] = mode
    env["LD_LIBRARY_PATH"] = os.path.join(meta["therock"], "lib")
    if csv:
        env.update(HIPDNN_RESULTS_CSV=csv, HIPDNN_RESULTS_OP="matmul_nbits",
                   HIPDNN_RESULTS_LEAF=meta["leaf"], HIPDNN_RESULTS_ARCH=meta["arch"],
                   HIPDNN_RESULTS_MODE=mode)
    return env


def _run_shape(meta, s, mode, csv=None):
    cmd = [EXE, f"{s['M']}x{s['K']}x{s['N']}", "--group-size", str(s["gs"])]
    if not s.get("zp"):
        cmd.append("--no-zeros")
    r = _run(cmd, env=_leaf_env(meta, mode, csv))
    out = r.stdout + r.stderr
    case = parse_leaf_line(out)
    if r.returncode != 0 or case is None or not case["pass"] or case["status"] != 0:
        return None, out
    return case, out


def correctness(meta):
    err = _current(meta) or build_qmoe(meta)
    if err:
        return False, err
    coverage = str(meta["coverage"])
    with _gpu_lock(meta):
        for s in meta["shapes"] + meta["guard_shapes"]:
            case, out = _run_shape(meta, s, "online")
            print(out.strip().splitlines()[-1] if out.strip() else "", flush=True)
            if case is None:
                return False, f"{case_id(s)} failed (online autotune)"
        r = _run([EXE, "--coverage", coverage], env=_leaf_env(meta, "lookup"), timeout=3600)
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-6:])
        print(tail, flush=True)
        if r.returncode != 0 or "SOME FAILED" in r.stdout or "FAIL" in tail:
            return False, "the leaf's full run failed (coverage sweep or forced extended entries)"
        r = _run([QMOE_EXE, "--coverage", coverage], env=_leaf_env(meta, "lookup"), timeout=3600)
    out = r.stdout + r.stderr
    print("\n".join(out.strip().splitlines()[-3:]), flush=True)
    if not qmoe_passed(r.returncode, out):
        return False, "the qmoe_ragged_mm leaf failed: it instantiates the edited WMMA core"
    return True, None


def performance(meta):
    err = _current(meta)
    if err:
        return False, [], err
    reps = int(meta["perf_reps"])
    cases = []
    with _gpu_lock(meta):
        for s in meta["shapes"]:
            times = []
            for i in range(reps):
                csv = os.path.join(BUILD_DIR, f"perf_{case_id(s)}_{i}.csv")
                case, out = _run_shape(meta, s, "online", csv)
                if case is None:
                    print(out[-3000:])
                    return False, cases, f"{case_id(s)} failed; its timing is not valid"
                times.append(case["ms"])
            config = ""
            try:
                with open(csv) as f:
                    rows = f.read().strip().splitlines()
                config = rows[-1].split(",")[5] if len(rows) > 1 else ""
            except OSError:
                pass
            cases.append({"test_case_id": case_id(s), "execution_time_ms": statistics.median(times),
                          "params": {**s, "times_ms": times, "config": config}})
    return True, cases, None


def _write_report(name, report):
    with open(os.path.join(BUILD_DIR, name), "w") as f:
        json.dump(report, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="GEAK hip-ep leaf task runner")
    ap.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = ap.parse_args()
    os.makedirs(BUILD_DIR, exist_ok=True)
    meta = _meta()

    if args.mode in ("compile", "correctness"):
        ok, err = (compile_task if args.mode == "compile" else correctness)(meta)
        _write_report(f"{args.mode}_report.json", {"status": "ok" if ok else "fail", "error": err})
        print(f"{'Compilation' if args.mode == 'compile' else 'Correctness'}: "
              f"{'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    ok, cases, err = performance(meta)
    _write_report("performance_report.json",
                  {"status": "ok" if ok else "fail", "error": err, "test_cases": cases})
    if not ok:
        print("Performance: FAIL")
        print(f"Error: {err}")
        sys.exit(1)
    for c in cases:
        print(f"Perf: {c['execution_time_ms']:.4f} ms ({c['test_case_id']})")
    sys.exit(0)


if __name__ == "__main__":
    main()
