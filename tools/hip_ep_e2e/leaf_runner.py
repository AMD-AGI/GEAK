#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for a GEAK task built from hip-ep's kernel unit-test leaves (make_task.py).

make_task.py copies it into the task as scripts/task_runner.py, next to tools/ep_task_runner.py,
whose entry point and reports it uses (oracle "leaf"). Per-task data is ../task_meta.json;
the task's README says what each mode checks and times. Every mode first rebuilds the
objects whose sources, flags or headers changed, so no mode runs an older build. GPU runs
hold the serving-GPU lock, with the locking code of hipep_env.py, copied next to this runner.
"""
import hashlib
import json
import os
import re
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
            return {"status": int(m.group(7)), "ms": float(m.group(8)), "pass": m.group(10) == "PASS"}
    return None


def case_id(s):
    return f"M{s['M']}_K{s['K']}_N{s['N']}_gs{s['gs']}{'' if s.get('zp') else '_noz'}"


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
    h = hashlib.sha256(json.dumps(flags).replace(TASK_DIR, "<task>").encode())
    paths = list(paths)
    for d in dirs:
        for root, _, files in os.walk(_p(d)):
            paths += [os.path.join(root, f) for f in files if f.endswith((".h", ".hpp", ".inc"))]
    for path in sorted({_p(x) for x in paths}):
        with open(path, "rb") as f:
            h.update(os.path.relpath(path, TASK_DIR).encode() + b"\0" + f.read())
    return h.hexdigest()


def _includes(meta):
    return [f"-I{_p(d)}" for d in meta["include_dirs"]] + \
        ["-isystem", os.path.join(meta["therock"], "include"),
         f"-I{os.path.join(BUILD_DIR, 'gen')}", f"-I{_p(meta['generated_dir'])}",
         f"-I{_p(meta['flatbuffers_include'])}", f"-I{_p(meta['autotune_dir'])}"]


def _obj(meta, src, flags):
    """build/<src>.o, compiled only when its flags, its source, the lookup table or any header
    the task carries changed since it was last built. Returns (path, rebuilt, error)."""
    obj = os.path.join(BUILD_DIR, os.path.basename(src) + ".o")
    stamp = obj + ".sha256"
    digest = _digest(flags, [src, meta["lut_fb"], meta["compat_header"]],
                     meta["include_dirs"] + [meta["autotune_dir"], meta["generated_dir"]])
    if os.path.exists(obj) and os.path.exists(stamp) and open(stamp).read() == digest:
        return obj, False, None
    if os.path.exists(stamp):
        os.remove(stamp)
    r = _run([meta["hipcc"], "-c", *flags, "-x", "hip", *_includes(meta), _p(src), "-o", obj])
    if r.returncode != 0:
        sys.stdout.write(r.stdout[-4000:] + r.stderr[-8000:])
        return obj, True, f"build step failed: {os.path.basename(src)}"
    with open(stamp, "w") as f:
        f.write(digest)
    return obj, True, None


def compile_task(meta):
    """The leaf's executable and the QMoE leaf's, which links the task's kernel and autotune
    objects too: its test also calls the matmul_nbits small-M grouped entry."""
    os.makedirs(os.path.join(BUILD_DIR, "gen"), exist_ok=True)
    with open(os.path.join(BUILD_DIR, "gen", "lut_fb_path.h"), "w") as f:
        f.write(f'#define HIPDNN_LUT_FB "{_p(meta["lut_fb"])}"\n')
    q, test_flags = meta["qmoe"], meta["test_flags"]
    leaf_test = [*test_flags, "-DHIPDNN_LUT_LINKED_EXTERNALLY=1", "-include", _p(meta["compat_header"])]
    objs, rebuilt = [], False
    for src, flags in [(meta["editable"][0], meta["kernel_flags"]), (meta["autotune_cpp"], meta["autotune_flags"]),
                       (meta["test_cpp"], leaf_test), (q["test_cpp"], test_flags),
                       *((x["file"], x["flags"]) for x in q["sources"]), (q["lut_stub"], test_flags)]:
        obj, new, err = _obj(meta, src, flags)
        if err:
            return False, err
        objs.append(obj)
        rebuilt |= new
    kernel, autotune, test, *qmoe = objs
    offload = next(f for f in meta["kernel_flags"] if f.startswith("--offload-arch="))
    for exe, parts in ((EXE, [test, kernel, autotune]), (QMOE_EXE, [*qmoe, kernel, autotune])):
        if rebuilt or not os.path.exists(exe):
            if os.path.exists(exe):
                os.remove(exe)  # a failed link must not leave the previous build to be run
            r = _run([meta["hipcc"], "--hip-link", offload, *parts, "-o", exe])
            if r.returncode != 0:
                sys.stdout.write(r.stdout[-4000:] + r.stderr[-8000:])
                return False, f"link failed: {os.path.basename(exe)}"
    return True, None


def qmoe_passed(returncode, output):
    """The QMoE leaf prints a PASS line per case and ends with ALL PASS."""
    return returncode == 0 and "ALL PASS" in output and "FAIL" not in output


def _leaf_env(meta, mode):
    env = {k: v for k, v in os.environ.items() if not k.startswith("HIPDNN_")}
    env["HIPDNN_MATMUL_AUTOTUNE_MODE"] = mode
    env["LD_LIBRARY_PATH"] = os.path.join(meta["therock"], "lib")
    return env


def _run_shape(meta, s, mode):
    cmd = [EXE, f"{s['M']}x{s['K']}x{s['N']}", "--group-size", str(s["gs"])]
    if not s.get("zp"):
        cmd.append("--no-zeros")
    r = _run(cmd, env=_leaf_env(meta, mode))
    out = r.stdout + r.stderr
    case = parse_leaf_line(out)
    if r.returncode != 0 or case is None or not case["pass"] or case["status"] != 0:
        return None, out
    return case, out


def correctness(meta):
    ok, err = compile_task(meta)
    if not ok:
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
    """Each task shape once, timed by the leaf on cold weights: `Perf: <ms> ms (<case_id>)`."""
    ok, err = compile_task(meta)
    if not ok:
        return False, [], err
    cases = []
    with _gpu_lock(meta):
        for s in meta["shapes"]:
            case, out = _run_shape(meta, s, "online")
            if case is None:
                print(out[-3000:])
                return False, cases, f"{case_id(s)} failed; its timing is not valid"
            cases.append({"test_case_id": case_id(s), "execution_time_ms": case["ms"], "params": s})
    return True, cases, None


def main():
    import ep_task_runner  # copied next to this runner by make_task.py
    ep_task_runner.DISPATCH["leaf"] = (compile_task, correctness, performance)
    ep_task_runner.main()


if __name__ == "__main__":
    main()
