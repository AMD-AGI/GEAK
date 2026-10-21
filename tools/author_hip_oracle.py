#!/usr/bin/env python3
# GENERIC author-HIP oracle. Copied verbatim into each generated task as
# `unittest.py`. It is fully parameterized by `meta.json` (the arg schema), so
# adding a new op never edits this file.
#
# It compiles the authored kernel_src/<op>_kernel.hip into a shared lib with
# hipcc, binds the frozen `extern "C" <launch_symbol>` via ctypes, synthesizes
# inputs from the arg schema, runs it against a torch reference, and grades
# correctness + speedup with the shared harness_lib. Because the candidate is
# reached purely through the frozen C-ABI symbol, a PASS here means the authored
# .hip drops straight into the ONNX-Runtime HIP EP.
#
# Env toggles (validation only; not for a scored run):
#   GEAK_SELFTEST=1   bind the candidate to the torch reference (no kernel) ->
#                     proves oracle wiring, speedup ~= 1.0.
#   GEAK_FORCE_CPU=1  run on CPU (torch only; implies selftest — HIP needs a GPU).
#   GEAK_HIP_ARCH     offload arch for hipcc (default: meta.arch or gfx1151).
#   GEAK_STRICT_INTEGRITY=1  hard-fail on an integrity-hash mismatch.
import ctypes
import glob
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# This file is named unittest.py (oracle contract). Keep the script dir OFF
# sys.path so it does not shadow the stdlib 'unittest' that torch imports.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]

import importlib.util  # noqa: E402


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load_by_path("harness_lib", os.path.join(HERE, "harness_lib.py"))  # noqa: E402

META = json.load(open(os.path.join(HERE, "meta.json")))

ARGS = META["args"]
IN_ARGS = [a for a in ARGS if a["kind"] == "in_tensor"]
OUT_ARGS = [a for a in ARGS if a["kind"] == "out_tensor"]
LAUNCH = META["launch_symbol"]


# ---------------- integrity (same tree-sha scheme as freeze.py) ----------------
def _tree_sha(root):
    items = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("__pycache__", "build")]
        for f in fn:
            if f.endswith((".pyc", ".so", ".o")):
                continue
            p = os.path.join(dp, f)
            items.append((os.path.relpath(p, root), open(p, "rb").read()))
    hsh = hashlib.sha256()
    for rel, content in sorted(items, key=lambda x: x[0]):
        hsh.update(rel.encode())
        hsh.update(b"\0")
        hsh.update(content)
    return hsh.hexdigest()


def _file_sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _check_integrity():
    strict = os.environ.get("GEAK_STRICT_INTEGRITY") == "1"
    checks = {
        "baseline_src_sha256": _tree_sha(os.path.join(HERE, "baseline_src")),
        "harness_lib_sha256": _file_sha(os.path.join(HERE, "harness_lib.py")),
        "unittest_sha256": _file_sha(os.path.join(HERE, "unittest.py")),
    }
    ok = True
    for k, got in checks.items():
        want = META.get(k, "")
        if want and want != got:
            ok = False
            print(f"GEAK_INTEGRITY_MISMATCH: {k} want={want[:12]} got={got[:12]}")
    print(f"GEAK_INTEGRITY: {'ok' if ok else 'mismatch'} (strict={strict})")
    if strict and not ok:
        sys.exit(3)


def _resolve(spec, subdir):
    mod_name, attr = spec.split(":")
    sys.path.insert(0, os.path.join(HERE, subdir))
    mod = importlib.import_module(mod_name)
    importlib.reload(mod)
    return getattr(mod, attr)


# ---------------- hipcc build + ctypes bind of the authored kernel ----------------
def _torch_dtype(name):
    import torch

    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


def _ctype(kind):
    if kind in ("in_tensor", "out_tensor"):
        return ctypes.c_void_p
    if kind in ("dim", "scalar_i32"):
        return ctypes.c_int
    if kind == "scalar_f32":
        return ctypes.c_float
    raise ValueError(f"unknown arg kind: {kind}")


def _build_and_bind():
    arch = os.environ.get("GEAK_HIP_ARCH") or META.get("arch") or "gfx1151"
    hips = sorted(glob.glob(os.path.join(HERE, "kernel_src", "*.hip")))
    if not hips:
        raise FileNotFoundError("no kernel_src/*.hip to compile")
    hip = hips[0]
    so = os.path.join(HERE, f"_{META['op']}.so")
    if (not os.path.exists(so)) or os.path.getmtime(so) < os.path.getmtime(hip):
        hipcc = "/opt/rocm/bin/hipcc" if os.path.exists("/opt/rocm/bin/hipcc") else "hipcc"
        cmd = [hipcc, "-O3", "-fPIC", "-shared", f"--offload-arch={arch}",
               "-I", os.path.join(HERE, "include"), hip, "-o", so]
        print("BUILD:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(so)
    fn = getattr(lib, LAUNCH)
    fn.argtypes = [_ctype(a["kind"]) for a in ARGS]
    fn.restype = None
    return fn


# symbolic-dim sources: dim name -> (in_tensor index, axis), from the first
# in_tensor whose shape references that dim.
_SYM_SOURCE = {}
for _i, _a in enumerate(IN_ARGS):
    for _axis, _dim in enumerate(_a["shape"]):
        _SYM_SOURCE.setdefault(_dim, (_i, _axis))


def _make_current(fn, device):
    import torch

    def current_call(inputs):
        dims = {name: int(inputs[i].shape[ax]) for name, (i, ax) in _SYM_SOURCE.items()}
        outs = []
        for oa in OUT_ARGS:
            shp = [dims[d] for d in oa["shape"]]
            outs.append(torch.empty(shp, dtype=_torch_dtype(oa["dtype"]), device=device))
        call, ii, oi = [], 0, 0
        for a in ARGS:
            k = a["kind"]
            if k == "in_tensor":
                call.append(inputs[ii].data_ptr()); ii += 1
            elif k == "out_tensor":
                call.append(outs[oi].data_ptr()); oi += 1
            elif k == "dim":
                call.append(dims[a["name"]])
            elif k == "scalar_i32":
                call.append(int(a["value"]))
            elif k == "scalar_f32":
                call.append(float(a["value"]))
        torch.cuda.synchronize()
        fn(*call)
        torch.cuda.synchronize()
        return outs[0] if len(outs) == 1 else tuple(outs)

    return current_call


def _synth(arg, shape, rng, device):
    import torch

    dt = _torch_dtype(arg["dtype"])
    init = arg.get("init", "randn")
    if init == "ones_plus_noise":
        return (1.0 + 0.02 * torch.randn(shape, generator=rng, device=device,
                                         dtype=torch.float32)).to(dt)
    if init == "randn_small":
        return (0.1 * torch.randn(shape, generator=rng, device=device,
                                  dtype=torch.float32)).to(dt)
    if init == "uniform":
        return (2.0 * torch.rand(shape, generator=rng, device=device,
                                 dtype=torch.float32) - 1.0).to(dt)
    return torch.randn(shape, generator=rng, device=device, dtype=dt)


def _make_inputs_for(case, device):
    def make_inputs(rng):
        return tuple(_synth(a, [int(case[d]) for d in a["shape"]], rng, device)
                     for a in IN_ARGS)

    return make_inputs


def main():
    import torch

    _check_integrity()

    selftest = os.environ.get("GEAK_SELFTEST") == "1"
    force_cpu = os.environ.get("GEAK_FORCE_CPU") == "1"
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if force_cpu and not selftest:
        print("GEAK_FORCE_CPU implies GEAK_SELFTEST (HIP needs a GPU); enabling selftest.")
        selftest = True

    baseline_call = _resolve(META["baseline_callable"], "baseline_src")
    if selftest:
        current_call = baseline_call
    else:
        current_call = _make_current(_build_and_bind(), device)

    tol = float(META.get("tol", 1e-3))
    draws = int(META.get("random_draws", 5))
    cases = META["cases"]
    shapes = [{"sig": c["sig"], "make_inputs": _make_inputs_for(c, device)} for c in cases]

    all_ok, per_case = h.check_random_vs_baseline(
        baseline_call, current_call, shapes, tol, draws=draws, seed=0)
    for r in per_case:
        print(f"PARITY {r['case']}: correct={r['correct']} "
              f"max_rel_err={r.get('max_rel_err')} speedup={r.get('speedup')}")

    # output independence on the PRIMARY output (candidate may return a tuple)
    c0 = cases[0]
    mk = _make_inputs_for(c0, device)

    def _primary(a):
        out = current_call(a)
        return out[0] if isinstance(out, tuple) else out

    ind_ok, ind_reason = h.assert_independent_outputs(
        _primary,
        mk(torch.Generator(device=device).manual_seed(1)),
        mk(torch.Generator(device=device).manual_seed(2)))
    print(f"INDEPENDENCE: ok={ind_ok} {ind_reason}")
    all_ok = all_ok and ind_ok

    receipt = {"all_primed": True, "timer_unprimed": False, "cases": {}}
    ratios, base_ms = [], []
    for c in cases:
        mk = _make_inputs_for(c, device)
        args = mk(torch.Generator(device=device).manual_seed(int(c.get("seed", 0))))
        db = h.time_op(lambda: baseline_call(args), detail=True)
        dc = h.time_op(lambda: current_call(args), detail=True)
        if not db or not dc:
            print(f"TIMING {c['sig']}: FAILED (call raised)")
            all_ok = False
            continue
        sp = db["ms"] / dc["ms"] if dc["ms"] else None
        pb, pc = db.get("primed"), dc.get("primed")
        receipt["cases"][c["sig"]] = {"baseline": {"primed": pb, "host_ms": db.get("host_ms")},
                                      "current": {"primed": pc, "host_ms": dc.get("host_ms")}}
        if pb is None or pc is None:
            receipt["timer_unprimed"] = True
        if pb is False or pc is False:
            receipt["all_primed"] = False
        if sp:
            ratios.append(sp)
        base_ms.append(db["ms"])
        print(f"TIMING {c['sig']}: baseline={db['ms']:.4f}ms current={dc['ms']:.4f}ms speedup={sp}")

    def _geo(xs):
        xs = [x for x in xs if x and x > 0]
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

    print(f"GEAK_TIMING_RECEIPT: {json.dumps(receipt)}")
    print(f"BASELINE_GEOMEAN_MS: {_geo(base_ms):.4f}")
    print(f"SPEEDUP_GEOMEAN: {_geo(ratios):.4f}")
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
