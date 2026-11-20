#!/usr/bin/env python3
"""Correctness on the boundary shapes a task's cases do not cover, for every variant at once.

A task's correctness cases are usually a few production shapes, each a multiple of the tile. A
kernel that drops a tail mask, wraps an index with `%`, or assumes a dimension divides the block
passes them and is wrong elsewhere -- and a hand-written Gluon kernel owns every one of those masks
itself. This derives the boundary shapes from each dimension's value and block size, runs every
variant and a reference on each, and separates three cases: a variant that fails where the baseline
passes (a regression), a shape the baseline itself fails (a shared limitation of the kernel, to be
fixed or declared), and clean.

Adapter contract (--module adapter.py):
  variants() -> list (or dict keys) of variant names
  run(name, shape) -> the output for `shape` (a {dim: int} dict): a tensor or an array
  reference(shape) -> the expected output, in higher precision where the operator allows
  supports(shape) -> bool, optional: False skips a shape the operator does not define
Inputs must be generated deterministically from `shape` inside the adapter, so every variant and
the reference see the same data.

Usage:
  edge_shapes.py --module adapter.py --dims M=2048:128,N=2624:128,K=6144:128 [--baseline NAME]
                 [--cos 0.999] [--tol 1e-2] [--max-shapes 40] [--json out.json]
  edge_shapes.py --selftest

--dims gives each dimension as NAME=value:block (block = the tile size along it; 1 for an untiled
dim). Per dimension the boundary set is {value, 1, block-1, block+1, value rounded to the block
+1 and -1}; shapes vary one dimension at a time from the given values, plus one shape with every
tiled dimension at block+1. A pass needs finite output, cosine >= --cos and
max|out-ref| / max|ref| <= --tol (use the task's own tolerance).
Exit: 0 clean | 1 a regression vs the baseline | 3 no regression, but the baseline fails some
shape (shared limitation) | 2 usage.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path


def parse_dims(spec):
    dims = {}
    for part in spec.split(","):
        name, _, rest = part.partition("=")
        value, _, block = rest.partition(":")
        if not name or not value.isdigit() or (block and not block.isdigit()):
            raise ValueError(f"--dims entry {part!r}: expected NAME=value[:block]")
        dims[name.strip()] = (int(value), int(block or 1))
    return dims


def boundary_values(value, block):
    if block <= 1:
        return [value]
    down = (value // block) * block
    cands = {value, 1, block - 1, block + 1, down + 1, down - 1, down + block - 1}
    return sorted(v for v in cands if v > 0)


def edge_shapes(dims, max_shapes=40):
    base = {k: v for k, (v, _) in dims.items()}
    shapes = [dict(base)]
    for k, (v, b) in dims.items():
        for x in boundary_values(v, b):
            s = dict(base, **{k: x})
            if s not in shapes:
                shapes.append(s)
    tails = {k: (b + 1 if b > 1 else v) for k, (v, b) in dims.items()}
    if tails not in shapes:
        shapes.append(tails)
    return shapes[:max_shapes]


def _arr(x):
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def compare(out, ref, cos_min, tol):
    import numpy as np
    o, r = _arr(out).ravel(), _arr(ref).ravel()
    if o.shape != r.shape:
        return {"ok": False, "why": f"shape {o.shape} vs reference {r.shape}"}
    if not np.isfinite(o).all():
        return {"ok": False, "why": f"{int((~np.isfinite(o)).sum())} non-finite elements"}
    no, nr = float(np.linalg.norm(o)), float(np.linalg.norm(r))
    cos = float(o @ r / (no * nr)) if no and nr else (1.0 if no == nr else 0.0)
    scale = float(np.abs(r).max()) if r.size else 0.0
    rel = float(np.abs(o - r).max() / scale) if scale else float(np.abs(o - r).max()) if o.size else 0.0
    return {"ok": cos >= cos_min and rel <= tol, "cos": round(cos, 7), "max_rel": rel}


def check(mod, dims, baseline=None, cos_min=0.999, tol=1e-2, max_shapes=40):
    names = list(mod.variants())
    baseline = baseline or names[0]
    if baseline not in names:
        raise ValueError(f"--baseline {baseline!r} is not one of {names}")
    rows = []
    for shape in edge_shapes(dims, max_shapes):
        if hasattr(mod, "supports") and not mod.supports(shape):
            continue
        row = {"shape": shape, "results": {}}
        try:
            ref = mod.reference(shape)
        except Exception as e:  # a reference that cannot run is a usage problem, recorded per shape
            row["reference_error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            continue
        for n in names:
            try:
                row["results"][n] = compare(mod.run(n, shape), ref, cos_min, tol)
            except Exception as e:
                row["results"][n] = {"ok": False, "why": f"{type(e).__name__}: {e}"}
        rows.append(row)
    regressions, shared = [], []
    for row in rows:
        res = row["results"]
        if not res:
            continue
        if not res[baseline]["ok"]:
            shared.append(row["shape"])
            continue
        for n, r in res.items():
            if n != baseline and not r["ok"]:
                regressions.append({"variant": n, "shape": row["shape"], **r})
    return {"baseline": baseline, "variants": names, "cos_min": cos_min, "tol": tol,
            "rows": rows, "regressions": regressions, "baseline_fails": shared}


def _fmt(r):
    return "ok" if r["ok"] else ("FAIL " + (r.get("why") or f"cos {r.get('cos')} rel {r.get('max_rel', 0):.2e}"))


def _load(path):
    spec = importlib.util.spec_from_file_location("_edge_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(mod)
    return mod


def _selftest():
    import types

    import numpy as np
    fail = 0

    def chk(cond, msg):
        nonlocal fail
        if not cond:
            print("FAIL:", msg)
            fail = 1

    chk(boundary_values(2048, 128) == [1, 127, 129, 2047, 2048, 2049, 2175], boundary_values(2048, 128))
    chk(boundary_values(37, 1) == [37], "an untiled dim keeps its value")
    shapes = edge_shapes(parse_dims("M=256:64,K=512:128"))
    chk(shapes[0] == {"M": 256, "K": 512} and {"M": 65, "K": 129} in shapes, "base first, all-tails shape present")

    def make(bug_in):
        m = types.SimpleNamespace()
        blk = 128

        def data(s):
            rng = np.random.default_rng(s["M"] * 7919 + s["K"])
            return rng.standard_normal((s["M"], s["K"]))
        m.variants = lambda: ["base", "cand"]
        m.reference = lambda s: data(s).sum(axis=1)

        def run(n, s):
            x = data(s)
            if n in bug_in and s["K"] % blk:        # drops the K tail, as a missing mask would
                x = x[:, : (s["K"] // blk) * blk]
            return x.sum(axis=1)
        m.run = run
        return m
    dims = parse_dims("M=256:64,K=512:128")
    r = check(make({"cand"}), dims)
    chk(r["regressions"] and not r["baseline_fails"], "a tail bug only in the candidate is a regression")
    r = check(make({"base", "cand"}), dims)
    chk(not r["regressions"] and r["baseline_fails"], "a tail bug in both is a shared limitation, not a regression")
    r = check(make(set()), dims)
    chk(not r["regressions"] and not r["baseline_fails"], "clean")
    print("edge_shapes selftest:", "FAIL" if fail else "OK")
    return fail


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--selftest"]:
        return _selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", required=True)
    ap.add_argument("--dims", required=True, help="NAME=value:block,... (block 1 = untiled)")
    ap.add_argument("--baseline", help="variant whose failures are the kernel's own (default: the first)")
    ap.add_argument("--cos", type=float, default=0.999)
    ap.add_argument("--tol", type=float, default=1e-2, help="max_rel bound; pass the task's own tolerance")
    ap.add_argument("--max-shapes", type=int, default=40)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        dims = parse_dims(a.dims)
    except ValueError as e:
        ap.error(str(e))
    r = check(_load(a.module), dims, a.baseline, a.cos, a.tol, a.max_shapes)
    for row in r["rows"]:
        shape = ",".join(f"{k}={v}" for k, v in row["shape"].items())
        if "reference_error" in row:
            print(f"  {shape:32s} reference error: {row['reference_error']}")
            continue
        print(f"  {shape:32s} " + " | ".join(f"{n}: {_fmt(x)}" for n, x in row["results"].items()))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1, default=str) + "\n")
    if r["regressions"]:
        print(f"REGRESSION: {len(r['regressions'])} (variant, shape) fail where {r['baseline']} passes")
        return 1
    if r["baseline_fails"]:
        print(f"SHARED LIMITATION: {r['baseline']} itself fails {len(r['baseline_fails'])} shape(s); "
              "fix the kernel's boundary handling or declare the supported shapes")
        return 3
    print("clean: every variant passes every boundary shape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
