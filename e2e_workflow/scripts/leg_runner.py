#!/usr/bin/env python3
"""Run ONE measurement leg of a kernel task. WHICH leg is decided ONLY by the overlay on PYTHONPATH.

  baseline leg   PYTHONPATH=<task>/baseline_overlay    -> the REAL online serving stack (install +
                 every already-accepted kernel). Nothing in the task dir can change what it imports.
  candidate leg  PYTHONPATH=<task>/_cand_overlay       -> that SAME stack + exactly ONE entry built
                 from kernel_src/ (meta.candidate_bind).

Both legs execute THIS file and the task's OWN cases.py, so `speedup = baseline_ms / candidate_ms` is
always measured against the live path, whatever LANGUAGE the candidate is written in.

Modes: list (bucket sigs) | resolve (leg identity, for the direction assert) | time | oracle.
Driven by harness_lib.measure_legs — a unittest.py should not invoke it directly.
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# this file lives IN the task dir next to a file named unittest.py — drop it from sys.path before
# torch is imported, or that file shadows stdlib unittest.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve(dotted_attr):
    mod_name, _, attr = dotted_attr.partition(":")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


def _identity(target_callable):
    """Where this leg's callable ACTUALLY came from. The two legs must not report the same tuple."""
    try:
        fn = _resolve(target_callable)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    mod = getattr(fn, "__module__", "") or ""
    f = ""
    try:
        f = getattr(sys.modules.get(mod), "__file__", "") or ""
    except Exception:
        pass
    if not f:
        try:
            f = getattr(importlib.import_module(target_callable.split(":")[0]), "__file__", "") or ""
        except Exception:
            pass
    return {"module": mod, "file": os.path.realpath(f) if f else "",
            "qualname": getattr(fn, "__qualname__", repr(fn))}


def _snapshot(out):
    """Detach one op's output for the oracle blob.

    Attention entries commonly return `(out, lse)` and some return a dict, so a bare `out.detach()`
    raises AttributeError and — through `_run_leg` — surfaces as a generic non-zero-exit RuntimeError
    instead of naming the shape of the return value. Recurse over the container instead. Containers
    are matched BEFORE the tensor duck-check: a namedtuple carries both."""
    if isinstance(out, tuple) and hasattr(out, "_fields"):
        return type(out)(*(_snapshot(o) for o in out))
    if isinstance(out, (tuple, list)):
        return type(out)(_snapshot(o) for o in out)
    if isinstance(out, dict):
        return {k: _snapshot(v) for k, v in out.items()}
    if hasattr(out, "detach"):
        return out.detach().clone().cpu()
    if out is None or isinstance(out, (int, float, bool, str)):
        return out
    raise TypeError(
        f"oracle cannot record a {type(out).__name__} returned by the op — cases.call must return a "
        "tensor, or a tuple/list/dict of tensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", required=True, choices=("list", "resolve", "time", "oracle"))
    ap.add_argument("--bucket", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=0)
    a = ap.parse_args()

    task = os.path.abspath(a.task)
    with open(os.path.join(task, "meta.json")) as fh:
        meta = json.load(fh)

    if a.mode == "resolve":
        print(json.dumps(_identity(meta["target_callable"])))
        return

    h = _load("harness_lib", os.path.join(task, "harness_lib.py"))
    cases = _load("cases", os.path.join(task, "cases.py"))
    regime = meta.get("regime", {})

    if a.mode == "list":
        print(json.dumps({"sigs": [c["sig"] for c in cases.timing_cases(h, meta)]}))
        return

    call = cases.call
    if h.deployment_compile_mode(regime):
        call = h.compiled_op(call, regime)
    graph = h.deployment_graph_mode(regime)

    if a.mode == "time":
        sel = [c for c in cases.timing_cases(h, meta) if not a.bucket or c["sig"] == a.bucket]
        out = []
        for c in sel:
            r = h.time_op(lambda c=c: call(c["args"]), graph=graph, detail=True)
            out.append({"sig": c["sig"], "regime": c.get("regime", ""), "m": c.get("m"),
                        "ms": (r or {}).get("ms"), "wall_ms": (r or {}).get("wall_ms"),
                        "timer": (r or {}).get("timer")})
        print(json.dumps({"cases": out, "identity": _identity(meta["target_callable"])}))
        return

    torch = h._torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    draws = a.draws or int(meta.get("random_draws", 3))
    blob = {}
    for shape in cases.random_shapes(h, meta):
        for i in range(max(1, draws)):
            rng = torch.Generator(device=device).manual_seed(int(a.seed) + i)
            out = call(shape["make_inputs"](rng))
            blob[f"{shape['sig']}|{i}"] = _snapshot(out)
    torch.save(blob, a.out)
    print(json.dumps({"out": a.out, "n": len(blob)}))


if __name__ == "__main__":
    main()
