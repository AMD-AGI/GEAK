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
import time

_STARTED = time.perf_counter()
_SETUP = {}

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


def timing_rows(cases, h, meta, bucket="", sigs=None, metadata_only=False):
    """Timed cases from metadata-only `timing_specs(h, meta)`, filtered by --bucket and ordered by
    --sigs; `make_timing_case(spec)` builds ONE shape's operands at a time, so --mode list builds
    none and --mode time never holds two shapes' operands at once."""
    specs = [c for c in cases.timing_specs(h, meta) if not bucket or c["sig"] == bucket]
    if sigs is not None:
        order = {sig: i for i, sig in enumerate(json.loads(sigs))}
        specs = sorted((c for c in specs if c["sig"] in order), key=lambda c: order[c["sig"]])
    for spec in specs:
        yield spec if metadata_only else cases.make_timing_case(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", required=True, choices=("list", "resolve", "time", "oracle"))
    ap.add_argument("--bucket", default="")
    ap.add_argument("--sigs", default=None, help="time mode: JSON list of sigs, timed in THIS order")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draws", type=int, default=0)
    ap.add_argument("--target-callable", default="", help="identity probe only; defaults to meta target")
    a = ap.parse_args()

    task = os.path.abspath(a.task)
    with open(os.path.join(task, "meta.json")) as fh:
        meta = json.load(fh)

    if a.mode == "resolve":
        print(json.dumps(_identity(a.target_callable or meta["target_callable"])))
        return
    if a.target_callable:
        ap.error("--target-callable is only valid with --mode resolve")

    _SETUP["environment_s"] = time.perf_counter() - _STARTED
    imports_started = time.perf_counter()
    h = _load("harness_lib", os.path.join(task, "harness_lib.py"))
    cases = _load("cases", os.path.join(task, "cases.py"))
    _SETUP["task_import_s"] = time.perf_counter() - imports_started
    regime = meta.get("regime", {})

    if a.mode == "list":
        print(json.dumps({"sigs": [c["sig"] for c in timing_rows(cases, h, meta, metadata_only=True)]}))
        return

    call = cases.call
    if meta.get("phase_regimes") is None and h.deployment_compile_mode(regime):
        call = h.compiled_op(call, regime)
    graph = h.deployment_graph_mode(regime)

    def case_execution(case):
        phase_regimes = meta.get("phase_regimes")
        if phase_regimes is None:
            return call, graph
        phase = case.get("regime", "")
        if not isinstance(phase_regimes, dict) or not isinstance(phase_regimes.get(phase), dict):
            raise ValueError(f"missing phase_regimes entry for {phase!r}")
        phase_regime = phase_regimes[phase]
        phase_call = cases.call
        if h.deployment_compile_mode(phase_regime):
            phase_call = h.compiled_op(phase_call, phase_regime)
        return phase_call, h.deployment_graph_mode(phase_regime)

    if a.mode == "time":
        import gc
        init_started = time.perf_counter()
        torch = h._torch()
        have_cuda = torch.cuda.is_available()
        if have_cuda:
            torch.cuda.init()
            h.sync(torch)
        _SETUP["runtime_init_s"] = time.perf_counter() - init_started
        sel = iter(timing_rows(cases, h, meta, a.bucket, a.sigs))
        out = []
        while True:
            input_started = time.perf_counter()
            try:
                c = next(sel)
            except StopIteration:
                break
            h.sync(torch)
            inputs_s = time.perf_counter() - input_started
            case_call, case_graph = case_execution(c)
            timing_started = time.perf_counter()
            r = h.time_op(lambda c=c: case_call(c["args"]), graph=case_graph, detail=True)
            timing_s = time.perf_counter() - timing_started
            row = {"sig": c["sig"], "regime": c.get("regime", ""), "m": c.get("m"),
                   "ms": (r or {}).get("ms"), "wall_ms": (r or {}).get("wall_ms"),
                   "timer": (r or {}).get("timer"),
                   # Which cache condition produced `ms`. A decode A/B read under 'write-evict' is
                   # not comparable to one read under 'read-evict' (1.40 vs 1.12 on MI355X --
                   # see harness_lib.cache_policy), so the two must never be folded together.
                   "cache_condition": (r or {}).get("cache_condition"),
                   **({"error": r["error"]} if r and r.get("error") else {}),
                   # Setup cost OUTSIDE the device-event window (ms is unaffected by any of it).
                   "diagnostics": dict((r or {}).get("stages_s") or {}, inputs_s=inputs_s,
                                       time_op_s=timing_s)}
            # Whether `ms` is device work or host dispatch latency (time_op's three states:
            # True / False / ABSENT); carry both keys or neither.
            if r and "primed" in r:
                row["primed"], row["host_ms"] = r["primed"], r.get("host_ms")
            out.append(row)
            # One process streams every shape: do not keep this shape's operands alive while the
            # next one is built. JIT/autotune state stays warm; unused tensor storage is freed.
            cleanup_started = time.perf_counter()
            del c, case_call, r
            # A full scan of torch/serving-stack's long-lived Python heap per shape can cost hundreds
            # of ms. Refcounts release normal tensor owners; collect young cycles here and scan the
            # older generations periodically.
            gc.collect(0)
            if len(out) % 32 == 0:
                gc.collect()
            if have_cuda:
                h.sync(torch)
                torch.cuda.empty_cache()
                row["diagnostics"]["live_tensor_bytes_after"] = torch.cuda.memory_allocated()
            row["diagnostics"]["cleanup_s"] = time.perf_counter() - cleanup_started
        print(json.dumps({"cases": out, "identity": _identity(meta["target_callable"]),
                          "process": dict(_SETUP, child_s=time.perf_counter() - _STARTED)}))
        return

    torch = h._torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    draws = a.draws or int(meta.get("random_draws", 3))
    blob = {}
    for shape in cases.random_shapes(h, meta):
        case_call, _ = case_execution(shape)
        for i in range(max(1, draws)):
            rng = torch.Generator(device=device).manual_seed(int(a.seed) + i)
            out = case_call(shape["make_inputs"](rng))
            blob[f"{shape['sig']}|{i}"] = _snapshot(out)
    torch.save(blob, a.out)
    print(json.dumps({"out": a.out, "n": len(blob)}))


if __name__ == "__main__":
    main()
