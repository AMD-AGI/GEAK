#!/usr/bin/env python3
"""Run ONE measurement leg of a kernel task. WHICH leg is decided ONLY by the overlay on PYTHONPATH.

  baseline leg   PYTHONPATH=<task>/baseline_overlay    -> the REAL online serving stack (install +
                 every already-accepted kernel). Nothing in the task dir can change what it imports.
  candidate leg  PYTHONPATH=<task>/_cand_overlay       -> that SAME stack + exactly ONE entry built
                 from kernel_src/ (meta.candidate_bind).

Both legs execute THIS file and the task's OWN cases.py, so `speedup = baseline_ms / candidate_ms` is
always measured against the live path, whatever LANGUAGE the candidate is written in.

Modes: list (bucket sigs) | resolve (leg identity, for the direction assert) | time | oracle | check.
Driven by harness_lib.measure_legs — a unittest.py should not invoke it directly.

MULTI-RANK (meta.world_size > 1, e.g. an EP8 MegaKernel): harness_lib launches this file under torchrun,
one process per rank. Each rank joins the process group (meta.dist_backend, default nccl), runs
cases.setup_dist(h, meta) when the task defines it, and then the SAME cases in the SAME order with the same
fixed iteration counts -- ranks that disagree on the order would wait on each other forever. `time`
reports the slowest rank's ms (the operator is as fast as its slowest rank) plus every rank's; `oracle`
saves one blob per rank (`<out>.rank<r>`); `check` compares every rank's outputs with that rank's blob.
Only rank 0 prints the JSON result.
"""
import argparse
import importlib
import importlib.util
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# this file lives IN the task dir next to a file named unittest.py — drop it from sys.path before
# torch is imported, or that file shadows stdlib unittest. Python puts the script's REAL directory there:
# a re-run dir that symlinks the task's files (fusion_handback) puts the task dir itself on sys.path.
_DIRS = {_HERE, os.path.dirname(os.path.realpath(__file__))}
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") not in _DIRS]


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


class _Dist:
    """This leg's ranks. One process (world 1) is a no-op, so single-rank tasks run exactly as before."""

    def __init__(self, meta):
        self.world = int(os.environ.get("WORLD_SIZE", "1") or 1)
        self.rank = int(os.environ.get("RANK", "0") or 0)
        self.local = int(os.environ.get("LOCAL_RANK", "0") or 0)
        self.meta = meta
        self.dist = None

    def start(self, h, cases):
        if self.world <= 1:
            return
        import datetime
        torch = h._torch()
        import torch.distributed as dist
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local)
        backend = self.meta.get("dist_backend") or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend, timeout=datetime.timedelta(
            seconds=int(self.meta.get("dist_timeout_s", 600))))
        self.dist = dist
        if hasattr(cases, "setup_dist"):
            cases.setup_dist(h, self.meta)

    def gather(self, obj):
        if not self.dist:
            return [obj]
        out = [None] * self.world
        self.dist.all_gather_object(out, obj)
        return out

    def barrier(self):
        if self.dist:
            self.dist.barrier()

    def stop(self, cases):
        if not self.dist:
            return
        if hasattr(cases, "teardown_dist"):
            cases.teardown_dist()
        self.dist.barrier()
        self.dist.destroy_process_group()

    def emit(self, obj):
        if self.rank == 0:
            print(json.dumps(obj))


def _check(h, cases, meta, case_execution, ref_prefix, seed, draws, base_tol, rank, override_tol=False):
    """This rank's outputs on the seeded draws against the same rank's recorded blob (eager), then the
    same draws through one captured graph per shape (static inputs refreshed in place, as the server
    replays it). Returns {"pass", "worst", "cases": [...]}."""
    torch = h._torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = f"{ref_prefix}.rank{rank}" if rank is not None else ref_prefix
    ref = torch.load(path, map_location="cpu")
    rows, ok_all, worst = [], True, 0.0

    def compare(sig, out, key, how):
        nonlocal ok_all, worst
        if key not in ref:
            ok, err = False, float("inf")
        elif meta.get("correct_metric") == "rel_l2":
            ok, err = _rel_l2(h, out, h.to_device_like(ref[key], _device_of(out, device)), tol)
        else:
            ok, err = h.correct(out, h.to_device_like(ref[key], _device_of(out, device)), tol)
        ok_all = ok_all and ok
        if err == err and err != float("inf"):
            worst = max(worst, err)
        rows.append({"case": f"{sig}|{key.split('|')[-1]}", "check": how, "correct": bool(ok),
                     "max_rel_err": None if err != err or err == float("inf") else round(float(err), 6)})

    for shape in cases.random_shapes(h, meta):
        case_call, case_graph = case_execution(shape)
        keys = [f"{shape['sig']}|{i}" for i in range(max(1, draws))]
        # An explicit --tol wins; else the case's own (measured-noise) tolerance; else meta.tol.
        tol = base_tol if override_tol else float(shape.get("tol") or base_tol)
        inputs = [shape["make_inputs"](torch.Generator(device=device).manual_seed(int(seed) + i))
                  for i in range(max(1, draws))]
        for key, args in zip(keys, inputs):
            compare(shape["sig"], case_call(args), key, "eager")
        if not (case_graph and torch.cuda.is_available()) or meta.get("graph_check") is False:
            continue
        static = shape["make_inputs"](torch.Generator(device=device).manual_seed(int(seed)))
        if str(_device_of(static, "cpu")).split(":")[0] != "cuda":
            continue                       # host inputs: there is no device graph to replay
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    case_call(static)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = case_call(static)
        except Exception as exc:          # a seam that cannot be captured fails the graph gate
            ok_all = False
            rows.append({"case": shape["sig"], "check": "graph", "correct": False, "note": f"capture: {exc!r}"[:300]})
            continue
        for key, args in zip(keys, inputs):
            _copy_into(static, args)
            g.replay()
            torch.cuda.synchronize()
            compare(shape["sig"], _snapshot(gout), key, "graph_replay")
    return {"pass": ok_all, "worst": worst, "cases": rows}


def _rel_l2(h, out, ref, tol):
    """meta.correct_metric = "rel_l2": ||out - ref|| / ||ref|| per output tensor (worst of them) <= tol. For
    operators whose own baseline is not elementwise reproducible (e.g. an EP MoE whose cross-rank arrival
    order changes the accumulation order), where tol is set from the baseline's measured self-noise."""
    po, pr = h.flatten_outputs(out), h.flatten_outputs(ref)
    if not po or len(po) != len(pr):
        return False, float("inf")
    worst = 0.0
    for o, r in zip(po, pr):
        if tuple(o.shape) != tuple(r.shape):
            return False, float("inf")
        o, r = o.float(), r.float()
        if not bool(o.isfinite().all()):
            return False, float("inf")
        worst = max(worst, float((o - r).norm() / r.norm().clamp_min(1e-12)))
    return worst <= tol, worst


def _device_of(out, default):
    """The device of the first tensor in an op's output (a graph-replay snapshot is on the host)."""
    if isinstance(out, dict):
        out = list(out.values())
    if isinstance(out, (tuple, list)):
        for o in out:
            dev = _device_of(o, None)
            if dev is not None:
                return dev
        return default
    return getattr(out, "device", default)


def _copy_into(dst, src):
    """Refresh a captured call's static inputs in place (tensors only; containers recursed)."""
    if isinstance(dst, dict):
        for k in dst:
            _copy_into(dst[k], src[k])
    elif isinstance(dst, (tuple, list)):
        for a, b in zip(dst, src):
            _copy_into(a, b)
    elif hasattr(dst, "copy_") and hasattr(src, "shape"):
        dst.copy_(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--mode", required=True, choices=("list", "resolve", "time", "oracle", "check"))
    ap.add_argument("--ref", default="", help="check: the baseline leg's oracle blob (per rank: <ref>.rank<r>)")
    ap.add_argument("--tol", type=float, default=0.0, help="check: tolerance (default meta.tol)")
    ap.add_argument("--bucket", default="")
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

    h = _load("harness_lib", os.path.join(task, "harness_lib.py"))
    cases = _load("cases", os.path.join(task, "cases.py"))
    regime = meta.get("regime", {})

    d = _Dist(meta)
    if a.mode == "list" and d.world <= 1:
        print(json.dumps({"sigs": [c["sig"] for c in cases.timing_cases(h, meta)]}))
        return
    if d.world > 1:
        d.start(h, cases)
        try:
            _main_ranks(a, h, cases, meta, regime, d)
        finally:
            d.stop(cases)
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
        sel = [c for c in cases.timing_cases(h, meta) if not a.bucket or c["sig"] == a.bucket]
        out = []
        for c in sel:
            case_call, case_graph = case_execution(c)
            r = h.time_op(lambda c=c: case_call(c["args"]), graph=case_graph, detail=True)
            out.append({"sig": c["sig"], "regime": c.get("regime", ""), "m": c.get("m"),
                        "ms": (r or {}).get("ms"), "wall_ms": (r or {}).get("wall_ms"),
                        "timer": (r or {}).get("timer"),
                        # Which cache condition produced `ms`. A decode A/B read under 'write-evict' is
                        # not comparable to one read under 'read-evict' (1.40 vs 1.12 on MI355X --
                        # see harness_lib.cache_policy), so the two must never be folded together.
                        "cache_condition": (r or {}).get("cache_condition")})
        print(json.dumps({"cases": out, "identity": _identity(meta["target_callable"])}))
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


def _case_execution_for(h, cases, meta, regime):
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
    return case_execution


def _main_ranks(a, h, cases, meta, regime, d):
    """time / oracle / check on every rank of a multi-rank leg (see the module docstring)."""
    case_execution = _case_execution_for(h, cases, meta, regime)
    torch = h._torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    draws = a.draws or int(meta.get("random_draws", 3))
    if a.mode == "list":                     # a multi-rank task's cases build their inputs on every rank
        d.emit({"sigs": [c["sig"] for c in cases.timing_cases(h, meta)]})
        return
    if a.mode == "time":
        sel = [c for c in cases.timing_cases(h, meta) if not a.bucket or c["sig"] == a.bucket]
        out = []
        for c in sel:
            case_call, case_graph = case_execution(c)
            d.barrier()
            r = h.time_op(lambda c=c: case_call(c["args"]), graph=case_graph, detail=True) or {}
            per = d.gather({"ms": r.get("ms"), "wall_ms": r.get("wall_ms"), "timer": r.get("timer")})
            ms = [x["ms"] for x in per]
            out.append({"sig": c["sig"], "regime": c.get("regime", ""), "m": c.get("m"),
                        # The slowest rank is the operator's time; a rank that could not be timed voids it.
                        "ms": max(ms) if all(isinstance(x, (int, float)) for x in ms) else None,
                        "wall_ms": max((x["wall_ms"] or 0) for x in per) or None,
                        "timer": per[0]["timer"], "per_rank_ms": ms,
                        "cache_condition": r.get("cache_condition")})
        d.emit({"cases": out, "world_size": d.world, "identity": _identity(meta["target_callable"])})
        return
    if a.mode == "oracle":
        blob = {}
        for shape in cases.random_shapes(h, meta):
            case_call, _ = case_execution(shape)
            for i in range(max(1, draws)):
                rng = torch.Generator(device=device).manual_seed(int(a.seed) + i)
                blob[f"{shape['sig']}|{i}"] = _snapshot(case_call(shape["make_inputs"](rng)))
        torch.save(blob, f"{a.out}.rank{d.rank}")
        n = d.gather(len(blob))
        d.emit({"out": a.out, "per_rank": n, "world_size": d.world})
        return
    if a.mode == "check":
        tol = a.tol or float(meta.get("tol", 0.02))
        mine = _check(h, cases, meta, case_execution, a.ref, a.seed, draws, tol, d.rank, override_tol=bool(a.tol))
        per = d.gather(mine)
        by_case = {}
        for p in per:
            for c in p["cases"]:
                k = f"{c['case'].rsplit('|', 1)[0]}:{c['check']}"
                e = c.get("max_rel_err")
                by_case[k] = None if e is None or by_case.get(k, 0.0) is None else max(by_case.get(k, 0.0), e)
        d.emit({"pass": all(p["pass"] for p in per), "worst": max(p["worst"] for p in per),
                "worst_by_case": by_case,
                "per_rank": [{"rank": r, "pass": p["pass"], "worst": p["worst"],
                              "failed": [c for c in p["cases"] if not c["correct"]][:8]}
                             for r, p in enumerate(per)], "world_size": d.world, "tol": tol})
        return
    raise SystemExit(f"mode {a.mode} is not a multi-rank mode")


if __name__ == "__main__":
    main()
