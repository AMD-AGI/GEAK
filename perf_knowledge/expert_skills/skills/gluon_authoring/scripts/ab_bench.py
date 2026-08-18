#!/usr/bin/env python3
"""Same-window INTERLEAVED A/B of N variants inside ONE process.

Why this exists (references/benchmark-hygiene.md mandates it; nothing implemented it):
on a box whose memory clock is not pinnable, two numbers produced by two processes are NOT
comparable -- `do_bench` goes bimodal and a 3% "win" is indistinguishable from a clock
excursion. The only construction that makes a small delta mean anything is to run every
variant inside ONE process, interleaved cell by cell, with the variant ORDER ROTATED per cell
so no variant always occupies the same position in the clock trajectory.

Correctness is gated BEFORE timing, in the same process. A variant whose oracle fails is never
timed at all: a faster-and-wrong candidate that gets pinned poisons the comparator and
everything transcribed from it downstream.

The tool also refuses to let you over-read the result: if a pairwise delta is smaller than the
measured cell-to-cell spread, it is reported as NOT RESOLVED rather than as a speedup.

Two failure modes are checked here rather than left to the adapter, because both have been
observed to pass every adapter-level gate while destroying the result:

* **A tolerance comparison cannot fail on NaN.** `NaN > tol` is False, so an all-NaN output
  scores ZERO out-of-tolerance elements and prints ALL PASS. Every non-finite metric is a
  failure here regardless of what the adapter's arithmetic concluded, and the optional
  `outputs()` hook lets this file scan the tensors itself, one level below the adapter.
* **Two variants differing only by a `constexpr` share a Triton cache entry**, and the second
  silently runs the first's binary. It is numerically perfect -- every variant computes the
  right answer, just not with its own code -- and it produces the most seductive possible
  artifact: a FLAT result set that reads as a clean scientific negative. The `fingerprint()`
  hook detects it outright; without the hook, a flat set is flagged as a COLLISION SUSPECT and
  `--permute` provides the discriminating experiment (do the numbers follow the code, or the
  position?).

------------------------------------------------------------------------------------------
Adapter contract -- your module (--module bench_mod.py) defines:

    def variants() -> dict[str, callable]:
        '''name -> zero-arg callable that runs ONE full iteration of the thing being timed.
        Build/compile everything OUTSIDE the callable; only the timed work goes inside.'''

    def oracle(name: str) -> dict:        # OPTIONAL but strongly recommended
        '''name -> {"dQ": 0.0014, "dK": 0.0016, ...} error metrics. ANY value above --tol
        fails that variant, which is then excluded from timing and reported as FAILED.
        A non-finite metric is ALWAYS a failure, whatever it is compared against.'''

    def outputs(name: str) -> object:     # OPTIONAL, strongly recommended
        '''name -> the variant's output tensor(s): one tensor, or a list/tuple/dict of them.
        Scanned here for non-finite values independently of oracle()'s tolerance math.'''

    def fingerprint(name: str) -> str:    # OPTIONAL, strongly recommended for layout sweeps
        '''name -> a hash of the COMPILED artifact (e.g. the .amdgcn text, or the Triton
        cache key). Two variants that are meant to differ and share a fingerprint are a
        cache collision, and this file exits non-zero rather than reporting their timings.'''

    def sync() -> None:                   # OPTIONAL; defaults to torch.cuda.synchronize()

Usage:
    python3 ab_bench.py --module bench_mod.py [--cells 5] [--iters 20] [--warmup 10]
                        [--tol 0.002] [--baseline NAME] [--json ab.json] [--permute]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("_ab_bench_mod", path)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot import {path!r}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ab_bench_mod"] = mod
    # the adapter lives next to the harness it drives -> let it import its siblings
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


def _default_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001  - a non-torch adapter syncs inside its own callable
        pass


def _iter_tensors(obj):
    """Flatten whatever outputs() handed back into individual tensors."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            for name, t in _iter_tensors(v):
                yield (f"{k}.{name}" if name else str(k)), t
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            for name, t in _iter_tensors(v):
                yield (f"[{i}].{name}" if name else f"[{i}]"), t
        return
    yield "", obj


def scan_non_finite(obj):
    """Count non-finite elements per output tensor, independently of any tolerance.

    This is the check a tolerance comparison structurally cannot make: `NaN > tol` is False,
    so an all-NaN output passes every `(ref - cand).abs() > tol` gate ever written. Returns
    {tensor_name: count} for the tensors that have any, or {} when everything is finite.
    """
    bad = {}
    for name, t in _iter_tensors(obj):
        try:
            import torch
            if isinstance(t, torch.Tensor):
                n = int((~torch.isfinite(t.float())).sum().item())
                if n:
                    bad[name or "out"] = n
                continue
        except Exception:  # noqa: BLE001  - no torch, or a tensor type it cannot cast
            pass
        try:  # numpy / array-likes / plain scalars
            if isinstance(t, (int, float)):
                if not math.isfinite(t):
                    bad[name or "out"] = 1
                continue
            import numpy as np
            arr = np.asarray(t, dtype=float)
            n = int((~np.isfinite(arr)).sum())
            if n:
                bad[name or "out"] = n
        except Exception:  # noqa: BLE001  - not scannable; silence is honest, a 0 would not be
            continue
    return bad


def gate_correctness(mod, names, tol):
    """Run the oracle for every variant BEFORE any timing. Returns (passed, report).

    Three independent reasons to fail, in the order they are checked: the oracle raised, a
    metric is non-finite, or a metric exceeds `tol`. The middle one is not the adapter's job --
    a NaN metric slips through `v > tol` as a PASS, so it is caught here.
    """
    oracle = getattr(mod, "oracle", None)
    outputs = getattr(mod, "outputs", None)
    if oracle is None and outputs is None:
        return list(names), {n: {"_oracle": "absent"} for n in names}
    passed, report = [], {}
    for n in names:
        rep = {}
        failed = False
        if outputs is not None:
            try:
                nf = scan_non_finite(outputs(n))
            except Exception as e:  # noqa: BLE001  - an outputs() that raises IS a failure
                rep["_error"] = f"outputs(): {type(e).__name__}: {e}"
                report[n] = rep
                continue
            if nf:
                rep["_non_finite_outputs"] = nf
                failed = True
        if oracle is not None:
            try:
                res = oracle(n) or {}
            except Exception as e:  # noqa: BLE001  - an oracle that raises IS a failure
                rep["_error"] = f"oracle(): {type(e).__name__}: {e}"
                report[n] = rep
                continue
            rep.update(res)
            nan_metrics = {k: v for k, v in res.items()
                           if isinstance(v, (int, float)) and not math.isfinite(v)}
            if nan_metrics:
                # NOT folded into _fail: a NaN metric means the comparison never happened,
                # which is a different fact from "the error is too large".
                rep["_non_finite_metrics"] = nan_metrics
                failed = True
            bad = {k: v for k, v in res.items()
                   if isinstance(v, (int, float)) and math.isfinite(v) and v > tol}
            if bad:
                rep["_fail"] = bad
                failed = True
        report[n] = rep
        if not failed:
            passed.append(n)
    return passed, report


def gate_fingerprints(mod, names):
    """Distinct variants sharing a compiled-artifact hash are a cache collision.

    Returns {fingerprint: [names]} for the colliding groups only. An adapter without a
    `fingerprint()` hook returns {} -- which is not evidence of absence, so the flat-result
    heuristic in `collision_suspect()` still applies.
    """
    fp = getattr(mod, "fingerprint", None)
    if fp is None:
        return {}
    groups: dict[str, list[str]] = {}
    for n in names:
        try:
            key = str(fp(n))
        except Exception as e:  # noqa: BLE001
            key = f"_error:{type(e).__name__}:{e}:{n}"
        groups.setdefault(key, []).append(n)
    return {k: v for k, v in groups.items() if len(v) > 1}


def collision_suspect(res, baseline):
    """A flat set across >=3 variants is a cache-collision suspect, not a finding.

    The seductive part of a cache collision is that it does not look broken: every arm passes
    its oracle and they all report the same number, which reads as 'the levers do not move the
    clock'. Flatness alone cannot distinguish that from a real negative, so it is escalated to
    a suspicion with a named discriminating experiment rather than reported as a result.
    """
    others = [n for n in res if n != baseline]
    if len(others) < 2:
        return None
    if all(res[n].get("resolved") is False for n in others):
        return ("FLAT RESULT SET across %d variants -- treat as a Triton cache-collision "
                "suspect until the arm ORDER has been permuted. Re-run with --permute (or "
                "reverse the arm order) and check whether each number follows the CODE or the "
                "POSITION. Do not record 'the levers do not move the clock' from this window."
                % (len(others) + 1))
    return None


def measure(mod, names, cells, iters, warmup, sync):
    """Interleaved cells with a ROTATED order. Per-cell value = median of `iters` timings;
    the rotation is what removes 'variant 0 always runs on the cold clock' bias."""
    fns = mod.variants()
    for n in names:                      # warm every variant before the first timed cell
        for _ in range(warmup):
            fns[n]()
    sync()
    cell_ms: dict[str, list[float]] = {n: [] for n in names}
    for c in range(cells):
        order = names[c % len(names):] + names[: c % len(names)]
        for n in order:
            fn, samples = fns[n], []
            for _ in range(iters):
                sync()
                t0 = time.perf_counter()
                fn()
                sync()
                samples.append((time.perf_counter() - t0) * 1e3)
            cell_ms[n].append(statistics.median(samples))
    return cell_ms


def summarize(cell_ms, baseline):
    out = {}
    for n, cells in cell_ms.items():
        lo, hi = min(cells), max(cells)
        out[n] = {
            "cells_ms": [round(v, 4) for v in cells],
            "stable_min": round(lo, 4),
            "median": round(statistics.median(cells), 4),
            # spread is the resolution floor of this window: a delta below it is not a result
            "spread_pct": round(100.0 * (hi - lo) / lo, 2) if lo else None,
        }
    b = out.get(baseline)
    for n, r in out.items():
        if not b or n == baseline:
            continue
        r["vs_baseline"] = {
            "min": round(b["stable_min"] / r["stable_min"], 4),
            "median": round(b["median"] / r["median"], 4),
        }
        delta_pct = abs(100.0 * (b["stable_min"] - r["stable_min"]) / b["stable_min"])
        noise = max(b["spread_pct"] or 0, r["spread_pct"] or 0)
        r["resolved"] = delta_pct > noise
        if not r["resolved"]:
            r["note"] = (f"delta {delta_pct:.2f}% <= spread {noise:.2f}% -- NOT RESOLVED in this "
                         f"window. Add cells, or discriminate with a clock-insensitive counter.")
    return out


def _selftest():
    # 1. a delta LARGER than the cell-to-cell spread is a result
    r = summarize({"base": [10.0, 10.1, 10.05], "cand": [8.0, 8.05, 8.02]}, "base")
    assert r["cand"]["resolved"] is True, r
    assert abs(r["cand"]["vs_baseline"]["min"] - 1.25) < 1e-3, r
    # 2. a delta SMALLER than the spread must NOT be sold as a speedup
    r = summarize({"base": [10.0, 11.0], "cand": [9.9, 10.9]}, "base")
    assert r["cand"]["resolved"] is False, r
    assert "NOT RESOLVED" in r["cand"]["note"], r
    # 3. correctness gates BEFORE timing: faster-and-wrong is excluded, not ranked
    class _M:
        @staticmethod
        def variants():
            return {"ok": lambda: None, "wrong": lambda: None}
        @staticmethod
        def oracle(n):
            return {"e": 0.9 if n == "wrong" else 1e-4}
    passed, rep = gate_correctness(_M, ["ok", "wrong"], 2e-3)
    assert passed == ["ok"] and "_fail" in rep["wrong"], (passed, rep)
    # 4. an oracle that RAISES is a failure, never a silent pass
    class _R:
        @staticmethod
        def oracle(n):
            raise RuntimeError("boom")
    passed, rep = gate_correctness(_R, ["x"], 2e-3)
    assert passed == [] and "_error" in rep["x"], (passed, rep)
    # 5. rotation actually rotates (no variant is always first)
    names = ["a", "b", "c"]
    firsts = {(names[c % 3:] + names[: c % 3])[0] for c in range(3)}
    assert firsts == {"a", "b", "c"}, firsts
    # 6. THE NaN TRAP: a non-finite metric slips through `v > tol` as a pass. It must not.
    class _N:
        @staticmethod
        def oracle(n):
            return {"max_rel": float("nan") if n == "allnan" else 1e-4}
    passed, rep = gate_correctness(_N, ["ok", "allnan"], 2e-3)
    assert passed == ["ok"], (passed, rep)
    assert "_non_finite_metrics" in rep["allnan"], rep
    assert (float("nan") > 2e-3) is False  # the trap this guards, stated as an assertion
    # 7. the outputs() hook catches non-finite tensors one level BELOW the adapter's arithmetic
    class _O:
        @staticmethod
        def outputs(n):
            return [1.0, float("inf")] if n == "poison" else [1.0, 2.0]
    passed, rep = gate_correctness(_O, ["clean", "poison"], 2e-3)
    assert passed == ["clean"], (passed, rep)
    assert rep["poison"]["_non_finite_outputs"], rep
    # 8. distinct variants sharing a compiled artifact are a collision, not a tie
    class _F:
        @staticmethod
        def fingerprint(n):
            return "same-amdgcn-hash" if n in ("v_a", "v_b") else n
    groups = gate_fingerprints(_F, ["anchor", "v_a", "v_b"])
    assert list(groups.values()) == [["v_a", "v_b"]], groups
    assert gate_fingerprints(object(), ["x", "y"]) == {}, "no hook -> no claim"
    # 9. a flat set across 3+ arms is escalated to a suspicion, never reported as a negative
    flat = summarize({"base": [4.05, 4.06], "v_a": [4.05, 4.06], "v_b": [4.06, 4.05]}, "base")
    assert collision_suspect(flat, "base") is not None, flat
    sharp = summarize({"base": [2.88, 2.89], "v_a": [3.40, 3.41], "v_b": [4.05, 4.06]}, "base")
    assert collision_suspect(sharp, "base") is None, sharp
    print("ab_bench selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", help="adapter module (see the contract above)")
    ap.add_argument("--selftest", action="store_true", help="offline checks, no GPU")
    ap.add_argument("--cells", type=int, default=5, help="interleaved cells per variant (default 5)")
    ap.add_argument("--iters", type=int, default=20, help="timed iterations per cell (default 20)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--tol", type=float, default=2e-3, help="oracle tolerance (default 0.002)")
    ap.add_argument("--baseline", help="variant to compare against (default: the first)")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--permute", action="store_true",
                    help="measure a second time with the arm order REVERSED and report whether "
                         "each number follows the code or the position (the discriminating "
                         "experiment for a suspected cache collision)")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.module:
        ap.error("--module is required (or use --selftest)")

    mod = _load_module(a.module)
    if not hasattr(mod, "variants"):
        sys.exit(f"{a.module}: no variants() -- see the adapter contract in this file's docstring")
    names = list(mod.variants().keys())
    if not names:
        sys.exit("variants() returned nothing")
    sync = getattr(mod, "sync", _default_sync)

    print(f"=== ab_bench: {len(names)} variants, {a.cells} interleaved cells x {a.iters} iters ===")

    collisions = gate_fingerprints(mod, names)
    if collisions:
        print("\n  CACHE COLLISION -- these variants share a compiled artifact:")
        for key, group in collisions.items():
            print(f"    {key[:24]}...  {group}")
        sys.exit("distinct variants are running the SAME binary. Give each arm its own kernel "
                 "object and its own TRITON_CACHE_DIR, then re-run. Timings from this window "
                 "would be numerically perfect and attributionally meaningless.")
    if getattr(mod, "fingerprint", None) is not None:
        print("  fingerprint gate PASS -- every variant has its own compiled artifact")

    passed, report = gate_correctness(mod, names, a.tol)
    for n in names:
        r = report[n]
        if n in passed:
            shown = {k: v for k, v in r.items() if not k.startswith("_")}
            print(f"  oracle PASS  {n:28s} {shown or '(no oracle -- results are UNGATED)'}")
        else:
            why = (r.get("_non_finite_outputs") and f"NON-FINITE outputs {r['_non_finite_outputs']}"
                   or r.get("_non_finite_metrics") and f"NON-FINITE metrics {r['_non_finite_metrics']}"
                   or r.get("_fail") or r.get("_error"))
            print(f"  oracle FAIL  {n:28s} {why}  -> NOT TIMED")
    if not passed:
        sys.exit("every variant failed its oracle; nothing to time")

    baseline = a.baseline or passed[0]
    if baseline not in passed:
        sys.exit(f"baseline {baseline!r} is not among the correctness-passing variants {passed}")

    res = summarize(measure(mod, passed, a.cells, a.iters, a.warmup, sync), baseline)
    print(f"\n{'variant':28s} {'stable_min':>11s} {'median':>9s} {'spread%':>8s} "
          f"{'vs base(min)':>13s}")
    for n in passed:
        r = res[n]
        vs = f"{r['vs_baseline']['min']:.4f}x" if n != baseline else "(baseline)"
        flag = "" if n == baseline or r.get("resolved", True) else "  <- NOT RESOLVED"
        print(f"{n:28s} {r['stable_min']:11.4f} {r['median']:9.4f} {r['spread_pct']:8.2f} "
              f"{vs:>13s}{flag}")
    for n in passed:
        if res[n].get("note"):
            print(f"\n  {n}: {res[n]['note']}")

    suspect = collision_suspect(res, baseline)
    if suspect:
        print(f"\n  COLLISION SUSPECT: {suspect}")

    permuted = None
    if a.permute:
        print("\n=== --permute: second window, arm order REVERSED ===")
        permuted = summarize(
            measure(mod, list(reversed(passed)), a.cells, a.iters, a.warmup, sync), baseline)
        print(f"{'variant':28s} {'fwd stable_min':>15s} {'rev stable_min':>15s} {'delta%':>8s}")
        followed_position = []
        for n in passed:
            f_ms, r_ms = res[n]["stable_min"], permuted[n]["stable_min"]
            d = 100.0 * (r_ms - f_ms) / f_ms if f_ms else 0.0
            noise = max(res[n]["spread_pct"] or 0, permuted[n]["spread_pct"] or 0)
            if abs(d) > max(noise, 1.0):
                followed_position.append(n)
            print(f"{n:28s} {f_ms:15.4f} {r_ms:15.4f} {d:8.2f}")
        if followed_position:
            print(f"\n  THE NUMBERS FOLLOWED THE POSITION, NOT THE CODE, for {followed_position}."
                  f"\n  That is a cache collision (or an un-cancelled clock trajectory), and NO "
                  f"ranking from either window is usable. Give each arm its own kernel object "
                  f"and its own TRITON_CACHE_DIR before re-measuring.")
        else:
            print("\n  Every number followed the CODE across the order flip -- the ranking is "
                  "order-independent and the result stands.")

    payload = {"baseline": baseline, "cells": a.cells, "iters": a.iters, "tol": a.tol,
               "oracle": report, "results": res,
               "fingerprint_gate": "pass" if getattr(mod, "fingerprint", None) else "absent",
               "collision_suspect": suspect,
               "permuted_results": permuted,
               "excluded_failing_oracle": [n for n in names if n not in passed]}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
