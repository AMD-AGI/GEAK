#!/usr/bin/env python3
"""Interleaved same-process A/B screening through GEAK's harness_lib.time_op.

The median is the ranking key. Optional identical-code controls measure a noise
band; rotated arm order and --permute test sensitivity to order/window effects.
A screen pass requires correctness evidence, valid device-timer receipts, complete
rotation, and a speedup above both the noise floor and --min-improve-pct (default 2).
That threshold is caller-configurable; final acceptance uses GEAK verify / measure_legs
and the workflow's own deployment, workload and improvement requirements.

Adapter contract (--module adapter.py):
  variants() -> {name: zero_argument_callable}. Construct/compile outside timing.
  oracle(name) -> non-empty dict of numeric, finite, non-negative error metrics.
    Every metric must be <= --tol. Failed variants are excluded before timing.
  outputs(name), reference(name) -> tensors or matching nested tensor containers.
    Optional pair: uses harness_lib.correct, including matching-infinity semantics.
    outputs() alone scans for non-finite values but cannot verify correctness.
  fingerprint(name) -> compiled-artifact identity. Optional collision guard.
  toolchain(name) -> compiler identity. Required when variants patch the compiler;
    differing toolchains require separate processes and GEAK verify.
  sync() -> device synchronization. Optional; defaults to torch.cuda.synchronize.

Provide oracle() or outputs()+reference() to verify correctness for screening.
Missing evidence permits diagnostic timing only (screen invalid, exit 2). These hooks
are not a replacement for GEAK's complete random/replay correctness verification.
A fingerprint collision rejects the run; identical timings alone are only a diagnostic
suspicion. Neither order stability nor a distinct fingerprint proves kernel correctness.

JSON schema v2 contains validity/reasons, final results and both windows' controls,
per-cell measurement receipts and effort. A rejected window revokes all screen passes.
Exit 0 = valid screening run (possibly no speedup), 2 = invalid screening/CLI input,
1 = execution/gate failure. --cache hot is diagnostic only and always invalid for screening.
Legacy min remains descriptive; removed stable_min is not the ranking metric.

Usage: python3 ab_bench.py --module adapter.py --baseline base --control auto \
           --cells 6 --permute --json ab.json
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
from pathlib import Path


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
    except Exception:  # noqa: BLE001, S110 -- non-torch adapter owns synchronization
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
        except Exception:  # noqa: BLE001, S110 -- no torch; try scalar/array fallback
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
        except Exception:  # noqa: BLE001 -- unsupported outputs cannot establish finite values
            bad[name or "out"] = "unscannable output"
    return bad


def gate_correctness(mod, names, tol):
    """Screening oracle. Absent evidence is unverified; failed evidence is never timed.

    An optional reference(name) paired with outputs(name) uses GEAK's comparator,
    including matching-infinity semantics. Legacy oracle(name) returns numeric error
    metrics; every metric must be finite and non-negative.
    """
    oracle = getattr(mod, "oracle", None)
    outputs = getattr(mod, "outputs", None)
    reference = getattr(mod, "reference", None)
    passed, report = [], {}
    for n in names:
        rep = {"_verification": "unverified"}
        failed, verified = False, False
        try:
            if reference is not None and outputs is None:
                raise ValueError("reference() requires outputs()")
            if outputs is not None:
                out = outputs(n)
                if reference is not None:
                    ok, err = load_harness_lib().correct(out, reference(n), tol)
                    rep["_reference"] = {"correct": bool(ok), "max_rel_err": err}
                    failed = not ok or not math.isfinite(err)
                    verified = not failed
                else:
                    nf = scan_non_finite(out)
                    if nf:
                        rep["_non_finite_outputs"] = nf
                        failed = True  # infinity needs a reference, NaN never matches
            if oracle is not None:
                res = oracle(n)
                if not isinstance(res, dict) or not res:
                    raise ValueError("oracle() must return a non-empty dict of numeric error metrics")
                if any(not isinstance(k, str) or k.startswith("_") or type(v) not in (int, float)
                       for k, v in res.items()):
                    raise ValueError("oracle() keys must be metric names, values numeric errors")
                rep.update(res)
                nf = {k: v for k, v in res.items() if not math.isfinite(v)}
                bad = {k: v for k, v in res.items() if math.isfinite(v) and (v < 0 or v > tol)}
                if nf:
                    rep["_non_finite_metrics"] = nf
                if bad:
                    rep["_fail"] = bad
                failed = failed or bool(nf or bad)
                verified = not failed
            elif reference is None:
                rep["_oracle"] = "absent"
        except Exception as e:  # noqa: BLE001 -- a failed hook cannot certify correctness
            rep["_error"] = f"{type(e).__name__}: {e}"
            failed = True
        rep["_verification"] = "fail" if failed else "pass" if verified else "unverified"
        report[n] = rep
        if not failed:
            passed.append(n)
    return passed, report


def gate_fingerprints(mod, names):
    """Distinct variants sharing a compiled-artifact hash are a cache collision.

    Returns `(collisions, errors)`: `{fingerprint: [names]}` for the colliding groups only,
    and `{name: "ExcType: msg"}` for the variants whose hook raised. An adapter without a
    `fingerprint()` hook returns `({}, {})` -- which is not evidence of absence, so the
    flat-result heuristic in `collision_suspect()` still applies.

    A raising hook is an ERROR, never a pass. An earlier version folded the exception into
    the group key together with the variant name, which made every failure its own unique
    key: the gate could not then find a collision, and printed PASS for a run in which it had
    not executed at all. That is the worst available outcome for a gate whose whole job is to
    refuse a window -- so the two states are kept apart and the caller exits on either.
    """
    fp = getattr(mod, "fingerprint", None)
    if fp is None:
        return {}, {}
    groups: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    for n in names:
        try:
            key = str(fp(n))
        except Exception as e:  # noqa: BLE001 - a hook that raises IS a gate failure
            errors[n] = f"{type(e).__name__}: {e}"
            continue
        groups.setdefault(key, []).append(n)
    return {k: v for k, v in groups.items() if len(v) > 1}, errors


def collision_suspect(res, baseline):
    """A flat set across >=3 variants is a cache-collision suspect, not a finding.

    The seductive part of a cache collision is that it does not look broken: every arm passes
    its oracle and they all report the same number, which reads as 'the levers do not move the
    clock'. Flatness alone cannot distinguish that from a real negative, so it is escalated to
    a suspicion with a named discriminating experiment rather than reported as a result.
    """
    # The control arm is a byte-identical duplicate ON PURPOSE, so it is flat against its twin by
    # construction. Counting it here would turn the instrument that MEASURES the noise band into
    # evidence of a cache collision, and would do it on exactly the windows that are best
    # instrumented.
    others = [n for n in res if n != baseline and not n.startswith(CONTROL_PREFIX)]
    if len(others) < 2:
        return None
    if all(res[n].get("resolved") is False for n in others):
        return (f"FLAT RESULT SET across {len(others) + 1} variants -- treat as a Triton cache-collision "
                "suspect until the arm ORDER has been permuted. Re-run with --permute (or "
                "reverse the arm order) and check whether each number follows the CODE or the "
                "POSITION. Do not record 'the levers do not move the clock' from this window.")
    return None


# --------------------------------------------------------------------------- the instrument
# GEAK's harness_lib is the single owner of timing in GEAK. This file never times a kernel itself:
# every sample comes from `harness_lib.time_op(detail=True)`, the same CUDA-event / per-sample sync /
# read-evict / median instrument GEAK verify uses, so a screen and an acceptance number cannot
# disagree because of the METER. (They still differ by process: acceptance is a fresh process per
# leg, this is one process per window -- which is exactly why this is screening only.)
_REPO_HARNESS_LIB = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "e2e_workflow", "scripts", "harness_lib.py"))


class HarnessLibUnavailable(RuntimeError):
    """harness_lib could not be resolved; the message says where it looked and how to fix it."""


def harness_lib_path() -> str:
    """Where harness_lib.py is expected: $GEAK_HARNESS_LIB (file or dir), else the repo copy."""
    env = os.environ.get("GEAK_HARNESS_LIB")
    if env:
        return env if env.endswith(".py") else os.path.join(env, "harness_lib.py")
    return _REPO_HARNESS_LIB


def load_harness_lib():
    """Import GEAK's e2e_workflow/scripts/harness_lib.py by PATH (no sys.path games).

    An explicit $GEAK_HARNESS_LIB that does not exist is an error, never a silent fallback to the
    repo copy: an override that is ignored is worse than one that fails."""
    path = harness_lib_path()
    if not os.path.isfile(path):
        src = "$GEAK_HARNESS_LIB" if os.environ.get("GEAK_HARNESS_LIB") else "the GEAK checkout"
        raise HarnessLibUnavailable(
            f"harness_lib.py not found at {path} (from {src}). ab_bench times ONLY through GEAK's "
            f"e2e_workflow/scripts/harness_lib.py:time_op -- point GEAK_HARNESS_LIB at it (file or "
            f"its directory), or run from a GEAK checkout.")
    spec = importlib.util.spec_from_file_location("geak_harness_lib", path)
    if spec is None or spec.loader is None:
        raise HarnessLibUnavailable(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "time_op", None)):
        raise HarnessLibUnavailable(f"{path} has no time_op() -- not GEAK's harness_lib")
    return mod


def _reps_for(fn, sync, budget_ms, probe=5):
    """How many back-to-back launches fit in `budget_ms`, from the kernel's OWN duration.

    Used only to size the HOT burst (`inner` of time_op). A fixed count spends the same effort on a
    16 us kernel and a 400 us one, so the short kernel's reading is dominated by whatever else the box
    did (measured on the Gluon pack's own cases: 13.9% spread at a fixed 60 iterations, 0.8% at a budget).
    This probe is a SIZING estimate, never a reported number."""
    for _ in range(probe):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(probe):
        fn()
    sync()
    est_ms = max(1e-4, (time.perf_counter() - t0) * 1e3 / probe)
    return max(1, int(budget_ms / est_ms)), est_ms


HOT_SAMPLES = 5


def _time_cell(fn, hl, iters, budget_ms, cold, sync, warmup=1):
    """One cell's reading through harness_lib: (ms_per_call, effort, receipt).

    cold (default, the acceptance protocol): `time_op(inner=1, repeats=iters)` -- one launch per
      sample, a read-evict flush before every sample, CUDA-event device time, median.
    hot (SEARCH-ONLY diagnostic): `time_op(inner=N, repeats=HOT_SAMPLES)` with N sized from the
      budget -- each sample is one read-evict and then N back-to-back launches, so all but the first
      read warm data. It is still harness_lib's events/sync/median; it is NOT what acceptance
      measures, and a hot ranking may disagree with a cold one (measured: on a per-token quant
      kernel hot preferred the tuned config, cold the shipped one).
    Raises RuntimeError when time_op returns None (the variant raised under timing)."""
    if cold:
        inner, repeats = 1, max(1, iters)
    else:
        n = _reps_for(fn, sync, budget_ms)[0] if budget_ms and budget_ms > 0 else max(1, iters)
        inner, repeats = max(1, n // HOT_SAMPLES), HOT_SAMPLES
    d = hl.time_op(fn, warmup=max(1, warmup), repeats=repeats, inner=inner, detail=True)
    if d is None:
        raise RuntimeError("harness_lib.time_op returned None -- the variant raised while timed")
    if not isinstance(d, dict):            # a time_op without detail support: wrap, label it
        d = {"ms": float(d), "timer": "unknown"}
    if not math.isfinite(float(d["ms"])) or float(d["ms"]) <= 0:
        raise RuntimeError("harness_lib returned a non-positive or non-finite timing")
    return float(d["ms"]), {"inner": inner, "repeats": repeats}, d


def gate_toolchain(mod, names):
    """One process per toolchain-patched variant. Returns (identities, error_or_None).

    A patched compiler (an injected pass, a reinjected pipeliner) is process-global: two arms that
    need different toolchains cannot both be built correctly in one process, and the arm built
    second silently runs under the first one's compiler. No hook -> ({}, None): every arm is taken
    to need the stock toolchain."""
    hook = getattr(mod, "toolchain", None)
    if hook is None:
        return {}, None
    ids = {}
    for n in names:
        try:
            ids[n] = str(hook(n))
        except Exception as e:  # noqa: BLE001 - a raising hook is a gate failure, not a pass
            return ids, f"toolchain({n!r}) raised {type(e).__name__}: {e}"
    if len(set(ids.values())) > 1:
        return ids, ("arms need different toolchains " + json.dumps(ids, sort_keys=True) + " -- run "
                     "ONE window per toolchain (one process per toolchain-patched variant) and compare "
                     "across them through GEAK verify / harness_lib.measure_legs, never in-process")
    return ids, None


CONTROL_PREFIX = "__control__"


def control_name(of: str) -> str:
    return CONTROL_PREFIX + of


def with_control(fns: dict, names: list, of: str) -> tuple[dict, list]:
    """Add a BYTE-IDENTICAL duplicate arm, so the window measures its own noise band.

    The duplicate is literally the same callable under a second name: same object, same compiled
    artifact, same everything. Its two readings differ by exactly what the box did between them, so
    the spread across its cells IS this window's resolution floor -- measured rather than assumed,
    and measured on THIS kernel's duration, which matters because the placebo band is strongly
    duration-dependent (roughly a factor of 200 across a measured duration range: 3.68% at 6.3 us,
    0.2-0.3% at tens of ms).

    Why this is not optional bookkeeping. Over one 1208-round campaign the MEDIAN kept round was
    +0.833%, 31.6% of rounds marked a win recorded a NON-POSITIVE delta, and p25 was exactly
    0.000%. When the median win sits inside the band, ranking arms by end-to-end time is not
    merely noisy -- it is arithmetic performed on numbers that do not order. Everything downstream
    that ranks (BRANCH convergence, a sweep's winner, a kept verdict) needs this number first.

    The technique was invented in the field and then criticised by its own author, and the
    criticism is the design constraint here: *"all eight such controls ever run put ONE inert arm
    in ONE slot"*. A control pinned to one position can only falsify a slot-0 bias; it says nothing
    about any other position effect. So the control is a FIRST-CLASS member of `names` and rides
    the same per-cell rotation as every real arm -- see `rotation_coverage()` for the check that
    the rotation actually completed."""
    if of not in fns:
        raise KeyError(of)
    cn = control_name(of)
    return dict(fns, **{cn: fns[of]}), [*names, cn]


def rotation_coverage(names, cells) -> dict:
    """Did every arm actually occupy every slot, or did the window stop mid-cycle?

    The rotation is `order = names[c % n:] + names[:c % n]`, so slot coverage is complete only
    after `n` cells. At `cells < n` some arms never reach slot 0 and others hold it repeatedly --
    which is a systematic position penalty, not merely weak de-biasing, and it is invisible in the
    output because every arm still produces a full row of numbers."""
    n = len(names)
    full = cells // n
    return {"variants": n, "cells": cells, "full_cycles": full,
            "complete": cells % n == 0 and full >= 1,
            "slots_per_variant": {nm: sorted({(c - names.index(nm)) % n for c in range(cells)})
                                  for nm in names} if n <= 8 else None}


def preload(seconds: float, fn, sync) -> None:
    """Burn `seconds` of real compute before the first timed cell.

    A GPU that has been idle spends its first timed window on a decaying clock/power excursion,
    and the excursion lands on whichever arm holds the early slot rather than on the arm that
    deserves it: one measured run read the SAME empty kernel at 10.7 us in slot 1 and 6.06 us in
    slot 5. A 3 s compute pre-load removed it 6 times out of 6. This is distinct from `--warmup`,
    which warms each variant's caches and compilation; this warms the CARD."""
    if seconds <= 0:
        return
    end = time.time() + seconds
    while time.time() < end:
        for _ in range(32):
            fn()
    sync()


def measure(mod, names, cells, iters, warmup, sync, budget_ms=100.0, cold=True,
            fns=None, preload_s=0.0, hl=None):
    """Interleaved cells with a ROTATED order. Per-cell value = one harness_lib.time_op reading;
    the rotation is what removes 'variant 0 always runs on the cold clock' bias."""
    hl = hl if hl is not None else load_harness_lib()
    fns = fns if fns is not None else mod.variants()
    preload(preload_s, fns[names[0]], sync)
    for n in names:                      # warm every variant before the first timed cell
        for _ in range(warmup):
            fns[n]()
    sync()
    cell_ms: dict[str, list[float]] = {n: [] for n in names}
    used: dict[str, dict] = {}
    receipts: dict[str, dict] = {}
    for c in range(cells):
        order = names[c % len(names):] + names[: c % len(names)]
        for n in order:
            ms, effort, receipt = _time_cell(fns[n], hl, iters, budget_ms, cold, sync)
            cell_ms[n].append(ms)
            used[n] = effort
            receipts.setdefault(n, []).append({k: receipt.get(k) for k in
                ("timer", "primed", "host_ms", "cache_condition", "wall_ms")})
    measure.iters_used = used            # reported, so a reading's effort is auditable
    measure.receipts = receipts          # harness_lib's own receipt (timer, primed, cache policy)
    return cell_ms


def control_band_pct(cell_ms, control_of=None):
    """The window's measured noise band, read off the byte-identical control arm.

    Two numbers, because they answer different questions and conflating them is how a placebo
    reads as a win. `spread_pct` is the control's own cell-to-cell spread; `vs_twin_pct` is how far
    the control's MEDIAN lands from its own twin's -- the same code compared against itself through
    the whole pipeline, which is the number a candidate's delta has to beat."""
    if not control_of:
        return None
    cn = control_name(control_of)
    if cn not in cell_ms or control_of not in cell_ms:
        return None
    c, t = cell_ms[cn], cell_ms[control_of]
    lo, hi = min(c), max(c)
    spread = 100.0 * (hi - lo) / lo if lo else 0.0
    mc, mt = statistics.median(c), statistics.median(t)
    twin = abs(100.0 * (mc - mt) / mt) if mt else 0.0
    return {"of": control_of, "control": cn,
            "spread_pct": spread, "vs_twin_pct": twin,
            "band_pct": max(spread, twin)}


MIN_IMPROVE_PCT = 2.0   # configured screening threshold; a measured band may only RAISE this bar


def summarize(cell_ms, baseline, band=None, min_improve_pct=MIN_IMPROVE_PCT):
    """Median is the headline and the ranking key; min rides along as an extra field.

    Verdicts per non-baseline arm:
      resolved          delta exceeds this window's noise floor (control band, else arm spread)
      clears_min_improve  median speedup >= 1 + min_improve_pct/100 (configured screening threshold)
      screen            'pass' only when BOTH hold -- the band can only make the bar stricter.
                        'pass' means "send to GEAK verify", never "accepted"."""
    out = {}
    for n, cells in cell_ms.items():
        lo, hi = min(cells), max(cells)
        out[n] = {
            "cells_ms": list(cells),
            "median": statistics.median(cells),
            "min": lo,                       # extra field; never the ranking key
            # spread is the resolution floor of this window: a delta below it is not a result
            "spread_pct": 100.0 * (hi - lo) / lo if lo else None,
        }
    b = out.get(baseline)
    for n, r in out.items():
        if not b or n == baseline:
            continue
        sp_med = b["median"] / r["median"] if r["median"] else None
        r["vs_baseline"] = {
            "median": round(sp_med, 4) if sp_med else None,
            "min": round(b["min"] / r["min"], 4) if r["min"] else None,
        }
        delta_pct = abs(100.0 * (b["median"] - r["median"]) / b["median"]) if b["median"] else 0.0
        # A measured control outranks the per-arm spread. Both are floors on what this window can
        # resolve, and the control's is the honest one: it is the same CODE read twice, so it
        # carries every position, thermal and cache effect a candidate's delta also rides on.
        noise = max(b["spread_pct"] or 0, r["spread_pct"] or 0)
        src = "arm_spread"
        if band and band["band_pct"] > noise:
            noise, src = band["band_pct"], "control_arm"
        r["noise_band_pct"] = round(noise, 3)
        r["noise_band_source"] = src
        r["resolved"] = delta_pct > noise
        r["clears_min_improve"] = bool(sp_med and sp_med >= 1.0 + min_improve_pct / 100.0)
        r["screen"] = "pass" if (r["resolved"] and r["clears_min_improve"]) else (
            "slower" if sp_med and sp_med < 1.0 and r["resolved"] else "no")
        if not r["resolved"]:
            r["note"] = (f"delta {delta_pct:.2f}% <= {src} band {noise:.2f}% -- NOT RESOLVED in "
                         f"this window. Add cells, or discriminate with a clock-insensitive "
                         f"counter. Record it as `--verdict null`, not as a small win.")
        elif not r["clears_min_improve"] and sp_med and sp_med >= 1.0:
            r["note"] = (f"resolved, but {100 * (sp_med - 1):.2f}% < the {min_improve_pct:.1f}% "
                         f"screening threshold -- not a screen pass (the band never lowers that bar).")
    return out


def window_invalid_reasons(window, coverage):
    reasons = []
    if not coverage["complete"]:
        reasons.append("incomplete arm rotation")
    for name, receipts in window["receipts"].items():
        if any(r.get("timer") not in ("cuda_event", "cuda_event_graph") or r.get("primed") is not True
               or (r.get("cache_condition") or {}).get("mode") != "read-evict"
               for r in receipts):
            reasons.append(f"unverified device timing/priming: {name}")
    return reasons


def invalidate_screens(results, reasons):
    for row in results.values():
        if "screen" in row:
            row["screen"] = "invalid"
            row["invalid_reasons"] = list(reasons)


class _MockHL:
    """A GPU-free stand-in for harness_lib: time_op returns whatever the callable returns as its
    ms, and records how it was asked to time (inner / repeats / detail)."""

    def __init__(self):
        self.calls = []

    def time_op(self, call, warmup=10, repeats=50, inner=1, graph=False, *, detail=False):
        self.calls.append({"warmup": warmup, "repeats": repeats, "inner": inner, "detail": detail})
        ms = call()
        if ms is None:
            return None
        d = {"ms": float(ms), "wall_ms": float(ms), "timer": "mock", "primed": True,
             "host_ms": 0.001, "cache_condition": {"mode": "read-evict"}}
        return d if detail else d["ms"]


def _selftest():
    nop = lambda: None

    # 0a. ADAPTIVE BUDGET (hot burst sizing): the count must come from the callable's own duration.
    # Test the sizing policy with a clock advanced by the callable. Real sleep
    # includes host scheduling pauses and made this CPU-only check flaky under load.
    from unittest.mock import patch
    ticks = [0.0]
    def timed_call(us):
        def call():
            ticks[0] += us / 1e6
        return call
    with patch.object(time, "perf_counter", side_effect=lambda: ticks[0]):
        n_fast, fast_ms = _reps_for(timed_call(20), nop, budget_ms=20.0)
        n_slow, slow_ms = _reps_for(timed_call(400), nop, budget_ms=20.0)
    assert math.isclose(fast_ms, 0.02) and math.isclose(slow_ms, 0.4)
    assert n_fast > n_slow * 3, (n_fast, n_slow)

    # 0b. EVERY reading goes through harness_lib.time_op(detail=True). Cold = the acceptance
    # protocol: ONE launch per sample (inner=1), `iters` samples, read-evict inside time_op.
    hl = _MockHL()
    ms, effort, rec = _time_cell(lambda: 0.25, hl, iters=7, budget_ms=100.0, cold=True, sync=nop)
    assert ms == 0.25 and effort == {"inner": 1, "repeats": 7}, (ms, effort)
    assert hl.calls[-1]["detail"] is True and hl.calls[-1]["inner"] == 1, hl.calls
    assert rec["cache_condition"]["mode"] == "read-evict", rec
    # hot (search-only): a burst of inner>1 launches per sample, HOT_SAMPLES samples
    ms, effort, _r = _time_cell(lambda: 0.0001, hl, iters=99, budget_ms=5.0, cold=False, sync=nop)
    assert effort["repeats"] == HOT_SAMPLES and effort["inner"] > 1, effort
    ms, effort, _r = _time_cell(lambda: 0.5, hl, iters=10, budget_ms=0, cold=False, sync=nop)
    assert effort["inner"] == 10 // HOT_SAMPLES, ("--budget-ms 0 sizes the burst from --iters", effort)
    # 0c. a variant that raises under timing is an error, never a 0 ms reading
    try:
        _time_cell(lambda: None, hl, iters=3, budget_ms=0, cold=True, sync=nop)
        raise AssertionError("time_op None must raise")
    except RuntimeError:
        pass

    # 0d. the REAL instrument resolves from the repo, is GEAK's, and read-evicts; an explicit
    # override that does not exist is a clear error, never a silent fallback.
    saved = os.environ.pop("GEAK_HARNESS_LIB", None)
    try:
        if os.path.isfile(_REPO_HARNESS_LIB):
            real = load_harness_lib()
            assert callable(real.time_op) and real.cache_policy()["mode"] == "read-evict", \
                "harness_lib must be the read-evict instrument"
            os.environ["GEAK_HARNESS_LIB"] = os.path.dirname(_REPO_HARNESS_LIB)
            assert harness_lib_path() == _REPO_HARNESS_LIB, "a directory override names its file"
        else:
            print(f"  -- {_REPO_HARNESS_LIB} absent (tool copied out): real-instrument check skipped")
        os.environ["GEAK_HARNESS_LIB"] = "/nonexistent/geak/harness_lib.py"
        try:
            load_harness_lib()
            raise AssertionError("a missing override must not fall back")
        except HarnessLibUnavailable as e:
            assert "GEAK_HARNESS_LIB" in str(e), e
    finally:
        os.environ.pop("GEAK_HARNESS_LIB", None)
        if saved is not None:
            os.environ["GEAK_HARNESS_LIB"] = saved
    # 0e. no hand-rolled write-evict survives in this file
    src = Path(__file__).read_text()
    assert ("_FLUSH" + "_BUF") not in src and ("def " + "flush_cache") not in src \
        and (".zero" + "_()") not in src, "a hand-rolled write-evict timer is left behind"

    # 1. a delta LARGER than the spread AND the 2% gate is a screen pass; median is the key
    r = summarize({"base": [10.0, 10.1, 10.05], "cand": [8.0, 8.05, 8.02]}, "base")
    assert r["cand"]["resolved"] is True and r["cand"]["screen"] == "pass", r
    assert abs(r["cand"]["vs_baseline"]["median"] - 10.05 / 8.02) < 1e-3, r
    assert "min" in r["cand"]["vs_baseline"] and r["cand"]["min"] == 8.0, r
    # 2. a delta SMALLER than the spread must NOT be sold as a speedup
    r = summarize({"base": [10.0, 11.0], "cand": [9.9, 10.9]}, "base")
    assert r["cand"]["resolved"] is False and r["cand"]["screen"] == "no", r
    assert "NOT RESOLVED" in r["cand"]["note"], r
    # 2b. resolved but under GEAK's 2% screening threshold: still not a screen pass (stricter only)
    r = summarize({"base": [10.0, 10.0, 10.0], "cand": [9.9, 9.9, 9.9]}, "base")
    assert r["cand"]["resolved"] is True and r["cand"]["clears_min_improve"] is False, r
    assert r["cand"]["screen"] == "no" and "screening threshold" in r["cand"]["note"], r
    # 2c. a resolved slowdown says so
    r = summarize({"base": [10.0, 10.0], "cand": [12.0, 12.0]}, "base")
    assert r["cand"]["screen"] == "slower", r
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
    # 5a. ...but only after a FULL cycle, and the incomplete case must be reported
    assert rotation_coverage(["a", "b", "c"], 3)["complete"] is True
    assert rotation_coverage(["a", "b", "c"], 6)["complete"] is True
    inc = rotation_coverage(["a", "b", "c"], 5)
    assert inc["complete"] is False and inc["full_cycles"] == 1, inc
    assert rotation_coverage(["a", "b", "c", "d"], 2)["complete"] is False
    # 5b. The control arm is the SAME callable under a second name, and rides the rotation.
    src_fns = {"cand": lambda: None, "base": lambda: None}
    fns2, names2 = with_control(src_fns, ["cand", "base"], "base")
    assert names2 == ["cand", "base", "__control__base"]
    assert fns2["__control__base"] is src_fns["base"], "the control must BE the twin"
    assert rotation_coverage(names2, 3)["complete"] is True
    # 5c. The band comes off the control, and a candidate delta under it is NOT RESOLVED.
    cm = {"base": [10.00, 10.02, 10.01], "__control__base": [10.00, 10.05, 10.02],
          "cand": [9.96, 9.98, 9.97]}
    band = control_band_pct(cm, "base")
    assert band["band_pct"] >= 0.5, band
    res5 = summarize(cm, "base", band)
    assert res5["cand"]["noise_band_source"] == "control_arm", res5["cand"]
    assert res5["cand"]["resolved"] is False, res5["cand"]
    assert "null" in res5["cand"]["note"]
    assert summarize(cm, "base")["cand"]["noise_band_source"] == "arm_spread"
    assert collision_suspect(res5, "base") is None, "the control faked a collision"
    # 5d. measure() drives harness_lib for every cell, rotated, and keeps its receipts
    hl2 = _MockHL()
    got = measure(None, ["a", "b"], 2, 3, 0, nop, cold=True,
                  fns={"a": lambda: 1.0, "b": lambda: 2.0}, hl=hl2)
    assert got == {"a": [1.0, 1.0], "b": [2.0, 2.0]}, got
    assert len(hl2.calls) == 4 and all(c["detail"] and c["inner"] == 1 for c in hl2.calls)
    assert measure.receipts["a"][0]["timer"] == "mock", measure.receipts
    # 6. THE NaN TRAP: a non-finite metric slips through `v > tol` as a pass. It must not.
    class _N:
        @staticmethod
        def oracle(n):
            return {"max_rel": float("nan") if n == "allnan" else 1e-4}
    passed, rep = gate_correctness(_N, ["ok", "allnan"], 2e-3)
    assert passed == ["ok"], (passed, rep)
    assert "_non_finite_metrics" in rep["allnan"], rep
    assert (float("nan") > 2e-3) is False  # noqa: PLW0177 -- the trap this guards, stated as an assertion
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
    groups, errs = gate_fingerprints(_F, ["anchor", "v_a", "v_b"])
    assert list(groups.values()) == [["v_a", "v_b"]] and not errs, (groups, errs)
    assert gate_fingerprints(object(), ["x", "y"]) == ({}, {}), "no hook -> no claim"
    # 8b. A RAISING hook is an error, never a pass.
    class _Fx:
        @staticmethod
        def fingerprint(n):
            if n == "bad":
                raise RuntimeError("no .amdgcn for this arm")
            return n
    groups, errs = gate_fingerprints(_Fx, ["ok1", "ok2", "bad"])
    assert groups == {}, groups
    assert list(errs) == ["bad"] and "RuntimeError" in errs["bad"], errs
    # 8c. one process per toolchain-patched variant
    class _T:
        @staticmethod
        def toolchain(n):
            return "patch_reinject@abc" if n == "injected" else "stock"
    ids, err = gate_toolchain(_T, ["base", "injected"])
    assert err and "one process per toolchain" in err, (ids, err)
    assert gate_toolchain(_T, ["base", "cand"])[1] is None
    assert gate_toolchain(object(), ["x"]) == ({}, None)
    # 9. a flat set across 3+ arms is escalated to a suspicion, never reported as a negative
    flat = summarize({"base": [4.05, 4.06], "v_a": [4.05, 4.06], "v_b": [4.06, 4.05]}, "base")
    assert collision_suspect(flat, "base") is not None, flat
    sharp = summarize({"base": [2.88, 2.89], "v_a": [3.40, 3.41], "v_b": [4.05, 4.06]}, "base")
    assert collision_suspect(sharp, "base") is None, sharp

    # 10. END TO END through the CLI, no GPU: a stub harness_lib via $GEAK_HARNESS_LIB and a
    # two-arm adapter whose callables return their own "ms".
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "harness_lib.py"), "w") as f:
            f.write("def time_op(call, warmup=10, repeats=50, inner=1, graph=False, *, detail=False):\n"
                    "    ms = call()\n"
                    "    d = {'ms': ms, 'wall_ms': ms, 'timer': 'cuda_event', 'primed': True, "
                    "'host_ms': 0.001, 'cache_condition': {'mode': 'read-evict'}}\n"
                    "    return d if detail else ms\n")
        adapter = os.path.join(td, "bench_mod.py")
        with open(adapter, "w") as f:
            f.write("def variants():\n    return {'base': lambda: 1.0, 'cand': lambda: 0.8}\n"
                    "def oracle(n):\n    return {'max_rel': 1e-4}\n"
                    "def sync():\n    pass\n")
        out_json = os.path.join(td, "ab.json")
        env = dict(os.environ, GEAK_HARNESS_LIB=td)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--module", adapter,
                            "--cells", "3", "--iters", "4", "--warmup", "1", "--control", "auto",
                            "--json", out_json], capture_output=True, text=True, check=False, env=env,
                           timeout=120)
        assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
        payload = json.loads(Path(out_json).read_text())
        assert payload["cache"] == "cold" and payload["instrument"]["timer"].startswith(
            "harness_lib.time_op"), payload
        assert payload["results"]["cand"]["screen"] == "pass", payload["results"]["cand"]
        assert "measure_legs" in payload["role"], payload["role"]
        env["GEAK_HARNESS_LIB"] = os.path.join(td, "missing")
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--module", adapter],
                           capture_output=True, text=True, check=False, env=env, timeout=120)
        assert p.returncode != 0 and "harness_lib" in (p.stderr + p.stdout), (p.stdout, p.stderr)
    print("[ab_bench] SELFTEST PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", help="adapter module (see the contract above)")
    ap.add_argument("--selftest", action="store_true", help="offline checks, no GPU (mock timing)")
    ap.add_argument("--cells", type=int, default=6,
                    help="interleaved cells per variant (default 6; use a multiple of the arm count "
                         "so the rotation completes)")
    ap.add_argument("--iters", type=int, default=20,
                    help="harness_lib samples per cell in --cache cold (default 20); in hot mode with "
                         "--budget-ms 0, the burst length")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--budget-ms", type=float, default=100.0, metavar="MS",
                    help="hot mode only: size each burst from the kernel's own measured duration "
                         "(default 100). 0 sizes it from --iters")
    ap.add_argument("--cache", choices=("cold", "hot"), default="cold",
                    help="cold (default) = the acceptance protocol: harness_lib.time_op, one launch "
                         "per sample, a read-evict flush before every sample. hot = SEARCH-ONLY "
                         "diagnostic: one read-evict then a back-to-back burst per sample. Not a scale "
                         "factor -- it can REVERSE a ranking, and acceptance is always read-evict")
    ap.add_argument("--flush-mb", type=int, default=None,
                    help="size of harness_lib's read-evict buffer (sets HARNESS_CACHE_FLUSH_MB; "
                         "default: harness_lib's own, 512)")
    ap.add_argument("--tol", type=float, default=2e-3, help="oracle tolerance (default 0.002)")
    ap.add_argument("--baseline", help="variant to compare against (default: the first)")
    ap.add_argument("--min-improve-pct", type=float, default=MIN_IMPROVE_PCT,
                    help="screening threshold (default 2.0); pass the workflow value explicitly. "
                         "Final acceptance remains GEAK verify's responsibility")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--control", metavar="NAME|auto",
                    help="add a BYTE-IDENTICAL duplicate of NAME (or of the baseline, with "
                         "'auto') as a first-class arm. The spread across its cells is this "
                         "window's measured noise band, and it rides the same per-cell rotation "
                         "as every real arm -- a control pinned to one slot can only falsify a "
                         "slot-0 bias. Feed the printed band to "
                         "`round_record.py append --noise-band`.")
    ap.add_argument("--preload", type=float, default=0.0, metavar="SEC",
                    help="burn SEC of real compute before the first timed cell. An idle card "
                         "spends its first window on a decaying clock excursion that lands on "
                         "whichever arm holds the early slot (measured: the same empty kernel at "
                         "10.7us in slot 1 and 6.06us in slot 5; 3 s removed it 6/6). Distinct "
                         "from --warmup, which warms the VARIANT rather than the card.")
    ap.add_argument("--permute", action="store_true",
                    help="measure a second time with the arm order REVERSED and report whether "
                         "each number follows the code or the position (the discriminating "
                         "experiment for a suspected cache collision)")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if not a.module:
        ap.error("--module is required (or use --selftest)")

    for name in ("cells", "iters", "flush_mb"):
        value = getattr(a, name)
        if value is not None and value <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("warmup", "budget_ms", "tol", "preload", "min_improve_pct"):
        value = getattr(a, name)
        if not math.isfinite(value) or value < 0:
            ap.error(f"--{name.replace('_', '-')} must be finite and non-negative")

    # A failing rerun must never leave a previous run's actionable JSON in place.
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"schema_version": 2, "validity": {"valid": False,
                       "reasons": ["run incomplete; inspect process exit/log"]}, "results": {}}, f)

    try:
        hl = load_harness_lib()
    except HarnessLibUnavailable as e:
        sys.exit(f"ab_bench: {e}")
    if a.flush_mb is not None:
        os.environ["HARNESS_CACHE_FLUSH_MB"] = str(a.flush_mb)

    mod = _load_module(a.module)
    if not hasattr(mod, "variants"):
        sys.exit(f"{a.module}: no variants() -- see the adapter contract in this file's docstring")
    fns = mod.variants()
    names = list(fns.keys())
    if not names:
        sys.exit("variants() returned nothing")
    sync = getattr(mod, "sync", _default_sync)

    proto = ("cold: harness_lib.time_op, 1 launch/sample, read-evict" if a.cache == "cold" else
             "HOT (search-only): read-evict + burst/sample")
    print(f"=== ab_bench (SCREENING; acceptance = GEAK verify / harness_lib.measure_legs): "
          f"{len(names)} variants, {a.cells} interleaved cells, {proto} ===")
    print(f"    instrument: {harness_lib_path()}")

    collisions, fp_errors = gate_fingerprints(mod, names)
    if fp_errors:
        print("\n  FINGERPRINT GATE DID NOT RUN -- the hook raised for:")
        for n, err in fp_errors.items():
            print(f"    {n:28s} {err}")
        sys.exit("the collision gate could not be evaluated, so it cannot pass. Fix fingerprint() "
                 "(or remove it and rely on --permute) before trusting any ranking from this "
                 "window -- a gate that did not execute is not a gate that found nothing.")
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
            state = "PASS" if r["_verification"] == "pass" else "UNVERIFIED"
            print(f"  oracle {state}  {n:28s} {shown}")
        else:
            why = (r.get("_non_finite_outputs") and f"NON-FINITE outputs {r['_non_finite_outputs']}"
                   or r.get("_non_finite_metrics") and f"NON-FINITE metrics {r['_non_finite_metrics']}"
                   or r.get("_fail") or r.get("_error"))
            print(f"  oracle FAIL  {n:28s} {why}  -> NOT TIMED")
    if not passed:
        sys.exit("every variant failed its oracle; nothing to time")

    toolchains, tc_err = gate_toolchain(mod, passed)
    if tc_err:
        sys.exit(f"TOOLCHAIN GATE: {tc_err}")

    baseline = a.baseline or names[0]
    if baseline not in passed:
        sys.exit(f"baseline {baseline!r} is not among the correctness-passing variants {passed}")

    control_of = None
    if a.control:
        control_of = baseline if a.control == "auto" else a.control
        if control_of not in passed:
            sys.exit(f"--control {control_of!r} is not among the correctness-passing variants "
                     f"{passed}")
        fns, passed = with_control(fns, passed, control_of)
        print(f"  control arm {control_name(control_of)} = a byte-identical duplicate of "
              f"{control_of}; the spread across ITS cells is this window's noise band")

    cov = rotation_coverage(passed, a.cells)
    if not cov["complete"]:
        print(f"\n  ROTATION INCOMPLETE: {a.cells} cells over {cov['variants']} variants. The "
              f"per-cell rotation completes a slot cycle every {cov['variants']} cells, so some "
              f"arms never reach slot 0 and others hold it repeatedly -- a systematic position "
              f"penalty, not weak de-biasing, and invisible in the table below because every arm "
              f"still prints a full row. Use --cells {cov['variants'] * max(1, a.cells // cov['variants'] + 1)} "
              f"(any multiple of {cov['variants']}).")

    try:
        cells_ms = measure(mod, passed, a.cells, a.iters, a.warmup, sync,
                           budget_ms=a.budget_ms, cold=(a.cache == "cold"),
                           fns=fns, preload_s=a.preload, hl=hl)
    except RuntimeError as e:
        sys.exit(f"ab_bench: {e}")
    receipts = dict(getattr(measure, "receipts", {}))
    band = control_band_pct(cells_ms, control_of)
    res = summarize(cells_ms, baseline, band, a.min_improve_pct)
    used = dict(getattr(measure, "iters_used", {}))
    windows = [{"order": list(passed), "control": band, "receipts": receipts,
                "iters_used": used}]
    invalid = window_invalid_reasons(windows[0], cov)
    unverified = [n for n in names if n in passed and report[n]["_verification"] != "pass"]
    if unverified:
        invalid.append(f"correctness unverified: {unverified}")
    if a.cache != "cold":
        invalid.append("hot-cache diagnostic; no screen verdict")
    if invalid:
        invalidate_screens(res, invalid)
    if band:
        print(f"\n  MEASURED NOISE BAND {band['band_pct']:.3f}% "
              f"(control spread {band['spread_pct']:.3f}%, control-vs-twin "
              f"{band['vs_twin_pct']:.3f}%). Pass this to "
              f"`round_record.py append --noise-band {band['band_pct']:.3f}`. It can only make a "
              f"verdict stricter than the {a.min_improve_pct:.1f}% screening threshold, never looser.")
    used = getattr(measure, "iters_used", {})
    if used:
        print("    effort per cell: " + ", ".join(
            f"{n}=inner {k['inner']} x {k['repeats']} samples" for n, k in used.items()))
    print("\nFirst-window diagnostics (final validity follows the permutation check):")
    print(f"\n{'variant':28s} {'median':>9s} {'min':>9s} {'spread%':>8s} {'vs base':>9s} "
          f"{'screen':>7s}")
    for n in passed:
        r = res[n]
        vs = f"{r['vs_baseline']['median']:.4f}x" if n != baseline else "(base)"
        scr = r.get("screen", "-") if n != baseline else "-"
        flag = "" if n == baseline or r.get("resolved", True) else "  <- NOT RESOLVED"
        print(f"{n:28s} {r['median']:9.4f} {r['min']:9.4f} {r['spread_pct']:8.2f} "
              f"{vs:>9s} {scr:>7s}{flag}")
    for n in passed:
        if res[n].get("note"):
            print(f"\n  {n}: {res[n]['note']}")
    if any(res[n].get("screen") == "pass" for n in passed):
        print("\n  screen pass = worth a GEAK verify run (fresh process per leg, measure_legs). It "
              "is NOT an acceptance number and must not be recorded as one.")

    suspect = collision_suspect(res, baseline)
    if suspect:
        print(f"\n  COLLISION SUSPECT: {suspect}")

    permuted = None
    if a.permute:
        print("\n=== --permute: second window, arm order REVERSED ===")
        order = list(reversed(passed))
        try:
            reverse_cells = measure(mod, order, a.cells, a.iters, a.warmup, sync,
                                    budget_ms=a.budget_ms, cold=(a.cache == "cold"),
                                    fns=fns, preload_s=a.preload, hl=hl)
        except RuntimeError as e:
            sys.exit(f"ab_bench: {e}")
        reverse_band = control_band_pct(reverse_cells, control_of)
        permuted = summarize(reverse_cells, baseline, reverse_band, a.min_improve_pct)
        windows.append({"order": order, "control": reverse_band,
                        "receipts": dict(measure.receipts), "iters_used": dict(measure.iters_used)})
        invalid.extend(window_invalid_reasons(windows[-1], rotation_coverage(order, a.cells)))
        for n in passed:
            f_ms, r_ms = res[n]["median"], permuted[n]["median"]
            delta = abs(100.0 * (r_ms - f_ms) / f_ms)
            noise = max(res[n]["spread_pct"] or 0, permuted[n]["spread_pct"] or 0,
                        (band or {}).get("band_pct", 0), (reverse_band or {}).get("band_pct", 0))
            if delta > max(noise, 1.0):
                invalid.append(f"order/window-sensitive timing: {n} ({delta:.2f}%)")
            # A forward pass must also clear the reversed window's own measured band.
            if res[n].get("screen") == "pass" and permuted[n].get("screen") != "pass":
                invalid.append(f"screen result not reproduced in reversed window: {n}")
            print(f"  {n}: forward={f_ms:.6g} ms, reverse={r_ms:.6g} ms")
    if invalid:
        invalidate_screens(res, invalid)
        if permuted:
            invalidate_screens(permuted, invalid)
        print("\nSCREEN INVALID: " + "; ".join(invalid))
    else:
        print("\nSCREEN VALID: final results may be sent to GEAK verify; acceptance remains separate.")

    payload = {"schema_version": 2, "validity": {"valid": not invalid, "reasons": invalid},
               "windows": windows, "role": ("search/screening -- acceptance numbers come from GEAK verify / "
                        "e2e_workflow/scripts/harness_lib.py:measure_legs"),
               "instrument": {"timer": "harness_lib.time_op(detail=True)",
                              "path": harness_lib_path(), "receipts": receipts},
               "baseline": baseline, "cells": a.cells, "iters": a.iters, "tol": a.tol,
               "budget_ms": a.budget_ms, "cache": a.cache,
               "flush_mb": a.flush_mb, "preload_s": a.preload,
               "min_improve_pct": a.min_improve_pct,
               "control": band, "rotation": cov, "toolchain": toolchains,
               "iters_used": used,
               "oracle": report, "results": res,
               "fingerprint_gate": "pass" if getattr(mod, "fingerprint", None) else "absent",
               "collision_suspect": suspect,
               "permuted_results": permuted,
               "excluded_failing_oracle": [n for n in names if n not in passed]}
    if a.json:
        with open(a.json, "w") as f:
            payload = json.loads(json.dumps(payload), parse_constant=str)
            json.dump(payload, f, indent=2, allow_nan=False)
        print(f"\nwrote {a.json}")
    return 2 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
