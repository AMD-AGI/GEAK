#!/usr/bin/env python3
"""Same-window INTERLEAVED A/B of N variants inside ONE process -- a SEARCH / SCREENING instrument.

A GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/; the Gluon pack keeps a shim at
scripts/ab_bench.py).

What this is, and what it is not. GEAK owns acceptance: a candidate is accepted on GEAK verify /
`e2e_workflow/scripts/harness_lib.py:measure_legs` numbers (fresh process per leg, interleaved
baseline/candidate pairs, median) against the commit gate MIN_IMPROVE=2%. This tool ranks arms
WHILE SEARCHING, so its per-sample timing is the SAME instrument as acceptance --
`harness_lib.time_op(detail=True)`: CUDA-event device time, a sync per sample, a read-evict
cache flush before every sample (never a `zero_()` write-evict), median -- and everything it adds
on top (the control arm, the measured noise band, --permute, the fingerprint gate) can only make
a verdict STRICTER, never looser. A "screen pass" here means "worth sending to GEAK verify", not
"accepted". harness_lib resolves from $GEAK_HARNESS_LIB (file or dir), else
<repo>/e2e_workflow/scripts/harness_lib.py; if neither exists the tool exits with a clear message
rather than falling back to a hand-rolled timer.

ONE PROCESS PER TOOLCHAIN-PATCHED VARIANT. Arms that need a different compiler (a patched
Triton, an injected pass, gluon_swp / patch_reinject / patch_async_reinject) cannot share a
process -- the patch is process-global. Give the adapter a `toolchain(name) -> str` hook; when
the passing arms report more than one toolchain identity this tool refuses, and you run one
window per toolchain and compare them through GEAK verify instead.

Why interleaving (the Gluon pack's references/method/benchmark-hygiene.md): on a box whose memory clock is not
pinnable, two numbers produced by two processes are NOT comparable -- a 3% "win" is
indistinguishable from a clock excursion. The construction that makes a small delta mean anything
while SEARCHING is to run every variant inside ONE process, interleaved cell by cell, with the
variant ORDER ROTATED per cell so no variant always occupies the same position in the clock
trajectory.

Correctness is gated BEFORE timing, in the same process. A variant whose oracle fails is never
timed at all: a faster-and-wrong candidate that gets pinned poisons the comparator and
everything transcribed from it downstream.

The tool also refuses to let you over-read the result: if a pairwise delta is smaller than the
measured cell-to-cell spread, it is reported as NOT RESOLVED rather than as a speedup.

`--control` makes that floor MEASURED instead of inferred, and it is the single cheapest thing
here. It adds a byte-identical duplicate of one arm as a first-class variant, so the window reads
the same code twice and the difference is exactly what the box did in between. Cost: one more arm.
Why it matters: over one 1208-round campaign the MEDIAN kept round was +0.833%, 31.6% of rounds
marked a win recorded a NON-POSITIVE delta, and p25 was exactly 0.000% -- when the median win sits
inside the band, ranking arms by end-to-end time is arithmetic on numbers that do not order. The
technique was invented in the field and criticised by its own author in the same campaign --
*"all eight such controls ever run put ONE inert arm in ONE slot"* -- so here the control rides the
same per-cell rotation as every real arm, and `rotation_coverage()` says whether that rotation
actually completed.

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
                                          # (used for --preload and the hot-mode burst sizing;
                                          # harness_lib.time_op does its own per-sample sync)

    def toolchain(name: str) -> str:      # OPTIONAL; REQUIRED when any arm patches the compiler
        '''name -> identity of the compiler stack the arm needs ("stock", "patch_reinject@<sha>").
        More than one identity among the passing arms -> refused: one process per toolchain.'''

Reading the result. `median` (the median over cells of each cell's harness_lib median) is the
headline and the ranking key; `min` is reported beside it as an extra field only. A pairwise
delta is a SCREEN PASS only when it clears BOTH the GEAK commit gate (--min-improve-pct, default
2.0) AND this window's noise floor (the measured control band when --control is given, else the
per-arm cell spread). The noise floor can only raise the bar above 2%, never lower it.

Usage:
    python3 ab_bench.py --module bench_mod.py [--cells 6] [--iters 20] [--warmup 10]
                        [--cache cold|hot] [--tol 0.002] [--baseline NAME] [--json ab.json]
                        [--permute] [--control auto|NAME] [--preload 3] [--min-improve-pct 2]
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
        return ("FLAT RESULT SET across %d variants -- treat as a Triton cache-collision "
                "suspect until the arm ORDER has been permuted. Re-run with --permute (or "
                "reverse the arm order) and check whether each number follows the CODE or the "
                "POSITION. Do not record 'the levers do not move the clock' from this window."
                % (len(others) + 1))
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
            receipts[n] = {k: receipt.get(k) for k in ("timer", "primed", "host_ms",
                                                       "cache_condition", "wall_ms")}
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
            "spread_pct": round(spread, 3), "vs_twin_pct": round(twin, 3),
            "band_pct": round(max(spread, twin), 3)}


MIN_IMPROVE_PCT = 2.0   # GEAK's commit gate; a measured band may only RAISE this bar


def summarize(cell_ms, baseline, band=None, min_improve_pct=MIN_IMPROVE_PCT):
    """Median is the headline and the ranking key; min rides along as an extra field.

    Verdicts per non-baseline arm:
      resolved          delta exceeds this window's noise floor (control band, else arm spread)
      clears_min_improve  median speedup >= 1 + min_improve_pct/100 (GEAK's commit gate)
      screen            'pass' only when BOTH hold -- the band can only make the bar stricter.
                        'pass' means "send to GEAK verify", never "accepted"."""
    out = {}
    for n, cells in cell_ms.items():
        lo, hi = min(cells), max(cells)
        out[n] = {
            "cells_ms": [round(v, 4) for v in cells],
            "median": round(statistics.median(cells), 4),
            "min": round(lo, 4),                       # extra field; never the ranking key
            # spread is the resolution floor of this window: a delta below it is not a result
            "spread_pct": round(100.0 * (hi - lo) / lo, 2) if lo else None,
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
                         f"commit gate -- not a screen pass (the band never lowers that bar).")
    return out


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
    nop = lambda: None                                                        # noqa: E731

    # 0a. ADAPTIVE BUDGET (hot burst sizing): the count must come from the callable's own duration.
    def _sleep(us):
        return lambda: time.sleep(us / 1e6)
    n_fast, _e = _reps_for(_sleep(20), nop, budget_ms=20.0)
    n_slow, _e = _reps_for(_sleep(400), nop, budget_ms=20.0)
    assert n_fast > n_slow * 3, ("a 20us callable must get many more iterations than a 400us one at "
                                 "the same budget", n_fast, n_slow)

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
    src = open(os.path.abspath(__file__)).read()
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
    # 2b. resolved but under GEAK's 2% commit gate: still not a screen pass (stricter only)
    r = summarize({"base": [10.0, 10.0, 10.0], "cand": [9.9, 9.9, 9.9]}, "base")
    assert r["cand"]["resolved"] is True and r["cand"]["clears_min_improve"] is False, r
    assert r["cand"]["screen"] == "no" and "commit gate" in r["cand"]["note"], r
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
    assert measure.receipts["a"]["timer"] == "mock", measure.receipts
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
                    "    d = {'ms': ms, 'wall_ms': ms, 'timer': 'stub', 'primed': True, "
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
                            "--json", out_json], capture_output=True, text=True, env=env,
                           timeout=120)
        assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
        payload = json.load(open(out_json))
        assert payload["cache"] == "cold" and payload["instrument"]["timer"].startswith(
            "harness_lib.time_op"), payload
        assert payload["results"]["cand"]["screen"] == "pass", payload["results"]["cand"]
        assert "measure_legs" in payload["role"], payload["role"]
        env["GEAK_HARNESS_LIB"] = os.path.join(td, "missing")
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--module", adapter],
                           capture_output=True, text=True, env=env, timeout=120)
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
                    help="GEAK commit gate a screen pass must clear (default 2.0). Lowering it does "
                         "not lower GEAK's gate; the measured band can only raise the bar")
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

    try:
        hl = load_harness_lib()
    except HarnessLibUnavailable as e:
        sys.exit(f"ab_bench: {e}")
    if a.flush_mb is not None:
        os.environ["HARNESS_CACHE_FLUSH_MB"] = str(a.flush_mb)

    mod = _load_module(a.module)
    if not hasattr(mod, "variants"):
        sys.exit(f"{a.module}: no variants() -- see the adapter contract in this file's docstring")
    names = list(mod.variants().keys())
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
            print(f"  oracle PASS  {n:28s} {shown or '(no oracle -- results are UNGATED)'}")
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

    baseline = a.baseline or passed[0]
    if baseline not in passed:
        sys.exit(f"baseline {baseline!r} is not among the correctness-passing variants {passed}")

    fns, control_of = mod.variants(), None
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
    for n, rc in receipts.items():
        if rc.get("timer") not in ("cuda_event", "cuda_event_graph"):
            print(f"  WARNING {n}: timer={rc.get('timer')} -- not CUDA-event device time; these "
                  f"numbers do not screen anything on a GPU box")
        elif rc.get("primed") is False:
            print(f"  WARNING {n}: primed=False (host {rc.get('host_ms')} ms/launch) -- dispatch is "
                  f"slower than the kernel, so this reading is HOST-bound, not a kernel time")
    if band:
        print(f"\n  MEASURED NOISE BAND {band['band_pct']:.3f}% "
              f"(control spread {band['spread_pct']:.3f}%, control-vs-twin "
              f"{band['vs_twin_pct']:.3f}%). Pass this to "
              f"`round_record.py append --noise-band {band['band_pct']:.3f}`. It can only make a "
              f"verdict stricter than the {a.min_improve_pct:.1f}% commit gate, never looser.")
    used = getattr(measure, "iters_used", {})
    if used:
        print("    effort per cell: " + ", ".join(
            f"{n}=inner {k['inner']} x {k['repeats']} samples" for n, k in used.items()))
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
        permuted = summarize(
            measure(mod, list(reversed(passed)), a.cells, a.iters, a.warmup, sync,
                    budget_ms=a.budget_ms, cold=(a.cache == "cold"),
                    fns=fns, preload_s=a.preload, hl=hl),
            baseline, band, a.min_improve_pct)
        print(f"{'variant':28s} {'fwd median':>11s} {'rev median':>11s} {'delta%':>8s}")
        followed_position = []
        for n in passed:
            f_ms, r_ms = res[n]["median"], permuted[n]["median"]
            d = 100.0 * (r_ms - f_ms) / f_ms if f_ms else 0.0
            noise = max(res[n]["spread_pct"] or 0, permuted[n]["spread_pct"] or 0)
            if abs(d) > max(noise, 1.0):
                followed_position.append(n)
            print(f"{n:28s} {f_ms:11.4f} {r_ms:11.4f} {d:8.2f}")
        if followed_position:
            print(f"\n  THE NUMBERS FOLLOWED THE POSITION, NOT THE CODE, for {followed_position}."
                  f"\n  That is a cache collision (or an un-cancelled clock trajectory), and NO "
                  f"ranking from either window is usable. Give each arm its own kernel object "
                  f"and its own TRITON_CACHE_DIR before re-measuring.")
        else:
            print("\n  Every number followed the CODE across the order flip -- the ranking is "
                  "order-independent and the result stands.")

    payload = {"role": ("search/screening -- acceptance numbers come from GEAK verify / "
                        "e2e_workflow/scripts/harness_lib.py:measure_legs"),
               "instrument": {"timer": "harness_lib.time_op(detail=True)",
                              "path": harness_lib_path(), "receipts": receipts},
               "baseline": baseline, "cells": a.cells, "iters": a.iters, "tol": a.tol,
               "budget_ms": a.budget_ms, "cache": a.cache,
               "flush_mb": a.flush_mb, "preload_s": a.preload,
               "min_improve_pct": a.min_improve_pct,
               "control": band, "rotation": cov, "toolchain": toolchains,
               "iters_used": getattr(measure, "iters_used", {}),
               "oracle": report, "results": res,
               "fingerprint_gate": "pass" if getattr(mod, "fingerprint", None) else "absent",
               "collision_suspect": suspect,
               "permuted_results": permuted,
               "excluded_failing_oracle": [n for n in names if n not in passed]}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
