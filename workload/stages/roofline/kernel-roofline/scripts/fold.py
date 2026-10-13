#!/usr/bin/env python3
"""The shared fold: per-shape rooflines -> one number per (kernel, regime).

Every mode produces rooflines **per shape**. `tl-only` predicts one per shape from
the shape itself; `rocprof` measures one per shape, one profiled command at a time.
A report wants one number per kernel per regime, and that collapse has exactly one
correct arithmetic -- see `references/folding.md`, which this module implements.

It lives in its own file for one reason: "both legs are folded the same way" is the
precondition for comparing them at all, and a rule stated in prose and implemented
twice is a rule that will drift. `join_legs.py` and `compare_legs.py` both import
from here, and a single-engine leg can get the identical rollup from the CLI below
without having a second leg to join:

    python3 fold.py --rows results/tl-only/rows_candidate.json --label TL

The three primitives, in the order a leg uses them:

  fold_kernels   within one case: per-kernel efficiencies -> that case's roof_us
  fold_cases     across shapes:   per-case roofs -> one pct per (kernel, regime)
  fold_ratio     across shapes:   any ratio (AI, bound share) folded as sums

Engine-agnostic by construction: nothing here parses a profiler, knows a peak, or
cares which tool produced `eff`. That is the point -- the arithmetic cannot differ
between legs because there is only one copy of it.
"""
import argparse
import json
import sys
from pathlib import Path


def fold_kernels(per_kernel, eff_of, measured_us, missing="drop"):
    """One case's roof, folded over the kernels it launched.

        roof_us = measured_us * ( sum_k t_k*Eff_k / sum_k t_k )

    `per_kernel` is a list of dicts carrying at least `ms` (device ms, from a
    timeline that reflects deployment -- see folding.md). `eff_of(pk)` returns that
    kernel's efficiency in (0,1], or None when the engine could not score it.

    Folding TIME is what makes this axis-safe: each kernel's own `attainable`
    (min of its compute peak and AI*BW) is already inside its `Eff`, so a seam whose
    kernels straddle the ridge never ends up with two peaks over one denominator.

    `missing` decides what an unscorable kernel does to the result, and the two
    choices are NOT cosmetic:

      "drop"  -- out of numerator AND denominator; the case is scored on its covered
                 time and the shortfall is reported as coverage. Neutral.
      "zero"  -- out of the numerator, still in the denominator; i.e. scored as 0%
                 efficient. This DEFLATES the roof and therefore INFLATES the
                 reported headroom, by exactly (1 - coverage).

    folding.md prescribes "drop", and both legs of `join_legs.py` now use it. "zero"
    is kept only to reproduce empirical-leg tables published before 2026-10-12, when
    that leg scored unscorable kernels 0 instead. Either way the returned `coverage`
    is what makes the choice visible, so publish it.

    -> (roof_us, coverage, unscored) where `unscored` is [(name, ms), ...].
    """
    if missing not in ("drop", "zero"):
        raise ValueError(f"missing={missing!r} must be 'drop' or 'zero'")
    num = den = total = 0.0
    unscored = []
    for pk in per_kernel:
        t = pk["ms"]
        total += t
        eff = eff_of(pk)
        if eff is None:
            unscored.append((pk.get("kernel", "?")[:50], t))
            if missing == "zero":
                den += t          # counted as 0% efficient
            continue
        num += t * eff
        den += t
    if not total or not den:
        return None, 0.0, unscored
    coverage = 1.0 - sum(t for _, t in unscored) / total
    return (num / den) * measured_us, coverage, unscored


def _weight(row, key):
    """`share_of_op_calls` if the leg supplied one, else 1.0 (every shape equal).

    A weight is how often that shape is actually dispatched in deployment, so an
    ABSENT weight means "this leg has no deployment distribution" -- the honest
    fallback is to weight the shapes equally, not to drop them. An EXPLICIT 0 is a
    different statement ("never dispatched") and is honoured as such. The
    distinction matters: `or 1.0` collapses the two and turns a deliberate 0 into a
    full vote.
    """
    w = row.get(key)
    return 1.0 if w is None else float(w)


def fold_cases(rows, roof_key="roof_us", meas_key="measured_us", weight_key="w",
               cov_key="coverage"):
    """Across shapes: sum numerator and denominator separately, divide once.

        pct(kernel, regime) = sum_c w_c*roof_us_c / sum_c w_c*measured_us_c

    NEVER mean(pct_c). A percentage is a ratio, and averaging ratios gives a shape
    carrying 0.3% of the calls the same vote as the one holding 90% of the clock.

    A row whose `roof_key` is None is excluded from BOTH sums -- a missing metric is
    not a zero -- and counted in `n_unscored`. Weights are each case's own
    `share_of_op_calls`; they are normalised WITHIN a regime because `(kernel,
    regime)` is the grouping key, so decode and prefill never blend.

    -> {(kernel, regime): {w_meas, w_roof, pct, speedup, n, n_unscored, coverage}}
    """
    agg = {}
    for r in rows:
        k = (r["kernel"], r["regime"])
        a = agg.setdefault(k, {"w_meas": 0.0, "w_roof": 0.0, "n": 0,
                               "n_unscored": 0, "cov": []})
        roof = r.get(roof_key)
        if roof is None:
            a["n_unscored"] += 1
            continue
        w = _weight(r, weight_key)
        a["w_meas"] += w * r[meas_key]
        a["w_roof"] += w * roof
        a["n"] += 1
        if r.get(cov_key) is not None:
            a["cov"].append(r[cov_key])
    for a in agg.values():
        a["pct"] = a["w_roof"] / a["w_meas"] if a["w_meas"] else None
        # An efficiency above 1 is a disproof of the denominator, not a measurement:
        # report it, publish no speedup from it, and never clamp (a clamped 100%
        # reads as "saturated", the opposite of what the row says).
        a["suspect"] = a["pct"] is not None and a["pct"] > 1.0
        a["speedup"] = (1.0 / a["pct"]) if a["pct"] and not a["suspect"] else None
        # The weakest case governs: a seam is only as covered as its worst shape.
        a["coverage"] = min(a["cov"]) if a["cov"] else None
    return agg


def fold_ratio(rows, num_key, den_key, weight_key="w"):
    """Any ratio across shapes: `sum(w*num) / sum(w*den)`, never a mean of ratios.

    This is how AI folds (`sum(w*FLOPs)/sum(w*bytes)`) and how a bound "share"
    folds (`sum(w*compute_time)/sum(w*time)`). Same reason as `fold_cases`: AI is a
    ratio, and a tiny shape's AI must not outvote the shape that moves the clock.

    -> {(kernel, regime): {num, den, ratio, n}}
    """
    agg = {}
    for r in rows:
        k = (r["kernel"], r["regime"])
        a = agg.setdefault(k, {"num": 0.0, "den": 0.0, "n": 0})
        w = _weight(r, weight_key)
        a["num"] += w * r[num_key]
        a["den"] += w * r[den_key]
        a["n"] += 1
    for a in agg.values():
        a["ratio"] = a["num"] / a["den"] if a["den"] else None
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, type=Path,
                    help="JSON with a `rows` list (a TL leg's rows.json, or any "
                         "per-case table). Each row needs kernel, regime, case, "
                         "measured_us, the roof column, and optionally case_weight.")
    ap.add_argument("--roof-key", default="roofline_us",
                    help="Per-case roof column to fold (default: roofline_us, which is "
                         "what a TraceLens leg writes). A rocprof-only leg folds its "
                         "own per-case roof under whatever name it wrote.")
    ap.add_argument("--label", default="roof",
                    help="Column header for the folded percentage.")
    ap.add_argument("--out", default=None, type=Path, help="Optional JSON destination.")
    args = ap.parse_args()

    d = json.loads(args.rows.read_text())
    rows = [{**r, "w": r.get("case_weight")} for r in d["rows"]]
    missing_roof = [r["case"] for r in rows if r.get(args.roof_key) is None]
    agg = fold_cases(rows, roof_key=args.roof_key)

    print(f"[{args.rows.name}] folded per folding.md: "
          f"sum(w*{args.roof_key}) / sum(w*measured_us), weights within regime\n")
    print(f"{'kernel/regime':<40}{args.label + '%':>9}{'speedup':>9}{'n':>4}{'cov':>6}")
    for k in sorted(agg):
        a = agg[k]
        pct = f"{a['pct']:.1%}" if a["pct"] is not None else "--"
        if a["suspect"]:
            # Published as suspect, with no speedup beside it.
            sp = "suspect"
        else:
            sp = f"{a['speedup']:.2f}x" if a["speedup"] else "--"
        cov = f"{a['coverage']:.0%}" if a["coverage"] is not None else "--"
        print(f"{k[0] + '/' + k[1]:<40}{pct:>9}{sp:>9}{a['n']:>4}{cov:>6}")
        if a["n_unscored"]:
            print(f"{'':<40}[{a['n_unscored']} case(s) had no {args.roof_key} and are "
                  f"excluded from both sums -- a missing metric is not a zero]")
    if missing_roof:
        print(f"\n[{len(missing_roof)} of {len(rows)} rows carried no '{args.roof_key}']")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({f"{k[0]}/{k[1]}": v for k, v in agg.items()},
                  open(args.out, "w"), indent=1)
        print(f"\nfolded -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
