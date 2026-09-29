#!/usr/bin/env python3
"""Decide an apply-back serving A/B after each interleaved ref/cand pair.

Apply-back runs ref and cand legs alternately (R1 C1 R2 C2 ...). Every leg costs a
server launch plus warm-up plus timed rounds, so the A/B stops as soon as the answer
is clear instead of always spending a fixed number of pairs.

After n complete pairs (n = min(len(ref), len(cand))):

  n < min_pairs                                   -> continue
  cand not faster (delta <= 0 or max(cand) < min(ref))
                                                  -> reject
  non-overlapping AND delta > noise band AND delta >= clear_margin x spread
                                                  -> accept (clear win, stop early)
  n >= max_pairs: non-overlapping AND delta > noise band -> accept, else reject
  otherwise                                       -> continue

delta   = (median(cand) / median(ref) - 1) * 100
spread  = the larger within-side range, as % of that side's median (floored at
          --spread-floor-pct so two identical legs do not make any delta "clear")

Prints one JSON object and a final line ``AB_DECISION=<accept|reject|continue>``.
"""
import argparse
import json
import statistics
import sys


def _spread_pct(values):
    med = statistics.median(values)
    return (max(values) - min(values)) / med * 100.0 if med > 0 else float("inf")


def decide(ref, cand, noise_band_pct=0.5, min_pairs=2, max_pairs=3,
           clear_margin=10.0, spread_floor_pct=0.05):
    ref = [float(v) for v in ref]
    cand = [float(v) for v in cand]
    pairs = min(len(ref), len(cand))
    out = {"pairs": pairs, "noise_band_pct": noise_band_pct,
           "min_pairs": min_pairs, "max_pairs": max_pairs}
    if pairs < max(1, min_pairs):
        out.update(decision="continue", reason=f"{pairs} pair(s) < min_pairs {min_pairs}")
        return out
    if any(v <= 0 for v in ref + cand):
        out.update(decision="reject", reason="a leg reported a non-positive throughput")
        return out
    ref_med = statistics.median(ref)
    cand_med = statistics.median(cand)
    delta = (cand_med / ref_med - 1.0) * 100.0
    spread = max(_spread_pct(ref), _spread_pct(cand), spread_floor_pct)
    nonoverlap = min(cand) > max(ref)
    out.update(ref_median=round(ref_med, 4), cand_median=round(cand_med, 4),
               delta_pct=round(delta, 4), spread_pct=round(spread, 4),
               nonoverlap=nonoverlap)
    passes = nonoverlap and delta > noise_band_pct
    if delta <= 0 or max(cand) < min(ref):
        out.update(decision="reject", reason="candidate is not faster than the reference")
    elif passes and delta >= clear_margin * spread:
        out.update(decision="accept",
                   reason=f"clear win: delta {delta:.3f}% >= {clear_margin:g} x spread {spread:.3f}%")
    elif pairs >= max_pairs:
        if passes:
            out.update(decision="accept",
                       reason=f"non-overlapping and delta {delta:.3f}% > noise band {noise_band_pct}%")
        else:
            out.update(decision="reject",
                       reason=(f"after {pairs} pairs: nonoverlap={nonoverlap}, "
                               f"delta {delta:.3f}% vs noise band {noise_band_pct}%"))
    else:
        out.update(decision="continue", reason="not yet clear; run another pair")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", type=float, nargs="+", required=True, help="ref-leg tok/s, in run order")
    ap.add_argument("--cand", type=float, nargs="+", required=True, help="cand-leg tok/s, in run order")
    ap.add_argument("--noise-band-pct", type=float, default=0.5)
    ap.add_argument("--min-pairs", type=int, default=2)
    ap.add_argument("--max-pairs", type=int, default=3)
    ap.add_argument("--clear-margin", type=float, default=10.0)
    ap.add_argument("--spread-floor-pct", type=float, default=0.05)
    ap.add_argument("--out", default="", help="also write the JSON here")
    a = ap.parse_args(argv)
    out = decide(a.ref, a.cand, a.noise_band_pct, a.min_pairs, a.max_pairs,
                 a.clear_margin, a.spread_floor_pct)
    text = json.dumps(out, sort_keys=True)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    print(f"AB_DECISION={out['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
