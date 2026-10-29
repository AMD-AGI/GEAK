#!/usr/bin/env python3
"""The applyback slot's finalize adapter: e2e artifacts + baseline -> A/B ->
`$OUT/result.json`.

This file carries the pipeline's only authority to claim a gain, so its rules are
stricter than elsewhere:

* **Operator speedup and end-to-end delta are stated separately, never conflated.** The
  real failure mode of this scenario is an operator win with no system movement: MoE
  sorting 1.80x -> +0.2356% whole-machine; MoE split4 1.342x -> **-0.0872%**. Putting both
  numbers side by side in result.json makes it impossible to report the first as if it
  were the second.
* **Inside the noise band means "no movement"**, reported honestly as `no_change`. The
  noise band is calibrated by measurement (default 1.045), not the 1.02 in the docs.
* **No conclusion unless the overlay actually shadows.** Otherwise "no gain" may simply
  mean the patch was never loaded.
"""
import argparse, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts"))
from emit_result import emit, read_json  # noqa: E402

THROUGHPUT_KEYS = ("throughput_tok_s_median", "output_token_throughput",
                   "output_throughput_tok_s", "throughput_tok_s", "throughput")


def pick_throughput(d):
    for k in THROUGHPUT_KEYS:
        v = d.get(k)
        if not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            return v, k
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-rc", type=int, default=0)
    ap.add_argument("--e2e-dir", required=True)
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()

    out = os.environ.get("OUT")
    if not out:
        print("run this inside the environment prepared by stage.py prepare",
              file=sys.stderr)
        return 2

    def fail(reason):
        # Reporters read ab.json directly. Preserve an older A/B without presenting it
        # as the result of this failed invocation.
        previous = os.path.join(out, 'ab.json')
        if os.path.lexists(previous):
            os.replace(previous, os.path.join(out, 'ab.before-failed-run.%d.json' % time.time_ns()))
        return emit('fail', notes=reason)

    if args.bench_rc != 0:
        return fail('benchmark command exited %d; partial metrics are not an admitted A/B. '
                    'Preserve the failed run and use a fresh arm identity after repair.' % args.bench_rc)

    applied = read_json(os.path.join(out, "applied.json"))
    if applied.get('all_shadowed') is not True:
        return fail('the overlay did not shadow every target; see applied.json. '
                    'No end-to-end result is admitted.')
    base = read_json(args.baseline)
    if 'measured' in base and base['measured'] is not True:
        return fail('baseline is not a measured receipt; complete the baseline stage before A/B')
    base_tp, base_key = pick_throughput(base)
    if base_tp is None:
        return fail("baseline.json contains no finite positive throughput (tried %s)"
                    % ", ".join(THROUGHPUT_KEYS))

    summ = read_json(os.path.join(args.e2e_dir, "bench_summary.json"))
    cand_tp, cand_key = pick_throughput(summ)
    if cand_tp is None:
        return fail("the overlay e2e run measured no finite positive throughput; "
                    "see bench.log and %s/bench_summary.json" % args.e2e_dir)

    # Do not difference numbers on different bases; the same figure is not comparable
    # across two metric bases.
    basis_a = base.get("metric_basis") or base.get('metric') or base_key
    basis_b = summ.get("metric_basis") or cand_key
    basis_ok = (basis_a == basis_b)
    if not basis_ok:
        return fail('the baseline and candidate have different metric_basis (%s vs %s); '
                    'the two numbers are not comparable.' % (basis_a, basis_b))

    ratio = cand_tp / base_tp
    delta_pct = (ratio - 1.0) * 100.0
    noise = base.get("noise_band") or 1.045
    band_pct = (noise - 1.0) * 100.0

    if abs(delta_pct) <= band_pct:
        verdict = "no_change"
    elif delta_pct > 0:
        verdict = "gain"
    else:
        verdict = "regression"

    ops = [{"kernel": a["kernel"], "op_speedup": a.get("verified_speedup"),
            "target": a.get("target")} for a in applied.get("applied", [])]
    ab = {
        "baseline": {"throughput": base_tp, "metric": base_key, "basis": basis_a,
                     "provenance": base.get("provenance")},
        "candidate": {"throughput": cand_tp, "metric": cand_key, "basis": basis_b,
                      "repeats": summ.get("repeats")},
        "e2e_ratio": round(ratio, 6),
        "e2e_delta_pct": round(delta_pct, 4),
        "noise_band_pct": round(band_pct, 4),
        "verdict": verdict,
        "basis_comparable": basis_ok,
        "overlay_all_shadowed": applied.get("all_shadowed"),
        # Deliberately flattened into one level: how much the operator won and how much
        # the machine moved, both visible at a glance, so neither can pass for the other.
        "op_level": ops,
        "op_speedup_max": max([o["op_speedup"] for o in ops if o["op_speedup"]] or [None]),
        "kernels_applied": len(ops),
        "kernels_skipped": applied.get("skipped", []),
    }
    native_p90 = summ.get("native_interactivity_p90") or {}
    if native_p90.get("available") is True:
        ab["native_interactivity_p90"] = native_p90
    with open(os.path.join(out, "ab.json"), "w") as f:
        json.dump(ab, f, indent=1, ensure_ascii=False)

    notes = ["Operator speedup and end-to-end delta are two different things: %d kernels "
             "were installed, the best operator speedup was %s, and the whole machine "
             "moved %+.4f%% (verdict %s, noise band +/-%.2f%%)."
             % (len(ops), ab["op_speedup_max"], delta_pct, verdict, band_pct)]
    status = "ok"
    if verdict == "no_change":
        notes.append("Inside the noise band means the machine did not move. This is the "
                     "most common outcome in this scenario and must not be dressed up: "
                     "historically MoE sorting measured 1.80x on the microbenchmark for "
                     "+0.2356% whole-machine, and split4 measured 1.342x for -0.0872%.")

    native_metrics = {}
    if native_p90.get("available") is True:
        native_metrics = {
            "native_p90_interactivity_baseline": native_p90["baseline"],
            "native_p90_interactivity_candidate": native_p90["candidate"],
            "native_p90_interactivity_delta_pct": native_p90["delta_pct"],
        }
    return emit(status,
                artifacts={"ab": "ab.json", "applied": "applied.json",
                           "overlay": "overlay"},
                metrics={"baseline_tok_s": base_tp, "candidate_tok_s": cand_tp,
                         "e2e_delta_pct": ab["e2e_delta_pct"],
                         "op_speedup_max": ab["op_speedup_max"],
                         "kernels_applied": len(ops), "verdict": verdict, **native_metrics},
                notes=" ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
