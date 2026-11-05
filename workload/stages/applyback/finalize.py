#!/usr/bin/env python3
"""The applyback slot's verdict: the only place in the pipeline that may claim a gain.

It belongs to the slot, not to any implementation (`stages/CONTRACT.md` 3.1), and the
driver runs it after every session. It checks evidence; it never re-measures anything.

Four rules it enforces, each one written here and nowhere else:

* **Operator speedup and end-to-end delta are stated separately, never conflated.** The
  real failure mode of this scenario is an operator win with no system movement: MoE
  sorting 1.80x -> +0.2356% whole-machine; MoE split4 1.342x -> **-0.0872%**. Both numbers
  land side by side in `ab.json` so neither can pass for the other.
* **Inside the noise band means the machine did not move**, reported honestly as
  `no_change`. The band is calibrated by measurement (default 1.045), not quoted from
  documentation.
* **No conclusion unless the overlay is shown to have shadowed its targets.** Without it,
  "no gain" and "the patch was never loaded" are the same result. Once, a
  `candidate_bind.file` pointed at the wrong file and the overlay shadowed a dispatcher
  with an identical dispatcher while the image's prebuilt .so kept serving every token:
  that A/B measured the baseline against itself, and the shadow check was green. So the
  source hashes are checked too, against what optimize recorded -- in one place, which is
  what the five scattered copies of this check used to be.
* **Two numbers on different bases are not comparable.** A metric-basis mismatch is a
  `fail`, not a footnote.

Whatever native-validity fields a particular harness exposes are scenario config
(`params.native_validation`), read here as paths and required values. No harness's field
names live in this file.

    python3 stages/applyback/finalize.py      # inside the environment run.py prepares
"""
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_shared"))
from emit_result import emit, read_json  # noqa: E402

THROUGHPUT_KEYS = ("throughput_tok_s_median", "output_token_throughput",
                   "output_throughput_tok_s", "throughput_tok_s", "throughput")


def pick_throughput(d):
    for k in THROUGHPUT_KEYS:
        v = d.get(k)
        if not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            return v, k
    return None, None


def at(doc, path):
    """`a.b.c` into nested dicts and lists; None when any step is absent."""
    for part in str(path).split("."):
        if isinstance(doc, list):
            try:
                doc = doc[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(doc, dict) or part not in doc:
            return None
        doc = doc[part]
    return doc


def check_required(doc, required, where):
    """Every configured path must hold exactly the configured value."""
    for path, want in (required or {}).items():
        got = at(doc, path)
        if got != want:
            return "%s: %s is %r, the scenario requires %r" % (where, path, got, want)
    return None


def check_identity(out, run_root):
    """One hash check: what the overlay carries is what cold verification accepted.

    `applied.json` records each installed kernel with the hashes the overlay actually
    holds. `optimized.json` records what two cold runs measured. Equal or the stage is a
    fail -- a kernel whose bytes drifted between the two has no evidence behind it.
    """
    applied = read_json(os.path.join(out, "applied.json"))
    rows = applied.get("applied")
    if not isinstance(rows, list):
        return applied, "applied.json records no installed kernels"
    optimized = read_json(os.path.join(run_root, "optimize", "out", "optimized.json"))
    verified = {r.get("kernel"): r for r in (optimized.get("kernels") or [])
                if isinstance(r, dict)}
    for row in rows:
        name = row.get("kernel")
        source = verified.get(name)
        if not source:
            return applied, ("%s was installed but optimize never reported it" % name)
        expected = source.get("kernel_src_hashes")
        if not isinstance(expected, dict) or not expected:
            return applied, ("%s has no kernel_src_hashes from cold verification; recover "
                             "the recorded source identity rather than hashing current "
                             "work in progress" % name)
        got = row.get("kernel_src_hashes")
        if got != expected:
            return applied, ("%s: the overlay's source does not match the cold-verified "
                             "kernel_src_hashes; the bytes that were measured are not the "
                             "bytes that were installed" % name)
        meta = source.get("task_meta_sha256")
        if meta and row.get("task_meta_sha256") not in (None, meta):
            return applied, ("%s: task metadata differs from the cold-verified "
                             "task_meta_sha256, so the binding may point elsewhere" % name)
    return applied, None


def main():
    out = os.environ.get("OUT")
    if not out:
        print("run this inside the environment run.py prepares", file=sys.stderr)
        return 2
    run_root = os.environ.get("RUN_ROOT") or os.path.dirname(os.path.dirname(out))
    params = read_json(os.environ["PARAMS"]) if os.environ.get("PARAMS") else {}

    # A multi-config baseline (adopt-ci-artifacts) names representative points; each was
    # measured in $OUT/points/<pid>/ and is judged here against its own baseline, then folded
    # (aggregate_points.py): a gain at one concurrency cannot hide a regression at another.
    top = read_json(os.path.join(run_root, "baseline", "out", "baseline.json"))
    if top.get("multi_config") and top.get("representative") \
            and not os.environ.get("APPLYBACK_POINT"):
        pids = list(top["representative"])
        for pid in pids:
            pout = os.path.join(out, "points", pid)
            os.makedirs(pout, exist_ok=True)
            env = dict(os.environ, OUT=pout, RUN_ROOT=run_root, APPLYBACK_POINT=pid,
                       APPLYBACK_BASELINE=os.path.join(run_root, "baseline", "out", "points",
                                                       pid, "baseline.json"))
            subprocess.call([sys.executable, os.path.abspath(__file__)], env=env)
        sys.argv = [sys.argv[0]] + pids
        import aggregate_points
        return aggregate_points.main()

    def fail(reason, status="fail"):
        # Reporters read ab.json directly. Preserve an older A/B without presenting it as
        # the result of this invocation.
        previous = os.path.join(out, "ab.json")
        if os.path.lexists(previous):
            os.replace(previous,
                       os.path.join(out, "ab.before-failed-run.%d.json" % time.time_ns()))
        return emit(status, notes=reason)

    # --- The overlay: was anything installed, did it shadow, is it what was verified.
    applied, problem = check_identity(out, run_root)
    if not applied:
        return fail("there is no applied.json, so nothing records what was installed",
                    status="blocked")
    if not applied.get("applied"):
        return fail("there is no kernel to apply back: the optimize stage produced nothing "
                    "that passed cold re-verification. Every stage ran; the conclusion is "
                    "that no usable optimization has been found yet.", status="blocked")
    if problem:
        return fail(problem)

    shadow = read_json(os.path.join(out, "shadow_check.json"))
    shadowed = shadow.get("all_shadowed")
    if shadowed is None:
        shadowed = applied.get("all_shadowed")
    if shadowed is not True:
        return fail("the overlay did not shadow every target (shadow_check.json). Without "
                    "that proof, 'no gain' and 'the patch was never loaded' are the same "
                    "result, so no end-to-end number is admitted.")

    proof_cfg = params.get("serving_source_proof") or {}
    if proof_cfg:
        proof = read_json(os.path.join(out, proof_cfg.get("file") or
                                       "serving_source_proof.json"))
        ranks = proof.get("ranks")
        want = proof_cfg.get("world_size")
        have = len(ranks) if isinstance(ranks, (list, dict)) else 0
        if want and have != want:
            return fail("serving source proof covers %d of %d ranks; a local callable "
                        "check is preflight, not evidence of what the serving workers "
                        "loaded" % (have, want))

    # --- The measurement.
    summary = read_json(os.path.join(out, "e2e", "bench_summary.json"))
    if not summary:
        return fail("the stage left no e2e/bench_summary.json, so nothing records what "
                    "the candidate measured", status="blocked")
    rc = summary.get("rc")
    if isinstance(rc, int) and rc != 0:
        return fail("the benchmark command exited %d; partial metrics are not an admitted "
                    "A/B. Preserve the failed run and use a fresh arm identity after "
                    "repair." % rc)
    native = params.get("native_validation") or {}
    if native:
        doc = read_json(os.path.join(out, "e2e", native.get("file") or ""))
        if not doc:
            return fail("the scenario configures native_validation but %s is absent; a "
                        "missing validity receipt blocks the arm"
                        % (native.get("file") or "its file"), status="blocked")
        bad = check_required(doc, native.get("required_values"), "native validation")
        if bad:
            return fail(bad)

    historical = read_json(os.environ.get("APPLYBACK_BASELINE")
                           or os.path.join(run_root, "baseline", "out", "baseline.json"))
    if "measured" in historical and historical["measured"] is not True:
        return fail("the baseline is not a measured receipt; complete the baseline stage "
                    "before an A/B", status="blocked")
    base = summary.get("stock_control")
    if not isinstance(base, dict) or base.get("admissible") is not True:
        return fail("bench_summary.json has no admissible fresh stock_control; measure "
                    "the unchanged stack in this A/B before comparing the candidate",
                    status="blocked")
    base_tp, base_key = pick_throughput(base)
    if base_tp is None:
        return fail("stock_control contains no finite positive throughput (tried %s)"
                    % ", ".join(THROUGHPUT_KEYS))
    cand_tp, cand_key = pick_throughput(summary)
    if cand_tp is None:
        return fail("the overlay e2e run measured no finite positive throughput; see "
                    "e2e/bench_summary.json")

    basis_a = base.get("metric_basis") or base.get("metric") or base_key
    basis_b = summary.get("metric_basis") or cand_key
    if basis_a != basis_b:
        return fail("the baseline and candidate have different metric_basis (%s vs %s); "
                    "the two numbers are not comparable." % (basis_a, basis_b))

    ratio = cand_tp / base_tp
    delta_pct = (ratio - 1.0) * 100.0
    band_pct = ((historical.get("noise_band") or 1.045) - 1.0) * 100.0
    verdict = ("no_change" if abs(delta_pct) <= band_pct
               else "gain" if delta_pct > 0 else "regression")

    ops = [{"kernel": a.get("kernel"), "op_speedup": a.get("verified_speedup"),
            "target": a.get("target")} for a in applied["applied"]]
    ab = {
        "baseline": {"throughput": base_tp, "metric": base_key, "basis": basis_a,
                     "provenance": base.get("provenance")},
        "historical_baseline": historical,
        "candidate": {"throughput": cand_tp, "metric": cand_key, "basis": basis_b,
                      "repeats": summary.get("repeats")},
        "e2e_ratio": round(ratio, 6),
        "e2e_delta_pct": round(delta_pct, 4),
        "noise_band_pct": round(band_pct, 4),
        "verdict": verdict,
        "basis_comparable": True,
        "overlay_all_shadowed": True,
        # Deliberately flattened into one level: how much the operator won and how much
        # the machine moved, both visible at a glance, so neither can pass for the other.
        "op_level": ops,
        "op_speedup_max": max([o["op_speedup"] for o in ops if o["op_speedup"]] or [None]),
        "kernels_applied": len(ops),
        "kernels_skipped": applied.get("skipped", []),
        "stock_control": summary.get("stock_control"),
    }
    native_p90 = summary.get("native_interactivity_p90") or {}
    if params.get("require_native_interactivity_p90") and native_p90.get("available") is not True:
        return fail("the scenario requires native P90 interactivity and the harness "
                    "reported none; reciprocal rounded ITL is not a substitute")
    if native_p90.get("available") is True:
        ab["native_interactivity_p90"] = native_p90
    with open(os.path.join(out, "ab.json"), "w") as f:
        json.dump(ab, f, indent=1, ensure_ascii=False)

    notes = ["Operator speedup and end-to-end delta are two different things: %d kernels "
             "were installed, the best operator speedup was %s, and the whole machine "
             "moved %+.4f%% (verdict %s, noise band +/-%.2f%%)."
             % (len(ops), ab["op_speedup_max"], delta_pct, verdict, band_pct)]
    if verdict == "no_change":
        notes.append("Inside the noise band means the machine did not move. This is the "
                     "most common outcome in this scenario and must not be dressed up: "
                     "historically MoE sorting measured 1.80x on the microbenchmark for "
                     "+0.2356% whole-machine, and split4 measured 1.342x for -0.0872%.")

    metrics = {"baseline_tok_s": base_tp, "candidate_tok_s": cand_tp,
               "e2e_delta_pct": ab["e2e_delta_pct"], "op_speedup_max": ab["op_speedup_max"],
               "kernels_applied": len(ops), "verdict": verdict}
    if native_p90.get("available") is True:
        metrics.update({
            "native_p90_interactivity_baseline": native_p90["baseline"],
            "native_p90_interactivity_candidate": native_p90["candidate"],
            "native_p90_interactivity_delta_pct": native_p90["delta_pct"]})
    artifacts = {"ab": "ab.json", "applied": "applied.json", "overlay": "overlay"}
    for name in ("shadow_check.json", "REPORT.md"):
        if os.path.exists(os.path.join(out, name)):
            artifacts[name.split(".")[0].lower()] = name
    return emit("ok", artifacts=artifacts, metrics=metrics, notes=" ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
