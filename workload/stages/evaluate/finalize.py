#!/usr/bin/env python3
"""Judge a native serving comparison; shared by direct Evaluate and kernel applyback.

The benchmark owns its measurements and validity receipts. This module owns metric-basis
admission, the fresh-reference comparison and its report. It never installs a candidate
or re-measures it. Kernel integration is checked by applyback before it calls main with
that metadata. Direct system calls need their declared quality and effect evidence;
reading an E2E receipt alone does not establish that a proposed patch was installed.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_shared"))
from emit_result import emit, read_json  # noqa: E402


def pct(x):
    return "%+.4f%%" % x if isinstance(x, (int, float)) else "—"


def x_(v, nd=4):
    return ("%%.%df×" % nd) % v if isinstance(v, (int, float)) else "—"


def write_section(out, ab, summary):
    """The shared comparison report, with kernel details only when integration supplied them.

    The reference, candidate and verdict are rendered by their validation owner.
    The run-level renderer only combines admitted sections (PRINCIPLES R3/R6).
    """
    summary = summary if isinstance(summary, dict) else {}
    cand = ab.get("candidate") or {}
    L = ["## End to end", "",
         "| | tok/s |", "| --- | --- |",
         "| reference | %s |" % (ab.get("baseline") or {}).get("throughput")]
    for leg in summary.get("legs") or []:
        L.append("| candidate leg `%s` | %s |" % (leg.get("tag"), leg.get("throughput")))
    L += ["| **candidate median (%s legs)** | **%s** |"
          % (cand.get("repeats"), cand.get("throughput")),
          "", "| | |", "| --- | --- |"]
    if "op_level" in ab:
        L.append("| operator speedup, peak | %s |" % x_(ab.get("op_speedup_max")))
    L += ["| **whole machine** | **%s** |" % pct(ab.get("e2e_delta_pct")),
          "| noise band | ±%.2f%% |" % (ab.get("noise_band_pct") or 0),
          "| verdict | **%s** |" % ab.get("verdict")]
    if "op_level" in ab:
        L += ["| overlay shadowed every target | %s |" % ab.get("overlay_all_shadowed"),
              "| overlay loaded by every leg | %s |"
              % summary.get("overlay_loaded_by_every_leg")]
    if summary.get("spread_pct") is not None:
        L.append("| spread between legs | %.4f%% |" % summary["spread_pct"])

    p90 = ab.get("native_interactivity_p90") or {}
    if p90.get("available") is True:
        L += ["", "| Native P90 interactivity | tok/s/user |", "| --- | --- |",
              "| fresh stock | %s |" % p90["baseline"],
              "| median of candidate runs' native P90 | %s |" % p90["candidate"],
              "| change | %s |" % pct(p90["delta_pct"]),
              "", "Source: `%s`. This is the native rate percentile, without inverse-ITL "
                  "conversion." % p90["metric_path"]]

    if "op_level" in ab:
        L += ["", "| Kernel installed | Operator speedup | Target |", "| --- | --- | --- |"]
        for op in ab.get("op_level") or []:
            L.append("| `%s` | %s | `%s` |"
                     % (op.get("kernel"), x_(op.get("op_speedup")), op.get("target")))
        L += ["", "Operator speedup and serving delta are independent measurements. "
                  "A serving delta inside the noise band is reported as no_change."]
    try:
        with open(os.path.join(out, "REPORT.md"), "w") as f:
            f.write("\n".join(L) + "\n")
    except OSError:
        pass  # the verdict stands whether or not the prose could be written

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


def reject(out, reason, status="fail"):
    """Retain a previous comparison without presenting it as this call's result."""
    previous = os.path.join(out, "ab.json")
    if os.path.lexists(previous):
        os.replace(previous, os.path.join(out, "ab.before-failed-run.%d.json" % time.time_ns()))
    return emit(status, notes=reason)


def main(integration=None):
    out = os.environ.get("OUT")
    if not out:
        print("OUT is missing; prepare the Evaluate call environment", file=sys.stderr)
        return 2
    run_root = os.environ.get("RUN_ROOT") or os.path.dirname(os.path.dirname(out))
    params = read_json(os.environ["PARAMS"]) if os.environ.get("PARAMS") else {}

    def fail(reason, status="fail"):
        return reject(out, reason, status)

    # --- The measurement.
    summary = read_json(os.path.join(out, "e2e", "bench_summary.json"))
    if not summary:
        return fail("the stage left no e2e/bench_summary.json, so nothing records what "
                    "the candidate measured", status="blocked")
    if (summary.get("error") or summary.get("independent_arms") is False
            or summary.get("overlay_loaded_by_every_leg") is False):
        return fail("Native summary rejected the comparison: " + str(
            summary.get("error") or "arm independence or actual loading was not verified"))
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

    reference = params.get("reference_baseline") or os.path.join(
        run_root, "baseline", "out", "baseline.json")
    reference = os.path.expandvars(os.path.expanduser(reference))
    if not os.path.isabs(reference):
        reference = os.path.join(os.environ.get("GEAK_ROOT", ""), reference)
    historical = read_json(reference)
    if params.get("reference_baseline") and not historical:
        return fail("the configured reference_baseline is missing or unreadable: " + reference,
                    status="blocked")
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
        return fail("the candidate e2e run measured no finite positive throughput; see "
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
            "target": a.get("target")} for a in (integration or {}).get("applied", [])]
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
        "stock_control": summary.get("stock_control"),
    }
    if integration is not None:
        ab.update(overlay_all_shadowed=True, op_level=ops,
                  op_speedup_max=max([o["op_speedup"] for o in ops if o["op_speedup"]] or [None]),
                  kernels_applied=len(ops), kernels_skipped=integration.get("skipped", []))
    native_p90 = summary.get("native_interactivity_p90") or {}
    if params.get("require_native_interactivity_p90") and native_p90.get("available") is not True:
        return fail("the scenario requires native P90 interactivity and the harness "
                    "reported none; reciprocal rounded ITL is not a substitute")
    if native_p90.get("available") is True:
        ab["native_interactivity_p90"] = native_p90
    with open(os.path.join(out, "ab.json"), "w") as f:
        json.dump(ab, f, indent=1, ensure_ascii=False)

    notes = ["The measured system moved %+.4f%% (verdict %s, noise band +/-%.2f%%)."
             % (delta_pct, verdict, band_pct)]
    metrics = {"baseline_tok_s": base_tp, "candidate_tok_s": cand_tp,
               "e2e_delta_pct": ab["e2e_delta_pct"]}
    if integration is not None:
        notes.insert(0, "%d kernels installed; best operator speedup %s. Operator and "
                     "serving results are separate." % (len(ops), ab["op_speedup_max"]))
        metrics.update(op_speedup_max=ab["op_speedup_max"], kernels_applied=len(ops))
    metrics["verdict"] = verdict
    if native_p90.get("available") is True:
        metrics.update({
            "native_p90_interactivity_baseline": native_p90["baseline"],
            "native_p90_interactivity_candidate": native_p90["candidate"],
            "native_p90_interactivity_delta_pct": native_p90["delta_pct"]})
    write_section(out, ab, summary)

    artifacts = {"ab": "ab.json"}
    if integration is not None:
        artifacts.update(applied="applied.json", overlay="overlay")
    for name in ("shadow_check.json", "REPORT.md"):
        if os.path.exists(os.path.join(out, name)):
            artifacts[name.split(".")[0].lower()] = name
    return emit("ok", artifacts=artifacts, metrics=metrics, notes=" ".join(notes))


if __name__ == "__main__":
    sys.exit(main())
