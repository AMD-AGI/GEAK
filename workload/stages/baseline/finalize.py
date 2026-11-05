#!/usr/bin/env python3
"""The baseline slot's verdict on the number every later comparison is measured against.

It belongs to the slot, not to any implementation (`stages/CONTRACT.md` section 3.1), and
the driver runs it after every session. Until it existed the slot had no verdict at all:
each `run.sh` wrote its own `result.json`, so the same four rules were written out three
times, `agentx-trace-replay` checked none of them, and a baseline that nothing had
validated could still arrive at applyback as the denominator of a 48-hour A/B.

The five rules, which belong to the stage rather than to whichever skill produced the
number:

  1. An unmeasured baseline must not enter applyback's A/B, so `measured=false` is a
     `blocked` here rather than a footnote downstream.
  2. A false configured validity flag is `blocked`. A non-compliant submission cannot
     serve as a reference. Field names come from `params.valid_flags`.
  3. Drift past the configured band is `blocked`. This is the failure the slot exists to
     catch: a run that completes successfully and lands outside the band quietly
     re-labels every delta measured against it.
  4. The node name is required, because an A/B must be same-node. The same configuration
     varies by fractions of a percent across nodes, so a baseline taken on node A against
     a candidate measured on node B reports the node difference and nothing else.
  5. A scenario that declares `params.pins` declared a reproduction target, so the
     baseline must carry `provenance.alignment_verdict == "aligned"`. Requesting a pin is
     not having one: an unproven pin, or a pin proven wrong, is `blocked`. Without this
     the slot would accept "the replay ran" in place of "the replay reproduced".

It judges the receipt, not the measurement: it never recomputes a throughput and never
decides that a number is implausible.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_shared"))
from emit_result import emit, read_json  # noqa: E402


def write_section(out, baseline, run_root=None):
    """This slot's section of the run report, written where the numbers were accepted.

    The run-level renderer passes paths and reads `result.json`; it does not know that a
    baseline has a provenance method or that a node-local leg outranks a remote receipt.
    That is the slot's knowledge, so the slot writes the paragraph (PRINCIPLES R3/R6).
    Only a baseline this function was reached with is reported -- it runs after the five
    rules passed, never beside a `blocked`.
    """
    pristine = None
    if run_root:
        pristine = read_json(os.path.join(run_root, "profile", "out",
                                          "pristine_baseline.json"))
    L = ["## Baseline", "",
         "| Reference | tok/s | Provenance |", "| --- | --- | --- |",
         "| adopted receipt | %s | %s |"
         % (baseline.get("throughput"),
            (baseline.get("provenance") or {}).get("method") or "—")]
    if isinstance(pristine, dict) and pristine.get("output_tok_per_s") is not None:
        L.append("| node-local pristine leg | %s | measured on this node by the same "
                 "harness as the candidate legs |" % pristine["output_tok_per_s"])
        L += ["", "The A/B is against the **node-local** leg. A receipt from another "
                  "machine is a cross-check, not a reference: only the local leg shares "
                  "the harness, the client flags and the corpus with the candidate."]
    try:
        with open(os.path.join(out, "REPORT.md"), "w") as f:
            f.write("\n".join(L) + "\n")
    except OSError:
        pass  # the verdict stands whether or not the prose could be written


def truthy(value):
    """InferenceX records a valid submission as `true` or as the integer 1."""
    return value is True or value == 1


def find(doc, key):
    """The key as written at the top level, under `provenance`, or under `reference`."""
    for scope in (doc, doc.get("provenance") or {}, doc.get("reference") or {}):
        if isinstance(scope, dict) and key in scope:
            return scope[key]
    return None


def main():
    out = os.environ.get("OUT")
    params = read_json(os.environ["PARAMS"]) if os.environ.get("PARAMS") else {}
    baseline = read_json(os.path.join(out or ".", "baseline.json"))
    if not baseline:
        return emit("blocked", notes="the stage left no readable baseline.json; downstream "
                                     "has no reference to measure against")

    problems = []
    throughput = baseline.get("throughput")
    if not isinstance(throughput, (int, float)) or isinstance(throughput, bool):
        problems.append("baseline.json has no numeric throughput")

    # 1. Measured, or explicitly adopted from a measurement somebody else made.
    if baseline.get("measured") is not True:
        problems.append("measured is not true, so this number must not become the "
                        "denominator of an A/B; re-measure or adopt a receipt that was")

    # 2. A non-compliant submission is not a reference.
    for key in params.get("valid_flags") or []:
        value = find(baseline, key)
        if value is not None and not truthy(value):
            problems.append("%s is %r in the receipt" % (key, value))

    # 3. Drift against the named reference, within the band the config asked for.
    band = params.get("max_drift_pct")
    drift = find(baseline, "drift_pct")
    reference = baseline.get("reference") or {}
    if drift is None and isinstance(reference.get("ratio"), (int, float)):
        drift = (reference["ratio"] - 1.0) * 100.0
    if band is not None:
        if not isinstance(drift, (int, float)) or isinstance(drift, bool):
            problems.append("the config asks for drift within %.3f%% but the receipt "
                            "records no drift against a reference" % band)
        elif abs(drift) > band:
            problems.append("drift is %.3f%% against the %s reference, outside the "
                            "configured %.3f%% band" % (drift, reference.get("name") or
                                                        "named", band))

    # 4. An A/B is same-node or it is nothing.
    node = find(baseline, "node") or find(baseline, "node_name") or find(baseline, "host")
    if not str(node or "").strip():
        problems.append("no node name is recorded, so applyback cannot prove the "
                        "candidate was measured on the same machine")

    # 5. A declared reproduction target has to have been reproduced, and proven.
    verdict = find(baseline, "alignment_verdict")
    if params.get("pins"):
        if verdict is None:
            problems.append("the scenario declares pins, so this baseline is a "
                            "reproduction of a named run, but nothing records whether the "
                            "pins held; 'it ran' is not 'it reproduced'")
        elif verdict != "aligned":
            problems.append("alignment_verdict is %r: a pin was unproven or proven wrong, "
                            "and a baseline that is not the target point silently "
                            "relabels every downstream delta" % verdict)

    if problems:
        return emit("blocked", notes="; ".join(problems))

    metrics = {"throughput": throughput, "noise_band": baseline.get("noise_band"),
               "measured": True, "node": node}
    if verdict is not None:
        metrics["alignment"] = verdict
    if isinstance(drift, (int, float)) and not isinstance(drift, bool):
        metrics["drift_pct"] = round(drift, 5)
    if isinstance(reference.get("ratio"), (int, float)):
        metrics["reference_ratio"] = reference["ratio"]

    write_section(out, baseline, run_root=os.environ.get("RUN_ROOT"))

    artifacts = {"baseline": "baseline.json"}
    for name in ("REPORT.md", "measured_baseline.json", "provenance.json",
                 "alignment.json", "alignment_verified.json", "receipt.copy.json"):
        if os.path.exists(os.path.join(out, name)):
            artifacts[name.split(".")[0].lower()] = name

    return emit("ok", artifacts=artifacts, metrics=metrics,
                notes="%s %s on %s" % (throughput, baseline.get("unit") or "", node))


if __name__ == "__main__":
    sys.exit(main())
