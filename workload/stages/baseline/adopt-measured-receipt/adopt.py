#!/usr/bin/env python3
"""Read a measured receipt -> check it -> emit the unified baseline.json. See SKILL.md
alongside it."""

import hashlib
import json
import os
import shutil
import socket
import sys

sys.path.insert(0, os.path.join(os.environ.get("WORKLOAD_ROOT", ""), "scripts"))
from emit_result import emit, read_json  # noqa: E402

# Different receipts name "throughput" differently. Try in order, take the first hit, and
# record the key actually used in provenance.
THROUGHPUT_KEYS = [
    "output_tokens_per_second",
    "output_token_throughput",
    "output_throughput",
    "tokens_per_second",
    "throughput",
]
REFERENCE_KEYS = ["ci_output_tokens_per_second", "reference_output_tokens_per_second"]


def pick(doc, keys):
    for k in keys:
        v = doc.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return k, float(v)
    return None, None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = os.environ["OUT"]
    p = read_json(os.environ["PARAMS"], {})

    receipt = p.get("receipt")
    if not receipt:
        return emit("blocked", notes="params.receipt is not configured. Put the path to a "
                                     "measured receipt in the scenario config, or switch "
                                     "stages.baseline.skill to agentx-trace-replay to "
                                     "measure it directly.")
    receipt = os.path.expandvars(os.path.expanduser(receipt))
    if not os.path.isabs(receipt):
        receipt = os.path.join(os.environ.get("GEAK_ROOT", ""), receipt)
    if not os.path.exists(receipt):
        return emit("blocked", notes="no such receipt: %s. Check params.receipt." % receipt)

    try:
        with open(receipt) as f:
            doc = json.load(f)
    except ValueError as e:
        return emit("blocked", notes="the receipt is not valid JSON (%s): %s" % (e, receipt))

    key = p.get("throughput_key")
    if key:
        tkey, tput = (key, doc.get(key)) if isinstance(doc.get(key), (int, float)) else (None, None)
    else:
        tkey, tput = pick(doc, THROUGHPUT_KEYS)
    if tput is None:
        return emit("blocked",
                    notes="the receipt contains no throughput field (tried %s). Name it "
                          "with params.throughput_key. Keys in the receipt: %s"
                          % (key or THROUGHPUT_KEYS, sorted(doc.keys())))

    rkey = p.get("reference_key")
    if rkey:
        ref = doc.get(rkey) if isinstance(doc.get(rkey), (int, float)) else None
    else:
        rkey, ref = pick(doc, REFERENCE_KEYS)

    problems = []
    if p.get("require_valid", True):
        for flag in ("submission_valid", "power_valid"):
            if flag in doc and doc[flag] is not True:
                problems.append("the receipt has %s=%r. An invalid submission cannot serve "
                                "as a baseline; to use it anyway, set "
                                "params.require_valid to false and state why."
                                % (flag, doc[flag]))

    ratio = None
    if isinstance(ref, (int, float)) and ref:
        ratio = tput / float(ref)
        drift = abs(ratio - 1.0)
        limit = float(p.get("max_reference_drift", 0.05))
        if drift > limit:
            problems.append("drift from the reference is %.2f%%, above "
                            "max_reference_drift=%.2f%%. This reproduction is itself "
                            "misaligned, and adopting it as a baseline writes that drift "
                            "into every later comparison."
                            % (drift * 100, limit * 100))

    shutil.copyfile(receipt, os.path.join(out, "receipt.copy.json"))

    baseline = {
        "metric": p.get("metric", "output_token_throughput"),
        "unit": p.get("unit", "tok/s"),
        "throughput": tput,
        "higher_is_better": True,
        "noise_band": float(p.get("noise_band", 1.045)),
        "config": {k: doc[k] for k in ("tp", "ep", "gpus", "concurrency", "dp")
                   if k in doc},
        "reference": None if ratio is None else {
            "name": rkey, "value": float(ref), "ratio": round(ratio, 8)},
        "measured": True,
        "provenance": {
            "method": "adopt-measured-receipt",
            "receipt": receipt,
            "receipt_sha256": sha256(receipt),
            "throughput_key": tkey,
            "source_run": doc.get("source"),
            "node": p.get("node"),
            "adopted_on_host": socket.gethostname(),
        },
    }
    with open(os.path.join(out, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
        f.write("\n")

    metrics = {"throughput": tput, "noise_band": baseline["noise_band"]}
    if ratio is not None:
        metrics["reference_ratio"] = round(ratio, 6)

    arts = {"baseline": "baseline.json", "receipt_copy": "receipt.copy.json"}
    if problems:
        return emit("blocked", artifacts=arts, metrics=metrics,
                    notes="; ".join(problems))
    note = ""
    if baseline["provenance"]["node"] is None:
        note = ("the receipt records no node name. applyback's A/B must run on the same "
                "node — whole-machine throughput for the same configuration differs by a "
                "fraction of a percent across nodes. Fill in params.node if you know which "
                "machine it was.")
    return emit("ok", artifacts=arts, metrics=metrics, notes=note)


if __name__ == "__main__":
    sys.exit(main())
