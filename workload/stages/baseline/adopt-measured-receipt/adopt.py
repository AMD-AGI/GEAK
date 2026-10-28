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

# Flags that make a receipt inadmissible. Looked for in every document this skill reads,
# because the run that writes the throughput and the run that writes the validity verdict
# are often two files from the same harness.
VALID_FLAGS = ("submission_valid", "power_valid")

# Receipts that nest their numbers still describe the same configuration; these are the
# flat names this skill reports it under.
CONFIG_ALIASES = {"tp": "tp", "ep": "ep", "dp": "dp", "gpus": "gpus",
                  "num_gpus": "gpus", "concurrency": "concurrency", "conc": "concurrency"}


def resolve(doc, path):
    """Read a dotted path out of a nested document: `a.b.0.c`.

    A receipt is somebody else's format. The AgentX result document, for one, records
    throughput at `request_metrics.throughput.output.tokens_per_second` and nothing at the
    top level -- so insisting on a flat key would mean hand-copying the number into a file
    this pipeline then calls evidence, which is exactly the kind of retyping a receipt
    exists to avoid. A plain string without dots still behaves as a flat key.
    """
    cur = doc
    for part in str(path).split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.lstrip("-").isdigit():
            idx = int(part)
            cur = cur[idx] if -len(cur) <= idx < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    return cur


def number(doc, path):
    v = resolve(doc, path)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def pick(doc, keys):
    for k in keys:
        v = number(doc, k)
        if v is not None:
            return k, v
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

    # The reference figure and the validity verdict are not always in the same file as the
    # throughput. A harness that writes `result.json` and `COMPARISON.json` side by side
    # is not malformed, so the reference may be named in a sibling document -- which is
    # read, hashed and copied exactly like the receipt, never summarised by hand.
    ref_doc, ref_path = doc, None
    if p.get("reference_receipt"):
        ref_path = os.path.expandvars(os.path.expanduser(p["reference_receipt"]))
        if not os.path.isabs(ref_path):
            ref_path = os.path.join(os.environ.get("GEAK_ROOT", ""), ref_path)
        if not os.path.exists(ref_path):
            return emit("blocked", notes="no such reference_receipt: %s. It names the "
                                         "document holding the figure this reproduction is "
                                         "measured against." % ref_path)
        try:
            with open(ref_path) as f:
                ref_doc = json.load(f)
        except ValueError as e:
            return emit("blocked", notes="reference_receipt is not valid JSON (%s): %s"
                                         % (e, ref_path))

    key = p.get("throughput_key")
    if key:
        tkey, tput = (key, number(doc, key))
        if tput is None:
            tkey = None
    else:
        tkey, tput = pick(doc, THROUGHPUT_KEYS)
    if tput is None:
        return emit("blocked",
                    notes="the receipt contains no throughput field (tried %s). Name it "
                          "with params.throughput_key, which accepts a dotted path into a "
                          "nested document. Keys in the receipt: %s"
                          % (key or THROUGHPUT_KEYS, sorted(doc.keys())))

    rkey = p.get("reference_key")
    if rkey:
        ref = number(ref_doc, rkey)
    else:
        rkey, ref = pick(ref_doc, REFERENCE_KEYS)

    problems = []
    if p.get("require_valid", True):
        for src, d in (("the receipt", doc), ("reference_receipt", ref_doc)):
            for flag in VALID_FLAGS:
                # InferenceX result documents store power_valid as the integer 1. Accept
                # exactly True or 1; every other value (0, False, "true", null) is invalid.
                v = d.get(flag)
                if flag in d and not (v is True or (type(v) is int and v == 1)):
                    problems.append("%s has %s=%r. An invalid submission cannot serve "
                                    "as a baseline; to use it anyway, set "
                                    "params.require_valid to false and state why."
                                    % (src, flag, d[flag]))

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
    if ref_path:
        shutil.copyfile(ref_path, os.path.join(out, "reference.copy.json"))

    cfg = dict((CONFIG_ALIASES[k], doc[k]) for k in CONFIG_ALIASES if k in doc)
    cfg.update(p.get("config") or {})

    baseline = {
        "metric": p.get("metric", "output_token_throughput"),
        "unit": p.get("unit", "tok/s"),
        "throughput": tput,
        "higher_is_better": True,
        "noise_band": float(p.get("noise_band", 1.045)),
        "config": cfg,
        "reference": None if ratio is None else {
            "name": rkey, "value": float(ref), "ratio": round(ratio, 8),
            "document": ref_path or receipt},
        "measured": True,
        "provenance": {
            "method": "adopt-measured-receipt",
            "receipt": receipt,
            "receipt_sha256": sha256(receipt),
            "reference_receipt": ref_path,
            "reference_receipt_sha256": sha256(ref_path) if ref_path else None,
            "throughput_key": tkey,
            "model": resolve(doc, p.get("model_key") or "model"),
            "image": resolve(doc, p.get("image_key") or "image"),
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
    if ref_path:
        arts["reference_copy"] = "reference.copy.json"
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
