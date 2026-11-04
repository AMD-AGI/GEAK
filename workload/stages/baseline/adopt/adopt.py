#!/usr/bin/env python3
"""Read a measured receipt where it lies -> write the canonical baseline.json.

Translation only. Whether the receipt may serve as a reference -- measured, valid,
within drift, same node -- is `../finalize.py`, which belongs to the slot and runs after
this script. This file deliberately contains no alias tables and no thresholds: every
field name comes from `$PARAMS`, because they are one harness's JSON key names, which are
data, not skill. See SKILL.md alongside.
"""

import hashlib
import json
import os
import socket
import shutil
import sys

# The same rule every slot uses: the stages root, then `_shared`. Resolved from this
# file rather than from $WORKLOAD_ROOT, so running the script by hand needs no env.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_shared"))
from emit_result import emit, read_json  # noqa: E402

VALID_FLAGS = ("submission_valid", "power_valid")


def at(doc, path):
    """`a.b.0.c` into nested dicts and lists; None when any step is absent."""
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


def number(doc, path):
    v = at(doc, path) if path else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path, label):
    """Resolve a configured path against GEAK_ROOT and read it, or say why not."""
    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isabs(path):
        path = os.path.join(os.environ.get("GEAK_ROOT", ""), path)
    if not os.path.exists(path):
        return None, path, "no such %s: %s" % (label, path)
    try:
        with open(path) as f:
            return json.load(f), path, None
    except ValueError as e:
        return None, path, "%s is not valid JSON (%s): %s" % (label, e, path)


def main():
    out = os.environ["OUT"]
    p = read_json(os.environ["PARAMS"], {})

    if not p.get("receipt"):
        return emit("blocked", notes="params.receipt is not configured. Put the path to a "
                                     "measured receipt in the scenario config, or switch "
                                     "stages.baseline.skill to `measure` to measure it.")
    doc, receipt, why = load(p["receipt"], "receipt")
    if why:
        return emit("blocked", notes=why + ". Check params.receipt.")

    ref_doc, ref_path = doc, None
    if p.get("reference_receipt"):
        ref_doc, ref_path, why = load(p["reference_receipt"], "reference_receipt")
        if why:
            return emit("blocked", notes=why + ". It names the document holding the "
                                               "figure this reproduction is measured "
                                               "against.")

    tkey = p.get("throughput_key")
    tput = number(doc, tkey)
    if tput is None:
        return emit("blocked",
                    notes="params.throughput_key is %r, which holds no number in %s. Name "
                          "the field; a dotted path reaches into a nested document. Top "
                          "level keys: %s" % (tkey, receipt, sorted(doc)[:20]))
    rkey = p.get("reference_key")
    ref = number(ref_doc, rkey)

    shutil.copyfile(receipt, os.path.join(out, "receipt.copy.json"))
    if ref_path:
        shutil.copyfile(ref_path, os.path.join(out, "reference.copy.json"))

    flags = dict((f, at(doc, f) if at(doc, f) is not None else at(ref_doc, f))
                 for f in (p.get("valid_flags") or VALID_FLAGS))
    cfg = dict((k, doc[k]) for k in ("tp", "ep", "dp", "gpus", "concurrency")
               if isinstance(doc, dict) and k in doc)
    cfg.update(p.get("config") or {})

    baseline = {
        "metric": p.get("metric", "output_token_throughput"),
        "unit": p.get("unit", "tok/s"),
        "throughput": tput,
        "higher_is_better": True,
        "noise_band": float(p.get("noise_band", 1.045)),
        "config": cfg,
        "reference": None if not ref else {
            "name": rkey, "value": ref, "ratio": round(tput / ref, 8),
            "document": ref_path or receipt},
        "measured": True,
        "provenance": dict(flags, **{
            "method": "adopt",
            "receipt": receipt,
            "receipt_sha256": sha256(receipt),
            "reference_receipt": ref_path,
            "reference_receipt_sha256": sha256(ref_path) if ref_path else None,
            "throughput_key": tkey,
            "model": at(doc, p.get("model_key") or "model"),
            "image": at(doc, p.get("image_key") or "image"),
            "source_run": doc.get("source") if isinstance(doc, dict) else None,
            "node": p.get("node"),
            "adopted_on_host": socket.gethostname(),
        }),
    }
    with open(os.path.join(out, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
        f.write("\n")

    arts = {"baseline": "baseline.json", "receipt_copy": "receipt.copy.json"}
    if ref_path:
        arts["reference_copy"] = "reference.copy.json"
    return emit("ok", artifacts=arts,
                metrics={"throughput": tput, "noise_band": baseline["noise_band"]},
                notes="adopted %s from %s; ../finalize.py decides whether it is admissible"
                      % (tput, os.path.basename(receipt)))


if __name__ == "__main__":
    sys.exit(main())
