#!/usr/bin/env python3
"""Require DSR1 and Qwen3.5 together for GEAK K/P/U validation."""
import argparse
import json
import os


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _check(name, path):
    manifest = _load(path)
    counts = manifest.get("evidence_counts") or {}
    unavailable = manifest.get("unavailable") or []
    reasons_complete = all(
        item.get("reason_code") and item.get("reason")
        for item in unavailable)
    counts_complete = (
        sum(int(value) for value in counts.values())
        == int(manifest.get("row_count", -1)))
    passed = (
        manifest.get("status") == "pass"
        and manifest.get("classification_complete") is True
        and counts_complete
        and reasons_complete)
    return {
        "model": name,
        "status": "pass" if passed else "fail",
        "manifest": os.path.abspath(path),
        "row_count": manifest.get("row_count"),
        "evidence_counts": counts,
        "probe_scope_counts": manifest.get("probe_scope_counts", {}),
        "unavailable_reason_counts": manifest.get(
            "unavailable_reason_counts", {}),
        "classification_complete": manifest.get(
            "classification_complete", False),
        "unavailable_reasons_complete": reasons_complete,
    }


def validate(dsr1_manifest, qwen35_manifest, out_path):
    models = [
        _check("dsr1", dsr1_manifest),
        _check("qwen35", qwen35_manifest),
    ]
    result = {
        "schema_version": 1,
        "required_models": ["dsr1", "qwen35"],
        "status": (
            "pass" if all(item["status"] == "pass" for item in models)
            else "fail"),
        "models": models,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsr1-manifest", required=True)
    parser.add_argument("--qwen35-manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = validate(
        args.dsr1_manifest, args.qwen35_manifest, args.out)
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
