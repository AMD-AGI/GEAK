#!/usr/bin/env python3
"""Summarize final validation results into a stable JSON schema.

This small helper is intentionally generic: it can consume a CSV produced by an
external validation script (run,best_tag,correctness,round,speedup,latency_ms,cov,notes)
and emit per-kernel median/min/max speedups plus accepted/flagged status. It is meant
for post-processing AVO/ASM/IR validation jobs and for canvases/paper tables, so they
never treat lineage-only speedups as final results.
"""
from __future__ import annotations
import argparse, csv, json, statistics
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("-o", "--out", default="validation_report.json")
    ap.add_argument("--accept-threshold", type=float, default=1.0)
    ap.add_argument("--allow-legacy-accept", action="store_true",
                    help="allow CSV medians to use status=accepted; never use for ASM formal evidence")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv_path)))
    by = defaultdict(list)
    correctness = {}
    tags = {}
    for r in rows:
        name = r.get("run") or "unknown"
        correctness.setdefault(name, r.get("correctness") or "")
        tags.setdefault(name, r.get("best_tag") or "")
        try:
            if r.get("speedup"):
                by[name].append(float(r["speedup"]))
        except Exception:
            pass
    report = []
    for name in sorted(set(list(by.keys()) + list(correctness.keys()))):
        vals = by.get(name, [])
        corr = correctness.get(name, "")
        if corr != "OK":
            status = "correctness_failed" if corr else "no_result"
            med = None
        elif vals:
            med = statistics.median(vals)
            if med > args.accept_threshold:
                status = ("accepted" if args.allow_legacy_accept
                          else "legacy_above_threshold")
            else:
                status = "regressed"
        else:
            med = None
            status = "no_benchmark"
        report.append({
            "kernel": name,
            "best_tag": tags.get(name, ""),
            "correctness": corr,
            "validated_speedup": med,
            "rounds": vals,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "validation_status": status,
        })
    json.dump({
        "source": args.csv_path,
        "authoritative": False,
        "results": report,
    }, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
