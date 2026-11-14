#!/usr/bin/env python3
"""Attach deterministic per-rank/per-expert route details to repeated reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-report", type=Path, required=True)
    parser.add_argument("--route-detail", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.rank_report.read_text(encoding="utf-8"))
    by_id = {case["case_id"]: case for case in report["cases"]}
    for detail_path in args.route_detail:
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        if detail.get("status") != "pass":
            raise ValueError(f"{detail_path} status is not pass")
        for detail_case in detail["cases"]:
            case_id = detail_case["case_id"]
            if case_id not in by_id:
                raise ValueError(f"route detail has unknown case {case_id!r}")
            case = by_id[case_id]
            case["route_summary"] = dict(detail_case["route_summary"])
            detail_ranks = {
                rank["rank"]: rank for rank in detail_case["ranks"]
            }
            for record_set in [case["ranks"], *case.get("repetitions", [])]:
                for record in record_set:
                    route_record = detail_ranks[record["rank"]]
                    record["route_counts_by_destination"] = list(
                        route_record["route_counts_by_destination"]
                    )
                    record["expert_counts"] = list(
                        route_record["expert_counts"]
                    )
    report.setdefault("metadata", {})["route_detail_files"] = [
        str(path.resolve()) for path in args.route_detail
    ]
    _write(args.output, report)
    print(f"GEAK_DETAILED_RANK_REPORT={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
