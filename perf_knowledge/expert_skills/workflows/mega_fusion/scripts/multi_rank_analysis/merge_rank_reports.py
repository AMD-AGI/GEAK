#!/usr/bin/env python3
"""Merge repeated cases[].ranks[] reports into one repetition-aware report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError(f"{path} status is not pass")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain cases[]")
    return payload


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
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    metadata = None
    for path in args.input:
        payload = _read(path)
        current_metadata = payload.get("metadata", {})
        workload_metadata = {
            key: current_metadata.get(key)
            for key in ("world_size", "network")
        }
        if metadata is None:
            metadata = workload_metadata
        elif workload_metadata != metadata:
            raise ValueError("input reports have different workload metadata")
        for case in payload["cases"]:
            case_id = case.get("case_id")
            ranks = case.get("ranks")
            if not isinstance(case_id, str) or not isinstance(ranks, list):
                raise ValueError(f"{path} has malformed case")
            grouped[case_id].append((path, case))

    merged_cases = []
    source_files = []
    for case_id, runs in sorted(grouped.items()):
        first = dict(runs[0][1])
        invariant = {
            key: first.get(key)
            for key in (
                "network",
                "tokens_per_rank",
                "world_size",
                "route",
                "comparison_group",
            )
        }
        for path, case in runs:
            source_files.append(str(path.resolve()))
            if {
                key: case.get(key) for key in invariant
            } != invariant:
                raise ValueError(f"case {case_id!r} invariants differ")
        first["ranks"] = list(runs[0][1]["ranks"])
        first["repetitions"] = [list(case["ranks"]) for _, case in runs]
        merged_cases.append(first)

    _write(
        args.output,
        {
            "schema_version": "geak-repeated-rank-report-v1",
            "record_type": "repeated_runs",
            "status": "pass",
            "metadata": {
                **(metadata or {}),
                "repetitions": min(len(runs) for runs in grouped.values()),
                "source_files": sorted(set(source_files)),
            },
            "cases": merged_cases,
        },
    )
    print(f"GEAK_REPEATED_RANK_REPORT={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
