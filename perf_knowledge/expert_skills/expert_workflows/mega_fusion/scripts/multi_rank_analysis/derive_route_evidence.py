#!/usr/bin/env python3
"""Derive logical communication and GEMM padding from detailed route records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _padding(counts: list[int], block_m: int) -> int:
    return sum(
        math.ceil(count / block_m) * block_m - count
        for count in counts
        if count > 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-report", type=Path, required=True)
    parser.add_argument("--model-dim", type=int, required=True)
    parser.add_argument("--scale-group", type=int, default=32)
    parser.add_argument("--stage1-block-m", type=int, required=True)
    parser.add_argument("--stage2-block-m", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.rank_report.read_text(encoding="utf-8"))
    fp8_row_bytes = args.model_dim + args.model_dim // args.scale_group
    cases = {}
    for case in report["cases"]:
        rank_records = case["ranks"]
        remote_routes = 0
        local_routes = 0
        matrix = []
        for record in sorted(rank_records, key=lambda item: item["rank"]):
            counts = record["route_counts_by_destination"]
            rank = record["rank"]
            matrix.append(list(counts))
            local_routes += counts[rank]
            remote_routes += sum(counts) - counts[rank]
        expert_counts = case["route_summary"]["per_expert_routes"]
        stage1_padding = _padding(expert_counts, args.stage1_block_m)
        stage2_padding = _padding(expert_counts, args.stage2_block_m)
        useful_rows = sum(expert_counts)
        cases[case["case_id"]] = {
            "source_destination_route_matrix": matrix,
            "local_routes": local_routes,
            "remote_routes": remote_routes,
            "remote_route_fraction_pct": remote_routes
            / (local_routes + remote_routes)
            * 100.0,
            "dispatch_remote_payload_bytes": remote_routes * fp8_row_bytes,
            "stage2_remote_payload_bytes": remote_routes * fp8_row_bytes,
            "total_remote_payload_bytes": remote_routes * fp8_row_bytes * 2,
            "payload_row_bytes": fp8_row_bytes,
            "payload_assumption": (
                "FP8 row plus one E8M0 scale byte per 32 elements; "
                "metadata and protocol amplification excluded"
            ),
            "useful_expert_rows": useful_rows,
            "stage1_padded_rows": stage1_padding,
            "stage1_padding_pct": stage1_padding / useful_rows * 100.0,
            "stage2_padded_rows": stage2_padding,
            "stage2_padding_pct": stage2_padding / useful_rows * 100.0,
            "active_experts": sum(count > 0 for count in expert_counts),
            "expert_routes_max": max(expert_counts),
            "expert_routes_mean": sum(expert_counts) / len(expert_counts),
        }
    _write(
        args.output,
        {
            "schema_version": "geak-route-derived-evidence-v1",
            "source": str(args.rank_report.resolve()),
            "model_dim": args.model_dim,
            "scale_group": args.scale_group,
            "stage1_block_m": args.stage1_block_m,
            "stage2_block_m": args.stage2_block_m,
            "cases": cases,
        },
    )
    print(f"GEAK_ROUTE_EVIDENCE_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
