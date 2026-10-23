#!/usr/bin/env python3
"""Derive exposed Stage2 payload-path cost from matched no-payload controls."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected ROUTE=PATH")
    return name, Path(raw_path)


def _summary(values: list[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values),
        "span_pct": (max(values) - min(values)) / mean * 100.0 if mean else 0.0,
        "samples": values,
        "repetitions": len(values),
    }


def _load_run(path: Path, *, expect_no_payload: bool) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError(f"{path}: benchmark status must be pass")
    cases = payload.get("cases", [])
    if len(cases) != 1:
        raise ValueError(f"{path}: expected exactly one benchmark case")
    case = cases[0]
    ranks = case.get("ranks", [])
    world_size = int(case["world_size"])
    if len(ranks) != world_size:
        raise ValueError(f"{path}: expected {world_size} rank records, got {len(ranks)}")
    variant = case.get("variant", {})
    no_payload = bool(variant.get("analysis_no_p2p_payload", False))
    if no_payload != expect_no_payload:
        raise ValueError(
            f"{path}: analysis_no_p2p_payload={no_payload}, expected {expect_no_payload}"
        )

    stage2_rank_max = max(
        float(record["timing_ms"]["stage2_combine"]) for record in ranks
    )
    e2e_rank_max = max(float(record["timing_ms"]["e2e"]) for record in ranks)
    has_route_matrix = all("route_counts_by_destination" in record for record in ranks)
    if has_route_matrix:
        route_matrix = [
            [int(value) for value in record["route_counts_by_destination"]]
            for record in ranks
        ]
        if any(len(row) != world_size for row in route_matrix):
            raise ValueError(f"{path}: route-count vector does not match world size")
        total_rows = sum(sum(row) for row in route_matrix)
        remote_rows = sum(
            route_matrix[source][destination]
            for source in range(world_size)
            for destination in range(world_size)
            if source != destination
        )
        remote_rows_by_expert_rank = [
            sum(
                route_matrix[source][destination]
                for source in range(world_size)
                if source != destination
            )
            for destination in range(world_size)
        ]
    else:
        per_destination = case.get("route_summary", {}).get("per_destination_rank")
        if not per_destination:
            raise ValueError(f"{path}: missing both rank route matrix and route summary")
        total_rows = sum(int(value) for value in per_destination)
        remote_rows = None
        remote_rows_by_expert_rank = None
    return {
        "path": str(path.resolve()),
        "route": case["route"],
        "tokens_per_rank": int(case["tokens_per_rank"]),
        "world_size": world_size,
        "stage2_combine_rank_max_ms": stage2_rank_max,
        "e2e_rank_max_ms": e2e_rank_max,
        "total_rows": total_rows,
        "remote_rows": remote_rows,
        "remote_rows_by_expert_rank": remote_rows_by_expert_rank,
    }


def _aggregate_runs(runs: list[dict]) -> dict:
    first = runs[0]
    identity_fields = (
        "route",
        "tokens_per_rank",
        "world_size",
        "total_rows",
    )
    for run in runs[1:]:
        for field in identity_fields:
            if run[field] != first[field]:
                raise ValueError(f"run mismatch for {field}: {run[field]} != {first[field]}")
    return {
        "e2e_rank_max_ms": _summary([run["e2e_rank_max_ms"] for run in runs]),
        "stage2_combine_rank_max_ms": _summary(
            [run["stage2_combine_rank_max_ms"] for run in runs]
        ),
        "paths": [run["path"] for run in runs],
    }


def derive(
    baseline_paths: dict[str, list[Path]],
    control_paths: dict[str, list[Path]],
    *,
    row_bytes: int,
) -> dict:
    if set(baseline_paths) != set(control_paths):
        raise ValueError("baseline and control route sets must match")
    if row_bytes <= 0:
        raise ValueError("row_bytes must be positive")

    routes = {}
    route_deltas = {}
    for route in sorted(baseline_paths):
        baseline_runs = [
            _load_run(path, expect_no_payload=False) for path in baseline_paths[route]
        ]
        control_runs = [
            _load_run(path, expect_no_payload=True) for path in control_paths[route]
        ]
        first = baseline_runs[0]
        for run in baseline_runs + control_runs:
            for field in (
                "route",
                "tokens_per_rank",
                "world_size",
                "total_rows",
            ):
                if run[field] != first[field]:
                    raise ValueError(
                        f"{route}: baseline/control mismatch for {field}: "
                        f"{run[field]} != {first[field]}"
                    )
        if first["route"] != route:
            raise ValueError(f"{route}: artifact route is {first['route']}")
        routing = next(
            (
                run
                for run in baseline_runs + control_runs
                if run["remote_rows"] is not None
            ),
            None,
        )
        if routing is None:
            raise ValueError(f"{route}: no artifact contains a per-rank route matrix")
        for run in baseline_runs + control_runs:
            if run["remote_rows"] is not None and (
                run["remote_rows"] != routing["remote_rows"]
                or run["remote_rows_by_expert_rank"]
                != routing["remote_rows_by_expert_rank"]
            ):
                raise ValueError(f"{route}: per-rank route matrices do not agree")

        baseline = _aggregate_runs(baseline_runs)
        control = _aggregate_runs(control_runs)
        stage2_delta = (
            baseline["stage2_combine_rank_max_ms"]["mean"]
            - control["stage2_combine_rank_max_ms"]["mean"]
        )
        e2e_delta = (
            baseline["e2e_rank_max_ms"]["mean"]
            - control["e2e_rank_max_ms"]["mean"]
        )
        route_deltas[route] = stage2_delta
        total_bytes = first["total_rows"] * row_bytes
        remote_bytes = routing["remote_rows"] * row_bytes
        remote_rank_max_bytes = max(routing["remote_rows_by_expert_rank"]) * row_bytes
        routes[route] = {
            "workload": {
                "tokens_per_rank": first["tokens_per_rank"],
                "world_size": first["world_size"],
                "stage2_output_rows_global": first["total_rows"],
                "stage2_output_row_bytes": row_bytes,
                "stage2_output_bytes_global": total_bytes,
                "remote_stage2_output_rows_global": routing["remote_rows"],
                "remote_stage2_output_ratio": (
                    routing["remote_rows"] / first["total_rows"]
                    if first["total_rows"]
                    else 0.0
                ),
                "logical_remote_stage2_output_bytes_global": remote_bytes,
                "logical_remote_stage2_output_bytes_expert_rank_max": (
                    remote_rank_max_bytes
                ),
            },
            "baseline": baseline,
            "no_payload_control": control,
            "observed_delta": {
                "e2e_rank_max_ms": e2e_delta,
                "stage2_combine_rank_max_ms": stage2_delta,
                "stage2_combine_reduction_pct": (
                    stage2_delta
                    / baseline["stage2_combine_rank_max_ms"]["mean"]
                    * 100.0
                ),
            },
        }

    local_floor = route_deltas.get("local-only")
    if local_floor is not None:
        for route, evidence in routes.items():
            remote_incremental = max(route_deltas[route] - local_floor, 0.0)
            remote_bytes = evidence["workload"][
                "logical_remote_stage2_output_bytes_expert_rank_max"
            ]
            evidence["observed_delta"]["local_payload_path_floor_ms"] = local_floor
            evidence["observed_delta"][
                "remote_incremental_exposed_ms_approx"
            ] = remote_incremental
            evidence["observed_delta"][
                "logical_remote_bytes_over_exposed_time_gbps"
            ] = (
                remote_bytes / remote_incremental / 1.0e6
                if remote_incremental > 0.0 and remote_bytes > 0
                else None
            )

    return {
        "schema_version": "geak-payload-control-evidence-v1",
        "status": "complete_for_isolated_operator_scope",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "control_semantics": {
            "preserved": [
                "Stage2 GEMM and CShuffle computation",
                "scatter address generation",
                "buffer-store instruction issue",
                "Stage2-to-Combine stream ordering",
            ],
            "removed": (
                "Stage2 output payload transactions: all scatter offsets are forced "
                "to the buffer resource's bounded OOB sentinel"
            ),
            "correctness": "not_applicable_output_is_intentionally_invalid",
        },
        "routes": routes,
        "interpretation": {
            "primary_metric": "stage2_combine_rank_max_ms",
            "delta_meaning": (
                "Exposed payload-path latency in this serialized operator graph, including "
                "store/coherence/cache effects visible to Combine."
            ),
            "not_measured": [
                "physical XGMI transaction bytes",
                "protocol overhead",
                "payload transfer time hidden under Stage2 compute",
            ],
            "effective_bandwidth_warning": (
                "logical_remote_bytes_over_exposed_time_gbps is not link bandwidth; "
                "overlap can make it exceed a hardware transfer ceiling."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=_named_path,
        action="append",
        required=True,
        metavar="ROUTE=PATH",
    )
    parser.add_argument(
        "--control",
        type=_named_path,
        action="append",
        required=True,
        metavar="ROUTE=PATH",
    )
    parser.add_argument("--row-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_paths: dict[str, list[Path]] = defaultdict(list)
    control_paths: dict[str, list[Path]] = defaultdict(list)
    for route, path in args.baseline:
        baseline_paths[route].append(path)
    for route, path in args.control:
        control_paths[route].append(path)
    payload = derive(
        dict(baseline_paths),
        dict(control_paths),
        row_bytes=args.row_bytes,
    )
    _write(args.output, payload)
    print(f"GEAK_PAYLOAD_CONTROL_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
