#!/usr/bin/env python3
"""Aggregate instrumented Combine peer-readiness wait timing controls."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def _named_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def derive(samples: dict[str, Path]) -> dict:
    routes = {}
    for label, path in sorted(samples.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "mega-moe-v2-combine-wait-timing-v1":
            raise ValueError(f"{path}: unsupported Combine wait schema")
        rank_summaries = payload["rank_summaries"]
        means = [float(entry["mean_us"]) for entry in rank_summaries]
        p95s = [float(entry["p95_us"]) for entry in rank_summaries]
        maxima = [float(entry["max_us"]) for entry in rank_summaries]
        midpoint = len(rank_summaries) // 2
        routes[label] = {
            "route": payload["workload"]["route"],
            "analysis_no_p2p_payload": payload["workload"][
                "analysis_no_p2p_payload"
            ],
            "replays": int(payload["replays"]),
            "blocks_per_rank": int(rank_summaries[0]["samples"]) // int(
                payload["replays"]
            ),
            "rank_mean_wait_us": means,
            "rank_p95_wait_us": p95s,
            "rank_max_wait_us": maxima,
            "rank_max_of_mean_us": max(means),
            "rank_max_of_p95_us": max(p95s),
            "rank_max_of_max_us": max(maxima),
            "rank_mean_spread_us": max(means) - min(means),
            "lower_rank_half_mean_us": statistics.fmean(means[:midpoint]),
            "upper_rank_half_mean_us": statistics.fmean(means[midpoint:]),
            "instrumented_e2e_rank_max_ms": payload["workload"][
                "instrumented_e2e_rank_max_ms"
            ],
            "instrumented_stage2_combine_rank_max_ms": payload["workload"][
                "instrumented_stage2_combine_rank_max_ms"
            ],
            "raw_artifact": str(path.resolve()),
        }

    comparisons = {}
    skew = routes.get("skew.normal")
    uniform = routes.get("uniform.normal")
    allremote = routes.get("allremote.normal")
    skew_no_payload = routes.get("skew.no_payload")
    if skew and uniform:
        comparisons["skew_vs_uniform"] = {
            "rank_max_p95_delta_us": (
                skew["rank_max_of_p95_us"] - uniform["rank_max_of_p95_us"]
            ),
            "rank_mean_spread_delta_us": (
                skew["rank_mean_spread_us"] - uniform["rank_mean_spread_us"]
            ),
        }
    if skew and allremote:
        comparisons["skew_vs_allremote"] = {
            "rank_max_p95_delta_us": (
                skew["rank_max_of_p95_us"] - allremote["rank_max_of_p95_us"]
            ),
            "rank_mean_spread_delta_us": (
                skew["rank_mean_spread_us"] - allremote["rank_mean_spread_us"]
            ),
        }
    if skew and skew_no_payload:
        comparisons["skew_payload_control"] = {
            "normal_rank_max_p95_us": skew["rank_max_of_p95_us"],
            "no_payload_rank_max_p95_us": skew_no_payload["rank_max_of_p95_us"],
            "removed_rank_max_p95_us": (
                skew["rank_max_of_p95_us"]
                - skew_no_payload["rank_max_of_p95_us"]
            ),
            "removed_pct": (
                (
                    skew["rank_max_of_p95_us"]
                    - skew_no_payload["rank_max_of_p95_us"]
                )
                / skew["rank_max_of_p95_us"]
                * 100.0
            ),
        }
    return {
        "schema_version": "geak-combine-wait-evidence-v1",
        "status": "complete_for_instrumented_combine_peer_wait_scope",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routes": routes,
        "comparisons": comparisons,
        "timer": {
            "instruction": "s_memrealtime",
            "frequency_hz": 100_000_000,
            "tick_ns": 10,
        },
        "interpretation": {
            "measured_edge": (
                "Combine wave0 waits for all eight peer epoch flags and executes "
                "the paired system acquire fences"
            ),
            "not_measured": [
                "individual peer attribution inside the wave",
                "Stage1 readiness edges",
                "uninstrumented E2E contribution",
            ],
            "perturbation": (
                "two scalar timer reads plus one rank-local result store per Combine block"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", type=_named_path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = dict(args.sample)
    if len(samples) != len(args.sample):
        raise ValueError("duplicate sample labels are not allowed")
    payload = derive(samples)
    _write(args.output, payload)
    print(f"GEAK_COMBINE_WAIT_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
