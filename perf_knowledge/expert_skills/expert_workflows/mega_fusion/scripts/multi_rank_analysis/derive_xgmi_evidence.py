#!/usr/bin/env python3
"""Aggregate AMD-SMI XGMI accumulator intervals and matched payload controls."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _named_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def _summary(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    return {
        "samples": values,
        "repetitions": len(values),
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "stdev": statistics.pstdev(values),
        "span_pct": (max(values) - min(values)) / mean * 100.0 if mean else 0.0,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def derive(samples: list[tuple[str, Path]]) -> dict:
    groups: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    labels = set()
    for label, path in samples:
        if label in labels:
            raise ValueError(f"duplicate sample label {label}")
        labels.add(label)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "mega-moe-v2-amdsmi-xgmi-v1":
            raise ValueError(f"{path}: unsupported XGMI schema")
        workload = payload["workload"]
        key = (
            str(workload["route"]),
            bool(workload["analysis_no_p2p_payload"]),
        )
        groups[key].append(
            {
                "label": label,
                "path": str(path.resolve()),
                "replays": int(payload["replays"]),
                "counter_bytes_per_replay": float(
                    payload["derived"][
                        "idle_subtracted_paired_endpoint_bytes_per_replay"
                    ]
                ),
                "counter_amplification": (
                    float(
                        payload["derived"][
                            "counter_to_logical_useful_amplification"
                        ]
                    )
                    if payload["derived"][
                        "counter_to_logical_useful_amplification"
                    ]
                    is not None
                    else None
                ),
                "logical_stage1_payload_bytes_per_replay": int(
                    workload["logical_stage1_payload_bytes_per_replay"]
                ),
                "logical_stage2_payload_bytes_per_replay": int(
                    workload["logical_stage2_payload_bytes_per_replay"]
                ),
                "logical_route_metadata_bytes_per_replay": int(
                    workload["logical_route_metadata_bytes_per_replay"]
                ),
                "remote_route_rows_per_replay": int(
                    workload["remote_route_rows_per_replay"]
                ),
                "idle_paired_endpoint_bytes": float(
                    payload["idle_baseline"]["delta"][
                        "paired_endpoint_normalized_kb"
                    ]
                    * 1024.0
                ),
                "workload_wall_s": float(payload["workload_wall_s"]),
            }
        )

    output_groups = {}
    for (route, no_payload), entries in sorted(groups.items()):
        first = entries[0]
        invariant_fields = (
            "logical_stage1_payload_bytes_per_replay",
            "logical_stage2_payload_bytes_per_replay",
            "logical_route_metadata_bytes_per_replay",
            "remote_route_rows_per_replay",
        )
        for entry in entries[1:]:
            for field in invariant_fields:
                if entry[field] != first[field]:
                    raise ValueError(f"{route}: sample mismatch for {field}")
        name = f"{route}.{'no_payload' if no_payload else 'normal'}"
        amplification_samples = [
            entry["counter_amplification"]
            for entry in entries
            if entry["counter_amplification"] is not None
        ]
        output_groups[name] = {
            "route": route,
            "analysis_no_p2p_payload": no_payload,
            "counter_bytes_per_replay": _summary(
                [entry["counter_bytes_per_replay"] for entry in entries]
            ),
            "counter_to_logical_useful_amplification": (
                _summary(amplification_samples) if amplification_samples else None
            ),
            "idle_paired_endpoint_bytes": _summary(
                [entry["idle_paired_endpoint_bytes"] for entry in entries]
            ),
            "workload_wall_s": _summary(
                [entry["workload_wall_s"] for entry in entries]
            ),
            "logical_bytes_per_replay": {
                field: first[field] for field in invariant_fields
            },
            "raw_artifacts": [entry["path"] for entry in entries],
            "replays_per_sample": [entry["replays"] for entry in entries],
        }

    controls = {}
    for route in sorted({key[0] for key in groups}):
        normal = output_groups.get(f"{route}.normal")
        no_payload = output_groups.get(f"{route}.no_payload")
        if not normal or not no_payload:
            continue
        normal_counter = normal["counter_bytes_per_replay"]["mean"]
        control_counter = no_payload["counter_bytes_per_replay"]["mean"]
        removed_counter = normal_counter - control_counter
        stage2_logical = normal["logical_bytes_per_replay"][
            "logical_stage2_payload_bytes_per_replay"
        ]
        controls[route] = {
            "normal_counter_bytes_per_replay": normal_counter,
            "no_payload_counter_bytes_per_replay": control_counter,
            "removed_counter_bytes_per_replay": removed_counter,
            "removed_counter_pct": (
                removed_counter / normal_counter * 100.0
                if normal_counter
                else 0.0
            ),
            "logical_stage2_payload_bytes_per_replay": stage2_logical,
            "removed_counter_to_logical_stage2_amplification": (
                removed_counter / stage2_logical if stage2_logical else None
            ),
        }

    return {
        "schema_version": "geak-amdsmi-xgmi-evidence-v1",
        "status": "complete_for_firmware_accumulator_scope",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "groups": output_groups,
        "matched_payload_controls": controls,
        "semantics": {
            "counter": (
                "AMD-SMI xgmi_read_data_acc + xgmi_write_data_acc, summed over "
                "endpoints and divided by two for mirrored endpoint normalization"
            ),
            "unit": "bytes after AMD-SMI KB * 1024",
            "idle_subtraction": "matched wall-time idle interval per sample",
            "not_wire_bytes": (
                "firmware fabric accumulators are not protocol packet/header/CRC/retry "
                "wire-byte counters"
            ),
            "not_logical_bytes": (
                "counter values include fabric/coherence/transaction amplification"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=_named_path,
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = derive(args.sample)
    _write(args.output, payload)
    print(f"GEAK_XGMI_EVIDENCE_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
