#!/usr/bin/env python3
"""Build and validate a versioned analysis bundle from existing Profile artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__:
    from .bundle import bundle_from_rank_report, validate_analysis_bundle
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from multi_rank_analysis.bundle import (  # noqa: E402
        bundle_from_rank_report,
        validate_analysis_bundle,
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-report", type=Path, required=True)
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--metric-definitions", type=Path)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--case-artifacts", type=Path)
    parser.add_argument("--route-comparisons", type=Path)
    parser.add_argument("--hardware-context", type=Path)
    parser.add_argument("--measurement-tracks", type=Path)
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument("--derived-evidence", type=Path)
    parser.add_argument("--provenance-catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workload = _read_json(args.workload) if args.workload else None
    hardware_context = (
        _read_json(args.hardware_context)
        if args.hardware_context
        else None
    )
    metric_definitions = (
        _read_json(args.metric_definitions)
        if args.metric_definitions
        else None
    )
    bundle = bundle_from_rank_report(
        _read_json(args.rank_report),
        args.metric,
        workload=workload,
        metric_definitions=metric_definitions,
        expected_world_size=(
            args.expected_world_size
            or (
                hardware_context.get("device_count")
                if isinstance(hardware_context, dict)
                else None
            )
        ),
    )
    if args.case_artifacts:
        case_artifacts = _read_json(args.case_artifacts)
        if not isinstance(case_artifacts, dict):
            parser.error("--case-artifacts must contain a case_id -> fields mapping")
        malformed = sorted(
            case_id
            for case_id, fields in case_artifacts.items()
            if not isinstance(case_id, str) or not isinstance(fields, dict)
        )
        if malformed:
            parser.error(
                f"--case-artifacts entries must be mappings: {malformed}"
            )
        known = {case["case_id"] for case in bundle["cases"]}
        unknown = sorted(set(case_artifacts) - known)
        if unknown:
            parser.error(f"--case-artifacts references unknown cases: {unknown}")
        allowed_fields = {
            "repetitions",
            "workload",
            "comparison_group",
            "trace_files",
            "trace_replays",
            "trace_provenance",
            "att_stats_files",
            "att_occupancy_files",
            "att_provenance",
            "route_summary",
            "route_provenance",
            "software_counters",
        }
        forbidden = {
            case_id: sorted(set(fields) - allowed_fields)
            for case_id, fields in case_artifacts.items()
            if set(fields) - allowed_fields
        }
        if forbidden:
            parser.error(
                f"--case-artifacts contains immutable/core fields: {forbidden}"
            )
        bundle["cases"] = [
            {
                **case,
                **dict(case_artifacts.get(case["case_id"], {})),
            }
            for case in bundle["cases"]
        ]
    for field, path in (
        ("route_comparisons", args.route_comparisons),
        ("measurement_tracks", args.measurement_tracks),
        ("experiment_manifest", args.experiment_manifest),
        ("derived_evidence", args.derived_evidence),
        ("provenance_catalog", args.provenance_catalog),
    ):
        if path:
            bundle[field] = _read_json(path)
    if hardware_context is not None:
        bundle["hardware_context"] = hardware_context
    assembly_paths = {"rank_report": args.rank_report}
    for name, path in (
        ("workload", args.workload),
        ("metric_definitions", args.metric_definitions),
        ("case_artifacts", args.case_artifacts),
        ("route_comparisons", args.route_comparisons),
        ("hardware_context", args.hardware_context),
        ("measurement_tracks", args.measurement_tracks),
        ("experiment_manifest", args.experiment_manifest),
        ("derived_evidence", args.derived_evidence),
        ("provenance_catalog", args.provenance_catalog),
    ):
        if path:
            assembly_paths[name] = path
    bundle["artifacts"]["assembly_inputs"] = {
        name: _file_identity(path)
        for name, path in assembly_paths.items()
    }
    validated = validate_analysis_bundle(bundle)
    _write_json(args.output, validated)
    print(f"GEAK_ANALYSIS_BUNDLE_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
