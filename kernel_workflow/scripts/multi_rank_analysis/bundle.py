"""Versioned analysis-bundle contract joining Profile artifacts to analysis runners."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .experiments import validate_experiment_manifest
from .evidence import (
    EVIDENCE_CATALOG_SCHEMA_VERSION,
    validate_evidence_catalog,
)
from .hardware import validate_hardware_context
from .provenance import validate_collection_provenance

ANALYSIS_BUNDLE_SCHEMA_VERSION = "geak-analysis-bundle-v2"
_CASE_WORKLOAD_FIELDS = (
    "network",
    "tokens",
    "tokens_per_rank",
    "world_size",
    "model_dim",
    "inter_dim",
    "experts",
    "topk",
    "dtype",
    "config_tokens",
    "mtpr",
    "path",
    "p2p_quant",
    "config",
)

__all__ = [
    "ANALYSIS_BUNDLE_SCHEMA_VERSION",
    "bundle_from_rank_report",
    "validate_analysis_bundle",
    "validate_measurement_tracks",
]


def _case_id(case: Mapping[str, Any], index: int) -> str:
    value = case.get("case_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"case {index} must have a non-empty case_id")
    return value


def _metric_definition(path: str) -> dict:
    if path.startswith("timing_ms."):
        return {
            "unit": "ms",
            "direction": "lower",
            "reduction": "rank_max",
            "semantic": (
                "e2e_latency"
                if path == "timing_ms.e2e"
                else "stage_latency"
            ),
        }
    return {
        "unit": "unknown",
        "direction": "neutral",
        "reduction": "rank_max",
        "semantic": "opaque",
    }


def _case_workload(
    case: Mapping[str, Any],
    fallback: Mapping[str, Any],
) -> dict:
    explicit = case.get("workload")
    if isinstance(explicit, Mapping) and explicit:
        return dict(explicit)
    selected = {
        field: case[field]
        for field in _CASE_WORKLOAD_FIELDS
        if field in case
    }
    return selected or dict(fallback)


def _workload_fingerprint(workload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        workload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bundle_from_rank_report(
    report: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    metric_paths: Sequence[str],
    workload: Mapping[str, Any] | None = None,
    metric_definitions: Mapping[str, Mapping[str, Any]] | None = None,
    expected_world_size: int | None = None,
) -> dict:
    """Normalize existing UT shapes, including AITER ``cases[].ranks[]``."""
    if isinstance(report, list):
        report_mapping: Mapping[str, Any] = {}
        bare_records = report
    elif isinstance(report, Mapping):
        report_mapping = report
        bare_records = None
    else:
        raise TypeError("rank report must be a mapping or record list")
    metrics = list(metric_paths)
    if not metrics or not all(
        isinstance(path, str) and path for path in metrics
    ):
        raise ValueError(
            "metric_paths must be a non-empty list of non-empty strings"
        )
    normalized_workload = dict(
        workload
        or report_mapping.get("metadata")
        or {"source": "rank_report"}
    )
    inferred_world_size = (
        expected_world_size
        or normalized_workload.get("world_size")
        or normalized_workload.get("local_world_size")
    )
    world_size_source = (
        "caller"
        if expected_world_size is not None
        else "source_metadata"
    )
    if inferred_world_size is None:
        source_records = (
            bare_records
            if bare_records is not None
            else report_mapping.get("records")
        )
        if isinstance(source_records, list) and source_records:
            ranks = [
                record.get("rank", index)
                for index, record in enumerate(source_records)
                if isinstance(record, Mapping)
            ]
            inferred_world_size = max(ranks) + 1 if ranks else len(source_records)
            world_size_source = "inferred_rank_records"
    if inferred_world_size is None and isinstance(report_mapping.get("cases"), list):
        first_case = report_mapping["cases"][0]
        if isinstance(first_case, Mapping):
            inferred_world_size = first_case.get("world_size")
            if inferred_world_size is not None:
                world_size_source = "source_case"
            elif isinstance(first_case.get("ranks"), list) and first_case["ranks"]:
                ranks = [
                    record.get("rank", index)
                    for index, record in enumerate(first_case["ranks"])
                    if isinstance(record, Mapping)
                ]
                inferred_world_size = (
                    max(ranks) + 1 if ranks else len(first_case["ranks"])
                )
                world_size_source = "inferred_first_case_ranks"
    definitions = {
        path: dict((metric_definitions or {}).get(path) or _metric_definition(path))
        for path in metrics
    }
    if bare_records is not None:
        case_workload = dict(normalized_workload)
        raw_cases = [
            {
                "case_id": "default",
                "rank_records": bare_records,
                "repetitions": None,
                "metric_paths": metrics,
                "workload": case_workload,
                "comparison_group": _workload_fingerprint(case_workload),
            }
        ]
    elif isinstance(report_mapping.get("records"), list):
        case_workload = dict(normalized_workload)
        raw_cases = [
            {
                "case_id": report_mapping.get("case_id", "default"),
                "rank_records": report_mapping["records"],
                "repetitions": report_mapping.get("repetitions"),
                "metric_paths": metrics,
                "workload": case_workload,
                "comparison_group": _workload_fingerprint(case_workload),
            }
        ]
    else:
        cases = report_mapping.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("rank report must contain records[] or cases[]")
        raw_cases = []
        for index, case in enumerate(cases):
            if not isinstance(case, Mapping):
                raise TypeError(f"case {index} must be a mapping")
            ranks = case.get("ranks")
            if not isinstance(ranks, list):
                raise ValueError(
                    f"case {_case_id(case, index)!r} must contain ranks[]"
                )
            raw_cases.append(
                {
                    "case_id": _case_id(case, index),
                    "rank_records": ranks,
                    "metric_paths": metrics,
                    "repetitions": case.get("repetitions"),
                    "workload": _case_workload(case, normalized_workload),
                    "comparison_group": case.get("comparison_group")
                    or _workload_fingerprint(
                        _case_workload(case, normalized_workload)
                    ),
                    "route_summary": dict(case.get("route_summary") or {}),
                    "route_summary_source": (
                        "rank_report" if case.get("route_summary") else None
                    ),
                    "source_case": {
                        key: value
                        for key, value in case.items()
                        if key != "ranks"
                    },
                }
            )
    bundle = {
        "schema_version": ANALYSIS_BUNDLE_SCHEMA_VERSION,
        "source": {
            "schema_version": report_mapping.get("schema_version", "unknown"),
            "status": report_mapping.get("status", "unknown"),
            "record_type": report_mapping.get("record_type", "unknown"),
        },
        "workload": normalized_workload,
        "expected_world_size": inferred_world_size,
        "world_size_source": world_size_source,
        "metric_definitions": definitions,
        "cases": raw_cases,
        "route_comparisons": [],
        "measurement_tracks": {},
        "artifacts": {},
        "provenance_catalog": {},
    }
    return validate_analysis_bundle(bundle)


def validate_measurement_tracks(tracks: Mapping[str, Any]) -> dict:
    if not isinstance(tracks, Mapping):
        raise ValueError("measurement_tracks must be a mapping")
    for name, track in tracks.items():
        if not isinstance(track, Mapping):
            raise ValueError(f"measurement track {name!r} must be a mapping")
        if track.get("status") not in ("complete", "partial", "missing"):
            raise ValueError(
                f"measurement track {name!r} status must be complete, partial, or missing"
            )
        if track.get("status") == "complete" and not track.get("evidence"):
            raise ValueError(
                f"measurement track {name!r} complete status requires evidence"
            )
        if track.get("status") == "complete":
            evidence = track["evidence"]
            if not isinstance(evidence, Mapping):
                raise ValueError(
                    f"measurement track {name!r} evidence must be a mapping"
                )
            for field in ("artifact_refs", "metrics", "provenance_refs"):
                values = evidence.get(field)
                if (
                    not isinstance(values, list)
                    or not values
                    or not all(
                        isinstance(value, str) and value
                        for value in values
                    )
                ):
                    raise ValueError(
                        f"measurement track {name!r} evidence requires "
                        f"a non-empty string list {field}"
                    )
    return {str(name): dict(track) for name, track in tracks.items()}


def _validate_metric_definitions(
    definitions: Mapping[str, Any],
) -> dict:
    if not isinstance(definitions, Mapping) or not definitions:
        raise ValueError("metric_definitions must be a non-empty mapping")
    normalized = {}
    for path, definition in definitions.items():
        if not isinstance(path, str) or not path:
            raise ValueError("metric definition paths must be non-empty strings")
        if not isinstance(definition, Mapping):
            raise ValueError(f"metric definition {path!r} must be a mapping")
        for field in ("unit", "direction", "reduction", "semantic"):
            if not isinstance(definition.get(field), str) or not definition[field]:
                raise ValueError(
                    f"metric definition {path!r} requires non-empty {field}"
                )
        if definition["direction"] not in ("lower", "higher", "neutral"):
            raise ValueError(
                f"metric definition {path!r} has invalid direction"
            )
        if definition["reduction"] != "rank_max":
            raise ValueError(
                f"metric definition {path!r} reduction must be rank_max"
            )
        normalized[path] = dict(definition)
    return normalized


def _validate_rank_records(
    records: list,
    expected_world_size: int,
    label: str,
) -> None:
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{label}[{index}] must be a mapping")
        rank = record.get("rank", index)
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= expected_world_size
        ):
            raise ValueError(f"{label}[{index}] has invalid rank={rank!r}")
        if rank in seen:
            raise ValueError(f"{label} has duplicate rank={rank}")
        seen.add(rank)


def _validate_nonnegative_tree(value: Any, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{label} must not contain booleans")
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must contain non-negative finite values")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_nonnegative_tree(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_nonnegative_tree(item, f"{label}[{index}]")
        return
    raise ValueError(f"{label} contains unsupported value {value!r}")


def _validate_software_counters(
    counters: dict,
    case_id: str,
    expected_world_size: int,
) -> dict:
    records = counters.get("records")
    provenance = validate_collection_provenance(counters.get("provenance"))
    required_counter_fields = (
        "payload_bytes_by_peer",
        "metadata_bytes_by_peer",
        "publication_count_by_peer",
        "useful_rows_by_expert",
        "padded_rows_by_expert",
        "gemm1_tiles_by_expert",
        "gemm2_tiles_by_expert",
        "readiness_polls_by_edge",
        "wait_cycles_by_edge",
        "barrier_count_by_edge",
        "termination_polls",
    )
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"case {case_id!r} software counter {index} must be a mapping"
            )
        if record.get("case_id") != case_id:
            raise ValueError(
                f"case {case_id!r} software counter {index} has wrong case_id"
            )
        rank = record.get("rank")
        repetition = record.get("repetition")
        world_size = record.get("world_size")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= expected_world_size
        ):
            raise ValueError("software counter rank is invalid")
        if (
            isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or repetition < 0
        ):
            raise ValueError("software counter repetition is invalid")
        if world_size != expected_world_size:
            raise ValueError("software counter world_size mismatch")
        identity = (repetition, rank)
        if identity in seen:
            raise ValueError(
                f"duplicate software counter repetition/rank {identity}"
            )
        seen.add(identity)
        for field in required_counter_fields:
            if field not in record:
                raise ValueError(
                    f"software counter record requires {field}"
                )
            _validate_nonnegative_tree(
                record[field],
                f"software_counter[{index}].{field}",
            )
    expected = {
        (repetition, rank)
        for repetition in range(provenance["repetitions"])
        for rank in range(expected_world_size)
    }
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing or unexpected:
        raise ValueError(
            "software counter repetition/rank coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {"records": [dict(record) for record in records], "provenance": provenance}


def validate_analysis_bundle(bundle: Mapping[str, Any]) -> dict:
    if not isinstance(bundle, Mapping):
        raise TypeError("analysis bundle must be a mapping")
    if bundle.get("schema_version") != ANALYSIS_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported analysis bundle schema: {bundle.get('schema_version')!r}"
        )
    workload = bundle.get("workload")
    if not isinstance(workload, Mapping) or not workload:
        raise ValueError("analysis bundle workload must be a non-empty mapping")
    expected_world_size = bundle.get("expected_world_size")
    if (
        isinstance(expected_world_size, bool)
        or not isinstance(expected_world_size, int)
        or expected_world_size <= 0
    ):
        raise ValueError("expected_world_size must be a positive integer")
    world_size_source = bundle.get("world_size_source")
    if not isinstance(world_size_source, str) or not world_size_source:
        raise ValueError("world_size_source must be a non-empty string")
    source = bundle.get("source", {})
    if not isinstance(source, Mapping):
        raise ValueError("analysis bundle source must be a mapping")
    metric_definitions = _validate_metric_definitions(
        bundle.get("metric_definitions", {})
    )
    cases = bundle.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("analysis bundle cases must be a non-empty list")
    normalized_cases = []
    seen = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise TypeError(f"case {index} must be a mapping")
        case_id = _case_id(case, index)
        if case_id in seen:
            raise ValueError(f"duplicate case_id={case_id!r}")
        seen.add(case_id)
        records = case.get("rank_records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"case {case_id!r} must contain rank_records[]")
        _validate_rank_records(
            records,
            expected_world_size,
            f"case {case_id!r} rank_records",
        )
        metric_paths = case.get("metric_paths")
        if not isinstance(metric_paths, list) or not metric_paths:
            raise ValueError(f"case {case_id!r} must contain metric_paths[]")
        if not all(isinstance(path, str) and path for path in metric_paths):
            raise ValueError(
                f"case {case_id!r} metric_paths entries must be non-empty strings"
            )
        undefined = sorted(set(metric_paths) - set(metric_definitions))
        if undefined:
            raise ValueError(
                f"case {case_id!r} uses undefined metrics: {undefined}"
            )
        repetitions = case.get("repetitions")
        if repetitions is not None and not isinstance(repetitions, list):
            raise ValueError(f"case {case_id!r} repetitions must be a list")
        if repetitions is not None:
            for repetition_index, repetition in enumerate(repetitions):
                if not isinstance(repetition, list) or not repetition:
                    raise ValueError(
                        f"case {case_id!r} repetition {repetition_index} "
                        "must be a non-empty rank-record list"
                    )
                _validate_rank_records(
                    repetition,
                    expected_world_size,
                    f"case {case_id!r} repetition {repetition_index}",
                )
        case_workload = case.get("workload")
        if not isinstance(case_workload, Mapping) or not case_workload:
            raise ValueError(f"case {case_id!r} workload must be non-empty")
        comparison_group = case.get("comparison_group")
        if not isinstance(comparison_group, str) or not comparison_group:
            raise ValueError(
                f"case {case_id!r} comparison_group must be non-empty"
            )
        trace_value = case.get("trace_files", [])
        att_value = case.get("att_stats_files", [])
        occupancy_value = case.get("att_occupancy_files", [])
        if not isinstance(trace_value, list) or not all(
            isinstance(path, str) and path for path in trace_value
        ):
            raise ValueError(f"case {case_id!r} trace_files must be a string list")
        if not isinstance(att_value, list) or not all(
            isinstance(path, str) and path for path in att_value
        ):
            raise ValueError(
                f"case {case_id!r} att_stats_files must be a string list"
            )
        if not isinstance(occupancy_value, list) or not all(
            isinstance(path, str) and path for path in occupancy_value
        ):
            raise ValueError(
                f"case {case_id!r} att_occupancy_files must be a string list"
            )
        trace_files = list(trace_value)
        att_files = list(att_value)
        occupancy_files = list(occupancy_value)
        if occupancy_files and len(occupancy_files) != len(att_files):
            raise ValueError(
                f"case {case_id!r} ATT stats/occupancy file counts must match"
            )
        trace_replays = case.get("trace_replays")
        if trace_replays is not None and (
            isinstance(trace_replays, bool)
            or not isinstance(trace_replays, int)
            or trace_replays <= 0
        ):
            raise ValueError(
                f"case {case_id!r} trace_replays must be a positive integer"
            )
        if trace_files and trace_replays is None:
            raise ValueError(
                f"case {case_id!r} must define trace_replays when trace_files are present"
            )
        route_summary = dict(case.get("route_summary", {}))
        software_counters = dict(case.get("software_counters", {}))
        provenance_fields = {}
        for artifact_name, present in (
            ("trace", bool(trace_files)),
            ("att", bool(att_files)),
            ("route", bool(route_summary)),
        ):
            field = f"{artifact_name}_provenance"
            value = case.get(field)
            if (
                artifact_name == "route"
                and present
                and case.get("route_summary_source") == "rank_report"
            ):
                continue
            if present:
                provenance_fields[field] = validate_collection_provenance(value)
                files = {
                    "trace": trace_files,
                    "att": att_files + occupancy_files,
                    "route": [],
                }[artifact_name]
                if files:
                    raw_paths = set(provenance_fields[field]["raw_artifacts"])
                    unprovenanced = [
                        str(path) for path in files if str(path) not in raw_paths
                    ]
                    if unprovenanced:
                        raise ValueError(
                            f"case {case_id!r} {artifact_name} files missing from "
                            f"provenance raw_artifacts: {unprovenanced}"
                        )
            elif value is not None:
                provenance_fields[field] = validate_collection_provenance(value)
        if software_counters:
            if not isinstance(software_counters.get("records"), list):
                raise ValueError(
                    f"case {case_id!r} software_counters.records must be a list"
                )
            software_counters = _validate_software_counters(
                software_counters,
                case_id,
                expected_world_size,
            )
        normalized_cases.append(
            {
                **dict(case),
                "case_id": case_id,
                "rank_records": list(records),
                "metric_paths": list(metric_paths),
                "repetitions": repetitions,
                "workload": dict(case_workload),
                "comparison_group": comparison_group,
                "trace_files": trace_files,
                "trace_replays": trace_replays,
                "att_stats_files": att_files,
                "att_occupancy_files": occupancy_files,
                "software_counters": software_counters,
                "route_summary": route_summary,
                "route_summary_source": case.get("route_summary_source"),
                **provenance_fields,
            }
        )
    comparisons = bundle.get("route_comparisons", [])
    if not isinstance(comparisons, list):
        raise ValueError("route_comparisons must be a list")
    normalized_comparisons = []
    cases_by_id = {case["case_id"]: case for case in normalized_cases}
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise TypeError("route comparison must be a mapping")
        for field in ("baseline_case_id", "candidate_case_id"):
            value = comparison.get(field)
            if value not in seen:
                raise ValueError(
                    f"route comparison {field} references unknown case {value!r}"
                )
        metric_path = comparison.get("metric_path")
        if not isinstance(metric_path, str) or not metric_path:
            raise ValueError("route comparison metric_path must be a non-empty string")
        if metric_path not in metric_definitions:
            raise ValueError(
                f"route comparison references undefined metric {metric_path!r}"
            )
        baseline_group = cases_by_id[comparison["baseline_case_id"]][
            "comparison_group"
        ]
        candidate_group = cases_by_id[comparison["candidate_case_id"]][
            "comparison_group"
        ]
        if baseline_group != candidate_group:
            raise ValueError(
                "route comparison cases do not share comparison_group"
            )
        declared_group = comparison.get("comparison_group", baseline_group)
        if declared_group != baseline_group:
            raise ValueError(
                "route comparison comparison_group does not match its cases"
            )
        tokens = comparison.get("tokens_per_rank")
        if tokens is not None and (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
        ):
            raise ValueError(
                "route comparison tokens_per_rank must be a positive integer or null"
            )
        normalized_comparisons.append(
            {**dict(comparison), "comparison_group": baseline_group}
        )
    tracks = validate_measurement_tracks(bundle.get("measurement_tracks", {}))
    artifacts = bundle.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be a mapping")
    provenance_payload = bundle.get("provenance_catalog", {})
    if not isinstance(provenance_payload, Mapping):
        raise ValueError("provenance_catalog must be a mapping")
    provenance_catalog = {
        str(provenance_id): validate_collection_provenance(provenance)
        for provenance_id, provenance in provenance_payload.items()
    }
    derived_evidence_payload = bundle.get("derived_evidence", {})
    if not isinstance(derived_evidence_payload, Mapping):
        raise ValueError("derived_evidence must be a mapping")
    derived_evidence = validate_evidence_catalog(
        {
            "schema_version": EVIDENCE_CATALOG_SCHEMA_VERSION,
            "entries": derived_evidence_payload,
            "provenance": {
                provenance_id: {
                    "kind": "collection",
                    "status": "complete",
                    "data": provenance,
                }
                for provenance_id, provenance in provenance_catalog.items()
            },
        }
    )["entries"]
    hardware_context = (
        validate_hardware_context(bundle["hardware_context"])
        if bundle.get("hardware_context") is not None
        else None
    )
    experiment_manifest = (
        validate_experiment_manifest(bundle["experiment_manifest"])
        if bundle.get("experiment_manifest") is not None
        else None
    )
    if experiment_manifest is not None:
        unknown_variant_cases = sorted(
            {
                variant["case_id"]
                for variant in experiment_manifest["variants"]
                if variant["case_id"] not in seen
            }
        )
        if unknown_variant_cases:
            raise ValueError(
                "experiment variants reference unknown cases: "
                f"{unknown_variant_cases}"
            )
        variant_groups = {
            cases_by_id[variant["case_id"]]["comparison_group"]
            for variant in experiment_manifest["variants"]
        }
        if len(variant_groups) != 1:
            raise ValueError(
                "experiment variants must share one comparison_group"
            )
        unresolved_variant_provenance = sorted(
            {
                variant["provenance_ref"]
                for variant in experiment_manifest["variants"]
                if variant["provenance_ref"] not in provenance_catalog
            }
        )
        if unresolved_variant_provenance:
            raise ValueError(
                "experiment variants have unresolved provenance refs: "
                f"{unresolved_variant_provenance}"
            )
    return {
        **dict(bundle),
        "workload": dict(workload),
        "source": dict(source),
        "expected_world_size": expected_world_size,
        "world_size_source": world_size_source,
        "metric_definitions": metric_definitions,
        "cases": normalized_cases,
        "route_comparisons": normalized_comparisons,
        "measurement_tracks": tracks,
        "artifacts": dict(artifacts),
        "provenance_catalog": provenance_catalog,
        "derived_evidence": derived_evidence,
        **(
            {"hardware_context": hardware_context}
            if hardware_context is not None
            else {}
        ),
        **(
            {"experiment_manifest": experiment_manifest}
            if experiment_manifest is not None
            else {}
        ),
    }
