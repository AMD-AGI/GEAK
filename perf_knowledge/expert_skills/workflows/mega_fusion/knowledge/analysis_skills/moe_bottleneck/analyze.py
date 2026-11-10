#!/usr/bin/env python3
"""Deterministic measurement-first MoE analysis generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from multi_rank_analysis import (  # noqa: E402
    critical_path,
    resolve_measurement_tracks,
    validate_evidence_catalog,
    validate_hardware_context,
)

SCHEMA_VERSION = "geak-moe-bottleneck-analysis-v4"
REQUIRED_TRACKS = (
    "ep_baseline_decomposition",
    "communication_bytes",
    "wait_padding",
    "publication_granularity",
    "fusion_dag",
    "resource_residency",
)
REQUIRED_EVIDENCE_FIELDS = (
    "artifact_refs",
    "metrics",
    "provenance_refs",
)
TRACK_REQUIRED_KINDS = {
    "ep_baseline_decomposition": ({"rank_metric"}, {"trace"}),
    "communication_bytes": ({"software_counters"}, {"xgmi"}),
    "wait_padding": ({"software_counters"},),
    "publication_granularity": ({"controlled_experiment"},),
    "fusion_dag": ({"dependency_dag"},),
    "resource_residency": ({"resource_residency"},),
}
CATEGORY_LABELS = {
    "stage1": "stage1_dispatch_gemm1",
    "stage1_dispatch_gemm1": "stage1_dispatch_gemm1",
    "stage2": "stage2",
    "combine": "combine",
    "quantize": "quantize",
    "pre_dispatch_quant": "pre_dispatch_quant",
}
REFERENCE_PATTERNS = [
    {
        "name": "DeepGEMM Mega MoE",
        "mechanism": "symmetric workspace, arrival counts, expert-wave persistent scheduler",
        "applicability": "dependency and scheduler reference",
        "caveat": "Blackwell SM100 implementation details do not transfer to gfx950",
        "source": "https://github.com/deepseek-ai/DeepGEMM",
    },
    {
        "name": "DeepEP SBO",
        "mechanism": "block-granular producer signals consumed by combine send",
        "applicability": "readiness/publication protocol reference",
        "caveat": "CUDA/NVSHMEM memory ordering must be revalidated on MORI/gfx950",
        "source": "https://github.com/deepseek-ai/DeepEP/pull/483",
    },
    {
        "name": "Comet / Flux",
        "mechanism": "shared-tensor decomposition, work reordering, block specialization",
        "applicability": "candidate pattern catalog",
        "caveat": "CUTLASS/NVSHMEM implementation and speedups are not local evidence",
        "source": "https://arxiv.org/abs/2502.19811",
    },
    {
        "name": "tile-level producer/consumer overlap",
        "mechanism": "persistent GEMM producer, communication consumer, epilogue signals",
        "applicability": "candidate protocol experiment",
        "caveat": "A100 evidence; no dedicated public ROCm implementation identified",
        "source": "https://arxiv.org/abs/2607.19539",
    },
]


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


def _finite(value, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _case_map(report: dict) -> dict:
    cases = report.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("report.cases must be a non-empty list")
    result = {}
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every case must have a non-empty case_id")
        if case_id in result:
            raise ValueError(f"duplicate case_id={case_id!r}")
        result[case_id] = case
    return result


def _comparison_evidence(
    comparison: dict,
    default_timing_confidence: str,
) -> dict:
    if comparison.get("status") == "incomplete":
        return {
            "status": "incomplete",
            "baseline_case_id": comparison.get("baseline_case_id"),
            "candidate_case_id": comparison.get("candidate_case_id"),
            "comparison_group": comparison.get("comparison_group"),
            "metric": dict(comparison.get("metric") or {}),
            "incomplete_reasons": list(
                comparison.get("incomplete_reasons") or []
            ),
            "categories": {},
        }
    tokens = comparison.get("tokens_per_rank")
    timing_confidence = comparison.get(
        "timing_confidence",
        default_timing_confidence,
    )
    if timing_confidence not in ("high", "medium", "low"):
        timing_confidence = "low"
    profile_confidence = comparison.get(
        "profile_confidence",
        default_timing_confidence,
    )
    if profile_confidence not in ("high", "medium", "low"):
        profile_confidence = "low"
    profile = comparison.get("profile_category_delta") or {}
    categories = {}
    positive_total = 0.0
    for category, values in profile.items():
        delta = _finite(
            values.get("rank_max_delta_ms"),
            f"{category}.rank_max_delta_ms",
        )
        relative = _finite(
            values.get("rank_max_delta_pct"),
            f"{category}.rank_max_delta_pct",
        )
        positive_total += max(delta, 0.0)
        categories[category] = {
            "label": CATEGORY_LABELS.get(category, category),
            "baseline_rank_max_ms": values.get("baseline_rank_max"),
            "candidate_rank_max_ms": values.get("candidate_rank_max"),
            "absolute_delta_ms": delta,
            "relative_growth_pct": relative,
        }
    for values in categories.values():
        values["positive_absolute_delta_share_pct"] = (
            max(values["absolute_delta_ms"], 0.0) / positive_total * 100.0
            if positive_total
            else 0.0
        )
    if "metric" in comparison:
        metric = dict(comparison["metric"])
        if (
            metric.get("semantic") != "e2e_latency"
            or metric.get("unit") != "ms"
            or metric.get("reduction") != "rank_max"
        ):
            raise ValueError(
                "MoE route comparison requires rank-max e2e_latency in ms"
            )
        baseline = dict(comparison.get("baseline") or {})
        candidate = dict(comparison.get("candidate") or {})
        delta_ms = _finite(comparison.get("delta"), "delta")
        delta_pct = _finite(comparison.get("delta_pct"), "delta_pct")
        expected_delta = _finite(candidate.get("rank_max"), "candidate.rank_max") - _finite(
            baseline.get("rank_max"),
            "baseline.rank_max",
        )
        if not math.isclose(delta_ms, expected_delta, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("route comparison delta is inconsistent with case values")
        return {
            "status": "complete",
            "baseline_case_id": comparison.get("baseline_case_id"),
            "candidate_case_id": comparison.get("candidate_case_id"),
            "comparison_group": comparison.get("comparison_group"),
            "tokens_per_rank": tokens,
            "metric": metric,
            "baseline": baseline,
            "candidate": candidate,
            "e2e_rank_max_delta_ms": delta_ms,
            "e2e_rank_max_delta_pct": delta_pct,
            "categories": categories,
            "timing_confidence": timing_confidence,
            "profile_confidence": profile_confidence,
            "stage_deltas_non_additive": True,
        }
    return {
        "status": "complete",
        "baseline_case_id": comparison.get("baseline_case_id", "legacy_unknown"),
        "candidate_case_id": comparison.get(
            "candidate_case_id",
            "legacy_unknown",
        ),
        "comparison_group": comparison.get("comparison_group", "legacy_unknown"),
        "tokens_per_rank": tokens,
        "metric": {
            "path": "legacy.e2e_rank_max",
            "unit": "ms",
            "direction": "lower",
            "reduction": "rank_max",
            "semantic": "e2e_latency",
        },
        "e2e_rank_max_delta_ms": _finite(
            comparison.get("e2e_rank_max_delta_ms", 0.0),
            "e2e_rank_max_delta_ms",
        ),
        "e2e_rank_max_delta_pct": _finite(
            comparison.get("e2e_rank_max_delta_pct", 0.0),
            "e2e_rank_max_delta_pct",
        ),
        "categories": categories,
        "timing_confidence": timing_confidence,
        "profile_confidence": profile_confidence,
        "stage_deltas_non_additive": True,
    }


def _validate_bounds(bounds) -> list[dict]:
    if not isinstance(bounds, list) or not bounds:
        raise ValueError("fusion_dag bounds must be a non-empty list")
    validated = []
    for index, bound in enumerate(bounds):
        if not isinstance(bound, dict):
            raise ValueError(f"fusion bound {index} must be a mapping")
        name = bound.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"fusion bound {index} requires name")
        baseline_ms = _finite(bound.get("baseline_ms"), f"bound {name}.baseline_ms")
        lower_bound_ms = _finite(
            bound.get("lower_bound_ms"),
            f"bound {name}.lower_bound_ms",
        )
        ceiling_speedup = _finite(
            bound.get("ceiling_speedup"),
            f"bound {name}.ceiling_speedup",
        )
        if baseline_ms <= 0 or lower_bound_ms <= 0 or ceiling_speedup <= 0:
            raise ValueError(f"fusion bound {name!r} values must be positive")
        expected_speedup = baseline_ms / lower_bound_ms
        if not math.isclose(
            ceiling_speedup,
            expected_speedup,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"fusion bound {name!r} speedup is inconsistent"
            )
        assumptions = bound.get("assumptions")
        provenance_refs = bound.get("provenance_refs")
        if not isinstance(assumptions, list) or not assumptions:
            raise ValueError(f"fusion bound {name!r} requires assumptions")
        if not isinstance(provenance_refs, list) or not provenance_refs:
            raise ValueError(
                f"fusion bound {name!r} requires provenance_refs"
            )
        validated.append(
            {
                **dict(bound),
                "unit": "ms",
                "baseline_ms": baseline_ms,
                "lower_bound_ms": lower_bound_ms,
                "ceiling_speedup": ceiling_speedup,
            }
        )
    return validated


def _validate_dependency_dag(data: dict) -> None:
    if not isinstance(data.get("nodes"), dict) or not isinstance(
        data.get("edges"),
        list,
    ):
        raise ValueError("dependency_dag requires nodes and edges")
    measured = critical_path(data["nodes"], data["edges"])
    declared = data.get("critical_path")
    if not isinstance(declared, dict):
        raise ValueError("dependency_dag requires critical_path")
    if declared.get("path") != measured["path"] or not math.isclose(
        _finite(declared.get("critical_path_ms"), "critical_path_ms"),
        measured["critical_path_ms"],
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("dependency_dag critical_path is inconsistent")


def _validate_xgmi(data: dict) -> None:
    values = {}
    for field in (
        "logical_bytes",
        "physical_bytes",
        "duration_ms",
        "effective_gbps",
        "ceiling_gbps",
        "utilization_pct",
    ):
        values[field] = _finite(data.get(field), f"xgmi.{field}")
        if values[field] < 0:
            raise ValueError(f"xgmi.{field} must be non-negative")
    if values["duration_ms"] <= 0 or values["ceiling_gbps"] <= 0:
        raise ValueError("xgmi duration and ceiling must be positive")
    expected_gbps = (
        values["physical_bytes"] * 8.0 / (values["duration_ms"] * 1e6)
    )
    if not math.isclose(
        values["effective_gbps"],
        expected_gbps,
        rel_tol=1e-6,
        abs_tol=1e-9,
    ):
        raise ValueError("xgmi effective_gbps is inconsistent")
    expected_utilization = (
        values["effective_gbps"] / values["ceiling_gbps"] * 100.0
    )
    if not math.isclose(
        values["utilization_pct"],
        expected_utilization,
        rel_tol=1e-6,
        abs_tol=1e-9,
    ):
        raise ValueError("xgmi utilization_pct is inconsistent")


def _validate_residency(data: dict) -> None:
    for field in (
        "workgroup_size",
        "lds_bytes_per_workgroup",
        "vgpr_per_thread",
        "resident_workgroups_per_cu",
    ):
        value = data.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"resource_residency requires positive {field}")
    liveness = data.get("liveness")
    if not isinstance(liveness, dict) or not all(
        liveness.get(field) is True
        for field in (
            "producer_progress",
            "consumer_progress",
            "termination_proven",
        )
    ):
        raise ValueError("resource_residency liveness is not proven")


def _validate_publication_experiment(data: dict) -> None:
    variants = data.get("variants")
    if not isinstance(variants, list) or len(variants) < 3:
        raise ValueError(
            "publication experiment requires full plus at least two variants"
        )
    publication_variants = [
        variant
        for variant in variants
        if variant.get("name") != "full"
        and "publication_granularity"
        in variant.get("changed_components", [])
    ]
    if len(publication_variants) < 2:
        raise ValueError(
            "publication experiment requires two granularity variants"
        )


def _measurement_coverage(report: dict) -> tuple[dict, list[str], dict]:
    raw = report.get("measurement_tracks") or {}
    catalog_payload = report.get("evidence_catalog")
    catalog = None
    if catalog_payload is not None:
        catalog = validate_evidence_catalog(catalog_payload)
        raw = resolve_measurement_tracks(raw, catalog)
    coverage = {}
    missing = []
    for track in REQUIRED_TRACKS:
        value = raw.get(track)
        status = value.get("status") if isinstance(value, dict) else value
        evidence = value.get("evidence") if isinstance(value, dict) else None
        reason = None
        if status not in ("complete", "partial", "missing"):
            reason = "status must be complete, partial, or missing"
            status = "invalid" if value is not None else "missing"
        if status == "complete" and not evidence:
            reason = "complete requires non-empty evidence"
            status = "invalid"
        if status == "complete":
            if not isinstance(evidence, dict):
                reason = "complete evidence must be a mapping"
                status = "invalid"
            else:
                absent = [
                    field
                    for field in REQUIRED_EVIDENCE_FIELDS
                    if not evidence.get(field)
                ]
                if absent:
                    reason = (
                        "complete evidence is missing required fields: "
                        + ", ".join(absent)
                    )
                    status = "invalid"
                elif track == "fusion_dag" and not evidence.get("bounds"):
                    reason = "complete fusion_dag evidence requires bounds"
                    status = "invalid"
        if status == "complete" and catalog is None:
            reason = "complete evidence requires a validated evidence_catalog"
            status = "invalid"
        if status == "complete" and not value.get("resolved_evidence"):
            reason = "complete evidence references were not resolved"
            status = "invalid"
        if status == "complete":
            artifact_refs = value["resolved_evidence"]["artifacts"]
            kinds = {
                catalog["entries"][evidence_id]["kind"]
                for evidence_id in artifact_refs
            }
            missing_kind_groups = [
                sorted(group)
                for group in TRACK_REQUIRED_KINDS[track]
                if not (kinds & group)
            ]
            if missing_kind_groups:
                reason = (
                    "complete evidence is missing required evidence kinds: "
                    f"{missing_kind_groups}"
                )
                status = "invalid"
        if status == "complete":
            try:
                for evidence_id in value["resolved_evidence"]["artifacts"]:
                    entry = catalog["entries"][evidence_id]
                    data = entry.get("data") or {}
                    if entry["kind"] == "dependency_dag":
                        _validate_dependency_dag(data)
                    elif entry["kind"] == "xgmi":
                        _validate_xgmi(data)
                    elif entry["kind"] == "resource_residency":
                        _validate_residency(data)
                    elif (
                        track == "publication_granularity"
                        and entry["kind"] == "controlled_experiment"
                    ):
                        _validate_publication_experiment(data)
            except (TypeError, ValueError) as error:
                reason = str(error)
                status = "invalid"
        if status == "complete" and track == "fusion_dag":
            try:
                validated_bounds = _validate_bounds(evidence["bounds"])
                known_provenance = {
                    provenance_id
                    for provenance_id, provenance in catalog[
                        "provenance"
                    ].items()
                    if provenance["status"] == "complete"
                }
                unresolved_bound_provenance = sorted(
                    {
                        provenance_ref
                        for bound in validated_bounds
                        for provenance_ref in bound["provenance_refs"]
                        if provenance_ref not in known_provenance
                    }
                )
                if unresolved_bound_provenance:
                    raise ValueError(
                        "fusion bounds have unresolved provenance: "
                        f"{unresolved_bound_provenance}"
                    )
                evidence = {
                    **dict(evidence),
                    "bounds": validated_bounds,
                }
            except (TypeError, ValueError) as error:
                reason = str(error)
                status = "invalid"
        coverage[track] = {"status": status, "evidence": evidence}
        if reason:
            coverage[track]["reason"] = reason
        if status != "complete":
            missing.append(track)
    return coverage, missing, catalog or {
        "schema_version": "missing",
        "entries": {},
    }


def _hardware_guidance(context: dict | None) -> dict:
    if context is None:
        return {
            "status": "missing",
            "required": (
                "device/topology/runtime/capability context and measured hardware ceilings"
            ),
            "constraints": [],
        }
    validated = validate_hardware_context(context)
    measured = validated.get("measured", {})
    constraints = [
        (
            f"Interpret evidence for {validated['vendor']} {validated['model']} "
            f"({validated['arch']}), not for a reference system's hardware."
        ),
        (
            f"Report residency bounds with {validated['execution_units_per_device']} execution "
            f"units/device, thread-group width {validated['thread_group_width']}, and "
            f"{validated['local_memory_bytes_per_execution_unit']} bytes local memory/unit."
        ),
        (
            f"Interpret communication against {validated['interconnect']['type']} "
            f"{validated['interconnect']['topology']} topology."
        ),
    ]
    missing_ceilings = [
        name
        for name in (
            "pairwise_interconnect_gbps",
            "all_to_all_interconnect_gbps",
            "device_memory_gbps",
            "launch_overhead_us",
        )
        if measured.get(name) is None
    ]
    provenance = validated.get("provenance", {})
    low_confidence_fields = sorted(
        field
        for field, entry in provenance.items()
        if entry.get("confidence") == "low"
    )
    medium_confidence_fields = sorted(
        field
        for field, entry in provenance.items()
        if entry.get("confidence") == "medium"
    )
    return {
        "status": (
            "ready"
            if not missing_ceilings and not low_confidence_fields
            else "partial"
        ),
        "context": validated,
        "missing_measured_ceilings": missing_ceilings,
        "low_confidence_fields": low_confidence_fields,
        "medium_confidence_fields": medium_confidence_fields,
        "constraints": constraints,
    }


def _measurement_unknowns(missing: list[str]) -> list[dict]:
    descriptions = {
        "ep_baseline_decomposition": (
            "Decompose the frozen scattered EP kernel into quant, dispatch/planning, "
            "GEMM1, activation, GEMM2/publication, combine, launch/materialization, "
            "rank wait, and physical overlap."
        ),
        "communication_bytes": (
            "Collect per-source/destination payload and metadata bytes, XGMI samples, "
            "and full/no-payload/local-loopback controls."
        ),
        "wait_padding": (
            "Instrument per-expert/tile readiness, local wait cycles, useful rows, "
            "padded rows, and the per-rank expert-load matrix."
        ),
        "publication_granularity": (
            "Sweep token, owner-aligned row band, GEMM tile, and chunk publication "
            "granularity with signal/fence and XGMI efficiency."
        ),
        "fusion_dag": (
            "Build the measured dependency DAG and Amdahl ceiling for replacing the "
            "scattered launches with one complete persistent kernel per rank."
        ),
        "resource_residency": (
            "Measure common workgroup shape, LDS/VGPR union, resident CTA count, "
            "role liveness, scheduler-state pressure, and termination for the one-kernel target."
        ),
    }
    return [
        {
            "track": track,
            "question": descriptions[track],
            "required_data": f"measurement_tracks.{track}.evidence",
            "blocks": ["root_cause", "candidate_ranking"],
        }
        for track in missing
    ]


def _sum_numeric(value) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, dict):
        return sum(_sum_numeric(item) for item in value.values())
    if isinstance(value, list):
        return sum(_sum_numeric(item) for item in value)
    return 0.0


def _catalog_findings(catalog: dict) -> list[dict]:
    findings = []
    for evidence_id, entry in catalog.get("entries", {}).items():
        kind = entry["kind"]
        if (
            entry.get("status") != "complete"
            and kind not in (
                "publication_sweep",
                "resource_residency",
                "dependency_dag",
            )
        ):
            continue
        data = entry.get("data") or {}
        if kind == "att":
            categories = data.get("categories", {})
            if not categories:
                continue
            leader, metrics = max(
                categories.items(),
                key=lambda item: item[1].get(
                    "accounted_cycle_share_pct",
                    0.0,
                ),
            )
            att_label = Path(data.get("source", evidence_id)).parent.name
            findings.append(
                {
                    "observation": (
                        f"ATT sample {att_label} attributes the largest "
                        f"accounted-cycle share to {leader}."
                    ),
                    "evidence": {
                        "evidence_id": evidence_id,
                        "kernel_scope": att_label,
                        "category": leader,
                        "accounted_cycle_share_pct": metrics.get(
                            "accounted_cycle_share_pct"
                        ),
                        "stall_within_latency_pct": metrics.get(
                            "stall_within_latency_pct"
                        ),
                    },
                    "confidence": entry.get("confidence", "low"),
                    "scope": entry.get(
                        "scope",
                        data.get("scope_warning", "sampled ATT scope"),
                    ),
                }
            )
        elif kind == "trace":
            categories = data.get("categories", {})
            complete_categories = {
                path: metric
                for path, metric in categories.items()
                if metric.get("rank_max") is not None
                and not metric.get("missing_ranks")
            }
            if not complete_categories:
                continue
            leader, metric = max(
                complete_categories.items(),
                key=lambda item: item[1]["rank_max"],
            )
            findings.append(
                {
                    "observation": (
                        f"Normalized trace {evidence_id} has the largest "
                        f"rank-max category duration in {leader}."
                    ),
                    "evidence": {
                        "evidence_id": evidence_id,
                        "category": leader,
                        "rank_max_ms_per_replay": metric["rank_max"],
                        "replay_count": data.get("replay_count"),
                    },
                    "confidence": entry.get("confidence", "low"),
                    "scope": entry.get(
                        "scope",
                        data.get(
                            "duration_semantics",
                            "categorized kernels only",
                        ),
                    ),
                }
            )
        elif kind == "software_counters":
            records = data.get("records", [])
            payload_bytes = sum(
                _sum_numeric(record.get("payload_bytes_by_peer", {}))
                for record in records
            )
            wait_cycles = sum(
                _sum_numeric(record.get("wait_cycles_by_edge", {}))
                for record in records
            )
            useful_rows = sum(
                _sum_numeric(record.get("useful_rows_by_expert", {}))
                for record in records
            )
            padded_rows = sum(
                _sum_numeric(record.get("padded_rows_by_expert", {}))
                for record in records
            )
            findings.append(
                {
                    "observation": (
                        f"Software counters {evidence_id} provide exact "
                        "logical byte, wait, and padding totals for their scope."
                    ),
                    "evidence": {
                        "evidence_id": evidence_id,
                        "payload_bytes": payload_bytes,
                        "wait_cycles": wait_cycles,
                        "useful_rows": useful_rows,
                        "padded_rows": padded_rows,
                    },
                    "confidence": entry.get("confidence", "low"),
                    "scope": entry.get("scope", "counter records"),
                }
            )
        elif kind == "controlled_experiment":
            findings.append(
                {
                    "observation": (
                        f"Controlled experiment {evidence_id} contains "
                        f"{len(data.get('variants', []))} variants; deltas "
                        "remain non-additive."
                    ),
                    "evidence": {
                        "evidence_id": evidence_id,
                        "overlap_pairs": data.get("overlap_pairs", []),
                        "delta_additivity_allowed": data.get(
                            "delta_additivity_allowed",
                            False,
                        ),
                    },
                    "confidence": "medium",
                    "scope": "declared controlled-experiment invariants",
                }
            )
        elif kind == "route_derived_counts":
            for case_id, case_data in (data.get("cases") or {}).items():
                findings.append(
                    {
                        "observation": (
                            f"Route-derived evidence for {case_id} records "
                            f"{case_data['remote_route_fraction_pct']:.2f}% "
                            "remote routes and explicit GEMM padding."
                        ),
                        "evidence": {
                            "evidence_id": evidence_id,
                            "case_id": case_id,
                            "remote_routes": case_data["remote_routes"],
                            "remote_route_fraction_pct": case_data[
                                "remote_route_fraction_pct"
                            ],
                            "total_remote_payload_bytes": case_data[
                                "total_remote_payload_bytes"
                            ],
                            "stage1_padding_pct": case_data[
                                "stage1_padding_pct"
                            ],
                            "stage2_padding_pct": case_data[
                                "stage2_padding_pct"
                            ],
                        },
                        "confidence": entry.get("confidence", "medium"),
                        "scope": entry.get(
                            "scope",
                            "analytical route-derived lower bound",
                        ),
                    }
                )
        elif kind == "hardware_ceiling":
            findings.append(
                {
                    "observation": (
                        "Target-local launch, HBM, pairwise, and all-to-all "
                        "ceilings were measured."
                    ),
                    "evidence": {
                        "evidence_id": evidence_id,
                        "measurements": data.get("measurements", {}),
                        "units": data.get("units", {}),
                    },
                    "confidence": entry.get("confidence", "high"),
                    "scope": entry.get("scope", "target-local TransferBench"),
                }
            )
        elif kind == "publication_sweep":
            variants = data.get("variants", [])
            if variants:
                best = min(
                    variants,
                    key=lambda item: item["rank_max_ms"]["e2e"],
                )
                worst = max(
                    variants,
                    key=lambda item: item["rank_max_ms"]["e2e"],
                )
                findings.append(
                    {
                        "observation": (
                            f"Single-run publication sweep ranges from "
                            f"{best['variant']['name']}={best['rank_max_ms']['e2e']:.4f} ms "
                            f"to {worst['variant']['name']}={worst['rank_max_ms']['e2e']:.4f} ms."
                        ),
                        "evidence": {
                            "evidence_id": evidence_id,
                            "variants": variants,
                        },
                        "confidence": "low",
                        "scope": entry.get(
                            "scope",
                            "single-run publication sweep",
                        ),
                    }
                )
        elif kind == "liveness":
            findings.append(
                {
                    "observation": (
                        f"Liveness stress {evidence_id} completed "
                        f"{data.get('workload', {}).get('cuda_graph_replays')} "
                        "CUDA Graph replays without observed timeout/deadlock."
                    ),
                    "evidence": {
                        "evidence_id": evidence_id,
                        "result": data.get("result", {}),
                    },
                    "confidence": entry.get("confidence", "high"),
                    "scope": entry.get("scope", "current scheduler only"),
                }
            )
        elif kind in (
            "xgmi",
            "dependency_dag",
            "resource_residency",
            "xgmi_firmware_accumulator",
            "gmi_sector_counters",
            "occupancy_counters",
            "att_occupancy",
            "att_wait",
            "combine_wait_timing",
            "payload_control",
            "tile_dependency_dag",
        ):
            findings.append(
                {
                    "observation": (
                        f"Resolved {kind} evidence {evidence_id} is available "
                        "for bounded interpretation."
                    ),
                    "evidence": {
                        "evidence_id": evidence_id,
                        "data": data,
                    },
                    "confidence": entry.get("confidence", "medium"),
                    "scope": entry.get("scope", kind),
                }
            )
    return findings


def build_analysis(report: dict) -> dict:
    cases_by_id = _case_map(report)
    source_schema = report.get("schema_version", "unknown")
    generic_report = source_schema == "geak-multirank-analysis-v2"
    report_status = report.get("status", "legacy_unknown")
    if generic_report and report_status not in ("pass", "partial", "fail"):
        raise ValueError("generic report has invalid status")
    report_inputs = report.get("input_artifacts")
    upstream_identity_available = (
        isinstance(report_inputs, dict)
        and bool(report_inputs)
        and all(
            isinstance(identity, dict) and bool(identity.get("sha256"))
            for identity in report_inputs.values()
        )
    )
    source_provenance = {
        "status": (
            "upstream_input_identity_available"
            if upstream_identity_available
            else (
                "upstream_input_identity_incomplete"
                if isinstance(report_inputs, dict) and report_inputs
                else "legacy_upstream_identity_missing"
            )
        ),
        "input_artifacts": (
            dict(report_inputs)
            if isinstance(report_inputs, dict)
            else {}
        ),
        "raw_artifact_identities": dict(
            report.get("raw_artifact_identities") or {}
        ),
    }
    default_timing_confidence = (
        "medium"
        if (
            source_provenance["status"]
            == "upstream_input_identity_available"
            and report_status == "pass"
        )
        else "low"
    )
    comparisons = []
    for raw_comparison in report.get("route_comparisons", []):
        comparison = dict(raw_comparison)
        if (
            "baseline_case_id" not in comparison
            and comparison.get("tokens_per_rank") is not None
        ):
            prefix = f"t{comparison['tokens_per_rank']}_"
            matching = [
                case_id
                for case_id in cases_by_id
                if case_id.startswith(prefix)
            ]
            uniform = [
                case_id for case_id in matching if "uniform" in case_id
            ]
            skew = [
                case_id for case_id in matching if "skew" in case_id
            ]
            if len(uniform) == 1 and len(skew) == 1:
                comparison["baseline_case_id"] = uniform[0]
                comparison["candidate_case_id"] = skew[0]
                comparison["comparison_group"] = prefix.rstrip("_")
        comparisons.append(
            _comparison_evidence(comparison, default_timing_confidence)
        )
    coverage, missing, evidence_catalog = _measurement_coverage(report)
    hardware = _hardware_guidance(report.get("hardware_context"))
    unknowns = _measurement_unknowns(missing)
    if hardware["status"] != "ready":
        unknowns.insert(
            0,
            {
                "track": "hardware_context",
                "question": (
                    "Complete target-hardware provenance and measured ceilings before "
                    "interpreting utilization or hardware-sensitive reference patterns."
                ),
                "required_data": hardware.get("missing_measured_ceilings", []),
                "blocks": ["utilization_interpretation", "candidate_applicability"],
            },
        )
    findings = _catalog_findings(evidence_catalog)
    hypotheses = []
    for comparison in comparisons:
        if comparison["status"] != "complete":
            unknowns.append(
                {
                    "track": "route_comparison",
                    "question": (
                        "Repair incomplete rank/comparison evidence before "
                        f"interpreting {comparison.get('baseline_case_id')}→"
                        f"{comparison.get('candidate_case_id')}."
                    ),
                    "required_data": comparison.get("incomplete_reasons", []),
                    "blocks": ["route_sensitivity", "candidate_ranking"],
                }
            )
            continue
        token_scope = (
            f"{comparison['tokens_per_rank']} tokens/rank"
            if comparison["tokens_per_rank"] is not None
            else "an unspecified token count"
        )
        findings.append(
            {
                "observation": (
                    f"At {token_scope}, "
                    f"{comparison['baseline_case_id']}→"
                    f"{comparison['candidate_case_id']} "
                    f"changes E2E rank-max by {comparison['e2e_rank_max_delta_pct']:.3f}%."
                ),
                "evidence": {
                    "baseline_case_id": comparison["baseline_case_id"],
                    "candidate_case_id": comparison["candidate_case_id"],
                    "metric": comparison["metric"],
                    "baseline": comparison.get("baseline"),
                    "candidate": comparison.get("candidate"),
                    "tokens_per_rank": comparison["tokens_per_rank"],
                    "e2e_rank_max_delta_ms": comparison["e2e_rank_max_delta_ms"],
                    "e2e_rank_max_delta_pct": comparison["e2e_rank_max_delta_pct"],
                },
                "confidence": comparison["timing_confidence"],
                "scope": "tested route-case pair only",
            }
        )
        categories = comparison["categories"]
        if not categories:
            continue
        positive_categories = {
            category: values
            for category, values in categories.items()
            if values["absolute_delta_ms"] > 0.0
        }
        if not positive_categories:
            continue
        largest_absolute = max(
            positive_categories.items(),
            key=lambda item: item[1]["positive_absolute_delta_share_pct"],
        )
        largest_relative = max(
            positive_categories.items(),
            key=lambda item: item[1]["relative_growth_pct"],
        )
        findings.append(
            {
                "observation": (
                    f"At {token_scope}, "
                    f"{largest_absolute[1]['label']} has the largest positive absolute "
                    f"category-delta share, while {largest_relative[1]['label']} has the "
                    "largest relative growth."
                ),
                "evidence": {
                    "largest_absolute_category": largest_absolute[1]["label"],
                    "positive_absolute_delta_share_pct": largest_absolute[1][
                        "positive_absolute_delta_share_pct"
                    ],
                    "relative_growth_leader": largest_relative[1]["label"],
                    "relative_growth_pct": largest_relative[1]["relative_growth_pct"],
                    "e2e_rank_max_delta_pct": comparison["e2e_rank_max_delta_pct"],
                },
                "confidence": comparison["profile_confidence"],
                "scope": "profiler categories; not critical-path contribution",
            }
        )
        hypotheses.append(
            {
                "statement": (
                    f"Internal work or synchronization within "
                    f"{largest_absolute[1]['label']} may contribute materially to the "
                    "observed route sensitivity."
                ),
                "supports": findings[-1]["evidence"],
                "does_not_prove": [
                    "root cause",
                    "dispatch cost when fused with GEMM1",
                    "a preferred fusion or scheduling mechanism",
                ],
                "discriminating_measurement": (
                    "instrument internal readiness, wait, bytes, instruction stalls, "
                    "and physical overlap for this category"
                ),
                "confidence": min(
                    comparison["profile_confidence"],
                    "medium",
                    key=("low", "medium", "high").index,
                ),
            }
        )
    constraints = list(hardware.get("constraints", []))
    constraints.extend(
        [
            "Stage1 timing combines dispatch/planning/P2P publication with GEMM1.",
            "Separately captured category deltas are non-additive when execution overlaps.",
            "Public implementation mechanisms are references, not selected solutions.",
        ]
    )
    fusion_track = coverage.get("fusion_dag")
    bounds = []
    if isinstance(fusion_track, dict) and fusion_track.get("status") == "complete":
        evidence = fusion_track.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("bounds"), list):
            bounds = list(evidence["bounds"])
    degraded = []
    if missing or hardware["status"] != "ready" or report_status != "pass":
        degraded.append(
            {
                "reason": "analysis prerequisites are incomplete",
                "missing_tracks": missing,
                "hardware_context_status": hardware["status"],
                "source_report_status": report_status,
                "effect": "no high-confidence fusion or root-cause verdict is allowed",
            }
        )
    dispatch_metrics = []
    ep_track = coverage.get("ep_baseline_decomposition", {})
    if ep_track.get("status") == "complete":
        dispatch_metrics = [
            metric
            for metric in ep_track["evidence"].get("metrics", [])
            if "dispatch" in metric.lower()
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "framework_status": "ready",
        "analysis_boundary": {
            "workflow_step": 2,
            "emits": [
                "findings",
                "hypotheses",
                "constraints",
                "bounds",
                "unknowns",
                "reference_patterns",
            ],
            "decision_fields_emitted": False,
            "decision_owner": "Step-3 TechLead",
        },
        "analysis_status": (
            "awaiting_measurement"
            if (
                missing
                or hardware["status"] != "ready"
                or report_status != "pass"
                or not generic_report
            )
            else "evidence_complete"
        ),
        "source_schema": source_schema,
        "source_report_status": report_status,
        "source_provenance": source_provenance,
        "hardware_guidance": hardware,
        "measurement_coverage": coverage,
        "route_comparisons": comparisons,
        "findings": findings,
        "hypotheses": hypotheses,
        "constraints": constraints,
        "bounds": bounds,
        "unknowns": unknowns,
        "reference_patterns": list(REFERENCE_PATTERNS),
        "evidence_catalog_summary": {
            evidence_id: {
                "kind": entry["kind"],
                "status": entry["status"],
                "metric_ids": list(entry.get("metric_ids", [])),
                "provenance_refs": list(
                    entry.get("provenance_refs", [])
                ),
            }
            for evidence_id, entry in evidence_catalog.get("entries", {}).items()
        },
        "degraded": degraded,
        "claims": {
            "root_cause_proven": False,
            "dispatch_independently_measured": bool(dispatch_metrics),
            "dispatch_metric_refs": dispatch_metrics,
        },
    }


def validate_analysis_output(analysis: dict) -> dict:
    if not isinstance(analysis, dict):
        raise TypeError("analysis output must be a mapping")
    if analysis.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"analysis output must use schema {SCHEMA_VERSION}"
        )
    if analysis.get("framework_status") != "ready":
        raise ValueError("framework_status must be ready")
    if analysis.get("analysis_status") not in (
        "awaiting_measurement",
        "evidence_complete",
    ):
        raise ValueError("analysis_status is invalid")
    boundary = analysis.get("analysis_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("workflow_step") != 2
        or boundary.get("decision_fields_emitted") is not False
        or boundary.get("decision_owner") != "Step-3 TechLead"
    ):
        raise ValueError("analysis boundary is invalid")

    def walk(value, path="root"):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("directions", "specialty"):
                    raise ValueError(
                        f"decision field {key!r} is forbidden at {path}"
                    )
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite number at {path}")

    walk(analysis)
    for field in (
        "findings",
        "hypotheses",
        "constraints",
        "bounds",
        "unknowns",
        "reference_patterns",
        "degraded",
    ):
        if not isinstance(analysis.get(field), list):
            raise ValueError(f"analysis field {field} must be a list")
    if analysis["analysis_status"] == "evidence_complete":
        incomplete_tracks = [
            name
            for name, track in analysis["measurement_coverage"].items()
            if track.get("status") != "complete"
        ]
        if incomplete_tracks:
            raise ValueError(
                f"evidence_complete has incomplete tracks: {incomplete_tracks}"
            )
        if analysis.get("source_report_status") != "pass":
            raise ValueError("evidence_complete requires source report pass")
        if analysis.get("hardware_guidance", {}).get("status") != "ready":
            raise ValueError("evidence_complete requires ready hardware context")
        if analysis.get("degraded"):
            raise ValueError("evidence_complete must not be degraded")
    return analysis


def _render_markdown(analysis: dict) -> str:
    lines = [
        "# MoE bottleneck analysis",
        "",
        f"Framework status: `{analysis['framework_status']}`",
        f"Analysis status: `{analysis['analysis_status']}`",
        "",
        "## Source provenance",
        "",
        f"- upstream report provenance: {analysis['source_provenance']['status']}",
    ]
    for name, identity in analysis.get("analysis_inputs", {}).items():
        lines.append(
            f"- {name}: `{identity['path']}` / sha256 `{identity['sha256']}`"
        )
    lines.extend(["", "## Hardware context", ""])
    hardware = analysis["hardware_guidance"]
    lines.append(f"- status: {hardware['status']}")
    context = hardware.get("context")
    if context:
        lines.append(
            f"- target: {context['vendor']} {context['model']} / {context['arch']} / "
            f"{context['device_count']} devices"
        )
        for constraint in hardware["constraints"]:
            lines.append(f"- {constraint}")
    if hardware.get("missing_measured_ceilings"):
        lines.append(
            "- missing measured ceilings: "
            + ", ".join(hardware["missing_measured_ceilings"])
        )
    if hardware.get("medium_confidence_fields"):
        lines.append(
            "- medium-confidence provenance: "
            + ", ".join(hardware["medium_confidence_fields"])
        )
    if hardware.get("low_confidence_fields"):
        lines.append(
            "- low-confidence provenance: "
            + ", ".join(hardware["low_confidence_fields"])
        )
    lines.extend(["", "## Measurement coverage", ""])
    for track, details in analysis["measurement_coverage"].items():
        lines.append(f"- `{track}`: {details['status']}")
    lines.extend(["", "## Findings", ""])
    for finding in analysis["findings"]:
        lines.append(
            f"- [{finding['confidence']}] {finding['observation']} "
            f"(scope: {finding['scope']})"
        )
    lines.extend(["", "## Hypotheses", ""])
    for hypothesis in analysis["hypotheses"]:
        lines.append(
            f"- [{hypothesis['confidence']}] {hypothesis['statement']} "
            f"Next measurement: {hypothesis['discriminating_measurement']}"
        )
    lines.extend(["", "## Constraints", ""])
    for constraint in analysis["constraints"]:
        lines.append(f"- {constraint}")
    lines.extend(["", "## Unknowns / collection plan", ""])
    for unknown in analysis["unknowns"]:
        lines.append(
            f"- `{unknown['track']}`: {unknown['question']}"
        )
    lines.extend(["", "## Reference patterns (unranked)", ""])
    for pattern in analysis["reference_patterns"]:
        lines.append(
            f"- {pattern['name']}: {pattern['mechanism']} "
            f"(caveat: {pattern['caveat']}; source: {pattern['source']})"
        )
    if analysis["degraded"]:
        lines.extend(["", "## Degraded", ""])
        for entry in analysis["degraded"]:
            lines.append(f"- {entry['reason']}: {', '.join(entry['missing_tracks'])}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hardware-context", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    report = _read_json(args.report)
    if args.hardware_context:
        report["hardware_context"] = _read_json(args.hardware_context)
    analysis = build_analysis(report)
    analysis["analysis_inputs"] = {
        "report": _file_identity(args.report),
        **(
            {"hardware_context": _file_identity(args.hardware_context)}
            if args.hardware_context
            else {}
        ),
    }
    validate_analysis_output(analysis)
    _write_json(args.output, analysis)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            _render_markdown(analysis),
            encoding="utf-8",
        )
    print(f"GEAK_MOE_ANALYSIS_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
