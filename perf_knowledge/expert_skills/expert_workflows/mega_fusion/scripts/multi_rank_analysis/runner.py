#!/usr/bin/env python3
"""Deterministic CLI for generic multi-rank record and trace aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

if __package__:
    from .aggregate import merge_rank_records
    from .bundle import (
        bundle_from_rank_report,
        validate_analysis_bundle,
        validate_measurement_tracks,
    )
    from .experiments import validate_experiment_manifest
    from .evidence import (
        EVIDENCE_CATALOG_SCHEMA_VERSION,
        resolve_measurement_tracks,
        validate_evidence_catalog,
    )
    from .hardware import validate_hardware_context
    from .instruction_analysis import (
        load_instruction_category_map,
        parse_att_stats_csv,
        read_att_occupancy,
    )
    from .provenance import validate_collection_provenance
    from .intervals import analyze_category_overlap
    from .schema import build_report
    from .trace_categories import bucket_trace_events, load_category_map
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from multi_rank_analysis.aggregate import merge_rank_records
    from multi_rank_analysis.bundle import (
        bundle_from_rank_report,
        validate_analysis_bundle,
        validate_measurement_tracks,
    )
    from multi_rank_analysis.experiments import validate_experiment_manifest
    from multi_rank_analysis.evidence import (
        EVIDENCE_CATALOG_SCHEMA_VERSION,
        resolve_measurement_tracks,
        validate_evidence_catalog,
    )
    from multi_rank_analysis.hardware import validate_hardware_context
    from multi_rank_analysis.instruction_analysis import (
        load_instruction_category_map,
        parse_att_stats_csv,
        read_att_occupancy,
    )
    from multi_rank_analysis.provenance import validate_collection_provenance
    from multi_rank_analysis.intervals import analyze_category_overlap
    from multi_rank_analysis.schema import build_report
    from multi_rank_analysis.trace_categories import (
        bucket_trace_events,
        load_category_map,
    )

_RANK_RE = re.compile(r"(?:^|[_-])rank[_-]?(\d+)(?:\D|$)")


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


def _file_identity_or_error(path: Path) -> dict:
    try:
        return {"status": "available", **_file_identity(path)}
    except OSError as error:
        return {
            "status": "unavailable",
            "path": str(path.resolve()),
            "error": str(error),
        }


def _rank_from_name(path: Path, fallback: int) -> int:
    match = _RANK_RE.search(path.name)
    return int(match.group(1)) if match else fallback


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def _aggregate_trace_files(
    trace_files: list[Path],
    category_map,
    expected_ranks: list[int] | None,
    replay_count: int,
) -> dict:
    if not trace_files:
        raise ValueError("no trace files were provided")
    trace_records = []
    per_rank_overlap = {}
    per_rank_event_counts = {}
    per_rank_kernel_names = {}
    errors = []
    for index, path in enumerate(trace_files):
        rank = _rank_from_name(path, index)
        bucketed = bucket_trace_events(str(path), category_map)
        if bucketed.get("error"):
            errors.append({"rank": rank, "path": str(path), "error": bucketed["error"]})
            continue
        trace_records.append(
            {
                "rank": rank,
                "per_category_ms": {
                    category: duration / replay_count
                    for category, duration in bucketed["per_category_ms"].items()
                },
                "per_category_event_count": bucketed["per_category_event_count"],
            }
        )
        overlap = analyze_category_overlap(
            bucketed["per_category_intervals_us"]
        )
        overlap["category_active_ms"] = {
            category: duration / replay_count
            for category, duration in overlap["category_active_ms"].items()
        }
        for pair in overlap["pairwise"].values():
            pair["overlap_ms"] /= replay_count
        overlap["categorized_kernel_union_ms"] /= replay_count
        overlap["summed_category_active_ms"] /= replay_count
        overlap["normalization"] = {
            "replay_count": replay_count,
            "semantics": "average per replay",
        }
        per_rank_overlap[str(rank)] = overlap
        per_rank_event_counts[str(rank)] = {
            "raw": dict(bucketed["per_category_event_count"]),
            "per_replay": {
                category: count / replay_count
                for category, count in bucketed["per_category_event_count"].items()
            },
        }
        per_rank_kernel_names[str(rank)] = dict(bucketed["kernel_names"])
        if bucketed.get("malformed_events"):
            errors.append(
                {
                    "rank": rank,
                    "path": str(path),
                    "malformed_events": bucketed["malformed_events"],
                }
            )
        if bucketed["per_category_event_count"].get("unclassified", 0):
            errors.append(
                {
                    "rank": rank,
                    "path": str(path),
                    "unclassified_events": bucketed[
                        "per_category_event_count"
                    ]["unclassified"],
                }
            )
    categories = sorted(
        {
            category
            for record in trace_records
            for category in record["per_category_ms"]
        }
    )
    metrics = [f"per_category_ms.{category}" for category in categories]
    return {
        "trace_files": [str(path) for path in trace_files],
        "replay_count": replay_count,
        "duration_semantics": "average milliseconds per replay",
        "categories": merge_rank_records(
            trace_records,
            metrics,
            expected_ranks=expected_ranks,
        ),
        "per_rank_overlap": per_rank_overlap,
        "per_rank_event_counts": per_rank_event_counts,
        "per_rank_kernel_names": per_rank_kernel_names,
        "errors": errors,
    }


def _aggregate_traces(
    trace_dir: Path,
    trace_glob: str,
    category_map,
    expected_ranks: list[int] | None,
    replay_count: int,
) -> dict:
    trace_files = sorted(trace_dir.glob(trace_glob))
    if not trace_files:
        raise ValueError(f"no trace files matched {trace_dir / trace_glob}")
    return _aggregate_trace_files(
        trace_files,
        category_map,
        expected_ranks,
        replay_count,
    )


def _aggregate_att(
    att_files: list[Path],
    instruction_map,
    occupancy_files: list[Path] | None = None,
) -> dict:
    return {
        "files": [str(path) for path in att_files],
        "reports": [
            parse_att_stats_csv(path, instruction_map)
            for path in att_files
        ],
        "occupancy_files": [
            str(path) for path in (occupancy_files or [])
        ],
        "occupancy": [
            read_att_occupancy(path)
            for path in (occupancy_files or [])
        ],
        "scope_warning": (
            "ATT reports sampled instruction-pipeline evidence; they are not "
            "whole-device E2E time or cross-GPU traffic."
        ),
    }


def _require_provenanced_paths(
    provenance: dict,
    paths: list[Path],
    artifact_kind: str,
) -> None:
    raw_paths = set(provenance["raw_artifacts"])
    missing = [str(path) for path in paths if str(path) not in raw_paths]
    if missing:
        raise ValueError(
            f"{artifact_kind} files missing from provenance raw_artifacts: {missing}"
        )


def _build_route_comparisons(bundle: dict, cases: list[dict]) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    metric_definitions = bundle["metric_definitions"]
    output = []
    for spec in bundle.get("route_comparisons", []):
        baseline = by_id[spec["baseline_case_id"]]
        candidate = by_id[spec["candidate_case_id"]]
        metric_path = spec.get("metric_path")
        if not metric_path:
            raise ValueError("route comparison must define metric_path")
        baseline_metric = baseline["rank_metrics"].get(metric_path)
        candidate_metric = candidate["rank_metrics"].get(metric_path)
        if not baseline_metric or not candidate_metric:
            raise ValueError(
                f"route comparison metric {metric_path!r} missing from cases"
            )
        definition = metric_definitions[metric_path]
        comparison = {
            **dict(spec),
            "status": "complete",
            "metric": {
                "path": metric_path,
                **dict(definition),
            },
            "baseline": {
                "case_id": baseline["case_id"],
                **dict(baseline_metric),
            },
            "candidate": {
                "case_id": candidate["case_id"],
                **dict(candidate_metric),
            },
        }
        incomplete_reasons = []
        for role, metric in (
            ("baseline", baseline_metric),
            ("candidate", candidate_metric),
        ):
            if metric.get("missing_ranks"):
                incomplete_reasons.append(
                    f"{role} missing ranks {metric['missing_ranks']}"
                )
            if metric.get("repetition_missing_ranks"):
                incomplete_reasons.append(
                    f"{role} has incomplete repetitions"
                )
        baseline_max = baseline_metric["rank_max"]
        candidate_max = candidate_metric["rank_max"]
        if baseline_max is None or candidate_max is None:
            incomplete_reasons.append("rank_max is unavailable")
        elif (
            definition["direction"] == "lower"
            and (baseline_max <= 0 or candidate_max <= 0)
        ):
            incomplete_reasons.append(
                "lower-is-better metric values must be positive"
            )
        if incomplete_reasons:
            comparison["status"] = "incomplete"
            comparison["incomplete_reasons"] = incomplete_reasons
            output.append(comparison)
            continue
        comparison["delta"] = candidate_max - baseline_max
        comparison["delta_pct"] = (
            (candidate_max / baseline_max - 1.0) * 100.0
            if baseline_max != 0.0
            else None
        )
        repeated = all(
            len(metric.get("rank_max_runs", [])) >= 3
            for metric in (baseline_metric, candidate_metric)
        )
        low_noise = all(
            metric.get("rank_max_span_pct") is not None
            and metric["rank_max_span_pct"] <= 5.0
            for metric in (baseline_metric, candidate_metric)
        )
        comparison["timing_confidence"] = (
            "high" if repeated and low_noise else "medium"
        )
        baseline_categories = baseline.get("trace", {}).get("categories", {})
        candidate_categories = candidate.get("trace", {}).get("categories", {})
        category_delta = {}
        trace_incomplete = bool(
            baseline.get("trace", {}).get("errors")
            or candidate.get("trace", {}).get("errors")
        )
        for path in sorted(set(baseline_categories) & set(candidate_categories)):
            if trace_incomplete:
                break
            before = baseline_categories[path]["rank_max"]
            after = candidate_categories[path]["rank_max"]
            if (
                before is None
                or after is None
                or before <= 0
                or baseline_categories[path].get("missing_ranks")
                or candidate_categories[path].get("missing_ranks")
            ):
                continue
            category = path.removeprefix("per_category_ms.")
            if category.startswith("__"):
                continue
            category_delta[category] = {
                "unit": "ms",
                "baseline_rank_max": before,
                "candidate_rank_max": after,
                "rank_max_delta_ms": after - before,
                "rank_max_delta_pct": (after / before - 1.0) * 100.0,
            }
        if category_delta:
            comparison["profile_category_delta"] = category_delta
            confidence_order = {"low": 0, "medium": 1, "high": 2}
            trace_confidences = [
                baseline.get("trace", {})
                .get("provenance", {})
                .get("confidence", "low"),
                candidate.get("trace", {})
                .get("provenance", {})
                .get("confidence", "low"),
            ]
            comparison["profile_confidence"] = min(
                trace_confidences,
                key=lambda value: confidence_order.get(value, 0),
            )
        if trace_incomplete:
            comparison["trace_status"] = "incomplete"
        output.append(comparison)
    return output


def _provenance_ref(provenance: dict | None, fallback: str) -> str:
    if not provenance:
        return fallback
    return (
        f"collector:{provenance['collector_id']}:"
        f"{provenance['timestamp']}"
    )


def _build_evidence_catalog(
    cases: list[dict],
    comparisons: list[dict],
    hardware_context: dict | None,
    experiment_manifest: dict | None,
    derived_evidence: dict,
    declared_provenance: dict,
    source_identity: dict,
) -> dict:
    reserved = ("case:", "comparison:", "hardware:", "experiment:")
    collisions = sorted(
        evidence_id
        for evidence_id in derived_evidence
        if evidence_id.startswith(reserved)
    )
    if collisions:
        raise ValueError(
            f"derived evidence uses reserved IDs: {collisions}"
        )
    entries = {key: dict(value) for key, value in derived_evidence.items()}
    source_ref = "input:rank_report"
    provenance = {
        provenance_id: {
            "kind": "collection",
            "status": "complete",
            "data": value,
        }
        for provenance_id, value in declared_provenance.items()
    }
    provenance[source_ref] = {
        "kind": "input_artifact",
        "status": (
            "complete" if source_identity.get("sha256") else "invalid"
        ),
        "data": source_identity,
    }
    for case in cases:
        case_id = case["case_id"]
        for path, metric in case["rank_metrics"].items():
            evidence_id = f"case:{case_id}:rank_metric:{path}"
            complete = not (
                metric.get("missing_ranks")
                or metric.get("repetition_missing_ranks")
                or metric.get("rank_max") is None
            )
            entries[evidence_id] = {
                "kind": "rank_metric",
                "status": "complete" if complete else "partial",
                "metric_ids": [
                    f"{evidence_id}:rank_mean",
                    f"{evidence_id}:rank_max",
                    f"{evidence_id}:rank_tail_spread_pct",
                ],
                "provenance_refs": [source_ref],
                "data": dict(metric),
            }
        if case.get("trace"):
            trace = case["trace"]
            evidence_id = f"case:{case_id}:trace"
            trace_ref = _provenance_ref(
                trace.get("provenance"),
                f"missing:case:{case_id}:trace",
            )
            provenance[trace_ref] = {
                "kind": "collection",
                "status": (
                    "complete" if trace.get("provenance") else "invalid"
                ),
                "data": trace.get("provenance") or {},
            }
            entries[evidence_id] = {
                "kind": "trace",
                "status": "partial" if trace.get("errors") else "complete",
                "confidence": trace.get("provenance", {}).get(
                    "confidence",
                    "low",
                ),
                "scope": trace.get("provenance", {}).get(
                    "scope",
                    "trace",
                ),
                "metric_ids": [
                    f"{evidence_id}:{path}"
                    for path in sorted(trace.get("categories", {}))
                ],
                "provenance_refs": [trace_ref],
                "data": trace,
            }
        if case.get("att"):
            att = case["att"]
            att_ref = _provenance_ref(
                att.get("provenance"),
                f"missing:case:{case_id}:att",
            )
            provenance[att_ref] = {
                "kind": "collection",
                "status": (
                    "complete" if att.get("provenance") else "invalid"
                ),
                "data": att.get("provenance") or {},
            }
            for index, report in enumerate(att.get("reports", [])):
                evidence_id = f"case:{case_id}:att:{index}"
                report_data = dict(report)
                if index < len(att.get("occupancy", [])):
                    report_data["occupancy"] = att["occupancy"][index]
                entries[evidence_id] = {
                    "kind": "att",
                    "status": (
                        "complete" if report.get("categories") else "partial"
                    ),
                    "confidence": att.get("provenance", {}).get(
                        "confidence",
                        "low",
                    ),
                    "scope": att.get("provenance", {}).get(
                        "scope",
                        "sampled ATT",
                    ),
                    "metric_ids": [
                        f"{evidence_id}:{category}:accounted_cycles"
                        for category in sorted(report.get("categories", {}))
                    ]
                    + (
                        [
                            f"{evidence_id}:occupancy",
                            f"{evidence_id}:occupancy:avg_active_waves_per_cu",
                            f"{evidence_id}:occupancy:peak_active_waves_per_cu",
                            f"{evidence_id}:occupancy:wave_slot_occupancy_pct",
                            f"{evidence_id}:occupancy:balanced_events",
                        ]
                        if "occupancy" in report_data
                        else []
                    ),
                    "provenance_refs": [att_ref],
                    "data": report_data,
                }
        if case.get("route_summary"):
            evidence_id = f"case:{case_id}:route_summary"
            if case.get("route_summary_source") == "rank_report":
                route_ref = source_ref
            else:
                route_ref = _provenance_ref(
                    case.get("route_provenance"),
                    f"missing:case:{case_id}:route",
                )
                provenance[route_ref] = {
                    "kind": "collection",
                    "status": (
                        "complete"
                        if case.get("route_provenance")
                        else "invalid"
                    ),
                    "data": case.get("route_provenance") or {},
                }
            entries[evidence_id] = {
                "kind": "route_summary",
                "status": "complete",
                "metric_ids": [],
                "provenance_refs": [route_ref],
                "data": case["route_summary"],
            }
        if case.get("software_counters"):
            counters = case["software_counters"]
            evidence_id = f"case:{case_id}:software_counters"
            counter_ref = _provenance_ref(
                counters.get("provenance"),
                f"missing:case:{case_id}:software_counters",
            )
            provenance[counter_ref] = {
                "kind": "collection",
                "status": (
                    "complete" if counters.get("provenance") else "invalid"
                ),
                "data": counters.get("provenance") or {},
            }
            entries[evidence_id] = {
                "kind": "software_counters",
                "status": "complete" if counters.get("records") else "partial",
                "confidence": counters.get("provenance", {}).get(
                    "confidence",
                    "low",
                ),
                "scope": counters.get("provenance", {}).get(
                    "scope",
                    "software counters",
                ),
                "metric_ids": [
                    f"{evidence_id}:payload_bytes",
                    f"{evidence_id}:metadata_bytes",
                    f"{evidence_id}:publication_count",
                    f"{evidence_id}:useful_rows",
                    f"{evidence_id}:padded_rows",
                    f"{evidence_id}:wait_cycles",
                    f"{evidence_id}:readiness_polls",
                    f"{evidence_id}:barrier_count",
                ],
                "provenance_refs": [counter_ref],
                "data": counters,
            }
    for index, comparison in enumerate(comparisons):
        evidence_id = (
            f"comparison:{comparison['baseline_case_id']}:"
            f"{comparison['candidate_case_id']}:{index}"
        )
        metric_ids = []
        if comparison.get("status") == "complete":
            metric_ids.extend(
                [f"{evidence_id}:delta", f"{evidence_id}:delta_pct"]
            )
            metric_ids.extend(
                f"{evidence_id}:category:{category}:delta_ms"
                for category in sorted(
                    comparison.get("profile_category_delta", {})
                )
            )
        entries[evidence_id] = {
            "kind": "comparison",
            "status": (
                "complete"
                if comparison.get("status") == "complete"
                else "partial"
            ),
            "metric_ids": metric_ids,
            "provenance_refs": [source_ref],
            "data": comparison,
        }
    if hardware_context is not None:
        hardware_refs = []
        for field, hardware_provenance in hardware_context.get(
            "provenance",
            {},
        ).items():
            provenance_id = f"hardware:{field}"
            hardware_refs.append(provenance_id)
            provenance[provenance_id] = {
                "kind": "hardware_field",
                "status": "complete",
                "data": hardware_provenance,
            }
        entries["hardware:context"] = {
            "kind": "hardware_context",
            "status": "complete",
            "metric_ids": [
                f"hardware:measured:{metric}"
                for metric, value in hardware_context.get("measured", {}).items()
                if value is not None
            ],
            "provenance_refs": sorted(hardware_refs),
            "data": hardware_context,
        }
    if experiment_manifest is not None:
        entries[f"experiment:{experiment_manifest['experiment_id']}"] = {
            "kind": "controlled_experiment",
            "status": "complete",
            "metric_ids": [],
            "provenance_refs": [
                variant["provenance_ref"]
                for variant in experiment_manifest["variants"]
            ],
            "data": experiment_manifest,
        }
    return {
        "schema_version": EVIDENCE_CATALOG_SCHEMA_VERSION,
        "entries": entries,
        "provenance": provenance,
    }


def _derive_report_status(
    bundle: dict,
    cases: list[dict],
    comparisons: list[dict],
    measurement_tracks: dict | None,
) -> tuple[str, list[str]]:
    reasons = []
    source_status = str(bundle.get("source", {}).get("status", "unknown"))
    if source_status not in ("pass", "unknown"):
        return "fail", [f"source status is {source_status!r}"]
    if source_status != "pass":
        reasons.append("source status is not explicitly pass")
    if str(bundle.get("world_size_source", "")).startswith("inferred"):
        reasons.append("world size was inferred rather than declared")
    for case in cases:
        for path, metric in case["rank_metrics"].items():
            if metric.get("missing_ranks"):
                reasons.append(
                    f"{case['case_id']}:{path} missing ranks "
                    f"{metric['missing_ranks']}"
                )
            if metric.get("repetition_missing_ranks"):
                reasons.append(
                    f"{case['case_id']}:{path} has incomplete repetitions"
                )
        if case.get("trace", {}).get("errors"):
            reasons.append(f"{case['case_id']} has trace errors")
    for comparison in comparisons:
        if comparison.get("status") != "complete":
            reasons.append(
                "comparison "
                f"{comparison.get('baseline_case_id')}→"
                f"{comparison.get('candidate_case_id')} is incomplete"
            )
    for name, track in (measurement_tracks or {}).items():
        if track.get("status") == "invalid":
            reasons.append(f"measurement track {name!r} has invalid evidence")
    return ("partial" if reasons else "pass"), reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rank-records", type=Path)
    source.add_argument("--analysis-bundle", type=Path)
    parser.add_argument("--metric", action="append")
    parser.add_argument("--case-id")
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--primary-metric-path")
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-glob", default="*.json")
    parser.add_argument("--trace-replays", type=int)
    parser.add_argument("--category-map")
    parser.add_argument("--trace-provenance", type=Path)
    parser.add_argument("--att-stats", type=Path, action="append")
    parser.add_argument("--att-occupancy", type=Path, action="append")
    parser.add_argument("--instruction-map")
    parser.add_argument("--att-provenance", type=Path)
    parser.add_argument("--hardware-context", type=Path)
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument("--measurement-tracks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_world_size is not None and args.expected_world_size <= 0:
        parser.error("--expected-world-size must be positive")
    if args.trace_dir and not args.trace_provenance:
        parser.error("--trace-provenance is required with --trace-dir")
    if args.trace_dir and (args.trace_replays is None or args.trace_replays <= 0):
        parser.error("--trace-replays must be positive with --trace-dir")
    if args.trace_provenance and not args.trace_dir:
        parser.error("--trace-provenance requires --trace-dir")
    if args.trace_replays is not None and not args.trace_dir:
        parser.error("--trace-replays requires --trace-dir")
    if args.att_stats and not args.att_provenance:
        parser.error("--att-provenance is required with --att-stats")
    if args.att_provenance and not args.att_stats:
        parser.error("--att-provenance requires --att-stats")
    if args.att_occupancy and not args.att_stats:
        parser.error("--att-occupancy requires --att-stats")
    if args.att_occupancy and len(args.att_occupancy) != len(args.att_stats):
        parser.error("--att-occupancy count must match --att-stats")
    if args.analysis_bundle:
        bundle = validate_analysis_bundle(_read_json(args.analysis_bundle))
    else:
        if not args.metric:
            parser.error("--metric is required with --rank-records")
        raw_rank_report = _read_json(args.rank_records)
        bundle = bundle_from_rank_report(
            raw_rank_report,
            args.metric,
            expected_world_size=args.expected_world_size,
        )
        if args.case_id and (
            isinstance(raw_rank_report, list)
            or (isinstance(raw_rank_report, dict) and "records" in raw_rank_report)
        ):
            bundle["cases"][0]["case_id"] = args.case_id
    if args.case_id:
        selected = [
            case for case in bundle["cases"] if case["case_id"] == args.case_id
        ]
        if not selected:
            parser.error(f"--case-id {args.case_id!r} was not found")
        bundle = {**bundle, "cases": selected, "route_comparisons": []}
    if (args.trace_dir or args.att_stats) and len(bundle["cases"]) != 1:
        parser.error(
            "CLI trace/ATT overrides require exactly one selected case; "
            "use per-case bundle fields otherwise"
        )
    if (
        args.expected_world_size is not None
        and args.expected_world_size != bundle["expected_world_size"]
    ):
        parser.error(
            "--expected-world-size does not match analysis bundle"
        )
    expected_world_size = (
        args.expected_world_size or bundle["expected_world_size"]
    )
    primary_metric_path = args.primary_metric_path
    if primary_metric_path is None:
        if len(bundle["metric_definitions"]) != 1:
            parser.error(
                "--primary-metric-path is required when multiple metrics exist"
            )
        primary_metric_path = next(iter(bundle["metric_definitions"]))
    if primary_metric_path not in bundle["metric_definitions"]:
        parser.error("--primary-metric-path is not defined in the bundle")
    expected_ranks = list(range(expected_world_size))
    category_map_path = Path(
        args.category_map
        or str(Path(__file__).with_name("default_category_map.json"))
    )
    instruction_map_path = Path(
        args.instruction_map
        or str(Path(__file__).with_name("default_instruction_map.json"))
    )
    category_map = load_category_map(str(category_map_path))
    instruction_map = load_instruction_category_map(str(instruction_map_path))
    cli_trace_files = (
        sorted(args.trace_dir.glob(args.trace_glob))
        if args.trace_dir
        else []
    )
    if args.trace_dir and not cli_trace_files:
        parser.error(f"no trace files matched {args.trace_dir / args.trace_glob}")
    cases = []
    all_trace_files = []
    all_att_files = []
    for index, case_spec in enumerate(bundle["cases"]):
        case = {
            "case_id": case_spec["case_id"],
            "rank_metrics": merge_rank_records(
                case_spec["rank_records"],
                case_spec["metric_paths"],
                repetitions=case_spec.get("repetitions"),
                expected_ranks=expected_ranks,
            ),
            "route_summary": case_spec.get("route_summary", {}),
            "route_summary_source": case_spec.get("route_summary_source"),
            "software_counters": case_spec.get("software_counters", {}),
        }
        trace_files = [Path(path) for path in case_spec.get("trace_files", [])]
        if args.trace_dir and index == 0:
            if trace_files:
                raise ValueError(
                    "CLI trace files cannot be mixed with bundle trace files"
                )
            trace_files.extend(cli_trace_files)
        all_trace_files.extend(trace_files)
        if trace_files:
            trace_replays = (
                args.trace_replays
                if args.trace_dir and index == 0
                else case_spec.get("trace_replays")
            )
            case["trace"] = _aggregate_trace_files(
                _dedupe_paths(trace_files),
                category_map,
                expected_ranks,
                trace_replays,
            )
            trace_provenance = case_spec.get("trace_provenance")
            if args.trace_provenance and index == 0:
                trace_provenance = validate_collection_provenance(
                    _read_json(args.trace_provenance)
                )
            if trace_provenance:
                _require_provenanced_paths(
                    trace_provenance,
                    _dedupe_paths(trace_files),
                    "trace",
                )
                case["trace"]["provenance"] = trace_provenance
        att_files = [Path(path) for path in case_spec.get("att_stats_files", [])]
        occupancy_files = [
            Path(path)
            for path in case_spec.get("att_occupancy_files", [])
        ]
        if args.att_stats and index == 0:
            if att_files or occupancy_files:
                raise ValueError(
                    "CLI ATT files cannot be mixed with bundle ATT files"
                )
            att_files.extend(args.att_stats)
            occupancy_files.extend(args.att_occupancy or [])
        all_att_files.extend(att_files + occupancy_files)
        if att_files:
            case["att"] = _aggregate_att(
                _dedupe_paths(att_files),
                instruction_map,
                _dedupe_paths(occupancy_files),
            )
            att_provenance = case_spec.get("att_provenance")
            if args.att_provenance and index == 0:
                att_provenance = validate_collection_provenance(
                    _read_json(args.att_provenance)
                )
            if att_provenance:
                _require_provenanced_paths(
                    att_provenance,
                    _dedupe_paths(att_files + occupancy_files),
                    "ATT",
                )
                case["att"]["provenance"] = att_provenance
        if case_spec.get("route_provenance"):
            case["route_provenance"] = case_spec["route_provenance"]
        if case_spec.get("source_case"):
            case["source_case"] = case_spec["source_case"]
        cases.append(case)
    experiment_manifest_payload = (
        _read_json(args.experiment_manifest)
        if args.experiment_manifest
        else bundle.get("experiment_manifest")
    )
    experiment_manifest = (
        validate_experiment_manifest(experiment_manifest_payload)
        if experiment_manifest_payload
        else None
    )
    measurement_tracks_payload = (
        _read_json(args.measurement_tracks)
        if args.measurement_tracks
        else bundle.get("measurement_tracks")
    )
    measurement_tracks = (
        validate_measurement_tracks(measurement_tracks_payload)
        if measurement_tracks_payload
        else None
    )
    hardware_context = (
        validate_hardware_context(_read_json(args.hardware_context))
        if args.hardware_context
        else (
            validate_hardware_context(bundle["hardware_context"])
            if bundle.get("hardware_context")
            else None
        )
    )
    route_comparisons = _build_route_comparisons(bundle, cases)
    source_identity = _file_identity_or_error(
        args.analysis_bundle or args.rank_records
    )
    evidence_catalog = validate_evidence_catalog(
        _build_evidence_catalog(
            cases,
            route_comparisons,
            hardware_context,
            experiment_manifest,
            bundle.get("derived_evidence", {}),
            bundle.get("provenance_catalog", {}),
            source_identity,
        )
    )
    measurement_tracks = (
        resolve_measurement_tracks(measurement_tracks, evidence_catalog)
        if measurement_tracks
        else None
    )
    report_status, status_reasons = _derive_report_status(
        bundle,
        cases,
        route_comparisons,
        measurement_tracks,
    )
    report = build_report(
        {
            "description": args.primary_metric,
            "path": primary_metric_path,
            **dict(bundle["metric_definitions"][primary_metric_path]),
        },
        cases,
        status=report_status,
        route_comparisons=route_comparisons,
        hardware_context=hardware_context,
        measurement_tracks=measurement_tracks,
        experiment_manifest=experiment_manifest,
        workload=bundle["workload"],
        source=bundle["source"],
        expected_world_size=expected_world_size,
        metric_definitions=bundle["metric_definitions"],
    )
    report["status_reasons"] = status_reasons
    report["evidence_catalog"] = evidence_catalog
    input_paths = {
        (
            "analysis_bundle" if args.analysis_bundle else "rank_records"
        ): args.analysis_bundle or args.rank_records,
        "category_map": category_map_path,
        "instruction_map": instruction_map_path,
    }
    for name, path in (
        ("hardware_context", args.hardware_context),
        ("experiment_manifest", args.experiment_manifest),
        ("measurement_tracks", args.measurement_tracks),
        ("trace_provenance", args.trace_provenance),
        ("att_provenance", args.att_provenance),
    ):
        if path is not None:
            input_paths[name] = path
    report["input_artifacts"] = {
        name: _file_identity_or_error(Path(path))
        for name, path in input_paths.items()
    }
    report["raw_artifact_identities"] = {
        "trace_files": [
            _file_identity_or_error(path)
            for path in sorted(set(all_trace_files))
        ],
        "att_stats_files": [
            _file_identity_or_error(path)
            for path in sorted(set(all_att_files))
        ],
    }
    _write_json(args.output, report)
    print(f"GEAK_MULTIRANK_ANALYSIS_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
