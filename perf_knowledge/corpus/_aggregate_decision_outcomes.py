#!/usr/bin/env python3
"""Aggregate decision outcomes from run-local validation manifests.

The recorded ratio is named ``speedup_vs_frozen_baseline`` because its denominator is the
run's frozen baseline.  A direction citing multiple decisions is retained as a bundle-level
observation and is not attributed to any one member of the bundle.

A run's ``ranking_basis`` says why its results may order candidates: ``accepted`` by the
Director, or ``graph_timed_lower_bound`` when the only flags are the timing-receipt ones and
every timed case was a CUDA-graph replay.  Both legs of a graph replay carry the same
submission bubble, which pulls the ratio toward 1.0, so a speedup measured that way is a lower
bound.  Eager timing carries each implementation's own host dispatch and gets no such basis.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


SCHEMA_VERSION = 1
MANIFEST_NAMES = ("validation_environment.yaml", "validation_environment.json")
CONTEXT_FIELDS = (
    "observed_language", "flydsl_version", "provider", "gfx", "op_spec",
    "aiter_commit", "geak_commit", "workspace_commit", "rocm_version",
)


def _error(path, message):
    return {"path": str(path), "error": str(message)}


def discover_manifests(run_paths):
    """Return one manifest per run directory, deduplicating overlapping input paths."""
    errors = []
    by_run = {}
    priority = {name: index for index, name in enumerate(MANIFEST_NAMES)}

    roots = []
    seen_roots = set()
    for raw_path in run_paths:
        path = Path(raw_path).expanduser().resolve()
        if path in seen_roots:
            continue
        seen_roots.add(path)
        roots.append(path)

    candidates = []
    for root in roots:
        if not root.exists():
            errors.append(_error(root, "runs path does not exist"))
            continue
        if root.is_file():
            if root.name in MANIFEST_NAMES:
                candidates.append(root)
            else:
                errors.append(_error(root, "file is not a validation environment manifest"))
            continue
        for name in MANIFEST_NAMES:
            candidates.extend(root.rglob(name))

    for candidate in sorted({path.resolve() for path in candidates}, key=str):
        run = candidate.parent.resolve()
        current = by_run.get(run)
        if current is None or priority[candidate.name] < priority[current.name]:
            by_run[run] = candidate
    return [by_run[run] for run in sorted(by_run, key=str)], errors


def _load_manifest(path):
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a mapping")
    return data


TIMING_ONLY_REASONS = frozenset({"timing_receipt_absent", "timer_unprimed"})
TIMING_BASIS_REASONS = {
    "unknown": "timing_receipt_absent",
    "unprimed": "timer_unprimed",
    "host_bound": "host_bound",
}


def _read_json(path):
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _graph_timed(meta):
    """True when every timed case of a task meta is timed as a CUDA-graph replay."""
    cases = [case for case in (meta or {}).get("cases") or []
             if isinstance(case, dict) and case.get("timed") is not False]
    if not cases or any("graph" not in case for case in cases):
        return None
    return all(case.get("graph") is True for case in cases)


def _flag_evidence(manifest, run_dir):
    """Why a run is not plainly accepted, from the manifest or, for older runs, the Director's file.

    Manifests written since flag reasons were recorded carry ``validation.flag_reasons``,
    ``validation.timing_basis`` and ``validation.graph_timed``.  Older ones do not; for those the
    same facts are read from ``director_validation.json`` and the task ``meta.json`` in the run.
    """
    validation = manifest.get("validation") if isinstance(manifest.get("validation"), dict) else {}
    if any(key in validation for key in ("flag_reasons", "timing_basis", "graph_timed")):
        return {
            "timing_basis": validation.get("timing_basis"),
            "flag_reasons": sorted({str(reason) for reason in validation.get("flag_reasons") or []}),
            "graph_timed": validation.get("graph_timed"),
            "flag_evidence": "manifest",
        }
    director = _read_json(run_dir / "director_validation.json") if run_dir else None
    if director is None:
        return {"timing_basis": None, "flag_reasons": [], "graph_timed": None,
                "flag_evidence": "not_recorded"}
    reasons = set()
    basis = director.get("timing_basis")
    if basis in TIMING_BASIS_REASONS:
        reasons.add(TIMING_BASIS_REASONS[basis])
    if not str(director.get("correctness") or "").lower().startswith("pass"):
        reasons.add("correctness_fail")
    claim = director.get("tech_lead_reported_speedup_geomean")
    measured = director.get("director_verified_speedup_geomean")
    if _finite_positive(claim) and _finite_positive(measured) and measured < 0.9 * claim:
        reasons.add("below_claim")
    gate = director.get("prevalidation_gate")
    if isinstance(gate, dict) and gate.get("passed") is False:
        reasons.add("prevalidation_failed")
    meta = None
    for candidate in ("workspace/meta.json", "validation_workspace/meta.json"):
        meta = _read_json(run_dir / candidate)
        if meta is not None:
            break
    return {"timing_basis": basis, "flag_reasons": sorted(reasons),
            "graph_timed": _graph_timed(meta), "flag_evidence": "director_validation.json"}


def ranking_basis(validation_status, flag_reasons, graph_timed):
    """The reason a run's results may order candidates, or None when they may not."""
    status = str(validation_status or "").lower()
    if status == "accepted":
        return "accepted"
    if (status == "flagged" and flag_reasons and set(flag_reasons) <= TIMING_ONLY_REASONS
            and graph_timed is True):
        return "graph_timed_lower_bound"
    return None


def _context(manifest, run_dir=None):
    context = {field: manifest.get(field) for field in CONTEXT_FIELDS}
    measurement = manifest.get("measurement") if isinstance(manifest.get("measurement"), dict) else {}
    validation = manifest.get("validation") if isinstance(manifest.get("validation"), dict) else {}
    protocol = {
        "method": measurement.get("method"),
        "benchmark_cmd": measurement.get("benchmark_cmd"),
        "warmup_iterations": measurement.get("warmup_iterations"),
        "benchmark_iterations": measurement.get("benchmark_iterations"),
        "workload_aligned": measurement.get("workload_aligned"),
    }
    # The lane names the metric its direction scores were computed in. Older manifests do not, and
    # their scores are time-weighted only when a caller workload spec was in play.
    metric_kind = measurement.get("primary_metric")
    if metric_kind not in ("time_weighted", "geomean"):
        metric_kind = "time_weighted" if measurement.get("workload_aligned") else "geomean"
    evidence = _flag_evidence(manifest, run_dir)
    observed = context.get("observed_language")
    captured = observed not in (None, "", "not_captured")
    context.update({
        "observed_language": observed if captured else None,
        "observed_language_basis": validation.get("observed_language_basis")
        or ("detected" if captured else None),
        "metric_kind": metric_kind,
        "measurement_reliable": measurement.get("reliable") is True,
        "validation_status": validation.get("status"),
        **evidence,
        "ranking_basis": ranking_basis(
            validation.get("status"), evidence["flag_reasons"], evidence["graph_timed"],
        ),
        "protocol_fingerprint": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
    })
    return context


def _run_identity(manifest):
    explicit = manifest.get("run_id") or manifest.get("session_id")
    if explicit:
        return str(explicit)
    durable = {
        "kernel": manifest.get("kernel"),
        "captured_at": manifest.get("captured_at"),
        "workspace_commit": manifest.get("workspace_commit"),
        "validation": manifest.get("validation"),
        "decision_outcomes": manifest.get("decision_outcomes"),
    }
    return "run_" + hashlib.sha256(
        json.dumps(durable, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _count_key(value):
    if value is None or value == "":
        return "not_recorded"
    return str(value)


def _common_value(rows, field):
    values = []
    for row in rows:
        value = row.get(field)
        if value not in values:
            values.append(value)
    return values[0] if len(values) == 1 else values


def _finite_positive(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _common_number(rows, field):
    values = [row.get(field) for row in rows]
    if not values or not all(_finite_positive(value) for value in values):
        return None, bool([value for value in values if value is not None])
    unique = {float(value) for value in values}
    return (unique.pop(), False) if len(unique) == 1 else (None, True)


def _quality_eligible(context, status, correctness, attribution, metric_conflict):
    statuses = status if isinstance(status, list) else [status]
    correctness_values = correctness if isinstance(correctness, list) else [correctness]
    measured_status = all(
        str(value or "").lower().startswith(("verified", "regression"))
        for value in statuses
    )
    correct = all(str(value or "").lower().startswith("pass") for value in correctness_values)
    return (
        attribution in {"single_ref_direction", "bundle"}
        and context.get("measurement_reliable") is True
        and context.get("ranking_basis") is not None
        and measured_status
        and correct
        and not metric_conflict
    )


def _direction_result(run, manifest_path, context, rows):
    decisions = sorted({str(row["decision"]) for row in rows})
    frozen, frozen_conflict = _common_number(rows, "cited_then_verified")
    incremental, incremental_conflict = _common_number(rows, "incremental_vs_incumbent")
    recorded_attribution = _common_value(rows, "attribution")
    if isinstance(recorded_attribution, list):
        recorded_attribution = None
    if recorded_attribution in {"single_ref_direction", "single_ref_seed", "bundle"}:
        attribution = recorded_attribution
    elif len(decisions) == 1:
        attribution = "single_ref_direction"
    else:
        attribution = "bundle"
    status = _common_value(rows, "status")
    correctness = _common_value(rows, "correctness")
    result = {
        "run": str(run),
        "manifest": str(manifest_path),
        "round": _common_value(rows, "round"),
        "direction": _common_value(rows, "direction"),
        "hypothesis_source": _common_value(rows, "hypothesis_source"),
        "decisions": decisions,
        "decision_count": len(decisions),
        "speedup_vs_frozen_baseline": frozen,
        "incremental_vs_incumbent": incremental,
        "incremental_basis": _common_value(rows, "incremental_basis"),
        "status": status,
        "correctness": correctness,
        "became_winner": any(bool(row.get("became_winner")) for row in rows),
        "context": context,
        "quality_eligible_for_ranking": _quality_eligible(
            context, status, correctness, attribution, frozen_conflict or incremental_conflict,
        ),
    }
    if frozen_conflict or incremental_conflict:
        result["metric_conflict"] = {
            "speedup_vs_frozen_baseline": frozen_conflict,
            "incremental_vs_incumbent": incremental_conflict,
        }
    specialties = sorted({
        str(row["specialty"]) for row in rows if row.get("specialty") not in (None, "")
    })
    if specialties:
        result["specialties"] = specialties
    if attribution == "single_ref_direction":
        # One ref is attributable as a planner input, not necessarily as a causal code change: an
        # engineer can pivot after reading the card. `engaged_refs` can strengthen this later.
        result["attribution_scope"] = "single_ref_direction"
    elif attribution == "single_ref_seed":
        result["attribution_scope"] = "single_ref_seed"
        result["individual_decision_attribution"] = False
    else:
        result["attribution_scope"] = "bundle_only"
        result["individual_decision_attribution"] = False
    return result


def aggregate_decision_outcomes(run_paths, assumed_languages=None):
    """Aggregate all parseable manifests below ``run_paths`` into a stable dictionary.

    ``assumed_languages`` maps a run directory to the language its task is written in, for runs
    whose manifest did not capture one (older optimize lanes over a seam that carries two backends).
    It is applied only to such runs and is recorded as ``observed_language_basis: assumed_by_operator``.
    """
    assumed = {str(Path(path).expanduser().resolve()): str(language)
               for path, language in (assumed_languages or {}).items()}
    manifests, errors = discover_manifests(run_paths)
    decision_stats = {}
    single_ref_results = []
    single_ref_seed_results = []
    bundled_results = []
    runs_scanned = 0
    runs_deduplicated = 0
    seen_run_ids = set()

    for manifest_path in manifests:
        try:
            manifest = _load_manifest(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
            errors.append(_error(manifest_path, exc))
            continue

        run_id = _run_identity(manifest)
        if run_id in seen_run_ids:
            runs_deduplicated += 1
            continue
        seen_run_ids.add(run_id)
        runs_scanned += 1
        run = manifest_path.parent.resolve()
        context = _context(manifest, run)
        if context["observed_language"] is None and str(run) in assumed:
            context["observed_language"] = assumed[str(run)]
            context["observed_language_basis"] = "assumed_by_operator"
        context["run_id"] = run_id
        outcomes = manifest.get("decision_outcomes") or []
        if not isinstance(outcomes, list):
            errors.append(_error(manifest_path, "decision_outcomes must be a list"))
            continue

        grouped = defaultdict(list)
        valid_rows = []
        for index, row in enumerate(outcomes):
            if not isinstance(row, dict):
                errors.append(_error(manifest_path, "decision_outcomes[%d] is not a mapping" % index))
                continue
            decision = row.get("decision")
            if decision in (None, ""):
                errors.append(_error(manifest_path, "decision_outcomes[%d] has no decision" % index))
                continue
            copied = dict(row)
            copied["decision"] = str(decision)
            valid_rows.append(copied)
            phase = copied.get("phase")
            round_id = copied.get("round")
            direction = copied.get("direction")
            if phase == "author_seed" or direction == "author_seed":
                group_key = ("author_seed", round_id, direction or "author_seed")
            elif round_id is None and direction in (None, ""):
                # Unknown rows are not evidence that they came from one bundle.
                group_key = ("unscoped_row", index, None)
            else:
                group_key = (phase or "round", round_id, direction)
            copied["_group_key"] = group_key
            grouped[group_key].append(copied)

        group_results = {}
        for group_key, rows in grouped.items():
            result = _direction_result(run, manifest_path, context, rows)
            group_results[group_key] = result
            if result["attribution_scope"] == "single_ref_direction":
                single_ref_results.append(result)
            elif result["attribution_scope"] == "single_ref_seed":
                single_ref_seed_results.append(result)
            else:
                bundled_results.append(result)

        for row in valid_rows:
            decision = row["decision"]
            stats = decision_stats.setdefault(decision, {
                "attempts": 0,
                "_runs": set(),
                "_status_counts": Counter(),
                "_correctness_counts": Counter(),
                "winner_count": 0,
                "_single_ref_groups": set(),
                "_single_ref_seed_groups": set(),
                "_bundled_groups": set(),
                "_single_ref_incremental": [],
                "_single_ref_frozen": [],
                "contexts": [],
            })
            stats["attempts"] += 1
            stats["_runs"].add(str(run))
            stats["_status_counts"][_count_key(row.get("status"))] += 1
            stats["_correctness_counts"][_count_key(row.get("correctness"))] += 1
            group_key = row["_group_key"]
            qualified_group = (str(run), *group_key)
            result = group_results[group_key]
            if result["attribution_scope"] == "single_ref_direction":
                stats["_single_ref_groups"].add(qualified_group)
                stats["winner_count"] += int(bool(row.get("became_winner")))
                incremental = row.get("incremental_vs_incumbent")
                frozen = row.get("cited_then_verified")
                if result["quality_eligible_for_ranking"] and _finite_positive(incremental):
                    stats["_single_ref_incremental"].append(float(incremental))
                if result["quality_eligible_for_ranking"] and _finite_positive(frozen):
                    stats["_single_ref_frozen"].append(float(frozen))
            elif result["attribution_scope"] == "single_ref_seed":
                stats["_single_ref_seed_groups"].add(qualified_group)
            else:
                stats["_bundled_groups"].add(qualified_group)
            if context not in stats["contexts"]:
                stats["contexts"].append(context)

    decisions = {}
    for decision in sorted(decision_stats):
        stats = decision_stats[decision]
        decisions[decision] = {
            "attempts": stats["attempts"],
            "unique_runs": len(stats["_runs"]),
            "runs": sorted(stats["_runs"]),
            "status_counts": dict(sorted(stats["_status_counts"].items())),
            "correctness_counts": dict(sorted(stats["_correctness_counts"].items())),
            "winner_count": stats["winner_count"],
            "single_ref_direction_results": len(stats["_single_ref_groups"]),
            "single_ref_seed_results": len(stats["_single_ref_seed_groups"]),
            "bundled_direction_results": len(stats["_bundled_groups"]),
            "contexts": stats["contexts"],
        }
        if stats["_single_ref_incremental"]:
            values = stats["_single_ref_incremental"]
            decisions[decision]["single_ref_incremental_vs_incumbent"] = {
                "count": len(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        if stats["_single_ref_frozen"]:
            values = stats["_single_ref_frozen"]
            decisions[decision]["single_ref_speedup_vs_frozen_baseline"] = {
                "count": len(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }

    result_key = lambda item: (
        item["run"], str(item["round"]), str(item["direction"]), tuple(item["decisions"])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "runs_scanned": runs_scanned,
        "runs_deduplicated": runs_deduplicated,
        "errors": sorted(errors, key=lambda item: (item["path"], item["error"])),
        "semantics": {
            "speedup_vs_frozen_baseline": "measured speedup ratio versus the frozen baseline",
            "incremental_vs_incumbent": (
                "ratio of the candidate and round-start incumbent scores against the same frozen "
                "baseline; planner attribution, not proof that the referenced card caused the change"
            ),
            "winner_count": "winning outcomes from single-ref directions only; not causal credit",
            "bundled_direction_results": (
                "bundle-level observations without attribution to an individual decision"
            ),
        },
        "decisions": decisions,
        "single_ref_direction_results": sorted(single_ref_results, key=result_key),
        "single_ref_seed_results": sorted(single_ref_seed_results, key=result_key),
        "bundled_direction_results": sorted(bundled_results, key=result_key),
    }


def aggregate(run_paths):
    """Short programmatic alias for ``aggregate_decision_outcomes``."""
    return aggregate_decision_outcomes(run_paths)


def _write_output(data, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".json":
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    else:
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    output.write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", action="append", required=True, metavar="PATH",
        help="directory or manifest to scan; may be repeated",
    )
    parser.add_argument(
        "--assume-language", action="append", default=[], metavar="RUN=LANGUAGE",
        help="language of a run whose manifest captured none; recorded as assumed_by_operator",
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--json", action="store_true", help="write JSON to stdout")
    output.add_argument("--output", metavar="PATH", help="write YAML, or JSON for a .json path")
    args = parser.parse_args(argv)

    assumed = {}
    for spec in args.assume_language:
        path, sep, language = spec.rpartition("=")
        if not sep or not path or not language:
            parser.error(f"--assume-language expects RUN=LANGUAGE, got {spec!r}")
        assumed[path] = language
    result = aggregate_decision_outcomes(args.runs, assumed)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _write_output(result, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
