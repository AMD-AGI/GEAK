"""Neutral top-level report schema for multi-rank analysis output.

Generalizes the shape of the one existing multi-rank analysis artifact this framework was built
from (schema ``geak-megamoe-analysis-v1``) into an operator-agnostic version: ``primary_metric`` and
every ``case_id`` are caller-supplied strings rather than hardcoded to one operator's name. See this
package's README for the full field-by-field mapping to that artifact.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "geak-multirank-analysis-v2"

__all__ = ["SCHEMA_VERSION", "build_report"]


def build_report(
    primary_metric: Mapping[str, Any],
    cases: Sequence[dict],
    route_comparisons: Sequence[dict] | None = None,
    status: str = "pass",
    secondary_comparator_role: str | None = None,
    hardware_context: dict | None = None,
    measurement_tracks: dict | None = None,
    experiment_manifest: dict | None = None,
    workload: dict | None = None,
    source: dict | None = None,
    expected_world_size: int | None = None,
    metric_definitions: dict | None = None,
) -> dict:
    """Assemble the top-level report object.

    ``primary_metric``: typed path/unit/direction/reduction/semantic contract plus description.
    ``cases``: list of case dicts, each expected to at least carry a ``case_id`` — but this module
        does not enforce a case's internal shape (that is the caller's/Skill's concern; this
        function only assembles the envelope).
    ``route_comparisons``: optional list of cross-case delta dicts (e.g. uniform vs. skewed route).
    ``secondary_comparator_role``: optional free-text note (e.g. "secondary comparison only; never
        the speedup denominator") — generalizes the artifact's ``mori_role`` field without naming
        MORI; omitted entirely when the caller has no secondary comparator.
    ``measurement_tracks``: optional resolved completion/evidence map consumed by comprehensive analysis
        skills; absent tracks must not be inferred.
    ``hardware_context``: optional device/topology/runtime/capability context used to interpret
        counters and constrain candidate applicability; it does not select an implementation.
    ``experiment_manifest``: optional normalized controlled-experiment contract.
    """
    if status not in ("pass", "partial", "fail"):
        raise ValueError("report status must be pass, partial, or fail")
    if not isinstance(primary_metric, Mapping):
        raise ValueError("primary_metric must be a mapping")
    for field in (
        "description",
        "path",
        "unit",
        "direction",
        "reduction",
        "semantic",
    ):
        if not isinstance(primary_metric.get(field), str) or not primary_metric[field]:
            raise ValueError(f"primary_metric requires non-empty {field}")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "primary_metric": dict(primary_metric),
        "cases": list(cases),
    }
    if workload:
        report["workload"] = dict(workload)
    if source:
        report["source"] = dict(source)
    if expected_world_size is not None:
        report["expected_world_size"] = expected_world_size
    if metric_definitions:
        report["metric_definitions"] = dict(metric_definitions)
    if secondary_comparator_role:
        report["secondary_comparator_role"] = secondary_comparator_role
    if route_comparisons:
        report["route_comparisons"] = list(route_comparisons)
    if hardware_context:
        report["hardware_context"] = dict(hardware_context)
    if measurement_tracks:
        report["measurement_tracks"] = dict(measurement_tracks)
    if experiment_manifest:
        report["experiment_manifest"] = dict(experiment_manifest)
    return report
