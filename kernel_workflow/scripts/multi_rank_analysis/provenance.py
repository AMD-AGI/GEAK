"""Collector provenance contract for every non-UT analysis artifact."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping

COLLECTION_PROVENANCE_SCHEMA_VERSION = "geak-collection-provenance-v1"

__all__ = [
    "COLLECTION_PROVENANCE_SCHEMA_VERSION",
    "validate_collection_provenance",
]


def validate_collection_provenance(provenance: Mapping[str, Any]) -> dict:
    if not isinstance(provenance, Mapping):
        raise TypeError("collection provenance must be a mapping")
    if provenance.get("schema_version") != COLLECTION_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            "unsupported collection provenance schema: "
            f"{provenance.get('schema_version')!r}"
        )
    for field in (
        "collector_id",
        "tool_version",
        "command",
        "timestamp",
        "scope",
        "confidence",
    ):
        value = provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"collection provenance {field} must be non-empty")
    if provenance["confidence"] not in ("high", "medium", "low"):
        raise ValueError("collection provenance confidence must be high, medium, or low")
    try:
        datetime.fromisoformat(provenance["timestamp"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            "collection provenance timestamp must be ISO-8601"
        ) from error
    repetitions = provenance.get("repetitions")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions <= 0
    ):
        raise ValueError("collection provenance repetitions must be a positive integer")
    raw_artifacts = provenance.get("raw_artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("collection provenance raw_artifacts must be a non-empty list")
    if not all(isinstance(path, str) and path for path in raw_artifacts):
        raise ValueError("collection provenance raw_artifacts entries must be non-empty")
    perturbation = provenance.get("profiler_perturbation_pct")
    if perturbation is not None and (
        isinstance(perturbation, bool)
        or not isinstance(perturbation, (int, float))
        or not math.isfinite(perturbation)
        or perturbation < 0
    ):
        raise ValueError(
            "collection provenance profiler_perturbation_pct must be finite, "
            "non-negative, or null"
        )
    cross_checks = provenance.get("cross_checks", [])
    if not isinstance(cross_checks, list):
        raise ValueError("collection provenance cross_checks must be a list")
    units = provenance.get("units")
    if not isinstance(units, Mapping) or not units:
        raise ValueError("collection provenance units must be a non-empty mapping")
    if not all(
        isinstance(metric, str)
        and metric
        and isinstance(unit, str)
        and unit
        for metric, unit in units.items()
    ):
        raise ValueError(
            "collection provenance units must map non-empty strings"
        )
    return {
        **dict(provenance),
        "raw_artifacts": list(raw_artifacts),
        "cross_checks": list(cross_checks),
        "units": dict(units),
        "profiler_perturbation_pct": perturbation,
    }
