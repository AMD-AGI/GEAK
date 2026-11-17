"""Typed evidence catalog and measurement-track reference resolution."""

from __future__ import annotations

from typing import Any, Mapping

EVIDENCE_CATALOG_SCHEMA_VERSION = "geak-evidence-catalog-v1"

__all__ = [
    "EVIDENCE_CATALOG_SCHEMA_VERSION",
    "resolve_measurement_tracks",
    "validate_evidence_catalog",
]


def validate_evidence_catalog(catalog: Mapping[str, Any]) -> dict:
    if not isinstance(catalog, Mapping):
        raise TypeError("evidence catalog must be a mapping")
    if catalog.get("schema_version") != EVIDENCE_CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported evidence catalog schema: {catalog.get('schema_version')!r}"
        )
    entries = catalog.get("entries")
    if not isinstance(entries, Mapping):
        raise ValueError("evidence catalog entries must be a mapping")
    provenance = catalog.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("evidence catalog provenance must be a mapping")
    normalized_provenance = {}
    for provenance_id, entry in provenance.items():
        if not isinstance(provenance_id, str) or not provenance_id:
            raise ValueError("provenance IDs must be non-empty strings")
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"provenance {provenance_id!r} must be a mapping"
            )
        if entry.get("status") not in ("complete", "partial", "invalid"):
            raise ValueError(
                f"provenance {provenance_id!r} has invalid status"
            )
        kind = entry.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"provenance {provenance_id!r} requires kind")
        normalized_provenance[provenance_id] = dict(entry)
    normalized_entries = {}
    metric_owners = {}
    provenance_refs = set()
    for evidence_id, entry in entries.items():
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("evidence IDs must be non-empty strings")
        if not isinstance(entry, Mapping):
            raise ValueError(f"evidence {evidence_id!r} must be a mapping")
        if entry.get("status") not in ("complete", "partial", "invalid"):
            raise ValueError(f"evidence {evidence_id!r} has invalid status")
        kind = entry.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"evidence {evidence_id!r} requires kind")
        metric_ids = entry.get("metric_ids", [])
        refs = entry.get("provenance_refs", [])
        if not isinstance(metric_ids, list) or not all(
            isinstance(metric, str) and metric for metric in metric_ids
        ):
            raise ValueError(
                f"evidence {evidence_id!r} metric_ids must be a string list"
            )
        if not isinstance(refs, list) or not all(
            isinstance(ref, str) and ref for ref in refs
        ):
            raise ValueError(
                f"evidence {evidence_id!r} provenance_refs must be a string list"
            )
        for metric_id in metric_ids:
            if metric_id in metric_owners:
                raise ValueError(
                    f"duplicate evidence metric ID {metric_id!r}"
                )
            metric_owners[metric_id] = evidence_id
        provenance_refs.update(refs)
        normalized_entries[evidence_id] = {
            **dict(entry),
            "metric_ids": list(metric_ids),
            "provenance_refs": list(refs),
        }
    unresolved_provenance = sorted(
        provenance_refs - set(normalized_provenance)
    )
    if unresolved_provenance:
        raise ValueError(
            "evidence catalog has unresolved provenance refs: "
            f"{unresolved_provenance}"
        )
    return {
        "schema_version": EVIDENCE_CATALOG_SCHEMA_VERSION,
        "entries": normalized_entries,
        "metric_owners": metric_owners,
        "provenance_refs": sorted(provenance_refs),
        "provenance": normalized_provenance,
    }


def resolve_measurement_tracks(
    tracks: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> dict:
    validated_catalog = validate_evidence_catalog(catalog)
    entries = validated_catalog["entries"]
    metric_owners = validated_catalog["metric_owners"]
    known_provenance = {
        provenance_id
        for provenance_id, entry in validated_catalog["provenance"].items()
        if entry["status"] == "complete"
    }
    resolved = {}
    for name, raw_track in tracks.items():
        track = dict(raw_track)
        if track.get("status") != "complete":
            resolved[str(name)] = track
            continue
        evidence = track.get("evidence")
        errors = []
        if not isinstance(evidence, Mapping):
            errors.append("complete evidence must be a mapping")
            evidence = {}
        artifact_refs = evidence.get("artifact_refs", [])
        metric_refs = evidence.get("metrics", [])
        provenance_refs = evidence.get("provenance_refs", [])
        missing_artifacts = sorted(
            ref
            for ref in artifact_refs
            if ref not in entries or entries[ref]["status"] != "complete"
        )
        missing_metrics = sorted(
            metric for metric in metric_refs if metric not in metric_owners
        )
        missing_provenance = sorted(
            ref for ref in provenance_refs if ref not in known_provenance
        )
        if missing_artifacts:
            errors.append(f"unresolved/incomplete artifacts: {missing_artifacts}")
        if missing_metrics:
            errors.append(f"unresolved metrics: {missing_metrics}")
        if missing_provenance:
            errors.append(f"unresolved provenance: {missing_provenance}")
        if errors:
            track["status"] = "invalid"
            track["resolution_errors"] = errors
        else:
            track["resolved_evidence"] = {
                "artifacts": list(artifact_refs),
                "metric_owners": {
                    metric: metric_owners[metric] for metric in metric_refs
                },
                "provenance_refs": list(provenance_refs),
            }
        resolved[str(name)] = track
    return resolved
