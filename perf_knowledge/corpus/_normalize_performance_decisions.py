#!/usr/bin/env python3
"""Project backend-specific GEMM knobs onto shared performance-decision axes.

Normalization preserves backend spelling and provenance.  It makes bundles comparable by question;
it does not infer that unlike representations are equivalent and never assigns speed to an axis.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_AXES = HERE / "performance_axes" / "gemm.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a top-level mapping")
    return data


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def validate_schema(schema: dict[str, Any]) -> None:
    if schema.get("schema_version") != 1:
        raise ValueError("unsupported performance-axis schema")
    axes = schema.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("performance-axis schema has no axes")
    allowed_kinds = {
        "performance_choice", "hard_constraint", "search_space",
        "structural_choice", "integration_contract",
    }
    axis_ids: set[str] = set()
    source_bindings: set[tuple[str, str]] = set()
    for axis in axes:
        axis_id = str(axis.get("id") or "")
        if not axis_id or axis_id in axis_ids:
            raise ValueError(f"missing or duplicate performance axis {axis_id!r}")
        axis_ids.add(axis_id)
        if axis.get("kind") not in allowed_kinds:
            raise ValueError(f"{axis_id}: unsupported axis kind {axis.get('kind')!r}")
        components: set[str] = set()
        aliases: dict[str, str] = {}
        for component, names in (axis.get("knob_aliases") or {}).items():
            if component in components:
                raise ValueError(f"{axis_id}: duplicate component {component!r}")
            components.add(component)
            for name in names or []:
                normalized = _key(name)
                previous = aliases.get(normalized)
                if previous and previous != component:
                    raise ValueError(
                        f"{axis_id}: alias {name!r} maps to both {previous!r} and {component!r}"
                    )
                aliases[normalized] = component
        unknown_required = set(axis.get("required_components") or []) - components
        if unknown_required:
            raise ValueError(f"{axis_id}: unknown required components {sorted(unknown_required)}")
        for mapping in axis.get("source_mappings") or []:
            if not mapping.get("representation"):
                raise ValueError(f"{axis_id}: source mapping lacks representation")
            for category in mapping.get("categories") or []:
                binding = (axis_id, str(category))
                if binding in source_bindings:
                    raise ValueError(f"{axis_id}: duplicate source category {category!r}")
                source_bindings.add(binding)
    for dependency in schema.get("dependencies") or []:
        unknown = set(dependency.get("axes") or []) - axis_ids
        if unknown:
            raise ValueError(
                f"{dependency.get('id')}: unknown dependency axes {sorted(unknown)}"
            )


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, str, Any]]:
    """Return (path, leaf-key, scalar value), retaining the original field path."""
    rows: list[tuple[str, str, Any]] = []
    if isinstance(value, dict):
        for name in sorted(value, key=str):
            if name == "performance_decisions":
                continue
            path = f"{prefix}.{name}" if prefix else str(name)
            rows.extend(_flatten(value[name], path))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        rows.append((prefix, prefix.rsplit(".", 1)[-1], value))
    return rows


def _axis_aliases(axis: dict[str, Any]) -> dict[str, str]:
    return {
        _key(alias): str(component)
        for component, aliases in (axis.get("knob_aliases") or {}).items()
        for alias in aliases or []
    }


def normalize_implementation(
    implementation: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Return a loss-aware decision vector for one complete implementation."""
    validate_schema(schema)
    language = str(implementation.get("language") or "unknown")
    config = implementation.get("config")
    fields = _flatten(config if isinstance(config, dict) else {})
    if implementation.get("name") not in (None, ""):
        fields.append(("implementation_name", "implementation_name", implementation["name"]))
    normalized_axes: dict[str, Any] = {}
    for axis in schema["axes"]:
        aliases = _axis_aliases(axis)
        matches: dict[str, list[tuple[str, Any]]] = defaultdict(list)
        for path, leaf, value in fields:
            component = aliases.get(_key(leaf))
            if component:
                matches[component].append((path, value))
        if not matches:
            continue
        values: dict[str, Any] = {}
        source_fields: dict[str, Any] = {}
        conflicts: list[str] = []
        for component in sorted(matches):
            component_rows = matches[component]
            distinct: list[Any] = []
            for _, value in component_rows:
                if value not in distinct:
                    distinct.append(value)
            values[component] = distinct[0] if len(distinct) == 1 else distinct
            paths = [path for path, _ in component_rows]
            source_fields[component] = paths[0] if len(paths) == 1 else paths
            if len(distinct) > 1:
                conflicts.append(component)
        required = set(axis.get("required_components") or [])
        present = set(values)
        completeness = (
            "complete" if required and required <= present
            else "partial" if required
            else "observed"
        )
        normalized_axes[str(axis["id"])] = {
            "values": values,
            "source_fields": source_fields,
            "completeness": completeness,
            "conflicts": conflicts,
            "comparison_contract": axis["comparison_contract"],
        }

    limitations = []
    if not isinstance(config, dict):
        limitations.append("configuration is opaque; no backend knobs were decoded")
    else:
        opaque_payload = config.get("config")
        if opaque_payload is not None and not isinstance(opaque_payload, dict):
            limitations.append(
                f"backend config payload is opaque: {opaque_payload}"
            )
        if config.get("identity_complete") is False:
            limitations.append(
                str(config.get("identity_gap") or "configuration identity is incomplete")
            )
    return {
        "schema_version": 1,
        "family": schema["family"],
        "backend": language,
        "backend_representation": (
            schema.get("backend_representations") or {}
        ).get(language, "unregistered_backend"),
        "required_comparison_context": (
            (schema.get("comparison_context") or {}).get("required") or []
        ),
        "axes": normalized_axes,
        "limitations": limitations,
        "contract": (
            "This vector aligns decision questions and preserves original fields. "
            "It is not a speed claim or a causal attribution."
        ),
    }


def _source_mapping_index(schema: dict[str, Any]) -> dict[str, list[tuple[str, str]]]:
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for axis in schema["axes"]:
        for mapping in axis.get("source_mappings") or []:
            for category in mapping.get("categories") or []:
                value = (str(axis["id"]), str(mapping["representation"]))
                if value not in index[str(category)]:
                    index[str(category)].append(value)
    return dict(index)


def summarize_source_evidence(
    source: dict[str, Any],
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Group source observations by shared axis without pretending values are equivalent."""
    validate_schema(schema)
    mappings = _source_mapping_index(schema)
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    unmapped: dict[str, int] = defaultdict(int)
    for row in source.get("source_evidence") or []:
        category = str(row.get("category") or "")
        mapped_axes = mappings.get(category) or []
        if not mapped_axes:
            unmapped[category] += 1
            continue
        language = str(row.get("language") or "unknown")
        for axis, representation in mapped_axes:
            key = (axis, language, category, representation)
            group = groups.setdefault(key, {
                "axis": axis,
                "language": language,
                "source_category": category,
                "representation": representation,
                "records": 0,
                "evidence_ids": [],
                "examples": [],
            })
            group["records"] += 1
            evidence_id = row.get("evidence_id")
            if evidence_id:
                group["evidence_ids"].append(str(evidence_id))
            example = row.get("match")
            if example and example not in group["examples"] and len(group["examples"]) < 5:
                group["examples"].append(example)

    observations = []
    for key in sorted(groups):
        group = groups[key]
        group["evidence_ids"] = sorted(set(group["evidence_ids"]))
        observations.append(group)
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in observations:
        coverage[row["axis"]][row["language"]] += row["records"]
    return {
        "observations": observations,
        "coverage": {
            axis: dict(sorted(languages.items()))
            for axis, languages in sorted(coverage.items())
        },
        "unmapped_categories": dict(sorted((k, v) for k, v in unmapped.items() if k)),
        "contract": (
            "Coverage means a backend exposes evidence for the same performance question. "
            "Different representations and values are not asserted equivalent."
        ),
    }


def normalize_tuned_configs(
    tuned: dict[str, Any],
    schema: dict[str, Any],
    config_id: str | None = None,
) -> list[dict[str, Any]]:
    profiles = []
    for row in tuned.get("tuned_configs") or []:
        if config_id and row.get("config_id") != config_id:
            continue
        vector = normalize_implementation(
            {"language": "triton", "config": row.get("knobs") or {}},
            schema,
        )
        profiles.append({
            "config_id": row.get("config_id"),
            "gfx": row.get("gfx"),
            "tags": row.get("tags"),
            "m_bucket": row.get("m_bucket"),
            "source_files": row.get("source_files") or [],
            "performance_decisions": vector,
        })
    return profiles


def axis_summary(schema: dict[str, Any]) -> list[dict[str, Any]]:
    validate_schema(schema)
    return [{
        "id": axis["id"],
        "kind": axis["kind"],
        "question": axis["question"],
        "comparison_contract": axis["comparison_contract"],
        "components": sorted((axis.get("knob_aliases") or {}).keys()),
        "source_categories": sorted({
            str(category)
            for mapping in axis.get("source_mappings") or []
            for category in mapping.get("categories") or []
        }),
    } for axis in schema["axes"]]


def model_summary(schema: dict[str, Any]) -> dict[str, Any]:
    validate_schema(schema)
    return {
        "axes": axis_summary(schema),
        "comparison_context": schema.get("comparison_context") or {},
        "dependencies": schema.get("dependencies") or [],
        "contract": schema.get("contract") or {},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", type=Path, default=DEFAULT_AXES)
    parser.add_argument("--source-evidence", type=Path)
    parser.add_argument("--tuned-evidence", type=Path)
    parser.add_argument("--config-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    schema = load_yaml(args.axes)
    result: dict[str, Any] = {
        "schema_version": 1,
        "family": schema.get("family"),
        **model_summary(schema),
    }
    if args.source_evidence:
        result["source_evidence"] = summarize_source_evidence(
            load_yaml(args.source_evidence), schema,
        )
    if args.tuned_evidence:
        result["tuned_profiles"] = normalize_tuned_configs(
            load_yaml(args.tuned_evidence), schema, args.config_id,
        )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
