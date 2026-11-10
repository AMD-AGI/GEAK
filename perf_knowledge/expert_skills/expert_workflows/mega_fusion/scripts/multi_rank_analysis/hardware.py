"""Portable hardware/topology context validation for analysis reports."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping

HARDWARE_CONTEXT_SCHEMA_VERSION = "geak-hardware-context-v2"
_PROVENANCE_FIELDS = (
    "vendor",
    "model",
    "arch",
    "device_count",
    "execution_units_per_device",
    "thread_group_width",
    "local_memory_bytes_per_execution_unit",
    "device_memory_bytes",
    "interconnect",
    "runtime",
)

__all__ = ["HARDWARE_CONTEXT_SCHEMA_VERSION", "validate_hardware_context"]


def _positive_int(context: Mapping[str, Any], field: str) -> int:
    value = context.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"hardware_context.{field} must be a positive integer")
    return value


def validate_hardware_context(context: Mapping[str, Any]) -> dict:
    """Validate context needed to interpret counters and candidate applicability."""
    if not isinstance(context, Mapping):
        raise TypeError("hardware_context must be a mapping")
    if context.get("schema_version") != HARDWARE_CONTEXT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported hardware context schema: {context.get('schema_version')!r}"
        )
    for field in ("vendor", "model", "arch"):
        value = context.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"hardware_context.{field} must be a non-empty string")
    _positive_int(context, "device_count")
    _positive_int(context, "execution_units_per_device")
    _positive_int(context, "thread_group_width")
    _positive_int(context, "local_memory_bytes_per_execution_unit")
    _positive_int(context, "device_memory_bytes")
    interconnect = context.get("interconnect")
    if not isinstance(interconnect, Mapping):
        raise ValueError("hardware_context.interconnect must be a mapping")
    for field in ("type", "topology"):
        value = interconnect.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"hardware_context.interconnect.{field} must be a non-empty string"
            )
    runtime = context.get("runtime")
    if not isinstance(runtime, Mapping) or not runtime:
        raise ValueError("hardware_context.runtime must be a non-empty mapping")
    for section in ("theoretical", "measured"):
        values = context.get(section, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"hardware_context.{section} must be a mapping")
        for metric, value in values.items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"hardware_context.{section}.{metric} "
                    "must be positive finite or null"
                )
    capabilities = context.get("capabilities", {})
    if not isinstance(capabilities, Mapping):
        raise ValueError("hardware_context.capabilities must be a mapping")
    provenance = context.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("hardware_context.provenance must be a mapping")
    required_provenance = list(_PROVENANCE_FIELDS)
    required_provenance.extend(
        f"measured.{metric}"
        for metric, value in context.get("measured", {}).items()
        if value is not None
    )
    required_provenance.extend(
        f"theoretical.{metric}"
        for metric, value in context.get("theoretical", {}).items()
        if value is not None
    )
    required_provenance.extend(
        f"capabilities.{capability}"
        for capability in capabilities
    )
    for field in required_provenance:
        entry = provenance.get(field)
        if not isinstance(entry, Mapping):
            raise ValueError(f"hardware_context.provenance.{field} is required")
        for key in ("collector", "timestamp", "confidence", "raw_artifact"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"hardware_context.provenance.{field}.{key} "
                    "must be a non-empty string"
                )
        if entry["confidence"] not in ("high", "medium", "low"):
            raise ValueError(
                f"hardware_context.provenance.{field}.confidence must be "
                "high, medium, or low"
            )
        try:
            datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(
                f"hardware_context.provenance.{field}.timestamp must be ISO-8601"
            ) from error
    return {
        **dict(context),
        "vendor": context["vendor"].strip(),
        "model": context["model"].strip(),
        "arch": context["arch"].strip(),
        "interconnect": dict(interconnect),
        "runtime": dict(runtime),
        "capabilities": dict(capabilities),
        "theoretical": dict(context.get("theoretical", {})),
        "measured": dict(context.get("measured", {})),
        "provenance": {
            str(field): dict(entry) for field, entry in provenance.items()
        },
    }
