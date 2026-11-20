#!/usr/bin/env python3
"""Canonical, machine-readable records for optimization runs.

This module owns the portable artifact contract.  It distinguishes a measured value
from its identity, a stage outcome from final arbitration, and an arm's verdict from
whether that arm completed.  Call ``validate`` before a record crosses a worker,
captain, or finalization boundary; call ``reconcile`` or ``finalize`` for a final run.

IN GEAK this is optional bookkeeping a deep_engineer may use for its records, not a run mode.
Role ids such as `captain` / `deep` / `skeptic` are upstream record-schema identifiers (`deep` = the
deep_engineer, pass `--role deep`); nothing spawns them, and final arbitration in GEAK is Director's.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
from typing import Any


MEASUREMENT_SCHEMA = "kernel_opt.measurement/1"
STAGE_RESULT_SCHEMA = "kernel_opt.stage_result/1"
RUN_DECISION_SCHEMA = "kernel_opt.run_decision/1"
ARM_RESULT_SCHEMA = "kernel_opt.arm_result/1"
BRANCH_CONVERGE_SCHEMA = "kernel_opt.branch_converge/1"
WORKER_RESULT_SCHEMA = "kernel_opt.worker_result/1"
BRANCH_REQUEST_SCHEMA = "kernel_opt.branch_request/1"
STRUCTURE_SUSPECT_SCHEMA = "structure_suspect/1"
RESWEEP_REQUEST_SCHEMA = "kernel_opt.resweep_request/1"
RESWEEP_RESULT_SCHEMA = "kernel_opt.resweep_result/1"
SWEEP_REQUEST_SCHEMA = "kernel_opt.sweep_request/1"
SWEEP_RESULT_SCHEMA = "kernel_opt.sweep_result/1"
FINAL_REPORT_SCHEMA = "kernel_opt.final_report/3"
LEGACY_FINAL_REPORT_SCHEMAS = frozenset({
    "kernel_opt.final_report/1",
    "kernel_opt.final_report/2",
    "kernel_opt.final_report",
    "kernel_opt_run.final_report",
    "kernel_opt_run.final_report/1",
})
PRODUCER_RECEIPT_SCHEMA = "kernel_opt.producer_receipt/1"
LEGACY_ARTIFACT_REF_SCHEMA = "kernel_opt.legacy_artifact_ref/1"
ARTIFACT_REF_SCHEMA = "kernel_opt.artifact_ref/1"
LIFECYCLE_PROFILES = ("front-end", "deep-consumer", "direct-owner", "blocked")
PRODUCER_ARTIFACT_ROLES = ("sweep", "branch", "arm", "champion", "audit", "final_report")

PHASES = ("anchor", "search", "branch", "converge", "resweep", "finalization")
STAGE_STATUSES = ("completed", "skipped", "blocked", "superseded")
TIMELINE_EVENTS = ("entered", "completed", "reopened", "superseded", "finalized")
OUTCOMES = ("win", "partial", "negative_keep_baseline", "negative_revert_plain", "timeout")
ARBITRATIONS = ("accept", "reject")
RUN_STATUSES = ("closed", "deferred_needs_user", "blocked", "partial_time_limit")
RUN_DECISION_STATES = ("open", "finalizing", *RUN_STATUSES)
FINAL_REPORT_OWNERS = ("captain", "direct_owner")
BRANCH_CONVERGE_OWNERS = ("direction", "direct_owner")
WORKER_ROLES = ("direction", "deep", "independent_direction")
WORKER_STATUSES = ("running", "completed", "blocked", "deferred_needs_user")
MEASUREMENT_UNITS = ("ms", "us", "ns", "ratio", "percent", "cycles", "unknown")
MEASUREMENT_SOURCES = (
    "device_timing", "host_timing", "derived", "static_model",
    "legacy_latency", "legacy_ratio", "unknown",
)
MEASUREMENT_SCOPES = ("device", "host", "derived", "unknown")
ARM_VERDICTS = ("supported", "not_supported", "inconclusive")
ARM_COMPLETION = ("complete", "truncated", "crashed")
ARM_DIRECTION_CLASSES = ("structural", "knob")
ARM_GATE_STAGES = ("a", "b", "c")
POST_MERGE_RESWEEP = ("completed", "not_required")
REQUEST_STATES = ("requested", "consumed", "resolved")
RESWEEP_REQUESTERS = ("deep", "captain", "direct_owner")
RESWEEP_PARENT_SCHEMAS = (WORKER_RESULT_SCHEMA, BRANCH_CONVERGE_SCHEMA)
SWEEP_PURPOSES = ("initial", "arm_local", "post_merge", "body_refresh", "handoff")
SWEEP_REQUESTERS = ("arm", "captain", "direct_owner")
SWEEP_EXECUTORS = ("triton_sweep",) + FINAL_REPORT_OWNERS
SWEEP_WORK_KINDS = ("sweep",)
CHANGE_SCOPES = ("config", "body", "layout", "launch", "pipeline")
AXIS_KINDS = ("config", "shape", "layout", "launch", "resource", "pipeline")
REALIZATION_KINDS = ("request", "execution")
FINAL_REPORT_BASE_FIELDS = frozenset({
    "schema", "run_id", "generation", "stage", "outcome", "arbitration", "status",
    "measurement", "canonical_ref", "champion_ref", "champion_gate_ref", "anchor",
    "close_audit_ref", "scope", "deferred", "deep_arbitration", "known_wrong", "producer",
    "lifecycle_profile", "refs", "headline", "served_range", "audit", "provenance", "integrity",
})

# Public JSON Schema descriptors.  The validators below add relationship checks
# (set inclusion, identity equality, and state-dependent requirements) that JSON
# Schema alone cannot express portably.
_REF = {"anyOf": [{"type": "string", "minLength": 1},
                  {"type": "object", "required": ["kind"],
                   "properties": {"kind": {"type": "string", "minLength": 1}}}]}
_MEASUREMENT_IDENTITY = {
    "type": "object",
    "required": ["shape_set", "aggregation", "unit", "boundary", "source", "scope", "sample",
                 "baseline_ref", "comparator_ref"],
    "properties": {
        "shape_set": {"type": "array", "minItems": 1, "uniqueItems": True,
                      "items": {"type": "string", "minLength": 1}},
        "aggregation": {"type": "string", "minLength": 1},
        "unit": {"enum": list(MEASUREMENT_UNITS)},
        "boundary": {"type": "string", "minLength": 1},
        "source": {"enum": list(MEASUREMENT_SOURCES)},
        "scope": {"enum": list(MEASUREMENT_SCOPES)},
        "sample": {"type": "object", "required": ["count", "method"],
                   "properties": {"count": {"anyOf": [{"type": "integer", "minimum": 1},
                                                       {"const": "unknown"}]},
                                  "method": {"type": "string", "minLength": 1}}},
        "baseline_ref": _REF,
        "comparator_ref": _REF,
    },
}
MEASUREMENT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": MEASUREMENT_SCHEMA,
    "type": "object",
    "required": ["schema", "measurement_id", "value", "collected_at", "identity"],
    "properties": {
        "schema": {"const": MEASUREMENT_SCHEMA},
        "measurement_id": {"type": "string", "minLength": 1},
        "value": {"type": "number"},
        "collected_at": {"type": "string", "format": "date-time"},
        "identity": _MEASUREMENT_IDENTITY,
    },
}
STAGE_RESULT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": STAGE_RESULT_SCHEMA,
    "type": "object",
    "required": ["schema", "stage_id", "phase", "status", "collected_at", "profile_ref"],
    "properties": {
        "schema": {"const": STAGE_RESULT_SCHEMA},
        "stage_id": {"type": "string", "minLength": 1},
        "phase": {"enum": list(PHASES)},
        "status": {"enum": list(STAGE_STATUSES)},
        "collected_at": {"type": "string", "format": "date-time"},
        "profile_ref": _REF,
        "measurement": MEASUREMENT_JSON_SCHEMA,
    },
}
ARM_RESULT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": ARM_RESULT_SCHEMA,
    "type": "object",
    "required": ["schema", "arm_id", "lever", "direction_class", "iters_used", "completion",
                 "verdict", "gate_stage"],
    "properties": {
        "schema": {"const": ARM_RESULT_SCHEMA},
        "arm_id": {"type": "string", "minLength": 1},
        "lever": {"type": "string", "minLength": 1},
        "direction_class": {"enum": list(ARM_DIRECTION_CLASSES)},
        "iters_used": {"type": "integer", "minimum": 0},
        "completion": {"enum": list(ARM_COMPLETION)},
        "verdict": {"enum": list(ARM_VERDICTS)},
        "gate_stage": {"enum": list(ARM_GATE_STAGES)},
        "measurement": MEASUREMENT_JSON_SCHEMA,
    },
}
WORKER_RESULT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": WORKER_RESULT_SCHEMA,
    "type": "object",
    "required": ["schema", "run_id", "generation", "producer_role", "stage", "status",
                 "produced_at", "measurement"],
    "properties": {
        "schema": {"const": WORKER_RESULT_SCHEMA},
        "run_id": {"type": "string", "minLength": 1},
        "generation": {"type": "integer", "minimum": 1},
        "producer_role": {"enum": list(WORKER_ROLES)},
        "stage": {"enum": list(PHASES)},
        "status": {"enum": list(WORKER_STATUSES)},
        "produced_at": {"type": "string", "format": "date-time"},
        "measurement": MEASUREMENT_JSON_SCHEMA,
        "requests": {"type": "array"},
        "work_kinds": {"type": "array", "items": {"type": "string"}},
    },
}
BRANCH_REQUEST_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": BRANCH_REQUEST_SCHEMA,
    "type": "object",
    "required": ["schema", "request_id", "run_id", "generation", "requester_role",
                 "requested_at", "candidates", "evidence_ref"],
    "properties": {
        "schema": {"const": BRANCH_REQUEST_SCHEMA},
        "request_id": {"type": "string", "minLength": 1},
        "run_id": {"type": "string", "minLength": 1},
        "generation": {"type": "integer", "minimum": 1},
        "requester_role": {"const": "direction"},
        "requested_at": {"type": "string", "format": "date-time"},
        "candidates": {"type": "array", "minItems": 1},
        "evidence_ref": _REF,
        "state": {"enum": list(REQUEST_STATES)},
    },
}
STRUCTURE_SUSPECT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": STRUCTURE_SUSPECT_SCHEMA,
    "type": "object",
    "required": ["schema", "candidate_id", "state", "independent", "candidate_ref",
                 "source_ref", "evidence_ref"],
    "properties": {
        "schema": {"const": STRUCTURE_SUSPECT_SCHEMA},
        "candidate_id": {"type": "string", "minLength": 1},
        "state": {"enum": ["untried", "deferred"]},
        "independent": {"type": "boolean"},
        "candidate_ref": _REF,
        "source_ref": _REF,
        "evidence_ref": _REF,
        "requester_role": {"enum": list(RESWEEP_REQUESTERS)},
    },
}
RESWEEP_REQUEST_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": RESWEEP_REQUEST_SCHEMA,
    "type": "object",
    "required": ["schema", "request_id", "run_id", "generation", "requester_role",
                 "requested_at", "parent_result_ref", "evidence_ref", "state"],
    "properties": {
        "schema": {"const": RESWEEP_REQUEST_SCHEMA},
        "request_id": {"type": "string", "minLength": 1},
        "run_id": {"type": "string", "minLength": 1},
        "generation": {"type": "integer", "minimum": 1},
        "requester_role": {"const": "deep"},
        "requested_at": {"type": "string", "format": "date-time"},
        "parent_result_ref": _REF,
        "evidence_ref": _REF,
        "state": {"const": "requested"},
    },
}
RESWEEP_RESULT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": RESWEEP_RESULT_SCHEMA,
    "type": "object",
    "required": ["schema", "request_ref", "run_id", "generation", "executor_role",
                 "completed_at", "measurement"],
    "properties": {
        "schema": {"const": RESWEEP_RESULT_SCHEMA},
        "request_ref": _REF,
        "run_id": {"type": "string", "minLength": 1},
        "generation": {"type": "integer", "minimum": 1},
        "executor_role": {"enum": list(FINAL_REPORT_OWNERS)},
        "completed_at": {"type": "string", "format": "date-time"},
        "measurement": MEASUREMENT_JSON_SCHEMA,
    },
}
_SWEEP_ATTEMPT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["attempt_id"],
    "properties": {"attempt_id": {"type": "string", "minLength": 1}},
}
_SWEEP_LINEAGE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lineage_id", "root_request_id"],
    "properties": {
        "lineage_id": {"type": "string", "minLength": 1},
        "root_request_id": {"type": "string", "minLength": 1},
    },
}
SWEEP_REQUEST_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": SWEEP_REQUEST_SCHEMA,
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema", "request_id", "run_id", "generation", "purpose", "requester_role",
        "requested_at", "source", "context", "parent", "attempt", "lineage", "state",
        "work_kind", "change_scope", "axis_kind", "realization_kind",
    ],
    "properties": {
        "schema": {"const": SWEEP_REQUEST_SCHEMA},
        "request_id": {"type": "string", "minLength": 1},
        "run_id": {"type": "string", "minLength": 1},
        "generation": {"type": "integer", "minimum": 1},
        "purpose": {"enum": list(SWEEP_PURPOSES)},
        "requester_role": {"enum": list(SWEEP_REQUESTERS)},
        "requested_at": {"type": "string", "format": "date-time"},
        "source": _REF,
        "context": {"type": "object", "minProperties": 1},
        "parent": _REF,
        "attempt": _SWEEP_ATTEMPT,
        "lineage": _SWEEP_LINEAGE,
        "state": {"const": "requested"},
        "work_kind": {"enum": list(SWEEP_WORK_KINDS)},
        "change_scope": {"enum": list(CHANGE_SCOPES)},
        "axis_kind": {"enum": list(AXIS_KINDS)},
        "realization_kind": {"const": "request"},
    },
}
SWEEP_RESULT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": SWEEP_RESULT_SCHEMA,
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema", "request_ref", "run_id", "generation", "purpose", "executor_role",
        "completed_at", "measurement", "attempt", "lineage", "work_kind", "change_scope",
        "axis_kind", "realization_kind",
    ],
    "properties": {
        "schema": {"const": SWEEP_RESULT_SCHEMA},
        "request_ref": _REF,
        "run_id": {"type": "string", "minLength": 1},
        "generation": {"type": "integer", "minimum": 1},
        "purpose": {"enum": list(SWEEP_PURPOSES)},
        "executor_role": {"enum": list(SWEEP_EXECUTORS)},
        "completed_at": {"type": "string", "format": "date-time"},
        "measurement": MEASUREMENT_JSON_SCHEMA,
        "attempt": _SWEEP_ATTEMPT,
        "lineage": _SWEEP_LINEAGE,
        "work_kind": {"enum": list(SWEEP_WORK_KINDS)},
        "change_scope": {"enum": list(CHANGE_SCOPES)},
        "axis_kind": {"enum": list(AXIS_KINDS)},
        "realization_kind": {"const": "execution"},
    },
}
BRANCH_CONVERGE_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": BRANCH_CONVERGE_SCHEMA,
    "type": "object",
    "required": ["schema", "planned", "completed", "eligible", "winner", "measurement",
                 "collected_at", "arm_results", "post_merge_resweep", "direction_decision",
                 "producer"],
    "properties": {
        "schema": {"const": BRANCH_CONVERGE_SCHEMA},
        "run_id": {"type": "string", "minLength": 1},
        "generation": {"type": "integer", "minimum": 1},
        "planned": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True},
        "completed": {"type": "array", "items": {"type": "string", "minLength": 1},
                      "uniqueItems": True},
        "eligible": {"type": "array", "items": {"type": "string", "minLength": 1},
                     "uniqueItems": True},
        "winner": {"type": ["string", "null"]},
        "measurement": MEASUREMENT_JSON_SCHEMA,
        "collected_at": {"type": "string", "format": "date-time"},
        "arm_results": {"type": "array"},
        "post_merge_resweep": {"type": "object"},
        "direction_decision": {"type": "object", "required": ["winner", "evidence_ref"],
                               "properties": {"winner": {"type": ["string", "null"]},
                                              "evidence_ref": _REF}},
        "producer": {"type": "object", "required": ["role"],
                     "properties": {"role": {"enum": list(BRANCH_CONVERGE_OWNERS)}}},
    },
}
RUN_DECISION_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": RUN_DECISION_SCHEMA,
    "type": "object",
    "required": ["schema", "run_id", "generation", "state", "producer", "measurement",
                 "timeline", "stages"],
    "properties": {
        "schema": {"const": RUN_DECISION_SCHEMA},
        "run_id": {"type": "string", "minLength": 1},
        "generation": {"type": "integer", "minimum": 1},
        "state": {"enum": list(RUN_DECISION_STATES)},
        "producer": {"type": "object", "required": ["role"],
                     "properties": {"role": {"enum": list(FINAL_REPORT_OWNERS)}}},
        "measurement": MEASUREMENT_JSON_SCHEMA,
        "timeline": {"type": "array"},
        "stages": {"type": "array", "minItems": 1, "items": STAGE_RESULT_JSON_SCHEMA},
        "final": {"type": "object"},
        "branch_converge": BRANCH_CONVERGE_JSON_SCHEMA,
    },
}
FINAL_REPORT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": FINAL_REPORT_SCHEMA,
    "type": "object",
    "required": ["schema", "run_id", "generation", "lifecycle_profile", "stage", "outcome",
                 "arbitration", "status", "refs", "audit", "provenance", "integrity",
                 "known_wrong", "producer"],
    "properties": {
        "schema": {"const": FINAL_REPORT_SCHEMA},
        "run_id": {"type": "string", "minLength": 1},
        "generation": {"type": "integer", "minimum": 1},
        "lifecycle_profile": {"enum": list(LIFECYCLE_PROFILES)},
        "stage": {"enum": list(PHASES)},
        "outcome": {"enum": list(OUTCOMES)},
        "arbitration": {"enum": list(ARBITRATIONS)},
        "status": {"enum": list(RUN_STATUSES)},
        "measurement": MEASUREMENT_JSON_SCHEMA,
        "canonical_ref": _REF,
        "champion_ref": _REF,
        "champion_gate_ref": _REF,
        "anchor": {"type": "object", "required": ["type", "ref"],
                   "properties": {"type": {"enum": ["pinned_comparator",
                                                     "production_default_anchor"]},
                                  "ref": _REF}},
        "close_audit_ref": _REF,
        "scope": {"type": "object", "required": ["authorization"],
                  "properties": {"authorization": {"type": "string", "minLength": 1}}},
        "deferred": {"type": "array"},
        "deep_arbitration": {
            "type": "object",
            "required": ["state", "champion_ms"],
            "properties": {
                "state": {"enum": ["not_authorized", "not_run", "compared"]},
                "champion_ms": {"type": "number"},
                "result_ref": {"anyOf": [_REF, {"type": "null"}]},
            },
        },
        "known_wrong": {"type": "array"},
        "refs": {"type": "object"},
        "headline": {"type": "object"},
        "served_range": {"type": "array"},
        "audit": {"type": "object"},
        "provenance": {"type": "object", "required": ["receipts"],
                       "properties": {"receipts": {"type": "array"}}},
        "integrity": {"type": "object", "required": [
            "source_schema", "normalization_status", "legacy_unverified", "eligible_clean"]},
        "producer": {"type": "object", "required": ["role"],
                     "properties": {"role": {"enum": list(FINAL_REPORT_OWNERS)}}},
    },
}
PRODUCER_RECEIPT_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": PRODUCER_RECEIPT_SCHEMA,
    "type": "object",
    "required": ["schema", "receipt_id", "artifact_role", "produced_at", "timing",
                 "producer", "artifact", "invariant_validator"],
    "properties": {
        "schema": {"const": PRODUCER_RECEIPT_SCHEMA},
        "artifact_role": {"enum": list(PRODUCER_ARTIFACT_ROLES)},
        "timing": {"enum": ["process_time", "late_reconstruction"]},
    },
}
JSON_SCHEMAS = {
    "measurement": MEASUREMENT_JSON_SCHEMA,
    "stage-result": STAGE_RESULT_JSON_SCHEMA,
    "run-decision": RUN_DECISION_JSON_SCHEMA,
    "arm-result": ARM_RESULT_JSON_SCHEMA,
    "worker-result": WORKER_RESULT_JSON_SCHEMA,
    "branch-request": BRANCH_REQUEST_JSON_SCHEMA,
    "structure-suspect": STRUCTURE_SUSPECT_JSON_SCHEMA,
    "resweep-request": RESWEEP_REQUEST_JSON_SCHEMA,
    "resweep-result": RESWEEP_RESULT_JSON_SCHEMA,
    "sweep-request": SWEEP_REQUEST_JSON_SCHEMA,
    "sweep-result": SWEEP_RESULT_JSON_SCHEMA,
    "branch-converge": BRANCH_CONVERGE_JSON_SCHEMA,
    "final-report": FINAL_REPORT_JSON_SCHEMA,
    "producer-receipt": PRODUCER_RECEIPT_JSON_SCHEMA,
}


def _finding(out: list[dict], field: str, code: str, detail: str) -> None:
    out.append({"field": field, "code": code, "detail": detail, "severity": "error"})


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _timestamp(value: Any) -> bool:
    if not _nonempty(value):
        return False
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _reference(value: Any) -> bool:
    """A reference is a non-empty path/id string or an explicit typed object."""
    if _nonempty(value):
        return True
    return isinstance(value, dict) and (
        _nonempty(value.get("kind")) or _nonempty(value.get("uri")))


def _path_reference(value: Any) -> str | None:
    """Return the path component of a portable string/typed artifact reference."""
    if _nonempty(value):
        return value
    if isinstance(value, dict):
        for key in ("path", "artifact", "uri"):
            if _nonempty(value.get(key)):
                uri = value[key]
                return uri[len("work:/"):] if uri.startswith("work:/") else uri
    return None


def _known_reference(value: Any) -> bool:
    """A device comparison cannot use a typed ``unknown`` placeholder."""
    if _nonempty(value):
        return True
    return (isinstance(value, dict)
            and (_nonempty(value.get("kind")) or _nonempty(value.get("uri")))
            and any(_nonempty(value.get(key)) for key in ("path", "id", "value", "uri")))


def _enum(out: list[dict], doc: dict, field: str, allowed: tuple[str, ...],
          required: bool = True) -> str | None:
    value = doc.get(field)
    if value is None:
        if required:
            _finding(out, field, "missing", f"expected one of {list(allowed)}")
        return None
    if not isinstance(value, str) or value not in allowed:
        _finding(out, field, "enum", f"expected one of {list(allowed)}, got {value!r}")
        return None
    return value


def identity_of(measurement: dict | None) -> dict | None:
    """Return the exact fields that determine whether measurements are comparable."""
    if not isinstance(measurement, dict) or not isinstance(measurement.get("identity"), dict):
        return None
    return measurement["identity"]


def same_identity(left: dict | None, right: dict | None) -> bool:
    """Comparison identity is exact; unknown is a value, never a wildcard."""
    a, b = identity_of(left), identity_of(right)
    return a is not None and b is not None and a == b


def validate_measurement(doc: Any, prefix: str = "measurement") -> list[dict]:
    """Validate a timing/rate value without inferring provenance from its numeric unit."""
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != MEASUREMENT_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {MEASUREMENT_SCHEMA!r}")
    if not _nonempty(doc.get("measurement_id")):
        _finding(out, f"{prefix}.measurement_id", "missing", "must be a stable non-empty id")
    if not _number(doc.get("value")):
        _finding(out, f"{prefix}.value", "type", "must be a finite number")
    if not _timestamp(doc.get("collected_at")):
        _finding(out, f"{prefix}.collected_at", "timestamp", "must be an ISO-8601 timestamp")

    identity = doc.get("identity")
    if not isinstance(identity, dict):
        _finding(out, f"{prefix}.identity", "missing", "expected measurement identity object")
        return out
    shapes = identity.get("shape_set")
    if (not isinstance(shapes, list) or not shapes or
            any(not _nonempty(x) for x in shapes) or len(set(shapes)) != len(shapes)):
        _finding(out, f"{prefix}.identity.shape_set", "shape_set",
                 "must be a non-empty list of distinct non-empty shape ids")
    elif "unknown" in shapes and len(shapes) != 1:
        _finding(out, f"{prefix}.identity.shape_set", "shape_set",
                 "'unknown' cannot be mixed with named shapes")
    if not _nonempty(identity.get("aggregation")):
        _finding(out, f"{prefix}.identity.aggregation", "missing",
                 "must name the aggregation, including 'unspecified' when legacy")
    _enum(out, identity, "unit", MEASUREMENT_UNITS)
    source = _enum(out, identity, "source", MEASUREMENT_SOURCES)
    scope = _enum(out, identity, "scope", MEASUREMENT_SCOPES)
    if not _nonempty(identity.get("boundary")):
        _finding(out, f"{prefix}.identity.boundary", "missing",
                 "must name the timing boundary, including 'unknown' when legacy")
    for ref in ("baseline_ref", "comparator_ref"):
        if not _reference(identity.get(ref)):
            _finding(out, f"{prefix}.identity.{ref}", "missing",
                     "must be a reference or an explicit {kind: ...} unknown reference")

    sample = identity.get("sample")
    if not isinstance(sample, dict):
        _finding(out, f"{prefix}.identity.sample", "missing",
                 "must describe sample count/method, or explicitly mark both unknown")
    elif sample.get("count") != "unknown" and (
            not isinstance(sample.get("count"), int) or isinstance(sample.get("count"), bool)
            or sample["count"] < 1):
        _finding(out, f"{prefix}.identity.sample.count", "sample",
                 "must be a positive integer or 'unknown'")
    elif not _nonempty(sample.get("method")):
        _finding(out, f"{prefix}.identity.sample.method", "missing",
                 "must name the sampling method, including 'unknown' when legacy")

    # A latency field says only a unit.  This is the guard that prevents an old
    # ``latency_ms`` from being silently promoted to a measured device duration.
    if source == "device_timing":
        if scope != "device":
            _finding(out, f"{prefix}.identity.scope", "device_scope",
                     "source=device_timing requires scope=device")
        if identity.get("boundary") == "unknown":
            _finding(out, f"{prefix}.identity.boundary", "device_boundary",
                     "source=device_timing requires a named boundary")
        if not isinstance(sample, dict) or not isinstance(sample.get("count"), int):
            _finding(out, f"{prefix}.identity.sample.count", "device_sample",
                     "source=device_timing requires a positive observed sample count")
        for ref in ("baseline_ref", "comparator_ref"):
            if not _known_reference(identity.get(ref)):
                _finding(out, f"{prefix}.identity.{ref}", "comparator_missing",
                         "source=device_timing requires a dereferenceable baseline/comparator")
    if source in ("legacy_latency", "legacy_ratio") and scope != "unknown":
        _finding(out, f"{prefix}.identity.scope", "legacy_scope",
                 "legacy measurements must retain scope=unknown")
    return out


def validate_timeline(events: Any, prefix: str = "timeline") -> list[dict]:
    out: list[dict] = []
    if not isinstance(events, list):
        _finding(out, prefix, "type", "expected a list")
        return out
    ids, sequences, last_sequence = set(), set(), 0
    for i, event in enumerate(events):
        path = f"{prefix}[{i}]"
        if not isinstance(event, dict):
            _finding(out, path, "type", "expected an object")
            continue
        event_id = event.get("event_id")
        if not _nonempty(event_id) or event_id in ids:
            _finding(out, f"{path}.event_id", "unique", "must be a unique non-empty id")
        ids.add(event_id)
        _enum(out, event, "phase", PHASES)
        _enum(out, event, "event", TIMELINE_EVENTS)
        if not _timestamp(event.get("at")):
            _finding(out, f"{path}.at", "timestamp", "must be an ISO-8601 timestamp")
        seq = event.get("sequence")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1 or seq in sequences:
            _finding(out, f"{path}.sequence", "sequence", "must be a unique positive integer")
        elif seq <= last_sequence:
            _finding(out, f"{path}.sequence", "order", "must be strictly increasing")
        else:
            last_sequence = seq
        sequences.add(seq)
        if not isinstance(event.get("generation"), int) or event["generation"] < 1:
            _finding(out, f"{path}.generation", "generation", "must be a positive integer")
    return out


def validate_stage_result(doc: Any, prefix: str = "stage") -> list[dict]:
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != STAGE_RESULT_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {STAGE_RESULT_SCHEMA!r}")
    if not _nonempty(doc.get("stage_id")):
        _finding(out, f"{prefix}.stage_id", "missing", "must be a stable non-empty id")
    _enum(out, doc, "phase", PHASES)
    status = _enum(out, doc, "status", STAGE_STATUSES)
    if not _timestamp(doc.get("collected_at")):
        _finding(out, f"{prefix}.collected_at", "timestamp", "must be an ISO-8601 timestamp")
    measurement = doc.get("measurement")
    if status == "completed":
        if measurement is None:
            _finding(out, f"{prefix}.measurement", "missing",
                     "completed stage must carry its measurement")
        else:
            out.extend(validate_measurement(measurement, f"{prefix}.measurement"))
    elif measurement is not None:
        out.extend(validate_measurement(measurement, f"{prefix}.measurement"))
    if not _reference(doc.get("profile_ref")):
        _finding(out, f"{prefix}.profile_ref", "profile_missing",
                 "every stage result must name its profile/evidence artifact")
    return out


def validate_arm_result(doc: Any, prefix: str = "arm") -> list[dict]:
    """Strict arm contract: completion and verdict are independent axes."""
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != ARM_RESULT_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {ARM_RESULT_SCHEMA!r}")
    for key in ("arm_id", "lever"):
        if not _nonempty(doc.get(key)):
            _finding(out, f"{prefix}.{key}", "missing", "must be a non-empty string")
    _enum(out, doc, "direction_class", ARM_DIRECTION_CLASSES)
    completion = _enum(out, doc, "completion", ARM_COMPLETION)
    verdict = _enum(out, doc, "verdict", ARM_VERDICTS)
    gate = _enum(out, doc, "gate_stage", ARM_GATE_STAGES)
    iters = doc.get("iters_used")
    if not isinstance(iters, int) or isinstance(iters, bool) or iters < 0:
        _finding(out, f"{prefix}.iters_used", "type", "must be a non-negative integer")
    if "result" in doc:
        _finding(out, f"{prefix}.result", "forbidden",
                 "ArmResult uses verdict; result is a final-report outcome field")
    measurement = doc.get("measurement")
    if completion == "complete":
        if measurement is None:
            _finding(out, f"{prefix}.measurement", "missing",
                     "a completed arm must carry a comparable measurement")
        else:
            out.extend(validate_measurement(measurement, f"{prefix}.measurement"))
    elif measurement is not None:
        _finding(out, f"{prefix}.measurement", "incomplete_measurement",
                 "truncated/crashed arms must not supply a rankable measurement")
    if completion in ("truncated", "crashed") and verdict not in (None, "inconclusive"):
        _finding(out, f"{prefix}.verdict", "completion_verdict",
                 "truncated/crashed arms can only be inconclusive")
    return out


def _role(out: list[dict], doc: dict, field: str, allowed: tuple[str, ...],
          prefix: str) -> str | None:
    value = doc.get(field)
    if not isinstance(value, str) or value not in allowed:
        _finding(out, f"{prefix}.{field}", "role",
                 f"expected one of {list(allowed)}, got {value!r}")
        return None
    return value


def _run_identity(out: list[dict], doc: dict, prefix: str) -> None:
    if not _nonempty(doc.get("run_id")):
        _finding(out, f"{prefix}.run_id", "missing", "must be a non-empty run id")
    generation = doc.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        _finding(out, f"{prefix}.generation", "generation", "must be a positive integer")


def validate_worker_result(doc: Any, prefix: str = "worker_result") -> list[dict]:
    """Validate the handoff a direction worker returns to its captain."""
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != WORKER_RESULT_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {WORKER_RESULT_SCHEMA!r}")
    _run_identity(out, doc, prefix)
    role = _role(out, doc, "producer_role", WORKER_ROLES, prefix)
    _enum(out, doc, "stage", PHASES)
    status = _enum(out, doc, "status", WORKER_STATUSES)
    if not _timestamp(doc.get("produced_at")):
        _finding(out, f"{prefix}.produced_at", "timestamp", "must be an ISO-8601 timestamp")
    measurement = doc.get("measurement")
    if status == "completed":
        if not isinstance(measurement, dict):
            _finding(out, f"{prefix}.measurement", "missing",
                     "a completed worker result must carry its comparable measurement")
        else:
            out.extend(validate_measurement(measurement, f"{prefix}.measurement"))
    elif measurement is not None:
        out.extend(validate_measurement(measurement, f"{prefix}.measurement"))
    requests = doc.get("requests", [])
    if not isinstance(requests, list):
        _finding(out, f"{prefix}.requests", "type", "must be a list")
    work_kinds = doc.get("work_kinds", [])
    if not isinstance(work_kinds, list) or any(not _nonempty(x) for x in work_kinds):
        _finding(out, f"{prefix}.work_kinds", "type", "must be a list of non-empty work kinds")
    if role == "deep" and any(kind in {"branch", "config_resweep", "resweep"}
                              for kind in work_kinds if isinstance(kind, str)):
        _finding(out, f"{prefix}.work_kinds", "forbidden_deep_branch",
                 "deep workers may emit structure_suspect/resweep_request, never execute branch or resweep")
    return out


def validate_branch_request(doc: Any, prefix: str = "branch_request") -> list[dict]:
    """Validate a direction's request for a captain-owned fan-out."""
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != BRANCH_REQUEST_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {BRANCH_REQUEST_SCHEMA!r}")
    _run_identity(out, doc, prefix)
    if not _nonempty(doc.get("request_id")):
        _finding(out, f"{prefix}.request_id", "missing", "must be a stable non-empty id")
    _role(out, doc, "requester_role", ("direction",), prefix)
    if not _timestamp(doc.get("requested_at")):
        _finding(out, f"{prefix}.requested_at", "timestamp", "must be an ISO-8601 timestamp")
    candidates = doc.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        _finding(out, f"{prefix}.candidates", "missing", "must declare every requested branch candidate")
    elif any(not _reference(candidate) for candidate in candidates):
        _finding(out, f"{prefix}.candidates", "reference", "every candidate must be a reference")
    if not _reference(doc.get("evidence_ref")):
        _finding(out, f"{prefix}.evidence_ref", "missing", "must name evidence for the request")
    if "state" in doc:
        _enum(out, doc, "state", REQUEST_STATES)
    return out


def validate_structure_suspect(doc: Any, prefix: str = "structure_suspect") -> list[dict]:
    """Validate the existing deep-worker request schema without renaming its public token."""
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != STRUCTURE_SUSPECT_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {STRUCTURE_SUSPECT_SCHEMA!r}")
    if not _nonempty(doc.get("candidate_id")):
        _finding(out, f"{prefix}.candidate_id", "missing", "must be a non-empty candidate id")
    _enum(out, doc, "state", ("untried", "deferred"))
    if not isinstance(doc.get("independent"), bool):
        _finding(out, f"{prefix}.independent", "type", "must be a boolean")
    for field in ("candidate_ref", "source_ref", "evidence_ref"):
        if not _reference(doc.get(field)):
            _finding(out, f"{prefix}.{field}", "missing", "must be a canonical artifact reference")
    if "requester_role" in doc:
        _role(out, doc, "requester_role", ("deep",), prefix)
    return out


def validate_resweep_request(doc: Any, prefix: str = "resweep_request") -> list[dict]:
    """A deep worker, captain, or direct owner can request a re-sweep, never execute it."""
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != RESWEEP_REQUEST_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {RESWEEP_REQUEST_SCHEMA!r}")
    _run_identity(out, doc, prefix)
    for field in ("request_id",):
        if not _nonempty(doc.get(field)):
            _finding(out, f"{prefix}.{field}", "missing", "must be a stable non-empty id")
    _role(out, doc, "requester_role", RESWEEP_REQUESTERS, prefix)
    if not _timestamp(doc.get("requested_at")):
        _finding(out, f"{prefix}.requested_at", "timestamp", "must be an ISO-8601 timestamp")
    for field in ("parent_result_ref", "evidence_ref"):
        if not _reference(doc.get(field)):
            _finding(out, f"{prefix}.{field}", "missing", "must be a canonical artifact reference")
    if doc.get("state") != "requested":
        _finding(out, f"{prefix}.state", "state",
                 "a resweep request remains requested; only a captain/direct owner records a result")
    if "measurement" in doc or "completed_at" in doc or "executor_role" in doc:
        _finding(out, prefix, "unauthorized_resweep",
                 "a resweep request cannot include execution fields; a separate captain/direct-owner "
                 "resweep result owns execution")
    return out


def validate_resweep_result(doc: Any, prefix: str = "resweep_result") -> list[dict]:
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != RESWEEP_RESULT_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {RESWEEP_RESULT_SCHEMA!r}")
    if not _reference(doc.get("request_ref")):
        _finding(out, f"{prefix}.request_ref", "missing", "must reference the deep-worker request")
    _run_identity(out, doc, prefix)
    _role(out, doc, "executor_role", FINAL_REPORT_OWNERS, prefix)
    if not _timestamp(doc.get("completed_at")):
        _finding(out, f"{prefix}.completed_at", "timestamp", "must be an ISO-8601 timestamp")
    if not isinstance(doc.get("measurement"), dict):
        _finding(out, f"{prefix}.measurement", "missing", "must carry the measured re-sweep")
    else:
        out.extend(validate_measurement(doc["measurement"], f"{prefix}.measurement"))
    return out


def _validate_sweep_axes(out: list[dict], doc: dict, prefix: str, *, request: bool) -> None:
    """Validate the immutable semantic identity shared by a sweep request and result."""
    _enum(out, doc, "purpose", SWEEP_PURPOSES)
    for field, allowed in (("work_kind", SWEEP_WORK_KINDS),
                           ("change_scope", CHANGE_SCOPES),
                           ("axis_kind", AXIS_KINDS)):
        _enum(out, doc, field, allowed)
    required_realization = "request" if request else "execution"
    if doc.get("realization_kind") != required_realization:
        _finding(out, f"{prefix}.realization_kind", "realization_kind",
                 f"must be {required_realization!r}")
    for field in ("attempt", "lineage"):
        value = doc.get(field)
        if not isinstance(value, dict):
            _finding(out, f"{prefix}.{field}", "missing", "must be an immutable object")
            continue
        unexpected = sorted(set(value) - ({"attempt_id"} if field == "attempt"
                                          else {"lineage_id", "root_request_id"}))
        if unexpected:
            _finding(out, f"{prefix}.{field}", "immutable",
                     f"contains mutable/unrecognised fields {unexpected}")
        required = ("attempt_id",) if field == "attempt" else ("lineage_id", "root_request_id")
        for key in required:
            if not _nonempty(value.get(key)):
                _finding(out, f"{prefix}.{field}.{key}", "missing",
                         "must be a stable non-empty id")


def validate_sweep_request(doc: Any, prefix: str = "sweep_request") -> list[dict]:
    """A typed sweep may be requested by an arm but only executed by its owner."""
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != SWEEP_REQUEST_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {SWEEP_REQUEST_SCHEMA!r}")
    _run_identity(out, doc, prefix)
    if not _nonempty(doc.get("request_id")):
        _finding(out, f"{prefix}.request_id", "missing", "must be a stable non-empty id")
    _role(out, doc, "requester_role", SWEEP_REQUESTERS, prefix)
    if not _timestamp(doc.get("requested_at")):
        _finding(out, f"{prefix}.requested_at", "timestamp", "must be an ISO-8601 timestamp")
    for field in ("source", "parent"):
        if not _reference(doc.get(field)):
            _finding(out, f"{prefix}.{field}", "missing", "must name a source/parent artifact")
    if not isinstance(doc.get("context"), dict) or not doc["context"]:
        _finding(out, f"{prefix}.context", "missing", "must be a non-empty context object")
    if doc.get("state") != "requested":
        _finding(out, f"{prefix}.state", "state", "a sweep request remains requested")
    _validate_sweep_axes(out, doc, prefix, request=True)
    if any(key in doc for key in ("measurement", "completed_at", "executor_role")):
        _finding(out, prefix, "unauthorized_sweep",
                 "a sweep request cannot contain execution fields; only captain/direct_owner "
                 "may write a separate sweep result")
    return out


def validate_sweep_result(doc: Any, prefix: str = "sweep_result") -> list[dict]:
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != SWEEP_RESULT_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {SWEEP_RESULT_SCHEMA!r}")
    if not _reference(doc.get("request_ref")):
        _finding(out, f"{prefix}.request_ref", "missing", "must reference the typed sweep request")
    _run_identity(out, doc, prefix)
    _role(out, doc, "executor_role", SWEEP_EXECUTORS, prefix)
    if not _timestamp(doc.get("completed_at")):
        _finding(out, f"{prefix}.completed_at", "timestamp", "must be an ISO-8601 timestamp")
    if not isinstance(doc.get("measurement"), dict):
        _finding(out, f"{prefix}.measurement", "missing", "must carry the measured sweep")
    else:
        out.extend(validate_measurement(doc["measurement"], f"{prefix}.measurement"))
    _validate_sweep_axes(out, doc, prefix, request=False)
    return out


def validate_sweep_pair(request: Any, result: Any, prefix: str = "sweep") -> list[dict]:
    """Verify that an execution resolves, rather than replaces, its immutable request."""
    out = validate_sweep_request(request, f"{prefix}.request")
    out.extend(validate_sweep_result(result, f"{prefix}.result"))
    if not isinstance(request, dict) or not isinstance(result, dict):
        return out
    for field in ("run_id", "generation", "purpose", "attempt", "lineage", "work_kind",
                  "change_scope", "axis_kind"):
        if request.get(field) != result.get(field):
            _finding(out, f"{prefix}.{field}", "lineage_mismatch",
                     "request and result must preserve the same immutable sweep identity")
    return out


def validate_branch_converge(doc: Any, prefix: str = "branch_converge") -> list[dict]:
    """Validate roster relations without imposing a fixed number of branch arms."""
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != BRANCH_CONVERGE_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {BRANCH_CONVERGE_SCHEMA!r}")
    if "run_id" in doc or "generation" in doc:
        _run_identity(out, doc, prefix)

    sets: dict[str, set[str]] = {}
    for name in ("planned", "completed", "eligible"):
        values = doc.get(name)
        if not isinstance(values, list) or any(not _nonempty(v) for v in values):
            _finding(out, f"{prefix}.{name}", "roster", "must be a list of non-empty arm ids")
            sets[name] = set()
        elif len(values) != len(set(values)):
            _finding(out, f"{prefix}.{name}", "roster", "must not contain duplicate arm ids")
            sets[name] = set(values)
        else:
            sets[name] = set(values)
    if not sets["planned"]:
        _finding(out, f"{prefix}.planned", "roster", "must state the arms actually planned")
    if sets["completed"] != sets["planned"]:
        _finding(out, f"{prefix}.completed", "roster",
                 "all planned arms must complete or be explicitly deferred before convergence")
    if not sets["eligible"] <= sets["completed"]:
        _finding(out, f"{prefix}.eligible", "arm_gate_missing",
                 "eligible arms must be completed arms")

    winner = doc.get("winner")
    if winner is not None and not _nonempty(winner):
        _finding(out, f"{prefix}.winner", "type", "must be an eligible arm id or null")
    elif winner is not None and winner not in sets["eligible"]:
        _finding(out, f"{prefix}.winner", "winner", "must be one of eligible arms")
    elif winner is None and sets["eligible"]:
        _finding(out, f"{prefix}.winner", "winner", "must select an eligible winner or record why none")

    measurement = doc.get("measurement")
    if measurement is None:
        _finding(out, f"{prefix}.measurement", "missing",
                 "must carry the measurement used to compare eligible arms")
    else:
        out.extend(validate_measurement(measurement, f"{prefix}.measurement"))
    if not _timestamp(doc.get("collected_at")):
        _finding(out, f"{prefix}.collected_at", "timestamp", "must be an ISO-8601 timestamp")
    decision = doc.get("direction_decision")
    if not isinstance(decision, dict):
        _finding(out, f"{prefix}.direction_decision", "missing",
                 "a direction-owned winner decision and evidence are required")
    else:
        if decision.get("winner") != winner:
            _finding(out, f"{prefix}.direction_decision.winner", "winner",
                     "must match the convergence winner")
        if not _reference(decision.get("evidence_ref")):
            _finding(out, f"{prefix}.direction_decision.evidence_ref", "missing",
                     "must name the direction's evidence for this selection")
    producer = doc.get("producer")
    if not isinstance(producer, dict):
        _finding(out, f"{prefix}.producer", "missing",
                 "must identify the direction or direct owner that selected the arm result")
    else:
        _role(out, producer, "role", BRANCH_CONVERGE_OWNERS, f"{prefix}.producer")

    results = doc.get("arm_results")
    result_ids: set[str] = set()
    if not isinstance(results, list):
        _finding(out, f"{prefix}.arm_results", "missing",
                 "must list each completed arm and its result reference")
    else:
        for i, entry in enumerate(results):
            path = f"{prefix}.arm_results[{i}]"
            if not isinstance(entry, dict):
                _finding(out, path, "type", "expected an object")
                continue
            arm_id = entry.get("arm_id")
            if not _nonempty(arm_id) or arm_id in result_ids:
                _finding(out, f"{path}.arm_id", "unique", "must be a unique non-empty arm id")
                continue
            result_ids.add(arm_id)
            if arm_id not in sets["completed"]:
                _finding(out, f"{path}.arm_id", "roster", "must be listed in completed")
            if not _reference(entry.get("ref")):
                _finding(out, f"{path}.ref", "missing", "must point to the arm record")
            record = entry.get("record")
            if not isinstance(record, dict):
                _finding(out, f"{path}.record", "arm_gate_missing",
                         "strict convergence requires the machine-readable ArmResult")
                continue
            out.extend(validate_arm_result(record, f"{path}.record"))
            if record.get("arm_id") != arm_id:
                _finding(out, f"{path}.record.arm_id", "identity",
                         "must match arm_results[].arm_id")
            if arm_id in sets["eligible"]:
                if record.get("completion") != "complete" or record.get("gate_stage") == "a":
                    _finding(out, path, "arm_gate_missing",
                             "eligible arm must be complete and past gate A")
                if not same_identity(record.get("measurement"), measurement):
                    _finding(out, f"{path}.record.measurement", "identity_mismatch",
                             "eligible arm measurement identity must match convergence measurement")
        if result_ids != sets["completed"]:
            _finding(out, f"{prefix}.arm_results", "roster",
                     "must contain exactly the completed arm ids")

    resweep = doc.get("post_merge_resweep")
    if not isinstance(resweep, dict):
        _finding(out, f"{prefix}.post_merge_resweep", "resweep_missing",
                 "must state completed/not_required; waived resweeps are not valid")
    else:
        status = _enum(out, resweep, "status", POST_MERGE_RESWEEP)
        if winner is not None and status != "completed":
            _finding(out, f"{prefix}.post_merge_resweep.status", "resweep_missing",
                     "a selected winner requires a completed post-merge resweep")
        if status == "completed":
            if not _reference(resweep.get("request_ref")):
                _finding(out, f"{prefix}.post_merge_resweep.request_ref", "resweep_missing",
                         "must reference the captain/direct-owner or deep-worker resweep request")
            if not _reference(resweep.get("result_ref")):
                _finding(out, f"{prefix}.post_merge_resweep.result_ref", "resweep_missing",
                         "must reference the captain/direct-owner resweep result")
            if isinstance(resweep.get("result"), dict):
                result_validator = (validate_sweep_result
                                    if resweep["result"].get("schema") == SWEEP_RESULT_SCHEMA
                                    else validate_resweep_result)
                out.extend(result_validator(resweep["result"], f"{prefix}.post_merge_resweep.result"))
                if resweep["result"].get("schema") == SWEEP_RESULT_SCHEMA and (
                        resweep["result"].get("purpose") != "post_merge"):
                    _finding(out, f"{prefix}.post_merge_resweep.result.purpose", "resweep_missing",
                             "typed post-merge execution must have purpose=post_merge")
                if ("run_id" in doc and resweep["result"].get("run_id") != doc.get("run_id")):
                    _finding(out, f"{prefix}.post_merge_resweep.result.run_id", "identity_mismatch",
                             "resweep result must share branch convergence run_id")
                if ("generation" in doc and
                        resweep["result"].get("generation") != doc.get("generation")):
                    _finding(out, f"{prefix}.post_merge_resweep.result.generation",
                             "identity_mismatch",
                             "resweep result must share branch convergence generation")
            if not _timestamp(resweep.get("collected_at")):
                _finding(out, f"{prefix}.post_merge_resweep.collected_at", "timestamp",
                         "must be an ISO-8601 timestamp")
            if not isinstance(resweep.get("measurement"), dict):
                _finding(out, f"{prefix}.post_merge_resweep.measurement", "resweep_missing",
                         "must carry the post-merge measurement")
            else:
                out.extend(validate_measurement(
                    resweep["measurement"], f"{prefix}.post_merge_resweep.measurement"))
                if not same_identity(resweep["measurement"], measurement):
                    _finding(out, f"{prefix}.post_merge_resweep.measurement", "identity_mismatch",
                             "post-merge resweep must use convergence measurement identity")
    return out


_REPORT_HEADLINE_ALIASES = {
    "default_ms": ("default_ms", "baseline_ms", "comparator_ms"),
    "champion_ms": ("champion_ms", "current_ms"),
    "best_ms": ("best_ms", "final_ms"),
    "vs_champion": ("vs_champion", "result_vs_champion_ms", "final_vs_champion"),
}
_REPORT_HEADLINE_CONTAINERS = ("headline", "result", "measurement", "verdict")
_REPORT_SERVED_ALIASES = (
    "served_range", "served_range_table", "served_shape_table", "served_envelope",
)


def _report_scalar(doc: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = doc.get(key)
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        if value not in (None, ""):
            return value
    return default


def _legacy_ref(value: Any, *, work_root: str | None = None) -> dict | None:
    """Project a historical path into a typed, explicitly unverified reference.

    This is deliberately not ``normalize_artifact_ref``: that function verifies bytes and emits a
    trusted v1 ArtifactRef.  A path found in an old report proves only what the writer named, so the
    projection keeps it addressable without retroactively claiming that the bytes were verified.
    """
    if isinstance(value, dict):
        if value.get("schema") == ARTIFACT_REF_SCHEMA and isinstance(value.get("sha256"), str):
            return dict(value)
        if value.get("schema") == LEGACY_ARTIFACT_REF_SCHEMA:
            return dict(value)
        raw = value.get("uri") or value.get("path") or value.get("ref")
    else:
        raw = value
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()
    selector = None
    if "#" in raw:
        raw, selector = raw.split("#", 1)
    if raw.startswith(("work:/", "pack:/", "snapshot:/")):
        uri = raw
    else:
        path = os.path.abspath(os.path.expanduser(raw)) if os.path.isabs(raw) else None
        root = os.path.abspath(work_root) if work_root else None
        if path and root:
            try:
                rel = os.path.relpath(path, root)
            except ValueError:
                rel = ".."
        else:
            rel = raw
        rel = rel.replace("\\", "/").lstrip("./")
        if not rel or rel == ".." or rel.startswith("../") or "/../" in rel:
            rel = "legacy-unresolved/" + hashlib.sha256(raw.encode()).hexdigest()
        uri = "work:/" + rel
    out = {
        "schema": LEGACY_ARTIFACT_REF_SCHEMA,
        "uri": uri,
        "verification": "legacy_unverified",
    }
    if selector:
        out["selector"] = selector
    return out


def _writer_ref(value: Any, *, work_root: str) -> dict:
    """Emit a verified ArtifactRef for a new writer; unlike legacy projection, bytes are required."""
    if isinstance(value, dict) and value.get("schema") == ARTIFACT_REF_SCHEMA:
        return dict(value)
    raw = value.get("path") if isinstance(value, dict) else value
    if not isinstance(raw, str) or not raw:
        raise ValueError("new final-report references must name a file or ArtifactRef")
    path = raw if os.path.isabs(raw) else os.path.join(work_root, raw)
    path = os.path.realpath(path)
    root = os.path.realpath(work_root)
    try:
        rel = os.path.relpath(path, root)
    except ValueError as exc:
        raise ValueError(f"artifact path is not portable under work root: {raw}") from exc
    if rel == ".." or rel.startswith("../") or not os.path.isfile(path):
        raise ValueError(f"artifact path is missing or escapes work root: {raw}")
    return {
        "schema": ARTIFACT_REF_SCHEMA,
        "uri": "work:/" + rel.replace(os.sep, "/"),
        "sha256": _file_sha256(path),
        "bytes": os.path.getsize(path),
    }


def _headline_projection(doc: dict) -> dict:
    scopes = [doc] + [doc[name] for name in _REPORT_HEADLINE_CONTAINERS
                      if isinstance(doc.get(name), dict)]
    primary = _report_scalar(doc, "primary_case")
    for scope in scopes:
        primary = primary or _report_scalar(scope, "primary_case")
    out: dict[str, Any] = {}
    for canonical, aliases in _REPORT_HEADLINE_ALIASES.items():
        for scope in scopes:
            hit = next((scope[name] for name in aliases if name in scope), None)
            if hit is None:
                continue
            if isinstance(hit, (int, float)) and not isinstance(hit, bool):
                out[canonical] = hit
                break
            if isinstance(hit, dict):
                if isinstance(hit.get("value"), (int, float)) and not isinstance(hit["value"], bool):
                    out[canonical] = hit["value"]
                    break
                selected = hit.get(primary) if isinstance(primary, str) else None
                if isinstance(selected, (int, float)) and not isinstance(selected, bool):
                    out[canonical] = selected
                    break
        # An unresolved per-case map stays absent.  Consumers must report incompleteness, not pick.
    if isinstance(primary, str) and primary:
        out["primary_case"] = primary
    return out


def _served_projection(doc: dict) -> list:
    for name in _REPORT_SERVED_ALIASES:
        value = doc.get(name)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            rows = value.get("rows") or value.get("served_range") or value.get("cases")
            if isinstance(rows, list):
                return rows
            keyed = [dict(row, case=row.get("case", label))
                     for label, row in value.items() if isinstance(row, dict)]
            if keyed:
                return keyed
    return []


def validate_producer_receipt(doc: Any, prefix: str = "producer_receipt") -> list[dict]:
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != PRODUCER_RECEIPT_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {PRODUCER_RECEIPT_SCHEMA!r}")
    if not _nonempty(doc.get("receipt_id")):
        _finding(out, f"{prefix}.receipt_id", "missing", "must be non-empty")
    _enum(out, doc, "artifact_role", PRODUCER_ARTIFACT_ROLES)
    if not _timestamp(doc.get("produced_at")):
        _finding(out, f"{prefix}.produced_at", "timestamp", "must be an ISO-8601 timestamp")
    _enum(out, doc, "timing", ("process_time", "late_reconstruction"))
    producer = doc.get("producer")
    if not isinstance(producer, dict):
        _finding(out, f"{prefix}.producer", "missing", "must identify the executable")
    else:
        mode = producer.get("mode")
        if mode not in ("primary", "fallback"):
            _finding(out, f"{prefix}.producer.mode", "enum",
                     "expected one of ['primary', 'fallback']")
        if not _nonempty(producer.get("tool")):
            _finding(out, f"{prefix}.producer.tool", "missing", "must be non-empty")
        if not isinstance(producer.get("executable_sha256"), str) or len(
                producer["executable_sha256"]) != 64:
            _finding(out, f"{prefix}.producer.executable_sha256", "hash",
                     "must be a complete SHA-256")
        fallback = doc.get("fallback")
        if producer.get("mode") == "fallback":
            if not isinstance(fallback, dict) or any(
                    not _nonempty(fallback.get(key))
                    for key in ("reason", "input_schema", "output_schema")):
                _finding(out, f"{prefix}.fallback", "fallback_contract",
                         "fallback requires reason, input_schema, and output_schema")
        elif fallback is not None:
            _finding(out, f"{prefix}.fallback", "fallback_contract",
                     "primary producers must not carry a fallback block")
    artifact = doc.get("artifact")
    if not isinstance(artifact, dict):
        _finding(out, f"{prefix}.artifact", "missing", "must identify produced bytes")
    else:
        for key in ("ref", "schema"):
            if not _nonempty(artifact.get(key)):
                _finding(out, f"{prefix}.artifact.{key}", "missing", "must be non-empty")
        if not isinstance(artifact.get("sha256"), str) or len(artifact["sha256"]) != 64:
            _finding(out, f"{prefix}.artifact.sha256", "hash", "must be a complete SHA-256")
    validator = doc.get("invariant_validator")
    if not isinstance(validator, dict):
        _finding(out, f"{prefix}.invariant_validator", "missing",
                 "must name the invariant validator")
    else:
        if validator.get("passed") is not True:
            _finding(out, f"{prefix}.invariant_validator.passed", "invariant",
                     "must be true")
        for key in ("tool", "schema"):
            if not _nonempty(validator.get(key)):
                _finding(out, f"{prefix}.invariant_validator.{key}", "missing",
                         "must be non-empty")
        if not isinstance(validator.get("executable_sha256"), str) or len(
                validator["executable_sha256"]) != 64:
            _finding(out, f"{prefix}.invariant_validator.executable_sha256", "hash",
                     "must be a complete SHA-256")
    fallback = doc.get("fallback")
    if isinstance(fallback, dict) and isinstance(artifact, dict) and isinstance(validator, dict):
        if fallback.get("output_schema") != artifact.get("schema"):
            _finding(out, f"{prefix}.fallback.output_schema", "fallback_contract",
                     "must equal the produced artifact schema")
        if validator.get("schema") != artifact.get("schema"):
            _finding(out, f"{prefix}.invariant_validator.schema", "fallback_contract",
                     "fallback and primary outputs must use the same invariant schema")
    return out


def write_producer_receipt(artifact_path: str, artifact_role: str, *,
                           tool_path: str | None = None, timing: str = "process_time",
                           fallback_reason: str | None = None,
                           input_schema: str | None = None,
                           output_schema: str | None = None,
                           validator_path: str | None = None,
                           receipt_path: str | None = None) -> dict:
    """Write a sidecar receipt after validating and hashing the produced artifact."""
    artifact = _load_json_value(artifact_path, "producer artifact")
    if not isinstance(artifact, dict):
        raise ValueError("producer artifacts must be JSON objects")
    schema = str(artifact.get("schema") or "legacy.unspecified/1")
    executable = os.path.abspath(tool_path or __file__)
    validator = os.path.abspath(validator_path or __file__)
    mode = "fallback" if fallback_reason else "primary"
    receipt = {
        "schema": PRODUCER_RECEIPT_SCHEMA,
        "receipt_id": f"{artifact_role}-{hashlib.sha256((artifact_path + _now()).encode()).hexdigest()[:20]}",
        "artifact_role": artifact_role,
        "produced_at": _now(),
        "timing": timing,
        "producer": {
            "tool": os.path.basename(executable),
            "executable_sha256": _file_sha256(executable),
            "mode": mode,
        },
        "artifact": {
            "ref": os.path.abspath(artifact_path),
            "schema": schema,
            "sha256": _file_sha256(artifact_path),
        },
        "invariant_validator": {
            "tool": os.path.basename(validator),
            "executable_sha256": _file_sha256(validator),
            "schema": schema,
            "passed": True,
        },
    }
    if mode == "fallback":
        receipt["fallback"] = {
            "reason": fallback_reason,
            "input_schema": input_schema or "",
            "output_schema": output_schema or schema,
        }
    findings = validate_producer_receipt(receipt)
    if findings:
        raise ValueError(findings[0]["detail"])
    _atomic_json(receipt_path or artifact_path + ".receipt.json", receipt)
    return receipt


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt_projection(value: Any, *, work_root: str | None) -> tuple[dict | None, list[dict]]:
    if isinstance(value, dict) and value.get("schema") == PRODUCER_RECEIPT_SCHEMA:
        receipt = value
    else:
        ref = _legacy_ref(value, work_root=work_root)
        if not ref:
            return None, [{"field": "provenance.receipts", "code": "receipt_unreadable",
                           "detail": "receipt reference is not readable", "severity": "error"}]
        uri = ref.get("uri", "")
        path = uri[len("work:/"):] if uri.startswith("work:/") else None
        try:
            receipt = _load(os.path.join(work_root or ".", path)) if path else None
        except (OSError, json.JSONDecodeError):
            receipt = None
        if not isinstance(receipt, dict):
            return None, [{"field": "provenance.receipts", "code": "receipt_unreadable",
                           "detail": f"cannot read producer receipt {uri!r}", "severity": "error"}]
    findings = validate_producer_receipt(receipt)
    artifact = receipt.get("artifact") if isinstance(receipt, dict) else None
    raw_artifact = artifact.get("ref") if isinstance(artifact, dict) else None
    if not findings and isinstance(raw_artifact, str):
        if work_root is None:
            findings.append({
                "field": "provenance.receipts", "code": "receipt_unverified_context",
                "detail": "receipt structure is valid but no work root was supplied to verify bytes",
                "severity": "note",
            })
        else:
            artifact_path = (raw_artifact if os.path.isabs(raw_artifact)
                             else os.path.join(work_root, raw_artifact))
            if not os.path.isfile(artifact_path):
                findings.append({
                    "field": "provenance.receipts", "code": "receipt_artifact_missing",
                    "detail": f"receipt artifact is unavailable: {raw_artifact}",
                    "severity": "error",
                })
            elif _file_sha256(artifact_path) != artifact.get("sha256"):
                findings.append({
                    "field": "provenance.receipts", "code": "receipt_artifact_hash",
                    "detail": f"receipt hash does not match artifact bytes: {raw_artifact}",
                    "severity": "error",
                })
            else:
                try:
                    artifact_doc = _load(artifact_path)
                except (OSError, json.JSONDecodeError):
                    artifact_doc = None
                observed_schema = (artifact_doc.get("schema") if isinstance(artifact_doc, dict)
                                   else "legacy.unspecified/1")
                if str(observed_schema) != artifact.get("schema"):
                    findings.append({
                        "field": "provenance.receipts", "code": "receipt_artifact_schema",
                        "detail": f"receipt schema does not match artifact: {raw_artifact}",
                        "severity": "error",
                    })
    return receipt, findings


def project_final_report(report: Any, *, work_root: str | None = None,
                         source_path: str | None = None) -> dict:
    """Normalize every supported report spelling into the portable v3 projection.

    Unknown majors are represented as a degraded projection with an explicit finding.  The CLI's
    ``--strict`` option turns that state into a non-zero exit; library consumers can still display
    it without guessing fields from an unknown contract.
    """
    if not isinstance(report, dict):
        return {
            "schema": FINAL_REPORT_SCHEMA, "run_id": "unknown", "generation": 1,
            "lifecycle_profile": "blocked", "stage": "finalization", "outcome": "partial",
            "arbitration": "reject", "status": "blocked", "refs": {}, "audit": {},
            "provenance": {"receipts": []}, "caveats": [], "deferred": [], "known_wrong": [],
            "integrity": {"source_schema": None, "normalization_status": "failed",
                          "legacy_unverified": True, "eligible_clean": False,
                          "findings": [{"field": "<root>", "code": "type",
                                        "detail": "final report must be an object",
                                        "severity": "error"}]},
        }
    source_schema = report.get("schema")
    native = source_schema == FINAL_REPORT_SCHEMA
    legacy = source_schema in LEGACY_FINAL_REPORT_SCHEMAS or source_schema in (None, "")
    unknown = not native and not legacy
    status = str(_report_scalar(report, "status", default="closed"))
    producer = report.get("producer") if isinstance(report.get("producer"), dict) else {}
    producer_role = producer.get("role") or report.get("producer_role") or report.get("owner")
    if status in ("blocked", "deferred_needs_user", "partial_time_limit"):
        lifecycle = "blocked"
    else:
        lifecycle = report.get("lifecycle_profile")
        if lifecycle not in LIFECYCLE_PROFILES:
            if producer_role == "direct_owner":
                lifecycle = "direct-owner"
            elif ((report.get("deep_arbitration") or {}).get("state") in ("not_run", "compared")
                  if isinstance(report.get("deep_arbitration"), dict) else False) or report.get(
                      "deep_result_ref"):
                lifecycle = "deep-consumer"
            elif any(key in report for key in ("champion_ref", "champion_gate_ref", "plain_champion")):
                lifecycle = "front-end"
            else:
                lifecycle = "direct-owner"
    outcome = str(_report_scalar(
        report, "outcome", "result", default="partial" if status != "closed" else "timeout"
    )).lower()
    arbitration = str(_report_scalar(
        report, "arbitration", "verdict", default="reject" if status != "closed" else "accept"
    )).lower()
    refs: dict[str, dict] = {}
    ref_aliases = {
        "canonical": ("canonical_ref",),
        "champion": ("champion_ref", "plain_champion_ref"),
        "champion_gate": ("champion_gate_ref",),
        "close_audit": ("close_audit_ref", "audit_ref"),
        "deep_result": ("deep_result_ref",),
    }
    native_refs = report.get("refs") if isinstance(report.get("refs"), dict) else {}
    for name, aliases in ref_aliases.items():
        value = native_refs.get(name)
        if value is None:
            value = next((report[key] for key in aliases if report.get(key) is not None), None)
        typed = _legacy_ref(value, work_root=work_root)
        if typed:
            refs[name] = typed
    anchor = report.get("anchor")
    if isinstance(anchor, dict):
        typed = _legacy_ref(anchor.get("ref"), work_root=work_root)
        if typed:
            refs["anchor"] = typed
    audit_source = report.get("audit") if isinstance(report.get("audit"), dict) else (
        report.get("close_audit") if isinstance(report.get("close_audit"), dict) else {})
    audit = {
        "strict_pass": audit_source.get("strict_pass"),
        "high_severity_count": audit_source.get(
            "high_severity_count", audit_source.get("high_findings", 0)),
        "blocked_count": audit_source.get(
            "blocked_count", len(audit_source.get("blocked_checks") or [])),
        "verified_by": audit_source.get("verified_by"),
    }
    raw_receipts = ((report.get("provenance") or {}).get("receipts")
                    if isinstance(report.get("provenance"), dict) else report.get("producer_receipts"))
    raw_receipts = raw_receipts if isinstance(raw_receipts, list) else []
    if source_path:
        sidecar = source_path + ".receipt.json"
        if os.path.isfile(sidecar):
            raw_receipts = [*raw_receipts, sidecar]
    receipts, findings = [], []
    for value in raw_receipts:
        receipt, receipt_findings = _receipt_projection(value, work_root=work_root)
        findings.extend(receipt_findings)
        if receipt:
            receipts.append(receipt)
    if unknown:
        findings.append({
            "field": "schema", "code": "unknown_schema_major",
            "detail": f"unsupported final-report schema {source_schema!r}; supported legacy=/1,/2 "
                      f"and canonical={FINAL_REPORT_SCHEMA}",
            "severity": "error",
        })
    for field in ("caveats", "deferred"):
        if field not in report:
            findings.append({
                "field": field, "code": "legacy_field_absent",
                "detail": f"{field} was absent in the source report; projection uses [] but does "
                          "not treat absence as an explicit answer",
                "severity": "note",
            })
    roles = {item.get("artifact_role") for item in receipts}
    required_roles = {
        "front-end": {"sweep", "champion", "audit", "final_report"},
        "deep-consumer": {"champion", "audit", "final_report"},
        "direct-owner": {"audit", "final_report"},
        "blocked": set(),
    }[lifecycle]
    missing_roles = sorted(required_roles - roles)
    if native and missing_roles:
        findings.append({
            "field": "provenance.receipts", "code": "producer_receipt_missing",
            "detail": f"{lifecycle} report misses process provenance for {missing_roles}",
            "severity": "error",
        })
    late = any(item.get("timing") == "late_reconstruction" for item in receipts)
    fallback_invalid = any(item for item in findings if item.get("code") == "fallback_contract")
    legacy_unverified = legacy or any(
        ref.get("schema") != ARTIFACT_REF_SCHEMA for ref in refs.values())
    normalization_status = "failed" if unknown else (
        "degraded" if legacy_unverified or findings else "canonical")
    eligible_clean = bool(
        native and status == "closed" and arbitration == "accept"
        and audit.get("strict_pass") is True
        and audit.get("high_severity_count") == 0 and audit.get("blocked_count") == 0
        and not legacy_unverified and not late and not fallback_invalid and not findings
    )
    projected = dict(report)
    projected.update({
        "schema": FINAL_REPORT_SCHEMA,
        "run_id": str(report.get("run_id") or report.get("kernel_id") or "legacy-run"),
        "generation": report.get("generation") if isinstance(report.get("generation"), int) else 1,
        "lifecycle_profile": lifecycle,
        "stage": str(_report_scalar(report, "stage", "winning_stage", default="finalization")),
        "outcome": outcome,
        "arbitration": arbitration,
        "status": status,
        "refs": refs,
        "audit": audit,
        "provenance": {"receipts": receipts},
        "headline": _headline_projection(report),
        "served_range": _served_projection(report),
        "measurement": report.get("measurement") if isinstance(report.get("measurement"), dict) else None,
        "caveats": (report.get("caveats") if native and "caveats" in report else
                    report.get("caveats") if isinstance(report.get("caveats"), list) else []),
        "deferred": (report.get("deferred") if native and "deferred" in report else
                     report.get("deferred") if isinstance(report.get("deferred"), list) else []),
        "known_wrong": (report.get("known_wrong") if native and "known_wrong" in report else
                        report.get("known_wrong")
                        if isinstance(report.get("known_wrong"), list) else []),
        "producer": dict(producer, role=producer_role or (
            "direct_owner" if lifecycle == "direct-owner" else "captain")),
        "integrity": {
            "source_schema": source_schema,
            "normalization_status": normalization_status,
            "legacy_unverified": legacy_unverified,
            "late_reconstruction": late,
            "fallback_invalid": fallback_invalid,
            "missing_receipt_roles": missing_roles,
            "eligible_clean": eligible_clean,
            "findings": findings,
        },
    })
    for name, value in projected["headline"].items():
        projected.setdefault(name, value)
    for name, value in refs.items():
        projected[name + "_ref"] = value
    return projected


def validate_final_report(doc: Any, captain: bool = False,
                          prefix: str = "final_report") -> list[dict]:
    """Validate the portable v3 report with lifecycle-specific obligations."""
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != FINAL_REPORT_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {FINAL_REPORT_SCHEMA!r}")
    if not _nonempty(doc.get("run_id")):
        _finding(out, f"{prefix}.run_id", "missing", "must be a non-empty id")
    generation = doc.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        _finding(out, f"{prefix}.generation", "generation", "must be a positive integer")
    lifecycle = _enum(out, doc, "lifecycle_profile", LIFECYCLE_PROFILES)
    _enum(out, doc, "stage", PHASES)
    _enum(out, doc, "outcome", OUTCOMES)
    _enum(out, doc, "arbitration", ARBITRATIONS)
    _enum(out, doc, "status", RUN_STATUSES)
    producer = doc.get("producer")
    if not isinstance(producer, dict):
        _finding(out, f"{prefix}.producer", "missing",
                 "must identify the captain or direct owner that wrote this report")
    else:
        _role(out, producer, "role", FINAL_REPORT_OWNERS, f"{prefix}.producer")
    refs = doc.get("refs")
    if not isinstance(refs, dict):
        _finding(out, f"{prefix}.refs", "type", "must be an object of typed references")
        refs = {}
    if lifecycle != "blocked" and not (_reference(refs.get("canonical"))
                                        or _reference(doc.get("canonical_ref"))):
        _finding(out, f"{prefix}.refs.canonical", "canonical_entry_missing",
                 "non-blocked reports must reference the canonical run-decision artifact")
    if lifecycle in ("front-end", "deep-consumer"):
        if not (_reference(refs.get("champion")) or _reference(doc.get("champion_ref"))):
            _finding(out, f"{prefix}.refs.champion", "missing",
                     f"{lifecycle} reports must name their champion input/output")
    if lifecycle == "front-end" and not (
            _reference(refs.get("champion_gate")) or _reference(doc.get("champion_gate_ref"))):
        _finding(out, f"{prefix}.refs.champion_gate", "missing",
                 "front-end reports must name the champion acceptance gate")
    if lifecycle != "blocked" and not (
            _reference(refs.get("close_audit")) or _reference(doc.get("close_audit_ref"))):
        _finding(out, f"{prefix}.refs.close_audit", "missing",
                 "non-blocked reports must name the strict close audit")
    anchor = doc.get("anchor")
    if anchor is not None:
        if not isinstance(anchor, dict) or anchor.get("type") not in (
                "pinned_comparator", "production_default_anchor"):
            _finding(out, f"{prefix}.anchor", "anchor",
                     "must declare pinned_comparator or production_default_anchor")
        elif not (_reference(anchor.get("ref")) or _reference(refs.get("anchor"))):
            _finding(out, f"{prefix}.anchor.ref", "missing",
                     "must reference the anchor evidence")
    scope = doc.get("scope")
    if not isinstance(scope, dict) or not _nonempty(scope.get("authorization")):
        _finding(out, f"{prefix}.scope", "missing",
                 "must state the authorized optimization scope")
    if not isinstance(doc.get("deferred"), list):
        _finding(out, f"{prefix}.deferred", "type",
                 "must explicitly list deferred work; [] is valid")
    deep = doc.get("deep_arbitration")
    if lifecycle == "deep-consumer":
        if not isinstance(deep, dict) or deep.get("state") not in ("not_run", "compared"):
            _finding(out, f"{prefix}.deep_arbitration", "missing",
                     "deep-consumer must declare whether its deep result was compared")
        elif deep["state"] == "compared" and not (
                _reference(deep.get("result_ref")) or _reference(refs.get("deep_result"))):
            _finding(out, f"{prefix}.deep_arbitration.result_ref", "missing",
                     "a compared deep result must reference its measured artifact")
    elif deep is not None and (not isinstance(deep, dict) or deep.get("state") not in (
            "not_authorized", "not_run", "compared")):
        _finding(out, f"{prefix}.deep_arbitration", "type",
                 "when present, deep arbitration must use a known state")
    if lifecycle != "blocked" and not isinstance(doc.get("measurement"), dict):
        _finding(out, f"{prefix}.measurement", "missing",
                 "must carry the final comparable measurement")
    elif isinstance(doc.get("measurement"), dict):
        out.extend(validate_measurement(doc["measurement"], f"{prefix}.measurement"))
    if "known_wrong" not in doc:
        _finding(out, f"{prefix}.known_wrong", "missing",
                 "must carry known_wrong; [] is a valid explicit answer")
    elif not isinstance(doc["known_wrong"], list):
        _finding(out, f"{prefix}.known_wrong", "type", "must be a list")
    provenance = doc.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("receipts"), list):
        _finding(out, f"{prefix}.provenance", "missing",
                 "must carry the producer receipt projection")
    else:
        for index, receipt in enumerate(provenance["receipts"]):
            out.extend(validate_producer_receipt(receipt, f"{prefix}.provenance.receipts[{index}]"))
    integrity = doc.get("integrity")
    if not isinstance(integrity, dict):
        _finding(out, f"{prefix}.integrity", "missing", "must carry normalization integrity")
    elif integrity.get("normalization_status") == "failed":
        _finding(out, f"{prefix}.integrity", "unknown_schema_major",
                 "a failed/unknown projection cannot pass canonical validation")
    return out


def validate_run_decision(doc: Any, prefix: str = "canonical") -> list[dict]:
    out: list[dict] = []
    if not isinstance(doc, dict):
        _finding(out, prefix, "type", "expected an object")
        return out
    if doc.get("schema") != RUN_DECISION_SCHEMA:
        _finding(out, f"{prefix}.schema", "schema", f"expected {RUN_DECISION_SCHEMA!r}")
    if not _nonempty(doc.get("run_id")):
        _finding(out, f"{prefix}.run_id", "missing", "must be a non-empty id")
    if not isinstance(doc.get("generation"), int) or doc.get("generation", 0) < 1:
        _finding(out, f"{prefix}.generation", "generation", "must be a positive integer")
    state = _enum(out, doc, "state", RUN_DECISION_STATES)
    producer = doc.get("producer")
    if not isinstance(producer, dict):
        _finding(out, f"{prefix}.producer", "missing",
                 "must identify the captain or direct owner operating the canonical record")
    else:
        _role(out, producer, "role", FINAL_REPORT_OWNERS, f"{prefix}.producer")
    measurement = doc.get("measurement")
    if not isinstance(measurement, dict):
        _finding(out, f"{prefix}.measurement", "missing", "must carry final measurement")
    else:
        out.extend(validate_measurement(measurement, f"{prefix}.measurement"))
    out.extend(validate_timeline(doc.get("timeline"), f"{prefix}.timeline"))
    stages = doc.get("stages")
    if not isinstance(stages, list) or not stages:
        _finding(out, f"{prefix}.stages", "missing", "must be a non-empty list of StageResults")
    else:
        ids = set()
        for i, stage in enumerate(stages):
            out.extend(validate_stage_result(stage, f"{prefix}.stages[{i}]"))
            sid = stage.get("stage_id") if isinstance(stage, dict) else None
            if sid in ids:
                _finding(out, f"{prefix}.stages[{i}].stage_id", "unique", "must be unique")
            ids.add(sid)
    final = doc.get("final")
    if state in ("finalizing", *RUN_STATUSES) and not isinstance(final, dict):
        _finding(out, f"{prefix}.final", "missing",
                 "a finalizing or terminal run must declare final domains")
    elif isinstance(final, dict):
        _enum(out, final, "stage", PHASES)
        _enum(out, final, "outcome", OUTCOMES)
        _enum(out, final, "arbitration", ARBITRATIONS)
        _enum(out, final, "status", RUN_STATUSES)
        if not _reference(final.get("report_ref")):
            _finding(out, f"{prefix}.final.report_ref", "missing", "must reference final report")
        elif state in RUN_STATUSES and final.get("status") != state:
            _finding(out, f"{prefix}.final.status", "state_mismatch",
                     "terminal canonical state must equal final.status")
    elif final is not None:
        _finding(out, f"{prefix}.final", "type", "must be an object when present")
    branch = doc.get("branch_converge")
    if branch is not None:
        out.extend(validate_branch_converge(branch, f"{prefix}.branch_converge"))
        if isinstance(branch, dict) and isinstance(measurement, dict) and not same_identity(
                measurement, branch.get("measurement")):
            _finding(out, f"{prefix}.branch_converge.measurement", "identity_mismatch",
                     "branch convergence and final measurement identities must match")
    return out


def validate(kind: str, doc: Any, captain: bool = False) -> list[dict]:
    validators = {
        "measurement": validate_measurement,
        "stage-result": validate_stage_result,
        "run-decision": validate_run_decision,
        "arm-result": validate_arm_result,
        "worker-result": validate_worker_result,
        "branch-request": validate_branch_request,
        "structure-suspect": validate_structure_suspect,
        "resweep-request": validate_resweep_request,
        "resweep-result": validate_resweep_result,
        "sweep-request": validate_sweep_request,
        "sweep-result": validate_sweep_result,
        "branch-converge": validate_branch_converge,
    }
    if kind == "final-report":
        return validate_final_report(doc, captain=captain)
    if kind == "producer-receipt":
        return validate_producer_receipt(doc)
    return validators[kind](doc)


def legacy_measurement(record: dict, *, shape_set: list[str] | None = None,
                       aggregation: str = "unspecified", boundary: str = "unknown",
                       source: str | None = None, sample_count: int | None = None,
                       measurement_scope: str | None = None, baseline_ref: Any = None,
                       comparator_ref: Any = None) -> dict | None:
    """Produce an honest compatibility record for a legacy round ledger row.

    Existing ``latency_ms`` has no implicit device provenance.  It becomes
    ``legacy_latency`` with ``scope=unknown`` unless a caller supplies all fields
    required for a genuine ``device_timing`` measurement.
    """
    metric = "latency_ms" if record.get("latency_ms") is not None else (
        "speedup_vs_comparator" if record.get("speedup_vs_comparator") is not None else None)
    if metric is None:
        return None
    value = record[metric]
    legacy_source = "legacy_latency" if metric == "latency_ms" else "legacy_ratio"
    source = source or legacy_source
    scope = measurement_scope or ("device" if source == "device_timing" else "unknown")
    count: int | str = sample_count if sample_count is not None else "unknown"
    seed = json.dumps({
        "round": record.get("round"), "metric": metric, "value": value,
        "shape_set": shape_set or ["unknown"], "aggregation": aggregation,
        "boundary": boundary, "source": source, "scope": scope,
    }, sort_keys=True)
    return {
        "schema": MEASUREMENT_SCHEMA,
        "measurement_id": "legacy-" + hashlib.sha256(seed.encode()).hexdigest()[:16],
        "value": value,
        "collected_at": record.get("measured_at") or dt.datetime.now(dt.timezone.utc).isoformat(),
        "identity": {
            "shape_set": shape_set or ["unknown"],
            "aggregation": aggregation,
            "unit": "ms" if metric == "latency_ms" else "ratio",
            "boundary": boundary,
            "source": source,
            "scope": scope,
            "sample": {"count": count, "method": "unknown" if count == "unknown" else "declared"},
            "baseline_ref": baseline_ref or {"kind": "legacy_unknown"},
            "comparator_ref": comparator_ref or {
                "kind": "named", "value": record.get("comparator", "unknown")},
        },
    }


def _load(path: str) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _load_finalization_input(path: str, label: str) -> tuple[Any | None, dict | None]:
    try:
        return _load(path), None
    except FileNotFoundError:
        code = {
            "report": "final_report_missing",
            "canonical": "canonical_entry_missing",
            "state": "state_missing",
        }.get(label, "artifact_missing")
        return None, {"field": label, "code": code,
                      "detail": f"required {label} artifact is missing: {path}", "severity": "error"}
    except (OSError, json.JSONDecodeError) as exc:
        return None, {"field": label, "code": "unreadable",
                      "detail": f"cannot read {label} artifact {path}: {exc}", "severity": "error"}


def _report_is_captain(report: dict) -> bool:
    return any(key in report for key in ("champion_ref", "winning_stage", "served_range"))


def reconcile(canonical: dict, report: dict, state: dict,
              canonical_ref: str | None = None) -> list[dict]:
    """Check that final artifacts describe one run, one generation, one identity."""
    report = project_final_report(report)
    out = validate_run_decision(canonical)
    out.extend(validate_final_report(report, captain=_report_is_captain(report)))
    try:
        from run_state import validate_state
        out.extend(validate_state(state))
    except ImportError:
        _finding(out, "state", "unavailable", "run_state module is unavailable")
        return out
    if not all(isinstance(x, dict) for x in (canonical, report, state)):
        return out
    values = {key: (canonical.get(key), report.get(key), state.get(key))
              for key in ("run_id", "generation")}
    for key, triple in values.items():
        if len(set(triple)) != 1:
            _finding(out, key, "mismatch",
                     f"canonical/report/state disagree: {triple!r}")
    if not same_identity(canonical.get("measurement"), report.get("measurement")):
        _finding(out, "measurement.identity", "identity_mismatch",
                 "canonical and final-report measurements must have identical identity")
    final = canonical.get("final")
    if isinstance(final, dict):
        for key in ("stage", "outcome", "arbitration", "status"):
            if final.get(key) != report.get(key):
                _finding(out, f"final.{key}", "mismatch",
                         "canonical final domains must equal final-report domains")
    if canonical_ref and _reference(report.get("canonical_ref")):
        ref = report["canonical_ref"]
        if isinstance(ref, str) and os.path.normpath(ref) != os.path.normpath(canonical_ref):
            _finding(out, "canonical_ref", "mismatch",
                     f"report references {ref!r}, expected {canonical_ref!r}")
    if state.get("state") != "finalizing":
        _finding(out, "state.state", "state", "strict finalization requires state=finalizing")
    open_obligations = [o.get("obligation_id") for o in state.get("obligations", [])
                        if isinstance(o, dict) and o.get("status") == "open"]
    if open_obligations:
        _finding(out, "state.obligations", "obligations_open",
                 f"cannot finalize with open obligations: {open_obligations}")
    done_kinds = {o.get("kind") for o in state.get("obligations", [])
                  if isinstance(o, dict) and o.get("status") == "done"}
    required = {
        "canonical_entry": "canonical_entry_missing",
        "comparator": "comparator_missing",
        "profile": "profile_missing",
        "report": "final_report_missing",
    }
    if canonical.get("branch_converge") is not None:
        required.update({"arm_gate": "arm_gate_missing", "sweep": "resweep_missing"})
    for kind, code in required.items():
        if kind not in done_kinds:
            _finding(out, f"state.obligations.{kind}", code,
                     f"finalization requires a completed {kind} obligation")
    return out


def _print_result(kind: str, findings: list[dict]) -> None:
    print(json.dumps({"kind": kind, "ok": not findings, "findings": findings}, indent=2))


def _atomic_json(path: str, doc: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".canonical.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _require_valid(kind: str, doc: Any) -> dict:
    if not isinstance(doc, dict):
        raise ValueError(f"{kind} must be a JSON object")
    findings = validate(kind, doc)
    if findings:
        first = findings[0]
        raise ValueError(f"invalid {kind}: {first['field']}: {first['detail']}")
    return doc


def _load_object(path: str, label: str) -> dict:
    try:
        return _require_valid(label, _load(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} at {path}: {exc}") from exc


def _load_json_value(path: str, label: str) -> Any:
    try:
        return _load(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} at {path}: {exc}") from exc


def _load_report_extensions(path: str | None) -> dict:
    """Load user-authored report fields without allowing canonical identity to be overwritten."""
    if not path:
        return {}
    extensions = _load_json_value(path, "final report extensions")
    if not isinstance(extensions, dict):
        raise ValueError("--extra-fields must name a JSON object")
    protected = sorted(FINAL_REPORT_BASE_FIELDS & set(extensions))
    if protected:
        raise ValueError("--extra-fields cannot overwrite canonical base fields: "
                         + ", ".join(protected))
    return extensions


def _split_csv(value: str, label: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be a non-empty comma-separated list of unique ids")
    return values


def _timeline_event(canonical: dict, phase: str, event: str, *, once: bool = False) -> None:
    """Append one timeline event. `once` makes the append idempotent within a generation.

    The close path has to reach a fixed point. `finalize` rewrites this file, so an unconditional
    append changes its sha256 on every call -- including the call that happens after the final
    report has already hashed it into `canonical_ref`. Re-running the pair in either order then
    keeps invalidating the other's snapshot, and no honest owner converges: all four in one
    campaign hit it and had to reconcile by hand. A terminal event is a fact about the generation,
    not a counter, so recording it twice is the bug rather than the second call being one.
    """
    timeline = canonical.setdefault("timeline", [])
    if once and any(isinstance(item, dict) and item.get("phase") == phase
                    and item.get("event") == event
                    and item.get("generation") == canonical["generation"]
                    for item in timeline):
        return
    sequence = max((item.get("sequence", 0) for item in timeline
                    if isinstance(item, dict) and isinstance(item.get("sequence"), int)),
                   default=0) + 1
    timeline.append({
        "event_id": f"{phase}-{event}-g{canonical['generation']}-{sequence}",
        "phase": phase,
        "event": event,
        "at": _now(),
        "sequence": sequence,
        "generation": canonical["generation"],
    })


def _state_init(args: argparse.Namespace) -> int:
    from run_state import init_state
    measurement = _load_object(args.measurement, "measurement")
    stage = _load_object(args.stage, "stage-result")
    if stage.get("status") == "completed" and not same_identity(
            measurement, stage.get("measurement")):
        raise ValueError("initial stage measurement identity must equal the canonical measurement")
    state = init_state(args.state, args.run_id)
    canonical = {
        "schema": RUN_DECISION_SCHEMA,
        "run_id": args.run_id,
        "generation": state["generation"],
        "state": state["state"],
        "producer": {"role": args.owner, "mechanism": "canonical_record.py init-run"},
        "measurement": measurement,
        "timeline": [],
        "stages": [stage],
    }
    _timeline_event(canonical, stage["phase"],
                    "completed" if stage["status"] == "completed" else "entered")
    _require_valid("run-decision", canonical)
    _atomic_json(args.canonical, canonical)
    print(json.dumps({"run_id": args.run_id, "generation": state["generation"],
                      "state": state["state"], "canonical": args.canonical,
                      "run_state": args.state}, indent=2))
    return 0


def _record_stage(args: argparse.Namespace) -> int:
    canonical = _load_object(args.canonical, "run-decision")
    stage = _load_object(args.stage, "stage-result")
    stages = canonical.setdefault("stages", [])
    stages[:] = [item for item in stages if item.get("stage_id") != stage["stage_id"]]
    stages.append(stage)
    _timeline_event(canonical, stage["phase"],
                    "completed" if stage["status"] == "completed" else "entered")
    _require_valid("run-decision", canonical)
    _atomic_json(args.canonical, canonical)
    print(json.dumps({"canonical": args.canonical, "stage_id": stage["stage_id"],
                      "generation": canonical["generation"]}, indent=2))
    return 0


def _write_worker_result(args: argparse.Namespace) -> int:
    measurement = _load_object(args.measurement, "measurement") if args.measurement else None
    doc = {
        "schema": WORKER_RESULT_SCHEMA,
        "run_id": args.run_id,
        "generation": args.generation,
        "producer_role": args.role,
        "stage": args.stage,
        "status": args.status,
        "produced_at": _now(),
        "work_kinds": args.work_kind or [],
        "requests": [{"kind": "request", "path": ref} for ref in args.request_ref or []],
    }
    if measurement is not None:
        doc["measurement"] = measurement
    _require_valid("worker-result", doc)
    _atomic_json(args.out, doc)
    print(json.dumps({"worker_result": args.out, "role": args.role,
                      "status": args.status}, indent=2))
    return 0


def _request_branch(args: argparse.Namespace) -> int:
    doc = {
        "schema": BRANCH_REQUEST_SCHEMA,
        "request_id": args.request_id,
        "run_id": args.run_id,
        "generation": args.generation,
        "requester_role": "direction",
        "requested_at": _now(),
        "candidates": args.candidate_ref,
        "evidence_ref": args.evidence_ref,
        "state": "requested",
    }
    _require_valid("branch-request", doc)
    _atomic_json(args.out, doc)
    print(json.dumps({"branch_request": args.out, "request_id": args.request_id,
                      "candidates": len(args.candidate_ref)}, indent=2))
    return 0


def _request_structure_suspect(args: argparse.Namespace) -> int:
    source = _load_object(args.source_ref, "worker-result")
    if source.get("producer_role") != "deep":
        raise ValueError("only a deep worker result may emit structure_suspect.json")
    doc = {
        "schema": STRUCTURE_SUSPECT_SCHEMA,
        "candidate_id": args.candidate_id,
        "state": args.state,
        "independent": args.independent,
        "candidate_ref": {"kind": "structure_candidate", "path": args.source_ref,
                          "candidate_id": args.candidate_id},
        "source_ref": {"kind": "deep_worker_result", "path": args.source_ref},
        "evidence_ref": {"kind": "evidence", "path": args.evidence_ref},
        "requester_role": "deep",
    }
    _require_valid("structure-suspect", doc)
    _atomic_json(args.out, doc)
    print(json.dumps({"structure_suspect": args.out, "candidate_id": args.candidate_id}, indent=2))
    return 0


def _request_resweep(args: argparse.Namespace) -> int:
    parent = _load_json_value(args.parent_result, "resweep request parent")
    if not isinstance(parent, dict) or parent.get("schema") not in RESWEEP_PARENT_SCHEMAS:
        raise ValueError("resweep request parent must be a canonical worker-result or branch-converge")
    requester_role = getattr(args, "requester_role", "deep")
    if requester_role == "deep":
        if parent.get("schema") != WORKER_RESULT_SCHEMA or parent.get("producer_role") != "deep":
            raise ValueError("a deep worker may request a resweep only from its own worker-result")
        run_id, generation = parent["run_id"], parent["generation"]
    else:
        run_id, generation = getattr(args, "run_id", None), getattr(args, "generation", None)
        if not _nonempty(run_id) or not isinstance(generation, int) or generation < 1:
            raise ValueError("captain/direct_owner resweep requests require --run-id and --generation")
        if parent.get("schema") == WORKER_RESULT_SCHEMA and (
                parent.get("run_id") != run_id or parent.get("generation") != generation):
            raise ValueError("captain/direct_owner request identity must match its parent worker-result")
    doc = {
        "schema": RESWEEP_REQUEST_SCHEMA,
        "request_id": args.request_id,
        "run_id": run_id,
        "generation": generation,
        "requester_role": requester_role,
        "requested_at": _now(),
        "parent_result_ref": args.parent_result,
        "evidence_ref": args.evidence_ref,
        "state": "requested",
    }
    _require_valid("resweep-request", doc)
    _atomic_json(args.out, doc)
    print(json.dumps({"resweep_request": args.out, "request_id": args.request_id,
                      "requested_by": requester_role}, indent=2))
    return 0


def _resolve_resweep(args: argparse.Namespace) -> int:
    request = _load_object(args.request, "resweep-request")
    measurement = _load_object(args.measurement, "measurement")
    doc = {
        "schema": RESWEEP_RESULT_SCHEMA,
        "request_ref": args.request,
        "executor_role": args.owner,
        "completed_at": _now(),
        "measurement": measurement,
        "run_id": request["run_id"],
        "generation": request["generation"],
    }
    _require_valid("resweep-result", doc)
    _atomic_json(args.out, doc)
    print(json.dumps({"resweep_result": args.out, "request_id": request["request_id"],
                      "executed_by": args.owner}, indent=2))
    return 0


def _request_sweep(args: argparse.Namespace) -> int:
    """Write the immutable, execution-free request for any sweep purpose."""
    context = _load_json_value(args.context, "sweep context")
    if not isinstance(context, dict) or not context:
        raise ValueError("--context must name a non-empty JSON object")
    doc = {
        "schema": SWEEP_REQUEST_SCHEMA,
        "request_id": args.request_id,
        "run_id": args.run_id,
        "generation": args.generation,
        "purpose": args.purpose,
        "requester_role": args.requester_role,
        "requested_at": _now(),
        "source": args.source_ref,
        "context": context,
        "parent": args.parent_ref,
        "attempt": {"attempt_id": args.attempt_id},
        "lineage": {"lineage_id": args.lineage_id,
                    "root_request_id": args.root_request_id or args.request_id},
        "state": "requested",
        "work_kind": args.work_kind,
        "change_scope": args.change_scope,
        "axis_kind": args.axis_kind,
        "realization_kind": "request",
    }
    _require_valid("sweep-request", doc)
    _atomic_json(args.out, doc)
    print(json.dumps({"sweep_request": args.out, "request_id": args.request_id,
                      "purpose": args.purpose, "requested_by": args.requester_role}, indent=2))
    return 0


def _resolve_sweep(args: argparse.Namespace) -> int:
    """Write a sweep-worker or owner-executed result without changing request identity."""
    request = _load_object(args.request, "sweep-request")
    measurement = _load_object(args.measurement, "measurement")
    doc = {
        "schema": SWEEP_RESULT_SCHEMA,
        "request_ref": args.request,
        "run_id": request["run_id"],
        "generation": request["generation"],
        "purpose": request["purpose"],
        "executor_role": args.owner,
        "completed_at": _now(),
        "measurement": measurement,
        "attempt": request["attempt"],
        "lineage": request["lineage"],
        "work_kind": request["work_kind"],
        "change_scope": request["change_scope"],
        "axis_kind": request["axis_kind"],
        "realization_kind": "execution",
    }
    _require_valid("sweep-result", doc)
    _atomic_json(args.out, doc)
    write_producer_receipt(args.out, "sweep")
    print(json.dumps({"sweep_result": args.out, "request_id": request["request_id"],
                      "purpose": request["purpose"], "executed_by": args.owner}, indent=2))
    return 0


def _write_arm_result(args: argparse.Namespace) -> int:
    measurement = _load_object(args.measurement, "measurement") if args.measurement else None
    doc = {
        "schema": ARM_RESULT_SCHEMA,
        "arm_id": args.arm_id,
        "lever": args.lever,
        "direction_class": args.direction_class,
        "iters_used": args.iters_used,
        "completion": args.completion,
        "verdict": args.verdict,
        "gate_stage": args.gate_stage,
    }
    if measurement is not None:
        doc["measurement"] = measurement
    if getattr(args, "sweep_result", None):
        doc["config_basis"] = {"mode": "own_sweep", "sweep_ref": args.sweep_result}
    _require_valid("arm-result", doc)
    _atomic_json(args.out, doc)
    write_producer_receipt(args.out, "arm")
    print(json.dumps({"arm_result": args.out, "arm_id": args.arm_id,
                      "verdict": args.verdict}, indent=2))
    return 0


def _branch_converge(args: argparse.Namespace) -> int:
    planned = _split_csv(args.planned, "--planned")
    records = []
    for path in args.arm_result:
        try:
            record = _load_object(path, "arm-result")
        except ValueError as exc:
            raise ValueError(
                f"{path} is not a canonical ArmResult. Run `adapt-arm-result` first; "
                "branch-converge never mixes legacy and canonical arm formats.") from exc
        records.append({"arm_id": record["arm_id"], "ref": path, "record": record})
    completed = [entry["arm_id"] for entry in records]
    if len(completed) != len(set(completed)):
        raise ValueError("each --arm-result must have a unique arm_id")
    if set(completed) - set(planned):
        raise ValueError("every supplied ArmResult must be listed in --planned")
    if set(planned) != set(completed):
        raise ValueError("supply one complete ArmResult for every planned arm")
    measurement = _load_object(args.measurement, "measurement")
    eligible = [
        entry["arm_id"] for entry in records
        if entry["record"]["completion"] == "complete"
        and entry["record"]["gate_stage"] in ("b", "c")
        and entry["record"]["verdict"] == "supported"
        and same_identity(entry["record"].get("measurement"), measurement)
    ]
    winner = None if args.winner == "none" else args.winner
    post_merge: dict[str, Any] = {"status": "not_required"}
    if winner is not None:
        sweep_result = getattr(args, "sweep_result", None)
        legacy_resweep = getattr(args, "resweep_result", None)
        if bool(sweep_result) == bool(legacy_resweep):
            raise ValueError("a selected winner requires exactly one --sweep-result or legacy --resweep-result")
        result_path = sweep_result or legacy_resweep
        result_kind = "sweep-result" if sweep_result else "resweep-result"
        result = _load_object(result_path, result_kind)
        request_path = _path_reference(result.get("request_ref"))
        if not request_path:
            raise ValueError("sweep result request_ref must contain a resolvable artifact path")
        if not os.path.isabs(request_path):
            sibling_request = os.path.join(os.path.dirname(result_path), request_path)
            if os.path.exists(sibling_request):
                request_path = sibling_request
        request = _load_object(request_path, "sweep-request" if sweep_result else "resweep-request")
        if (request["run_id"] != result["run_id"] or
                request["generation"] != result["generation"]):
            raise ValueError("sweep request and result must share run_id and generation")
        if sweep_result:
            if request.get("purpose") != "post_merge" or result.get("purpose") != "post_merge":
                raise ValueError("a branch winner requires purpose=post_merge typed sweep artifacts")
            if validate_sweep_pair(request, result):
                raise ValueError("typed sweep request/result must preserve immutable lineage")
        if not same_identity(result["measurement"], measurement):
            raise ValueError("sweep measurement identity must equal branch convergence identity")
        post_merge = {
            "status": "completed",
            "request_ref": result["request_ref"],
            "result_ref": result_path,
            "result": result,
            "collected_at": result["completed_at"],
            "measurement": result["measurement"],
        }
    doc = {
        "schema": BRANCH_CONVERGE_SCHEMA,
        "planned": planned,
        "completed": completed,
        "eligible": eligible,
        "winner": winner,
        "measurement": measurement,
        "collected_at": _now(),
        "arm_results": records,
        "post_merge_resweep": post_merge,
        "direction_decision": {
            "winner": winner,
            "evidence_ref": args.decision_evidence,
        },
        "producer": {"role": args.owner, "mechanism": "canonical_record.py branch-converge"},
    }
    if winner is not None:
        doc["run_id"] = result["run_id"]
        doc["generation"] = result["generation"]
    _require_valid("branch-converge", doc)
    _atomic_json(args.out, doc)
    write_producer_receipt(args.out, "branch")
    print(json.dumps({"branch_converge": args.out, "winner": winner,
                      "eligible": eligible}, indent=2))
    return 0


def _attach_branch_converge(args: argparse.Namespace) -> int:
    canonical = _load_object(args.canonical, "run-decision")
    converge = _load_object(args.branch_converge, "branch-converge")
    if not same_identity(canonical["measurement"], converge["measurement"]):
        raise ValueError("branch convergence identity must equal canonical run identity")
    if ("run_id" in converge and converge["run_id"] != canonical["run_id"]) or (
            "generation" in converge and converge["generation"] != canonical["generation"]):
        raise ValueError("branch convergence run_id and generation must equal the canonical run")
    canonical["branch_converge"] = converge
    _timeline_event(canonical, "converge", "completed")
    _require_valid("run-decision", canonical)
    _atomic_json(args.canonical, canonical)
    print(json.dumps({"canonical": args.canonical, "branch_converge": args.branch_converge}, indent=2))
    return 0


def _write_final_report(args: argparse.Namespace) -> int:
    measurement = _load_object(args.measurement, "measurement")
    known_wrong = _load_json_value(args.known_wrong, "known_wrong") if args.known_wrong else []
    if not isinstance(known_wrong, list):
        raise ValueError("--known-wrong must name a JSON array")
    extensions = _load_report_extensions(getattr(args, "extra_fields", None))
    work_root = os.path.dirname(os.path.abspath(args.out)) or "."
    anchor = _load_json_value(args.anchor, "anchor")
    if not isinstance(anchor, dict):
        raise ValueError("--anchor must name an object")
    refs = {
        "canonical": _writer_ref(args.canonical_ref, work_root=work_root),
        "close_audit": _writer_ref(args.close_audit_ref, work_root=work_root),
    }
    lifecycle = getattr(args, "lifecycle_profile", None)
    if not lifecycle:
        lifecycle = "direct-owner" if args.owner == "direct_owner" else "front-end"
    if lifecycle in ("front-end", "deep-consumer"):
        refs["champion"] = _writer_ref(args.champion_ref, work_root=work_root)
    if lifecycle == "front-end":
        refs["champion_gate"] = _writer_ref(args.champion_gate_ref, work_root=work_root)
    if anchor.get("ref"):
        refs["anchor"] = _writer_ref(anchor["ref"], work_root=work_root)
        anchor = dict(anchor, ref=refs["anchor"])
    audit_path = (args.close_audit_ref if os.path.isabs(args.close_audit_ref)
                  else os.path.join(work_root, args.close_audit_ref))
    audit_doc = _load_json_value(audit_path, "close audit")
    audit = {
        "strict_pass": audit_doc.get("strict_pass") if isinstance(audit_doc, dict) else None,
        "high_severity_count": audit_doc.get("high_severity_count", 0)
        if isinstance(audit_doc, dict) else 0,
        "blocked_count": len(audit_doc.get("blocked_checks") or [])
        if isinstance(audit_doc, dict) else 0,
        "verified_by": audit_doc.get("verified_by") if isinstance(audit_doc, dict) else None,
    }
    receipt_values = list(getattr(args, "producer_receipt", None) or [])
    # THIS WRITER OWNS THE `final_report` RECEIPT, so a caller-supplied one is refused rather than
    # embedded. Embedding it cannot work and cannot be made to work: the receipt would go inside the
    # bytes it hashes, and `_atomic_json` below then changes those bytes, so the digest it just
    # recorded is stale the instant it lands. Every later verification reads it as a mismatch
    # forever, and no write order fixes it. The sidecar written after `_atomic_json` is the route
    # that does work -- a separate file, hashing a finished one -- and this function already writes
    # it, so there was never a reason to accept the caller's.
    #
    # Measured on one campaign: three of four results were held below clean acceptance by
    # exactly this, and it was reported to the pack as "self-referential receipts can never
    # validate". Half right -- the embedded route cannot, the sidecar route can. The run that used
    # the sidecar closed with zero high findings.
    out_real = os.path.realpath(args.out)
    for value in receipt_values:
        if isinstance(value, dict):
            doc = value
        else:
            try:
                doc = _load(str(value))
            except (OSError, TypeError, json.JSONDecodeError):
                continue          # unreadable here is reported by the projection below, not swallowed
        if not isinstance(doc, dict):
            continue
        artifact = doc.get("artifact") if isinstance(doc.get("artifact"), dict) else {}
        ref = artifact.get("ref")
        covers_out = isinstance(ref, str) and os.path.realpath(
            ref if os.path.isabs(ref) else os.path.join(work_root, ref)) == out_real
        if doc.get("artifact_role") == "final_report" or covers_out:
            raise ValueError(
                "--producer-receipt may not carry a receipt for this command's own output "
                f"({args.out}). A receipt embedded in the artifact it covers is invalidated by the "
                "write that embeds it, so it can never verify. This command writes that receipt "
                "itself, as a sidecar, after the report is on disk -- drop this one and let it. "
                "Pass --producer-receipt only for the artifacts the report REFERENCES (sweep, "
                "champion, audit), which are separate files and hash stably")
    audit_receipt = audit_path + ".receipt.json"
    if os.path.isfile(audit_receipt) and audit_receipt not in receipt_values:
        receipt_values.append(audit_receipt)
    receipts = []
    for value in receipt_values:
        receipt, findings = _receipt_projection(value, work_root=work_root)
        if findings:
            raise ValueError(f"invalid producer receipt {value}: {findings[0]['detail']}")
        if receipt:
            receipts.append(receipt)
    doc = {
        "schema": FINAL_REPORT_SCHEMA,
        "run_id": args.run_id,
        "generation": args.generation,
        "lifecycle_profile": lifecycle,
        "stage": args.stage,
        "outcome": args.outcome,
        "arbitration": args.arbitration,
        "status": args.status,
        "measurement": measurement,
        "refs": refs,
        "canonical_ref": refs["canonical"],
        "anchor": anchor,
        "close_audit_ref": refs["close_audit"],
        "scope": _load_json_value(args.scope, "scope"),
        "deferred": _load_json_value(args.deferred, "deferred"),
        "deep_arbitration": _load_json_value(args.deep_arbitration, "deep arbitration"),
        "known_wrong": known_wrong,
        "producer": {"role": args.owner, "mechanism": "canonical_record.py write-final-report"},
        "audit": audit,
        "provenance": {"receipts": receipts},
        "integrity": {
            "source_schema": FINAL_REPORT_SCHEMA,
            "normalization_status": "canonical",
            "legacy_unverified": False,
            "eligible_clean": False,
        },
    }
    if "champion" in refs:
        doc["champion_ref"] = refs["champion"]
    if "champion_gate" in refs:
        doc["champion_gate_ref"] = refs["champion_gate"]
    doc.update(extensions)
    _require_valid("final-report", doc)
    _atomic_json(args.out, doc)
    write_producer_receipt(args.out, "final_report")
    print(json.dumps({"final_report": args.out, "owner": args.owner,
                      "status": args.status}, indent=2))
    return 0


def _record_final(args: argparse.Namespace) -> int:
    canonical = _load_object(args.canonical, "run-decision")
    report = _load_object(args.report, "final-report")
    if canonical["run_id"] != report["run_id"] or canonical["generation"] != report["generation"]:
        raise ValueError("canonical and final report must share run_id and generation")
    if not same_identity(canonical["measurement"], report["measurement"]):
        raise ValueError("canonical and final report measurements must have identical identity")
    if canonical["producer"].get("role") != report["producer"].get("role"):
        raise ValueError("the canonical record and final report must name the same accountable owner")
    canonical["state"] = "finalizing"
    canonical["final"] = {
        key: report[key] for key in ("stage", "outcome", "arbitration", "status")
    }
    canonical["final"]["report_ref"] = args.report
    _timeline_event(canonical, "finalization", "entered")
    _require_valid("run-decision", canonical)
    _atomic_json(args.canonical, canonical)
    print(json.dumps({"canonical": args.canonical, "report": args.report,
                      "state": canonical["state"]}, indent=2))
    return 0


def _state_open(args: argparse.Namespace) -> int:
    from run_state import open_obligation
    state = open_obligation(args.state, args.kind, args.reason, args.obligation_id)
    print(json.dumps(state, indent=2))
    return 0


def _state_done(args: argparse.Namespace) -> int:
    from run_state import complete_obligation
    state = complete_obligation(args.state, args.obligation, args.generation, args.evidence_ref)
    print(json.dumps(state, indent=2))
    return 0


def _state_begin_finalization(args: argparse.Namespace) -> int:
    from run_state import begin_finalization
    state = begin_finalization(args.state, args.generation)
    print(json.dumps(state, indent=2))
    return 0


def _sync_run_state(args: argparse.Namespace) -> int:
    """Advance the canonical record only from the authoritative state-machine generation."""
    canonical = _load_object(args.canonical, "run-decision")
    try:
        from run_state import validate_state
        state = _load(args.state)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run state at {args.state}: {exc}") from exc
    findings = validate_state(state)
    if findings:
        first = findings[0]
        raise ValueError(f"invalid run state: {first['field']}: {first['detail']}")
    if canonical["run_id"] != state["run_id"]:
        raise ValueError("canonical record and run state must share run_id")
    if state["state"] != "open":
        raise ValueError("sync-run-state only accepts open state; record final before entering finalization")
    canonical["generation"] = state["generation"]
    canonical["state"] = state["state"]
    _timeline_event(canonical, "finalization", "reopened")
    _require_valid("run-decision", canonical)
    _atomic_json(args.canonical, canonical)
    print(json.dumps({"canonical": args.canonical, "run_state": args.state,
                      "generation": canonical["generation"]}, indent=2))
    return 0


def _finalize(args: argparse.Namespace) -> int:
    canonical, canonical_error = _load_finalization_input(args.canonical, "canonical")
    report, report_error = _load_finalization_input(args.report, "report")
    state, state_error = _load_finalization_input(args.state, "state")
    load_errors = [x for x in (canonical_error, report_error, state_error) if x]
    if load_errors:
        _print_result("finalize", load_errors)
        return 1
    findings = reconcile(canonical, report, state, args.canonical_ref)
    if findings:
        _print_result("finalize", findings)
        return 1
    from run_state import finalize_state
    try:
        final_state = finalize_state(args.state, expected_generation=state["generation"])
    except ValueError as exc:
        _print_result("finalize", [{"field": "state", "code": "state", "detail": str(exc),
                                    "severity": "error"}])
        return 1
    canonical["state"] = final_state["state"]
    _timeline_event(canonical, "finalization", "finalized", once=True)
    _atomic_json(args.canonical, canonical)
    if args.out:
        _atomic_json(args.out, canonical)
    print(json.dumps({"ok": True, "generation": final_state["generation"],
                      "state": final_state["state"]}, indent=2))
    return 0


def _project_command(args: argparse.Namespace) -> int:
    report = _load_json_value(args.report, "final report")
    source_path = os.path.abspath(args.report)
    work_root = os.path.abspath(args.work_root or os.path.dirname(source_path))
    projected = project_final_report(report, work_root=work_root, source_path=source_path)
    _atomic_json(args.out, projected)
    failed = projected["integrity"]["normalization_status"] == "failed"
    print(json.dumps({
        "projection": args.out,
        "source_schema": projected["integrity"]["source_schema"],
        "normalization_status": projected["integrity"]["normalization_status"],
        "eligible_clean": projected["integrity"]["eligible_clean"],
    }, indent=2))
    return 1 if args.strict and failed else 0


def _write_receipt_command(args: argparse.Namespace) -> int:
    write_producer_receipt(
        args.artifact, args.artifact_role, tool_path=args.executable,
        timing=args.timing, fallback_reason=args.fallback_reason,
        input_schema=args.input_schema, output_schema=args.output_schema,
        validator_path=args.validator, receipt_path=args.out,
    )
    print(json.dumps({"producer_receipt": args.out or args.artifact + ".receipt.json"}, indent=2))
    return 0


def _selftest() -> int:
    import shutil
    now = "2026-09-01T00:00:00Z"
    legacy = legacy_measurement({"round": 1, "latency_ms": 1.2, "measured_at": now})
    assert legacy and legacy["identity"]["scope"] == "unknown"
    assert legacy["identity"]["source"] == "legacy_latency"
    assert not validate_measurement(legacy), validate_measurement(legacy)
    forged = json.loads(json.dumps(legacy))
    forged["identity"]["source"] = "device_timing"
    assert any(x["code"] == "device_scope" for x in validate_measurement(forged))

    measurement = {
        "schema": MEASUREMENT_SCHEMA, "measurement_id": "m1", "value": 1.1,
        "collected_at": now,
        "identity": {
            "shape_set": ["s1"], "aggregation": "geomean", "unit": "ratio",
            "boundary": "kernel", "source": "device_timing", "scope": "device",
            "sample": {"count": 20, "method": "interleaved"},
            "baseline_ref": "baseline.json", "comparator_ref": "champion.json",
        },
    }
    arm = {
        "schema": ARM_RESULT_SCHEMA, "arm_id": "a", "lever": "split_k",
        "direction_class": "structural", "iters_used": 3, "completion": "complete",
        "verdict": "supported", "gate_stage": "b", "measurement": measurement,
    }
    assert not validate_arm_result(arm), validate_arm_result(arm)
    assert any(x["code"] == "forbidden" for x in validate_arm_result(dict(arm, result="win")))
    worker = {
        "schema": WORKER_RESULT_SCHEMA, "run_id": "run-a", "generation": 1,
        "producer_role": "deep", "stage": "resweep", "status": "completed",
        "produced_at": now, "measurement": measurement,
        "work_kinds": ["climb", "resweep_request"], "requests": [],
    }
    assert not validate_worker_result(worker), validate_worker_result(worker)
    assert any(x["code"] == "forbidden_deep_branch" for x in validate_worker_result(
        dict(worker, work_kinds=["resweep"])))
    branch_request = {
        "schema": BRANCH_REQUEST_SCHEMA, "request_id": "branch-a", "run_id": "run-a",
        "generation": 1, "requester_role": "direction", "requested_at": now,
        "candidates": ["census.json#candidate-a"], "evidence_ref": "profile.json",
        "state": "requested",
    }
    assert not validate_branch_request(branch_request), validate_branch_request(branch_request)
    resweep_request = {
        "schema": RESWEEP_REQUEST_SCHEMA, "request_id": "resweep-a", "run_id": "run-a",
        "generation": 1, "requester_role": "deep", "requested_at": now,
        "parent_result_ref": "worker_result.json", "evidence_ref": "profile.json",
        "state": "requested",
    }
    assert not validate_resweep_request(resweep_request), validate_resweep_request(resweep_request)
    captain_resweep_request = dict(resweep_request, request_id="resweep-captain",
                                   requester_role="captain")
    assert not validate_resweep_request(captain_resweep_request), \
        validate_resweep_request(captain_resweep_request)
    direct_owner_resweep_request = dict(resweep_request, request_id="resweep-direct-owner",
                                        requester_role="direct_owner")
    assert not validate_resweep_request(direct_owner_resweep_request), \
        validate_resweep_request(direct_owner_resweep_request)
    assert any(x["code"] == "unauthorized_resweep" for x in validate_resweep_request(
        dict(resweep_request, measurement=measurement)))
    resweep_result = {
        "schema": RESWEEP_RESULT_SCHEMA, "request_ref": "resweep_request.json",
        "run_id": "run-a", "generation": 1, "executor_role": "captain",
        "completed_at": now, "measurement": measurement,
    }
    assert not validate_resweep_result(resweep_result), validate_resweep_result(resweep_result)
    sweep_request = {
        "schema": SWEEP_REQUEST_SCHEMA, "request_id": "sweep-a", "run_id": "run-a",
        "generation": 1, "purpose": "arm_local", "requester_role": "arm",
        "requested_at": now, "source": "arm/profile.json", "context": {"objective": "geomean"},
        "parent": "arm_result.json", "attempt": {"attempt_id": "attempt-a"},
        "lineage": {"lineage_id": "lineage-a", "root_request_id": "sweep-a"},
        "state": "requested", "work_kind": "sweep", "change_scope": "config",
        "axis_kind": "config", "realization_kind": "request",
    }
    assert not validate_sweep_request(sweep_request), validate_sweep_request(sweep_request)
    assert any(x["code"] == "unauthorized_sweep" for x in validate_sweep_request(
        dict(sweep_request, measurement=measurement)))
    sweep_result = {
        "schema": SWEEP_RESULT_SCHEMA, "request_ref": "sweep_request.json", "run_id": "run-a",
        "generation": 1, "purpose": "arm_local", "executor_role": "captain",
        "completed_at": now, "measurement": measurement, "attempt": {"attempt_id": "attempt-a"},
        "lineage": {"lineage_id": "lineage-a", "root_request_id": "sweep-a"},
        "work_kind": "sweep", "change_scope": "config", "axis_kind": "config",
        "realization_kind": "execution",
    }
    assert not validate_sweep_pair(sweep_request, sweep_result), \
        validate_sweep_pair(sweep_request, sweep_result)
    assert any(x["code"] == "lineage_mismatch" for x in validate_sweep_pair(
        sweep_request, dict(sweep_result, attempt={"attempt_id": "attempt-b"})))
    converge = {
        "schema": BRANCH_CONVERGE_SCHEMA, "planned": ["a"], "completed": ["a"],
        "eligible": ["a"], "winner": "a", "measurement": measurement, "collected_at": now,
        "arm_results": [{"arm_id": "a", "ref": "arms/a/arm_result.json", "record": arm}],
        "post_merge_resweep": {"status": "completed", "request_ref": "resweep_request.json",
                                "result_ref": "resweep_result.json", "result": resweep_result,
                                "collected_at": now, "measurement": measurement},
        "direction_decision": {"winner": "a", "evidence_ref": "direction_evidence.json"},
        "producer": {"role": "direction"},
    }
    assert not validate_branch_converge(converge), validate_branch_converge(converge)
    assert any(x["code"] == "resweep_missing" for x in validate_branch_converge(
        dict(converge, post_merge_resweep={"status": "not_required"})))
    root = tempfile.mkdtemp(prefix="canonical_record_selftest_")
    try:
        state_path = os.path.join(root, "run_state.json")
        canonical_path = os.path.join(root, "canonical_record.json")
        report_path = os.path.join(root, "final_report.json")
        from run_state import begin_finalization, complete_obligation, init_state, open_obligation
        init_state(state_path, "run-a")
        state = _load(state_path)
        for kind, ref in (("canonical_entry", "canonical_record.json"),
                          ("comparator", "champion.json"),
                          ("profile", "exp/final_profile.json"),
                          ("report", "final_report.json")):
            state = open_obligation(state_path, kind, f"record {kind}")
            obligation = state["obligations"][-1]
            state = complete_obligation(state_path, obligation["obligation_id"],
                                        state["generation"], ref)
        state = begin_finalization(state_path, state["generation"])
        report = {
            "schema": FINAL_REPORT_SCHEMA, "run_id": "run-a", "generation": state["generation"],
            "stage": "finalization", "outcome": "win", "arbitration": "accept",
            "status": "closed", "measurement": measurement,
            "canonical_ref": "canonical_record.json", "champion_ref": "champion.json",
            "champion_gate_ref": "champion_gate.json",
            "anchor": {"type": "pinned_comparator", "ref": "plain_best_config.json"},
            "close_audit_ref": "close_audit.json",
            "scope": {"authorization": "plain-only"}, "deferred": [],
            "deep_arbitration": {"state": "not_authorized", "champion_ms": 9.0},
            "known_wrong": [],
            "producer": {"role": "captain"},
        }
        canonical = {
            "schema": RUN_DECISION_SCHEMA, "run_id": "run-a", "generation": state["generation"],
            "state": "finalizing", "producer": {"role": "captain"}, "measurement": measurement,
            "timeline": [{"event_id": "e1", "phase": "finalization", "event": "entered",
                          "at": now, "sequence": 1, "generation": state["generation"]}],
            "stages": [{"schema": STAGE_RESULT_SCHEMA, "stage_id": "final",
                        "phase": "finalization", "status": "completed",
                        "collected_at": now, "measurement": measurement,
                        "profile_ref": "exp/final_profile.json"}],
            "final": {"stage": "finalization", "outcome": "win", "arbitration": "accept",
                      "status": "closed", "report_ref": "final_report.json"},
        }
        _atomic_json(canonical_path, canonical)
        _atomic_json(report_path, report)
        reconciled = reconcile(canonical, report, state, "canonical_record.json")
        assert not reconciled, reconciled
        args = argparse.Namespace(canonical=canonical_path, report=report_path, state=state_path,
                                  canonical_ref="canonical_record.json", out=None)
        assert _finalize(args) == 0
        assert _load(state_path)["state"] == "closed"
        bad_report = json.loads(json.dumps(report))
        bad_report["measurement"]["identity"]["boundary"] = "end_to_end"
        assert any(x["code"] == "identity_mismatch"
                   for x in reconcile(canonical, bad_report,
                                      dict(state, state="finalizing"), "canonical_record.json"))

        cli_root = os.path.join(root, "cli")
        measurement_path = os.path.join(cli_root, "measurement.json")
        stage_path = os.path.join(cli_root, "stage.json")
        cli_state = os.path.join(cli_root, "run_state.json")
        cli_canonical = os.path.join(cli_root, "canonical_record.json")
        cli_worker = os.path.join(cli_root, "worker_result.json")
        cli_plain_worker = os.path.join(cli_root, "plain_worker_result.json")
        cli_branch_request = os.path.join(cli_root, "branch_request.json")
        cli_request = os.path.join(cli_root, "resweep_request.json")
        cli_captain_request = os.path.join(cli_root, "captain_resweep_request.json")
        cli_direct_owner_request = os.path.join(cli_root, "direct_owner_resweep_request.json")
        cli_result = os.path.join(cli_root, "resweep_result.json")
        cli_sweep_context = os.path.join(cli_root, "sweep_context.json")
        cli_sweep_request = os.path.join(cli_root, "sweep_request.json")
        cli_sweep_result = os.path.join(cli_root, "sweep_result.json")
        cli_post_sweep_request = os.path.join(cli_root, "post_merge_sweep_request.json")
        cli_post_sweep_result = os.path.join(cli_root, "post_merge_sweep_result.json")
        cli_suspect = os.path.join(cli_root, "structure_suspect.json")
        cli_arm_a = os.path.join(cli_root, "arm-a.json")
        cli_arm_b = os.path.join(cli_root, "arm-b.json")
        cli_converge = os.path.join(cli_root, "branch_converge.json")
        cli_report = os.path.join(cli_root, "final_report.json")
        cli_report_fields = os.path.join(cli_root, "final_report_fields.json")
        _atomic_json(measurement_path, measurement)
        _atomic_json(cli_sweep_context, {"objective": "geomean", "axis": "BLOCK_M"})
        _atomic_json(cli_report_fields, {
            "caveats": [],
            "close_audit": {"verified_by": "self", "findings": 0},
            "skeptic_verdict": "not_run",
            "operator_summary": "free-text report fields remain caller-owned",
        })
        cli_anchor = os.path.join(cli_root, "anchor.json")
        cli_scope = os.path.join(cli_root, "scope.json")
        cli_deferred = os.path.join(cli_root, "deferred.json")
        cli_deep = os.path.join(cli_root, "deep.json")
        for name, value in (
            ("champion.json", {"schema": "anonymous.champion/1"}),
            ("champion_gate.json", {"schema": "anonymous.champion_gate/1", "pass": True}),
            ("plain_best_config.json", {"schema": "anonymous.comparator/1"}),
            ("close_audit.json", {"schema": "close_audit", "strict_pass": True,
                                  "high_severity_count": 0, "blocked_checks": []}),
        ):
            _atomic_json(os.path.join(cli_root, name), value)
        _atomic_json(cli_anchor, {"type": "pinned_comparator", "ref": "plain_best_config.json"})
        _atomic_json(cli_scope, {"authorization": "plain-only"})
        _atomic_json(cli_deferred, [])
        _atomic_json(cli_deep, {"state": "not_authorized", "champion_ms": 9.0})
        _atomic_json(stage_path, {
            "schema": STAGE_RESULT_SCHEMA, "stage_id": "anchor", "phase": "anchor",
            "status": "completed", "collected_at": now, "profile_ref": "profile.json",
            "measurement": measurement,
        })
        assert _state_init(argparse.Namespace(
            run_id="cli-run", owner="direct_owner", measurement=measurement_path, stage=stage_path,
            state=cli_state, canonical=cli_canonical)) == 0
        assert _write_worker_result(argparse.Namespace(
            out=cli_worker, run_id="cli-run", generation=1, role="deep", stage="resweep",
            status="completed", measurement=measurement_path,
            work_kind=["climb", "resweep_request"], request_ref=[])) == 0
        assert _write_worker_result(argparse.Namespace(
            out=cli_plain_worker, run_id="cli-run", generation=1, role="direction", stage="converge",
            status="completed", measurement=measurement_path,
            work_kind=["climb"], request_ref=[])) == 0
        assert _request_branch(argparse.Namespace(
            out=cli_branch_request, request_id="cli-branch", run_id="cli-run", generation=1,
            candidate_ref=["census.json#candidate-a", "census.json#candidate-b"],
            evidence_ref="profile.json")) == 0
        assert _request_resweep(argparse.Namespace(
            out=cli_request, request_id="cli-resweep", parent_result=cli_worker,
            evidence_ref="profile.json", requester_role="deep", run_id=None,
            generation=None)) == 0
        assert _request_resweep(argparse.Namespace(
            out=cli_captain_request, request_id="cli-captain-resweep", parent_result=cli_plain_worker,
            evidence_ref="profile.json", requester_role="captain", run_id="cli-run",
            generation=1)) == 0
        assert _load(cli_captain_request)["requester_role"] == "captain"
        assert _request_resweep(argparse.Namespace(
            out=cli_direct_owner_request, request_id="cli-direct-owner-resweep",
            parent_result=cli_plain_worker, evidence_ref="profile.json",
            requester_role="direct_owner", run_id="cli-run", generation=1)) == 0
        assert _request_structure_suspect(argparse.Namespace(
            out=cli_suspect, candidate_id="candidate-a", source_ref=cli_worker,
            evidence_ref="profile.json", state="untried", independent=True)) == 0
        assert _resolve_resweep(argparse.Namespace(
            out=cli_result, request=cli_direct_owner_request, measurement=measurement_path,
            owner="direct_owner")) == 0
        assert not validate_resweep_result(_load(cli_result))
        assert _request_sweep(argparse.Namespace(
            out=cli_sweep_request, request_id="cli-sweep", run_id="cli-run", generation=1,
            purpose="handoff", requester_role="arm", source_ref="profile.json",
            context=cli_sweep_context, parent_ref=cli_worker, attempt_id="attempt-1",
            lineage_id="lineage-1", root_request_id=None, work_kind="sweep",
            change_scope="config", axis_kind="config")) == 0
        assert _resolve_sweep(argparse.Namespace(
            out=cli_sweep_result, request=cli_sweep_request, measurement=measurement_path,
            owner="direct_owner")) == 0
        assert not validate_sweep_pair(_load(cli_sweep_request), _load(cli_sweep_result))
        assert _request_sweep(argparse.Namespace(
            out=cli_post_sweep_request, request_id="cli-post-merge", run_id="cli-run", generation=1,
            purpose="post_merge", requester_role="captain", source_ref="merge.json",
            context=cli_sweep_context, parent_ref=cli_plain_worker, attempt_id="attempt-2",
            lineage_id="lineage-2", root_request_id=None, work_kind="sweep",
            change_scope="body", axis_kind="config")) == 0
        assert _resolve_sweep(argparse.Namespace(
            out=cli_post_sweep_result, request=cli_post_sweep_request,
            measurement=measurement_path, owner="direct_owner")) == 0
        for arm_id, arm_path in (("arm-a", cli_arm_a), ("arm-b", cli_arm_b)):
            assert _write_arm_result(argparse.Namespace(
                out=arm_path, arm_id=arm_id, lever="split_k", direction_class="structural",
                iters_used=3, completion="complete", verdict="supported", gate_stage="b",
                measurement=measurement_path)) == 0
        assert _branch_converge(argparse.Namespace(
            out=cli_converge, owner="direct_owner", planned="arm-a,arm-b",
            arm_result=[cli_arm_a, cli_arm_b], measurement=measurement_path, winner="arm-a",
            sweep_result=cli_post_sweep_result, resweep_result=None,
            decision_evidence="direction_evidence.json")) == 0
        assert _load(cli_converge)["run_id"] == "cli-run"
        assert _attach_branch_converge(argparse.Namespace(
            canonical=cli_canonical, branch_converge=cli_converge)) == 0
        assert _write_final_report(argparse.Namespace(
            out=cli_report, owner="direct_owner", run_id="cli-run", generation=1,
            stage="finalization", outcome="win", arbitration="accept", status="closed",
            measurement=measurement_path, canonical_ref=cli_canonical, known_wrong=None,
            extra_fields=cli_report_fields, champion_ref="champion.json",
            champion_gate_ref="champion_gate.json", anchor=cli_anchor,
            close_audit_ref="close_audit.json", scope=cli_scope, deferred=cli_deferred,
            deep_arbitration=cli_deep)) == 0
        assert _load(cli_report)["operator_summary"].startswith("free-text")
        # THE WRITER OWNS ITS OWN RECEIPT, and writes it as a sidecar against the finished bytes.
        assert os.path.isfile(cli_report + ".receipt.json")
        # A caller-supplied receipt covering that same output is refused, because embedding it puts
        # the digest inside the bytes it hashes and the write invalidates it permanently. Three of
        # four results on one campaign were held below clean acceptance by exactly this, and
        # it was reported upstream as "self-referential receipts can never validate" -- true of the
        # embedded route, false of the sidecar, which is why the refusal names the sidecar.
        self_ref = os.path.join(root, "self.receipt.json")
        write_producer_receipt(cli_report, "final_report", receipt_path=self_ref)
        for supplied in ([self_ref], [_load(self_ref)]):
            try:
                _write_final_report(argparse.Namespace(
                    out=cli_report, owner="direct_owner", run_id="cli-run", generation=1,
                    stage="finalization", outcome="win", arbitration="accept", status="closed",
                    measurement=measurement_path, canonical_ref=cli_canonical, known_wrong=None,
                    extra_fields=cli_report_fields, champion_ref="champion.json",
                    champion_gate_ref="champion_gate.json", anchor=cli_anchor,
                    close_audit_ref="close_audit.json", scope=cli_scope, deferred=cli_deferred,
                    deep_arbitration=cli_deep, producer_receipt=supplied))
                raise AssertionError("a receipt covering this command's own output was accepted")
            except ValueError as exc:
                assert "own output" in str(exc), exc
        assert _record_final(argparse.Namespace(
            canonical=cli_canonical, report=cli_report)) == 0
        assert _load(cli_canonical)["final"]["outcome"] == "win"
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("[canonical_record] SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=False)
    for command in ("normalize", "project"):
        projection_parser = sub.add_parser(
            command, help="read historical/current final report shapes and emit portable v3")
        projection_parser.add_argument("--report", required=True)
        projection_parser.add_argument("--out", required=True)
        projection_parser.add_argument("--work-root")
        projection_parser.add_argument("--strict", action="store_true",
                                       help="fail on unsupported schema majors")
    receipt_parser = sub.add_parser(
        "write-producer-receipt", help="write process-time or explicitly late provenance sidecar")
    receipt_parser.add_argument("--artifact", required=True)
    receipt_parser.add_argument("--artifact-role", choices=PRODUCER_ARTIFACT_ROLES, required=True)
    receipt_parser.add_argument("--out")
    receipt_parser.add_argument("--executable")
    receipt_parser.add_argument("--validator")
    receipt_parser.add_argument("--timing", choices=("process_time", "late_reconstruction"),
                                default="process_time")
    receipt_parser.add_argument("--fallback-reason")
    receipt_parser.add_argument("--input-schema")
    receipt_parser.add_argument("--output-schema")
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--kind", required=True, choices=(
        "measurement", "stage-result", "run-decision", "arm-result", "worker-result",
        "branch-request", "structure-suspect", "resweep-request", "resweep-result",
        "sweep-request", "sweep-result", "branch-converge", "final-report",
        "producer-receipt"))
    validate_parser.add_argument("--file", required=True)
    validate_parser.add_argument("--captain", action="store_true")
    validate_parser.add_argument("--strict", action="store_true")
    init_parser = sub.add_parser("init-run", help="atomically create a canonical decision and run state")
    init_parser.add_argument("--run-id", required=True)
    init_parser.add_argument("--owner", choices=FINAL_REPORT_OWNERS, required=True)
    init_parser.add_argument("--measurement", required=True)
    init_parser.add_argument("--stage", required=True)
    init_parser.add_argument("--state", required=True)
    init_parser.add_argument("--canonical", required=True)
    stage_parser = sub.add_parser("record-stage", help="append or replace one StageResult")
    stage_parser.add_argument("--canonical", required=True)
    stage_parser.add_argument("--stage", required=True)
    worker_parser = sub.add_parser("write-worker-result",
                                   help="write a direction/deep worker handoff for a captain")
    worker_parser.add_argument("--out", required=True)
    worker_parser.add_argument("--run-id", required=True)
    worker_parser.add_argument("--generation", type=int, required=True)
    worker_parser.add_argument("--role", choices=WORKER_ROLES, required=True)
    worker_parser.add_argument("--stage", choices=PHASES, required=True)
    worker_parser.add_argument("--status", choices=WORKER_STATUSES, required=True)
    worker_parser.add_argument("--measurement",
                               help="required for status=completed; omit for a blocked/deferred handoff")
    worker_parser.add_argument("--work-kind", action="append", default=[])
    worker_parser.add_argument("--request-ref", action="append", default=[])
    request_branch_parser = sub.add_parser("request-branch",
                                           help="write a direction request for captain-owned fan-out")
    request_branch_parser.add_argument("--out", required=True)
    request_branch_parser.add_argument("--request-id", required=True)
    request_branch_parser.add_argument("--run-id", required=True)
    request_branch_parser.add_argument("--generation", type=int, required=True)
    request_branch_parser.add_argument("--candidate-ref", action="append", required=True)
    request_branch_parser.add_argument("--evidence-ref", required=True)
    suspect_parser = sub.add_parser("request-structure-suspect",
                                    help="write a deep-worker request for later structural review")
    suspect_parser.add_argument("--out", required=True)
    suspect_parser.add_argument("--candidate-id", required=True)
    suspect_parser.add_argument("--source-ref", required=True)
    suspect_parser.add_argument("--evidence-ref", required=True)
    suspect_parser.add_argument("--state", choices=("untried", "deferred"), default="untried")
    suspect_parser.add_argument("--independent", action="store_true")
    request_resweep_parser = sub.add_parser("request-resweep",
                                            help="write a request only; a separate captain/direct-owner record "
                                                 "owns execution")
    request_resweep_parser.add_argument("--out", required=True)
    request_resweep_parser.add_argument("--request-id", required=True)
    request_resweep_parser.add_argument("--parent-result", required=True)
    request_resweep_parser.add_argument("--evidence-ref", required=True)
    request_resweep_parser.add_argument("--requester-role", choices=RESWEEP_REQUESTERS, default="deep",
                                        help="deep may only request from its own worker result; captain and "
                                             "direct_owner may request the plain post-merge re-sweep")
    request_resweep_parser.add_argument("--run-id",
                                        help="required for a captain/direct_owner request")
    request_resweep_parser.add_argument("--generation", type=int,
                                        help="required for a captain/direct_owner request")
    resolve_resweep_parser = sub.add_parser("resolve-resweep",
                                            help="record a captain/direct-owner resweep result")
    resolve_resweep_parser.add_argument("--out", required=True)
    resolve_resweep_parser.add_argument("--request", required=True)
    resolve_resweep_parser.add_argument("--measurement", required=True)
    resolve_resweep_parser.add_argument("--owner", choices=FINAL_REPORT_OWNERS, required=True)
    request_sweep_parser = sub.add_parser(
        "request-sweep", help="write one typed, immutable sweep request; arms request but never execute")
    request_sweep_parser.add_argument("--out", required=True)
    request_sweep_parser.add_argument("--request-id", required=True)
    request_sweep_parser.add_argument("--run-id", required=True)
    request_sweep_parser.add_argument("--generation", type=int, required=True)
    request_sweep_parser.add_argument("--purpose", choices=SWEEP_PURPOSES, required=True)
    request_sweep_parser.add_argument("--requester-role", choices=SWEEP_REQUESTERS, required=True)
    request_sweep_parser.add_argument("--source-ref", required=True)
    request_sweep_parser.add_argument("--context", required=True,
                                      help="path to a non-empty JSON context object")
    request_sweep_parser.add_argument("--parent-ref", required=True)
    request_sweep_parser.add_argument("--attempt-id", required=True)
    request_sweep_parser.add_argument("--lineage-id", required=True)
    request_sweep_parser.add_argument("--root-request-id")
    request_sweep_parser.add_argument("--work-kind", choices=SWEEP_WORK_KINDS, required=True)
    request_sweep_parser.add_argument("--change-scope", choices=CHANGE_SCOPES, required=True)
    request_sweep_parser.add_argument("--axis-kind", choices=AXIS_KINDS, required=True)
    resolve_sweep_parser = sub.add_parser(
        "resolve-sweep", help="record a triton-sweep/captain/direct-owner result for a typed sweep request")
    resolve_sweep_parser.add_argument("--out", required=True)
    resolve_sweep_parser.add_argument("--request", required=True)
    resolve_sweep_parser.add_argument("--measurement", required=True)
    resolve_sweep_parser.add_argument("--owner", choices=SWEEP_EXECUTORS, required=True)
    arm_parser = sub.add_parser("write-arm-result",
                                help="write the only accepted ArmResult schema")
    arm_parser.add_argument("--out", required=True)
    arm_parser.add_argument("--arm-id", required=True)
    arm_parser.add_argument("--lever", required=True)
    arm_parser.add_argument("--direction-class", choices=ARM_DIRECTION_CLASSES, required=True)
    arm_parser.add_argument("--iters-used", type=int, required=True)
    arm_parser.add_argument("--completion", choices=ARM_COMPLETION, required=True)
    arm_parser.add_argument("--verdict", choices=ARM_VERDICTS, required=True)
    arm_parser.add_argument("--gate-stage", choices=ARM_GATE_STAGES, required=True)
    arm_parser.add_argument("--measurement")
    arm_parser.add_argument("--sweep-result",
                            help="typed purpose=arm_local result used for this arm's own sweep")
    converge_parser = sub.add_parser("branch-converge",
                                     help="converge canonical ArmResults and a captain-owned resweep")
    converge_parser.add_argument("--out", required=True)
    converge_parser.add_argument("--owner", choices=BRANCH_CONVERGE_OWNERS, required=True)
    converge_parser.add_argument("--decision-evidence", required=True,
                                 help="direction-owned evidence supporting winner selection")
    converge_parser.add_argument("--planned", required=True)
    converge_parser.add_argument("--arm-result", action="append", required=True)
    converge_parser.add_argument("--measurement", required=True)
    converge_parser.add_argument("--winner", required=True,
                                 help="eligible arm id, or literal 'none'")
    converge_parser.add_argument("--sweep-result",
                                 help="typed purpose=post_merge result (preferred)")
    converge_parser.add_argument("--resweep-result")
    attach_parser = sub.add_parser("attach-branch-converge",
                                   help="attach a captain convergence record to run decision")
    attach_parser.add_argument("--canonical", required=True)
    attach_parser.add_argument("--branch-converge", required=True)
    report_parser = sub.add_parser("write-final-report",
                                   help="write final report from a captain or direct owner")
    report_parser.add_argument("--out", required=True)
    report_parser.add_argument("--owner", choices=FINAL_REPORT_OWNERS, required=True)
    report_parser.add_argument("--lifecycle-profile", choices=LIFECYCLE_PROFILES,
                               help="defaults to direct-owner for direct owners, front-end otherwise")
    report_parser.add_argument("--run-id", required=True)
    report_parser.add_argument("--generation", type=int, required=True)
    report_parser.add_argument("--stage", choices=PHASES, required=True)
    report_parser.add_argument("--outcome", choices=OUTCOMES, required=True)
    report_parser.add_argument("--arbitration", choices=ARBITRATIONS, required=True)
    report_parser.add_argument("--status", choices=RUN_STATUSES, required=True)
    report_parser.add_argument("--measurement", required=True)
    report_parser.add_argument("--canonical-ref", required=True)
    report_parser.add_argument("--champion-ref", required=True)
    report_parser.add_argument("--champion-gate-ref", required=True)
    report_parser.add_argument("--anchor", required=True,
                               help="JSON object/file: {type, ref}")
    report_parser.add_argument("--close-audit-ref", required=True)
    report_parser.add_argument("--scope", required=True,
                               help="JSON object/file: {authorization}")
    report_parser.add_argument("--deferred", required=True,
                               help="JSON array/file; [] is valid")
    report_parser.add_argument("--deep-arbitration", required=True,
                               help="JSON object/file: {state, champion_ms, result_ref?}")
    report_parser.add_argument("--known-wrong")
    report_parser.add_argument("--extra-fields",
                               help="JSON object of caller-owned/free-text report fields; canonical base "
                                    "keys are protected")
    report_parser.add_argument("--producer-receipt", action="append", default=[],
                               help="process-time receipt to carry into report provenance")
    record_final_parser = sub.add_parser("record-final",
                                         help="copy the report's domains into the canonical decision")
    record_final_parser.add_argument("--canonical", required=True)
    record_final_parser.add_argument("--report", required=True)
    state_open_parser = sub.add_parser("state-open", help="open a state obligation through this CLI")
    state_open_parser.add_argument("--state", required=True)
    state_open_parser.add_argument("--kind", required=True)
    state_open_parser.add_argument("--reason", required=True)
    state_open_parser.add_argument("--obligation-id")
    state_done_parser = sub.add_parser("state-done", help="complete a state obligation through this CLI")
    state_done_parser.add_argument("--state", required=True)
    state_done_parser.add_argument("--obligation", required=True)
    state_done_parser.add_argument("--generation", type=int, required=True)
    state_done_parser.add_argument("--evidence-ref", required=True)
    state_begin_parser = sub.add_parser("state-begin-finalization",
                                        help="enter finalization after obligations complete")
    state_begin_parser.add_argument("--state", required=True)
    state_begin_parser.add_argument("--generation", type=int, required=True)
    sync_parser = sub.add_parser("sync-run-state",
                                 help="copy the current open run-state generation into canonical decision")
    sync_parser.add_argument("--state", required=True)
    sync_parser.add_argument("--canonical", required=True)
    schema_parser = sub.add_parser("schema")
    schema_parser.add_argument("--kind", choices=(
        "measurement", "stage-result", "run-decision", "arm-result", "worker-result",
        "branch-request", "structure-suspect", "resweep-request", "resweep-result",
        "sweep-request", "sweep-result", "branch-converge", "final-report",
        "producer-receipt"))
    reconcile_parser = sub.add_parser("reconcile")
    reconcile_parser.add_argument("--canonical", required=True)
    reconcile_parser.add_argument("--report", required=True)
    reconcile_parser.add_argument("--state", required=True)
    reconcile_parser.add_argument("--canonical-ref")
    reconcile_parser.add_argument("--strict", action="store_true")
    finalize_parser = sub.add_parser("finalize")
    finalize_parser.add_argument("--canonical", required=True)
    finalize_parser.add_argument("--report", required=True)
    finalize_parser.add_argument("--state", required=True)
    finalize_parser.add_argument("--canonical-ref")
    finalize_parser.add_argument("--out", help="optional atomically-written canonical copy")
    sub.add_parser("selftest")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest or args.command == "selftest":
        return _selftest()
    if args.command in ("normalize", "project"):
        return _project_command(args)
    if args.command == "write-producer-receipt":
        return _write_receipt_command(args)
    if args.command == "schema":
        print(json.dumps(JSON_SCHEMAS if args.kind is None
                         else JSON_SCHEMAS[args.kind], indent=2))
        return 0
    if args.command == "validate":
        findings = validate(args.kind, _load(args.file), captain=args.captain)
        _print_result(args.kind, findings)
        return 1 if args.strict and findings else 0
    if args.command == "init-run":
        return _state_init(args)
    if args.command == "record-stage":
        return _record_stage(args)
    if args.command == "write-worker-result":
        return _write_worker_result(args)
    if args.command == "request-branch":
        return _request_branch(args)
    if args.command == "request-structure-suspect":
        return _request_structure_suspect(args)
    if args.command == "request-resweep":
        return _request_resweep(args)
    if args.command == "resolve-resweep":
        return _resolve_resweep(args)
    if args.command == "request-sweep":
        return _request_sweep(args)
    if args.command == "resolve-sweep":
        return _resolve_sweep(args)
    if args.command == "write-arm-result":
        return _write_arm_result(args)
    if args.command == "branch-converge":
        return _branch_converge(args)
    if args.command == "attach-branch-converge":
        return _attach_branch_converge(args)
    if args.command == "write-final-report":
        return _write_final_report(args)
    if args.command == "record-final":
        return _record_final(args)
    if args.command == "state-open":
        return _state_open(args)
    if args.command == "state-done":
        return _state_done(args)
    if args.command == "state-begin-finalization":
        return _state_begin_finalization(args)
    if args.command == "sync-run-state":
        return _sync_run_state(args)
    if args.command == "reconcile":
        canonical, canonical_error = _load_finalization_input(args.canonical, "canonical")
        report, report_error = _load_finalization_input(args.report, "report")
        state, state_error = _load_finalization_input(args.state, "state")
        findings = [x for x in (canonical_error, report_error, state_error) if x]
        if not findings:
            findings = reconcile(canonical, report, state, args.canonical_ref)
        _print_result("reconcile", findings)
        return 1 if args.strict and findings else 0
    if args.command == "finalize":
        return _finalize(args)
    parser.error("a subcommand is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
