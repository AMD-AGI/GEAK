# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind normalized measurements to their sealed accepted-source observations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import statistics
from pathlib import Path
from typing import Any

try:
    from interface.source_materialization import (
        SourceMaterializationError,
        read_source_request,
    )
except ModuleNotFoundError:
    from source_materialization import SourceMaterializationError, read_source_request


class SourceMeasurementError(ValueError):
    """A selected measurement lacks matching durable source evidence."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SourceMeasurementError(reason)


def _read(path: Path) -> bytes:
    _require(path.is_absolute(), "relative_evidence_path")
    _require(path.resolve() == path, "symlink_in_measurement_evidence")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        _require(
            stat.S_ISREG(os.fstat(stream.fileno()).st_mode),
            "non_regular_measurement_evidence",
        )
        return stream.read()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        _require(key not in value, "duplicate_evidence_key")
        value[key] = item
    return value


def _object(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=_pairs)
    _require(isinstance(value, dict), "invalid_measurement_evidence")
    return value


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hashed(path: Path, expected: Any) -> bytes:
    _require(
        isinstance(expected, str)
        and re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
        "invalid_evidence_digest",
    )
    data = _read(path)
    _require(_digest(data) == expected, "measurement_evidence_digest_mismatch")
    return data


def _positive(value: Any) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(value) and value > 0,
        "invalid_measurement_value",
    )
    return float(value)


def _same_value(actual: Any, expected: Any) -> None:
    _require(
        math.isclose(
            _positive(actual), _positive(expected), rel_tol=1e-9, abs_tol=1e-9
        ),
        "published_measurement_mismatch",
    )


def _identity(value: Any) -> dict[str, int]:
    _require(
        isinstance(value, dict) and set(value) == {"pid", "pgid", "start_ticks"},
        "invalid_server_identity",
    )
    _require(
        all(type(item) is int and item > 0 for item in value.values()),
        "invalid_server_identity",
    )
    return value


def _bindings(
    row: dict[str, Any], seal: dict[str, Any], request_sha: str, manifest_sha: str
) -> None:
    _require(
        row.get("status") == "verified"
        and row.get("request_sha256") == request_sha
        and row.get("manifest_sha256") == manifest_sha
        and row.get("launch_nonce") == seal.get("launch_nonce"),
        "source_measurement_binding_mismatch",
    )


def _module_rows(rows: Any, source: Any, overlay_roots: Any) -> set[str]:
    _require(
        isinstance(rows, list)
        and isinstance(overlay_roots, list)
        and all(
            isinstance(root, str) and Path(root).is_absolute() for root in overlay_roots
        ),
        "invalid_process_module_evidence",
    )
    hashes = {
        str(source.bundle_root / item["path"]): item["sha256"]
        for item in source.manifest["files"]
    }
    names: set[str] = set()
    for row in rows:
        _require(
            isinstance(row, dict)
            and isinstance(row.get("name"), str)
            and row["name"] not in names
            and isinstance(row.get("origin"), str)
            and row["origin"].endswith(".py"),
            "invalid_process_module_evidence",
        )
        name, origin = row["name"], Path(row["origin"])
        names.add(name)
        _require(
            origin.is_absolute() and origin.resolve() == origin,
            "invalid_process_module_origin",
        )
        if row.get("source") == "accepted_source":
            _require(
                str(origin) in hashes and hashes[str(origin)] == row.get("sha256"),
                "accepted_module_digest_mismatch",
            )
            prefixes = [
                root
                for root in source.pythonpath_prefixes
                if origin.is_relative_to(root)
            ]
            _require(len(prefixes) == 1, "accepted_module_origin_mismatch")
            parts = list(origin.relative_to(prefixes[0]).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            _require(name == ".".join(parts), "accepted_module_name_mismatch")
        else:
            _require(
                row.get("source") == "authored_overlay"
                and any(origin.is_relative_to(root) for root in overlay_roots),
                "unknown_module_source",
            )
            _hashed(origin, row.get("sha256"))
    return names


def _gate(
    directory: Path,
    phase: str,
    reference: Any,
    seal: dict[str, Any],
    source: Any,
    request_sha: str,
    minimum_observed_at_ns: int = 0,
) -> dict[str, Any]:
    _require(
        isinstance(reference, dict)
        and set(reference) == {"path", "sha256"}
        and reference["path"] == f"gate-{phase}.json",
        "invalid_measurement_gate_reference",
    )
    row = _object(_hashed(directory / reference["path"], reference["sha256"]))
    _require(
        row.get("schema") == "geak.source_runtime.gate.v1"
        and row.get("phase") == phase,
        "invalid_measurement_gate",
    )
    _bindings(row, seal, request_sha, source.manifest_sha256)
    owner = _identity(row.get("server_identity"))
    _require(
        owner == seal.get("server_identity") and owner["pid"] == owner["pgid"] > 1,
        "measurement_server_identity_mismatch",
    )
    _require(
        isinstance(row.get("boot_id"), str)
        and bool(row["boot_id"])
        and isinstance(row.get("challenge_id"), str)
        and bool(row["challenge_id"])
        and type(row.get("observed_at_ns")) is int
        and row["observed_at_ns"] > 0
        and row.get("base_url") == seal.get("base_url"),
        "invalid_measurement_gate_identity",
    )
    processes = row.get("processes")
    listeners = row.get("listener_pids")
    if not (
        isinstance(processes, list)
        and bool(processes)
        and isinstance(listeners, list)
        and bool(listeners)
        and all(type(pid) is int and pid > 1 for pid in listeners)
    ):
        raise SourceMeasurementError("missing_serving_process_evidence")
    seen: set[int] = set()
    touched = {item["name"] for item in source.manifest["modules"]}
    deleted = set(source.manifest["deleted_modules"])
    for process in processes:
        _require(
            isinstance(process, dict)
            and set(process) == {"pid", "pgid", "start_ticks", "receipt", "sha256"},
            "invalid_process_evidence",
        )
        identity = _identity(
            {key: process[key] for key in ("pid", "pgid", "start_ticks")}
        )
        _require(
            identity["pgid"] == owner["pgid"] and identity["pid"] not in seen,
            "invalid_process_membership",
        )
        seen.add(identity["pid"])
        filename = f"process-{identity['pid']}-{identity['start_ticks']}-{row['challenge_id']}.json"
        _require(
            process["receipt"] == filename and Path(filename).name == filename,
            "invalid_process_receipt_path",
        )
        receipt = _object(_hashed(directory / filename, process["sha256"]))
        _bindings(receipt, seal, request_sha, source.manifest_sha256)
        _require(
            receipt.get("schema") == "geak.source_runtime.process.v1"
            and all(receipt.get(key) == value for key, value in identity.items())
            and receipt.get("boot_id") == row["boot_id"]
            and receipt.get("challenge_id") == row["challenge_id"]
            and type(receipt.get("observed_at_ns")) is int
            and minimum_observed_at_ns
            < receipt["observed_at_ns"]
            <= row["observed_at_ns"]
            and receipt.get("accepted_roots") == list(source.pythonpath_prefixes)
            and receipt.get("guard") == "owned_module_specs_unchanged",
            "invalid_process_receipt",
        )
        resolved = receipt.get("resolved_modules")
        absent = receipt.get("deleted_modules_absent")
        loaded = receipt.get("loaded_modules")
        resolved_names = _module_rows(resolved, source, receipt.get("overlay_roots"))
        loaded_names = _module_rows(loaded, source, receipt.get("overlay_roots"))
        _require(
            touched.issubset(resolved_names)
            and not deleted.intersection(resolved_names | loaded_names)
            and isinstance(absent, list)
            and len(absent) == len(deleted)
            and set(absent) == deleted,
            "incomplete_process_source_coverage",
        )
    _require(
        set(listeners).issubset(seen) and len(set(listeners)) == len(listeners),
        "unobserved_listener_process",
    )
    return row


def _leaf(
    directory: Path, source: Any, request_sha: str
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    runtime = directory / "source_runtime"
    raw_seal = _read(runtime / "measurement.json")
    seal = _object(raw_seal)
    _require(
        seal.get("schema") == "geak.source_runtime.measurement.v1"
        and isinstance(seal.get("launch_nonce"), str)
        and bool(seal["launch_nonce"])
        and isinstance(seal.get("base_url"), str)
        and bool(seal["base_url"]),
        "invalid_measurement_seal",
    )
    _bindings(seal, seal, request_sha, source.manifest_sha256)
    _identity(seal.get("server_identity"))
    gates = seal.get("gates")
    if not isinstance(gates, dict) or set(gates) != {"ready", "finished"}:
        raise SourceMeasurementError("incomplete_measurement_gates")
    ready = _gate(runtime, "ready", gates["ready"], seal, source, request_sha)
    finished = _gate(
        runtime,
        "finished",
        gates["finished"],
        seal,
        source,
        request_sha,
        ready["observed_at_ns"],
    )
    _require(
        ready["boot_id"] == finished["boot_id"]
        and ready["observed_at_ns"] < finished["observed_at_ns"]
        and ready["challenge_id"] != finished["challenge_id"],
        "measurement_gate_order_mismatch",
    )
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "bench_runs.jsonl",
        "bench_summary.json",
    }:
        raise SourceMeasurementError("incomplete_measurement_artifacts")
    raw_summary = _hashed(
        directory / "bench_summary.json", artifacts["bench_summary.json"]
    )
    runs = _hashed(directory / "bench_runs.jsonl", artifacts["bench_runs.jsonl"])
    summary = _object(raw_summary)
    _positive(summary.get("throughput_tok_s_median"))
    _require(
        bool(runs.strip()) and type(summary.get("runs")) is int and summary["runs"] > 0,
        "empty_sealed_measurement",
    )
    return (
        summary,
        runs,
        {
            "summary_path": str(directory / "bench_summary.json"),
            "summary_sha256": artifacts["bench_summary.json"],
            "seal_sha256": _digest(raw_seal),
            "launch_nonce": seal["launch_nonce"],
            "server_identity": seal["server_identity"],
        },
    )


def _measurement(directory: Path, source: Any, request_sha: str) -> dict[str, Any]:
    raw_summary = _read(directory / "bench_summary.json")
    summary = _object(raw_summary)
    if summary.get("measurement_mode") != "isolated_server":
        checked, _, evidence = _leaf(directory, source, request_sha)
        _require(checked == summary, "measurement_changed_during_verification")
        return {
            **evidence,
            "throughput_tok_s": _positive(checked.get("throughput_tok_s_median")),
        }
    replicas = summary.get("replicas")
    if not isinstance(replicas, list) or not replicas:
        raise SourceMeasurementError("missing_selected_replicas")
    selections, values, runs = [], [], []
    indices: list[int] = []
    for replica in replicas:
        _require(
            isinstance(replica, dict)
            and set(replica) == {"replica", "attempt", "throughput_tok_s"},
            "invalid_replica_selection",
        )
        index, attempt = replica["replica"], replica["attempt"]
        _require(
            type(index) is int
            and index > 0
            and type(attempt) is int
            and attempt in (1, 2),
            "invalid_replica_selection",
        )
        indices.append(index)
        selected = directory / f"replica_{index:03d}"
        _require(
            _read(selected / "selected_attempt").strip() == str(attempt).encode(),
            "replica_selection_mismatch",
        )
        leaf_dir = selected / f"attempt_{attempt}"
        leaf, leaf_runs, evidence = _leaf(leaf_dir, source, request_sha)
        _require(
            _read(selected / "selected_summary.json")
            == _read(leaf_dir / "bench_summary.json"),
            "selected_summary_mismatch",
        )
        _require(leaf.get("runs") == 1, "invalid_replica_run_count")
        _same_value(leaf.get("throughput_tok_s_median"), replica["throughput_tok_s"])
        values.append(_positive(replica["throughput_tok_s"]))
        runs.append(leaf_runs)
        selections.append({"replica": index, "attempt": attempt, **evidence})
    _require(indices == sorted(set(indices)), "duplicate_or_unordered_replicas")
    _require(
        len({row["launch_nonce"] for row in selections}) == len(selections),
        "duplicate_replica_launch",
    )
    _require(
        type(summary.get("requested_replicas")) is int
        and type(summary.get("successful_replicas")) is int
        and type(summary.get("runs")) is int
        and indices == list(range(1, summary["requested_replicas"] + 1))
        and summary.get("successful_replicas") == len(indices) == summary.get("runs")
        and summary.get("usable_for_acceptance") is True
        and summary.get("status") == "complete",
        "incomplete_isolated_measurement",
    )
    _require(
        _read(directory / "bench_runs.jsonl") == b"".join(runs),
        "aggregate_runs_mismatch",
    )
    median = round(statistics.median(values), 3)
    _same_value(summary.get("throughput_tok_s_median"), median)
    _require(
        summary.get("all_throughput") == values, "aggregate_replica_values_mismatch"
    )
    _require(
        _read(directory / "bench_summary.json") == raw_summary,
        "measurement_changed_during_verification",
    )
    return {
        "summary_path": str(directory / "bench_summary.json"),
        "summary_sha256": _digest(raw_summary),
        "throughput_tok_s": median,
        "selected_replicas": selections,
    }


def verify_source_measurement(
    request_path: str | Path,
    summary_path: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_required_layer_ids: list[str] | tuple[str, ...],
    expected_throughput_tok_s: float,
) -> dict[str, Any]:
    """Verify one explicitly selected measurement; never search for alternatives."""
    request = Path(request_path)
    request_raw = _read(request)
    source = read_source_request(str(request))
    _require(
        source.manifest_sha256 == expected_manifest_sha256
        and source.required_layer_ids == tuple(expected_required_layer_ids),
        "handoff_source_identity_mismatch",
    )
    path = Path(summary_path)
    _require(path.name == "bench_summary.json", "unsupported_summary_path")
    evidence = _measurement(path.parent, source, _digest(request_raw))
    _same_value(evidence["throughput_tok_s"], expected_throughput_tok_s)
    _require(
        _read(request) == request_raw, "source_request_changed_during_verification"
    )
    return evidence


def verify_normalized_source_measurements(
    request_path: str | Path,
    *,
    eval_dir: Path,
    result_source: str,
    baseline_basis_source: str,
    setup_tput: float,
    baseline_tput: float,
    final_tput: float,
    expected_manifest_sha256: str,
    expected_required_layer_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Assess fixed normalized measurement provenance without changing results.

    Recovery schemas without exact source-sealed summary mappings remain
    unavailable. A numerically matching alternative does not prove provenance.
    """
    evidence: dict[str, Any] = {
        "status": "unavailable",
        "reason": "unverified_source_measurements",
    }
    try:
        _require(
            result_source in {"workflow_return", "disk_director_validation"}
            and baseline_basis_source == "validation_base_bench_summary",
            "unsupported_measurement_provenance",
        )
        selected = {
            "setup": ("baseline", setup_tput),
            "baseline": ("validation/base", baseline_tput),
            "final": ("validation/final", final_tput),
        }
        measurements = {}
        for role, (relative, value) in selected.items():
            measurements[role] = verify_source_measurement(
                request_path,
                eval_dir / relative / "bench_summary.json",
                expected_manifest_sha256=expected_manifest_sha256,
                expected_required_layer_ids=expected_required_layer_ids,
                expected_throughput_tok_s=value,
            )
        evidence = {
            "status": "verified",
            "reason": "sealed_selected_measurements",
            "request_sha256": _digest(_read(Path(request_path))),
            "manifest_sha256": expected_manifest_sha256,
            "required_layer_ids": list(expected_required_layer_ids),
            "measurements": measurements,
        }
    except (SourceMeasurementError, SourceMaterializationError) as exc:
        evidence["reason"] = str(exc)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        evidence["reason"] = "unreadable_source_measurement_evidence"
        evidence["error_type"] = type(exc).__name__
    return evidence
