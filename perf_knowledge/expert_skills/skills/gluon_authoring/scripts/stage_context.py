#!/usr/bin/env python3
"""Build a bounded, hash-addressed context package for one optimization stage.

The package is a cooperative context-I/O boundary: it emits references and fixed
metadata, never a ledger, profile dump, source file, or decision transcript.  It
does not and cannot prevent a host from reading files directly.

Usage:
  stage_context.py build --stage STAGE [--canonical FILE] [--run-state FILE]
                         [--sweep FILE] [--branch FILE] [--converge FILE]
                         [--champion FILE] [--recall FILE] [--out FILE]
  stage_context.py --describe json|--format=json
  stage_context.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import context_contracts as contracts
import round_record


SCHEMA = "kernel_opt.stage_context/1"
MAX_REFS = 24
MAX_ITEMS = 16
MAX_MEASUREMENTS = 4
MAX_RECALL_PER_BUCKET = 2
MAX_REF_CHARS = 256
MAX_ACTION_CHARS = 96
_KINDS = ("canonical", "run_state", "sweep", "branch", "converge", "champion", "recall")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _clip(value: Any, limit: int = MAX_REF_CHARS) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value[:limit] if value else None


def _bounded_strings(values: Any, limit: int, width: int) -> list[str]:
    out: list[str] = []
    for value in values or []:
        text = _clip(value, width)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _file_hash(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _load_reference(kind: str, path: str) -> tuple[dict, dict | None]:
    """Return a fixed ref receipt and, only for JSON objects, its metadata source."""
    ref = _clip(path) or ""
    receipt = {"kind": kind, "ref": ref, "status": "missing", "sha256": None}
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return receipt, None
    except (OSError, json.JSONDecodeError):
        receipt["status"] = "unreadable"
        return receipt, None
    receipt["sha256"] = _file_hash(path)
    if not isinstance(raw, dict):
        receipt["status"] = "not_object"
        return receipt, None
    receipt["status"] = "read"
    return receipt, raw


def _measurement_summary(doc: dict) -> dict | None:
    measurement = doc.get("measurement") if isinstance(doc.get("measurement"), dict) else {}
    identity = measurement.get("identity") if isinstance(measurement.get("identity"), dict) else {}
    value = measurement.get("value")
    if value is None:
        if doc.get("latency_ms") is not None:
            value, unit = doc.get("latency_ms"), "ms"
        elif doc.get("speedup_vs_comparator") is not None:
            value, unit = doc.get("speedup_vs_comparator"), "ratio"
        else:
            return None
    else:
        unit = identity.get("unit")
    summary = {
        "measurement_id": _clip(measurement.get("measurement_id"), 96),
        "value": value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
        "unit": _clip(unit, 24),
        "aggregation": _clip(identity.get("aggregation"), 48),
        "boundary": _clip(identity.get("boundary"), 48),
        "source": _clip(identity.get("source"), 48),
        "scope": _clip(identity.get("scope"), 24),
        "comparator": _clip(doc.get("comparator"), 64),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _identity(docs: list[dict]) -> dict:
    """Extract only comparison identity, never an input document's free-form fields."""
    out: dict[str, Any] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for key in ("run_id", "generation"):
            if key not in out and doc.get(key) is not None:
                out[key] = doc[key]
        measurement = doc.get("measurement") if isinstance(doc.get("measurement"), dict) else {}
        ident = measurement.get("identity") if isinstance(measurement.get("identity"), dict) else {}
        if "measurement_id" not in out and _clip(measurement.get("measurement_id"), 96):
            out["measurement_id"] = _clip(measurement.get("measurement_id"), 96)
        if "shape_set" not in out and isinstance(ident.get("shape_set"), list):
            out["shape_set"] = _bounded_strings(ident["shape_set"], MAX_ITEMS, 96)
        for key in ("aggregation", "unit", "boundary", "source", "scope"):
            if key not in out and _clip(ident.get(key), 64):
                out[key] = _clip(ident.get(key), 64)
        if "body_sha" not in out and _clip(doc.get("body_sha"), 64):
            out["body_sha"] = _clip(doc.get("body_sha"), 64)
        if "layout_sig" not in out and _clip(doc.get("layout_sig"), 64):
            out["layout_sig"] = _clip(doc.get("layout_sig"), 64)
    return out


def _obligations(state: dict | None) -> list[dict]:
    if not isinstance(state, dict):
        return []
    out = []
    for item in state.get("obligations") or []:
        if not isinstance(item, dict):
            continue
        entry = {
            "obligation_id": _clip(item.get("obligation_id"), 96),
            "kind": _clip(item.get("kind"), 64),
            "status": _clip(item.get("status"), 32),
            "generation": item.get("generation") if isinstance(item.get("generation"), int) else None,
            "evidence_ref": _clip(item.get("evidence_ref")),
        }
        if entry["obligation_id"] and entry["kind"] and entry["status"]:
            out.append({key: value for key, value in entry.items() if value is not None})
        if len(out) >= MAX_ITEMS:
            break
    return out


def _recall_summary(doc: dict | None) -> dict:
    """Accept the bounded `round_record summary` shape and degrade without guessing."""
    if not isinstance(doc, dict):
        return {"status": "unavailable", "counts": {}, "refs": []}
    source = doc.get("summary") if isinstance(doc.get("summary"), dict) else doc
    counts = source.get("counts") if isinstance(source.get("counts"), dict) else {}
    safe_counts = {
        name: int(counts.get(name, 0))
        for name in ("valid", "stale", "unfinished", "recheck")
        if isinstance(counts.get(name, 0), int) and counts.get(name, 0) >= 0
    }
    refs = _bounded_strings(source.get("refs") or source.get("artifact_refs"), MAX_ITEMS, MAX_REF_CHARS)
    return {"status": _clip(source.get("status"), 32) or "available",
            "counts": safe_counts, "refs": refs}


def build_package(stage: str, references: dict[str, list[str]], *,
                  allowed: list[str] | None = None, forbidden: list[str] | None = None) -> dict:
    if not _clip(stage, 96):
        raise ValueError("stage must be a non-empty string")
    artifacts, docs_by_kind = [], {}
    for kind in _KINDS:
        for path in references.get(kind, []):
            if len(artifacts) >= MAX_REFS:
                break
            receipt, doc = _load_reference(kind, path)
            artifacts.append(receipt)
            if doc is not None:
                docs_by_kind.setdefault(kind, []).append(doc)

    docs = [doc for kind in _KINDS for doc in docs_by_kind.get(kind, [])]
    measurements = []
    for doc in docs:
        summary = _measurement_summary(doc)
        if summary and summary not in measurements:
            measurements.append(summary)
        if len(measurements) >= MAX_MEASUREMENTS:
            break
    state = (docs_by_kind.get("run_state") or [None])[0]
    recall = (docs_by_kind.get("recall") or [None])[0]
    payload = {
        "schema": SCHEMA,
        "current_stage": _clip(stage, 96),
        "identity": _identity(docs),
        "measurement_summary": measurements,
        "obligations": _obligations(state),
        "recall_summary": _recall_summary(recall),
        "actions": {
            "allowed": _bounded_strings(allowed, MAX_ITEMS, MAX_ACTION_CHARS),
            "forbidden": _bounded_strings(forbidden, MAX_ITEMS, MAX_ACTION_CHARS),
        },
        "artifact_refs": artifacts,
    }
    payload["context_hash"] = _sha(payload)
    return payload


def _atomic_write(path: str, doc: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".stage_context.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


DEFAULT_POLICY = {
    "schema": contracts.CONTEXT_POLICY_SCHEMA,
    "policy_id": "portable-default-v1",
    "max_context_bytes": contracts.MAX_CONTEXT_BYTES,
    "max_capsule_bytes": contracts.MAX_CAPSULE_BYTES,
    "max_query_return_bytes": 16 * 1024,
    "lease": {"max_stage_events": 64, "max_returned_bytes": 128 * 1024},
    "token_telemetry": "optional",
}


def context_paths(work: Path) -> dict[str, Path]:
    """Return the fixed control-plane paths.  No experiment-tree discovery is performed."""
    kod = work / ".kod"
    return {
        "dir": kod,
        "context": kod / "stage_context.json",
        "lease": kod / "context_lease.json",
        "policy": kod / "context_policy.json",
        "capsule": kod / "resume_capsule.json",
        "recall": kod / "recall_summary.json",
        "obligations": kod / "open_obligations.json",
        "receipts": kod / "context_receipts.jsonl",
        "snapshot": kod / "snapshot",
    }


def artifact_roots(work: Path, pack: Path) -> dict[str, Path]:
    paths = context_paths(work)
    paths["snapshot"].mkdir(parents=True, exist_ok=True)
    return {"work": work.resolve(), "pack": pack.resolve(), "snapshot": paths["snapshot"].resolve()}


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _lease_identity(stage_doc: dict, state: dict) -> tuple[str, int, str, str]:
    run_id = state.get("run_id", stage_doc.get("run", "untracked"))
    generation = state.get("generation", 1)
    role = stage_doc.get("role")
    stage = stage_doc.get("current")
    if not isinstance(run_id, str) or not run_id:
        run_id = "untracked"
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        generation = 1
    if not isinstance(role, str) or not role or not isinstance(stage, str) or not stage:
        raise ValueError("stage projection must contain non-empty role and current")
    return run_id, generation, role, stage


def _named_refs(stage_doc: dict, explicit_artifacts: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    refs = stage_doc.get("artifact_refs")
    out: dict[str, Any] = {}
    if isinstance(refs, dict):
        out.update((str(name), ref) for name, ref in sorted(refs.items()))
    explicit_seen: set[str] = set()
    for name, ref in explicit_artifacts:
        if not name or name in explicit_seen:
            raise ValueError(f"explicit artifact names must be unique and non-empty: {name!r}")
        explicit_seen.add(name)
        out[name] = ref
    return list(out.items())


def _context_artifacts(
    stage_doc: dict,
    explicit_artifacts: list[tuple[str, Any]],
    roots: dict[str, Path],
) -> list[dict]:
    artifacts: list[dict] = []
    for name, source in _named_refs(stage_doc, explicit_artifacts):
        ref = contracts.normalize_artifact_ref(source, roots)
        artifacts.append({
            "name": name,
            "kind": "explicit" if (name, source) in explicit_artifacts else "registry",
            "ref": ref["uri"],
            "sha256": ref["sha256"],
            "bytes": ref["bytes"],
            "artifact_ref": ref,
        })
        if len(artifacts) >= MAX_REFS:
            break
    return artifacts


def _journal_projection(work: Path, artifacts: list[dict], roots: dict[str, Path]) -> tuple[dict, list[dict], list[str]]:
    """Derive bounded memory from declared/fixed journals without walking the work tree."""
    journal_paths: list[str] = []
    fixed = work / round_record.DEFAULT_JOURNAL
    fixed.touch(exist_ok=True)
    if fixed.is_file():
        journal_paths.append(str(fixed))
        if not any(item.get("name") == "optimization_journal" for item in artifacts):
            ref = contracts.normalize_artifact_ref(str(fixed), roots)
            artifacts.append({
                "name": "optimization_journal",
                "kind": "journal",
                "ref": ref["uri"],
                "sha256": ref["sha256"],
                "bytes": ref["bytes"],
                "artifact_ref": ref,
            })
    for item in artifacts:
        name = str(item.get("name") or "").lower()
        ref = item.get("artifact_ref")
        if not isinstance(ref, dict) or not any(token in name for token in ("journal", "rounds")):
            continue
        try:
            path = contracts.resolve_artifact_ref(ref, roots)
        except contracts.ContextContractError:
            continue
        if path.suffix == ".jsonl" and str(path) not in journal_paths:
            journal_paths.append(str(path))
    if not journal_paths:
        return (
            {"status": "empty", "counts": {name: 0 for name in round_record.SUMMARY_BUCKETS},
             "refs": [], "journal_head": None},
            [],
            [],
        )
    bounded = round_record.summary(journal_paths, limit=MAX_RECALL_PER_BUCKET)
    journal_uris = {
        path: contracts.normalize_artifact_ref(path, roots)["uri"] for path in journal_paths
    }
    bounded["refs"] = [journal_uris[path] for path in journal_paths]
    for bucket in bounded.get("entries", {}).values():
        for entry in bucket:
            refs = []
            for ref in entry.get("refs") or []:
                if ref in journal_uris:
                    refs.append(journal_uris[ref])
                elif isinstance(ref, str) and re_portable_ref(ref):
                    refs.append(ref)
            entry["refs"] = refs[:4]
    latest_measurements = []
    for ledger in journal_paths:
        for row in reversed(round_record._load(ledger)):
            if row.get("schema_version") != round_record.JOURNAL_SCHEMA_VERSION:
                continue
            measurement = row.get("measurement_summary")
            if isinstance(measurement, dict):
                latest_measurements.append(measurement)
            if len(latest_measurements) >= MAX_MEASUREMENTS:
                break
        if len(latest_measurements) >= MAX_MEASUREMENTS:
            break
    projection = {
        "status": "available",
        "counts": bounded["counts"],
        "entries": bounded["entries"],
        "truncated": bounded["truncated"],
        "refs": bounded["refs"],
        "journal_head": bounded.get("journal_head"),
    }
    return projection, latest_measurements, journal_paths


def re_portable_ref(value: str) -> bool:
    return value.startswith(("work:/", "pack:/", "snapshot:/"))


def verify_context_document(doc: dict) -> str:
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"context schema must be {SCHEMA}")
    observed = contracts.verify_document_hash(doc, "context_hash")
    contracts.validate_context_document(doc)
    return observed


def _write_policy(path: Path) -> dict:
    policy = dict(DEFAULT_POLICY)
    policy["lease"] = dict(DEFAULT_POLICY["lease"])
    contracts.validate_context_policy(policy)
    if path.is_file():
        existing = _read_object(path)
        contracts.validate_context_policy(existing)
        return existing
    _atomic_write(str(path), policy)
    return policy


def _settle_lease(path: Path) -> None:
    if not path.is_file():
        return
    lease = _read_object(path)
    if lease.get("status") != "settled":
        lease["status"] = "settled"
        _atomic_write(str(path), lease)


def acquire_context(
    work: Path,
    pack: Path,
    stage_doc: dict,
    explicit_artifacts: list[tuple[str, Any]] | None = None,
) -> tuple[dict, dict]:
    """Create or reuse the exact context and active lease for a stage projection."""
    work, pack = work.resolve(), pack.resolve()
    work.mkdir(parents=True, exist_ok=True)
    paths = context_paths(work)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    roots = artifact_roots(work, pack)
    policy = _write_policy(paths["policy"])
    state = _read_object(work / "run_state.json") if (work / "run_state.json").is_file() else {}
    run_id, generation, role, stage = _lease_identity(stage_doc, state)
    artifacts = _context_artifacts(stage_doc, explicit_artifacts or [], roots)
    recall_summary, measurements, journal_paths = _journal_projection(work, artifacts, roots)
    payload = {
        "schema": SCHEMA,
        "run_id": run_id,
        "generation": generation,
        "role": role,
        "current_stage": stage,
        "identity": {
            "run_id": run_id,
            "generation": generation,
            "tracking": "tracked" if state else "untracked",
        },
        "measurement_summary": measurements,
        "obligations": _obligations(state),
        "recall_summary": recall_summary,
        "journal": {
            "refs": [contracts.normalize_artifact_ref(path, roots)["uri"]
                     for path in journal_paths[:16]],
            "head": recall_summary.get("journal_head"),
        },
        "actions": {
            "allowed": list(stage_doc.get("allowed") or [])[:MAX_ITEMS],
            "forbidden": list(stage_doc.get("forbidden") or [])[:MAX_ITEMS],
        },
        "artifact_refs": artifacts,
    }
    payload = contracts.stamp_document_hash(payload, "context_hash")
    contracts.validate_context_document(payload)

    current_context = None
    if paths["context"].is_file():
        try:
            current_context = _read_object(paths["context"])
            verify_context_document(current_context)
        except (ValueError, contracts.ContextContractError):
            current_context = None
    current_lease = None
    if paths["lease"].is_file():
        try:
            current_lease = _read_object(paths["lease"])
            contracts.validate_context_lease(current_lease)
        except (ValueError, contracts.ContextContractError):
            current_lease = None
    if (
        current_context == payload
        and isinstance(current_lease, dict)
        and current_lease.get("status") == "active"
        and current_lease.get("run_id") == run_id
        and current_lease.get("generation") == generation
        and current_lease.get("role") == role
        and current_lease.get("stage") == stage
        and current_lease.get("context_sha256") == payload["context_hash"]
    ):
        return current_context, current_lease

    _settle_lease(paths["lease"])
    _atomic_write(str(paths["context"]), payload)
    context_ref = contracts.normalize_artifact_ref(str(paths["context"]), roots)
    policy_ref = contracts.normalize_artifact_ref(str(paths["policy"]), roots)
    lease = {
        "schema": contracts.CONTEXT_LEASE_SCHEMA,
        "lease_id": uuid.uuid4().hex,
        "run_id": run_id,
        "generation": generation,
        "role": role,
        "stage": stage,
        "acquired_at": dt_now(),
        "policy_ref": policy_ref,
        "context_ref": context_ref,
        "context_sha256": payload["context_hash"],
        "remaining": {
            "stage_events": policy["lease"]["max_stage_events"],
            "returned_bytes": policy["lease"]["max_returned_bytes"],
        },
        "status": "active",
    }
    contracts.validate_context_lease(lease)
    _atomic_write(str(paths["lease"]), lease)
    return payload, lease


def dt_now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def context_projection(work: Path, pack: Path, context: dict, lease: dict) -> dict:
    roots = artifact_roots(work.resolve(), pack.resolve())
    return {
        "context_ref": contracts.normalize_artifact_ref(
            str(context_paths(work)["context"]), roots
        ),
        "context_hash": context["context_hash"],
        "lease_ref": contracts.normalize_artifact_ref(
            str(context_paths(work)["lease"]), roots
        ),
        "lease_id": lease["lease_id"],
    }


def rotate_context(work: Path, pack: Path, stage_doc: dict) -> dict:
    """Create a bounded disposable capsule and settle the active lease."""
    paths = context_paths(work.resolve())
    roots = artifact_roots(work.resolve(), pack.resolve())
    if not paths["context"].is_file() or not paths["lease"].is_file():
        context, lease = acquire_context(work, pack, stage_doc)
    else:
        context, lease = _read_object(paths["context"]), _read_object(paths["lease"])
    verify_context_document(context)
    if lease.get("status") != "active" or lease.get("context_sha256") != context["context_hash"]:
        raise ValueError("rotation requires an active lease bound to the current context")
    paths["receipts"].touch(exist_ok=True)
    context_ref = contracts.normalize_artifact_ref(str(paths["context"]), roots)
    policy_ref = contracts.normalize_artifact_ref(str(paths["policy"]), roots)
    _atomic_write(str(paths["recall"]), context.get("recall_summary") or {
        "status": "empty", "counts": {}, "refs": [], "journal_head": None,
    })
    _atomic_write(str(paths["obligations"]), {
        "schema": "kernel_opt.open_obligations/1",
        "run_id": context["run_id"],
        "generation": context["generation"],
        "obligations": [
            item for item in context.get("obligations") or []
            if isinstance(item, dict) and item.get("status") == "open"
        ],
    })
    recall_ref = contracts.normalize_artifact_ref(str(paths["recall"]), roots)
    obligations_ref = contracts.normalize_artifact_ref(str(paths["obligations"]), roots)
    journal_entry = next((
        item for item in context.get("artifact_refs") or []
        if isinstance(item, dict) and item.get("name") == "optimization_journal"
        and isinstance(item.get("artifact_ref"), dict)
    ), None)
    if journal_entry is None:
        raise ValueError("rotation requires the canonical optimization journal in context")
    journal_ref = journal_entry["artifact_ref"]
    next_refs = [
        item["artifact_ref"] for item in context.get("artifact_refs", [])
        if isinstance(item, dict) and isinstance(item.get("artifact_ref"), dict)
    ]
    checkpoint_refs = [
        item["artifact_ref"] for item in context.get("artifact_refs", [])
        if isinstance(item, dict) and isinstance(item.get("artifact_ref"), dict)
        and any(token in str(item.get("name") or "").lower()
                for token in ("checkpoint", "champion", "canonical"))
    ] or [context_ref]
    capsule_base = {
        "schema": contracts.RESUME_CAPSULE_SCHEMA,
        "run_id": context["run_id"],
        "generation": context["generation"],
        "role": context["role"],
        "stage": context["current_stage"],
        "policy_ref": policy_ref,
        "context_ref": context_ref,
        "context_sha256": context["context_hash"],
        "checkpoint_ref": checkpoint_refs[0],
        "checkpoint_refs": checkpoint_refs,
        "journal_head_ref": journal_ref,
        "journal_head_sha256": (context.get("journal") or {}).get("head", {}).get("entry_sha256")
        if isinstance((context.get("journal") or {}).get("head"), dict) else None,
        "recall_ref": recall_ref,
        "open_obligation_refs": [obligations_ref],
        "next_refs": next_refs,
    }
    capsule = contracts.stamp_document_hash(capsule_base, "capsule_sha256")
    while len(contracts.canonical_json_bytes(capsule)) > contracts.MAX_CAPSULE_BYTES \
            and (next_refs or len(checkpoint_refs) > 1):
        if next_refs:
            next_refs.pop()
            capsule_base["next_refs"] = next_refs
        else:
            checkpoint_refs.pop()
            capsule_base["checkpoint_refs"] = checkpoint_refs
        capsule = contracts.stamp_document_hash(capsule_base, "capsule_sha256")
    contracts.validate_resume_capsule(capsule)
    _atomic_write(str(paths["capsule"]), capsule)
    _settle_lease(paths["lease"])
    return capsule


def resume_context(work: Path, pack: Path, role: str, stage_doc: dict, capsule_path: Path) -> tuple[dict, dict]:
    """Verify a capsule against durable identity and reacquire a fresh lease."""
    work, pack = work.resolve(), pack.resolve()
    roots = artifact_roots(work, pack)
    capsule = _read_object(capsule_path.resolve())
    contracts.validate_resume_capsule(capsule)
    for field in ("policy_ref", "context_ref", "checkpoint_ref", "journal_head_ref", "recall_ref"):
        contracts.resolve_artifact_ref(capsule[field], roots)
    for ref in (list(capsule.get("checkpoint_refs") or [])
                + list(capsule.get("open_obligation_refs") or [])
                + list(capsule.get("next_refs") or [])):
        contracts.resolve_artifact_ref(ref, roots)
    old_context = _read_object(contracts.resolve_artifact_ref(capsule["context_ref"], roots))
    observed_hash = verify_context_document(old_context)
    recall_doc = _read_object(contracts.resolve_artifact_ref(capsule["recall_ref"], roots))
    if recall_doc != old_context.get("recall_summary"):
        raise ValueError("resume recall summary differs from the rotated context")
    obligation_doc = _read_object(
        contracts.resolve_artifact_ref(capsule["open_obligation_refs"][0], roots)
    )
    expected_obligations = [
        item for item in old_context.get("obligations") or []
        if isinstance(item, dict) and item.get("status") == "open"
    ]
    if obligation_doc.get("obligations") != expected_obligations:
        raise ValueError("resume open obligations differ from the rotated context")
    journal_path = contracts.resolve_artifact_ref(capsule["journal_head_ref"], roots)
    observed_head = round_record.journal_head(str(journal_path)).get("entry_sha256")
    if observed_head != capsule.get("journal_head_sha256"):
        raise ValueError("resume journal head differs from the rotated context")
    state = _read_object(work / "run_state.json") if (work / "run_state.json").is_file() else {}
    run_id, generation, expected_role, stage = _lease_identity(stage_doc, state)
    expected = {
        "run_id": run_id,
        "generation": generation,
        "role": expected_role,
        "stage": stage,
        "context_sha256": observed_hash,
    }
    if role != expected_role:
        raise ValueError(f"resume role {role!r} does not match active role {expected_role!r}")
    for field, value in expected.items():
        if capsule.get(field) != value:
            raise ValueError(
                f"resume capsule {field} mismatch: capsule={capsule.get(field)!r}, current={value!r}"
            )
    stage_names = set((stage_doc.get("artifact_refs") or {}).keys())
    explicit = [
        (str(item.get("name") or f"capsule-{index}"), item["artifact_ref"])
        for index, item in enumerate(old_context.get("artifact_refs") or [])
        if isinstance(item, dict) and isinstance(item.get("artifact_ref"), dict)
        and item.get("name") not in stage_names
        and item.get("name") != "optimization_journal"
    ]
    context, lease = acquire_context(work, pack, stage_doc, explicit)
    if context["context_hash"] != capsule["context_sha256"]:
        _settle_lease(context_paths(work)["lease"])
        raise ValueError("reacquired context hash differs from the resume capsule")
    if (context.get("journal") or {}).get("head") != (old_context.get("journal") or {}).get("head"):
        _settle_lease(context_paths(work)["lease"])
        raise ValueError("cold resume journal projection is not equivalent")
    if context.get("recall_summary") != old_context.get("recall_summary"):
        _settle_lease(context_paths(work)["lease"])
        raise ValueError("cold resume recall projection is not equivalent")
    if context.get("obligations") != old_context.get("obligations"):
        _settle_lease(context_paths(work)["lease"])
        raise ValueError("cold resume obligations are not equivalent")
    return context, lease


def describe() -> dict:
    return {
        "tool": "stage_context",
        "schema": SCHEMA,
        "commands": ["build", "selftest"],
        "inputs": list(_KINDS),
        "output_fields": ["current_stage", "identity", "measurement_summary", "obligations",
                          "recall_summary", "journal", "actions", "artifact_refs", "context_hash"],
        "bounds": {"artifact_refs": MAX_REFS, "list_items": MAX_ITEMS,
                   "measurement_summaries": MAX_MEASUREMENTS},
        "guarantees": [
            "requires an explicit current stage",
            "does not scan directories or replay ledgers/raw dumps",
            "prints only a concise receipt unless --out is supplied",
        ],
        "non_guarantees": ["cannot prevent a host or agent from directly reading a file"],
    }


def _selftest() -> int:
    root = tempfile.mkdtemp(prefix="stage_context_selftest_")
    try:
        canonical = os.path.join(root, "canonical.json")
        state = os.path.join(root, "run_state.json")
        recall = os.path.join(root, "recall.json")
        with open(canonical, "w", encoding="utf-8") as fh:
            json.dump({"run_id": "run-a", "generation": 2, "measurement": {
                "measurement_id": "m-a", "value": 1.05,
                "identity": {"shape_set": ["s1"], "unit": "ratio", "aggregation": "geomean",
                             "boundary": "kernel", "source": "device_timing", "scope": "device"}}}, fh)
        with open(state, "w", encoding="utf-8") as fh:
            json.dump({"obligations": [{"obligation_id": "profile@g2", "kind": "profile",
                                        "status": "open", "generation": 2}]}, fh)
        with open(recall, "w", encoding="utf-8") as fh:
            json.dump({"schema": "kernel_opt.round_summary/1",
                       "counts": {"valid": 2, "stale": 1, "unfinished": 0, "recheck": 1},
                       "refs": ["rounds.jsonl"]}, fh)
        package = build_package("search", {"canonical": [canonical], "run_state": [state],
                                            "recall": [recall]}, allowed=["measure"],
                                forbidden=["read raw ledger"])
        assert package["current_stage"] == "search"
        assert package["identity"]["run_id"] == "run-a"
        assert package["obligations"][0]["status"] == "open"
        assert package["recall_summary"]["counts"]["stale"] == 1
        assert "hypothesis" not in _canonical(package).decode("utf-8")
        again = build_package("search", {"canonical": [canonical], "run_state": [state],
                                          "recall": [recall]}, allowed=["measure"],
                              forbidden=["read raw ledger"])
        assert package["context_hash"] == again["context_hash"]
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    print("[stage_context] SELFTEST PASS")
    return 0


def main() -> int:
    if sys.argv[1:] in (["--describe", "json"], ["--describe", "--format=json"]):
        print(json.dumps(describe(), ensure_ascii=False, sort_keys=True))
        return 0
    if sys.argv[1:] == ["--selftest"]:
        return _selftest()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build one bounded stage context package")
    build.add_argument("--stage", "--current-stage", dest="stage", required=True,
                       help="the one current stage; no all-stage default")
    for kind in _KINDS:
        build.add_argument("--" + kind.replace("_", "-"), action="append", default=[],
                           help=f"explicit {kind} JSON reference (repeatable)")
    build.add_argument("--canonical-registry", dest="canonical", action="append",
                       help="alias for --canonical")
    build.add_argument("--allowed-action", action="append", default=[])
    build.add_argument("--forbidden-action", action="append", default=[])
    build.add_argument("--out", help="write the complete package atomically")
    sub.add_parser("selftest", help="run no-GPU fixture checks")
    args = parser.parse_args()
    if args.command == "selftest":
        return _selftest()
    references = {kind: getattr(args, kind) for kind in _KINDS}
    package = build_package(args.stage, references, allowed=args.allowed_action,
                            forbidden=args.forbidden_action)
    if args.out:
        _atomic_write(args.out, package)
    print(json.dumps({"context_hash": package["context_hash"], "current_stage": package["current_stage"],
                      "artifact_refs": len(package["artifact_refs"]),
                      "out": args.out or None}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
