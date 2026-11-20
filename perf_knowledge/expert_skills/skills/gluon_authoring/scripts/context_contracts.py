#!/usr/bin/env python3
"""Portable context-contract primitives shared by control-plane tools.

This module deliberately has no pack, host, GPU, or third-party dependency.  It
accepts legacy string/absolute-path artifact references at read boundaries, but
always emits a root-relative URI plus a complete SHA-256 digest.

It does not implement the toolctl context lifecycle.  Later lifecycle code should
use these functions rather than creating another path or hashing convention.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ARTIFACT_REF_SCHEMA = "kernel_opt.artifact_ref/1"
CONTEXT_POLICY_SCHEMA = "kernel_opt.context_policy/1"
CONTEXT_LEASE_SCHEMA = "kernel_opt.context_lease/1"
RESUME_CAPSULE_SCHEMA = "kernel_opt.resume_capsule/1"

PORTABLE_SCHEMES = ("work", "pack", "snapshot")
MAX_CONTEXT_BYTES = 32 * 1024
MAX_CAPSULE_BYTES = 8 * 1024

_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_SHA256 = re.compile(r"^[0-9a-f]{8,64}$")
_SCHEMA_ID = re.compile(r"^(?P<family>[A-Za-z0-9_.-]+)/(?P<major>[1-9][0-9]*)$")
_URI = re.compile(r"^(?P<scheme>work|pack|snapshot):/(?P<path>.*)$")


class ContextContractError(ValueError):
    """Base class for invalid or unsafe contract input."""


class RootEscapeError(ContextContractError):
    """A reference resolves outside its declared portable root."""


class ArtifactHashMismatch(ContextContractError):
    """An artifact's observed bytes do not match its declared digest."""


class SchemaMajorError(ContextContractError):
    """A document uses an unsupported or malformed schema major."""


class ContractSizeError(ContextContractError):
    """A canonical JSON document exceeds its portable byte budget."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes suitable for content addressing."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContextContractError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    """Return the complete SHA-256 of a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_document_hash(document: Mapping[str, Any], *, hash_field: str | None = None) -> str:
    """Hash a document, optionally excluding its self-referential digest field."""
    payload = dict(document)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return canonical_json_hash(payload)


def stamp_document_hash(document: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    """Return a copy carrying a full canonical digest."""
    stamped = copy.deepcopy(dict(document))
    stamped[hash_field] = canonical_document_hash(stamped, hash_field=hash_field)
    return stamped


def verify_document_hash(document: Mapping[str, Any], hash_field: str) -> str:
    """Verify and return a document's complete canonical digest."""
    expected = document.get(hash_field)
    if not isinstance(expected, str) or not _FULL_SHA256.fullmatch(expected):
        raise ArtifactHashMismatch(f"{hash_field} must be a complete lowercase SHA-256")
    observed = canonical_document_hash(document, hash_field=hash_field)
    if not hmac.compare_digest(expected, observed):
        raise ArtifactHashMismatch(
            f"{hash_field} mismatch: declared {expected}, observed {observed}"
        )
    return observed


def parse_schema_id(schema: Any) -> tuple[str, int]:
    """Split ``family/major`` schema IDs without accepting ambiguous variants."""
    if not isinstance(schema, str):
        raise SchemaMajorError("schema must be a family/major string")
    match = _SCHEMA_ID.fullmatch(schema)
    if match is None:
        raise SchemaMajorError(f"malformed schema id: {schema!r}")
    return match.group("family"), int(match.group("major"))


def require_schema_major(
    document: Mapping[str, Any],
    expected_family: str,
    supported_major: int = 1,
) -> None:
    """Reject a wrong schema family or an unsupported major version."""
    family, major = parse_schema_id(document.get("schema"))
    if family != expected_family:
        raise SchemaMajorError(
            f"schema family {family!r} is not expected family {expected_family!r}"
        )
    if major != supported_major:
        raise SchemaMajorError(
            f"unsupported {expected_family} schema major {major}; supported={supported_major}"
        )


def _roots(roots: Mapping[str, os.PathLike[str] | str]) -> dict[str, Path]:
    unknown = sorted(set(roots) - set(PORTABLE_SCHEMES))
    if unknown:
        raise ContextContractError(f"unknown artifact root scheme(s): {unknown}")
    if not roots:
        raise ContextContractError("at least one artifact root must be declared")
    result: dict[str, Path] = {}
    for scheme in PORTABLE_SCHEMES:
        if scheme not in roots:
            continue
        root = Path(roots[scheme]).expanduser().resolve()
        if not root.is_dir():
            raise ContextContractError(f"{scheme} root is not a directory: {root}")
        result[scheme] = root
    return result


def _portable_relative(raw: str) -> PurePosixPath:
    if not raw or raw.startswith("/") or "\\" in raw or "\x00" in raw:
        raise RootEscapeError(f"portable artifact path is not root-relative: {raw!r}")
    if "?" in raw or "#" in raw:
        raise RootEscapeError("portable artifact URI must not contain query or fragment")
    path = PurePosixPath(raw)
    if any(part in ("", ".", "..") for part in path.parts):
        raise RootEscapeError(f"portable artifact path contains a dot/empty segment: {raw!r}")
    return path


def _confined(root: Path, candidate: Path, *, source: str) -> tuple[Path, Path]:
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RootEscapeError(f"artifact reference escapes its root: {source!r}") from exc
    if not resolved.is_file():
        raise ContextContractError(f"artifact is not a readable regular file: {source!r}")
    return resolved, relative


def _from_portable_uri(uri: str, roots: Mapping[str, Path]) -> tuple[str, Path, Path]:
    match = _URI.fullmatch(uri)
    if match is None:
        raise ContextContractError(f"unsupported artifact URI: {uri!r}")
    scheme = match.group("scheme")
    if scheme not in roots:
        raise ContextContractError(f"artifact URI uses undeclared root scheme: {scheme!r}")
    relative = _portable_relative(match.group("path"))
    resolved, final_relative = _confined(
        roots[scheme], roots[scheme].joinpath(*relative.parts), source=uri
    )
    return scheme, resolved, final_relative


def _from_legacy_path(
    reference: str,
    roots: Mapping[str, Path],
    default_scheme: str,
) -> tuple[str, Path, Path]:
    path = Path(reference).expanduser()
    if not path.is_absolute():
        if default_scheme not in roots:
            raise ContextContractError(f"unknown default artifact scheme: {default_scheme!r}")
        return (
            default_scheme,
            *_confined(roots[default_scheme], roots[default_scheme] / path, source=reference),
        )

    # Select by the lexical location first, then resolve inside that same root.  A symlink from
    # work into pack must be rejected as a work-root escape, not silently relabelled as pack.
    lexical = Path(os.path.abspath(path))
    candidates: list[tuple[int, str, Path]] = []
    for scheme, root in roots.items():
        try:
            lexical.relative_to(root)
        except ValueError:
            continue
        candidates.append((len(root.parts), scheme, root))
    if not candidates:
        raise RootEscapeError(f"absolute artifact path is outside all declared roots: {reference!r}")
    _, scheme, root = max(candidates)
    resolved, relative = _confined(root, lexical, source=reference)
    return scheme, resolved, relative


def file_sha256(path: os.PathLike[str] | str) -> str:
    """Hash a file without truncation."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_artifact_ref(
    reference: str | Mapping[str, Any],
    roots: Mapping[str, os.PathLike[str] | str],
    *,
    default_scheme: str = "work",
    media_type: str | None = None,
) -> dict[str, Any]:
    """Ingest a v1 or legacy reference and emit a portable, fully hashed ArtifactRef.

    Accepted legacy object keys are ``ref`` or ``path`` plus an optional truncated
    ``sha256``/``hash``.  A legacy digest is accepted only when it matches the observed
    full digest prefix; output is always the complete digest.
    """
    declared_hash: Any = None
    declared_bytes: Any = None
    source_media_type: Any = None
    if isinstance(reference, str):
        source = reference
    elif isinstance(reference, Mapping):
        if "schema" in reference:
            require_schema_major(reference, "kernel_opt.artifact_ref")
        source = reference.get("uri") or reference.get("ref") or reference.get("path")
        declared_hash = reference.get("sha256", reference.get("hash"))
        declared_bytes = reference.get("bytes")
        source_media_type = reference.get("media_type")
        if not isinstance(source, str) or not source:
            raise ContextContractError("artifact reference object needs uri, ref, or path")
    else:
        raise ContextContractError("artifact reference must be a string or object")

    normalized_roots = _roots(roots)
    if _URI.fullmatch(source):
        scheme, path, relative = _from_portable_uri(source, normalized_roots)
    elif re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", source):
        raise ContextContractError(f"unsupported artifact URI scheme: {source!r}")
    else:
        scheme, path, relative = _from_legacy_path(source, normalized_roots, default_scheme)

    observed = file_sha256(path)
    if declared_hash is not None:
        digest = str(declared_hash).lower()
        if digest.startswith("sha256:"):
            digest = digest[len("sha256:") :]
        if not _LEGACY_SHA256.fullmatch(digest):
            raise ArtifactHashMismatch("declared hash must be 8-64 hexadecimal SHA-256 characters")
        if not hmac.compare_digest(digest, observed[: len(digest)]):
            raise ArtifactHashMismatch(
                f"artifact hash mismatch: declared {digest}, observed {observed}"
            )
    size = path.stat().st_size
    if declared_bytes is not None and declared_bytes != size:
        raise ContextContractError(
            f"artifact byte count mismatch: declared {declared_bytes!r}, observed {size}"
        )

    result: dict[str, Any] = {
        "schema": ARTIFACT_REF_SCHEMA,
        "uri": f"{scheme}:/{relative.as_posix()}",
        "sha256": observed,
        "bytes": size,
    }
    chosen_media_type = media_type if media_type is not None else source_media_type
    if chosen_media_type is not None:
        if not isinstance(chosen_media_type, str) or not chosen_media_type:
            raise ContextContractError("media_type must be a non-empty string")
        result["media_type"] = chosen_media_type
    return result


def resolve_artifact_ref(
    reference: Mapping[str, Any],
    roots: Mapping[str, os.PathLike[str] | str],
) -> Path:
    """Resolve and integrity-check a normalized ArtifactRef."""
    normalized = normalize_artifact_ref(reference, roots)
    match = _URI.fullmatch(normalized["uri"])
    assert match is not None
    root = _roots(roots)[match.group("scheme")]
    return root.joinpath(*_portable_relative(match.group("path")).parts).resolve()


def enforce_canonical_json_size(document: Any, max_bytes: int, label: str) -> int:
    """Enforce a portable byte bound against canonical JSON, not Python object size."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ContextContractError("max_bytes must be a positive integer")
    size = len(canonical_json_bytes(document))
    if size > max_bytes:
        raise ContractSizeError(f"{label} is {size} bytes; maximum is {max_bytes}")
    return size


def validate_context_document(document: Mapping[str, Any]) -> int:
    """Enforce the universal 32 KiB derived-context ceiling."""
    return enforce_canonical_json_size(document, MAX_CONTEXT_BYTES, "stage context")


def validate_context_policy(document: Mapping[str, Any]) -> None:
    """Validate the load-bearing policy bounds without a JSON Schema dependency."""
    require_schema_major(document, "kernel_opt.context_policy")
    maxima = (
        ("max_context_bytes", MAX_CONTEXT_BYTES),
        ("max_capsule_bytes", MAX_CAPSULE_BYTES),
        ("max_query_return_bytes", MAX_CONTEXT_BYTES),
    )
    for field, ceiling in maxima:
        value = document.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= ceiling:
            raise ContextContractError(f"{field} must be an integer in [1, {ceiling}]")
    lease = document.get("lease")
    if not isinstance(lease, Mapping):
        raise ContextContractError("context policy needs a lease object")
    for field in ("max_stage_events", "max_returned_bytes"):
        value = lease.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ContextContractError(f"lease.{field} must be a positive integer")


def validate_context_lease(document: Mapping[str, Any]) -> None:
    """Validate portable lease identity and non-negative remaining budgets."""
    require_schema_major(document, "kernel_opt.context_lease")
    for field in ("lease_id", "run_id", "role", "stage", "acquired_at"):
        if not isinstance(document.get(field), str) or not document[field]:
            raise ContextContractError(f"{field} must be a non-empty string")
    generation = document.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise ContextContractError("generation must be a positive integer")
    digest = document.get("context_sha256")
    if not isinstance(digest, str) or not _FULL_SHA256.fullmatch(digest):
        raise ContextContractError("context_sha256 must be a complete lowercase SHA-256")
    for field in ("policy_ref", "context_ref"):
        ref = document.get(field)
        if not isinstance(ref, Mapping):
            raise ContextContractError(f"{field} must be an ArtifactRef object")
        require_schema_major(ref, "kernel_opt.artifact_ref")
    remaining = document.get("remaining")
    if not isinstance(remaining, Mapping):
        raise ContextContractError("remaining must be an object")
    for field in ("stage_events", "returned_bytes"):
        value = remaining.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ContextContractError(f"remaining.{field} must be a non-negative integer")
    if document.get("status") not in ("active", "rotation_required", "settled"):
        raise ContextContractError("status must be active, rotation_required, or settled")


def validate_resume_capsule(document: Mapping[str, Any]) -> int:
    """Check schema major, self-hash, and the universal 8 KiB capsule ceiling."""
    require_schema_major(document, "kernel_opt.resume_capsule")
    verify_document_hash(document, "capsule_sha256")
    for field in ("run_id", "role", "stage", "context_sha256"):
        if not isinstance(document.get(field), str) or not document[field]:
            raise ContextContractError(f"resume capsule {field} must be a non-empty string")
    generation = document.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise ContextContractError("resume capsule generation must be a positive integer")
    for field in ("policy_ref", "context_ref", "checkpoint_ref", "journal_head_ref", "recall_ref"):
        ref = document.get(field)
        if not isinstance(ref, Mapping):
            raise ContextContractError(f"resume capsule {field} must be an ArtifactRef")
        require_schema_major(ref, "kernel_opt.artifact_ref")
    for field in ("checkpoint_refs", "open_obligation_refs", "next_refs"):
        refs = document.get(field)
        if not isinstance(refs, list) or (field != "next_refs" and not refs):
            raise ContextContractError(f"resume capsule {field} must be a "
                                       f"{'non-empty ' if field != 'next_refs' else ''}list")
        for ref in refs:
            if not isinstance(ref, Mapping):
                raise ContextContractError(f"resume capsule {field} entries must be ArtifactRefs")
            require_schema_major(ref, "kernel_opt.artifact_ref")
    head = document.get("journal_head_sha256")
    if head is not None and (not isinstance(head, str) or not _FULL_SHA256.fullmatch(head)):
        raise ContextContractError("journal_head_sha256 must be null or a complete SHA-256")
    return enforce_canonical_json_size(document, MAX_CAPSULE_BYTES, "resume capsule")


def _selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="context-contracts-") as tmp:
        base = Path(tmp)
        roots_a = {name: base / "a" / name for name in PORTABLE_SCHEMES}
        roots_b = {name: base / "b" / name for name in PORTABLE_SCHEMES}
        for roots in (roots_a, roots_b):
            for root in roots.values():
                root.mkdir(parents=True)
            target = roots["work"] / "records" / "result.json"
            target.parent.mkdir()
            target.write_text('{"value":1}\n', encoding="utf-8")

        legacy = str(roots_a["work"] / "records" / "result.json")
        ref_a = normalize_artifact_ref(legacy, roots_a)
        ref_b = normalize_artifact_ref(
            {"path": "records/result.json", "sha256": ref_a["sha256"][:16]}, roots_b
        )
        assert ref_a == ref_b
        assert len(ref_a["sha256"]) == 64

        try:
            normalize_artifact_ref("work:/../outside", roots_a)
        except RootEscapeError:
            pass
        else:
            raise AssertionError("path escape was accepted")

        try:
            normalize_artifact_ref({"ref": legacy, "sha256": "0" * 16}, roots_a)
        except ArtifactHashMismatch:
            pass
        else:
            raise AssertionError("hash tamper was accepted")

        assert canonical_json_hash(ref_a) == canonical_json_hash(ref_b)
        validate_context_document({"payload": "x" * 100})
        try:
            validate_context_document({"payload": "x" * MAX_CONTEXT_BYTES})
        except ContractSizeError:
            pass
        else:
            raise AssertionError("oversized context was accepted")

        capsule_base = {
            "schema": RESUME_CAPSULE_SCHEMA,
            "run_id": "selftest",
            "generation": 1,
            "role": "direction",
            "stage": "search",
            "policy_ref": ref_a,
            "context_ref": ref_a,
            "context_sha256": "0" * 64,
            "checkpoint_ref": ref_a,
            "checkpoint_refs": [ref_a],
            "journal_head_ref": ref_a,
            "journal_head_sha256": None,
            "recall_ref": ref_a,
            "open_obligation_refs": [ref_a],
            "next_refs": [],
        }
        capsule = stamp_document_hash(capsule_base, "capsule_sha256")
        validate_resume_capsule(capsule)
        oversized = stamp_document_hash(
            dict(capsule_base, padding="x" * MAX_CAPSULE_BYTES),
            "capsule_sha256",
        )
        try:
            validate_resume_capsule(oversized)
        except ContractSizeError:
            pass
        else:
            raise AssertionError("oversized capsule was accepted")
    print("context_contracts selftest: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run pure-CPU contract fixtures")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
