#!/usr/bin/env python3
"""Persist and retrieve Deep Research Agent findings.

The Researcher remains the sole author of the knowledge content.  This module
only validates, normalizes, deduplicates, and materializes its existing Stage-7
artifacts.  It deliberately has no model or network dependency.

Storage layout::

    <kb_dir>/
      observations/<run_id>.json
      cards/<operator>/<card_id>.json
      cards/<operator>/<card_id>.md
      snapshots/<snapshot_id>.json
      channels/latest.json
      index.json
      INDEX.md

Online mode calls ``ingest`` immediately after ``research_synthesize``.
Offline mode calls ``retrieve`` and hands the generated compact brief to the
unchanged TechLead ``DEEP_SEARCH_BRIEF`` input.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

try:  # Linux CI/runtime.  The fallback still keeps atomic file replacement.
    import fcntl
except ImportError:  # pragma: no cover - GEAK targets Linux
    fcntl = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
MERGE_THRESHOLD = 0.62
CONFLICT_THRESHOLD = 0.72
DEFAULT_MAX_DIRECTIONS = 8
UNKNOWN = {"", "unknown", "none", "null", "n/a", "na"}
SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".h",
    ".hip",
    ".hpp",
    ".js",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
}
IGNORED_DIRS = {
    ".git",
    ".torch_ext",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "full",
    "in",
    "into",
    "is",
    "it",
    "level",
    "of",
    "on",
    "or",
    "then",
    "the",
    "this",
    "to",
    "entire",
    "use",
    "via",
    "with",
}
TOKEN_ALIASES = {
    "cudagraph": "graph_capture",
    "cudagraphs": "graph_capture",
    "hipgraph": "graph_capture",
    "hipgraphs": "graph_capture",
    "wavefront": "wave",
    "wavefronts": "wave",
    "warps": "wave",
    "warp": "wave",
    "shared_memory": "lds",
    "sharedmemory": "lds",
    "registers": "vgpr",
    "register": "vgpr",
    "pipelining": "pipeline",
    "pipelined": "pipeline",
    "dispatches": "dispatch",
    "launches": "launch",
}
CONFIDENCE_VALUE = {"low": 1, "medium": 2, "high": 3}
VALUE_CONFIDENCE = {1: "low", 2: "medium", 3: "high"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()[:length]


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@contextlib.contextmanager
def _exclusive_lock(kb_dir: Path) -> Iterator[None]:
    kb_dir.mkdir(parents=True, exist_ok=True)
    with (kb_dir / ".merge.lock").open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _canonical(value: Any) -> str:
    text = _clean(value).lower()
    text = re.sub(r"[^a-z0-9_+.-]+", "_", text)
    return text.strip("_")


def _slug(value: Any, fallback: str = "unknown") -> str:
    text = _canonical(value).replace(".", "-").replace("+", "-")
    return (text or fallback)[:80]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, list) else re.split(r"[,|]", str(value))
    out: list[str] = []
    for item in raw:
        val = _canonical(item)
        if val and val not in UNKNOWN and val not in out:
            out.append(val)
    return out


def _confidence(value: Any) -> str:
    val = _canonical(value)
    return val if val in CONFIDENCE_VALUE else "medium"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _tokens(text: Any) -> set[str]:
    normalized = _clean(text).lower()
    normalized = re.sub(r"cuda[\s_-]*graphs?", " graph_capture ", normalized)
    normalized = re.sub(r"hip[\s_-]*graphs?", " graph_capture ", normalized)
    normalized = re.sub(r"shared[\s_-]*memory", " shared_memory ", normalized)
    words = re.findall(r"[a-z0-9][a-z0-9_+]*", normalized)
    out: set[str] = set()
    for word in words:
        word = TOKEN_ALIASES.get(word, word)
        if word in STOP_WORDS or len(word) < 2:
            continue
        # Very light normalization makes weekly wording variation merge without
        # pretending to be a general-purpose stemmer.
        if len(word) > 5 and word.endswith("ing"):
            word = word[:-3]
        elif len(word) > 4 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(TOKEN_ALIASES.get(word, word))
    # Small mechanism families make paraphrases searchable without an embedding
    # dependency. They do not add a claim; they only provide merge keys.
    if "graph" in out and ({"capture", "replay"} & out):
        out.add("graph_capture")
    if "graph_capture" in out:
        out.difference_update({"graph", "capture", "replay"})
    if any(token.startswith("wrapper") for token in out):
        out.add("wrapper")
    if "overhead" in out and ({"launch", "dispatch"} & out):
        out.add("launch_overhead")
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _containment(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _finding_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    lt, rt = _tokens(left.get("title")), _tokens(right.get("title"))
    lm, rm = _tokens(left.get("mechanism")), _tokens(right.get("mechanism"))
    title = 0.6 * _containment(lt, rt) + 0.4 * _jaccard(lt, rt)
    mechanism = 0.6 * _containment(lm, rm) + 0.4 * _jaccard(lm, rm)
    return 0.35 * title + 0.65 * mechanism


def _rejection_similarity(card: dict[str, Any], rejected: dict[str, Any]) -> float:
    left, right = _tokens(card.get("title")), _tokens(rejected.get("title"))
    title = 0.6 * _containment(left, right) + 0.4 * _jaccard(left, right)
    return max(title, _finding_similarity(card, rejected))


def _known(value: Any) -> bool:
    return _canonical(value) not in UNKNOWN


def _list_compatible(left: Any, right: Any) -> bool:
    lvals, rvals = set(_as_list(left)), set(_as_list(right))
    return not lvals or not rvals or bool(lvals & rvals)


def _scope_compatible(
    left: dict[str, Any], right: dict[str, Any], *, exact_kernel_ok: bool = True
) -> bool:
    exact_kernel = bool(
        exact_kernel_ok
        and left.get("source_kernel_fingerprint")
        and left.get("source_kernel_fingerprint") == right.get("source_kernel_fingerprint")
    )
    for key in ("operator", "language"):
        if _known(left.get(key)) and _known(right.get(key)):
            if _canonical(left[key]) != _canonical(right[key]):
                return False
    # Unknown/custom operators are unsafe to merge across unrelated kernel names.
    if (
        not exact_kernel
        and not _known(left.get("operator"))
        and not _known(right.get("operator"))
    ):
        if _known(left.get("kernel_name")) and _known(right.get("kernel_name")):
            if _canonical(left["kernel_name"]) != _canonical(right["kernel_name"]):
                return False
    return all(
        _list_compatible(left.get(key), right.get(key))
        for key in ("gfx", "dtypes", "regimes")
    )


def _source_fingerprint(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    digest = hashlib.sha256()
    if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        return ""
    files: list[Path] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if any(part in IGNORED_DIRS for part in candidate.relative_to(path).parts):
            continue
        files.append(candidate)
    for candidate in sorted(files):
        rel = candidate.relative_to(path).as_posix()
        try:
            data = candidate.read_bytes()
        except OSError:
            continue
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest() if files else ""


def _infer_gfx(*values: Any) -> list[str]:
    found: list[str] = []
    for value in values:
        for gfx in re.findall(r"\bgfx[0-9a-z]+\b", _clean(value).lower()):
            if gfx not in found:
                found.append(gfx)
    return found


def _scope_from_inputs(
    facts: dict[str, Any],
    *,
    operator: str = "",
    language: str = "",
    backend: str = "",
    gfx: str = "",
    dtype: str = "",
    regime: str = "",
    bottleneck: str = "",
    kernel_name: str = "",
    kernel_path: Path | None = None,
) -> dict[str, Any]:
    facts_backend = _clean(facts.get("kernel_backend"))
    return {
        "operator": _canonical(operator) or "unknown",
        "language": _canonical(language or facts.get("kernel_language")) or "unknown",
        "backend": _canonical(backend or facts_backend) or "unknown",
        "gfx": _infer_gfx(gfx) or _as_list(gfx) or _infer_gfx(facts_backend, facts.get("notes")),
        "dtypes": _as_list(dtype),
        "regimes": _as_list(regime),
        "bottleneck": _canonical(bottleneck or facts.get("bottleneck_type")) or "unknown",
        "kernel_name": _canonical(kernel_name) or "unknown",
        "source_kernel_fingerprint": _source_fingerprint(kernel_path),
    }


def _sanitize_evidence(
    evidence: Any, *, eval_dir: Path, run_id: str
) -> list[dict[str, str]]:
    if not isinstance(evidence, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    eval_prefix = str(eval_dir.resolve())
    for item in evidence:
        if not isinstance(item, dict):
            continue
        url = _clean(item.get("url"))
        if eval_prefix and eval_prefix in url:
            url = url.replace(eval_prefix, f"run://{run_id}")
        elif url.startswith(("local:/home/", "local:///home/", "/home/", "/root/")):
            url = f"run://{run_id}/local-evidence"
        title = _clean(item.get("title"))
        key = (url, title)
        if not any(key) or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "title": title,
                "url": url,
                "kind": _canonical(item.get("kind") or item.get("source_type")) or "unknown",
                "note": _clean(item.get("note") or item.get("snippet")),
            }
        )
    return out


def _normalize_direction(
    raw: dict[str, Any],
    *,
    scope: dict[str, Any],
    run_id: str,
    artifact_sha256: str,
    eval_dir: Path,
) -> dict[str, Any] | None:
    title = _clean(raw.get("title"))
    mechanism = _clean(raw.get("mechanism"))
    if not title or not mechanism:
        return None
    direction_id = _clean(raw.get("id") or raw.get("direction_id"))
    finding = {
        "direction_id": direction_id,
        "title": title,
        "specialty": _canonical(raw.get("specialty")) or "deep_explore",
        "bottleneck": _clean(raw.get("bottleneck") or raw.get("bottleneck_addressed")),
        "mechanism": mechanism,
        "expected_upside": _clean(raw.get("expected_upside")),
        "implementation_cost": _clean(raw.get("implementation_cost") or raw.get("cost")),
        "confidence": _confidence(raw.get("confidence")),
        "kill_criterion": _clean(raw.get("kill_criterion") or raw.get("kill_criteria")),
        "rank_score": _number(raw.get("rank_score")),
        "rationale_for_rank": _clean(raw.get("rationale_for_rank")),
        "evidence": _sanitize_evidence(raw.get("evidence"), eval_dir=eval_dir, run_id=run_id),
        "scope": dict(scope),
        "provenance": {
            "source_run_id": run_id,
            "artifact_sha256": artifact_sha256,
            "direction_id": direction_id,
        },
    }
    finding["observation_id"] = "obs-" + _stable_hash(
        {
            "run": run_id,
            "direction": direction_id,
            "title": title,
            "mechanism": mechanism,
            "scope": scope,
        },
        20,
    )
    return finding


def _normalize_rejected(
    raw: Any, *, scope: dict[str, Any], run_id: str
) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        title = _clean(raw.get("title") or raw.get("direction"))
        reason = _clean(
            raw.get("reason") or raw.get("notes") or raw.get("mechanism")
        )
    else:
        title, reason = _clean(raw), ""
    if not title:
        return None
    return {
        "title": title,
        "mechanism": reason or title,
        "reason": reason,
        "scope": dict(scope),
        "run_id": run_id,
        "rejection_id": "reject-"
        + _stable_hash(
            {"run": run_id, "title": title, "reason": reason, "scope": scope},
            20,
        ),
    }


def _load_cards(kb_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    cards: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((kb_dir / "cards").glob("*/*.json")):
        card = _read_json(path)
        if isinstance(card, dict) and card.get("card_id"):
            cards.append((path, card))
    return cards


def _merge_evidence(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out = list(existing)
    seen = {
        (_clean(item.get("url")), _clean(item.get("title")))
        for item in existing
        if isinstance(item, dict)
    }
    for item in incoming:
        key = (_clean(item.get("url")), _clean(item.get("title")))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _aggregate_confidence(values: list[str]) -> str:
    nums = [CONFIDENCE_VALUE.get(_confidence(value), 2) for value in values]
    if not nums:
        return "medium"
    # Round down on disagreement: recurring research can strengthen retrieval,
    # but variable confidence should not be silently upgraded.
    return VALUE_CONFIDENCE[max(1, min(3, math.floor(sum(nums) / len(nums))))]


def _new_card(finding: dict[str, Any], now: str) -> dict[str, Any]:
    scope = finding["scope"]
    identity = {
        "scope": scope,
        "specialty": finding["specialty"],
        "tokens": sorted(_tokens(finding["title"] + " " + finding["mechanism"])),
    }
    card_id = f"research-{_slug(scope.get('operator'))}-{_stable_hash(identity, 14)}"
    return {
        "schema_version": SCHEMA_VERSION,
        "card_id": card_id,
        "collection": "researcher_findings",
        "kind": "direction",
        "title": finding["title"],
        "specialty": finding["specialty"],
        "bottleneck": finding["bottleneck"],
        "mechanism": finding["mechanism"],
        "expected_upside": finding["expected_upside"],
        "expected_upside_observed": [finding["expected_upside"]]
        if finding["expected_upside"]
        else [],
        "implementation_cost": finding["implementation_cost"],
        "confidence": finding["confidence"],
        "confidence_observed": [finding["confidence"]],
        "kill_criterion": finding["kill_criterion"],
        "rank_score": finding["rank_score"],
        "rank_scores": [finding["rank_score"]],
        "rationale_for_rank": finding["rationale_for_rank"],
        "scope": scope,
        "evidence": finding["evidence"],
        "observation_ids": [finding["observation_id"]],
        "source_runs": [finding["provenance"]["source_run_id"]],
        "support_count": 1,
        "contested": False,
        "contested_observations": [],
        "first_seen": now,
        "last_seen": now,
    }


def _merge_card(
    card: dict[str, Any], finding: dict[str, Any], now: str
) -> tuple[dict[str, Any], bool]:
    observation_id = finding["observation_id"]
    if observation_id in card.get("observation_ids", []):
        return card, False
    card = dict(card)
    card["observation_ids"] = list(card.get("observation_ids", [])) + [observation_id]
    run_id = finding["provenance"]["source_run_id"]
    card["source_runs"] = list(dict.fromkeys(card.get("source_runs", []) + [run_id]))
    card["support_count"] = len(card["observation_ids"])
    card["last_seen"] = now
    card["evidence"] = _merge_evidence(card.get("evidence", []), finding["evidence"])
    ranks = list(card.get("rank_scores", [])) + [finding["rank_score"]]
    card["rank_scores"] = ranks
    card["rank_score"] = round(sum(ranks) / len(ranks), 4)
    confidences = list(card.get("confidence_observed", [])) + [finding["confidence"]]
    card["confidence_observed"] = confidences
    card["confidence"] = _aggregate_confidence(confidences)
    upsides = list(card.get("expected_upside_observed", []))
    if finding["expected_upside"] and finding["expected_upside"] not in upsides:
        upsides.append(finding["expected_upside"])
    card["expected_upside_observed"] = upsides
    return card, True


def _contest_card(
    card: dict[str, Any], rejected: dict[str, Any], now: str
) -> tuple[dict[str, Any], bool]:
    rejection_id = rejected["rejection_id"]
    existing = list(card.get("contested_observations", []))
    if any(item.get("rejection_id") == rejection_id for item in existing):
        return card, False
    card = dict(card)
    existing.append(
        {
            "rejection_id": rejection_id,
            "source_run_id": rejected["run_id"],
            "title": rejected["title"],
            "reason": rejected["reason"],
        }
    )
    card["contested_observations"] = existing
    card["contested"] = True
    card["last_seen"] = now
    return card, True


def _card_markdown(card: dict[str, Any]) -> str:
    scope = card.get("scope", {})
    meta = {
        "schema_version": card.get("schema_version"),
        "card_id": card.get("card_id"),
        "collection": card.get("collection"),
        "operator": scope.get("operator"),
        "language": scope.get("language"),
        "backend": scope.get("backend"),
        "gfx": scope.get("gfx", []),
        "dtypes": scope.get("dtypes", []),
        "regimes": scope.get("regimes", []),
        "bottleneck": scope.get("bottleneck"),
        "confidence": card.get("confidence"),
        "support_count": card.get("support_count"),
        "contested": bool(card.get("contested")),
        "source_runs": card.get("source_runs", []),
        "last_seen": card.get("last_seen"),
    }
    lines = ["---"]
    lines.extend(f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in meta.items())
    lines.extend(
        [
            "---",
            "",
            f"# {card.get('title', '')}",
            "",
            f"- specialty: {card.get('specialty', '')}",
            f"- mechanism: {card.get('mechanism', '')}",
            f"- expected_upside: {card.get('expected_upside', '')}",
            f"- implementation_cost: {card.get('implementation_cost', '')}",
            f"- kill_criterion: {card.get('kill_criterion', '')}",
            f"- researcher_rank: {card.get('rank_score', 0)}",
            f"- contested: {str(bool(card.get('contested'))).lower()}",
            "",
            "## Evidence",
        ]
    )
    for item in card.get("evidence", []):
        title, url, note = item.get("title", ""), item.get("url", ""), item.get("note", "")
        lines.append(f"- {title}" + (f" — {url}" if url else "") + (f" — {note}" if note else ""))
    return "\n".join(lines).rstrip() + "\n"


def _write_card(kb_dir: Path, card: dict[str, Any]) -> Path:
    operator = _slug(card.get("scope", {}).get("operator"))
    base = kb_dir / "cards" / operator / card["card_id"]
    _atomic_write_json(base.with_suffix(".json"), card)
    _atomic_write(base.with_suffix(".md"), _card_markdown(card))
    return base.with_suffix(".json")


def _build_index(kb_dir: Path, cards: list[dict[str, Any]], now: str) -> None:
    ordered = sorted(cards, key=lambda card: card["card_id"])
    index = {
        "schema_version": SCHEMA_VERSION,
        "collection": "researcher_findings",
        "updated_at": now,
        "cards": [
            {
                "card_id": card["card_id"],
                "title": card["title"],
                "specialty": card["specialty"],
                "scope": card["scope"],
                "confidence": card["confidence"],
                "support_count": card["support_count"],
                "contested": bool(card.get("contested")),
                "rank_score": card["rank_score"],
                "path": (
                    Path("cards")
                    / _slug(card["scope"].get("operator"))
                    / f"{card['card_id']}.json"
                ).as_posix(),
            }
            for card in ordered
        ],
    }
    _atomic_write_json(kb_dir / "index.json", index)
    lines = [
        "# Researcher findings — generated index",
        "",
        "These cards are merged immediately from unchanged online Researcher artifacts.",
        "They are advisory findings, not measured truth.",
        "",
    ]
    for item in index["cards"]:
        scope = item["scope"]
        label = " · ".join(
            filter(
                None,
                [
                    scope.get("operator"),
                    scope.get("language"),
                    ",".join(scope.get("gfx", [])),
                    ",".join(scope.get("regimes", [])),
                ],
            )
        )
        md_path = item["path"][:-5] + ".md"
        lines.append(
            f"- [{item['title']}]({md_path}) — {label}; "
            f"confidence={item['confidence']}, observations={item['support_count']}"
            + (", contested" if item.get("contested") else "")
        )
    _atomic_write(kb_dir / "INDEX.md", "\n".join(lines).rstrip() + "\n")


def _publish_snapshot(kb_dir: Path, cards: list[dict[str, Any]], now: str) -> str:
    entries = []
    for card in sorted(cards, key=lambda item: item["card_id"]):
        rel = (
            Path("cards")
            / _slug(card["scope"].get("operator"))
            / f"{card['card_id']}.json"
        )
        entries.append(
            {
                "card_id": card["card_id"],
                "path": rel.as_posix(),
                "sha256": hashlib.sha256(_json_bytes(card)).hexdigest(),
                # Snapshot manifests embed the exact card version. Canonical
                # cards continue to merge in place, while every published
                # snapshot remains reproducible after future weekly updates.
                "card": card,
            }
        )
    snapshot_id = "research-" + _stable_hash(entries, 20)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "created_at": now,
        "cards": entries,
    }
    _atomic_write_json(kb_dir / "snapshots" / f"{snapshot_id}.json", manifest)
    _atomic_write_json(
        kb_dir / "channels" / "latest.json",
        {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id},
    )
    return snapshot_id


def ingest(
    *,
    eval_dir: Path,
    kb_dir: Path,
    kernel_path: Path | None = None,
    operator: str = "",
    language: str = "",
    backend: str = "",
    gfx: str = "",
    dtype: str = "",
    regime: str = "",
    bottleneck: str = "",
    kernel_name: str = "",
) -> dict[str, Any]:
    deep_path = eval_dir / "deep_search.json"
    deep = _read_json(deep_path)
    if not isinstance(deep, dict):
        raise ValueError(f"missing or invalid Researcher artifact: {deep_path}")
    facts_path = eval_dir / "research" / "facts.json"
    facts = _read_json(facts_path, {})
    if not isinstance(facts, dict):
        facts = {}
    artifact_sha = hashlib.sha256(
        deep_path.read_bytes()
        + (facts_path.read_bytes() if facts_path.exists() else b"")
    ).hexdigest()
    run_id = f"{_slug(eval_dir.name, 'run')}-{artifact_sha[:12]}"
    scope = _scope_from_inputs(
        facts,
        operator=operator,
        language=language,
        backend=backend,
        gfx=gfx,
        dtype=dtype,
        regime=regime,
        bottleneck=bottleneck,
        kernel_name=kernel_name,
        kernel_path=kernel_path,
    )
    directions = [
        finding
        for raw in deep.get("directions", [])
        if isinstance(raw, dict)
        for finding in [
            _normalize_direction(
                raw,
                scope=scope,
                run_id=run_id,
                artifact_sha256=artifact_sha,
                eval_dir=eval_dir,
            )
        ]
        if finding is not None
    ]
    rejected = [
        finding
        for raw in deep.get("rejected_directions", [])
        for finding in [_normalize_rejected(raw, scope=scope, run_id=run_id)]
        if finding is not None
    ]
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "artifact_sha256": artifact_sha,
        "scope": scope,
        "directions": directions,
        # Retain these for audit and future card kinds, but only final ranked
        # directions enter planner-visible cards in this first version.
        "open_measurements": deep.get("open_measurements", []),
        "rejected_directions": deep.get("rejected_directions", []),
    }

    now = _utc_now()
    created = merged = unchanged = contested = 0
    touched_card_ids: list[str] = []
    with _exclusive_lock(kb_dir):
        _atomic_write_json(kb_dir / "observations" / f"{run_id}.json", run_record)
        cards_with_paths = _load_cards(kb_dir)
        for finding in directions:
            candidates = [
                (path, card, _finding_similarity(card, finding))
                for path, card in cards_with_paths
                if card.get("specialty") == finding.get("specialty")
                and _scope_compatible(card.get("scope", {}), finding["scope"])
            ]
            best = max(candidates, key=lambda item: item[2], default=None)
            if best is not None and best[2] >= MERGE_THRESHOLD:
                path, card, _ = best
                touched_card_ids.append(card["card_id"])
                updated, changed = _merge_card(card, finding, now)
                if changed:
                    _write_card(kb_dir, updated)
                    cards_with_paths = [
                        (p, updated if p == path else c) for p, c in cards_with_paths
                    ]
                    merged += 1
                else:
                    unchanged += 1
                continue
            card = _new_card(finding, now)
            touched_card_ids.append(card["card_id"])
            path = _write_card(kb_dir, card)
            cards_with_paths.append((path, card))
            created += 1
        # A later online Researcher may reject a mechanism that an earlier run
        # preferred. Keep both observations and mark the canonical card
        # contested; never overwrite or delete either finding.
        for negative in rejected:
            candidates = [
                (path, card, _rejection_similarity(card, negative))
                for path, card in cards_with_paths
                if _scope_compatible(card.get("scope", {}), negative["scope"])
            ]
            best = max(candidates, key=lambda item: item[2], default=None)
            if best is None or best[2] < CONFLICT_THRESHOLD:
                continue
            path, card, _ = best
            touched_card_ids.append(card["card_id"])
            updated, changed = _contest_card(card, negative, now)
            if changed:
                _write_card(kb_dir, updated)
                cards_with_paths = [
                    (p, updated if p == path else c)
                    for p, c in cards_with_paths
                ]
                contested += 1
        cards = [card for _, card in cards_with_paths]
        _build_index(kb_dir, cards, now)
        snapshot_id = _publish_snapshot(kb_dir, cards, now)

    result = {
        "ok": True,
        "mode": "ingest",
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "directions_seen": len(directions),
        "cards_created": created,
        "cards_merged": merged,
        "cards_contested": contested,
        "observations_unchanged": unchanged,
        "card_count": len(cards),
        "card_ids": list(dict.fromkeys(touched_card_ids)),
        "kb_dir": str(kb_dir),
    }
    return result


def _load_snapshot_cards(kb_dir: Path, snapshot_id: str = "") -> tuple[str, list[dict[str, Any]]]:
    if not snapshot_id:
        channel = _read_json(kb_dir / "channels" / "latest.json", {})
        snapshot_id = _clean(channel.get("snapshot_id")) if isinstance(channel, dict) else ""
    if not snapshot_id:
        return "", []
    manifest = _read_json(kb_dir / "snapshots" / f"{snapshot_id}.json", {})
    if not isinstance(manifest, dict):
        return "", []
    cards: list[dict[str, Any]] = []
    for entry in manifest.get("cards", []):
        if not isinstance(entry, dict):
            continue
        card = entry.get("card")
        path = kb_dir / _clean(entry.get("path"))
        if not isinstance(card, dict) and entry.get("path"):
            # Compatibility with early manifests that referenced mutable cards.
            card = _read_json(path)
        if not isinstance(card, dict):
            continue
        actual = hashlib.sha256(_json_bytes(card)).hexdigest()
        if entry.get("sha256") and actual != entry["sha256"]:
            raise ValueError(
                f"snapshot checksum mismatch: {entry.get('card_id') or path}"
            )
        cards.append(card)
    return snapshot_id, cards


def _retrieval_score(card: dict[str, Any], query: dict[str, Any]) -> float | None:
    scope = card.get("scope", {})
    exact_kernel = bool(
        query.get("source_kernel_fingerprint")
        and query["source_kernel_fingerprint"] == scope.get("source_kernel_fingerprint")
    )
    # Exact source identity boosts replay ranking, but it never overrides an
    # explicit operator/language/architecture incompatibility.
    if not _scope_compatible(scope, query, exact_kernel_ok=True):
        return None
    score = 100.0 if exact_kernel else 0.0
    for key, weight in (
        ("operator", 30.0),
        ("kernel_name", 12.0),
        ("language", 10.0),
        ("backend", 4.0),
        ("bottleneck", 8.0),
    ):
        if _known(query.get(key)) and _canonical(query[key]) == _canonical(scope.get(key)):
            score += weight
    for key, weight in (("gfx", 12.0), ("dtypes", 10.0), ("regimes", 10.0)):
        qvals, cvals = set(_as_list(query.get(key))), set(_as_list(scope.get(key)))
        if qvals and cvals and qvals & cvals:
            score += weight
    score += min(10.0, _number(card.get("rank_score")))
    score += min(3.0, math.log2(max(1, int(card.get("support_count", 1)))))
    if card.get("contested"):
        score -= 12.0
    return score


def _render_offline_brief(
    selected: list[tuple[float, dict[str, Any]]], snapshot_id: str
) -> str:
    lines = [
        "# Deep Search Brief — offline Researcher knowledge",
        "",
        f"{len(selected)} directions retrieved from snapshot `{snapshot_id}`. "
        "These are advisory Researcher findings; the planner and measured benchmark remain the judge.",
        "",
    ]
    for index, (_, card) in enumerate(selected, 1):
        lines.extend(
            [
                f"### D{index}: {card.get('title', '')}",
                f"**Specialty:** {card.get('specialty', 'deep_explore')}  ",
                f"**Mechanism:** {card.get('mechanism', '')}  ",
                f"**Expected upside:** {card.get('expected_upside', '')}  ",
                f"**Confidence:** {card.get('confidence', 'medium')}",
            ]
        )
        if card.get("contested"):
            lines.append(
                "**Caution:** Conflicting Researcher observations exist; "
                "treat this mechanism as unresolved and re-measure."
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def retrieve(
    *,
    kb_dir: Path,
    output: Path,
    kernel_path: Path | None = None,
    operator: str = "",
    language: str = "",
    backend: str = "",
    gfx: str = "",
    dtype: str = "",
    regime: str = "",
    bottleneck: str = "",
    kernel_name: str = "",
    snapshot_id: str = "",
    max_directions: int = DEFAULT_MAX_DIRECTIONS,
) -> dict[str, Any]:
    query = _scope_from_inputs(
        {},
        operator=operator,
        language=language,
        backend=backend,
        gfx=gfx,
        dtype=dtype,
        regime=regime,
        bottleneck=bottleneck,
        kernel_name=kernel_name,
        kernel_path=kernel_path,
    )
    resolved_snapshot, cards = _load_snapshot_cards(kb_dir, snapshot_id)
    scored = [
        (score, card)
        for card in cards
        for score in [_retrieval_score(card, query)]
        if score is not None
    ]
    scored.sort(key=lambda item: (-item[0], -_number(item[1].get("rank_score")), item[1]["card_id"]))
    selected = scored[: max(1, max_directions)]
    if not selected:
        return {
            "ok": True,
            "mode": "retrieve",
            "snapshot_id": resolved_snapshot,
            "brief_path": "",
            "cards_retrieved": 0,
            "card_ids": [],
            "query": query,
        }
    _atomic_write(output, _render_offline_brief(selected, resolved_snapshot))
    retrieval_path = output.parent / "research_kb_retrieval.json"
    retrieval = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": resolved_snapshot,
        "query": query,
        "cards": [
            {"card_id": card["card_id"], "score": round(score, 4)}
            for score, card in selected
        ],
        "brief_path": str(output),
    }
    _atomic_write_json(retrieval_path, retrieval)
    return {
        "ok": True,
        "mode": "retrieve",
        "snapshot_id": resolved_snapshot,
        "brief_path": str(output),
        "retrieval_path": str(retrieval_path),
        "cards_retrieved": len(selected),
        "card_ids": [card["card_id"] for _, card in selected],
        "query": query,
    }


def record_validation(
    *,
    kb_dir: Path,
    snapshot_id: str,
    eval_dir: Path,
    kernel_path: Path | None = None,
    kernel_name: str = "",
    dra_mode: str = "",
    card_ids: str = "",
    final_speedup: float = 0.0,
    validation_status: str = "",
    correctness: str = "",
) -> dict[str, Any]:
    cards = [item for item in _as_list(card_ids) if item]
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": "",
        "snapshot_id": _clean(snapshot_id),
        "source_run_id": _clean(eval_dir.name),
        "kernel_name": _canonical(kernel_name) or "unknown",
        "source_kernel_fingerprint": _source_fingerprint(kernel_path),
        "dra_mode": _canonical(dra_mode) or "unknown",
        "card_ids": cards,
        "final_speedup": _number(final_speedup),
        "validation_status": _clean(validation_status),
        "correctness": _clean(correctness),
        "recorded_at": _utc_now(),
    }
    event["event_id"] = "validation-" + _stable_hash(
        {key: value for key, value in event.items() if key not in {"event_id", "recorded_at"}},
        20,
    )
    with _exclusive_lock(kb_dir):
        path = kb_dir / "validation" / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_ids: set[str] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(prior, dict) and prior.get("event_id"):
                    existing_ids.add(prior["event_id"])
        if event["event_id"] not in existing_ids:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            recorded = True
        else:
            recorded = False
    return {
        "ok": True,
        "mode": "validate",
        "validation_event_id": event["event_id"],
        "validation_recorded": recorded,
        "snapshot_id": event["snapshot_id"],
        "card_ids": cards,
        "kb_dir": str(kb_dir),
    }


def compare_online_offline(
    *,
    kb_dir: Path,
    online_json: Path,
    offline_retrieval: Path,
    output: Path | None = None,
    match_threshold: float = 0.55,
) -> dict[str, Any]:
    online = _read_json(online_json, {})
    retrieval = _read_json(offline_retrieval, {})
    if not isinstance(online, dict) or not isinstance(retrieval, dict):
        raise ValueError("online or offline comparison artifact is invalid JSON")
    snapshot_id, snapshot_cards = _load_snapshot_cards(
        kb_dir, _clean(retrieval.get("snapshot_id"))
    )
    wanted = {
        _clean(item.get("card_id"))
        for item in retrieval.get("cards", [])
        if isinstance(item, dict)
    }
    available = [
        card for card in snapshot_cards if not wanted or card.get("card_id") in wanted
    ]
    directions = [
        item for item in online.get("directions", []) if isinstance(item, dict)
    ]
    matches: list[dict[str, Any]] = []
    used: set[str] = set()
    for direction in directions:
        candidates = [
            (_finding_similarity(direction, card), card)
            for card in available
            if card.get("card_id") not in used
        ]
        best = max(candidates, key=lambda item: item[0], default=None)
        if best is None or best[0] < match_threshold:
            matches.append(
                {
                    "online_direction_id": _clean(
                        direction.get("id") or direction.get("direction_id")
                    ),
                    "online_title": _clean(direction.get("title")),
                    "offline_card_id": "",
                    "offline_title": "",
                    "similarity": round(best[0], 4) if best else 0.0,
                    "specialty_match": False,
                }
            )
            continue
        similarity, card = best
        used.add(card["card_id"])
        matches.append(
            {
                "online_direction_id": _clean(
                    direction.get("id") or direction.get("direction_id")
                ),
                "online_title": _clean(direction.get("title")),
                "offline_card_id": card["card_id"],
                "offline_title": card["title"],
                "similarity": round(similarity, 4),
                "specialty_match": _canonical(direction.get("specialty"))
                == _canonical(card.get("specialty")),
            }
        )
    matched = [item for item in matches if item["offline_card_id"]]
    recall = len(matched) / len(directions) if directions else 1.0
    mean_similarity = (
        sum(item["similarity"] for item in matched) / len(matched)
        if matched
        else 0.0
    )
    specialty_agreement = (
        sum(bool(item["specialty_match"]) for item in matched) / len(matched)
        if matched
        else 0.0
    )
    result = {
        "ok": True,
        "mode": "compare",
        "snapshot_id": snapshot_id,
        "online_directions": len(directions),
        "offline_directions": len(available),
        "matched_directions": len(matched),
        "direction_recall": round(recall, 4),
        "mean_mechanism_similarity": round(mean_similarity, 4),
        "specialty_agreement": round(specialty_agreement, 4),
        "equivalent": bool(
            recall >= 0.8
            and mean_similarity >= 0.65
            and specialty_agreement >= 0.8
        ),
        "matches": matches,
    }
    if output is not None:
        _atomic_write_json(output, result)
    return result


def _add_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kernel-path", type=Path)
    parser.add_argument("--operator", default="")
    parser.add_argument("--language", default="")
    parser.add_argument("--backend", default="")
    parser.add_argument("--gfx", default="")
    parser.add_argument("--dtype", default="")
    parser.add_argument("--regime", default="")
    parser.add_argument("--bottleneck", default="")
    parser.add_argument("--kernel-name", default="")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest_parser = sub.add_parser("ingest", help="merge one online Researcher run")
    ingest_parser.add_argument("--eval-dir", type=Path, required=True)
    ingest_parser.add_argument("--kb-dir", type=Path, required=True)
    _add_scope_args(ingest_parser)

    retrieve_parser = sub.add_parser("retrieve", help="materialize an offline planner brief")
    retrieve_parser.add_argument("--kb-dir", type=Path, required=True)
    retrieve_parser.add_argument("--output", type=Path, required=True)
    retrieve_parser.add_argument("--snapshot-id", default="")
    retrieve_parser.add_argument("--max-directions", type=int, default=DEFAULT_MAX_DIRECTIONS)
    _add_scope_args(retrieve_parser)

    validate_parser = sub.add_parser(
        "validate", help="append one Director-validated online/offline outcome"
    )
    validate_parser.add_argument("--kb-dir", type=Path, required=True)
    validate_parser.add_argument("--snapshot-id", default="")
    validate_parser.add_argument("--eval-dir", type=Path, required=True)
    validate_parser.add_argument("--kernel-path", type=Path)
    validate_parser.add_argument("--kernel-name", default="")
    validate_parser.add_argument("--dra-mode", default="")
    validate_parser.add_argument("--card-ids", default="")
    validate_parser.add_argument("--final-speedup", type=float, default=0.0)
    validate_parser.add_argument("--validation-status", default="")
    validate_parser.add_argument("--correctness", default="")

    compare_parser = sub.add_parser(
        "compare", help="compare the online portfolio with an offline retrieval"
    )
    compare_parser.add_argument("--kb-dir", type=Path, required=True)
    compare_parser.add_argument("--online-json", type=Path, required=True)
    compare_parser.add_argument("--offline-retrieval", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.add_argument("--match-threshold", type=float, default=0.55)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    kwargs = vars(args)
    command = kwargs.pop("command")
    try:
        if command == "ingest":
            result = ingest(**kwargs)
        elif command == "retrieve":
            result = retrieve(**kwargs)
        elif command == "validate":
            result = record_validation(**kwargs)
        else:
            result = compare_online_offline(**kwargs)
    except Exception as exc:  # A single compact error is easy for the Workflow agent to relay.
        print(json.dumps({"ok": False, "mode": command, "error": str(exc)}))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
