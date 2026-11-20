#!/usr/bin/env python3
"""Build a content-addressed skill index and detect shadowing conflicts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "kernel_opt.snapshot_skill_index/1"
RUNTIME_INDEX_SCHEMA = "toolctl.skill-index/1"
SKIP_PARTS = {"__pycache__", ".git", ".nfs"}


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(
            part in SKIP_PARTS or part.startswith(".nfs") for part in path.parts
        ):
            continue
        rel = path.relative_to(root).as_posix()
        content = path.read_bytes()
        digest.update(f"{rel}\0{hashlib.sha256(content).hexdigest()}\0{len(content)}\n".encode())
    return digest.hexdigest()


def _frontmatter_name(path: Path) -> str | None:
    try:
        text = path.read_text(errors="strict")
    except OSError:
        return None
    if not text.startswith("---\n"):
        return None
    for line in text.split("---\n", 2)[1].splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "name" and value.strip():
            return value.strip().strip("\"'")
    return None


def _identity(skill_dir: Path) -> tuple[str, str | None]:
    runtime_index = skill_dir / "runtime" / "skill-index.json"
    if runtime_index.is_file():
        doc = json.loads(runtime_index.read_text())
        skill = doc.get("skill") if isinstance(doc, dict) else None
        if (
            doc.get("schema") != RUNTIME_INDEX_SCHEMA
            or not isinstance(skill, dict)
            or not isinstance(skill.get("id"), str)
            or not skill["id"]
        ):
            raise ValueError(f"invalid runtime skill index: {runtime_index}")
        version = skill.get("version")
        return skill["id"], version if isinstance(version, str) else None
    name = _frontmatter_name(skill_dir / "SKILL.md")
    if not name:
        raise ValueError(f"skill has no runtime index or frontmatter name: {skill_dir}")
    return name, None


def _is_skill_dir(path: Path) -> bool:
    # A skill dir carries SKILL.md (upstream layout) or a runtime skill index. GEAK carries this
    # pack as perf_knowledge/expert_skills/skills/gluon_authoring with its entry folded into
    # skill.md (GEAK's expert-skill entry, no SKILL.md), so the runtime index is what identifies
    # it there (_identity already reads the index first).
    return (path / "SKILL.md").is_file() or (path / "runtime" / "skill-index.json").is_file()


def _skill_dirs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        return []
    if _is_skill_dir(root):
        return [root]
    container = root / "skills" if (root / "skills").is_dir() else root
    return sorted(path for path in container.iterdir() if path.is_dir() and _is_skill_dir(path))


def _entries(root: Path, source: str) -> list[dict[str, Any]]:
    entries = []
    for skill_dir in _skill_dirs(root):
        skill_id, version = _identity(skill_dir)
        entries.append({
            "skill_id": skill_id,
            "version": version,
            "sha256": _tree_hash(skill_dir),
            "path": str(skill_dir),
            "source": source,
        })
    return entries


def configured_user_roots(
    explicit: Iterable[Path] = (),
    *,
    use_claude_config: bool = False,
    environ: dict[str, str] | None = None,
) -> tuple[list[Path], list[str]]:
    roots = [Path(path) for path in explicit]
    warnings: list[str] = []
    if use_claude_config:
        config = (environ or os.environ).get("CLAUDE_CONFIG_DIR")
        if config:
            roots.append(Path(config) / "skills")
        else:
            warnings.append(
                "CLAUDE_CONFIG_DIR is unset; user skill root remains unknown unless supplied explicitly"
            )
    unique: list[Path] = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique, warnings


def build_snapshot_skill_index(
    skill_root: Path,
    *,
    user_skill_roots: Iterable[Path] = (),
    use_claude_config: bool = False,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    primary = _entries(skill_root, "snapshot")
    if not primary:
        raise ValueError(f"no skills found below {skill_root}")
    roots, warnings = configured_user_roots(
        user_skill_roots, use_claude_config=use_claude_config, environ=environ
    )
    users = [
        entry
        for root in roots
        for entry in _entries(root, f"user:{root}")
    ]
    seen: dict[str, dict[str, Any]] = {}
    for entry in primary:
        previous = seen.get(entry["skill_id"])
        if previous:
            raise ValueError(
                f"snapshot has duplicate skill_id {entry['skill_id']!r}: "
                f"{previous['path']} and {entry['path']}"
            )
        seen[entry["skill_id"]] = entry
    conflicts = []
    for entry in users:
        previous = seen.get(entry["skill_id"])
        if previous is None:
            continue
        relation = "same_hash" if previous["sha256"] == entry["sha256"] else "different_hash"
        conflict = {
            "skill_id": entry["skill_id"],
            "snapshot": previous,
            "user": entry,
            "relation": relation,
        }
        conflicts.append(conflict)
        if relation == "same_hash":
            warnings.append(
                f"user skill {entry['skill_id']!r} duplicates the snapshot with the same hash"
            )
        else:
            raise ValueError(
                f"skill conflict for {entry['skill_id']!r}: snapshot hash "
                f"{previous['sha256']} differs from user hash {entry['sha256']}"
            )
    return {
        "schema": SCHEMA,
        "skills": primary,
        "user_roots": [str(root) for root in roots],
        "conflicts": conflicts,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", required=True, type=Path)
    parser.add_argument("--user-skill-root", action="append", default=[], type=Path)
    parser.add_argument("--check-claude-config", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_snapshot_skill_index(
            args.skill_root,
            user_skill_roots=args.user_skill_root,
            use_claude_config=args.check_claude_config,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[skill_index] ERROR: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
