#!/usr/bin/env python3
"""Integrity + re-sync tool for the VENDORED tuning skillset.

The skillset is vendored into GEAK as ONE INTACT TREE (`<repo>/tuning_skillset/`), byte-identical to
the upstream standalone repo. It is deliberately NOT decomposed into GEAK's own knowledge/ files: it is
developed and validated standalone (`validate/claims.py`, the per-skill SKILL.md set), and that
validation only transfers to GEAK if the copy GEAK runs is the copy that was validated.

This tool is the enforcement point:

    --verify   recompute every file hash and compare against the recorded manifest (CI / preflight)
    --update   re-record the manifest from the current vendored tree (after an intentional re-sync)
    --sync SRC copy an upstream skillset tree over the vendored one, then re-record the manifest

Stdlib only; no GPU, no network. Safe to run anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys

# <repo>/e2e_workflow/scripts/tuning_skillset_sync.py -> <repo>
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SKILLSET_DIR = os.path.join(REPO_ROOT, "tuning_skillset")
DEFAULT_MANIFEST = os.path.join(
    REPO_ROOT, "e2e_workflow", "knowledge", "tuning_skillset.manifest.sha256"
)

# Build/cache noise never belongs in the vendored tree or the manifest.
EXCLUDE_DIRS = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo")


def iter_files(root: str):
    """Yield repo-relative POSIX paths of every vendored file, sorted and cache-free."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for name in sorted(filenames):
            if name.endswith(EXCLUDE_SUFFIXES):
                continue
            abs_path = os.path.join(dirpath, name)
            out.append(os.path.relpath(abs_path, root).replace(os.sep, "/"))
    return sorted(out)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(skillset_dir: str) -> dict:
    return {rel: sha256_file(os.path.join(skillset_dir, rel)) for rel in iter_files(skillset_dir)}


def read_manifest(path: str) -> dict:
    manifest = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, _, rel = line.partition("  ")
            if not rel:
                raise ValueError(f"malformed manifest line: {line!r}")
            manifest[rel] = digest
    return manifest


def write_manifest(path: str, manifest: dict, skillset_dir: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rel_dir = os.path.relpath(skillset_dir, REPO_ROOT).replace(os.sep, "/")
    lines = [
        "# sha256 manifest of the VENDORED tuning skillset — do not hand-edit.",
        f"# tree: <repo>/{rel_dir}/   files: {len(manifest)}",
        "# regenerate: python3 e2e_workflow/scripts/tuning_skillset_sync.py --update",
        "# verify:     python3 e2e_workflow/scripts/tuning_skillset_sync.py --verify",
        "#",
        "# The tree is vendored WHOLE and UNMODIFIED from the standalone skillset repo so that the",
        "# standalone validation (validate/claims.py + the SKILL.md set) applies to what GEAK runs.",
        "# A mismatch here means the vendored copy drifted: re-sync it, never patch it in place.",
    ]
    lines += [f"{digest}  {rel}" for rel, digest in sorted(manifest.items())]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def verify(skillset_dir: str, manifest_path: str) -> int:
    if not os.path.isdir(skillset_dir):
        print(f"FAIL: vendored skillset missing: {skillset_dir}", file=sys.stderr)
        return 1
    if not os.path.isfile(manifest_path):
        print(f"FAIL: manifest missing: {manifest_path}", file=sys.stderr)
        return 1

    expected = read_manifest(manifest_path)
    actual = build_manifest(skillset_dir)

    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = sorted(r for r in set(expected) & set(actual) if expected[r] != actual[r])

    for rel in missing:
        print(f"MISSING  {rel}")
    for rel in added:
        print(f"UNTRACKED {rel}")
    for rel in changed:
        print(f"MODIFIED {rel}")

    if missing or added or changed:
        print(
            f"\nFAIL: vendored skillset drifted from the manifest "
            f"({len(missing)} missing, {len(added)} untracked, {len(changed)} modified).\n"
            "The skillset is vendored WHOLE and validated standalone — re-sync it with\n"
            "  --sync <upstream_tuning_skillset_dir>\n"
            "rather than editing files inside the vendored tree.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: vendored skillset matches the manifest ({len(actual)} files).")
    return 0


def sync(src: str, skillset_dir: str) -> None:
    src = os.path.abspath(src)
    if not os.path.isdir(src):
        raise SystemExit(f"--sync source is not a directory: {src}")
    if not os.path.isfile(os.path.join(src, "README.md")):
        raise SystemExit(f"--sync source does not look like a tuning skillset (no README.md): {src}")

    if os.path.isdir(skillset_dir):
        shutil.rmtree(skillset_dir)
    shutil.copytree(
        src,
        skillset_dir,
        ignore=shutil.ignore_patterns(*EXCLUDE_DIRS, "*.pyc", "*.pyo"),
    )
    print(f"Synced {src} -> {skillset_dir} ({len(iter_files(skillset_dir))} files).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skillset-dir", default=DEFAULT_SKILLSET_DIR, help="vendored tree (default: <repo>/tuning_skillset)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="manifest path")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="check the vendored tree against the manifest (default)")
    mode.add_argument("--update", action="store_true", help="re-record the manifest from the vendored tree")
    mode.add_argument("--sync", metavar="SRC", help="copy an upstream skillset tree in, then re-record the manifest")
    args = ap.parse_args(argv)

    if args.sync:
        sync(args.sync, args.skillset_dir)
        write_manifest(args.manifest, build_manifest(args.skillset_dir), args.skillset_dir)
        print(f"Manifest written: {args.manifest}")
        return 0

    if args.update:
        manifest = build_manifest(args.skillset_dir)
        write_manifest(args.manifest, manifest, args.skillset_dir)
        print(f"Manifest written: {args.manifest} ({len(manifest)} files).")
        return 0

    return verify(args.skillset_dir, args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
