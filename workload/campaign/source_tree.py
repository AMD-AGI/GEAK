#!/usr/bin/env python3
"""Source paths shared by the optimizer worker and cold verifier.

Snapshots contain the complete source tree, excluding interpreter caches. Editable
metadata controls permissions, not which computation survives KEEP or rollback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile


def files_equal(a, b):
    # Read bytes each time: filecmp caches by size/mtime, which can be preserved by
    # source restoration between the pre-run and post-run integrity checks.
    with open(a, "rb") as left, open(b, "rb") as right:
        while True:
            chunk = left.read(1 << 20)
            if chunk != right.read(1 << 20):
                return False
            if not chunk:
                return True


def source_files(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"not a regular source directory: {root}")
    files = {}
    for parent, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in dirs + names:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError(f"symlink in source tree: {path}")
        for name in sorted(names):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = Path(parent) / name
            if not path.is_file():
                raise ValueError(f"non-regular source: {path}")
            files[path.relative_to(root).as_posix()] = path
    return dict(sorted(files.items()))


def source_hashes(root):
    """Identity of the complete computation tree, using the snapshot's file policy."""
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_files(root).items()}


def editable_sources(task):
    task = Path(task)
    meta_path = task / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    leaves = []
    for section in ("kernel_extraction", "candidate_bind"):
        obj = meta.get(section) or {}
        if not isinstance(obj, dict):
            raise ValueError(f"{section} must be an object")
        values = obj.get("editable_leaves") or []
        leaves.extend(values if isinstance(values, list) else [values])
    if not leaves:
        value = (meta.get("candidate_bind") or {}).get("file")
        if value:
            leaves = [value]
    files = source_files(task / "kernel_src")
    if not leaves:
        # Legacy tasks without declarations retain the old flat-file preference.
        names = [n for n in files if "/" not in n and not n.startswith("__")
                 and n != "cand_bind.py"]
        leaves = (sorted(n for n in names if n.endswith(".py")) or names)[:1]
    result = []
    for leaf in leaves:
        if isinstance(leaf, dict):
            # Extraction contracts use both spellings for the same source path.
            # Reject ambiguous aliases instead of silently widening editable scope.
            value = leaf.get("source_file", leaf.get("file"))
            if "source_file" in leaf and "file" in leaf and (
                not isinstance(value, str) or not isinstance(leaf["file"], str)
                or value.removeprefix("kernel_src/") != leaf["file"].removeprefix("kernel_src/")
            ):
                raise ValueError(f"conflicting editable source paths: {leaf!r}")
        else:
            value = leaf
        if not isinstance(value, str) or not value or any(c.isspace() for c in value):
            raise ValueError(f"invalid editable source declaration: {leaf!r}")
        path = PurePosixPath(value)
        if (path.is_absolute() or "\\" in value
                or any(p in ("", ".", "..") for p in value.split("/"))):
            raise ValueError(f"unsafe editable source path: {value}")
        name = value.removeprefix("kernel_src/")
        if name == "cand_bind.py" or name not in files:
            raise ValueError(f"declared editable source is missing or scaffolding: {value}")
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("kernel_src/ holds no declared editable file")
    return result


def copy_source_tree(src, dst):
    """Replace a snapshot/working tree; remove stale files and interpreter caches."""
    src, dst = Path(src), Path(dst)
    files = source_files(src)
    if dst.is_symlink() or src.resolve() == dst.resolve():
        raise ValueError(f"invalid source copy destination: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".source-copy-", dir=dst.parent))
    try:
        for name, path in files.items():
            target = temp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        if dst.exists():
            shutil.rmtree(dst)
        temp.rename(dst)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def changed_sources(before, after):
    a, b = source_files(before), source_files(after)
    return [name for name in sorted(a.keys() | b.keys())
            if name not in a or name not in b
            or not files_equal(a[name], b[name])]


def freeze_harness(task, destination):
    """Keep the declared harness files once, across attempts and worker resumes."""
    task, destination = Path(task), Path(destination)
    if destination.exists():
        source_files(destination)  # Validate the saved snapshot without replacing it.
        return
    meta = json.loads((task / "meta.json").read_text()) if (task / "meta.json").exists() else {}
    names = {"meta.json", "unittest.py", "cases.py", "harness_lib.py",
             "leg_runner.py", "overlay_setup.py"}
    declared = (meta.get("kernel_extraction") or {}).get("source_delivery_files") or []
    for name in declared:
        if (not isinstance(name, str) or PurePosixPath(name).is_absolute()
                or "\\" in name or any(p in ("", ".", "..") for p in name.split("/"))):
            raise ValueError(f"invalid source delivery path: {name!r}")
        if not name.startswith("kernel_src/"):
            names.add(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".frozen-harness-", dir=destination.parent))
    try:
        for name in sorted(names):
            source = task / name
            if not source.exists():
                if name in declared:
                    raise ValueError(f"missing declared harness file: {name}")
                continue
            if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(task.resolve()):
                raise ValueError(f"invalid harness file: {source}")
            target = temp / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        temp.rename(destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def changed_harness(task, snapshot):
    task = Path(task)
    changed = []
    for name, reference in source_files(snapshot).items():
        current = task / name
        if (current.is_symlink() or not current.is_file()
                or not current.resolve().is_relative_to(task.resolve())
                or not files_equal(reference, current)):
            changed.append(name)
    return changed


def restore_harness(task, snapshot):
    task = Path(task)
    for name, reference in source_files(snapshot).items():
        target = task / name
        if not target.parent.resolve().is_relative_to(task.resolve()):
            raise ValueError(f"harness destination escaped task: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        shutil.copy2(reference, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("editable", "hashes", "copy", "check", "freeze", "check-frozen", "restore-frozen"))
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--allowed", default="")
    args = parser.parse_args()
    try:
        if args.action == "editable":
            print(" ".join(editable_sources(args.paths[0])))
        elif args.action == "hashes":
            print(json.dumps(source_hashes(args.paths[0]), indent=2))
        elif args.action == "copy":
            copy_source_tree(*args.paths)
        elif args.action == "freeze":
            freeze_harness(*args.paths)
        elif args.action == "restore-frozen":
            restore_harness(*args.paths)
        elif args.action == "check-frozen":
            changes = changed_harness(*args.paths)
            if changes:
                print("frozen harness changed: " + ", ".join(changes))
                return 2
        else:
            changes = changed_sources(*args.paths)
            allowed = set(args.allowed.split())
            missing = allowed - source_files(args.paths[1]).keys()
            forbidden = set(changes) - allowed
            if forbidden or missing:
                print("invalid source edits: " + ", ".join(sorted(forbidden | missing)))
                return 2
            print(" ".join(changes))
            return 1 if changes else 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"source tree error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
