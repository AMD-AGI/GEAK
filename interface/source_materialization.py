# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate complete accepted source bundles without importing their code.

Validation establishes content and layer coverage, not serving-process source
engagement. Launchers must separately verify the runtime that uses these roots.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class SourceMaterializationError(ValueError):
    """Required accepted source cannot be reproduced from the handoff."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"unresolved_baseline_source:{reason}")


@dataclass(frozen=True)
class MaterializedSource:
    bundle_root: Path
    manifest_sha256: str
    required_layer_ids: tuple[str, ...]
    pythonpath_prefixes: tuple[str, ...]
    manifest: dict[str, Any]


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise SourceMaterializationError(reason)


def _keys(value: Any, expected: set[str], reason: str) -> None:
    _require(isinstance(value, dict) and set(value) == expected, reason)


def _strings(value: Any, reason: str, *, empty: bool = False) -> list[str]:
    _require(isinstance(value, list), reason)
    _require((empty or bool(value)) and all(
        isinstance(item, str) and bool(item.strip()) and not any(
            ord(char) < 32 for char in item
        ) for item in value
    ), reason)
    _require(len(set(value)) == len(value), reason)
    return value


def _relative(value: Any) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), "invalid_relative_path")
    _require(not any(char in value for char in "\\:\x00\n\r"), "invalid_relative_path")
    _require(all(part not in {"", ".", ".."} for part in value.split("/")),
             "invalid_relative_path")
    path = PurePosixPath(value)
    _require(not path.is_absolute(), "invalid_relative_path")
    return path


def _sha(value: Any, reason: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             reason)
    return value


def _path(root: Path, relative: Any, *, directory: bool = False,
          absent: bool = False) -> Path:
    rel = _relative(relative)
    current = root
    for index, part in enumerate(rel.parts):
        current = current / part
        last = index == len(rel.parts) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            if absent:
                return root / str(rel)
            raise SourceMaterializationError("missing_bundle_path") from None
        _require(not stat.S_ISLNK(info.st_mode), "symlink_in_bundle")
        if not last or directory:
            _require(stat.S_ISDIR(info.st_mode), "non_directory_bundle_path")
        elif absent:
            raise SourceMaterializationError("deleted_path_present")
        else:
            _require(stat.S_ISREG(info.st_mode), "non_regular_bundle_file")
    return current


def _read_regular(path: Path, *, limit: int | None = None) -> bytes:
    # O_NONBLOCK prevents a changed FIFO from hanging between lstat and open.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode), "non_regular_bundle_file")
        _require(limit is None or info.st_size <= limit, "manifest_too_large")
        data = stream.read() if limit is None else stream.read(limit + 1)
        _require(limit is None or len(data) <= limit, "manifest_too_large")
        return data


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate_manifest_key")
        result[key] = value
    return result


def _under(path: str, roots: list[str]) -> str:
    matches = [root for root in roots if path.startswith(root + "/")]
    _require(len(matches) == 1, "ambiguous_or_uncovered_path")
    return matches[0]


def _module(path: str, prefixes: list[str]) -> str:
    prefix = _under(path, prefixes)
    relative = path[len(prefix) + 1:]
    _require(relative.endswith(".py"), "non_python_module")
    parts = relative[:-3].split("/")
    if parts[-1] == "__init__":
        parts.pop()
    _require(bool(parts) and all(part.isidentifier() for part in parts),
             "invalid_module_name")
    return ".".join(parts)


def validate_source_materialization(baseline: dict[str, Any]) -> MaterializedSource | None:
    """Verify a v1 bundle and every required layer; no-source is a no-op.

    Raw sparse snapshots cannot establish coverage. A producer must provide a
    complete verified bundle even when the original accepted tree still exists.
    """
    snapshots = baseline.get("source_snapshots")
    if snapshots is None:
        snapshots = []
    _require(isinstance(snapshots, list), "invalid_source_layers")
    descriptor = baseline.get("source_materialization")
    if not snapshots and descriptor is None:
        return None
    try:
        return _validate(snapshots, descriptor)
    except SourceMaterializationError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SourceMaterializationError("unreadable_or_invalid_bundle") from exc


def read_source_request(path: str) -> MaterializedSource:
    """Revalidate the per-run source request before each serving launch."""
    try:
        request = json.loads(_read_regular(Path(path), limit=1024 * 1024),
                             object_pairs_hook=_json_object)
        _keys(request, {"schema_version", "source_snapshots", "source_materialization"},
              "invalid_source_request")
        _require(type(request["schema_version"]) is int and request["schema_version"] == 1,
                 "unsupported_source_request")
        source = validate_source_materialization(request)
        _require(source is not None, "empty_source_request")
        return source
    except SourceMaterializationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise SourceMaterializationError("unreadable_source_request") from exc


def compose_pythonpath(request: str, overlay: str, backend_defaults: str,
                       inherited: str, bootstrap: str = "") -> str:
    """Compose authored overlay, accepted source, defaults and inherited paths.

    An optional verification bootstrap is first and contains no framework
    packages. Empty path components are omitted so they cannot insert the CWD.
    """
    source = read_source_request(request)
    paths = []
    for group in (bootstrap, overlay, *source.pythonpath_prefixes, backend_defaults, inherited):
        for path in group.split(os.pathsep):
            if path and path not in paths:
                paths.append(path)
    return os.pathsep.join(paths)


def _validate(snapshots: Any, descriptor: Any) -> MaterializedSource:
    _require(isinstance(snapshots, list) and bool(snapshots), "missing_source_layers")
    _require(all(isinstance(row, dict) for row in snapshots), "invalid_source_layers")
    required = _strings([row.get("id") for row in snapshots], "invalid_layer_ids")
    _keys(descriptor, {"schema_version", "status", "bundle_root", "manifest_path",
                       "manifest_sha256", "required_layer_ids", "pythonpath_prefixes"},
          "missing_or_invalid_descriptor")
    _require(type(descriptor["schema_version"]) is int and descriptor["schema_version"] == 1,
             "unsupported_schema_version")
    _require(descriptor["status"] == "ready", "materialization_not_ready")
    _require(descriptor["required_layer_ids"] == required, "layer_coverage_mismatch")
    _require(descriptor["manifest_path"] == "manifest.json", "invalid_manifest_path")
    root_name = descriptor["bundle_root"]
    _require(isinstance(root_name, str) and Path(root_name).is_absolute(), "invalid_bundle_root")
    _require(not any(char in root_name for char in "\x00\n\r:"), "invalid_bundle_root")
    root = Path(root_name)
    _require(stat.S_ISDIR(root.lstat().st_mode), "invalid_bundle_root")
    _require(root.resolve() == root, "noncanonical_bundle_root")
    sha = _sha(descriptor["manifest_sha256"], "invalid_manifest_digest")
    raw = _read_regular(_path(root, "manifest.json"), limit=32 * 1024 * 1024)
    _require(hashlib.sha256(raw).hexdigest() == sha, "manifest_digest_mismatch")
    manifest = json.loads(raw, object_pairs_hook=_json_object)
    _keys(manifest, {"schema_version", "required_layer_ids", "pythonpath_prefixes",
                     "trees", "files", "deleted_paths", "modules", "deleted_modules"},
          "invalid_manifest")
    _require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
             "unsupported_manifest_version")
    _require(manifest["required_layer_ids"] == required, "layer_coverage_mismatch")
    prefixes = _strings(manifest["pythonpath_prefixes"], "invalid_import_prefixes")
    _require(prefixes == descriptor["pythonpath_prefixes"], "import_prefix_mismatch")
    trees = manifest["trees"]
    _require(isinstance(trees, list) and bool(trees), "invalid_trees")
    roots: list[str] = []
    tree_ids: list[str] = []
    layers: list[str] = []
    first_positions: list[int] = []
    for tree in trees:
        _keys(tree, {"tree_id", "root", "accepted_commit", "layer_ids"}, "invalid_tree")
        tree_ids.append(tree["tree_id"])
        path = str(_relative(tree["root"]))
        _require(not any(path == old or path.startswith(old + "/") or
                         old.startswith(path + "/") for old in roots), "overlapping_trees")
        roots.append(path)
        _path(root, path, directory=True)
        commit = tree["accepted_commit"]
        _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)
                 is not None, "invalid_accepted_commit")
        owned = _strings(tree["layer_ids"], "invalid_tree_layers")
        _require(all(item in required for item in owned), "layer_coverage_mismatch")
        _require(owned == [item for item in required if item in owned], "layer_order_mismatch")
        first_positions.append(required.index(owned[0]))
        layers.extend(owned)
    _strings(tree_ids, "invalid_tree_ids")
    _require(first_positions == sorted(first_positions), "tree_order_mismatch")
    _require(len(layers) == len(set(layers)) and set(layers) == set(required),
             "layer_coverage_mismatch")
    for prefix in prefixes:
        _relative(prefix)
        _require(prefix in roots or bool(_under(prefix, roots)), "uncovered_import_prefix")
        _path(root, prefix, directory=True)
    for tree in roots:
        _require(any(prefix == tree or prefix.startswith(tree + "/") for prefix in prefixes),
                 "tree_without_import_prefix")
    _require(not any(a != b and a.startswith(b + "/") for a in prefixes for b in prefixes),
             "overlapping_import_prefixes")

    rows = manifest["files"]
    _require(isinstance(rows, list) and bool(rows), "invalid_file_inventory")
    files: dict[str, dict[str, Any]] = {}
    for row in rows:
        _keys(row, {"path", "sha256", "mode"}, "invalid_file_row")
        name = str(_relative(row["path"]))
        _under(name, roots)
        _require(name not in files, "duplicate_file_row")
        mode = row["mode"]
        _require(type(mode) is int and 0 <= mode <= 0o777, "invalid_file_mode")
        path = _path(root, name)
        _require(stat.S_IMODE(path.stat().st_mode) == mode, "file_mode_mismatch")
        _require(hashlib.sha256(_read_regular(path)).hexdigest() ==
                 _sha(row["sha256"], "invalid_file_digest"), "file_digest_mismatch")
        files[name] = row
    inventory: set[str] = set()
    expected_dirs = {str(parent) for name in files for parent in PurePosixPath(name).parents
                     if str(parent) != "."}
    for tree in roots:
        for parent, dirs, names in os.walk(root / tree, followlinks=False):
            for name in dirs:
                _require(not (Path(parent) / name).is_symlink(), "symlink_in_bundle")
                relative = (Path(parent) / name).relative_to(root).as_posix()
                _require(relative in expected_dirs, "unlisted_bundle_directory")
            for name in names:
                relative = (Path(parent) / name).relative_to(root).as_posix()
                _path(root, relative)
                inventory.add(relative)
    _require(inventory == set(files), "incomplete_file_inventory")
    import_owners: dict[str, str] = {}
    for prefix in prefixes:
        for child in (root / prefix).iterdir():
            # Directories can be namespace packages, including an otherwise
            # ancillary tests/ tree. A second root must not change an unchanged
            # dependency's meaning while touched module origins still match.
            candidate = child.name if child.is_dir() else child.stem if child.suffix == ".py" else ""
            if candidate.isidentifier():
                _require(import_owners.setdefault(candidate, prefix) == prefix,
                         "ambiguous_import_ownership")
    deleted = _strings(manifest["deleted_paths"], "invalid_deleted_paths", empty=True)
    for name in deleted:
        _relative(name)
        _under(name, roots)
        _require(not name.endswith("/__init__.py"), "unsupported_package_deletion")
        _require(name not in files, "conflicting_deletion")
        _path(root, name, absent=True)
    modules = manifest["modules"]
    _require(isinstance(modules, list), "invalid_modules")
    module_names: list[str] = []
    for row in modules:
        _keys(row, {"name", "path"}, "invalid_module_row")
        _require(row["path"] in files, "module_not_in_inventory")
        _require(row["name"] == _module(row["path"], prefixes), "module_name_mismatch")
        module_names.append(row["name"])
    _strings(module_names, "duplicate_modules", empty=True)
    deleted_modules = _strings(manifest["deleted_modules"], "invalid_deleted_modules", empty=True)
    possible_deleted = {_module(path, prefixes) for path in deleted if path.endswith(".py")}
    _require(set(deleted_modules) == possible_deleted, "deleted_module_coverage_mismatch")
    _require(not set(deleted_modules).intersection(module_names), "conflicting_module_deletion")
    # Validate the owned packages, including all their unchanged siblings.
    # Complete repository trees can also have unrelated setup/tests/tooling at
    # their import root; those are inventoried, not treated as served packages.
    owners: dict[str, str] = {}
    source_paths = [row["path"] for row in modules]
    source_paths.extend(path for path in deleted if path.endswith(".py"))
    for tree in roots:
        _require(any(path.startswith(tree + "/") for path in source_paths),
                 "tree_without_source_modules")
    for name in source_paths:
        prefix = _under(name, prefixes)
        parts = name[len(prefix) + 1:].split("/")
        _require(len(parts) > 1, "unsupported_top_level_module")
        owner = parts[0]
        _require(owners.setdefault(owner, prefix) == prefix, "ambiguous_package_ownership")
        _require(prefix + "/" + owner + "/__init__.py" in files,
                 "unsupported_namespace_package")
        for count in range(1, len(parts)):
            initializer = prefix + "/" + "/".join(parts[:count]) + "/__init__.py"
            _require(initializer in files, "unsupported_namespace_package")
        if name.endswith("/__init__.py"):
            alternate = name[:-len("/__init__.py")] + ".py"
        else:
            alternate = name[:-3]
        _require(not (root / alternate).exists(), "ambiguous_module_ownership")
        for other in prefixes:
            if other != prefix:
                candidates = (root / other / owner, root / other / (owner + ".py"))
                _require(not any(path.exists() for path in candidates), "ambiguous_package_ownership")
    for name in files:
        matches = [prefix for prefix in prefixes if name.startswith(prefix + "/")]
        if not matches:
            continue
        prefix = matches[0]
        relative = name[len(prefix) + 1:]
        parts = relative.split("/")
        _require(parts[0].split(".")[0] not in {"sitecustomize", "usercustomize"},
                 "unsupported_startup_hook")
        if owners.get(parts[0]) != prefix:
            continue
        if not name.endswith(".py"):
            runtime_suffixes = {".pyc", ".pyo", ".so", ".pyd", ".dylib",
                                ".dll", ".a", ".o", ".co", ".hsaco"}
            _require(not runtime_suffixes.intersection(PurePosixPath(name.lower()).suffixes),
                     "unsupported_runtime_artifact")
            continue
        for count in range(1, len(parts)):
            initializer = prefix + "/" + "/".join(parts[:count]) + "/__init__.py"
            _require(initializer in files, "unsupported_namespace_package")
    for name in deleted_modules:
        _require("." in name and name.split(".")[0] in owners, "unsupported_package_deletion")
    return MaterializedSource(root, sha, tuple(required),
                              tuple(str(root / prefix) for prefix in prefixes), manifest)


if __name__ == "__main__":
    if len(sys.argv) != 7 or sys.argv[1] != "--compose-pythonpath":
        raise SystemExit("usage: source_materialization.py --compose-pythonpath REQUEST OVERLAY DEFAULTS INHERITED BOOTSTRAP")
    try:
        print(compose_pythonpath(*sys.argv[2:]))
    except SourceMaterializationError as error:
        sys.stderr.write(str(error) + "\n")
        raise SystemExit(2)
