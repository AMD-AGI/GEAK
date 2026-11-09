# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage complete Python source for the existing GEAK benchmark entry point."""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

try:
    from interface.source_materialization import (
        SourceMaterializationError,
        _read_regular,
        validate_source_materialization,
    )
except ModuleNotFoundError:
    from source_materialization import (
        SourceMaterializationError,
        _read_regular,
        validate_source_materialization,
    )


def _publish_file(path: Path, data: bytes, mode: int = 0o644) -> None:
    """Retain an identical prior asset; never replace a different run asset."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError:
        if _read_regular(path) != data:
            raise SourceMaterializationError("conflicting_staged_asset") from None
    else:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        path.chmod(mode)


def stage_source_bundle(request: dict, eval_dir: Path) -> dict:
    """Copy and validate the declared tree before exposing its per-run request.

    Only source files and the pure verifier are staged. Existing benchmark,
    launcher, scheduler and measurement implementations remain authoritative.
    """
    if (set(request) != {"schema_version", "source_snapshots", "source_materialization"}
            or type(request["schema_version"]) is not int or request["schema_version"] != 1):
        raise SourceMaterializationError("invalid_source_request")
    source = validate_source_materialization(request)
    if source is None:
        raise SourceMaterializationError("empty_source_request")
    eval_dir = eval_dir.resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    parent = eval_dir / "source_materializations"
    parent.mkdir(exist_ok=True)
    if parent.is_symlink():
        raise SourceMaterializationError("unsafe_staging_directory")
    destination = parent / source.manifest_sha256
    staged = copy.deepcopy(request)
    staged["source_materialization"]["bundle_root"] = str(destination)
    if not destination.exists() and not destination.is_symlink():
        with tempfile.TemporaryDirectory(prefix=".materializing-", dir=parent) as temporary:
            tree = Path(temporary) / "bundle"
            tree.mkdir()
            for row in [*source.manifest["files"], {"path": "manifest.json", "mode": 0o644}]:
                target = tree / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                _publish_file(target, _read_regular(source.bundle_root / row["path"]), row["mode"])
            check = copy.deepcopy(staged)
            check["source_materialization"]["bundle_root"] = str(tree)
            validate_source_materialization(check)
            try:
                tree.rename(destination)
            except FileExistsError:
                # Concurrent setup can publish identical content for this run.
                pass
    validate_source_materialization(staged)
    _publish_file(eval_dir / "source_materialization.py",
                  _read_regular(Path(__file__).with_name("source_materialization.py")))
    _publish_file(eval_dir / "source_paths.sh", _read_regular(
        Path(__file__).resolve().parents[1] / "e2e_workflow/scripts/source_paths.sh"))
    _publish_file(eval_dir / "source_request.json",
                  (json.dumps(staged, indent=2, sort_keys=True) + "\n").encode())
    return staged
