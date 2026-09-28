# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive a seed contract before any actor process starts.

The official materializer runs only on trusted input files in private scratch
space. It executes no candidate code. The provenance bytes follow the existing
materializer rule for a canonical source without inherited provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path

from .quality_stop_controller import _actor_path, canonical, require, sha

GITIGNORE = b"build/\n__pycache__/\n*.pyc\nresults.*\n*.so\n.torch_ext/\n.rocprofv3/\n*.o\n/.geak/\n"
IGNORE_LINE = "printf '%s\\n' 'build/' '__pycache__/' '*.pyc' 'results.*' '*.so' '.torch_ext/' '.rocprofv3/' '*.o' '/.geak/' > .gitignore"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed_contract(*, task_root, actor_task_root, reference_io_sha256, workflow_root):
    """Return exact file records and metadata bytes without reading actor output."""
    task_root, workflow_root = Path(task_root), Path(workflow_root)
    require(task_root.is_absolute() and task_root.resolve() == task_root and task_root.is_dir(), "startup_task_root_invalid")
    require(workflow_root.is_absolute() and workflow_root.resolve() == workflow_root, "startup_workflow_root_invalid")
    actor_task_root = _actor_path(actor_task_root)
    require(not (task_root / ".geak").exists() and not (task_root / ".geak").is_symlink(), "startup_inherited_metadata_unsupported")
    reference = task_root / "reference_io.pt"
    require(reference.is_file() and not reference.is_symlink() and file_sha256(reference) == reference_io_sha256,
            "startup_reference_identity_changed")
    materializer = workflow_root / "scripts/materialize_workspace.sh"
    sources = [materializer, workflow_root / "scripts/workspace_sources.py", workflow_root / "roles/director.md"]
    require(IGNORE_LINE in sources[-1].read_text(), "startup_gitignore_profile_changed")
    source_bindings = {str(path): file_sha256(path) for path in sources}
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
                   "GEAK_QUALITY_REFERENCE_COPY": "1", "GEAK_QUALITY_REFERENCE_SHA256": reference_io_sha256}
    if "HOME" in os.environ:
        environment["HOME"] = os.environ["HOME"]
    with tempfile.TemporaryDirectory(prefix="geak-quality-seed-") as temporary:
        scratch = Path(temporary)
        workspace = scratch / "workspace"
        try:
            process = subprocess.run(["/bin/bash", str(materializer), "--src", str(task_root), "--dst", str(workspace),
                                      "--shared-root", str(scratch / "shared"), "--link-aiter"],
                                     capture_output=True, stdin=subprocess.DEVNULL, env=environment, timeout=600, check=False)
            require(process.returncode == 0, "startup_materializer_failed")
            (workspace / ".gitignore").write_bytes(GITIGNORE)
            host_metadata = (json.dumps({"ancestors": [str(task_root)]}) + "\n").encode()
            require((workspace / ".geak/workspace.json").read_bytes() == host_metadata, "startup_materializer_metadata_changed")
            records = []
            for directory, folders, names in os.walk(workspace, followlinks=False):
                for name in folders:
                    require(not (Path(directory) / name).is_symlink(), "startup_source_directory_link")
                for name in names:
                    path = Path(directory) / name
                    relative = path.relative_to(workspace).as_posix()
                    if relative == ".geak/workspace.json":
                        continue
                    info = path.lstat()
                    require(path.resolve() == path and stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "startup_source_not_regular")
                    records.append({"path": relative, "mode": "100755" if info.st_mode & 0o111 else "100644",
                                    "sha256": file_sha256(path), "bytes": info.st_size})
            records.sort(key=lambda record: record["path"])
            metadata = (json.dumps({"ancestors": [actor_task_root]}) + "\n").encode()
            metadata_files = {".geak/workspace.json": {"mode": "100644", "sha256": sha(metadata), "bytes": len(metadata)}}
            require(all(file_sha256(Path(path)) == digest for path, digest in source_bindings.items()), "startup_source_changed")
            return {"schema": "geak-quality-startup-v1", "task_root": str(task_root), "actor_task_root": actor_task_root,
                    "expected_seed_files": records, "metadata_files": metadata_files,
                    "metadata_bytes_utf8": {".geak/workspace.json": metadata.decode()},
                    "seed_manifest_sha256": sha(canonical({"files": records, "metadata": metadata_files}).encode("ascii")),
                    "reference_io_sha256": reference_io_sha256, "source_bindings": source_bindings,
                    "materializer_stdout_sha256": sha(process.stdout), "materializer_stderr_sha256": sha(process.stderr),
                    "actor_output_read": False, "candidate_code_executed": False,
                    "shell_environment": {key: environment[key] for key in
                                          ("GEAK_QUALITY_REFERENCE_COPY", "GEAK_QUALITY_REFERENCE_SHA256", "PYTHONDONTWRITEBYTECODE")}}
        finally:
            # Only the private scratch tree needs write permission for cleanup.
            for directory, folders, _names in os.walk(scratch, followlinks=False):
                Path(directory).chmod(stat.S_IMODE(Path(directory).stat().st_mode) | 0o700)
                for name in folders:
                    path = Path(directory) / name
                    if not path.is_symlink():
                        path.chmod(stat.S_IMODE(path.stat().st_mode) | 0o700)
