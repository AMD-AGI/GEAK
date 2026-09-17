# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage reproducible accepted source and matching launch helpers per GEAK run."""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import os
import shlex
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

try:
    from interface.source_materialization import (
        SourceMaterializationError,
        _read_regular,
        _relative,
        read_source_request,
        validate_source_materialization,
    )
except ModuleNotFoundError:
    from source_materialization import (
        SourceMaterializationError,
        _read_regular,
        _relative,
        read_source_request,
        validate_source_materialization,
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
    )


def _directory_fd(path: Path, *, create: bool = False) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise SourceMaterializationError("unsafe_staging_directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(path.anchor, flags)
    complete = False
    try:
        for name in path.parts[1:]:
            try:
                child = os.open(name, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(name, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(name, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        complete = True
        return fd
    finally:
        if not complete:
            os.close(fd)


@contextmanager
def _opened_directory(path: Path, *, create: bool = False):
    fd = _directory_fd(path, create=create)
    try:
        yield fd
    finally:
        os.close(fd)


def _same_directory(path: Path, fd: int) -> None:
    with _opened_directory(path) as current:
        if (os.fstat(current).st_dev, os.fstat(current).st_ino) != (
            os.fstat(fd).st_dev,
            os.fstat(fd).st_ino,
        ):
            raise SourceMaterializationError("staging_directory_changed")


def _directory(path: Path) -> None:
    with _opened_directory(path, create=True) as fd:
        _same_directory(path, fd)


def _target(root: Path, relative: str) -> Path:
    parts = _relative(relative).parts
    parent = root
    for part in parts[:-1]:
        parent = parent / part
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise SourceMaterializationError("unsafe_staging_directory")
    return root / relative


def _check_file(path: Path, data: bytes, mode: int) -> bool:
    try:
        with _opened_directory(path.parent) as fd:
            present = _check_at(fd, path.name, data, mode)
            _same_directory(path.parent, fd)
            return present
    except FileNotFoundError:
        return False


def _read_asset(path: Path) -> tuple[bytes, int]:
    path = path.absolute()
    with _opened_directory(path.parent) as parent:
        fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise SourceMaterializationError("non_regular_source_helper")
            data = stream.read()
        after = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if _file_identity(info) != _file_identity(after):
            raise SourceMaterializationError("source_helper_changed")
        _same_directory(path.parent, parent)
        return data, stat.S_IMODE(info.st_mode)


def _check_at(parent: int, name: str, data: bytes, mode: int) -> bool:
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(info.st_mode):
        raise SourceMaterializationError("conflicting_staged_asset")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    with os.fdopen(fd, "rb") as stream:
        current = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_IMODE(current.st_mode) != mode
            or stream.read() != data
        ):
            raise SourceMaterializationError("conflicting_staged_asset")
    after = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if _file_identity(current) != _file_identity(after):
        raise SourceMaterializationError("conflicting_staged_asset")
    return True


def _publish_file(path: Path, data: bytes, mode: int) -> None:
    with _opened_directory(path.parent, create=True) as parent:
        if _check_at(parent, path.name, data, mode):
            _same_directory(path.parent, parent)
            return
        temporary = ".source-stage-" + uuid.uuid4().hex
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fchmod(stream.fileno(), mode)
                os.fsync(stream.fileno())
            try:
                os.link(
                    temporary,
                    path.name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                pass
            if not _check_at(parent, path.name, data, mode):
                raise SourceMaterializationError("staged_asset_disappeared")
            _same_directory(path.parent, parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass


def _publish_tree(parent: int, temporary: str, destination: str) -> None:
    # rename() can replace an empty directory; the source protocol requires a
    # no-replace publication even when a conflicting stager left no files yet.
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename is None:
        raise SourceMaterializationError("atomic_source_publication_unavailable")
    rename.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename.restype = ctypes.c_int
    if rename(parent, os.fsencode(temporary), parent, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _remove_tree(parent: int, name: str) -> None:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    except FileNotFoundError:
        return
    try:
        with os.scandir(fd) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    _remove_tree(fd, entry.name)
                else:
                    os.unlink(entry.name, dir_fd=fd)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent)


def stage_source_bundle(
    request: dict, eval_dir: Path, assets: Mapping[str, Path]
) -> dict:
    """Copy complete verified source and helpers without replacing old assets.

    Request filenames are content-addressed, so moving a complete run can stage
    a new location-bound request without rewriting old measurement evidence.
    The caller selects the exact canonical helper files required by its runner.
    """
    try:
        source = validate_source_materialization(request)
        if source is None:
            raise SourceMaterializationError("empty_source_request")
        if (
            type(request.get("schema_version")) is not int
            or request["schema_version"] != 1
            or set(request)
            != {"schema_version", "source_snapshots", "source_materialization"}
        ):
            raise SourceMaterializationError("invalid_source_request")
        if not assets:
            raise SourceMaterializationError("missing_source_helpers")
        eval_dir = eval_dir.absolute()
        _directory(eval_dir)
        # Read and precheck the complete helper set before publishing any of it.
        marker = (source.manifest_sha256 + "\n").encode()
        payloads = {"source_manifest.sha256": (marker, 0o644)}
        _check_file(eval_dir / "source_manifest.sha256", marker, 0o644)
        for name, path in assets.items():
            if name in payloads:
                raise SourceMaterializationError("reserved_source_helper_path")
            target = _target(eval_dir, name)
            data, mode = _read_asset(path)
            _check_file(target, data, mode)
            payloads[name] = (data, mode)

        materializations = eval_dir / "source_materializations"
        destination = materializations / source.manifest_sha256
        staged_request = copy.deepcopy(request)
        staged_request["source_materialization"]["bundle_root"] = str(destination)
        with _opened_directory(materializations, create=True) as parent:
            if destination.exists() or destination.is_symlink():
                validate_source_materialization(staged_request)
            else:
                temporary_name = ".materializing-" + uuid.uuid4().hex
                os.mkdir(temporary_name, 0o700, dir_fd=parent)
                temporary = materializations / temporary_name
                try:
                    for row in source.manifest["files"]:
                        _publish_file(
                            temporary / row["path"],
                            _read_regular(source.bundle_root / row["path"]),
                            row["mode"],
                        )
                    _publish_file(
                        temporary / "manifest.json",
                        _read_regular(source.bundle_root / "manifest.json"),
                        0o644,
                    )
                    check = copy.deepcopy(staged_request)
                    check["source_materialization"]["bundle_root"] = str(temporary)
                    validate_source_materialization(check)
                    _same_directory(materializations, parent)
                    try:
                        _publish_tree(parent, temporary_name, destination.name)
                    except OSError as exc:
                        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                            raise
                        validate_source_materialization(staged_request)
                    _same_directory(materializations, parent)
                finally:
                    _remove_tree(parent, temporary_name)
        staged_source = validate_source_materialization(staged_request)
        for name, (data, mode) in payloads.items():
            _publish_file(_target(eval_dir, name), data, mode)

        request_bytes = (
            json.dumps(staged_request, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        request_sha = _sha(request_bytes)
        request_path = eval_dir / "source_requests" / (request_sha + ".json")
        _publish_file(request_path, request_bytes, 0o600)
        receipt = {
            "schema": "geak.source_bundle.v1",
            "status": "staged",
            "request_path": str(request_path.relative_to(eval_dir)),
            "request_sha256": request_sha,
            "manifest_sha256": staged_source.manifest_sha256,
            "assets": {
                name: {"sha256": _sha(data), "mode": mode}
                for name, (data, mode) in payloads.items()
            },
        }
        receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
        receipt_path = eval_dir / "source_requests" / (request_sha + ".staging.json")
        _publish_file(receipt_path, receipt_bytes, 0o600)
        return {
            "request_path": str(request_path),
            "request_sha256": request_sha,
            "staging_receipt": str(receipt_path),
            "manifest_sha256": staged_source.manifest_sha256,
            "pythonpath": os.pathsep.join(staged_source.pythonpath_prefixes),
        }
    except SourceMaterializationError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SourceMaterializationError("source_bundle_staging_failed") from exc


def stage_source_replay_launcher(
    request_path: str | Path, eval_dir: Path, original_launcher: str | Path
) -> str:
    """Stage a source-bound replay wrapper while preserving the original script."""
    try:
        eval_dir = eval_dir.absolute()
        source = read_source_request(str(request_path))
        request_path = Path(request_path)
        original = Path(original_launcher)
        if (
            not original.is_absolute()
            or original.resolve() != original
            or not original.is_relative_to(eval_dir)
            or request_path.resolve() != request_path
            or not request_path.is_relative_to(eval_dir)
        ):
            raise SourceMaterializationError("unsafe_source_replay_path")
        if (
            request_path
            != eval_dir
            / "source_requests"
            / (_sha(_read_regular(request_path)) + ".json")
            or source.bundle_root
            != eval_dir / "source_materializations" / source.manifest_sha256
        ):
            raise SourceMaterializationError("unstaged_source_replay_request")
        original_bytes, _ = _read_asset(original)
        marker = (source.manifest_sha256 + "\n").encode()
        if not _check_file(eval_dir / "source_manifest.sha256", marker, 0o644):
            raise SourceMaterializationError("missing_source_manifest_marker")
        pythonpath = os.pathsep.join(source.pythonpath_prefixes)
        script = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"# Original launcher SHA256: {_sha(original_bytes)}\n"
            f"export GEAK_SOURCE_REQUEST={shlex.quote(str(request_path))}\n"
            f"export GEAK_ACCEPTED_SOURCE_PYTHONPATH={shlex.quote(pythonpath)}\n"
            "unset GEAK_SOURCE_OBSERVATION_DIR GEAK_SOURCE_BOOTSTRAP_PYTHONPATH\n"
            f'exec bash {shlex.quote(str(original))} "$@"\n'
        ).encode()
        destination = eval_dir / "final" / ("source_replay_" + _sha(script) + ".sh")
        _publish_file(destination, script, 0o755)
        return str(destination)
    except SourceMaterializationError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SourceMaterializationError("source_replay_staging_failed") from exc
