# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Staging owns complete copies and never replaces conflicting old helpers."""

import errno
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from interface import source_bundle, test_source_materialization
from interface.source_materialization import (
    SourceMaterializationError,
    read_source_request,
)


@pytest.fixture(name="bundle")
def source_bundle_fixture(tmp_path):
    return test_source_materialization.bundle.__wrapped__(tmp_path)


@pytest.fixture
def assets(tmp_path):
    sources = tmp_path / "canonical"
    sources.mkdir()
    result = {}
    for name in ("bench_e2e.sh", "adapters/launchers/magpie.sh", "source_runtime.py"):
        path = sources / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# canonical test helper: " + name + "\n")
        path.chmod(0o755 if name.endswith(".sh") else 0o644)
        result[name] = path
    return result


def _request(baseline):
    return {"schema_version": 1, **baseline}


@pytest.fixture
def claim_publication(monkeypatch):
    def unsupported(*_args):
        raise OSError(errno.EOPNOTSUPP, "filesystem lacks renameat2 no-replace")

    monkeypatch.setattr(source_bundle, "_publish_tree", unsupported)


@pytest.mark.parametrize("publication", ["rename", "claim"])
def test_staging_keeps_working_after_original_source_is_removed(
    bundle, assets, tmp_path, request, publication
):
    if publication == "claim":
        request.getfixturevalue("claim_publication")
    baseline, _, original = bundle
    target = tmp_path / "run"
    result = source_bundle.stage_source_bundle(_request(baseline), target, assets)
    request_path = Path(result["request_path"])
    assert (
        hashlib.sha256(request_path.read_bytes()).hexdigest()
        == result["request_sha256"]
    )
    assert read_source_request(str(request_path)).bundle_root.is_relative_to(target)
    shutil.rmtree(original)
    verified = read_source_request(str(request_path))
    assert (
        verified.bundle_root / "trees/a/python/alpha/first.py"
    ).read_text() == "VALUE = 'accepted-first'\n"
    assert not (verified.bundle_root / "trees/a/python/alpha/old.py").exists()
    run = subprocess.run(
        [sys.executable, "-B", "-c", "import alpha.first; print(alpha.first.VALUE)"],
        env=dict(os.environ, PYTHONPATH=result["pythonpath"]),
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert run.stdout == "accepted-first\n"
    for name, original_asset in assets.items():
        assert (target / name).read_bytes() == original_asset.read_bytes()
        assert (
            target / name
        ).stat().st_mode & 0o777 == original_asset.stat().st_mode & 0o777
    assert (target / "source_manifest.sha256").read_text() == result[
        "manifest_sha256"
    ] + "\n"
    staged_request = json.loads(request_path.read_text())
    assert source_bundle.stage_source_bundle(staged_request, target, assets) == result


@pytest.mark.parametrize("publication", ["rename", "claim"])
def test_relocation_preserves_old_request_bytes_and_creates_new_location_binding(
    bundle, assets, tmp_path, request, publication
):
    if publication == "claim":
        request.getfixturevalue("claim_publication")
    baseline, _, original = bundle
    first = tmp_path / "first"
    initial = source_bundle.stage_source_bundle(_request(baseline), first, assets)
    old_request = Path(initial["request_path"])
    old_bytes = old_request.read_bytes()
    moved = tmp_path / "moved"
    shutil.copytree(first, moved)
    shutil.rmtree(first)
    shutil.rmtree(original)
    request = json.loads(old_bytes)
    request["source_materialization"]["bundle_root"] = str(
        moved / "source_materializations" / initial["manifest_sha256"]
    )
    result = source_bundle.stage_source_bundle(request, moved, assets)
    assert result["manifest_sha256"] == initial["manifest_sha256"]
    assert (moved / "source_manifest.sha256").read_text() == initial[
        "manifest_sha256"
    ] + "\n"
    assert result["request_sha256"] != initial["request_sha256"]
    assert (moved / old_request.relative_to(first)).read_bytes() == old_bytes
    assert read_source_request(result["request_path"]).bundle_root.is_relative_to(moved)


@pytest.mark.parametrize("kind", ["bytes", "mode", "symlink", "fifo", "parent_symlink"])
def test_conflicting_helpers_are_never_overwritten(bundle, assets, tmp_path, kind):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    target.mkdir()
    conflict = target / "bench_e2e.sh"
    if kind == "bytes":
        conflict.write_text("# old helper must remain\n")
    elif kind == "mode":
        conflict.write_bytes(assets["bench_e2e.sh"].read_bytes())
        conflict.chmod(0o600)
    elif kind == "symlink":
        conflict.symlink_to(assets["bench_e2e.sh"])
    elif kind == "fifo":
        os.mkfifo(conflict)
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        (target / "adapters").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SourceMaterializationError):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert not (target / "source_materializations").exists()
    if kind == "bytes":
        assert conflict.read_text() == "# old helper must remain\n"
    elif kind == "parent_symlink":
        assert list(outside.iterdir()) == []


def test_copy_failure_leaves_no_partial_published_materialization(
    bundle, assets, tmp_path, monkeypatch
):
    baseline, _, original = bundle
    original_bytes = (original / "trees/a/python/alpha/second.py").read_bytes()
    real_read = source_bundle._read_regular

    def failed_read(path, **kwargs):
        if path == original / "trees/a/python/alpha/second.py":
            raise OSError("simulated source-copy failure")
        return real_read(path, **kwargs)

    monkeypatch.setattr(source_bundle, "_read_regular", failed_read)
    target = tmp_path / "run"
    with pytest.raises(
        SourceMaterializationError, match="source_bundle_staging_failed"
    ):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert list((target / "source_materializations").iterdir()) == []
    assert not (target / "bench_e2e.sh").exists()
    assert (original / "trees/a/python/alpha/second.py").read_bytes() == original_bytes


def test_tampered_staged_source_is_refused_without_repairing_it(
    bundle, assets, tmp_path
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    result = source_bundle.stage_source_bundle(_request(baseline), target, assets)
    copied = read_source_request(result["request_path"]).bundle_root
    changed = copied / "trees/a/python/alpha/first.py"
    changed.write_text("VALUE='tampered'\n")
    with pytest.raises(SourceMaterializationError, match="file_digest_mismatch"):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert changed.read_text() == "VALUE='tampered'\n"


def test_complete_helper_set_is_required_before_staging_source(bundle, tmp_path):
    baseline, _, _ = bundle
    with pytest.raises(SourceMaterializationError, match="missing_source_helpers"):
        source_bundle.stage_source_bundle(_request(baseline), tmp_path / "run", {})


@pytest.mark.parametrize("kind", ["bytes", "mode", "symlink", "directory"])
def test_marker_conflict_refuses_before_other_publication(
    bundle, assets, tmp_path, kind
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    target.mkdir()
    marker = target / "source_manifest.sha256"
    expected = baseline["source_materialization"]["manifest_sha256"] + "\n"
    if kind == "directory":
        marker.mkdir()
    elif kind == "symlink":
        outside = tmp_path / "marker"
        outside.write_text(expected)
        marker.symlink_to(outside)
    else:
        marker.write_text(expected if kind == "mode" else "previous-manifest\n")
        marker.chmod(0o600 if kind == "mode" else 0o644)
    with pytest.raises(SourceMaterializationError, match="conflicting_staged_asset"):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert {path.name for path in target.iterdir()} == {"source_manifest.sha256"}
    if kind == "bytes":
        assert marker.read_text() == "previous-manifest\n"


@pytest.mark.parametrize("kind", ["missing", "symlink", "parent_symlink", "fifo"])
def test_invalid_original_helper_refuses_before_publication(
    bundle, assets, tmp_path, kind
):
    baseline, _, _ = bundle
    broken = tmp_path / "broken"
    if kind == "symlink":
        broken.symlink_to(assets["bench_e2e.sh"])
    elif kind == "parent_symlink":
        broken.symlink_to(assets["bench_e2e.sh"].parent, target_is_directory=True)
        broken = broken / "bench_e2e.sh"
    elif kind == "fifo":
        os.mkfifo(broken)
    assets["bench_e2e.sh"] = broken
    target = tmp_path / "run"
    with pytest.raises(SourceMaterializationError):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert list(target.iterdir()) == []


def test_reserved_marker_cannot_be_supplied_as_a_helper(bundle, assets, tmp_path):
    baseline, _, _ = bundle
    assets["source_manifest.sha256"] = assets["bench_e2e.sh"]
    target = tmp_path / "run"
    with pytest.raises(SourceMaterializationError, match="reserved_source_helper_path"):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert list(target.iterdir()) == []


def test_concurrent_identical_stagers_reuse_complete_publication(
    bundle, assets, tmp_path
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [
            pool.submit(
                source_bundle.stage_source_bundle, _request(baseline), target, assets
            )
            for _ in range(8)
        ]
        results = []
        for job in jobs:
            try:
                results.append(job.result(timeout=20))
            except SourceMaterializationError as exc:
                # NFS callers can observe the exclusive creator's incomplete
                # claim, including stale directory inventory during publication.
                assert exc.reason in {
                    "incomplete_staged_materialization",
                    "incomplete_file_inventory",
                }
    assert results
    assert all(result == results[0] for result in results)
    for _ in jobs:
        assert (
            source_bundle.stage_source_bundle(_request(baseline), target, assets)
            == results[0]
        )
    assert read_source_request(results[0]["request_path"])
    assert not list(target.rglob(".source-stage-*"))
    assert not list(target.rglob(".materializing-*"))


@pytest.mark.parametrize("identical", [True, False])
def test_concurrent_helper_winner_is_never_replaced(
    bundle, assets, tmp_path, monkeypatch, identical
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    real_link = os.link
    expected = assets["bench_e2e.sh"].read_bytes()
    winner = expected if identical else b"# concurrent conflicting helper\n"

    def racing_link(src, dst, **kwargs):
        if dst == "bench_e2e.sh":
            fd = os.open(
                dst,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o755,
                dir_fd=kwargs["dst_dir_fd"],
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(winner)
                os.fchmod(stream.fileno(), 0o755)
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    if identical:
        result = source_bundle.stage_source_bundle(_request(baseline), target, assets)
        assert read_source_request(result["request_path"])
    else:
        with pytest.raises(
            SourceMaterializationError, match="conflicting_staged_asset"
        ):
            source_bundle.stage_source_bundle(_request(baseline), target, assets)
        assert not (target / "source_requests").exists()
    assert (target / "bench_e2e.sh").read_bytes() == winner
    assert not list(target.rglob(".source-stage-*"))


def test_removed_publication_cannot_return_an_active_request(
    bundle, assets, tmp_path, monkeypatch
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    real_link = os.link

    def racing_link(src, dst, **kwargs):
        result = real_link(src, dst, **kwargs)
        if dst == "bench_e2e.sh":
            os.unlink(dst, dir_fd=kwargs["dst_dir_fd"])
        return result

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(SourceMaterializationError, match="staged_asset_disappeared"):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert not (target / "source_requests").exists()
    assert not list(target.rglob(".source-stage-*"))


@pytest.mark.parametrize("identical", [True, False])
def test_concurrent_materialization_winner_requires_complete_exact_source(
    bundle, assets, tmp_path, monkeypatch, identical
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    materializations = target / "source_materializations"
    real_publish = source_bundle._publish_tree
    winner_inode = []

    def racing_publish(parent, temporary, destination):
        winner = materializations / destination
        if identical:
            shutil.copytree(materializations / temporary, winner)
        else:
            winner.mkdir()
        winner_inode.append(winner.stat().st_ino)
        real_publish(parent, temporary, destination)

    monkeypatch.setattr(source_bundle, "_publish_tree", racing_publish)
    if identical:
        result = source_bundle.stage_source_bundle(_request(baseline), target, assets)
        assert read_source_request(result["request_path"])
    else:
        with pytest.raises(SourceMaterializationError):
            source_bundle.stage_source_bundle(_request(baseline), target, assets)
        assert not (target / "source_requests").exists()
    winner = materializations / baseline["source_materialization"]["manifest_sha256"]
    assert winner.stat().st_ino == winner_inode[0]
    if not identical:
        assert list(winner.iterdir()) == []
    assert not list(materializations.glob(".materializing-*"))


def _inventory(root):
    return {
        str(path.relative_to(root)): (
            path.stat().st_ino,
            path.read_bytes() if path.is_file() else None,
        )
        for path in root.rglob("*")
    }


def test_claim_interruption_preserves_unready_source_and_never_repairs_it(
    bundle, assets, tmp_path, monkeypatch, claim_publication
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    claimed = (
        target
        / "source_materializations"
        / baseline["source_materialization"]["manifest_sha256"]
    )
    publish = source_bundle._publish_claimed_file

    def interrupted(path, *args):
        if path == claimed and args[2].endswith("second.py"):
            raise OSError("simulated interrupted NFS copy")
        return publish(path, *args)

    monkeypatch.setattr(source_bundle, "_publish_claimed_file", interrupted)
    with pytest.raises(
        SourceMaterializationError, match="source_bundle_staging_failed"
    ):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert claimed.is_dir() and list(claimed.rglob("*.py"))
    assert not (claimed / "manifest.json").exists()
    assert not (target / "source_requests").exists()
    assert not (target / "bench_e2e.sh").exists()
    assert not list(target.rglob(".materializing-*"))
    before = _inventory(claimed)
    monkeypatch.setattr(source_bundle, "_publish_claimed_file", publish)
    with pytest.raises(
        SourceMaterializationError, match="incomplete_staged_materialization"
    ):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert _inventory(claimed) == before


def test_concurrent_claim_cannot_publish_until_manifest_is_complete(
    bundle, assets, tmp_path, monkeypatch, claim_publication
):
    baseline, manifest, _ = bundle
    target = tmp_path / "run"
    claimed = (
        target
        / "source_materializations"
        / baseline["source_materialization"]["manifest_sha256"]
    )
    publish = source_bundle._publish_claimed_file
    waiting, resume = threading.Event(), threading.Event()

    def paused(path, *args):
        if path == claimed and args[2] == "manifest.json":
            waiting.set()
            assert resume.wait(10)
        return publish(path, *args)

    monkeypatch.setattr(source_bundle, "_publish_claimed_file", paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        owner = pool.submit(
            source_bundle.stage_source_bundle, _request(baseline), target, assets
        )
        try:
            assert waiting.wait(10)
            assert all((claimed / row["path"]).is_file() for row in manifest["files"])
            assert not (claimed / "manifest.json").exists()
            before = _inventory(claimed)
            with pytest.raises(
                SourceMaterializationError, match="incomplete_staged_materialization"
            ):
                source_bundle.stage_source_bundle(_request(baseline), target, assets)
            assert _inventory(claimed) == before
            assert not (target / "source_requests").exists()
        finally:
            resume.set()
        result = owner.result(timeout=10)
    assert (
        source_bundle.stage_source_bundle(_request(baseline), target, assets) == result
    )
    assert read_source_request(result["request_path"])


def test_completed_claim_is_reusable_after_creator_fails_before_request(
    bundle, assets, tmp_path, monkeypatch, claim_publication
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    claimed = (
        target
        / "source_materializations"
        / baseline["source_materialization"]["manifest_sha256"]
    )
    publish = source_bundle._publish_claimed_file

    def failed_after_manifest(path, *args):
        identity = publish(path, *args)
        if path == claimed and args[2] == "manifest.json":
            raise OSError(errno.EIO, "creator stopped after manifest publication")
        return identity

    monkeypatch.setattr(source_bundle, "_publish_claimed_file", failed_after_manifest)
    with pytest.raises(
        SourceMaterializationError, match="source_bundle_staging_failed"
    ):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert (claimed / "manifest.json").is_file()
    assert not (target / "source_requests").exists()
    before = _inventory(claimed)
    monkeypatch.setattr(source_bundle, "_publish_claimed_file", publish)
    result = source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert read_source_request(result["request_path"])
    assert _inventory(claimed) == before


def test_killed_claim_creator_leaves_no_ready_request_and_retry_refuses(
    bundle, assets, tmp_path
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    marker = tmp_path / "waiting-for-manifest"
    code = """
import errno, json, sys, time
from pathlib import Path
from interface import source_bundle
request, assets = json.loads(sys.argv[1]), json.loads(sys.argv[2])
target, marker = Path(sys.argv[3]), Path(sys.argv[4])
claimed = target / 'source_materializations' / request['source_materialization']['manifest_sha256']
def unavailable(*args):
    raise OSError(errno.EOPNOTSUPP, 'force NFS publication')
source_bundle._publish_tree = unavailable
publish = source_bundle._publish_claimed_file
def paused(path, *args):
    if path == claimed and args[2] == 'manifest.json':
        marker.touch()
        time.sleep(60)
    return publish(path, *args)
source_bundle._publish_claimed_file = paused
source_bundle.stage_source_bundle(request, target, {key: Path(value) for key, value in assets.items()})
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            json.dumps(_request(baseline)),
            json.dumps({name: str(path) for name, path in assets.items()}),
            str(target),
            str(marker),
        ],
        cwd=Path(source_bundle.__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not marker.exists()
            and time.monotonic() < deadline
            and process.poll() is None
        ):
            time.sleep(0.02)
        assert marker.exists()
        assert not (target / "source_requests").exists()
    finally:
        process.kill()
        process.communicate(timeout=5)
    before = _inventory(target)
    with pytest.raises(
        SourceMaterializationError, match="incomplete_staged_materialization"
    ):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert _inventory(target) == before


@pytest.mark.parametrize("level", ["root", "nested"])
@pytest.mark.parametrize("replacement", ["directory", "symlink"])
def test_claim_writes_stay_in_owned_directories_after_replacement(
    bundle, assets, tmp_path, monkeypatch, claim_publication, level, replacement
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    claimed = (
        target
        / "source_materializations"
        / baseline["source_materialization"]["manifest_sha256"]
    )
    changed = claimed if level == "root" else claimed / "trees/a/python/alpha"
    parked, outside = tmp_path / "parked", tmp_path / "outside"
    outside.mkdir()
    link = os.link

    def replaced(src, dst, **kwargs):
        parent = Path(os.readlink(f"/proc/self/fd/{kwargs['dst_dir_fd']}"))
        if dst == "first.py" and parent.is_relative_to(claimed):
            changed.rename(parked)
            if replacement == "directory":
                changed.mkdir()
            else:
                changed.symlink_to(outside, target_is_directory=True)
        return link(src, dst, **kwargs)

    monkeypatch.setattr(os, "link", replaced)
    with pytest.raises(SourceMaterializationError):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert list(changed.iterdir()) == []
    assert list(outside.iterdir()) == []
    assert not (target / "source_requests").exists()


@pytest.mark.parametrize("level", ["root", "nested"])
def test_temporary_cleanup_preserves_replaced_directories(
    bundle, assets, tmp_path, monkeypatch, level
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    parked = tmp_path / "parked"
    replacements = []

    def failed(parent, temporary, destination):
        changed = target / "source_materializations" / temporary
        if level == "nested":
            changed /= "trees/a/python/alpha"
        changed.rename(parked)
        changed.mkdir()
        (changed / "unowned.txt").write_text("preserve this replacement\n")
        replacements.append(changed)
        raise OSError(errno.EIO, "simulated publication failure")

    monkeypatch.setattr(source_bundle, "_publish_tree", failed)
    with pytest.raises(
        SourceMaterializationError, match="source_bundle_staging_failed"
    ):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert (
        replacements[0] / "unowned.txt"
    ).read_text() == "preserve this replacement\n"
    assert not (target / "source_requests").exists()


def test_regular_file_cleanup_preserves_a_replaced_temporary(
    bundle, assets, tmp_path, monkeypatch
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    replacements = []

    def replaced(src, dst, **kwargs):
        parent = kwargs["src_dir_fd"]
        os.rename(src, src + "-parked", src_dir_fd=parent, dst_dir_fd=parent)
        fd = os.open(src, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(b"unowned replacement temporary\n")
        replacements.append(Path(os.readlink(f"/proc/self/fd/{parent}")) / src)
        raise OSError(errno.EIO, "simulated failed link")

    monkeypatch.setattr(os, "link", replaced)
    with pytest.raises(
        SourceMaterializationError, match="source_bundle_staging_failed"
    ):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert replacements[0].read_bytes() == b"unowned replacement temporary\n"
    assert not (target / "source_requests").exists()


@pytest.mark.parametrize("phase", ["mkdir", "link"])
def test_parent_symlink_swap_never_redirects_staging_writes(
    bundle, assets, tmp_path, monkeypatch, phase
):
    baseline, _, _ = bundle
    target = tmp_path / "run"
    target.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    parked = tmp_path / "parked"

    def swap():
        target.rename(parked)
        target.symlink_to(outside, target_is_directory=True)

    if phase == "mkdir":
        real_mkdir = os.mkdir

        def racing_mkdir(path, *args, **kwargs):
            if path == "source_materializations":
                swap()
            return real_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(os, "mkdir", racing_mkdir)
    else:
        real_link = os.link

        def racing_link(src, dst, **kwargs):
            if dst == "source_manifest.sha256":
                swap()
            return real_link(src, dst, **kwargs)

        monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(SourceMaterializationError):
        source_bundle.stage_source_bundle(_request(baseline), target, assets)
    assert list(outside.iterdir()) == []
    assert not (parked / "source_requests").exists()


def _stage_replay(bundle, assets, tmp_path):
    baseline, _, _ = bundle
    target = tmp_path / "run with 'quoted' paths"
    staged = source_bundle.stage_source_bundle(_request(baseline), target, assets)
    original = target / "final" / "original 'launcher'.sh"
    original.parent.mkdir()
    original.write_text(
        "#!/usr/bin/env bash\nexec "
        + shlex.quote(sys.executable)
        + " -B -c "
        + shlex.quote(
            "import json,os,sys; print(json.dumps({'argv':sys.argv[1:],"
            "'request':os.environ.get('GEAK_SOURCE_REQUEST'),"
            "'pythonpath':os.environ.get('GEAK_ACCEPTED_SOURCE_PYTHONPATH'),"
            "'observer':os.environ.get('GEAK_SOURCE_OBSERVATION_DIR'),"
            "'bootstrap':os.environ.get('GEAK_SOURCE_BOOTSTRAP_PYTHONPATH')}))"
        )
        + ' "$@"\n'
    )
    return target, staged, original


def test_replay_wrapper_binds_exact_source_and_preserves_argv(bundle, assets, tmp_path):
    target, staged, original = _stage_replay(bundle, assets, tmp_path)
    old_bytes = original.read_bytes()
    wrapper = source_bundle.stage_source_replay_launcher(
        staged["request_path"], target, original
    )
    args = [
        "space argument",
        "$(touch should-not-exist)",
        "single'quote",
        'double"quote',
    ]
    run = subprocess.run(
        [wrapper, *args],
        env=dict(
            os.environ,
            GEAK_SOURCE_REQUEST="stale-request",
            GEAK_ACCEPTED_SOURCE_PYTHONPATH="stale-pythonpath",
            GEAK_SOURCE_OBSERVATION_DIR="stale-observer",
            GEAK_SOURCE_BOOTSTRAP_PYTHONPATH="stale-bootstrap",
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(run.stdout) == {
        "argv": args,
        "request": staged["request_path"],
        "pythonpath": staged["pythonpath"],
        "observer": None,
        "bootstrap": None,
    }
    assert not (tmp_path / "should-not-exist").exists()
    assert original.read_bytes() == old_bytes
    assert Path(wrapper).stat().st_mode & 0o777 == 0o755
    assert (
        source_bundle.stage_source_replay_launcher(
            staged["request_path"], target, original
        )
        == wrapper
    )


@pytest.mark.parametrize("kind", ["outside", "symlink", "directory", "fifo", "missing"])
def test_replay_refuses_invalid_original_launcher(bundle, assets, tmp_path, kind):
    target, staged, original = _stage_replay(bundle, assets, tmp_path)
    if kind == "outside":
        original = tmp_path / "outside.sh"
        original.write_text("exit 0\n")
    else:
        original.unlink()
        if kind == "symlink":
            original.symlink_to(assets["bench_e2e.sh"])
        elif kind == "directory":
            original.mkdir()
        elif kind == "fifo":
            os.mkfifo(original)
    with pytest.raises(SourceMaterializationError):
        source_bundle.stage_source_replay_launcher(
            staged["request_path"], target, original
        )
    assert not list((target / "final").glob("source_replay_*"))


@pytest.mark.parametrize("kind", ["missing", "bytes", "mode", "symlink"])
def test_replay_requires_exact_regular_marker(bundle, assets, tmp_path, kind):
    target, staged, original = _stage_replay(bundle, assets, tmp_path)
    marker = target / "source_manifest.sha256"
    if kind == "bytes":
        marker.write_text("wrong\n")
    elif kind == "mode":
        marker.chmod(0o600)
    else:
        marker.unlink()
        if kind == "symlink":
            marker.symlink_to(assets["bench_e2e.sh"])
    with pytest.raises(SourceMaterializationError):
        source_bundle.stage_source_replay_launcher(
            staged["request_path"], target, original
        )
    assert not list((target / "final").glob("source_replay_*"))


def test_replay_conflict_is_never_overwritten(bundle, assets, tmp_path):
    target, staged, original = _stage_replay(bundle, assets, tmp_path)
    wrapper = Path(
        source_bundle.stage_source_replay_launcher(
            staged["request_path"], target, original
        )
    )
    wrapper.write_text("# retain conflicting wrapper\n")
    with pytest.raises(SourceMaterializationError, match="conflicting_staged_asset"):
        source_bundle.stage_source_replay_launcher(
            staged["request_path"], target, original
        )
    assert wrapper.read_text() == "# retain conflicting wrapper\n"


def test_replay_request_must_be_run_owned_and_content_addressed(
    bundle, assets, tmp_path
):
    target, staged, original = _stage_replay(bundle, assets, tmp_path)
    copied = target / "source_requests" / "unbound.json"
    copied.write_bytes(Path(staged["request_path"]).read_bytes())
    with pytest.raises(
        SourceMaterializationError, match="unstaged_source_replay_request"
    ):
        source_bundle.stage_source_replay_launcher(copied, target, original)
    external = json.loads(copied.read_text())
    external["source_materialization"]["bundle_root"] = str(bundle[2])
    data = (json.dumps(external, sort_keys=True) + "\n").encode()
    copied = target / "source_requests" / (hashlib.sha256(data).hexdigest() + ".json")
    copied.write_bytes(data)
    with pytest.raises(
        SourceMaterializationError, match="unstaged_source_replay_request"
    ):
        source_bundle.stage_source_replay_launcher(copied, target, original)
