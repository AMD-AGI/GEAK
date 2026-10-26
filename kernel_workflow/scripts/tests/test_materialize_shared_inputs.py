# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only workspace copies using a small synthetic native task manifest."""

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "kernel_workflow/scripts"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "shared_materializer_fixtures", ROOT / "tests/native/fixtures.py"
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.fixture(autouse=True)
def isolated_shared_settings(monkeypatch):
    for suffix in ("MANIFEST", "SHA256", "ROOT"):
        monkeypatch.delenv("GEAK_SHARED_INPUT_" + suffix, raising=False)


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "task inputs"
    contract = fixtures.fixture(root)
    additions = {
        "captures[1]/input?.bin": b"shared capture",
        "captures1/inputA.bin": b"ordinary protected input",
        "aiter/frozen.bin": b"shared vendor input",
        "reference_io.pt": b"ordinary golden",
    }
    for name, value in additions.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        contract["artifacts"][name] = hashlib.sha256(value).hexdigest()
    (root / "task.json").write_text(json.dumps(contract))
    shared = {name: digest for name, digest in contract["artifacts"].items()
              if name.startswith(("captures[1]/", "aiter/"))}
    manifest = {
        "schema": "geak-native-shared-inputs-v1", "task_root": "/producer/task",
        "artifacts": contract["artifacts"],
        "sizes": {name: (root / name).stat().st_size for name in contract["artifacts"]},
        "shared_directories": ["aiter", "captures[1]"], "shared_files": shared,
    }
    manifest_path = tmp_path / "shared inputs.json"
    manifest_path.write_text(json.dumps(manifest))
    pin = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return root, contract, manifest, manifest_path, pin


@pytest.fixture
def readonly_env(task, tmp_path):
    # Model only the read-only mount observation for this synthetic task. No real
    # mounts or native fixtures are used; unmocked writable-root rejection is
    # exercised separately below.
    modules = tmp_path / "readonly_observation"
    modules.mkdir()
    (modules / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "original_statvfs = os.statvfs\n"
        f"root = Path({str(task[0])!r})\n"
        "def statvfs(path):\n"
        "    info = original_statvfs(path)\n"
        "    if Path(path) == root or root in Path(path).parents:\n"
        "        values = list(info)\n"
        "        values[8] |= os.ST_RDONLY\n"
        "        return os.statvfs_result(values)\n"
        "    return info\n"
        "os.statvfs = statvfs\n"
    )
    return {"PATH": os.environ["PATH"],
            "PYTHONPATH": str(modules) + os.pathsep + os.environ.get("PYTHONPATH", "")}


def copy(src, dst, *extra, env=None):
    return subprocess.run(
        ["bash", str(SCRIPTS / "materialize_workspace.sh"), "--src", str(src),
         "--dst", str(dst), *map(str, extra)], capture_output=True, text=True, env=env,
    )


def options(task):
    root, _, _, manifest, pin = task
    return ["--shared-input-manifest", manifest, "--shared-input-sha256", pin,
            "--shared-input-root", root]


@pytest.mark.parametrize("configuration", ["cli", "environment"])
def test_exact_links_and_private_source_across_generations(task, tmp_path, readonly_env, configuration):
    root, contract, manifest, _, _ = task
    source = contract["source_paths"][0]
    original = (root / source).read_bytes()
    (root / source).unlink()
    os.link(root / "baseline_src/module.py", root / source)
    ordinary = tmp_path / "ordinary"
    result = copy(root, ordinary)
    assert result.returncode == 0, result.stderr
    assert all((ordinary / name).is_file() and not (ordinary / name).is_symlink()
               for name in manifest["shared_files"])
    env = dict(readonly_env)
    extra = options(task)
    if configuration == "environment":
        env.update(GEAK_SHARED_INPUT_MANIFEST=str(task[3]),
                   GEAK_SHARED_INPUT_SHA256=task[4], GEAK_SHARED_INPUT_ROOT=str(root))
        extra = []
    else:
        env.update(GEAK_SHARED_INPUT_MANIFEST="/unused/manifest.json",
                   GEAK_SHARED_INPUT_SHA256="0" * 64, GEAK_SHARED_INPUT_ROOT="/unused/task")
    prior = root
    for generation in ("canonical", "base", "engineer", "verify", "next_wave"):
        workspace = tmp_path / generation / "workspace"
        result = copy(prior, workspace, *extra,
                      "--shared-root", tmp_path / "legacy_shared", "--link-aiter", env=env)
        assert result.returncode == 0, result.stderr
        for name in manifest["shared_files"]:
            assert (workspace / name).is_symlink()
            assert os.readlink(workspace / name) == str(root / name)
            assert (workspace / name).read_bytes() == (root / name).read_bytes()
        # Literal brackets and '?' in shared names must not widen tar exclusions.
        assert (workspace / "captures1/inputA.bin").read_bytes() == b"ordinary protected input"
        assert not (workspace / "aiter").is_symlink()
        assert (workspace / "reference_io.pt").read_bytes() == b"ordinary golden"
        assert not (workspace / "reference_io.pt").is_symlink()
        candidate = workspace / source
        assert candidate.is_file() and not candidate.is_symlink()
        assert candidate.stat().st_ino != (prior / source).stat().st_ino
        prior_bytes = (prior / source).read_bytes()
        assert candidate.read_bytes() == prior_bytes
        candidate.write_bytes(prior_bytes + b"# private edit\n")
        assert (prior / source).read_bytes() == prior_bytes
        assert (workspace / "baseline_src/module.py").read_bytes() == original
        assert candidate.stat().st_ino != (workspace / "baseline_src/module.py").stat().st_ino
        assert not (workspace / ".geak/previous-evaluation").exists()
        (workspace / ".geak/previous-evaluation").write_text("must not copy")
        prior = workspace
    assert not (tmp_path / "legacy_shared").exists()
    assert (root / source).read_bytes() == original


@pytest.mark.skipif(os.geteuid() == 0, reason="requires real unprivileged directory permissions")
def test_readonly_fixture_parents_across_generations(task, tmp_path, readonly_env):
    root, _, manifest, _, _ = task
    shared_files = [root / name for name in manifest["shared_files"]]
    shared_parents = {path.parent for path in shared_files}
    for path in shared_files:
        path.chmod(0o444)
    for parent in shared_parents:
        assert parent.stat().st_uid == os.geteuid()
        parent.chmod(0o555)
    shared_state = {
        path: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in shared_files
    }
    try:
        prior = root
        for generation in ("base", "engineer", "verify"):
            workspace = tmp_path / generation / "workspace"
            result = copy(prior, workspace, *options(task), env=readonly_env)
            assert result.returncode == 0, result.stderr
            for name in manifest["shared_files"]:
                link = workspace / name
                assert link.is_symlink()
                assert os.readlink(link) == str(root / name)
                assert link.read_bytes() == shared_state[root / name][0]
                assert not link.parent.is_symlink()
                assert link.parent.stat().st_uid == os.geteuid()
                assert stat.S_IMODE(link.parent.stat().st_mode) == 0o755
            for path, (content, mode) in shared_state.items():
                assert path.read_bytes() == content
                assert stat.S_IMODE(path.stat().st_mode) == mode
            assert all(stat.S_IMODE(parent.stat().st_mode) == 0o555
                       for parent in shared_parents)
            prior = workspace
    finally:
        for parent in shared_parents:
            parent.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="requires real unprivileged directory permissions")
def test_readonly_source_becomes_private_mutable_copy(task, tmp_path, readonly_env):
    root, contract, manifest, _, _ = task
    script = root / "private-script.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o555)
    external = tmp_path / "external"
    external.mkdir()
    external_file = external / "header.h"
    external_file.write_bytes(b"immutable external header\n")
    external_file.chmod(0o444)
    external.chmod(0o555)
    (root / "external").symlink_to(external, target_is_directory=True)
    physical = [root, *(path for path in root.rglob("*") if not path.is_symlink())]
    for path in physical:
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)
    source_state = {
        path: (stat.S_IMODE(path.stat().st_mode), path.read_bytes() if path.is_file() else None)
        for path in physical
    }
    try:
        workspace = tmp_path / "private" / "workspace"
        workspace.mkdir(parents=True, mode=0o700)
        result = copy(root, workspace, *options(task), env=readonly_env)
        assert result.returncode == 0, result.stderr
        assert stat.S_IMODE(workspace.stat().st_mode) == 0o755
        assert (workspace / ".geak/workspace.json").is_file()
        for path in [workspace, *workspace.rglob("*")]:
            if not path.is_symlink():
                assert path.stat().st_mode & stat.S_IWUSR
        candidate = workspace / contract["source_paths"][0]
        original = (root / contract["source_paths"][0]).read_bytes()
        candidate.write_bytes(original + b"# private edit\n")
        assert stat.S_IMODE((workspace / script.name).stat().st_mode) == 0o755
        for name in manifest["shared_files"]:
            assert (workspace / name).is_symlink()
            assert os.readlink(workspace / name) == str(root / name)
        assert (workspace / "external").is_symlink()
        assert (workspace / "external").resolve() == external
        assert external_file.read_bytes() == b"immutable external header\n"
        assert stat.S_IMODE(external_file.stat().st_mode) == 0o444
        assert stat.S_IMODE(external.stat().st_mode) == 0o555
        for path, (mode, content) in source_state.items():
            assert stat.S_IMODE(path.stat().st_mode) == mode
            if content is not None:
                assert path.read_bytes() == content
        assert not (root / ".geak").exists()
    finally:
        for path in physical:
            if path.is_dir():
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
        external.chmod(0o755)


@pytest.mark.parametrize("invalid", ["partial", "pin", "source", "closure"])
def test_invalid_manifest_preserves_existing_destination(task, tmp_path, readonly_env, invalid):
    _, contract, manifest, manifest_path, _ = task
    args = options(task)
    if invalid == "partial":
        args = args[:2]
    elif invalid == "pin":
        args[3] = "0" * 64
    else:
        if invalid == "source":
            source = contract["source_paths"][0]
            manifest["shared_files"][source] = contract["artifacts"][source]
        else:
            manifest["shared_files"].pop(next(iter(manifest["shared_files"])))
        manifest_path.write_text(json.dumps(manifest))
        args[3] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("prior workspace")
    result = copy(task[0], destination, *args, env=readonly_env)
    assert result.returncode != 0
    assert sorted(p.name for p in destination.iterdir()) == ["keep.txt"]
    assert (destination / "keep.txt").read_text() == "prior workspace"


@pytest.mark.parametrize("invalid", ["writable_root", "aliased_root", "aliased_leaf", "changed_bytes"])
def test_untrusted_shared_target_preserves_destination(task, tmp_path, readonly_env, invalid):
    root, _, _, _, _ = task
    args = options(task)
    env = readonly_env
    if invalid == "writable_root":
        env = {"PATH": os.environ["PATH"], "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        assert not os.statvfs(root).f_flag & os.ST_RDONLY
    elif invalid == "aliased_root":
        alias = tmp_path / "root_alias"
        alias.symlink_to(root, target_is_directory=True)
        args[-1] = alias
    elif invalid == "aliased_leaf":
        leaf = root / "aiter/frozen.bin"
        target = root / "original_frozen.bin"
        leaf.rename(target)
        leaf.symlink_to(target)
    else:
        leaf = root / "aiter/frozen.bin"
        leaf.write_bytes(b"X" * leaf.stat().st_size)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("prior workspace")
    result = copy(root, destination, *args, env=env)
    assert result.returncode != 0
    assert sorted(p.name for p in destination.iterdir()) == ["keep.txt"]


def test_link_rejects_parent_alias_without_creating_external_directories(task, tmp_path, readonly_env):
    root, contract, manifest, manifest_path, _ = task
    old, nested = "aiter/frozen.bin", "aiter/nested/frozen.bin"
    (root / nested).parent.mkdir()
    (root / old).rename(root / nested)
    for mapping in (contract["artifacts"], manifest["shared_files"], manifest["sizes"]):
        mapping[nested] = mapping.pop(old)
    (root / "task.json").write_text(json.dumps(contract))
    manifest_path.write_text(json.dumps(manifest))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o555)
    (workspace / "aiter").symlink_to(outside, target_is_directory=True)
    args = options(task)
    args[3] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    try:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "task_shared_inputs.py"), "link",
             "--manifest", str(args[1]), "--sha256", args[3], "--task-root", str(args[5]),
             "--workspace", str(workspace)], capture_output=True, text=True, env=readonly_env,
        )
        assert result.returncode != 0
        assert "shared input workspace parent cannot be an alias" in result.stderr
        assert not list(outside.iterdir())
        assert stat.S_IMODE(outside.stat().st_mode) == 0o555
    finally:
        outside.chmod(0o755)
