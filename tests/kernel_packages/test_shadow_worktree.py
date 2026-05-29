"""Tests for ``shadow_worktree``: build a writable shadow tree for a
wheel-installed Python package without copying multi-GB ``.so`` files.

We use small fake source trees instead of the real ~1GB vllm wheel.
The structural invariants we check:

  * ``.py`` files become PHYSICAL COPIES (different inode → editing
    them never clobbers the baseline).
  * ``.so`` files become SYMLINKS (same content, no inode duplication).
  * ``__pycache__`` and dot-dirs are skipped (don't bloat the shadow).
  * The shadow root is a git repo with a baseline commit.
  * ``.gitignore`` excludes ``*.so`` / ``__pycache__`` so binaries
    never enter the index.
  * The shadow marker file is planted so ``is_shadow_worktree`` /
    profile detection can recognise it post-creation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from minisweagent.kernel_packages.shadow_worktree import (
    SHADOW_MARKER,
    is_shadow_worktree,
    shadow_worktree,
)


@pytest.fixture
def fake_vllm_source(tmp_path: Path) -> Path:
    """Build a minimal vllm-shaped source tree under tmp_path/vllm."""
    src = tmp_path / "vllm"
    src.mkdir()
    (src / "__init__.py").write_text("# vllm package\n__version__ = '0.0.0-test'\n")
    (src / "_C.abi3.so").write_bytes(b"\x7fELF binary blob\n")
    (src / "_moe_C.abi3.so").write_bytes(b"\x7fELF moe binary\n")

    # Sub-package with a nested .py and a nested .so
    sub = src / "engine"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "core.py").write_text("def go(): return 'baseline'\n")
    (sub / "_engine.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF engine\n")

    # __pycache__ that should be skipped.
    pyc_dir = src / "__pycache__"
    pyc_dir.mkdir()
    (pyc_dir / "stale.pyc").write_bytes(b"stale bytecode")

    # Dot-dir that should be skipped.
    dot = src / ".internal"
    dot.mkdir()
    (dot / "secret.py").write_text("# should not be copied")

    return src


def test_shadow_creates_expected_layout(fake_vllm_source, tmp_path):
    dst = tmp_path / "slot_0"
    result = shadow_worktree(fake_vllm_source, dst)
    assert result == dst
    pkg = dst / "vllm"
    assert pkg.is_dir()
    assert (pkg / "__init__.py").is_file()
    assert (pkg / "engine" / "core.py").is_file()


def test_py_files_are_physical_copies(fake_vllm_source, tmp_path):
    """A ``.py`` edit on the shadow MUST NOT modify the baseline."""
    dst = tmp_path / "slot_0"
    shadow_worktree(fake_vllm_source, dst)
    src_py = fake_vllm_source / "engine" / "core.py"
    dst_py = dst / "vllm" / "engine" / "core.py"
    assert src_py.stat().st_ino != dst_py.stat().st_ino, (
        "engine/core.py shares an inode with baseline — editing the shadow "
        "would corrupt the original via shared inode"
    )
    # Now actually edit the shadow and confirm the baseline is untouched.
    dst_py.write_text("def go(): return 'patched'\n")
    assert "baseline" in src_py.read_text(), (
        "Baseline source modified through shadow .py file — shadow_worktree "
        "is leaking writes"
    )


def test_so_files_are_symlinks(fake_vllm_source, tmp_path):
    """``.so`` files are symlinks to the baseline (no copy)."""
    dst = tmp_path / "slot_0"
    shadow_worktree(fake_vllm_source, dst)
    binary = dst / "vllm" / "_C.abi3.so"
    assert binary.is_symlink(), f"{binary} should be a symlink"
    assert binary.resolve() == (fake_vllm_source / "_C.abi3.so").resolve()
    nested = dst / "vllm" / "engine" / "_engine.cpython-312-x86_64-linux-gnu.so"
    assert nested.is_symlink()


def test_pycache_and_dot_dirs_skipped(fake_vllm_source, tmp_path):
    dst = tmp_path / "slot_0"
    shadow_worktree(fake_vllm_source, dst)
    pkg = dst / "vllm"
    assert not (pkg / "__pycache__").exists(), (
        "__pycache__ should be skipped to keep shadow lean"
    )
    assert not (pkg / ".internal").exists(), "dot-dirs should be skipped"


def test_shadow_marker_planted(fake_vllm_source, tmp_path):
    dst = tmp_path / "slot_0"
    shadow_worktree(fake_vllm_source, dst)
    marker = dst / SHADOW_MARKER
    assert marker.is_file()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["package_name"] == "vllm"
    assert is_shadow_worktree(dst)
    assert not is_shadow_worktree(fake_vllm_source)


def test_shadow_is_git_repo_with_baseline_commit(fake_vllm_source, tmp_path):
    dst = tmp_path / "slot_0"
    shadow_worktree(fake_vllm_source, dst)
    if not shutil.which("git"):
        pytest.skip("git not available")
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=dst,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "GEAK shadow_worktree baseline" in log.stdout


def test_gitignore_excludes_binaries(fake_vllm_source, tmp_path):
    dst = tmp_path / "slot_0"
    shadow_worktree(fake_vllm_source, dst)
    if not shutil.which("git"):
        pytest.skip("git not available")
    # ls-files must NOT include any *.so symlinks.
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=dst,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = set(out.stdout.splitlines())
    for p in tracked:
        assert not p.endswith(".so"), f".so file leaked into git index: {p}"
        assert not p.endswith(".pyd"), f".pyd file leaked into git index: {p}"
    assert "vllm/__init__.py" in tracked
    assert "vllm/engine/core.py" in tracked


def test_destination_replaced_when_pre_existing(fake_vllm_source, tmp_path):
    """A stale dst from a previous run is replaced cleanly."""
    dst = tmp_path / "slot_0"
    dst.mkdir()
    (dst / "stale.txt").write_text("old run")
    shadow_worktree(fake_vllm_source, dst)
    assert not (dst / "stale.txt").exists()
    assert (dst / "vllm" / "__init__.py").is_file()


def test_rejects_non_package_source(tmp_path):
    """``shadow_worktree`` should refuse a directory without ``__init__.py``."""
    bad = tmp_path / "not_a_package"
    bad.mkdir()
    (bad / "random.txt").write_text("nope")
    with pytest.raises(ValueError, match="not a Python package"):
        shadow_worktree(bad, tmp_path / "slot_0")
