"""End-to-end (real ``git``) capture test for the patch-capture fix.

Reproduces the failure that pinned GEAK's A/B speedup at 1.0x: a worktree that
holds the genuine ``get_default_config`` source edit PLUS the Triton JIT-cache
blobs (``.triton_cache_geak/<hash>/fused_moe_kernel.{hsaco,ttir,json}``),
``__pycache__``, and a stray ``.so`` that a round leaves behind. The captured
round patch must contain ONLY the source edit and must re-apply cleanly on a
fresh checkout of the baseline (the precondition for the reactor's A/B to ever
see the real kernel speedup).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from minisweagent.tools.save_and_test import SaveAndTestContext, SaveAndTestTool

_GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(_GIT is None, reason="git not available")


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "t@t.t", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    src = root / "python" / "sglang" / "fused_moe.py"
    src.parent.mkdir(parents=True)
    src.write_text("def get_default_config():\n    return {'BLOCK_SIZE_M': 64}\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-qm", "baseline", cwd=root)
    return src


def _pollute_with_jit_artifacts(root: Path) -> None:
    """Write exactly the kinds of artifacts that broke re-apply."""
    cache = root / ".triton_cache_geak" / "DEADBEEF"
    cache.mkdir(parents=True)
    (cache / "fused_moe_kernel.hsaco").write_bytes(b"\x7fELF binary")
    (cache / "fused_moe_kernel.ttir").write_text("triton ir dump\n")
    (cache / "fused_moe_kernel.json").write_text("{}\n")
    pyc = root / "python" / "sglang" / "__pycache__"
    pyc.mkdir()
    (pyc / "fused_moe.cpython-312.pyc").write_bytes(b"pyc")
    so_dir = root / "build_x"
    so_dir.mkdir()
    (so_dir / "common_ops.so").write_bytes(b"\x7fELF so")
    # a non-triton runtime cache dir caught by the *cache* substring rule
    inductor = root / "torchinductor_root" / "abc"
    inductor.mkdir(parents=True)
    (inductor / "out.py").write_text("# inductor codegen\n")


def _make_edit(src: Path) -> None:
    src.write_text("def get_default_config():\n    return {'BLOCK_SIZE_M': 256}\n")


def test_capture_is_source_only_and_reapplies_cleanly(tmp_path):
    repo = tmp_path / "wt"
    src = _init_repo(repo)
    _make_edit(src)
    _pollute_with_jit_artifacts(repo)

    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(repo),
            test_command=None,
            timeout=30,
            patch_output_dir=None,
            base_repo_path=None,
        )
    )
    patch = tool._get_patch_content()

    # The genuine edit is captured...
    assert "python/sglang/fused_moe.py" in patch
    assert "BLOCK_SIZE_M" in patch and "+    return {'BLOCK_SIZE_M': 256}" in patch
    # ...and NONE of the JIT / compiled / cache artifacts are.
    for forbidden in (
        ".triton_cache_geak", ".hsaco", ".ttir", "fused_moe_kernel.json",
        "__pycache__", ".pyc", "common_ops.so", "torchinductor_root",
    ):
        assert forbidden not in patch, f"captured patch leaked artifact: {forbidden}\n{patch}"

    # Re-apply cleanly on a FRESH checkout of the baseline (the reactor's A/B).
    fresh = tmp_path / "fresh"
    _init_repo(fresh)
    patch_file = tmp_path / "round.patch"
    patch_file.write_text(patch)
    # ``git apply --check`` is the atomic gate that previously aborted with
    # "already exists in working directory" when cache blobs rode along.
    proc = subprocess.run(
        ["git", "apply", "--check", str(patch_file)],
        cwd=fresh, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"patch did not re-apply cleanly: {proc.stderr}"
    subprocess.run(["git", "apply", str(patch_file)], cwd=fresh, check=True)
    assert "BLOCK_SIZE_M': 256" in (fresh / "python" / "sglang" / "fused_moe.py").read_text()


def test_capture_scoped_to_source_is_source_only(tmp_path):
    """With the editable set declared, the scoped capture is also artifact-free."""
    repo = tmp_path / "wt"
    src = _init_repo(repo)
    _make_edit(src)
    _pollute_with_jit_artifacts(repo)

    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(repo),
            test_command=None,
            timeout=30,
            patch_output_dir=None,
            base_repo_path=None,
            source_file_paths=["python/sglang/fused_moe.py"],
        )
    )
    patch = tool._get_patch_content()
    assert "fused_moe.py" in patch and "BLOCK_SIZE_M" in patch
    for forbidden in (".triton_cache_geak", ".hsaco", "__pycache__", "common_ops.so"):
        assert forbidden not in patch
