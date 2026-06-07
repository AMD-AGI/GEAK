"""Tests for the GEAK Layer-1/Layer-3 patch-capture fix in ``save_and_test``.

* Layer 3 (allowlist): when ``source_file_paths`` is configured, the round
  patch capture scopes ``git diff`` to that editable set so it can only ever
  record genuine source edits -- with a safe fall-back to the blanket worktree
  diff so an edit outside the declared set is never silently dropped.
* Layer 1 (relocate): ``_build_test_env`` injects an OUT-OF-WORKTREE
  ``TRITON_CACHE_DIR`` / ``GEAK_JIT_CACHE_DIR`` (overriding any stale in-worktree
  value) so the reactor's per-round ``git diff`` never sweeps compiled blobs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

from minisweagent.tools.save_and_test import SaveAndTestContext, SaveAndTestTool


def _tool(tmp_path: Path, *, source_file_paths=None, base_repo=None, env_vars=None) -> SaveAndTestTool:
    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command=None,
            timeout=10,
            patch_output_dir=None,
            base_repo_path=base_repo,
            env_vars=env_vars,
            source_file_paths=source_file_paths,
        )
    )
    return tool


def _cp(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args="git", returncode=returncode, stdout=stdout, stderr="")


def test_capture_scopes_to_declared_source(tmp_path):
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    (tmp_path / "fused_moe.py").write_text("new\n")
    tool = _tool(tmp_path, source_file_paths=["fused_moe.py"], base_repo=base_repo)

    patch = "diff --git a/fused_moe.py b/fused_moe.py\n@@ -1 +1 @@\n-old\n+new\n"
    cmds: list = []

    def _run(cmd, *a, **k):
        cmds.append(cmd)
        return _cp(patch)

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch("minisweagent.tools.save_and_test.subprocess.run", side_effect=_run),
    ):
        out = tool._get_patch_content()

    assert len(cmds) == 1, "scoped capture should not need the blanket fallback"
    assert "fused_moe.py" in cmds[0]
    assert "git add -N -- " in cmds[0]
    assert "git diff -- . " not in cmds[0], "must NOT use the blanket worktree diff"
    assert "+new" in out


def test_capture_falls_back_to_blanket_when_scoped_empty(tmp_path):
    """A real edit outside the declared set must never be dropped."""
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    (tmp_path / "kernel.py").write_text("x\n")  # declared, but unchanged here
    tool = _tool(tmp_path, source_file_paths=["kernel.py"], base_repo=base_repo)

    blanket = "diff --git a/other.py b/other.py\n@@ -1 +1 @@\n-old\n+new\n"
    side = [_cp("", 0), _cp(blanket, 0)]  # scoped empty -> blanket has the edit

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch("minisweagent.tools.save_and_test.subprocess.run", side_effect=side) as run,
    ):
        out = tool._get_patch_content()

    assert run.call_count == 2, "must fall back to the blanket diff"
    assert "+new" in out


def test_no_source_paths_uses_blanket(tmp_path):
    base_repo = tmp_path / "base"
    base_repo.mkdir()
    tool = _tool(tmp_path, source_file_paths=None, base_repo=base_repo)

    blanket = "diff --git a/kernel.py b/kernel.py\n@@ -1 +1 @@\n-old\n+new\n"
    cmds: list = []

    def _run(cmd, *a, **k):
        cmds.append(cmd)
        return _cp(blanket)

    with (
        mock.patch.object(SaveAndTestTool, "_is_git_repo", return_value=True),
        mock.patch("minisweagent.tools.save_and_test.subprocess.run", side_effect=_run),
    ):
        out = tool._get_patch_content()

    assert len(cmds) == 1
    assert "git diff -- . " in cmds[0], "blanket worktree diff when no editable set"
    assert "+new" in out


def test_build_test_env_relocates_triton_cache_outside_worktree(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    tool = _tool(wt)
    env = tool._build_test_env()
    assert "TRITON_CACHE_DIR" in env and "GEAK_JIT_CACHE_DIR" in env
    assert not env["TRITON_CACHE_DIR"].startswith(str(wt.resolve()))


def test_build_test_env_overrides_stale_in_worktree_cache(tmp_path):
    """A legacy in-worktree TRITON_CACHE_DIR (the root cause) must be overridden."""
    wt = tmp_path / "wt"
    wt.mkdir()
    stale = str(wt / ".triton_cache_geak")
    tool = _tool(wt, env_vars={"TRITON_CACHE_DIR": stale})
    env = tool._build_test_env()
    assert env["TRITON_CACHE_DIR"] != stale
    assert not env["TRITON_CACHE_DIR"].startswith(str(wt.resolve()))
