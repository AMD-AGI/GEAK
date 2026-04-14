"""Tests for multi-round starting_patch application (#110).

Verifies that cumulative patches (relative to HEAD) apply correctly
when the main repo has dirty working-tree changes from prior rounds.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path


def _git(args: list[str], cwd: Path, **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
        **kwargs,
    )


ORIGINAL_KERNEL = textwrap.dedent("""\
    def compute(x):
        result = x * 2        # slow multiplication
        result = result + 1    # add one
        return result
""")

ROUND1_KERNEL = textwrap.dedent("""\
    def compute(x):
        result = x << 1       # bitshift instead of multiply
        result = result + 1    # add one
        return result
""")

ROUND2_KERNEL = textwrap.dedent("""\
    def compute(x):
        result = x << 1       # bitshift instead of multiply
        result = result + True # fused add
        return result
""")


def _make_repo(tmp_path: Path) -> Path:
    """Create a git repo with the original kernel committed as HEAD."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "kernel.py").write_text(ORIGINAL_KERNEL)
    _git(["init"], repo)
    _git(["add", "."], repo)
    _git(["commit", "-m", "original"], repo)
    return repo


def _make_cumulative_patch(repo: Path, kernel_text: str, patch_path: Path) -> None:
    """Simulate save_and_test: write kernel, capture git diff from HEAD."""
    (repo / "kernel.py").write_text(kernel_text)
    result = _git(["diff", "--binary", "HEAD"], repo)
    patch_path.write_text(result.stdout)
    (repo / "kernel.py").write_text(ORIGINAL_KERNEL)
    _git(["checkout", "--", "."], repo)


class TestMultiRoundPatch:
    """Reproduce and verify the fix for issue #110."""

    def test_round2_patch_applies_on_clean_head(self, tmp_path: Path):
        """Cumulative round-2 patch applies on a bare HEAD worktree."""
        repo = _make_repo(tmp_path)
        patch = tmp_path / "round2.patch"

        (repo / "kernel.py").write_text(ROUND2_KERNEL)
        result = _git(["diff", "--binary", "HEAD"], repo)
        patch.write_text(result.stdout)
        _git(["checkout", "--", "."], repo)

        wt = tmp_path / "clean_wt"
        _git(["worktree", "add", "--detach", str(wt)], repo)
        _git(["apply", str(patch)], wt)

        assert (wt / "kernel.py").read_text() == ROUND2_KERNEL

    def test_dirty_sync_breaks_cumulative_patch(self, tmp_path: Path):
        """Demonstrates the bug: dirty sync + cumulative patch = failure."""
        repo = _make_repo(tmp_path)

        (repo / "kernel.py").write_text(ROUND1_KERNEL)

        round2_patch = tmp_path / "round2.patch"
        wt_for_patch = tmp_path / "patch_wt"
        _git(["worktree", "add", "--detach", str(wt_for_patch)], repo)
        (wt_for_patch / "kernel.py").write_text(ROUND2_KERNEL)
        result = _git(["diff", "--binary", "HEAD"], wt_for_patch)
        round2_patch.write_text(result.stdout)
        subprocess.run(["git", "worktree", "remove", str(wt_for_patch), "--force"], cwd=repo, capture_output=True)

        wt = tmp_path / "buggy_wt"
        _git(["worktree", "add", "--detach", str(wt)], repo)

        dirty_diff = _git(["diff", "--no-ext-diff", "--binary", "HEAD"], repo)
        if dirty_diff.stdout.strip():
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn", "--binary", "-"],
                cwd=wt,
                input=dirty_diff.stdout,
                capture_output=True,
                text=True,
            )

        apply_result = subprocess.run(
            ["git", "apply", str(round2_patch)],
            cwd=wt,
            capture_output=True,
            text=True,
        )
        assert apply_result.returncode != 0, "Expected patch to FAIL with dirty sync"
        assert "patch does not apply" in apply_result.stderr

    def test_clean_worktree_fixes_cumulative_patch(self, tmp_path: Path):
        """The fix: skip dirty sync, apply cumulative patch on clean HEAD."""
        repo = _make_repo(tmp_path)

        (repo / "kernel.py").write_text(ROUND1_KERNEL)

        round2_patch = tmp_path / "round2.patch"
        wt_for_patch = tmp_path / "patch_wt"
        _git(["worktree", "add", "--detach", str(wt_for_patch)], repo)
        (wt_for_patch / "kernel.py").write_text(ROUND2_KERNEL)
        result = _git(["diff", "--binary", "HEAD"], wt_for_patch)
        round2_patch.write_text(result.stdout)
        subprocess.run(["git", "worktree", "remove", str(wt_for_patch), "--force"], cwd=repo, capture_output=True)

        wt = tmp_path / "fixed_wt"
        _git(["worktree", "add", "--detach", str(wt)], repo)

        apply_result = subprocess.run(
            ["git", "apply", str(round2_patch)],
            cwd=wt,
            capture_output=True,
            text=True,
        )
        assert apply_result.returncode == 0, f"Patch should apply on clean HEAD: {apply_result.stderr}"
        assert (wt / "kernel.py").read_text() == ROUND2_KERNEL

    def test_untracked_files_break_patch(self, tmp_path: Path):
        """Type 2 bug: untracked files copied before patch causes 'already exists'."""
        repo = _make_repo(tmp_path)

        (repo / "baseline_metrics.json").write_text('{"latency": 0.1}')

        wt_for_patch = tmp_path / "patch_wt"
        _git(["worktree", "add", "--detach", str(wt_for_patch)], repo)
        (wt_for_patch / "baseline_metrics.json").write_text('{"latency": 0.1}')
        (wt_for_patch / "kernel.py").write_text(ROUND1_KERNEL)
        _git(["add", "-N", "."], wt_for_patch)
        result = subprocess.run(
            ["git", "diff", "--binary", "--", "."],
            cwd=wt_for_patch,
            capture_output=True,
            text=True,
        )
        patch = tmp_path / "with_artifacts.patch"
        patch.write_text(result.stdout)
        subprocess.run(["git", "worktree", "remove", str(wt_for_patch), "--force"], cwd=repo, capture_output=True)

        wt = tmp_path / "type2_wt"
        _git(["worktree", "add", "--detach", str(wt)], repo)

        import shutil

        for name in ["baseline_metrics.json"]:
            src = repo / name
            if src.exists():
                shutil.copy2(str(src), str(wt / name))

        apply_result = subprocess.run(
            ["git", "apply", str(patch)],
            cwd=wt,
            capture_output=True,
            text=True,
        )
        assert apply_result.returncode != 0, "Expected 'already exists' failure"
        assert "already exists" in apply_result.stderr

    def test_create_worktree_with_patch_integration(self, tmp_path: Path):
        """Integration test: create_worktree_with_patch uses clean HEAD."""
        from minisweagent.run.task_file import create_worktree_with_patch

        repo = _make_repo(tmp_path)

        (repo / "kernel.py").write_text(ROUND1_KERNEL)
        (repo / "baseline_metrics.json").write_text('{"latency": 0.1}')

        round2_patch = tmp_path / "round2.patch"
        wt_for_patch = tmp_path / "patch_wt"
        _git(["worktree", "add", "--detach", str(wt_for_patch)], repo)
        (wt_for_patch / "kernel.py").write_text(ROUND2_KERNEL)
        result = _git(["diff", "--binary", "HEAD"], wt_for_patch)
        round2_patch.write_text(result.stdout)
        subprocess.run(["git", "worktree", "remove", str(wt_for_patch), "--force"], cwd=repo, capture_output=True)

        wt = tmp_path / "integration_wt"
        create_worktree_with_patch(repo, wt, round2_patch)

        assert (wt / "kernel.py").read_text() == ROUND2_KERNEL
        assert not (wt / "baseline_metrics.json").exists(), (
            "Preprocessing artifacts should NOT be synced when starting_patch is used"
        )
