"""Regression tests for worktree copy helpers in task_file.py.

Covers the recursive-copy bug fixed in #189 / #181: previous GEAK run
artifacts under the output directory (e.g. optimization_logs/) must not
be copied into new worktrees.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from minisweagent.run.task_file import (
    _copy_nested_git_repos,
    _copy_untracked_files,
    _resolve_output_root,
)

# ---------------------------------------------------------------------------
# _resolve_output_root
# ---------------------------------------------------------------------------


class TestResolveOutputRoot:
    def test_worktree_inside_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        worktree = repo / "optimization_logs" / "run_1" / "results" / "worktrees" / "slot_0"
        result = _resolve_output_root(repo, worktree)
        assert result == repo / "optimization_logs"

    def test_worktree_outside_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        worktree = tmp_path / "elsewhere" / "slot_0"
        result = _resolve_output_root(repo, worktree)
        assert result is None

    def test_worktree_equals_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        result = _resolve_output_root(repo, repo)
        assert result is None

    def test_custom_output_directory(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        worktree = repo / "my_output" / "results" / "worktrees" / "slot_0"
        result = _resolve_output_root(repo, worktree)
        assert result == repo / "my_output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with one commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, check=True, capture_output=True,
    )
    tracked_file = repo / "tracked.py"
    tracked_file.write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo, check=True, capture_output=True,
    )
    return repo


# ---------------------------------------------------------------------------
# _copy_untracked_files
# ---------------------------------------------------------------------------


class TestCopyUntrackedFilesSkipsOutputRoot:
    def test_previous_run_artifacts_not_copied(self, git_repo: Path) -> None:
        """Files under optimization_logs/ from a prior run must be skipped."""
        old_run = git_repo / "optimization_logs" / "old_run" / "results" / "round_1"
        old_run.mkdir(parents=True)
        (old_run / "best_patch.diff").write_text("fake patch")

        worktree = git_repo / "optimization_logs" / "new_run" / "results" / "round_1" / "worktrees" / "slot_0"
        worktree.mkdir(parents=True)

        _copy_untracked_files(git_repo, worktree)

        assert not (worktree / "optimization_logs").exists()

    def test_regular_untracked_files_still_copied(self, git_repo: Path) -> None:
        """Untracked files outside the output directory must still be copied."""
        untracked = git_repo / "utils" / "helper.py"
        untracked.parent.mkdir(parents=True)
        untracked.write_text("helper = True\n")

        worktree = git_repo / "optimization_logs" / "run" / "results" / "round_1" / "worktrees" / "slot_0"
        worktree.mkdir(parents=True)

        _copy_untracked_files(git_repo, worktree)

        assert (worktree / "utils" / "helper.py").exists()
        assert (worktree / "utils" / "helper.py").read_text() == "helper = True\n"

    def test_worktree_outside_repo_copies_everything(self, git_repo: Path, tmp_path: Path) -> None:
        """When worktree is outside the repo, no output root filtering applies."""
        untracked = git_repo / "optimization_logs" / "data.txt"
        untracked.parent.mkdir(parents=True)
        untracked.write_text("data")

        worktree = tmp_path / "external_worktree"
        worktree.mkdir(parents=True)

        _copy_untracked_files(git_repo, worktree)

        assert (worktree / "optimization_logs" / "data.txt").exists()


# ---------------------------------------------------------------------------
# _copy_nested_git_repos
# ---------------------------------------------------------------------------


class TestCopyNestedGitReposSkipsOutputRoot:
    def test_nested_repo_inside_output_dir_not_copied(self, git_repo: Path) -> None:
        """Nested git repos inside the output directory must be skipped."""
        nested = git_repo / "optimization_logs" / "old_run" / "nested_repo"
        nested.mkdir(parents=True)
        (nested / ".git").mkdir()
        (nested / "code.py").write_text("nested = True\n")

        worktree = git_repo / "optimization_logs" / "new_run" / "results" / "worktrees" / "slot_0"
        worktree.mkdir(parents=True)

        _copy_nested_git_repos(git_repo, worktree)

        assert not (worktree / "optimization_logs").exists()

    def test_nested_repo_outside_output_dir_still_copied(self, git_repo: Path) -> None:
        """Nested git repos outside the output directory must still be copied."""
        nested = git_repo / "third_party" / "lib"
        nested.mkdir(parents=True)
        (nested / ".git").mkdir()
        (nested / "lib.py").write_text("lib = True\n")

        worktree = git_repo / "optimization_logs" / "run" / "results" / "worktrees" / "slot_0"
        worktree.mkdir(parents=True)

        _copy_nested_git_repos(git_repo, worktree)

        assert (worktree / "third_party" / "lib" / "lib.py").exists()
