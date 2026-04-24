"""Tests for symlinking gitignored files into worktrees.

Covers _symlink_gitignored_files: compiled extensions (.so), generated
version files, etc. must be symlinked from the original repo so the
worktree can use them without rebuilding.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from minisweagent.run.task_file import (
    _resolve_output_root,
    _symlink_gitignored_files,
)


@pytest.fixture()
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


class TestSymlinkGitignoredFiles:
    def _setup_gitignore(self, git_repo: Path, patterns: str) -> None:
        gitignore = git_repo / ".gitignore"
        gitignore.write_text(patterns)
        subprocess.run(["git", "add", ".gitignore"], cwd=git_repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add gitignore"], cwd=git_repo, check=True, capture_output=True)

    def _make_worktree_dir(self, git_repo: Path) -> Path:
        worktree = git_repo / "optimization_logs" / "run" / "results" / "worktrees" / "slot_0"
        worktree.mkdir(parents=True)
        return worktree

    def test_so_files_are_symlinked(self, git_repo: Path) -> None:
        """Gitignored .so files should be symlinked into the worktree."""
        self._setup_gitignore(git_repo, "*.so\n")

        so_file = git_repo / "pkg" / "_ext.so"
        so_file.parent.mkdir(parents=True)
        so_file.write_bytes(b"\x7fELF")

        worktree = self._make_worktree_dir(git_repo)
        _symlink_gitignored_files(git_repo, worktree)

        destination = worktree / "pkg" / "_ext.so"
        assert destination.is_symlink()
        assert destination.resolve() == so_file.resolve()

    def test_version_py_is_symlinked(self, git_repo: Path) -> None:
        """Gitignored _version.py files should be symlinked into the worktree."""
        self._setup_gitignore(git_repo, "_version.py\n")

        version_file = git_repo / "mypkg" / "_version.py"
        version_file.parent.mkdir(parents=True)
        version_file.write_text("__version__ = '1.0.0'\n")

        worktree = self._make_worktree_dir(git_repo)
        _symlink_gitignored_files(git_repo, worktree)

        destination = worktree / "mypkg" / "_version.py"
        assert destination.is_symlink()
        assert destination.resolve() == version_file.resolve()

    def test_files_inside_output_dir_not_symlinked(self, git_repo: Path) -> None:
        """Gitignored files inside the output directory must be skipped."""
        self._setup_gitignore(git_repo, "*.so\n")

        so_file = git_repo / "optimization_logs" / "old_run" / "lib.so"
        so_file.parent.mkdir(parents=True)
        so_file.write_bytes(b"\x7fELF")

        worktree = git_repo / "optimization_logs" / "new_run" / "results" / "worktrees" / "slot_0"
        worktree.mkdir(parents=True)

        _symlink_gitignored_files(git_repo, worktree)

        assert not (worktree / "optimization_logs").exists()

    def test_existing_file_not_overwritten(self, git_repo: Path) -> None:
        """If a gitignored file already exists in the worktree, do not replace it."""
        self._setup_gitignore(git_repo, "*.so\n")

        so_file = git_repo / "pkg" / "_ext.so"
        so_file.parent.mkdir(parents=True)
        so_file.write_bytes(b"\x7fELF")

        worktree = self._make_worktree_dir(git_repo)
        (worktree / "pkg").mkdir(parents=True)
        existing = worktree / "pkg" / "_ext.so"
        existing.write_bytes(b"already here")

        _symlink_gitignored_files(git_repo, worktree)

        assert not existing.is_symlink()
        assert existing.read_bytes() == b"already here"

    def test_logs_symlinked_count(self, git_repo: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Should log the number of symlinked gitignored files."""
        self._setup_gitignore(git_repo, "*.so\n*.pyc\n")

        (git_repo / "a.so").write_bytes(b"\x7fELF")
        (git_repo / "b.pyc").write_bytes(b"compiled")

        worktree = self._make_worktree_dir(git_repo)

        target_logger = logging.getLogger("minisweagent.run.task_file")
        with caplog.at_level(logging.INFO, logger=target_logger.name):
            target_logger.addHandler(caplog.handler)
            try:
                _symlink_gitignored_files(git_repo, worktree)
            finally:
                target_logger.removeHandler(caplog.handler)

        assert "Symlinked 2 gitignored file(s)" in caplog.text
