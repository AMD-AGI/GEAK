from __future__ import annotations

import subprocess
from pathlib import Path

from minisweagent.run.task_file import create_worktree


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_create_worktree_syncs_dirty_tracked_and_untracked_files(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    _run_git(repo, "init")

    tracked = repo / "tracked.txt"
    tracked.write_text("baseline\n")
    _run_git(repo, "add", "tracked.txt")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    tracked.write_text("dirty local edit\n")
    untracked = repo / "notes.txt"
    untracked.write_text("untracked helper\n")

    monkeypatch.setattr(
        "minisweagent.run.task_file._ensure_safe_directory",
        lambda *_args, **_kwargs: None,
    )

    worktree = create_worktree(repo, tmp_path / "worktree")

    assert (worktree / "tracked.txt").read_text() == "dirty local edit\n"
    assert (worktree / "notes.txt").read_text() == "untracked helper\n"
