"""Git-backed tests for the AVO lineage materialization (A1) + worktree clean (A2).

These verify that ``avo-v{N}`` tags reflect the *verified* patch (not whatever the
agent left in the worktree) so the next step resumes from the real best.
Skipped when git is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from minisweagent.run.avo.lineage_store import LineageStore
from minisweagent.run.avo.result import AttemptRecord, VariationResult

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    (repo / "kernel.txt").write_text("BASE\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


def _incremental_patch(repo: Path, new_content: str, out: Path) -> Path:
    """Produce a git-diff patch (HEAD -> new_content) like save_and_test does."""
    (repo / "kernel.txt").write_text(new_content, encoding="utf-8")
    _git(repo, "add", "-N", ".")
    patch_text = _git(repo, "diff").stdout
    # revert the working change; the patch is the artifact
    _git(repo, "checkout", "-f", ".")
    _git(repo, "clean", "-fd")
    out.write_text(patch_text, encoding="utf-8")
    return out


def test_seed_tags_baseline(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path, repo=repo)
    assert _git(repo, "rev-parse", "avo-v0").returncode == 0


def test_commit_tag_equals_verified_patch(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path, repo=repo)

    patch = _incremental_patch(repo, "OPT_v1\n", tmp_path / "v1.patch")
    # Simulate the agent leaving a *different/worse* edit in the worktree.
    (repo / "kernel.txt").write_text("WORSE_ATTEMPT\n", encoding="utf-8")

    result = VariationResult(
        step_index=1,
        step_dir=repo,
        strategy="opt",
        attempts=[AttemptRecord(correctness_passed=True, verified_speedup=1.5)],
        best_patch_path=patch,
        best_speedup=1.5,
        best_correct=True,
    )
    assert store.maybe_commit(result, repo=repo) is True
    assert store.best_id == "v1"

    # The avo-v1 tag must hold the VERIFIED patch, not the worktree's worse edit.
    _git(repo, "checkout", "-f", "avo-v1")
    assert (repo / "kernel.txt").read_text().strip() == "OPT_v1"


def test_reset_worktree_to_best_restores_verified(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    store = LineageStore(tmp_path / "avo_state")
    store.seed_from_baseline(tmp_path, repo=repo)
    patch = _incremental_patch(repo, "OPT_v1\n", tmp_path / "v1.patch")
    result = VariationResult(
        step_index=1,
        step_dir=repo,
        strategy="opt",
        attempts=[AttemptRecord(correctness_passed=True, verified_speedup=1.5)],
        best_patch_path=patch,
        best_speedup=1.5,
        best_correct=True,
    )
    store.maybe_commit(result, repo=repo)

    # Dirty the worktree with junk, then reset to best — junk must be gone and
    # the verified best restored.
    (repo / "junk.txt").write_text("garbage", encoding="utf-8")
    (repo / "kernel.txt").write_text("BROKEN\n", encoding="utf-8")
    store.reset_worktree_to_best(repo)
    assert (repo / "kernel.txt").read_text().strip() == "OPT_v1"
    assert not (repo / "junk.txt").exists()
