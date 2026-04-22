"""Unit tests for ``minisweagent.run.postprocess.finalize_apply``.

Exercises the ``apply_commit_and_cleanup`` hook end-to-end against a real
temporary git repo, covering:

- Happy path (apply + commit + cleanup keeps final_report.json + .diff).
- Dirty-repo refusal (no apply, no cleanup).
- Apply failure (no commit, no cleanup).
- Commit failure (apply stays, no cleanup).
"""

# pytest fixtures legitimately shadow their names when injected into tests.
# pylint: disable=redefined-outer-name

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from minisweagent.agents.parallel_agent import BestPatchResult
from minisweagent.run.postprocess import finalize_apply
from minisweagent.run.postprocess.finalize_apply import apply_commit_and_cleanup

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fresh git repo with one tracked file and an initial commit."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _git(repo_dir, "init", "--initial-branch=main")
    _git(repo_dir, "config", "user.email", "test@example.com")
    _git(repo_dir, "config", "user.name", "Test")

    (repo_dir / "kernel.py").write_text("def run():\n    return 0\n")
    _git(repo_dir, "add", "kernel.py")
    _git(repo_dir, "commit", "-m", "initial")
    return repo_dir


@pytest.fixture
def good_patch(repo: Path) -> str:
    """Generate a real diff that applies cleanly to ``repo``."""
    (repo / "kernel.py").write_text("def run():\n    return 42\n")
    diff = subprocess.run(
        ["git", "diff"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # Revert the working tree so the test starts from the clean committed state.
    _git(repo, "checkout", "--", "kernel.py")
    return diff


@pytest.fixture
def output_dir(tmp_path: Path, good_patch: str) -> Path:
    """A populated per-run output_dir with final_report.json, a winning .diff, and noise."""
    out = tmp_path / "optimization_logs" / "kernel_20260101_000000"
    out.mkdir(parents=True)
    (out / "final_report.json").write_text(json.dumps({"best_speedup": 1.5, "summary": "ok"}))
    (out / "best_patch.diff").write_text(good_patch)

    # Noise we expect cleanup to prune.
    (out / "logs").mkdir()
    (out / "logs" / "run.log").write_text("noisy log content\n")
    (out / "results").mkdir()
    (out / "results" / "round_1").mkdir()
    (out / "results" / "round_1" / "best_results.json").write_text("{}")
    return out


def _result_for(output_dir: Path, patch_name: str = "best_patch.diff") -> BestPatchResult:
    return BestPatchResult(
        agent_id=0,
        patch_id="winning_patch",
        test_output="",
        best_speedup=1.5,
        best_patch_file=str(output_dir / patch_name),
        patch_dir=output_dir,
        llm_conclusion="Fused reduction tree and eliminated redundant loads.",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_applies_commits_and_cleans(repo: Path, output_dir: Path) -> None:
    result = _result_for(output_dir)

    apply_commit_and_cleanup(result, repo, output_dir)

    # Commit was created on top of "initial"
    log = (
        subprocess.run(
            ["git", "log", "--oneline"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )
    assert len(log) == 2
    assert "geak: apply best patch" in log[0]
    assert "winning_patch" in log[0]

    # Working tree reflects the applied change.
    assert (repo / "kernel.py").read_text() == "def run():\n    return 42\n"

    # Summary files retained, everything else pruned.
    assert output_dir.is_dir()
    assert (output_dir / "final_report.json").is_file()
    assert (output_dir / "best_patch.diff").is_file()
    remaining = {p.name for p in output_dir.iterdir()}
    assert remaining == {"final_report.json", "best_patch.diff"}


def test_happy_path_commit_body_references_report(repo: Path, output_dir: Path) -> None:
    result = _result_for(output_dir)

    apply_commit_and_cleanup(result, repo, output_dir)

    body = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "final_report.json" in body
    assert "Fused reduction tree" in body


# ---------------------------------------------------------------------------
# Dirty repo refusal
# ---------------------------------------------------------------------------


def test_dirty_repo_refuses_apply(repo: Path, output_dir: Path) -> None:
    (repo / "kernel.py").write_text("def run():\n    return 99\n")  # uncommitted change
    result = _result_for(output_dir)
    before_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    apply_commit_and_cleanup(result, repo, output_dir)

    after_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert before_sha == after_sha, "No new commit should be made when repo is dirty"
    # User's uncommitted change must remain untouched.
    assert (repo / "kernel.py").read_text() == "def run():\n    return 99\n"
    # Artifacts must remain intact for debugging.
    assert (output_dir / "logs" / "run.log").is_file()
    assert (output_dir / "results" / "round_1" / "best_results.json").is_file()


# ---------------------------------------------------------------------------
# Apply failure
# ---------------------------------------------------------------------------


def test_apply_failure_leaves_repo_and_artifacts_untouched(repo: Path, output_dir: Path) -> None:
    # Overwrite with a patch that cannot apply to HEAD (wrong context lines).
    bogus = (
        "diff --git a/kernel.py b/kernel.py\n"
        "index 0000000..1111111 100644\n"
        "--- a/kernel.py\n"
        "+++ b/kernel.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def totally_different():\n"
        "-    return 0\n"
        "+def totally_different():\n"
        "+    return 1\n"
    )
    (output_dir / "best_patch.diff").write_text(bogus)
    result = _result_for(output_dir)
    before_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    apply_commit_and_cleanup(result, repo, output_dir)

    after_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert before_sha == after_sha
    # Repo working tree unchanged.
    assert (repo / "kernel.py").read_text() == "def run():\n    return 0\n"
    # Artifacts preserved.
    assert (output_dir / "logs" / "run.log").is_file()
    assert (output_dir / "results" / "round_1" / "best_results.json").is_file()


# ---------------------------------------------------------------------------
# Commit failure
# ---------------------------------------------------------------------------


def test_commit_failure_preserves_apply_and_artifacts(
    repo: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _result_for(output_dir)

    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and len(cmd) >= 2 and cmd[0] == "git" and cmd[1] == "commit":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="simulated commit failure",
            )
        return real_run(cmd, *args, **kwargs)  # pylint: disable=subprocess-run-check

    monkeypatch.setattr(finalize_apply.subprocess, "run", fake_run)

    apply_commit_and_cleanup(result, repo, output_dir)

    # Apply must have landed in the working tree (no rollback).
    assert (repo / "kernel.py").read_text() == "def run():\n    return 42\n"
    # No new commit (only the initial one).
    log = (
        real_run(
            ["git", "log", "--oneline"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )
    assert len(log) == 1
    # Artifacts preserved so the user can investigate.
    assert (output_dir / "logs" / "run.log").is_file()
    assert (output_dir / "results" / "round_1" / "best_results.json").is_file()


# ---------------------------------------------------------------------------
# Precondition short-circuits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_bad",
    [
        lambda r, o: (None, r, o),
        lambda r, o: (
            BestPatchResult(agent_id=0, patch_id="x", test_output="", best_patch_file=None),
            r,
            o,
        ),
        lambda r, o: (
            BestPatchResult(
                agent_id=0,
                patch_id="x",
                test_output="",
                best_patch_file=str(o / "does_not_exist.diff"),
            ),
            r,
            o,
        ),
    ],
    ids=["result_none", "best_patch_file_missing", "patch_file_nonexistent"],
)
def test_precondition_short_circuits(repo: Path, output_dir: Path, make_bad) -> None:
    result, r, o = make_bad(repo, output_dir)
    before_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    apply_commit_and_cleanup(result, r, o)

    after_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert before_sha == after_sha
    # Nothing pruned.
    assert (output_dir / "logs" / "run.log").is_file()


@pytest.fixture
def repo_without_identity(tmp_path: Path) -> Path:
    """A git repo with no user.name/user.email (mimics a fresh container).

    ``_git`` uses author overrides for the initial commit so the fixture itself
    doesn't need an identity, but no ``git config user.*`` values are set on
    the repo afterwards.
    """
    repo_dir = tmp_path / "repo_no_id"
    repo_dir.mkdir()
    _git(repo_dir, "init", "--initial-branch=main")
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": "Bootstrap",
            "GIT_AUTHOR_EMAIL": "bootstrap@example.com",
            "GIT_COMMITTER_NAME": "Bootstrap",
            "GIT_COMMITTER_EMAIL": "bootstrap@example.com",
        }
    )
    (repo_dir / "kernel.py").write_text("def run():\n    return 0\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
        env=env,
    )
    # Explicitly scrub any local identity git may have inherited.
    subprocess.run(["git", "config", "--unset", "user.name"], cwd=str(repo_dir), capture_output=True, check=False)
    subprocess.run(["git", "config", "--unset", "user.email"], cwd=str(repo_dir), capture_output=True, check=False)
    return repo_dir


def test_commit_falls_back_to_default_identity(
    repo_without_identity: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a fresh container (no user.name/email), commit must still succeed with defaults."""
    # Simulate an isolated container env: no host-level git config visible.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.delenv("GIT_AUTHOR_NAME", raising=False)
    monkeypatch.delenv("GIT_AUTHOR_EMAIL", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_NAME", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_EMAIL", raising=False)
    monkeypatch.delenv("GEAK_GIT_AUTHOR_NAME", raising=False)
    monkeypatch.delenv("GEAK_GIT_AUTHOR_EMAIL", raising=False)

    # Build a matching output_dir with a real diff against repo_without_identity.
    (repo_without_identity / "kernel.py").write_text("def run():\n    return 42\n")
    diff = subprocess.run(
        ["git", "diff"],
        cwd=str(repo_without_identity),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    subprocess.run(
        ["git", "checkout", "--", "kernel.py"],
        cwd=str(repo_without_identity),
        capture_output=True,
        check=True,
    )

    out = tmp_path / "optimization_logs" / "kernel_run"
    out.mkdir(parents=True)
    (out / "final_report.json").write_text(json.dumps({"best_speedup": 1.5}))
    (out / "best_patch.diff").write_text(diff)
    result = _result_for(out)

    apply_commit_and_cleanup(result, repo_without_identity, out)

    log = subprocess.run(
        ["git", "log", "-1", "--pretty=format:%an <%ae>"],
        cwd=str(repo_without_identity),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert log == "GEAK Agent <geak@amd.com>"
    # The patched change is committed.
    assert (repo_without_identity / "kernel.py").read_text() == "def run():\n    return 42\n"


def test_commit_uses_geak_env_identity_override(
    repo_without_identity: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GEAK_GIT_AUTHOR_* env vars override the hard-coded default."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.delenv("GIT_AUTHOR_NAME", raising=False)
    monkeypatch.delenv("GIT_AUTHOR_EMAIL", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_NAME", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_EMAIL", raising=False)
    monkeypatch.setenv("GEAK_GIT_AUTHOR_NAME", "Custom Bot")
    monkeypatch.setenv("GEAK_GIT_AUTHOR_EMAIL", "bot@custom.example")

    (repo_without_identity / "kernel.py").write_text("def run():\n    return 7\n")
    diff = subprocess.run(
        ["git", "diff"],
        cwd=str(repo_without_identity),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    subprocess.run(
        ["git", "checkout", "--", "kernel.py"],
        cwd=str(repo_without_identity),
        capture_output=True,
        check=True,
    )

    out = tmp_path / "run_override"
    out.mkdir()
    (out / "final_report.json").write_text("{}")
    (out / "best_patch.diff").write_text(diff)
    result = _result_for(out)

    apply_commit_and_cleanup(result, repo_without_identity, out)

    author = subprocess.run(
        ["git", "log", "-1", "--pretty=format:%an <%ae>"],
        cwd=str(repo_without_identity),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert author == "Custom Bot <bot@custom.example>"


def test_commit_preserves_existing_identity(repo: Path, output_dir: Path) -> None:
    """When user.name / user.email are configured, we must not override them."""
    _git(repo, "config", "user.name", "Pre Configured")
    _git(repo, "config", "user.email", "pre@configured.example")
    result = _result_for(output_dir)

    apply_commit_and_cleanup(result, repo, output_dir)

    author = subprocess.run(
        ["git", "log", "-1", "--pretty=format:%an <%ae>"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert author == "Pre Configured <pre@configured.example>"


def test_non_git_repo_refused(tmp_path: Path, output_dir: Path) -> None:
    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()
    result = _result_for(output_dir)

    apply_commit_and_cleanup(result, not_a_repo, output_dir)

    assert (output_dir / "logs" / "run.log").is_file()


def test_cli_flag_threaded() -> None:
    """Smoke-check that --cleanup exists as a Typer option on the geak CLI."""
    from typer.testing import CliRunner

    from minisweagent.run import mini as mini_module

    runner = CliRunner()
    help_result = runner.invoke(mini_module.app, ["--help"])
    assert help_result.exit_code == 0
    assert "--cleanup" in help_result.stdout


# Silence unused-import warning for `patch` (imported for symmetry with sibling tests).
_ = patch
