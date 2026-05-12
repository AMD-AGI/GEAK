"""Unit tests for ``minisweagent.run.postprocess.finalize_apply``.

Exercises the ``finalize_apply_and_cleanup`` hook end-to-end against a real
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
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from minisweagent.agents.parallel_agent import BestPatchResult
from minisweagent.run.postprocess import finalize_apply
from minisweagent.run.postprocess.finalize_apply import (
    apply_and_commit_best_patch,
    cleanup_run_artifacts,
    finalize_apply_and_cleanup,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


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

    finalize_apply_and_cleanup(result, repo, output_dir)

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

    finalize_apply_and_cleanup(result, repo, output_dir)

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


def test_dirty_repo_refuses_apply_only(repo: Path, output_dir: Path) -> None:
    """Dirty repo must skip apply but must not block the independent cleanup step."""
    (repo / "kernel.py").write_text("def run():\n    return 99\n")  # uncommitted change
    result = _result_for(output_dir)
    before_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    finalize_apply_and_cleanup(result, repo, output_dir)

    after_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert before_sha == after_sha, "No new commit should be made when repo is dirty"
    # User's uncommitted change must remain untouched.
    assert (repo / "kernel.py").read_text() == "def run():\n    return 99\n"
    # Cleanup is independent and still runs: only summary files survive.
    remaining = {p.name for p in output_dir.iterdir()}
    assert remaining == {"final_report.json", "best_patch.diff"}


def test_dirty_repo_with_no_cleanup_preserves_everything(repo: Path, output_dir: Path) -> None:
    """With --no-cleanup, a dirty repo leaves both the repo and the output_dir untouched."""
    (repo / "kernel.py").write_text("def run():\n    return 99\n")
    result = _result_for(output_dir)
    before_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    finalize_apply_and_cleanup(result, repo, output_dir, cleanup=False)

    after_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert before_sha == after_sha
    assert (repo / "kernel.py").read_text() == "def run():\n    return 99\n"
    assert (output_dir / "logs" / "run.log").is_file()
    assert (output_dir / "results" / "round_1" / "best_results.json").is_file()


# ---------------------------------------------------------------------------
# Apply failure
# ---------------------------------------------------------------------------


def _write_bogus_patch(output_dir: Path) -> None:
    """Replace the winning diff with one that cannot apply to HEAD."""
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


def test_apply_failure_still_runs_cleanup_independently(repo: Path, output_dir: Path) -> None:
    """Apply failure does not block cleanup (the two are independent)."""
    _write_bogus_patch(output_dir)
    result = _result_for(output_dir)
    before_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    finalize_apply_and_cleanup(result, repo, output_dir)

    after_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert before_sha == after_sha
    assert (repo / "kernel.py").read_text() == "def run():\n    return 0\n"
    # Cleanup still ran: only summary + winning .diff survive in output_dir.
    remaining = {p.name for p in output_dir.iterdir()}
    assert remaining == {"final_report.json", "best_patch.diff"}


def test_apply_failure_with_no_cleanup_preserves_artifacts(repo: Path, output_dir: Path) -> None:
    """Passing --no-cleanup keeps the full artifact dir intact on apply failure (debug mode)."""
    _write_bogus_patch(output_dir)
    result = _result_for(output_dir)
    before_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    finalize_apply_and_cleanup(result, repo, output_dir, cleanup=False)

    after_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert before_sha == after_sha
    assert (repo / "kernel.py").read_text() == "def run():\n    return 0\n"
    assert (output_dir / "logs" / "run.log").is_file()
    assert (output_dir / "results" / "round_1" / "best_results.json").is_file()


# ---------------------------------------------------------------------------
# Commit failure
# ---------------------------------------------------------------------------


def _install_commit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap subprocess.run so `git commit ...` always returns non-zero."""
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


def test_commit_failure_preserves_apply_with_no_cleanup(
    repo: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --no-cleanup and a commit failure, apply stays in working tree and artifacts survive."""
    _install_commit_failure(monkeypatch)
    result = _result_for(output_dir)

    finalize_apply_and_cleanup(result, repo, output_dir, cleanup=False)

    # Apply must have landed in the working tree (no rollback).
    assert (repo / "kernel.py").read_text() == "def run():\n    return 42\n"
    # No new commit (only the initial one). `_git` uses the unpatched
    # module-level ``subprocess.run``, so the commit-failure injection in
    # ``finalize_apply.subprocess`` does not affect it.
    log = _git(repo, "log", "--oneline").stdout.strip().splitlines()
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
def test_apply_precondition_short_circuits(repo: Path, output_dir: Path, make_bad) -> None:
    """Bad apply preconditions never produce a commit (cleanup skipped too for test clarity)."""
    result, r, o = make_bad(repo, output_dir)
    before_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    finalize_apply_and_cleanup(result, r, o, cleanup=False)

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

    finalize_apply_and_cleanup(result, repo_without_identity, out)

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

    finalize_apply_and_cleanup(result, repo_without_identity, out)

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

    finalize_apply_and_cleanup(result, repo, output_dir)

    author = subprocess.run(
        ["git", "log", "-1", "--pretty=format:%an <%ae>"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert author == "Pre Configured <pre@configured.example>"


def test_non_git_repo_refuses_apply(tmp_path: Path, output_dir: Path) -> None:
    """Non-git repo skips apply; cleanup is independent and runs."""
    not_a_repo = tmp_path / "plain_dir"
    not_a_repo.mkdir()
    result = _result_for(output_dir)

    finalize_apply_and_cleanup(result, not_a_repo, output_dir, cleanup=False)

    # Output dir untouched (cleanup=False).
    assert (output_dir / "logs" / "run.log").is_file()


# ---------------------------------------------------------------------------
# Independent-flag combinations
# ---------------------------------------------------------------------------


def test_apply_only_without_cleanup(repo: Path, output_dir: Path) -> None:
    """--apply-best-patch --no-cleanup: commit happens, output_dir fully preserved."""
    result = _result_for(output_dir)

    finalize_apply_and_cleanup(result, repo, output_dir, apply_best_patch=True, cleanup=False)

    log = _git(repo, "log", "--oneline").stdout.strip().splitlines()
    assert len(log) == 2
    # All artifacts survive.
    assert (output_dir / "logs" / "run.log").is_file()
    assert (output_dir / "results" / "round_1" / "best_results.json").is_file()


def test_cleanup_only_without_apply(repo: Path, output_dir: Path) -> None:
    """--no-apply-best-patch --cleanup: no commit, but artifacts pruned down to summary."""
    result = _result_for(output_dir)
    before_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    finalize_apply_and_cleanup(result, repo, output_dir, apply_best_patch=False, cleanup=True)

    after_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert before_sha == after_sha
    assert (repo / "kernel.py").read_text() == "def run():\n    return 0\n"  # repo untouched
    remaining = {p.name for p in output_dir.iterdir()}
    assert remaining == {"final_report.json", "best_patch.diff"}


def test_both_disabled_is_noop(repo: Path, output_dir: Path) -> None:
    """--no-apply-best-patch --no-cleanup: nothing happens."""
    result = _result_for(output_dir)
    before_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    finalize_apply_and_cleanup(result, repo, output_dir, apply_best_patch=False, cleanup=False)

    after_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert before_sha == after_sha
    assert (output_dir / "logs" / "run.log").is_file()
    assert (output_dir / "results" / "round_1" / "best_results.json").is_file()


def test_standalone_apply_api(repo: Path, output_dir: Path) -> None:
    """The standalone apply API returns a commit SHA on success."""
    result = _result_for(output_dir)

    sha = apply_and_commit_best_patch(result, repo)

    assert sha is not None and len(sha) >= 7
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    assert sha == head


def test_standalone_cleanup_api_without_patch_file(tmp_path: Path) -> None:
    """cleanup_run_artifacts works even when the BestPatchResult has no patch file."""
    out = tmp_path / "empty_run"
    out.mkdir()
    (out / "final_report.json").write_text(json.dumps({"best_speedup": None}))
    (out / "logs").mkdir()
    (out / "logs" / "run.log").write_text("logs")
    result = BestPatchResult(agent_id=0, patch_id="none", test_output="", best_patch_file=None)

    cleanup_run_artifacts(result, out)

    remaining = {p.name for p in out.iterdir()}
    assert remaining == {"final_report.json"}


def test_cli_flag_threaded() -> None:
    """Smoke-check that both independent flags are exposed on the geak CLI."""
    from typer.testing import CliRunner

    from minisweagent.run import mini as mini_module

    # Widen the virtual terminal to avoid Rich/Typer wrapping the option names
    # across lines in CI, and disable color so substring matches don't fight
    # ANSI escape codes. We still strip any surviving ANSI defensively.
    runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"})
    help_result = runner.invoke(mini_module.app, ["--help"])
    assert help_result.exit_code == 0
    plain = _strip_ansi(help_result.stdout)
    assert "--cleanup" in plain
    assert "--no-cleanup" in plain
    assert "--apply-best-patch" in plain
    assert "--no-apply-best-patch" in plain


# Silence unused-import warning for `patch` (imported for symmetry with sibling tests).
_ = patch
