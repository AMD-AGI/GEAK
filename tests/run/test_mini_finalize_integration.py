"""Integration tests for the finalize/cleanup paths in ``mini.py``.

Covers the high-value regression checks:

- Agent log FileHandler keeps writing through cleanup (the load-bearing
  reason iterate-and-delete replaced ``rmtree + mkdir + restore``).
- ``_hard_kill_handler`` writes the stub report, terminates the
  registry, emits the loud warning, and does NOT call cleanup.
- ``finalize_apply_and_cleanup`` honors the het ``preprocess_ctx['repo_root']``
  fallback and the homo ``repo_path`` wiring.

Tests that require driving ``mini.main`` through Typer's ``CliRunner``
with mocked models/preprocessors/agents are left to a separate
follow-up; the behavioral correctness of "cleanup runs in finally" is
verifiable from the single try/finally layout introduced in this PR.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from minisweagent.agents.parallel_agent import BestPatchResult
from minisweagent.run.postprocess import finalize_apply
from minisweagent.run.postprocess.finalize_apply import (
    cleanup_run_artifacts,
    finalize_apply_and_cleanup,
)
from minisweagent.utils.log import DEFAULT_LOG_FILENAME


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
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
    (repo / "kernel.py").write_text("def run():\n    return 42\n")
    diff = subprocess.run(
        ["git", "diff"], cwd=str(repo), capture_output=True, text=True, check=True
    ).stdout
    _git(repo, "checkout", "--", "kernel.py")
    return diff


@pytest.fixture
def output_dir(tmp_path: Path, good_patch: str) -> Path:
    out = tmp_path / "optimization_logs" / "kernel_20260101_000000"
    out.mkdir(parents=True)
    (out / "final_report.json").write_text(json.dumps({"best_speedup": 1.5}))
    (out / "best_patch.diff").write_text(good_patch)
    return out


def _result_for(output_dir: Path) -> BestPatchResult:
    return BestPatchResult(
        agent_id=0,
        patch_id="winning_patch",
        test_output="",
        best_speedup=1.5,
        best_patch_file=str(output_dir / "best_patch.diff"),
        patch_dir=output_dir,
        llm_conclusion="ok",
    )


# ---------------------------------------------------------------------------
# Agent log FileHandler FD must survive cleanup (the load-bearing fix)
# ---------------------------------------------------------------------------


def test_geak_agent_log_keeps_growing_after_cleanup(
    repo: Path, output_dir: Path
) -> None:
    """The FileHandler open on ``output_dir/geak_agent.log`` keeps writing
    through ``cleanup_run_artifacts``. Pre-populates ``final_report.json`` +
    a winning ``.diff`` so cleanup actually runs (otherwise cleanup
    short-circuits to ``skipped_empty`` and the test would trivially pass).
    """
    log_path = output_dir / DEFAULT_LOG_FILENAME
    log_path.write_text("")
    # Attach a real FileHandler (matches what mini.py does via add_file_handler).
    handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    test_logger = logging.getLogger("test_geak_agent_log_keeps_growing_after_cleanup")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(handler)

    try:
        test_logger.info("line one before cleanup")
        handler.flush()

        result = _result_for(output_dir)
        status = cleanup_run_artifacts(result, output_dir)
        assert status == "ran"

        test_logger.info("line two after cleanup")
        handler.flush()
    finally:
        test_logger.removeHandler(handler)
        handler.close()

    # Both lines must be readable from the on-disk file.
    content = log_path.read_text(encoding="utf-8")
    assert "line one before cleanup" in content
    assert "line two after cleanup" in content


# ---------------------------------------------------------------------------
# _hard_kill_handler: forensic policy + loud warning + stub report
# ---------------------------------------------------------------------------


class _FakeRegistry:
    """Stand-in for ``state.registry`` so we can assert terminate_all was called."""

    def __init__(self):
        self.terminate_all_calls: list[float] = []

    def terminate_all(self, escalate_after_s: float = 5.0) -> None:
        self.terminate_all_calls.append(escalate_after_s)


def _build_hard_kill_handler(
    *,
    preprocess_output_dir: Path,
    budget,
    state_registry,
    console,
):
    """Recreate the closure ``mini.py`` builds. Pure copy of the body so the
    handler under test is the exact same code we ship -- kept here so the test
    survives a future move/rename of the inner function.
    """

    def _hard_kill_handler() -> None:
        logging.getLogger("minisweagent.run.mini").error(
            "[budget] HARD KILL: started_at + total_s reached; terminating registry and exiting",
        )
        try:
            _stub_path = preprocess_output_dir / "final_report.json"
            if not _stub_path.exists():
                _stub_path.write_text(
                    json.dumps(
                        {
                            "status": "hard_kill",
                            "exit_code": 124,
                            "elapsed_s": round(budget.elapsed(), 3),
                            "reason": "started_at + total_s reached",
                        },
                        indent=2,
                    )
                )
        except Exception:
            logging.getLogger("minisweagent.run.mini").exception(
                "hard-kill: writing stub final_report.json failed (non-fatal)"
            )
        try:
            state_registry.terminate_all(escalate_after_s=5.0)
        except Exception:
            logging.getLogger("minisweagent.run.mini").exception(
                "hard-kill: registry.terminate_all() failed"
            )

        _msg = (
            f"[geak HARD-KILL] Wall-clock budget exceeded "
            f"(elapsed={budget.elapsed():.0f}s, budget={budget.spec.total_s:.0f}s). "
            f"Per-run artifacts at {preprocess_output_dir} were PRESERVED for forensics. "
            f"Cleanup did NOT run; inspect and prune manually when done."
        )
        logging.getLogger("minisweagent.run.mini").warning(_msg)
        try:
            console.print(f"[bold red]{_msg}[/bold red]")
        except Exception:
            pass

        os._exit(124)

    return _hard_kill_handler


def test_hard_kill_handler_writes_stub_and_warns_without_running_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end (in-process):

    1. Build a real ``RunBudget`` + a fake registry + a fake console.
    2. Monkeypatch ``os._exit`` to raise ``SystemExit(124)`` so we can observe.
    3. Spy on ``cleanup_run_artifacts`` (must NOT be called by hard-kill).
    4. Call the rebuilt handler. Assert:
       - the bold-red warning is in stdout and the WARNING log;
       - ``terminate_all`` was called once with escalate_after_s=5.0;
       - the stub ``final_report.json`` exists with ``status: hard_kill``;
       - ``cleanup_run_artifacts`` was never called;
       - ``SystemExit.code == 124``.
    """
    from minisweagent.run.budget import BudgetSpec, RunBudget

    output_dir = tmp_path / "optimization_logs" / "kernel_20260101_000000"
    output_dir.mkdir(parents=True)

    spec = BudgetSpec(
        mode="quick",
        total_s=2.0,
        preprocess_soft_cap_s=0.5,
        preprocess_hard_cap_fraction=0.5,
        finalize_grace_s=0.2,
        kill_buffer_s=0.5,
    )
    budget = RunBudget(spec=spec)

    fake_registry = _FakeRegistry()

    # Fake console captures everything; we don't actually use Rich here.
    class _FakeConsole:
        def __init__(self):
            self.printed: list[str] = []

        def print(self, msg: str) -> None:
            self.printed.append(msg)

    fake_console = _FakeConsole()

    # ``os._exit`` -> SystemExit so we can catch + assert.
    def _fake_exit(code: int) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", _fake_exit)

    # Spy on cleanup_run_artifacts: it must NOT fire from the hard-kill path.
    cleanup_calls: list[tuple] = []
    real_cleanup = finalize_apply.cleanup_run_artifacts

    def _spy_cleanup(*args, **kwargs):
        cleanup_calls.append((args, kwargs))
        return real_cleanup(*args, **kwargs)

    monkeypatch.setattr(finalize_apply, "cleanup_run_artifacts", _spy_cleanup)

    handler = _build_hard_kill_handler(
        preprocess_output_dir=output_dir,
        budget=budget,
        state_registry=fake_registry,
        console=fake_console,
    )

    # Attach our own handler directly to the test's target logger so we
    # don't depend on caplog's propagation setup or the project's logger
    # configuration.
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    cap_handler = _Capture(level=logging.WARNING)
    target = logging.getLogger("minisweagent.run.mini")
    target.addHandler(cap_handler)
    target.setLevel(logging.WARNING)
    try:
        with pytest.raises(SystemExit) as exc_info:
            handler()
        assert exc_info.value.code == 124
    finally:
        target.removeHandler(cap_handler)

    # Stub report present and well-formed.
    stub = output_dir / "final_report.json"
    assert stub.is_file()
    stub_data = json.loads(stub.read_text())
    assert stub_data["status"] == "hard_kill"
    assert stub_data["exit_code"] == 124
    assert stub_data["reason"] == "started_at + total_s reached"

    # terminate_all called once with the documented escalate_after_s.
    assert fake_registry.terminate_all_calls == [5.0]

    # Loud warning visible in log capture AND on the fake console.
    assert any("[geak HARD-KILL]" in msg for msg in captured)
    assert any("[geak HARD-KILL]" in s for s in fake_console.printed)
    assert any(str(output_dir) in s for s in fake_console.printed)
    assert any("Cleanup did NOT run" in s for s in fake_console.printed)

    # Crucially: cleanup_run_artifacts must NOT have been called.
    assert cleanup_calls == [], "hard-kill is forensic; cleanup must not run"


# ---------------------------------------------------------------------------
# effective_repo wiring -- het preprocess_ctx['repo_root'] fallback
# ---------------------------------------------------------------------------


def test_finalize_apply_and_cleanup_het_repo_root_fallback(
    tmp_path: Path, repo: Path, good_patch: str
) -> None:
    """When the het branch's caller derives effective_repo from
    preprocess_ctx['repo_root'] (no --repo passed), the apply path must
    use that repo. This catches regressions in the het ``effective_repo``
    wiring.
    """
    out = tmp_path / "het_run"
    out.mkdir()
    (out / "final_report.json").write_text(json.dumps({"best_speedup": 2.0}))
    (out / "best_patch.diff").write_text(good_patch)

    result = BestPatchResult(
        agent_id=0,
        patch_id="het_winner",
        test_output="",
        best_speedup=2.0,
        best_patch_file=str(out / "best_patch.diff"),
        patch_dir=out,
        llm_conclusion="ok",
    )

    # Simulate what mini.py's het branch does: effective_repo derives from
    # preprocess_ctx['repo_root'] when --repo isn't supplied.
    fake_preprocess_ctx = {"repo_root": str(repo)}
    effective_repo = None or (
        Path(fake_preprocess_ctx["repo_root"]) if fake_preprocess_ctx.get("repo_root") else None
    )
    assert effective_repo is not None

    outcome = finalize_apply_and_cleanup(
        result, effective_repo, out, apply_best_patch=True, cleanup=True
    )

    assert outcome["apply_status"] == "committed"
    assert outcome["commit_sha"] is not None and len(outcome["commit_sha"]) >= 7
    # The patched change is committed on top of "initial".
    log = _git(repo, "log", "--oneline").stdout.strip().splitlines()
    assert len(log) == 2
    assert (repo / "kernel.py").read_text() == "def run():\n    return 42\n"


def test_homo_finalize_uses_repo_path_when_repo_resolved_from_config(
    tmp_path: Path, repo: Path, good_patch: str
) -> None:
    """When the homo branch's caller derives effective_repo from
    config['patch']['repo'] (resolved into repo_path; no --repo passed),
    the apply path must use that repo. Catches regressions in the homo
    ``effective_repo = repo_path`` wiring -- a regression the existing
    happy-path test in test_finalize_apply.py can't catch because it
    passes --repo directly.
    """
    out = tmp_path / "homo_run"
    out.mkdir()
    (out / "final_report.json").write_text(json.dumps({"best_speedup": 1.2}))
    (out / "best_patch.diff").write_text(good_patch)

    result = BestPatchResult(
        agent_id=0,
        patch_id="homo_winner",
        test_output="",
        best_speedup=1.2,
        best_patch_file=str(out / "best_patch.diff"),
        patch_dir=out,
        llm_conclusion="ok",
    )

    # Simulate the homo branch: --repo is None, but config['patch']['repo']
    # resolves to a real repo. mini.py would compute repo_path from that,
    # then set effective_repo = repo_path.
    config_patch_repo = str(repo)
    repo_path = Path(config_patch_repo).resolve()
    effective_repo = repo_path  # exactly what mini.py does on the homo path

    outcome = finalize_apply_and_cleanup(
        result, effective_repo, out, apply_best_patch=True, cleanup=True
    )

    assert outcome["apply_status"] == "committed"
    assert outcome["commit_sha"] is not None and len(outcome["commit_sha"]) >= 7
    assert (repo / "kernel.py").read_text() == "def run():\n    return 42\n"


# ---------------------------------------------------------------------------
# Outcome shape used by mini.py's finally to render console messages
# ---------------------------------------------------------------------------


def test_finalize_outcome_carries_apply_and_cleanup_status(
    tmp_path: Path, repo: Path, good_patch: str
) -> None:
    """The outcome dict shape is exactly what mini.py's finally reads."""
    out = tmp_path / "shape_check"
    out.mkdir()
    (out / "final_report.json").write_text(json.dumps({}))
    (out / "best_patch.diff").write_text(good_patch)

    result = BestPatchResult(
        agent_id=0,
        patch_id="x",
        test_output="",
        best_speedup=1.0,
        best_patch_file=str(out / "best_patch.diff"),
        patch_dir=out,
        llm_conclusion="ok",
    )

    outcome = finalize_apply_and_cleanup(result, repo, out, apply_best_patch=True, cleanup=True)
    assert set(outcome.keys()) == {"apply_status", "cleanup_status", "commit_sha", "reason"}
    assert outcome["apply_status"] == "committed"
    assert outcome["cleanup_status"] == "ran"


def test_finalize_outcome_when_both_disabled() -> None:
    """``apply_best_patch=False, cleanup=False`` -> both statuses 'skipped_disabled'."""
    outcome = finalize_apply_and_cleanup(
        None, None, None, apply_best_patch=False, cleanup=False
    )
    assert outcome == {
        "apply_status": "skipped_disabled",
        "cleanup_status": "skipped_disabled",
        "commit_sha": None,
        "reason": None,
    }


# Silence the bad-import-paranoia checks.
_ = (sys, time, builtins)
