"""Hard-fail behavior in postprocess evaluation when the COMMANDMENT contract breaks.

Covers:
  - subprocess returncode != 0 in CORRECTNESS / FULL_BENCHMARK / PROFILE
    raises ``CommandmentExecutionError``;
  - argparse "unrecognized arguments" stderr produces a contract-broken message;
  - kernel-level correctness failures (no contract-broken signature) keep the
    legacy ``correctness_failed`` round status and do NOT raise.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["bash"], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _setup_path():
    import sys

    repo = Path(__file__).resolve().parent.parent.parent
    if str(repo / "src") not in sys.path:
        sys.path.insert(0, str(repo / "src"))


# ---------------------------------------------------------------------------
# CORRECTNESS
# ---------------------------------------------------------------------------


class TestCorrectnessHardFail:
    def test_argparse_unrecognized_iterations_raises(self, tmp_path):
        from minisweagent.run.postprocess.evaluation import (
            CommandmentExecutionError,
            run_correctness_and_benchmark,
        )

        round_eval: dict = {}
        stderr = (
            "usage: harness.py [-h] (--correctness | --profile | --benchmark | --full-benchmark)\n"
            "harness.py: error: unrecognized arguments: --iterations 30\n"
        )
        with patch(
            "minisweagent.run.postprocess.evaluation.build_eval_script",
            return_value=str(tmp_path / "fake.sh"),
        ), patch(
            "minisweagent.run.postprocess.evaluation.subprocess.run",
            return_value=_completed(2, stderr=stderr),
        ):
            with pytest.raises(CommandmentExecutionError) as exc_info:
                run_correctness_and_benchmark(
                    eval_worktree=tmp_path,
                    eval_env={},
                    commandment_path=tmp_path / "COMMANDMENT.md",
                    pp_dir=tmp_path,
                    round_eval=round_eval,
                    round_num=1,
                )
        assert exc_info.value.section == "CORRECTNESS"
        assert exc_info.value.returncode == 2
        assert "unrecognized arguments" in exc_info.value.detail
        assert round_eval["status"] == "commandment_execution_failed"

    def test_subprocess_exception_raises_commandment_error(self, tmp_path):
        from minisweagent.run.postprocess.evaluation import (
            CommandmentExecutionError,
            run_correctness_and_benchmark,
        )

        round_eval: dict = {}
        with patch(
            "minisweagent.run.postprocess.evaluation.build_eval_script",
            return_value=str(tmp_path / "fake.sh"),
        ), patch(
            "minisweagent.run.postprocess.evaluation.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="bash", timeout=1),
        ):
            with pytest.raises(CommandmentExecutionError) as exc_info:
                run_correctness_and_benchmark(
                    eval_worktree=tmp_path,
                    eval_env={},
                    commandment_path=tmp_path / "COMMANDMENT.md",
                    pp_dir=tmp_path,
                    round_eval=round_eval,
                    round_num=1,
                )
        assert exc_info.value.section == "CORRECTNESS"
        assert exc_info.value.returncode is None

    def test_kernel_assertion_failure_does_not_raise(self, tmp_path):
        """A non-zero exit without contract-broken stderr is a kernel failure,
        not a contract failure. We keep the legacy round_eval status and return
        normally so the orchestrator can move to the next candidate."""
        from minisweagent.run.postprocess.evaluation import run_correctness_and_benchmark

        round_eval: dict = {}
        stderr = "AssertionError: tensor mismatch at index 0\n"
        with patch(
            "minisweagent.run.postprocess.evaluation.build_eval_script",
            return_value=str(tmp_path / "fake.sh"),
        ), patch(
            "minisweagent.run.postprocess.evaluation.subprocess.run",
            return_value=_completed(1, stderr=stderr),
        ):
            run_correctness_and_benchmark(
                eval_worktree=tmp_path,
                eval_env={},
                commandment_path=tmp_path / "COMMANDMENT.md",
                pp_dir=tmp_path,
                round_eval=round_eval,
                round_num=1,
            )
        assert round_eval["status"] == "correctness_failed"
        assert round_eval["correctness"]["returncode"] == 1


# ---------------------------------------------------------------------------
# BENCHMARK / FULL_BENCHMARK
# ---------------------------------------------------------------------------


class TestBenchmarkHardFail:
    def _commandment_passing_correctness(self, tmp_path: Path) -> Path:
        # Build a fake commandment file (we mock build_eval_script anyway).
        cm = tmp_path / "COMMANDMENT.md"
        cm.write_text("# placeholder\n")
        # Pretend a baseline file exists so the benchmark loop runs.
        (tmp_path / "full_benchmark_baseline.txt").write_text("GEAK_RESULT_LATENCY_MS=10.0\n")
        return cm

    def test_benchmark_subprocess_failure_raises(self, tmp_path):
        from minisweagent.run.postprocess.evaluation import (
            CommandmentExecutionError,
            run_correctness_and_benchmark,
        )

        cm = self._commandment_passing_correctness(tmp_path)
        round_eval: dict = {}

        # First subprocess call (CORRECTNESS) succeeds; second (FULL_BENCHMARK) fails.
        call_results = iter([
            _completed(0, stdout="ok"),
            _completed(3, stderr="harness.py: error: unrecognized arguments: --iterations 30\n"),
        ])
        with patch(
            "minisweagent.run.postprocess.evaluation.build_eval_script",
            return_value=str(tmp_path / "fake.sh"),
        ), patch(
            "minisweagent.run.postprocess.evaluation.subprocess.run",
            side_effect=lambda *a, **kw: next(call_results),
        ):
            with pytest.raises(CommandmentExecutionError) as exc_info:
                run_correctness_and_benchmark(
                    eval_worktree=tmp_path,
                    eval_env={},
                    commandment_path=cm,
                    pp_dir=tmp_path,
                    round_eval=round_eval,
                    round_num=1,
                )
        assert exc_info.value.section == "FULL_BENCHMARK"
        assert exc_info.value.returncode == 3
        assert "unrecognized arguments" in exc_info.value.detail
        assert round_eval["status"] == "commandment_execution_failed"


# ---------------------------------------------------------------------------
# PROFILE
# ---------------------------------------------------------------------------


class TestProfileHardFail:
    def test_profile_subprocess_failure_raises(self, tmp_path):
        from minisweagent.run.postprocess.evaluation import (
            CommandmentExecutionError,
            run_profile,
        )

        round_eval: dict = {}
        with patch(
            "minisweagent.run.postprocess.evaluation.build_eval_script",
            return_value=str(tmp_path / "fake.sh"),
        ), patch(
            "minisweagent.run.postprocess.evaluation.subprocess.run",
            return_value=_completed(127, stderr="kernel-profile: command not found\n"),
        ):
            with pytest.raises(CommandmentExecutionError) as exc_info:
                run_profile(
                    eval_worktree=tmp_path,
                    eval_env={},
                    commandment_path=tmp_path / "COMMANDMENT.md",
                    pp_dir=tmp_path,
                    round_eval=round_eval,
                    round_num=1,
                    results_dir=tmp_path / "results",
                )
        assert exc_info.value.section == "PROFILE"
        assert exc_info.value.returncode == 127
        assert "command not found" in exc_info.value.detail
        assert round_eval["status"] == "commandment_execution_failed"


# ---------------------------------------------------------------------------
# preflight_commandment_contract
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_preflight_raises_on_contract_broken_stderr(self, tmp_path, monkeypatch):
        from minisweagent.run.postprocess import evaluation

        monkeypatch.delenv("GEAK_SKIP_COMMANDMENT_PREFLIGHT", raising=False)

        cm = tmp_path / "COMMANDMENT.md"
        cm.write_text("## SETUP\n## CORRECTNESS\n")

        with patch(
            "minisweagent.run.postprocess.evaluation.build_eval_script",
            return_value=str(tmp_path / "fake.sh"),
        ), patch(
            "minisweagent.run.postprocess.evaluation.subprocess.run",
            return_value=_completed(
                2, stderr="harness.py: error: unrecognized arguments: --iterations 1\n"
            ),
        ):
            with pytest.raises(evaluation.CommandmentExecutionError) as exc_info:
                evaluation.preflight_commandment_contract(
                    cm,
                    repo_root=str(tmp_path),
                    harness_path=str(tmp_path / "h.py"),
                    gpu_id=0,
                )
        assert exc_info.value.section == "PREFLIGHT"
        assert "unrecognized arguments" in exc_info.value.detail

    def test_preflight_skipped_via_env(self, tmp_path, monkeypatch):
        from minisweagent.run.postprocess import evaluation

        monkeypatch.setenv("GEAK_SKIP_COMMANDMENT_PREFLIGHT", "1")
        # Even with a missing commandment the call returns silently.
        evaluation.preflight_commandment_contract(
            tmp_path / "missing.md",
            repo_root=str(tmp_path),
            harness_path=str(tmp_path / "h.py"),
            gpu_id=0,
        )

    def test_preflight_pass(self, tmp_path, monkeypatch):
        from minisweagent.run.postprocess import evaluation

        monkeypatch.delenv("GEAK_SKIP_COMMANDMENT_PREFLIGHT", raising=False)
        cm = tmp_path / "COMMANDMENT.md"
        cm.write_text("## SETUP\n## CORRECTNESS\n")

        with patch(
            "minisweagent.run.postprocess.evaluation.build_eval_script",
            return_value=str(tmp_path / "fake.sh"),
        ), patch(
            "minisweagent.run.postprocess.evaluation.subprocess.run",
            return_value=_completed(0, stdout="ok"),
        ):
            evaluation.preflight_commandment_contract(
                cm,
                repo_root=str(tmp_path),
                harness_path=str(tmp_path / "h.py"),
                gpu_id=0,
            )

    def test_preflight_missing_commandment_raises(self, tmp_path, monkeypatch):
        from minisweagent.run.postprocess import evaluation

        monkeypatch.delenv("GEAK_SKIP_COMMANDMENT_PREFLIGHT", raising=False)
        with pytest.raises(evaluation.CommandmentExecutionError):
            evaluation.preflight_commandment_contract(
                tmp_path / "missing.md",
                repo_root=str(tmp_path),
                harness_path=str(tmp_path / "h.py"),
                gpu_id=0,
            )
