"""Tests for orchestrator evaluation: round-best selection, eval worktree setup,
and start_round resume behaviour."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from minisweagent.run.orchestrator import (
    _auto_finalize,
    _evaluate_round_best,
    _merge_round_evaluation_into_final_report,
    _parse_total_kernel_time_ms,
    _parse_reported_speedup,
    _setup_eval_worktree,
    run_orchestrator,
)


# ---------------------------------------------------------------------------
# _parse_total_kernel_time_ms
# ---------------------------------------------------------------------------


class TestParseTotalKernelTimeMs:
    def test_standard_output(self):
        output = (
            "Benchmark Results:\n"
            "TOTAL_KERNEL_TIME_MS: 1.2412\n"
            "AVG_KERNEL_TIME_MS: 0.0496\n"
        )
        assert _parse_total_kernel_time_ms(output) == pytest.approx(1.2412)

    def test_no_match(self):
        assert _parse_total_kernel_time_ms("no relevant output here") is None

    def test_scientific_notation(self):
        assert _parse_total_kernel_time_ms("TOTAL_KERNEL_TIME_MS: 1.5e-2") == pytest.approx(0.015)

    def test_embedded_in_longer_output(self):
        output = (
            "Running benchmark on 25 shapes with 20 iterations...\n"
            "(1,2,1,4)                   0.0531\n"
            "------------------------------------------------------------\n"
            "TOTAL_KERNEL_TIME_MS: 1.1880\n"
            "AVG_KERNEL_TIME_MS: 0.0475\n"
        )
        assert _parse_total_kernel_time_ms(output) == pytest.approx(1.1880)


# ---------------------------------------------------------------------------
# _setup_eval_worktree – path resolution (Bug 1)
# ---------------------------------------------------------------------------


class TestSetupEvalWorktree:
    def test_eval_dir_is_absolute_even_with_relative_output_dir(self, tmp_path):
        """Bug 1: _setup_eval_worktree must resolve eval_dir to absolute so
        that `git worktree add` and later `subprocess.run(cwd=eval_dir)` both
        refer to the same location."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        relative_output = Path("relative_output")

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("minisweagent.run.orchestrator.subprocess.run", return_value=mock_result) as mock_run:
            result = _setup_eval_worktree(
                repo_root=str(repo_dir),
                patch_file="nonexistent.patch",
                output_dir=relative_output,
            )

        assert result.is_absolute(), f"eval_dir should be absolute, got: {result}"

        worktree_call = mock_run.call_args_list[0]
        worktree_cmd = worktree_call[0][0]
        worktree_path_arg = worktree_cmd[4]
        assert Path(worktree_path_arg).is_absolute(), (
            f"git worktree add path should be absolute, got: {worktree_path_arg}"
        )

    def test_non_git_repo_uses_copytree(self, tmp_path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "somefile.py").write_text("pass")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("minisweagent.run.orchestrator.shutil.copytree") as mock_copy:
            result = _setup_eval_worktree(
                repo_root=str(repo_dir),
                patch_file="nonexistent.patch",
                output_dir=output_dir,
            )

        assert result.is_absolute()
        mock_copy.assert_called_once()
        dest_arg = mock_copy.call_args[0][1]
        assert Path(dest_arg).is_absolute()


# ---------------------------------------------------------------------------
# _evaluate_round_best – winner selection (Bug 2)
# ---------------------------------------------------------------------------


def _make_task_dir(
    results_dir: Path,
    name: str,
    speedup: float,
    patch_id: str = "patch_0",
    kernel_time_ms: float | None = None,
) -> None:
    """Helper: create a fake task directory with best_results.json and
    optionally a test output file containing TOTAL_KERNEL_TIME_MS."""
    task_dir = results_dir / name
    task_dir.mkdir(parents=True)

    test_output_path = str(task_dir / f"{patch_id}_test.txt")
    patch_file_path = str(task_dir / f"{patch_id}.patch")

    best_results = {
        "best_patch_id": patch_id,
        "best_patch_speedup": speedup,
        "best_patch_file": patch_file_path,
        "best_patch_test_output": test_output_path,
    }
    (task_dir / "best_results.json").write_text(json.dumps(best_results))

    if kernel_time_ms is not None:
        test_content = (
            "Correctness: 25/25 passed\n"
            "CORRECTNESS TEST PASSED\n"
            "Benchmark Results:\n"
            f"TOTAL_KERNEL_TIME_MS: {kernel_time_ms:.4f}\n"
            f"AVG_KERNEL_TIME_MS: {kernel_time_ms / 25:.4f}\n"
        )
        (task_dir / f"{patch_id}_test.txt").write_text(test_content)


class TestEvaluateRoundBestSelection:
    def test_selects_lowest_kernel_time_over_highest_speedup(self, tmp_path):
        """Bug 2: When test outputs are available, the winner should be the
        candidate with the lowest absolute kernel time, not the highest
        self-reported speedup."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        results_dir = output_dir / "results" / "round_1"
        results_dir.mkdir(parents=True)

        _make_task_dir(results_dir, "agent-high-speedup", speedup=1.05, kernel_time_ms=1.20)
        _make_task_dir(results_dir, "agent-low-time", speedup=1.02, kernel_time_ms=1.10)

        ctx = {
            "output_dir": str(output_dir),
            "preprocess_dir": str(tmp_path / "no_commandment"),
        }
        (tmp_path / "no_commandment").mkdir()

        messages: list[str] = []
        result = _evaluate_round_best(ctx, 1, results_dir, messages.append)

        assert result is not None
        assert result["best_task"] == "agent-low-time"
        assert any("kernel_time" in m for m in messages), (
            f"Expected 'kernel_time' selection method in output, got: {messages}"
        )

    def test_falls_back_to_speedup_when_test_outputs_missing(self, tmp_path):
        """When test output files don't exist, fall back to highest speedup."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        results_dir = output_dir / "results" / "round_1"
        results_dir.mkdir(parents=True)

        _make_task_dir(results_dir, "agent-high-speedup", speedup=1.05, kernel_time_ms=None)
        _make_task_dir(results_dir, "agent-low-time", speedup=1.02, kernel_time_ms=None)

        ctx = {
            "output_dir": str(output_dir),
            "preprocess_dir": str(tmp_path / "no_commandment"),
        }
        (tmp_path / "no_commandment").mkdir()

        messages: list[str] = []
        result = _evaluate_round_best(ctx, 1, results_dir, messages.append)

        assert result is not None
        assert result["best_task"] == "agent-high-speedup"

    def test_mixed_availability_falls_back_to_speedup(self, tmp_path):
        """If only some candidates have kernel times, fall back to speedup."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        results_dir = output_dir / "results" / "round_1"
        results_dir.mkdir(parents=True)

        _make_task_dir(results_dir, "agent-a", speedup=1.05, kernel_time_ms=1.20)
        _make_task_dir(results_dir, "agent-b", speedup=1.02, kernel_time_ms=None)

        ctx = {
            "output_dir": str(output_dir),
            "preprocess_dir": str(tmp_path / "no_commandment"),
        }
        (tmp_path / "no_commandment").mkdir()

        messages: list[str] = []
        result = _evaluate_round_best(ctx, 1, results_dir, messages.append)

        assert result is not None
        assert result["best_task"] == "agent-a"

    def test_no_candidates_returns_none(self, tmp_path):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        results_dir = output_dir / "results" / "round_1"
        results_dir.mkdir(parents=True)

        ctx = {
            "output_dir": str(output_dir),
            "preprocess_dir": str(tmp_path / "no_commandment"),
        }
        (tmp_path / "no_commandment").mkdir()

        result = _evaluate_round_best(ctx, 1, results_dir, lambda m: None)
        assert result is None

    def test_skips_worktrees_directory(self, tmp_path):
        """The 'worktrees' directory should be ignored during scanning."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        results_dir = output_dir / "results" / "round_1"
        results_dir.mkdir(parents=True)

        (results_dir / "worktrees").mkdir()
        _make_task_dir(results_dir, "real-agent", speedup=1.03, kernel_time_ms=1.15)

        ctx = {
            "output_dir": str(output_dir),
            "preprocess_dir": str(tmp_path / "no_commandment"),
        }
        (tmp_path / "no_commandment").mkdir()

        result = _evaluate_round_best(ctx, 1, results_dir, lambda m: None)
        assert result is not None
        assert result["best_task"] == "real-agent"


class TestFinalReportVerification:
    def test_parse_reported_speedup_handles_percent_and_multiplier(self):
        assert _parse_reported_speedup("0.46%") == pytest.approx(1.0046)
        assert _parse_reported_speedup("1.16446x") == pytest.approx(1.16446)

    @patch("minisweagent.memory.integration.record_optimization_outcome")
    @patch("minisweagent.memory.cross_session_memory.classify_kernel_category", return_value="unknown")
    def test_merge_round_evaluation_clamps_no_improvement(
        self,
        mock_classify,
        mock_record,
        tmp_path,
    ):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        patch_dir = output_dir / "results" / "round_1" / "dispatch-path-check"
        patch_dir.mkdir(parents=True)
        patch_file = patch_dir / "patch_9.patch"
        patch_file.write_text("diff --git a/kernel b/kernel\n")

        initial_report = {
            "status": "complete",
            "summary": (
                "### Best Patch: wrong-task/patch_1\n"
                "- **Speedup**: 0.46%\n"
                "### Why Improvement is Limited\n"
            ),
            "best_patch": str(output_dir / "wrong.patch"),
            "total_speedup": "0.46%",
        }
        (output_dir / "final_report.json").write_text(json.dumps(initial_report))

        round_eval = {
            "round": 1,
            "best_task": "dispatch-path-check",
            "best_patch": str(patch_file),
            "benchmark_speedup": 1.004644,
            "full_benchmark": {
                "verified_speedup": 0.998989,
                "baseline_ms": 0.012445,
                "candidate_ms": 0.0124576,
            },
        }
        ctx = {
            "output_dir": str(output_dir),
            "kernel_path": "/tmp/kernel.hpp",
            "baseline_metrics": {"bottleneck": "balanced"},
        }

        merged = _merge_round_evaluation_into_final_report(
            ctx,
            output_dir,
            dict(initial_report),
            round_eval,
        )

        assert merged["best_patch"] == str(patch_file)
        assert merged["total_speedup"] == "1.0000x"
        assert merged["verified_speedup_raw"] == pytest.approx(0.998989)
        assert merged["verified_improvement"] is False
        assert "## Verified Final Selection" in merged["summary"]
        assert "- Best patch: dispatch-path-check/patch_9" in merged["summary"]
        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["speedup_achieved"] == pytest.approx(1.0)
        assert mock_record.call_args.kwargs["success"] is False

    @patch("minisweagent.memory.integration.record_optimization_outcome")
    @patch("minisweagent.memory.cross_session_memory.classify_kernel_category", return_value="normalization")
    def test_merge_round_evaluation_uses_verified_positive_speedup(
        self,
        mock_classify,
        mock_record,
        tmp_path,
    ):
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        patch_dir = output_dir / "results" / "round_1" / "split-k-reduction-rewrite"
        patch_dir.mkdir(parents=True)
        patch_file = patch_dir / "patch_7.patch"
        patch_file.write_text("diff --git a/kernel b/kernel\n")

        report = {
            "status": "complete",
            "summary": "### Best Patch: stale-task/patch_1\n- **Speedup**: 1.16x\n",
            "best_patch": str(output_dir / "stale.patch"),
            "total_speedup": "1.16x",
        }

        round_eval = {
            "round": 1,
            "best_task": "split-k-reduction-rewrite",
            "best_patch": str(patch_file),
            "benchmark_speedup": 1.16446,
            "full_benchmark": {
                "verified_speedup": 1.14731,
                "baseline_ms": 0.053103,
                "candidate_ms": 0.046268,
            },
        }
        ctx = {
            "output_dir": str(output_dir),
            "kernel_path": "/tmp/kernel.py",
            "baseline_metrics": {"bottleneck": "latency"},
        }

        merged = _merge_round_evaluation_into_final_report(
            ctx,
            output_dir,
            report,
            round_eval,
        )

        assert merged["best_patch"] == str(patch_file)
        assert merged["total_speedup"] == "1.1473x"
        assert merged["verified_improvement"] is True
        assert "- Best patch: split-k-reduction-rewrite/patch_7" in merged["summary"]
        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["speedup_achieved"] == pytest.approx(1.14731)
        assert mock_record.call_args.kwargs["success"] is True

    @patch("minisweagent.memory.integration.record_optimization_outcome")
    @patch("minisweagent.memory.cross_session_memory.classify_kernel_category", return_value="unknown")
    def test_auto_finalize_prefers_best_verified_round_over_best_patch_speedup(
        self,
        mock_classify,
        mock_record,
        tmp_path,
    ):
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        round_2_dir = output_dir / "results" / "round_2"
        round_3_dir = output_dir / "results" / "round_3"
        round_2_dir.mkdir(parents=True)
        round_3_dir.mkdir(parents=True)

        _make_task_dir(round_2_dir, "kernel_optimization", speedup=1.360269, patch_id="patch_9")
        _make_task_dir(round_3_dir, "kernel_optimization", speedup=1.250000, patch_id="patch_2")

        round_2_patch = round_2_dir / "kernel_optimization" / "patch_9.patch"
        round_3_patch = round_3_dir / "kernel_optimization" / "patch_2.patch"
        round_2_patch.write_text("diff --git a/kernel.py b/kernel.py\n")
        round_3_patch.write_text("diff --git a/kernel.py b/kernel.py\n")

        (output_dir / "round_2_evaluation.json").write_text(
            json.dumps(
                {
                    "round": 2,
                    "best_task": "kernel_optimization",
                    "best_patch": str(round_2_patch),
                    "benchmark_speedup": 1.360269,
                    "full_benchmark": {
                        "verified_speedup": 1.3160,
                        "baseline_ms": 0.0404,
                        "candidate_ms": 0.0307,
                    },
                }
            )
        )
        (output_dir / "round_3_evaluation.json").write_text(
            json.dumps(
                {
                    "round": 3,
                    "best_task": "kernel_optimization",
                    "best_patch": str(round_3_patch),
                    "benchmark_speedup": 1.250000,
                    "full_benchmark": {
                        "verified_speedup": 1.3333,
                        "baseline_ms": 0.0404,
                        "candidate_ms": 0.0303,
                    },
                }
            )
        )

        ctx = {
            "output_dir": str(output_dir),
            "kernel_path": "/tmp/topk/kernel.py",
            "baseline_metrics": {"bottleneck": "latency"},
        }

        messages: list[str] = []
        report = _auto_finalize(ctx, messages.append)

        assert report["best_round"] == "round_3"
        assert report["best_task"] == "kernel_optimization"
        assert report["best_patch"] == str(round_3_patch)
        assert report["best_speedup"] == pytest.approx(1.3333)
        assert report["best_speedup_verified"] == pytest.approx(1.3333)
        assert report["total_speedup"] == "1.3333x"
        assert report["verified_improvement"] is True
        assert report["best_patch_analysis"].startswith("Verified FULL_BENCHMARK:")
        assert report["summary"].startswith("## Verified Final Selection")
        assert "round_2" not in report["summary"]
        assert "patch_9" not in report["summary"]

        persisted = json.loads((output_dir / "final_report.json").read_text())
        assert persisted["best_round"] == "round_3"
        assert persisted["best_patch"] == str(round_3_patch)
        assert persisted["total_speedup"] == "1.3333x"

        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["speedup_achieved"] == pytest.approx(1.3333)
        assert mock_record.call_args.kwargs["success"] is True


# ---------------------------------------------------------------------------
# run_orchestrator – start_round resume behaviour
# ---------------------------------------------------------------------------


def _minimal_preprocess_ctx(tmp_path: Path) -> dict:
    """Build a minimal preprocess context for testing run_orchestrator."""
    pp_dir = tmp_path / "pp"
    pp_dir.mkdir(parents=True, exist_ok=True)
    return {
        "kernel_path": str(pp_dir / "kernel.py"),
        "repo_root": str(pp_dir),
        "test_command": "echo ok",
        "discovery": {},
        "profiling": {},
        "baseline_metrics": {},
        "commandment": "",
        "output_dir": str(tmp_path / "output"),
    }


class TestStartRound:
    """Verify that start_round > 1 skips exploration, loads prior evals,
    and begins the round loop at the correct number."""

    @patch("minisweagent.run.orchestrator._run_llm_steps", return_value={"status": "done"})
    @patch("minisweagent.run.orchestrator._evaluate_round_best", return_value=None)
    def test_start_round_1_runs_exploration(
        self, mock_eval, mock_llm, tmp_path,
    ):
        """Default start_round=1 should invoke exploration phase."""
        ctx = _minimal_preprocess_ctx(tmp_path)
        out = tmp_path / "output"
        out.mkdir(parents=True, exist_ok=True)

        mock_model = MagicMock()
        mock_model_factory = MagicMock()

        run_orchestrator(
            ctx, [0], mock_model, mock_model_factory,
            output_dir=out, max_rounds=1, start_round=1,
            heterogeneous=True,
        )

        phases = [c.kwargs.get("phase") or c[1].get("phase", "")
                  for c in mock_llm.call_args_list]
        assert "explore" in phases, f"Exploration phase expected, got phases: {phases}"

    @patch("minisweagent.run.orchestrator._run_llm_steps", return_value={"status": "done"})
    @patch("minisweagent.run.orchestrator._evaluate_round_best", return_value=None)
    def test_start_round_2_skips_exploration(
        self, mock_eval, mock_llm, tmp_path,
    ):
        """start_round=2 should skip exploration and go straight to round 2."""
        ctx = _minimal_preprocess_ctx(tmp_path)
        out = tmp_path / "output"
        out.mkdir(parents=True, exist_ok=True)

        mock_model = MagicMock()
        mock_model_factory = MagicMock()

        run_orchestrator(
            ctx, [0], mock_model, mock_model_factory,
            output_dir=out, max_rounds=2, start_round=2,
            heterogeneous=True,
        )

        phases = [c.kwargs.get("phase") or c[1].get("phase", "")
                  for c in mock_llm.call_args_list]
        assert "explore" not in phases, f"Exploration should be skipped, got phases: {phases}"
        assert "round_2" in phases, f"Round 2 expected, got phases: {phases}"
        assert "round_1" not in phases, f"Round 1 should be skipped, got phases: {phases}"

    @patch("minisweagent.run.orchestrator._run_llm_steps", return_value=None)
    @patch("minisweagent.run.orchestrator._evaluate_round_best", return_value=None)
    def test_prior_round_eval_loaded_into_ctx(
        self, mock_eval, mock_llm, tmp_path,
    ):
        """Prior round evaluation JSON should be loaded into the internal ctx
        (observable via what's passed to _evaluate_round_best)."""
        pctx = _minimal_preprocess_ctx(tmp_path)
        out = tmp_path / "output"
        out.mkdir(parents=True, exist_ok=True)

        eval_data = {"round": 1, "best_task": "agent-a", "benchmark_speedup": 1.05}
        (out / "round_1_evaluation.json").write_text(json.dumps(eval_data))

        mock_model = MagicMock()
        mock_model_factory = MagicMock()

        mock_llm.side_effect = [None, {"status": "done"}]

        run_orchestrator(
            pctx, [0], mock_model, mock_model_factory,
            output_dir=out, max_rounds=3, start_round=2,
            heterogeneous=True,
        )

        # _evaluate_round_best receives the internal ctx as its first arg
        assert mock_eval.call_count >= 1
        internal_ctx = mock_eval.call_args_list[0][0][0]
        assert "round_1_eval" in internal_ctx
        assert internal_ctx["round_1_eval"]["best_task"] == "agent-a"

    @patch("minisweagent.run.orchestrator._run_llm_steps", return_value=None)
    @patch("minisweagent.run.orchestrator._evaluate_round_best", return_value=None)
    def test_prior_eval_injected_into_messages(
        self, mock_eval, mock_llm, tmp_path,
    ):
        """Prior round eval should appear in the messages list passed to the LLM."""
        ctx = _minimal_preprocess_ctx(tmp_path)
        out = tmp_path / "output"
        out.mkdir(parents=True, exist_ok=True)

        eval_data = {"round": 1, "best_task": "agent-a", "benchmark_speedup": 1.05}
        (out / "round_1_evaluation.json").write_text(json.dumps(eval_data))

        mock_model = MagicMock()
        mock_model_factory = MagicMock()

        mock_llm.side_effect = [None, {"status": "done"}]

        run_orchestrator(
            ctx, [0], mock_model, mock_model_factory,
            output_dir=out, max_rounds=3, start_round=2,
            heterogeneous=True,
        )

        first_call_messages = mock_llm.call_args_list[0][0][1]
        eval_msgs = [m for m in first_call_messages
                     if "prior run" in m.get("content", "")]
        assert len(eval_msgs) == 1, f"Expected 1 prior eval message, got {len(eval_msgs)}"
        assert "agent-a" in eval_msgs[0]["content"]

    @patch("minisweagent.run.orchestrator._run_llm_steps", return_value={"status": "done"})
    @patch("minisweagent.run.orchestrator._evaluate_round_best", return_value=None)
    def test_missing_prior_eval_is_tolerated(
        self, mock_eval, mock_llm, tmp_path,
    ):
        """Missing round_1_evaluation.json should not crash; just skip it."""
        ctx = _minimal_preprocess_ctx(tmp_path)
        out = tmp_path / "output"
        out.mkdir(parents=True, exist_ok=True)

        mock_model = MagicMock()
        mock_model_factory = MagicMock()

        run_orchestrator(
            ctx, [0], mock_model, mock_model_factory,
            output_dir=out, max_rounds=2, start_round=2,
            heterogeneous=True,
        )

        assert "round_1_eval" not in ctx
