"""Unit tests for ``minisweagent.run.orchestrator``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.orchestrator import _probe_preprocess_dir, run_orchestrator


class TestRunOrchestrator:
    def test_homogeneous_raises_not_implemented(self, tmp_path: Path) -> None:
        ctx = {"output_dir": str(tmp_path)}
        with pytest.raises(NotImplementedError, match="Homogeneous mode is not supported"):
            run_orchestrator(
                preprocess_ctx=ctx,
                gpu_ids=[0],
                model=MagicMock(),
                model_factory=MagicMock(),
                heterogeneous=False,
            )

    def test_heterogeneous_delegates_to_run_heterogeneous_orchestrator(self, tmp_path: Path) -> None:
        ctx = {"output_dir": str(tmp_path / "pp")}
        model = MagicMock()
        factory = MagicMock()
        sentinel = {"status": "ok"}

        with patch(
            "minisweagent.agents.heterogeneous.orchestrator.run_heterogeneous_orchestrator",
            return_value=sentinel,
        ) as mock_hetero:
            out = run_orchestrator(
                preprocess_ctx=ctx,
                gpu_ids=[0, 1],
                model=model,
                model_factory=factory,
                heterogeneous=True,
                max_rounds=3,
                start_round=2,
                console=None,
            )

        assert out is sentinel
        assert mock_hetero.call_count == 1
        args, kwargs = mock_hetero.call_args
        assert args[0] is ctx
        assert args[1] == [0, 1]
        assert args[2] is model
        assert args[3] is factory
        assert args[4] == Path(ctx["output_dir"]).resolve()
        assert args[5] == 3
        assert args[6] == 2
        assert callable(args[7])  # _print
        assert args[8] is None  # console

    def test_max_rounds_defaults_from_env_when_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = {"output_dir": str(tmp_path)}
        monkeypatch.setenv("GEAK_MAX_ROUNDS", "7")
        with patch(
            "minisweagent.agents.heterogeneous.orchestrator.run_heterogeneous_orchestrator",
            return_value={},
        ) as mock_hetero:
            run_orchestrator(
                preprocess_ctx=ctx,
                gpu_ids=[0],
                model=MagicMock(),
                model_factory=MagicMock(),
                heterogeneous=True,
                max_rounds=None,
            )
        assert mock_hetero.call_args[0][5] == 7

    def test_output_dir_override(self, tmp_path: Path) -> None:
        ctx = {"output_dir": str(tmp_path / "ignored")}
        override = tmp_path / "override_out"
        with patch(
            "minisweagent.agents.heterogeneous.orchestrator.run_heterogeneous_orchestrator",
            return_value={},
        ) as mock_hetero:
            run_orchestrator(
                preprocess_ctx=ctx,
                gpu_ids=[0],
                model=MagicMock(),
                model_factory=MagicMock(),
                heterogeneous=True,
                output_dir=override,
            )
        assert mock_hetero.call_args[0][4] == override


class TestProbePreprocessDir:
    def test_empty_dir_uses_preprocess_as_repo_root(self, tmp_path: Path) -> None:
        pc = _probe_preprocess_dir(tmp_path)
        assert pc.kernel_path == ""
        assert pc.repo_root == str(tmp_path)
        assert pc.harness_path == ""
        assert pc.preprocess_dir == str(tmp_path)
        assert pc.discovery is None

    def test_resolved_json_sets_kernel_and_finds_git_root(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        sub = repo / "kernels"
        sub.mkdir(parents=True)
        k = sub / "k.hip"
        k.write_text("kernel")
        (repo / ".git").mkdir()

        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "resolved.json").write_text(
            json.dumps(
                {
                    "local_file_path": str(k.resolve()),
                    "local_repo_path": None,
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert Path(pc.kernel_path).resolve() == k.resolve()
        assert Path(pc.repo_root).resolve() == repo.resolve()

    def test_resolved_json_no_git_uses_local_repo_path(self, tmp_path: Path) -> None:
        k = tmp_path / "solo.cu"
        k.write_text("x")
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "resolved.json").write_text(
            json.dumps(
                {
                    "local_file_path": str(k.resolve()),
                    "local_repo_path": "/workspace/myproject",
                }
            )
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.repo_root == "/workspace/myproject"

    def test_testcase_selection_sets_harness_path(self, tmp_path: Path) -> None:
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "testcase_selection.json").write_text(
            json.dumps({"harness_path": "/h/run.py"})
        )

        pc = _probe_preprocess_dir(pp)
        assert pc.harness_path == "/h/run.py"

    def test_discovery_json_loaded(self, tmp_path: Path) -> None:
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "discovery.json").write_text(json.dumps({"kernel": {"type": "triton"}}))

        pc = _probe_preprocess_dir(pp)
        assert pc.discovery == {"kernel": {"type": "triton"}}

    def test_optional_artifact_paths(self, tmp_path: Path) -> None:
        pp = tmp_path / "pp"
        pp.mkdir()
        (pp / "COMMANDMENT.md").write_text("cmd")
        (pp / "CODEBASE_CONTEXT.md").write_text("ctx")
        (pp / "baseline_metrics.json").write_text("{}")
        (pp / "profile.json").write_text("{}")

        pc = _probe_preprocess_dir(pp)
        assert pc.commandment_path == str(pp / "COMMANDMENT.md")
        assert pc.codebase_context_path == str(pp / "CODEBASE_CONTEXT.md")
        assert pc.baseline_metrics_path == str(pp / "baseline_metrics.json")
        assert pc.profiling_result_path == str(pp / "profile.json")
