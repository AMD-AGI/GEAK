"""Unit tests for evaluation-contract freezing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minisweagent.run.preprocess.contract_normalize import (
    build_evaluation_contract,
    infer_compile_command_from_eval,
)
from minisweagent.run.preprocess.phases.base import PhaseContext
from minisweagent.run.preprocess.phases.contract_resolution import ContractResolutionPhase


class TestInferCompileCommandFromEval:
    def test_none_and_empty(self) -> None:
        assert infer_compile_command_from_eval(None) is None
        assert infer_compile_command_from_eval("") is None
        assert infer_compile_command_from_eval("   ") is None

    def test_task_runner_style(self) -> None:
        ev = (
            "export ROCM_PATH=/opt/rocm && "
            "python3 scripts/task_runner.py compile && "
            "python3 scripts/task_runner.py correctness && "
            "python3 scripts/task_runner.py performance"
        )
        got = infer_compile_command_from_eval(ev)
        assert got is not None
        assert "compile" in got
        assert "correctness" not in got
        assert "performance" not in got

    def test_make_only_prefix(self) -> None:
        ev = "make -j8 && ./run_tests.sh"
        got = infer_compile_command_from_eval(ev)
        assert got == "make -j8"

    def test_no_build_token_returns_none(self) -> None:
        assert infer_compile_command_from_eval("pytest -q") is None


class TestBuildEvaluationContract:
    def test_shapes(self, tmp_path: Path) -> None:
        ctx = PhaseContext(
            kernel_url="x",
            output_dir=tmp_path,
            kernel_path=str(tmp_path / "k.hip"),
            repo_root=str(tmp_path),
            eval_command="make && python3 scripts/task_runner.py correctness",
        )
        lang = MagicMock()
        lang.name = "hip"
        ctx.language = lang
        ctx.discovery = {"tests": [], "kernel": {"type": "hip"}}
        contract = build_evaluation_contract(ctx)
        assert contract["version"] == 1
        assert contract["kernel_language"] == "hip"
        assert contract["compile_command"] == "make"
        assert contract["tier0_deterministic_compile"] is True


class TestContractResolutionPhase:
    def test_writes_json(self, tmp_path: Path) -> None:
        ctx = PhaseContext(
            kernel_url="local",
            output_dir=tmp_path,
            kernel_path=str(tmp_path / "k.py"),
            repo_root=str(tmp_path),
        )
        ContractResolutionPhase().run(ctx)
        out = tmp_path / "evaluation_contract.json"
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert ctx.evaluation_contract == data
        assert "contract_resolution" in ctx.phases_run

    def test_skips_without_kernel_path(self, tmp_path: Path) -> None:
        ctx = PhaseContext(kernel_url="x", output_dir=tmp_path, kernel_path="")
        ContractResolutionPhase().run(ctx)
        assert not (tmp_path / "evaluation_contract.json").exists()
        assert ctx.evaluation_contract is None
