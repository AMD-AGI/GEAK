from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from minisweagent.run.preprocess.phases.base import PhaseContext
from minisweagent.run.preprocess.phases.contract_resolution import ContractResolutionPhase


def test_contract_resolution_writes_evaluation_contract(tmp_path: Path) -> None:
    ctx = PhaseContext(
        kernel_url="local",
        output_dir=tmp_path,
        kernel_path=str(tmp_path / "kernel.hip"),
        repo_root=str(tmp_path),
        eval_command="make && python3 scripts/task_runner.py correctness",
    )
    ctx.language = SimpleNamespace(name="hip")
    ctx.discovery = {"kernel": {"type": "hip"}, "tests": []}

    ContractResolutionPhase().run(ctx)

    output = tmp_path / "evaluation_contract.json"
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["kernel_language"] == "hip"
    assert data["compile_command"] == "make"
    assert ctx.evaluation_contract == data
    assert "contract_resolution" in ctx.phases_run


def test_contract_resolution_skips_without_kernel_path(tmp_path: Path) -> None:
    ctx = PhaseContext(kernel_url="local", output_dir=tmp_path, kernel_path="")

    ContractResolutionPhase().run(ctx)

    assert not (tmp_path / "evaluation_contract.json").exists()
    assert ctx.evaluation_contract is None
    assert ctx.phases_skipped == [("contract_resolution", "no kernel_path")]


def test_phase_context_to_dict_includes_evaluation_contract(tmp_path: Path) -> None:
    ctx = PhaseContext(kernel_url="local", output_dir=tmp_path)
    ctx.evaluation_contract = {"version": 1}

    assert ctx.to_dict()["evaluation_contract"] == {"version": 1}

