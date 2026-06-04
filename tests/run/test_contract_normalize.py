from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from minisweagent.run.preprocess.contract_normalize import (
    build_evaluation_contract,
    codebase_context_excerpt,
    discovery_digest,
    infer_compile_command_from_eval,
    is_amalgamation_command,
)


class TestInferCompileCommandFromEval:
    def test_none_and_empty(self) -> None:
        assert infer_compile_command_from_eval(None) is None
        assert infer_compile_command_from_eval("") is None
        assert infer_compile_command_from_eval("   ") is None

    def test_task_runner_style(self) -> None:
        command = (
            "export ROCM_PATH=/opt/rocm && "
            "python3 scripts/task_runner.py compile && "
            "python3 scripts/task_runner.py correctness && "
            "python3 scripts/task_runner.py performance"
        )

        inferred = infer_compile_command_from_eval(command)

        assert inferred is not None
        assert "compile" in inferred
        assert "correctness" not in inferred
        assert "performance" not in inferred

    def test_make_only_prefix(self) -> None:
        assert infer_compile_command_from_eval("make -j8 && ./run_tests.sh") == "make -j8"

    def test_no_build_token_returns_none(self) -> None:
        assert infer_compile_command_from_eval("pytest -q") is None


class TestIsAmalgamationCommand:
    def test_no_double_amp_is_not_amalgamation(self) -> None:
        # A single command (flag-less or flag-bearing) is never an amalgamation.
        assert is_amalgamation_command("python t.py") is False
        assert is_amalgamation_command("python t.py --benchmark") is False

    def test_non_build_double_amp_is_amalgamation(self) -> None:
        # Same script run twice with different settings, no build step.
        assert is_amalgamation_command("python t.py --mode 1 && python t.py --mode 2") is True
        # Flag-bearing variant must also be caught (it would otherwise yield a
        # harness path and slip past the flag-less backstop).
        assert (
            is_amalgamation_command(
                "python t.py --benchmark --opt-a && python t.py --benchmark --opt-b"
            )
            is True
        )

    def test_build_bearing_double_amp_is_not_amalgamation(self) -> None:
        # A genuine compile + run contract has a leading build prefix and may split.
        assert is_amalgamation_command("make && python t.py --benchmark") is False
        assert (
            is_amalgamation_command(
                "python3 scripts/task_runner.py compile && "
                "python3 scripts/task_runner.py correctness && "
                "python3 scripts/task_runner.py performance"
            )
            is False
        )


def test_discovery_digest_truncates_large_payload() -> None:
    digest = discovery_digest({"blob": "x" * 100}, max_chars=20)

    assert digest["truncated"] is True
    assert "preview" in digest


def test_codebase_context_excerpt_reads_and_truncates(tmp_path: Path) -> None:
    path = tmp_path / "CODEBASE_CONTEXT.md"
    path.write_text("abcdef", encoding="utf-8")

    excerpt = codebase_context_excerpt(str(path), max_chars=3)
    assert excerpt.startswith("abc\n")
    assert len(excerpt) > 4


def test_build_evaluation_contract_shape(tmp_path: Path) -> None:
    language = SimpleNamespace(name="hip")
    ctx = SimpleNamespace(
        language=language,
        eval_command="make && python3 scripts/task_runner.py correctness",
        kernel_path=str(tmp_path / "kernel.hip"),
        repo_root=str(tmp_path),
        harness=None,
        correctness_command=None,
        performance_command=None,
        discovery={"tests": [], "kernel": {"type": "hip"}},
        codebase_context_path=None,
    )

    contract = build_evaluation_contract(ctx)

    assert contract["version"] == 1
    assert contract["kernel_language"] == "hip"
    assert contract["compile_command"] == "make"
    assert contract["tier0_deterministic_compile"] is True
    assert json.dumps(contract, default=str)

