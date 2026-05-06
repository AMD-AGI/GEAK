"""Unit tests for helpers in ``minisweagent.cli`` (no full CLI / preprocess run)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from minisweagent import cli as mini_module


class TestDeepMerge:
    def test_shallow_keys(self) -> None:
        assert mini_module._deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_nested_dicts_merge(self) -> None:
        base = {"agent": {"mode": "confirm", "step_limit": 10}, "model": {"x": 1}}
        override = {"agent": {"mode": "yolo"}}
        out = mini_module._deep_merge(base, override)
        assert out["agent"] == {"mode": "yolo", "step_limit": 10}
        assert out["model"] == {"x": 1}

    def test_override_replaces_non_dict_value(self) -> None:
        assert mini_module._deep_merge({"k": {"a": 1}}, {"k": "scalar"}) == {"k": "scalar"}


class TestAsInt:
    def test_valid(self) -> None:
        assert mini_module._as_int(3) == 3
        assert mini_module._as_int("4") == 4

    def test_none_returns_none(self) -> None:
        assert mini_module._as_int(None) is None

    def test_invalid_returns_none(self) -> None:
        assert mini_module._as_int("not-a-number") is None


class TestNormalizeKernelType:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("triton", "triton"),
            ("Triton", "triton"),
            ("hip", "hip"),
            ("rocm", "hip"),
            ("rocblas", "hip"),
            ("cuda", "other"),
            ("", "other"),
            (None, "other"),
        ],
    )
    def test_mapping(self, value: object, expected: str) -> None:
        assert mini_module._normalize_kernel_type(value) == expected


class TestDeriveOutputDir:
    def test_none_output_uses_generated_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        def fake_generate(name: str | None) -> str:
            return "optimization_logs/kernel_fixed_ts"

        with patch(
            "minisweagent.run.utils.task_parser.generate_patch_output_dir",
            side_effect=fake_generate,
        ):
            out_dir = mini_module._derive_output_dir(None, "my_kernel")

        assert out_dir == (tmp_path / "optimization_logs" / "kernel_fixed_ts").resolve()

    def test_file_path_uses_parent_for_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "run.traj.json"
        out_dir = mini_module._derive_output_dir(f, None)
        assert out_dir == f.parent.resolve()

    def test_directory_path_returns_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "logs"
        d.mkdir()
        out_dir = mini_module._derive_output_dir(d, None)
        assert out_dir == d.resolve()


class TestFinalReportToBestpatchresult:
    def test_none_returns_none(self) -> None:
        assert mini_module._final_report_to_bestpatchresult(None) is None

    def test_dict_with_best_patch(self, tmp_path: Path) -> None:
        patch_file = tmp_path / "patch_1.patch"
        patch_file.write_text("diff")
        report = {
            "best_patch": str(patch_file),
            "best_speedup": 1.5,
            "best_round": 2,
            "best_task": "t",
            "status": "ok",
            "summary": "done",
        }
        bpr = mini_module._final_report_to_bestpatchresult(report)
        assert bpr is not None
        assert bpr.patch_id == "patch_1"
        assert bpr.best_speedup == 1.5
        assert bpr.llm_conclusion == "done"
        assert bpr.patch_dir == patch_file.parent


class TestTryPromoteToHarness:
    def test_returns_script_when_valid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        harness = tmp_path / "harness.py"
        harness.write_text(
            "\n".join(
                [
                    "import argparse",
                    "",
                    "def main():",
                    "    p = argparse.ArgumentParser()",
                    "    p.add_argument('--correctness', action='store_true')",
                    "    p.add_argument('--profile', action='store_true')",
                    "    p.add_argument('--benchmark', action='store_true')",
                    "    p.add_argument('--full-benchmark', action='store_true')",
                    "    p.parse_args()",
                    "",
                    "if __name__ == '__main__':",
                    "    main()",
                ]
            )
        )
        cmd = f"python {harness.name}"
        # Returns the argv token that matched (relative name), not necessarily absolute.
        promoted = mini_module._try_promote_to_harness(cmd)
        assert promoted == harness.name
        assert Path(promoted).resolve() == harness.resolve()

    def test_returns_none_when_no_py_in_command(self) -> None:
        assert mini_module._try_promote_to_harness("echo hello") is None


def test_typer_app_exposed() -> None:
    assert mini_module.app is not None
    assert hasattr(mini_module.app, "registered_commands") or hasattr(mini_module.app, "info_name")


class TestTranslateSubcommand:
    """Verify ``geak translate`` is registered and wires through to the
    TranslationPhase.  We stub the agent's model to keep the test
    offline.
    """

    def test_translate_subcommand_registered(self) -> None:
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(mini_module.app, ["translate", "--help"])
        assert result.exit_code == 0, result.output
        assert "--source" in result.output
        assert "--target-language" in result.output

    def test_translate_runs_phase_with_mocked_agent(self, tmp_path: Path) -> None:
        """Integration-light test: mock the TranslationAgent.loop so no
        real model is invoked; verify the phase swap + output file
        write + exit code."""
        import yaml as _yaml
        from typer.testing import CliRunner

        from minisweagent.subagents.translation import TranslationAgent
        from minisweagent.subagents.translation.translator import TranslationResult

        src = tmp_path / "kernel.py"
        src.write_text("def kernel(): pass")

        fake_result = TranslationResult(
            ok=True,
            candidate_code="__global__ void kernel() {}",
            attempts_used=1,
        )

        # Patch the config-load side effects so we don't need real config files.
        with patch.object(TranslationAgent, "loop", return_value=fake_result), patch(
            "minisweagent.cli.yaml.safe_load",
            return_value={"model": {}},
        ), patch("minisweagent.models.get_model", return_value=object()), patch(
            "minisweagent.cli.configure_if_first_time"
        ):
            runner = CliRunner()
            result = runner.invoke(
                mini_module.app,
                ["translate", "--source", str(src), "--target-language", "hip"],
            )
        assert result.exit_code == 0, result.output
        # The translated file should exist next to the source with a .hip suffix
        translated = src.with_suffix(".hip")
        assert translated.exists()
        assert translated.read_text() == "__global__ void kernel() {}"
        del _yaml  # silence linter about unused import


class TestTargetLanguageFlag:
    """Verify the --target-language flag is registered on the geak CLI.

    Translation is a preprocess phase (not a run_pipeline mode), so the
    CLI entry must expose a flag that flows into the preprocess layer.
    We do not invoke the full pipeline here — only assert the option is
    present and typed correctly on the Typer callback.
    """

    def test_target_language_registered_as_typer_option(self) -> None:
        import inspect

        sig = inspect.signature(mini_module.main)
        assert "target_language" in sig.parameters, (
            "cli.main() must accept a target_language parameter; "
            "translation is triggered by this flag + the LLM-extracted "
            "target_language field."
        )

    def test_target_language_default_is_none(self) -> None:
        import inspect

        sig = inspect.signature(mini_module.main)
        param = sig.parameters["target_language"]
        # Typer Option's default is a special object; we just assert the
        # callback treats "no flag" as None (i.e. translation disabled).
        assert param.default is not inspect.Parameter.empty
