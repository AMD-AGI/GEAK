"""Unit tests for helpers in ``minisweagent.run.mini`` (no full CLI / preprocess run)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisweagent.run import mini as mini_module


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


class TestPatchConfigWarnings:
    def test_warns_on_unknown_patch_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        capsys.readouterr()

        mini_module._warn_unknown_patch_keys({"patch": {"profile_quickk": True}})

        captured = capsys.readouterr()
        assert "Unknown patch config key(s): profile_quickk" in captured.out


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


def test_heterogeneous_path_threads_patch_profile_flags_to_orchestrator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import minisweagent.run.utils.task_parser as task_parser_module

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "mini_kernel_strategy_list.yaml").write_text("agent: {}\nenv: {}\nmodel: {}\npatch: {}\ntools: {}\n")
    geak_yaml = config_dir / "geak.yaml"
    geak_yaml.write_text("patch:\n  profile_every_patch: true\n  profile_quick: false\n")

    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text("print('kernel')\n")

    monkeypatch.setattr(mini_module, "builtin_config_dir", config_dir)
    monkeypatch.setattr(mini_module.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(task_parser_module, "parse_pipeline_params", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mini_module, "parse_task_info", lambda *_args, **_kwargs: {"kernel_type": "triton"})
    monkeypatch.setattr(mini_module, "display_parsed_config", lambda *_args, **_kwargs: "resolved")
    monkeypatch.setattr(mini_module, "parse_gpu_ids", lambda *_args, **_kwargs: [0])
    monkeypatch.setattr(
        mini_module, "extract_user_constraints", lambda *_args, **_kwargs: {"constraints": [], "directives": []}
    )
    monkeypatch.setattr(mini_module, "add_file_handler", lambda *_args, **_kwargs: None)

    class _FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(model_name="fake-model")

    monkeypatch.setattr(mini_module, "get_model", lambda *_args, **_kwargs: _FakeModel())

    class _FakeEnv:
        def __init__(self, **kwargs):
            self.config = SimpleNamespace(**kwargs)

    monkeypatch.setattr(mini_module, "get_environment_class", lambda *_args, **_kwargs: _FakeEnv)

    monkeypatch.setattr(
        mini_module,
        "run_preprocessor",
        lambda **_kwargs: {
            "commandment": "## SETUP\ntrue\n",
            "discovery": {"kernel": {"type": "triton"}},
            "kernel_path": str(kernel_path),
            "repo_root": str(tmp_path),
            "test_command": "python test_harness.py --correctness",
        },
    )

    captured: dict = {}

    def _fake_run_orchestrator(*, preprocess_ctx, **kwargs):
        captured["preprocess_ctx"] = preprocess_ctx
        captured["kwargs"] = kwargs
        return {"best_patch": str(tmp_path / "best.patch"), "best_speedup": 1.1, "summary": "ok"}

    monkeypatch.setattr(mini_module, "run_orchestrator", _fake_run_orchestrator)

    result = mini_module.main(
        visual=False,
        model_name=None,
        model_class=None,
        task="optimize this triton kernel",
        yolo=False,
        cost_limit=None,
        kernel_url=str(kernel_path),
        config_spec=geak_yaml,
        output=tmp_path / "out",
        exit_immediately=False,
        repo=None,
        num_parallel=None,
        gpu_ids=None,
        test_command=None,
    )

    assert result is not None
    assert captured["preprocess_ctx"]["profile_every_patch"] is True
    assert captured["preprocess_ctx"]["patch_profile_quick"] is False
    assert captured["preprocess_ctx"]["user_instructions"] == "optimize this triton kernel"
    assert captured["kwargs"]["heterogeneous"] is True
