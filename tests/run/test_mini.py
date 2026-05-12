"""Unit tests for helpers in ``minisweagent.run.mini`` (no full CLI / preprocess run)."""

from __future__ import annotations

from pathlib import Path
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


class TestDeriveOutputDir:
    def test_none_output_uses_generated_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

        def fake_generate(name: str | None) -> str:
            return "optimization_logs/kernel_fixed_ts"

        with patch(
            "minisweagent.run.utils.task_parser.generate_patch_output_dir",
            side_effect=fake_generate,
        ):
            out_dir, auto = mini_module._derive_output_dir(None, "my_kernel")

        assert out_dir == (tmp_path / "optimization_logs" / "kernel_fixed_ts").resolve()
        # output=None -> geak generated the path. ``--keep-runs`` retention
        # only ever acts on auto dirs.
        assert auto is True

    def test_file_path_uses_parent_for_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "run.traj.json"
        out_dir, auto = mini_module._derive_output_dir(f, None)
        assert out_dir == f.parent.resolve()
        assert auto is False

    def test_directory_path_returns_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "logs"
        d.mkdir()
        out_dir, auto = mini_module._derive_output_dir(d, None)
        assert out_dir == d.resolve()
        assert auto is False

    # The next three tests pass RELATIVE paths -- which was the gap that let
    # commit c07285cc silently regress the load-bearing ``output.resolve()``
    # call originally added in 3b0ff0ac. The pre-existing tests in this class
    # passed only because tmp_path is always absolute, so .resolve() was a
    # no-op and the assertions held either way.

    def test_relative_directory_resolves_to_absolute(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        rel = Path("outputs/silu")
        out_dir, auto = mini_module._derive_output_dir(rel, None)
        assert out_dir.is_absolute(), f"_derive_output_dir must always return absolute; got {out_dir!r}"
        assert out_dir == (tmp_path / "outputs" / "silu").resolve()
        assert auto is False

    def test_relative_file_path_resolves_parent_to_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        rel = Path("outputs/silu/run.traj.json")
        out_dir, auto = mini_module._derive_output_dir(rel, None)
        assert out_dir.is_absolute()
        assert out_dir == (tmp_path / "outputs" / "silu").resolve()
        assert auto is False

    def test_bare_relative_name_resolves_to_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A user passing ``--output silu`` (no subdirs) must still get an absolute
        # path anchored at cwd, not be left as a bare relative name.
        monkeypatch.chdir(tmp_path)
        out_dir, auto = mini_module._derive_output_dir(Path("silu"), None)
        assert out_dir.is_absolute()
        assert out_dir == (tmp_path / "silu").resolve()
        assert auto is False


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
                    "    p.add_argument('--iterations', type=int, default=None)",
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
