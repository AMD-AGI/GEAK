from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

from minisweagent.run.preprocessor import run_preprocessor


def _write_demo_harness(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import argparse",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--correctness', action='store_true')",
                "parser.add_argument('--profile', action='store_true')",
                "parser.add_argument('--benchmark', action='store_true')",
                "parser.add_argument('--full-benchmark', action='store_true')",
                "parser.add_argument('--iterations', type=int, default=30)",
                "args = parser.parse_args()",
                "",
                "if args.profile:",
                "    print('PROFILE OK')",
                "elif args.correctness:",
                "    print('CORRECTNESS OK')",
                "elif args.benchmark or args.full_benchmark:",
                "    print('GEAK_RESULT_LATENCY_MS=1.234')",
                "    print('(1, 1): 1.234 ms')",
                "    print('1 shapes')",
                "else:",
                "    raise SystemExit(2)",
            ]
        )
        + "\n"
    )


def test_run_preprocessor_uses_explicit_deterministic_harness(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    task_dir = repo_root / "tasks" / "demo"
    task_dir.mkdir(parents=True)
    kernel_path = task_dir / "kernel.py"
    harness_path = task_dir / "test_demo_harness.py"
    kernel_path.write_text("def kernel():\n    return 1\n")
    _write_demo_harness(harness_path)

    output_dir = tmp_path / "output"

    fake_pkg = types.ModuleType("automated_test_discovery")
    fake_server = types.ModuleType("automated_test_discovery.server")

    def _discover(*, kernel_path: str, output_dir: str):
        return {"tests": [], "kernel": {"type": "python"}}

    fake_server.discover = _discover
    monkeypatch.setitem(sys.modules, "automated_test_discovery", fake_pkg)
    monkeypatch.setitem(sys.modules, "automated_test_discovery.server", fake_server)

    def _fake_parse(url: str):
        if url.endswith("kernel.py"):
            return {
                "owner": "AMD-AGI",
                "repo": "AIG-Eval",
                "ref": "main",
                "file_path": "tasks/demo/kernel.py",
            }
        if url.endswith("test_demo_harness.py"):
            return {
                "owner": "AMD-AGI",
                "repo": "AIG-Eval",
                "ref": "main",
                "file_path": "tasks/demo/test_demo_harness.py",
            }
        return None

    with (
        patch("minisweagent.run.preprocessor._ensure_mcp_importable", return_value=None),
        patch(
            "minisweagent.tools.resolve_kernel_url_impl.resolve_kernel_url",
            return_value={
                "error": None,
                "local_repo_path": str(repo_root),
                "local_file_path": str(kernel_path),
            },
        ),
        patch("minisweagent.tools.resolve_kernel_url_impl.parse_github_source_url", side_effect=_fake_parse),
        patch(
            "minisweagent.run.codebase_context.generate_codebase_context",
            side_effect=lambda repo_root, kernel_path, output_dir: output_dir / "CODEBASE_CONTEXT.md",
        ),
        patch("minisweagent.run.preprocessor.get_testcase_cache_dir", side_effect=AssertionError("cache should be skipped")),
        patch("minisweagent.run.preprocessor.materialize_cached_harness", side_effect=AssertionError("cache materialization should be skipped")),
        patch("minisweagent.run.preprocessor.create_validated_harness", side_effect=AssertionError("UnitTestAgent should be skipped")),
        patch("minisweagent.run.preprocessor.run_baseline_profile", return_value=None),
        patch("minisweagent.tools.commandment.generate_commandment", return_value="COMMANDMENT"),
    ):
        (output_dir / "CODEBASE_CONTEXT.md").parent.mkdir(parents=True, exist_ok=True)
        (output_dir / "CODEBASE_CONTEXT.md").write_text("context\n")
        ctx = run_preprocessor(
            "https://github.com/AMD-AGI/AIG-Eval/blob/main/tasks/demo/kernel.py",
            output_dir=output_dir,
            gpu_id=0,
            deterministic=True,
            deterministic_harness=(
                "https://github.com/AMD-AGI/AIG-Eval/blob/main/tasks/demo/test_demo_harness.py"
            ),
        )

    assert ctx["harness_path"] == str(harness_path.resolve())
    assert ctx["test_command"].endswith(f"{harness_path.resolve()} --correctness")
    assert ctx["testcase_selection"]["selected_source"] == "deterministic_harness"
    assert ctx["testcase_selection"]["deterministic_requested"] is True
    assert ctx["testcase_selection"]["reused_cache"] is False
    assert ctx["commandment"] == "COMMANDMENT"

    testcase_selection = json.loads((output_dir / "testcase_selection.json").read_text())
    assert testcase_selection["selected_source"] == "deterministic_harness"
    assert testcase_selection["harness_path"] == str(harness_path.resolve())


def test_run_preprocessor_accepts_local_deterministic_harness_path(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    task_dir = repo_root / "tasks" / "demo"
    task_dir.mkdir(parents=True)
    kernel_path = task_dir / "kernel.py"
    harness_path = task_dir / "test_demo_harness.py"
    kernel_path.write_text("def kernel():\n    return 1\n")
    _write_demo_harness(harness_path)

    output_dir = tmp_path / "output"

    fake_pkg = types.ModuleType("automated_test_discovery")
    fake_server = types.ModuleType("automated_test_discovery.server")

    def _discover(*, kernel_path: str, output_dir: str):
        return {"tests": [], "kernel": {"type": "python"}}

    fake_server.discover = _discover
    monkeypatch.setitem(sys.modules, "automated_test_discovery", fake_pkg)
    monkeypatch.setitem(sys.modules, "automated_test_discovery.server", fake_server)

    with (
        patch("minisweagent.run.preprocessor._ensure_mcp_importable", return_value=None),
        patch(
            "minisweagent.tools.resolve_kernel_url_impl.resolve_kernel_url",
            return_value={
                "error": None,
                "local_repo_path": str(repo_root),
                "local_file_path": str(kernel_path),
            },
        ),
        patch(
            "minisweagent.run.codebase_context.generate_codebase_context",
            side_effect=lambda repo_root, kernel_path, output_dir: output_dir / "CODEBASE_CONTEXT.md",
        ),
        patch("minisweagent.run.preprocessor.get_testcase_cache_dir", side_effect=AssertionError("cache should be skipped")),
        patch("minisweagent.run.preprocessor.materialize_cached_harness", side_effect=AssertionError("cache materialization should be skipped")),
        patch("minisweagent.run.preprocessor.create_validated_harness", side_effect=AssertionError("UnitTestAgent should be skipped")),
        patch("minisweagent.run.preprocessor.run_baseline_profile", return_value=None),
        patch("minisweagent.tools.commandment.generate_commandment", return_value="COMMANDMENT"),
    ):
        (output_dir / "CODEBASE_CONTEXT.md").parent.mkdir(parents=True, exist_ok=True)
        (output_dir / "CODEBASE_CONTEXT.md").write_text("context\n")
        ctx = run_preprocessor(
            "https://github.com/AMD-AGI/AIG-Eval/blob/main/tasks/demo/kernel.py",
            output_dir=output_dir,
            gpu_id=0,
            deterministic=True,
            deterministic_harness="tasks/demo/test_demo_harness.py",
        )

    assert ctx["harness_path"] == str(harness_path.resolve())
    assert ctx["test_command"].endswith(f"{harness_path.resolve()} --correctness")
    assert ctx["testcase_selection"]["selected_source"] == "deterministic_harness"
    assert ctx["testcase_selection"]["deterministic_requested"] is True
    assert ctx["testcase_selection"]["deterministic_resolution"]["source"] == "local_path"
    assert ctx["testcase_selection"]["deterministic_resolution"]["path"] == str(harness_path.resolve())
