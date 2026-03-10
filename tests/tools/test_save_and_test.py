import os
import subprocess
from pathlib import Path

from minisweagent.tools.save_and_test import SaveAndTestContext, SaveAndTestTool


def _init_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "kernel.py").write_text("print('base')\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_get_patch_content_excludes_generated_helpers(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)

    (repo / "kernel.py").write_text("print('patched')\n")
    (repo / "test_harness_demo.py").write_text("print('helper')\n")
    (repo / "run.sh").write_text("#!/bin/bash\n")

    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(repo),
            test_command="python -c 'print(0)'",
            timeout=10,
            patch_output_dir=None,
            env_vars={"GEAK_HARNESS": str(repo / "test_harness_demo.py")},
            base_repo_path=repo,
        )
    )

    patch = tool._get_patch_content()

    assert "kernel.py" in patch
    assert "test_harness_demo.py" not in patch
    assert "run.sh" not in patch


def test_get_patch_content_excludes_generated_helpers_without_git(tmp_path: Path) -> None:
    base_repo = tmp_path / "base"
    worktree = tmp_path / "worktree"
    base_repo.mkdir()
    worktree.mkdir()

    (base_repo / "kernel.py").write_text("print('base')\n")
    (worktree / "kernel.py").write_text("print('patched')\n")
    (worktree / "test_harness.py").write_text("print('generated')\n")
    (worktree / "test_harness_binary_search.cpp").write_text("// generated\n")
    (worktree / "rocprim_version.hpp").write_text("// generated version\n")

    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(worktree),
            test_command="python -c 'print(0)'",
            timeout=10,
            patch_output_dir=None,
            env_vars=None,
            base_repo_path=base_repo,
        )
    )

    patch = tool._get_patch_content()

    assert "kernel.py" in patch
    assert f"+++ {worktree / 'test_harness.py'}" not in patch
    assert f"+++ {worktree / 'test_harness_binary_search.cpp'}" not in patch
    assert f"+++ {worktree / 'rocprim_version.hpp'}" not in patch


def test_run_test_restores_missing_harness_helper_from_base_repo(tmp_path: Path) -> None:
    base_repo = tmp_path / "base"
    worktree = tmp_path / "worktree"
    base_repo.mkdir()
    worktree.mkdir()

    helper_name = "test_harness_demo.py"
    (base_repo / helper_name).write_text("print('restored from base')\n")
    helper_path = worktree / helper_name

    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(worktree),
            test_command='python "${GEAK_HARNESS}"',
            timeout=10,
            patch_output_dir=None,
            env_vars={"GEAK_HARNESS": str(helper_path)},
            base_repo_path=base_repo,
        )
    )

    output, passed, returncode = tool._run_test()

    assert passed is True
    assert returncode == 0
    assert helper_path.exists()
    assert "restored from base" in output


def test_run_test_recreates_missing_harness_helper_from_task_fallback(tmp_path: Path) -> None:
    worktree = tmp_path / "worktree"
    task_dir = worktree / "tasks" / "geak_eval" / "demo"
    task_dir.mkdir(parents=True)
    worktree.mkdir(exist_ok=True)

    actual_harness = task_dir / "test_harness.py"
    actual_harness.write_text("print('fallback harness')\n")
    helper_path = worktree / "test_harness_demo.py"

    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(worktree),
            test_command='python "${GEAK_HARNESS}"',
            timeout=10,
            patch_output_dir=None,
            env_vars={"GEAK_HARNESS": str(helper_path)},
            base_repo_path=None,
        )
    )

    output, passed, returncode = tool._run_test()

    assert passed is True
    assert returncode == 0
    assert helper_path.is_symlink()
    assert os.readlink(helper_path) == "tasks/geak_eval/demo/test_harness.py"
    assert "fallback harness" in output


def test_format_output_includes_true_baseline_speedups(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "fused_rms_fp8"
    task_dir = kernel_dir / "results" / "round_1" / "dispatch-path-check"
    task_dir.mkdir(parents=True)
    (kernel_dir / "benchmark_baseline.txt").write_text(
        "\n".join(
            [
                "Benchmark mode: 2 shapes, 10 iterations each",
                "  (32,4096): 0.0500 ms",
                "  (64,4096): 0.0600 ms",
                "Geomean latency: 0.0548 ms",
                "GEAK_RESULT_LATENCY_MS=0.054772",
            ]
        )
    )

    tool = SaveAndTestTool()
    tool.set_context(
        SaveAndTestContext(
            cwd=str(tmp_path),
            test_command="python -c 'print(0)'",
            timeout=10,
            patch_output_dir=str(task_dir),
            env_vars=None,
            base_repo_path=None,
        )
    )

    formatted = tool._format_output(
        patch_name="patch_1",
        patch_content="diff --git a/kernel.py b/kernel.py\n",
        test_output="\n".join(
            [
                "Benchmark mode: 2 shapes, 10 iterations each",
                "  (32,4096): 0.0400 ms",
                "  (64,4096): 0.0750 ms",
                "Geomean latency: 0.0548 ms",
                "GEAK_RESULT_LATENCY_MS=0.054772",
            ]
        ),
        test_passed=True,
        returncode=0,
    )

    assert "Speedup vs true baseline:" in formatted
    assert "Overall: 1.0000x (0.054772 ms -> 0.054772 ms)" in formatted
    assert "(32,4096): 1.2500x (0.050000 ms -> 0.040000 ms)" in formatted
    assert "(64,4096): 0.8000x (0.060000 ms -> 0.075000 ms)" in formatted
