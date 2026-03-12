"""Tests for canonical testcase/harness reuse."""

from __future__ import annotations

from pathlib import Path

from minisweagent.run.testcase_cache import (
    build_testcase_cache_key,
    get_testcase_cache_entry,
    materialize_cached_harness,
    save_cached_harness,
)


def test_repo_relative_harness_rewrites_to_current_repo(tmp_path: Path) -> None:
    old_repo = tmp_path / "old_repo"
    new_repo = tmp_path / "new_repo"
    for repo in (old_repo, new_repo):
        (repo / "tasks" / "demo").mkdir(parents=True)
        (repo / "tasks" / "demo" / "kernel.py").write_text("print('kernel')\n")
        (repo / "tasks" / "demo" / "test_harness.py").write_text("from kernel import *\n")

    old_output = tmp_path / "old_output"
    old_output.mkdir()
    cache_dir = tmp_path / "cache"
    cache_key = build_testcase_cache_key("https://example.com/kernel.py", old_repo / "tasks" / "demo" / "kernel.py")
    entry = get_testcase_cache_entry(cache_dir, cache_key)

    source_harness = old_repo / "tasks" / "demo" / "test_harness.py"
    source_command = f"python {source_harness} --correctness"
    save_cached_harness(
        entry,
        kernel_url="https://example.com/kernel.py",
        source="discovery_test",
        test_command=source_command,
        harness_path=source_harness,
        repo_root=old_repo,
        output_dir=old_output,
        kernel_path=old_repo / "tasks" / "demo" / "kernel.py",
        harness_results=[{"mode": "correctness", "success": True, "returncode": 0}],
    )

    new_output = tmp_path / "new_output"
    new_output.mkdir()
    materialized = materialize_cached_harness(
        entry,
        repo_root=new_repo,
        output_dir=new_output,
        kernel_path=new_repo / "tasks" / "demo" / "kernel.py",
    )

    assert materialized is not None
    test_command, harness_path, manifest = materialized
    assert harness_path == str((new_repo / "tasks" / "demo" / "test_harness.py").resolve())
    assert test_command == f"python {harness_path} --correctness"
    assert manifest["source_type"] == "repo_relative"


def test_standalone_harness_snapshot_rewrites_repo_and_output_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source_output = tmp_path / "old_output"
    source_output.mkdir()
    kernel_path = repo_root / "kernel.py"
    kernel_path.write_text("print('kernel')\n")
    harness_path = source_output / "generated_harness.py"
    harness_path.write_text(
        "\n".join(
            [
                f"REPO = {repo_root!r}",
                f"KERNEL = {str(kernel_path)!r}",
                f"OUT = {str(source_output)!r}",
            ]
        )
    )

    cache_dir = tmp_path / "cache"
    cache_key = build_testcase_cache_key("local://kernel", kernel_path)
    entry = get_testcase_cache_entry(cache_dir, cache_key)
    source_command = f"python {harness_path} --benchmark"
    save_cached_harness(
        entry,
        kernel_url="local://kernel",
        source="unit_test_agent",
        test_command=source_command,
        harness_path=harness_path,
        repo_root=repo_root,
        output_dir=source_output,
        kernel_path=kernel_path,
        harness_results=[{"mode": "benchmark", "success": True, "returncode": 0}],
    )

    new_repo = tmp_path / "new_repo"
    new_repo.mkdir()
    new_kernel = new_repo / "kernel.py"
    new_kernel.write_text("print('new kernel')\n")
    new_output = tmp_path / "new_output"
    new_output.mkdir()

    materialized = materialize_cached_harness(
        entry,
        repo_root=new_repo,
        output_dir=new_output,
        kernel_path=new_kernel,
    )

    assert materialized is not None
    test_command, new_harness_path, manifest = materialized
    new_harness = Path(new_harness_path)
    assert new_harness.is_file()
    text = new_harness.read_text()
    assert str(new_repo.resolve()) in text
    assert str(new_kernel.resolve()) in text
    assert str(new_output.resolve()) in text
    assert str(repo_root.resolve()) not in text
    assert str(source_output.resolve()) not in text
    assert test_command == f"python {new_harness} --benchmark"
    assert manifest["source_type"] == "standalone"
