"""Tests for staged dispatch selection in the orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

from minisweagent.run.orchestrator import _group_task_files_by_dispatch_stage, _stage_found_improvement
from minisweagent.run.task_file import write_task_file


def _make_task(tmp_path: Path, name: str, *, priority: int, label: str | None = None) -> Path:
    task_path = tmp_path / f"{name}.md"
    write_task_file(
        task_path,
        {
            "label": label or name,
            "priority": priority,
        },
        "Optimize the kernel.",
    )
    return task_path


def test_group_task_files_by_dispatch_stage_orders_priority_bands(tmp_path: Path) -> None:
    wrapper = _make_task(tmp_path, "wrapper", priority=15)
    tuning = _make_task(tmp_path, "tuning", priority=8)
    kernel_a = _make_task(tmp_path, "kernel_a", priority=5)
    kernel_b = _make_task(tmp_path, "kernel_b", priority=0)

    groups = _group_task_files_by_dispatch_stage([wrapper, tuning, kernel_a, kernel_b])

    assert [stage for stage, _files in groups] == [
        "kernel_body",
        "tuning_fallback",
        "wrapper_fallback",
    ]
    assert [path.stem for path in groups[0][1]] == ["kernel_b", "kernel_a"]
    assert [path.stem for path in groups[1][1]] == ["tuning"]
    assert [path.stem for path in groups[2][1]] == ["wrapper"]


def test_stage_found_improvement_checks_only_selected_tasks(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    improving_task = _make_task(tmp_path, "algo", priority=0, label="algo")
    neutral_task = _make_task(tmp_path, "dispatch", priority=15, label="dispatch")

    algo_dir = results_dir / "algo"
    algo_dir.mkdir()
    (algo_dir / "best_results.json").write_text(json.dumps({"best_patch_speedup": 1.07}))

    dispatch_dir = results_dir / "dispatch"
    dispatch_dir.mkdir()
    (dispatch_dir / "best_results.json").write_text(json.dumps({"best_patch_speedup": 0.99}))

    assert _stage_found_improvement(results_dir, [improving_task]) is True
    assert _stage_found_improvement(results_dir, [neutral_task]) is False
