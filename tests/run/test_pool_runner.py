"""Tests for ``run/pool_runner.py`` — the single GPU-pool execution path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from minisweagent.agents.agent_spec import AgentTask
from minisweagent.run.pool_runner import (
    build_homogeneous_tasks,
    execute,
)
from minisweagent.run.unified import PipelineContext


class _FakeAgent:
    """Stand-in for OptimizationAgent — never instantiated in these tests."""


def _make_ctx(**overrides) -> PipelineContext:
    base = {
        "preprocess_ctx": {"kernel_path": "/tmp/k.py"},
        "user_prompt": "Optimize kernel X",
    }
    base.update(overrides)
    return PipelineContext(**base)


# ── build_homogeneous_tasks ───────────────────────────────────────────


def test_build_homogeneous_tasks_shape():
    tasks = build_homogeneous_tasks(
        num_parallel=4,
        agent_class=_FakeAgent,
        task_body="body",
    )
    assert len(tasks) == 4
    assert all(isinstance(t, AgentTask) for t in tasks)
    assert all(t.agent_class is _FakeAgent for t in tasks)
    assert all(t.task == "body" for t in tasks)
    assert [t.label for t in tasks] == ["parallel_0", "parallel_1", "parallel_2", "parallel_3"]
    assert all(t.num_gpus == 1 for t in tasks)


def test_build_homogeneous_tasks_custom_args():
    tasks = build_homogeneous_tasks(
        num_parallel=2,
        agent_class=_FakeAgent,
        task_body="body",
        base_label="ablation",
        priority=0,
        kernel_language="triton",
        num_gpus_per_task=2,
    )
    assert [t.label for t in tasks] == ["ablation_0", "ablation_1"]
    assert all(t.priority == 0 for t in tasks)
    assert all(t.kernel_language == "triton" for t in tasks)
    assert all(t.num_gpus == 2 for t in tasks)


def test_build_homogeneous_tasks_rejects_zero():
    with pytest.raises(ValueError, match="num_parallel must be >= 1"):
        build_homogeneous_tasks(num_parallel=0, agent_class=_FakeAgent, task_body="x")


# ── execute ───────────────────────────────────────────────────────────


def test_execute_rejects_empty_tasks(caplog):
    ctx = _make_ctx(repo=Path("/tmp/repo"), output_dir=Path("/tmp/out"))
    result = execute(ctx, tasks=[], env_factory=lambda: SimpleNamespace(config=SimpleNamespace(__dict__={})))
    assert result == []


def test_execute_requires_repo():
    ctx = _make_ctx()  # no repo, no output_dir
    tasks = build_homogeneous_tasks(1, _FakeAgent, "x")
    with pytest.raises(ValueError, match="repo path"):
        execute(ctx, tasks=tasks, env_factory=lambda: SimpleNamespace(config=SimpleNamespace(__dict__={})))


def test_execute_requires_env_factory():
    ctx = _make_ctx(repo=Path("/tmp/r"), output_dir=Path("/tmp/o"))
    tasks = build_homogeneous_tasks(1, _FakeAgent, "x")
    with pytest.raises(ValueError, match="env_factory"):
        execute(ctx, tasks=tasks)


def test_execute_requires_model_factory():
    ctx = _make_ctx(repo=Path("/tmp/r"), output_dir=Path("/tmp/o"))
    tasks = build_homogeneous_tasks(1, _FakeAgent, "x")
    with pytest.raises(ValueError, match="model factory"):
        execute(
            ctx,
            tasks=tasks,
            env_factory=lambda: SimpleNamespace(config=SimpleNamespace(__dict__={})),
        )


def test_execute_delegates_to_run_pool_with_ctx_values():
    ctx = _make_ctx(
        repo=Path("/tmp/r"),
        output_dir=Path("/tmp/o"),
        gpu_ids=[0, 1, 2],
        model_factory=lambda: "model_instance",
    )
    tasks = build_homogeneous_tasks(2, _FakeAgent, "user task body")

    captured: dict = {}

    def _fake_run_pool(**kwargs):
        captured.update(kwargs)
        return [(0, None, "ok", None)]

    env_factory = lambda: SimpleNamespace(config=SimpleNamespace(__dict__={}))

    with patch("minisweagent.run.pool_runner._run_pool_impl", _fake_run_pool):
        result = execute(
            ctx,
            tasks=tasks,
            env_factory=env_factory,
            is_git_repo=True,
        )

    assert result == [(0, None, "ok", None)]
    assert captured["gpu_ids"] == [0, 1, 2]
    assert captured["repo_path"] == Path("/tmp/r")
    assert captured["is_git_repo"] is True
    assert captured["base_task_content"] == "user task body"
    assert captured["tasks"] is tasks
    # model_factory passed through
    assert captured["model_factory"]() == "model_instance"


def test_execute_falls_back_to_user_prompt_when_task_empty():
    ctx = _make_ctx(
        repo=Path("/tmp/r"),
        output_dir=Path("/tmp/o"),
        gpu_ids=[0],
        model_factory=lambda: "m",
        user_prompt="FALLBACK PROMPT",
    )
    # Task with empty ``task`` — base_task_content should come from ctx.user_prompt
    tasks = [AgentTask(agent_class=_FakeAgent, task="", label="x")]

    captured: dict = {}

    def _fake_run_pool(**kwargs):
        captured.update(kwargs)
        return []

    env_factory = lambda: SimpleNamespace(config=SimpleNamespace(__dict__={}))
    with patch("minisweagent.run.pool_runner._run_pool_impl", _fake_run_pool):
        execute(ctx, tasks=tasks, env_factory=env_factory, is_git_repo=False)

    assert captured["base_task_content"] == "FALLBACK PROMPT"
