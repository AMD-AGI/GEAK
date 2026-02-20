"""Tests for the agent-based task generator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.run.task_generator import _parse_llm_response, generate_tasks
from minisweagent.tools.discovery import DiscoveryResult, KernelInfo


class FakeAgentClass:
    """Stand-in for an agent class in tests."""

    pass


def _make_discovery(
    kernel_type: str = "triton",
    inner_kernel_path: Path | None = None,
) -> DiscoveryResult:
    kernel = KernelInfo(
        file_path=Path("/workspace/kernel.py"),
        kernel_name="test_kernel",
        kernel_type=kernel_type,
        kernel_language="python",
        function_names=["kernel_fwd"],
        has_jit_decorator=True,
        inner_kernel_path=inner_kernel_path,
    )
    return DiscoveryResult(kernels=[kernel], workspace_path=Path("/workspace"))


# ---- Agent submits valid JSON -> tasks produced ----


VALID_TASK_JSON = """[
    {
        "label": "evolve-inner",
        "priority": 0,
        "agent_type": "openevolve",
        "kernel_language": "python",
        "task_prompt": "Run OpenEvolve on /ws/inner.py"
    },
    {
        "label": "mem-opt",
        "priority": 10,
        "agent_type": "strategy_agent",
        "kernel_language": "python",
        "task_prompt": "Optimize memory patterns"
    }
]"""


@patch("minisweagent.run.task_generator._run_task_agent", return_value=VALID_TASK_JSON)
def test_agent_submits_valid_json(mock_agent):
    dr = _make_discovery("triton")
    model = MagicMock()
    tasks = generate_tasks(
        discovery_result=dr,
        base_task_context="ctx",
        agent_class=FakeAgentClass,
        model=model,
    )
    assert len(tasks) == 2
    assert tasks[0].label == "evolve-inner"
    assert tasks[0].priority == 0
    assert tasks[1].label == "mem-opt"
    mock_agent.assert_called_once()


@patch("minisweagent.run.task_generator._run_task_agent", return_value=VALID_TASK_JSON)
def test_openevolve_dispatch(mock_agent):
    """openevolve agent_type -> OpenEvolveWorker class + config."""
    from minisweagent.agents.openevolve_worker import OpenEvolveWorker

    dr = _make_discovery("triton")
    model = MagicMock()
    tasks = generate_tasks(
        discovery_result=dr,
        base_task_context="ctx",
        agent_class=FakeAgentClass,
        model=model,
        commandment_path=Path("/ws/COMMANDMENT.md"),
        baseline_metrics_path=Path("/ws/baseline.json"),
    )
    oe_tasks = [t for t in tasks if t.agent_class is OpenEvolveWorker]
    assert len(oe_tasks) == 1
    assert oe_tasks[0].label == "evolve-inner"
    assert oe_tasks[0].config["kernel_path"] == "/workspace/kernel.py"
    assert oe_tasks[0].config["commandment_path"] == "/ws/COMMANDMENT.md"
    assert oe_tasks[0].config["baseline_metrics_path"] == "/ws/baseline.json"

    strategy_tasks = [t for t in tasks if t.agent_class is FakeAgentClass]
    assert len(strategy_tasks) == 1
    assert strategy_tasks[0].label == "mem-opt"


# ---- Agent fails -> RuntimeError propagates ----


@patch(
    "minisweagent.run.task_generator._run_task_agent",
    side_effect=RuntimeError("agent did not submit"),
)
def test_agent_failure_propagates(mock_agent):
    dr = _make_discovery("triton")
    model = MagicMock()
    with pytest.raises(RuntimeError, match="agent did not submit"):
        generate_tasks(
            discovery_result=dr,
            base_task_context="ctx",
            agent_class=FakeAgentClass,
            model=model,
        )


# ---- No kernels -> empty ----


def test_no_kernels_returns_empty():
    dr = DiscoveryResult()
    tasks = generate_tasks(dr, "ctx", FakeAgentClass, model=MagicMock())
    assert tasks == []


# ---- _parse_llm_response edge cases ----


def test_parse_valid_json():
    tasks = _parse_llm_response(VALID_TASK_JSON, FakeAgentClass)
    assert len(tasks) == 2
    assert tasks[0].label == "evolve-inner"


def test_parse_openevolve_uses_correct_class():
    from minisweagent.agents.openevolve_worker import OpenEvolveWorker

    tasks = _parse_llm_response(
        VALID_TASK_JSON,
        FakeAgentClass,
        kernel_path="/ws/k.py",
        commandment_path="/ws/CMD.md",
        baseline_metrics_path="/ws/bm.json",
    )
    oe = [t for t in tasks if t.agent_class is OpenEvolveWorker]
    assert len(oe) == 1
    assert oe[0].config["kernel_path"] == "/ws/k.py"
    assert oe[0].config["commandment_path"] == "/ws/CMD.md"
    assert oe[0].config["baseline_metrics_path"] == "/ws/bm.json"


def test_parse_strategy_agent_uses_default_class():
    tasks = _parse_llm_response(
        '[{"label": "opt", "priority": 5, "agent_type": "strategy_agent", "task_prompt": "Do it"}]',
        FakeAgentClass,
    )
    assert tasks[0].agent_class is FakeAgentClass


def test_parse_rejects_non_array():
    with pytest.raises(TypeError, match="Expected JSON array"):
        _parse_llm_response('{"not": "array"}', FakeAgentClass)


def test_parse_rejects_empty_task_prompt():
    with pytest.raises(ValueError, match="no valid tasks"):
        _parse_llm_response(
            '[{"label": "x", "priority": 5, "task_prompt": ""}]',
            FakeAgentClass,
        )


def test_parse_clamps_priority():
    tasks = _parse_llm_response(
        '[{"label": "x", "priority": 99, "task_prompt": "Do it"}]',
        FakeAgentClass,
    )
    assert tasks[0].priority == 15


def test_parse_code_fenced_json():
    fenced = '```json\n[{"label": "opt-1", "priority": 5, "agent_type": "strategy_agent", "kernel_language": "python", "task_prompt": "Do something"}]\n```'
    tasks = _parse_llm_response(fenced, FakeAgentClass)
    assert len(tasks) == 1
    assert tasks[0].label == "opt-1"


def test_parse_sorts_by_priority():
    json_text = """[
        {"label": "low", "priority": 15, "task_prompt": "Low priority task"},
        {"label": "high", "priority": 0, "task_prompt": "High priority task"},
        {"label": "mid", "priority": 5, "task_prompt": "Mid priority task"}
    ]"""
    tasks = _parse_llm_response(json_text, FakeAgentClass)
    assert [t.label for t in tasks] == ["high", "mid", "low"]
