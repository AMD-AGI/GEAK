"""Tests for the LLM-assisted task generator with rule-based fallback."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from minisweagent.agents.agent_spec import AgentTask
from minisweagent.run.task_generator import generate_tasks, _parse_llm_response
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
    return DiscoveryResult(kernels=[kernel])


# ---- Fallback: model=None -> rule-based ----

def test_no_model_falls_back_to_rule_based():
    dr = _make_discovery("triton", inner_kernel_path=Path("/ws/inner.py"))
    tasks = generate_tasks(
        discovery_result=dr,
        base_task_context="ctx",
        agent_class=FakeAgentClass,
        model=None,
    )
    labels = [t.label for t in tasks]
    assert "openevolve-inner" in labels
    assert "triton-autotune" in labels
    assert "profile-guided" in labels


# ---- Fallback: LLM raises exception -> rule-based ----

def test_llm_exception_falls_back():
    mock_model = MagicMock()
    mock_model.query.side_effect = RuntimeError("API error")
    dr = _make_discovery("triton")
    tasks = generate_tasks(
        discovery_result=dr,
        base_task_context="ctx",
        agent_class=FakeAgentClass,
        model=mock_model,
    )
    labels = [t.label for t in tasks]
    assert "profile-guided" in labels


# ---- Fallback: LLM returns garbage -> rule-based ----

def test_llm_garbage_falls_back():
    mock_model = MagicMock()
    mock_model.query.return_value = {"content": "This is not JSON at all"}
    dr = _make_discovery("hip")
    tasks = generate_tasks(
        discovery_result=dr,
        base_task_context="ctx",
        agent_class=FakeAgentClass,
        model=mock_model,
    )
    labels = [t.label for t in tasks]
    assert "hip-launch-config" in labels


# ---- Fallback: LLM returns empty array -> rule-based ----

def test_llm_empty_array_falls_back():
    mock_model = MagicMock()
    mock_model.query.return_value = {"content": "[]"}
    dr = _make_discovery("triton")
    tasks = generate_tasks(
        discovery_result=dr,
        base_task_context="ctx",
        agent_class=FakeAgentClass,
        model=mock_model,
    )
    labels = [t.label for t in tasks]
    assert "profile-guided" in labels


# ---- Success: LLM returns valid JSON ----

def test_llm_valid_response_produces_tasks():
    mock_model = MagicMock()
    mock_model.query.return_value = {"content": """[
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
    ]"""}
    dr = _make_discovery("triton")
    tasks = generate_tasks(
        discovery_result=dr,
        base_task_context="ctx",
        agent_class=FakeAgentClass,
        model=mock_model,
    )
    assert len(tasks) == 2
    assert tasks[0].label == "evolve-inner"
    assert tasks[0].priority == 0
    assert tasks[1].label == "mem-opt"


# ---- LLM wraps JSON in code fences -> still parsed ----

def test_llm_code_fenced_json_parsed():
    mock_model = MagicMock()
    mock_model.query.return_value = {"content": """```json
[{"label": "opt-1", "priority": 5, "agent_type": "strategy_agent", "kernel_language": "python", "task_prompt": "Do something"}]
```"""}
    dr = _make_discovery("triton")
    tasks = generate_tasks(
        discovery_result=dr,
        base_task_context="ctx",
        agent_class=FakeAgentClass,
        model=mock_model,
    )
    assert len(tasks) == 1
    assert tasks[0].label == "opt-1"


# ---- _parse_llm_response edge cases ----

def test_parse_rejects_non_array():
    with pytest.raises(TypeError, match="Expected JSON array"):
        _parse_llm_response('{"not": "array"}', FakeAgentClass, "ctx")


def test_parse_rejects_empty_task_prompt():
    # Tasks with empty task_prompt are skipped; if all are empty -> ValueError
    with pytest.raises(ValueError, match="no valid tasks"):
        _parse_llm_response(
            '[{"label": "x", "priority": 5, "task_prompt": ""}]',
            FakeAgentClass,
            "ctx",
        )


def test_parse_clamps_priority():
    tasks = _parse_llm_response(
        '[{"label": "x", "priority": 99, "task_prompt": "Do it"}]',
        FakeAgentClass,
        "ctx",
    )
    assert tasks[0].priority == 15  # clamped to max


# ---- No kernels -> empty ----

def test_no_kernels_returns_empty():
    dr = DiscoveryResult()
    tasks = generate_tasks(dr, "ctx", FakeAgentClass, model=MagicMock())
    assert tasks == []
