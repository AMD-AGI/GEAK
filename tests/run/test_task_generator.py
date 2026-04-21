"""Tests for the agent-based task generator."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from minisweagent.agents.heterogeneous.task_generator import (
    _SYSTEM_PROMPT,
    _build_workload_guidance,
    _parse_llm_response,
    generate_tasks,
)


class FakeAgentClass:
    """Stand-in for an agent class in tests."""

    pass


def _make_kernel_kwargs(
    kernel_type: str = "triton",
) -> dict[str, Any]:
    return {
        "kernel_path": "/workspace/kernel.py",
        "kernel_name": "test_kernel",
        "kernel_type": kernel_type,
        "kernel_language": "python",
        "function_names": ["kernel_fwd"],
        "workspace_path": "/workspace",
    }


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


@patch("minisweagent.agents.heterogeneous.task_generator._run_task_agent", return_value=VALID_TASK_JSON)
def test_agent_submits_valid_json(mock_agent):
    model = MagicMock()
    tasks = generate_tasks(
        base_task_context="ctx",
        agent_class=FakeAgentClass,
        model=model,
        **_make_kernel_kwargs("triton"),
    )
    assert len(tasks) == 2
    assert tasks[0].label == "evolve-inner"
    assert tasks[0].priority == 0
    assert tasks[1].label == "mem-opt"
    mock_agent.assert_called_once()


# ---- Agent fails -> RuntimeError propagates ----


@patch(
    "minisweagent.agents.heterogeneous.task_generator._run_task_agent",
    side_effect=RuntimeError("agent did not submit"),
)
def test_agent_failure_propagates(mock_agent):
    model = MagicMock()
    with pytest.raises(RuntimeError, match="agent did not submit"):
        generate_tasks(
            base_task_context="ctx",
            agent_class=FakeAgentClass,
            model=model,
            **_make_kernel_kwargs("triton"),
        )


# ---- No kernels -> empty ----


def test_no_kernel_path_returns_empty():
    tasks = generate_tasks("ctx", FakeAgentClass, model=MagicMock(), kernel_path="")
    assert tasks == []


# ---- _parse_llm_response edge cases ----


def test_parse_valid_json():
    tasks = _parse_llm_response(VALID_TASK_JSON, FakeAgentClass)
    assert len(tasks) == 2
    assert tasks[0].label == "evolve-inner"


def test_parse_strategy_agent_uses_mapped_class():
    from minisweagent.agents.strategy_interactive import StrategyInteractiveAgent

    tasks = _parse_llm_response(
        '[{"label": "opt", "priority": 5, "agent_type": "strategy_agent", "task_prompt": "Do it"}]',
        FakeAgentClass,
    )
    assert tasks[0].agent_class is StrategyInteractiveAgent


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


def test_build_workload_guidance_classifies_hip_search_as_latency_bound():
    kernel = {
        "file_path": "/workspace/rocprim/device_binary_search.hpp",
        "kernel_name": "device_binary_search",
        "kernel_type": "unknown",
    }
    baseline_metrics = {
        "kernel_name": "rocprim::detail::binary_search lower_bound",
        "bottleneck": "latency",
        "top_kernels": [
            {
                "name": "transform_kernel<binary_search<lower_bound>>",
                "bottleneck": "latency",
                "duration_us": 10.0,
                "pct_of_selected": 100.0,
                "observations": [],
                "metrics": {
                    "memory.hbm_bandwidth_utilization": 0.3,
                    "memory.l2_hit_rate": 70.6,
                },
            }
        ],
    }

    guidance = _build_workload_guidance(kernel, baseline_metrics)

    assert "HIP backend detected." in guidance
    assert "Prefer First:" in guidance
    assert "Branchless/control-flow simplification" in guidance
    assert "Size-specialized kernel variants" in guidance
    assert "Bandwidth-maximization or generic vectorization ideas as the main strategy." in guidance
    assert "Search / pointer-chasing classifier:" in guidance


def test_build_workload_guidance_for_triton_deprioritizes_dispatch():
    kernel = {
        "file_path": "/workspace/kernel.py",
        "kernel_name": "fused_rms",
        "kernel_type": "triton",
    }
    baseline_metrics = {
        "kernel_name": "fused_rms_fp8",
        "bottleneck": "memory-bound",
        "top_kernels": [{
            "name": "fused_rms_fp8",
            "bottleneck": "memory-bound",
            "duration_us": 12.4,
            "pct_of_selected": 100.0,
            "observations": [],
            "metrics": {
                "memory.hbm_bandwidth_utilization": 71.2,
                "memory.l2_hit_rate": 44.0,
            },
        }],
    }

    guidance = _build_workload_guidance(kernel, baseline_metrics)

    assert "Triton backend detected." in guidance
    assert "Prefer First:" in guidance
    assert "Memory-access rewrites inside the kernel body" in guidance
    assert "@triton.autotune-only config sweeps." in guidance
    assert "Python dispatch, import-routing, or wrapper-only edits" in guidance


def test_build_workload_guidance_for_mixed_triton_mentions_dual_focus() -> None:
    kernel = {
        "file_path": "/workspace/kernel.py",
        "kernel_name": "fused_mix",
        "kernel_type": "triton",
    }
    baseline_metrics = {
        "kernel_name": "fused_mix",
        "top_kernels": [
            {
                "name": "mem_hot",
                "bottleneck": "memory",
                "duration_us": 60.0,
                "pct_of_selected": 60.0,
                "observations": [],
                "metrics": {"memory.hbm_bandwidth_utilization": 65.0},
            },
            {
                "name": "compute_hot",
                "bottleneck": "compute",
                "duration_us": 40.0,
                "pct_of_selected": 40.0,
                "observations": [],
                "metrics": {"memory.hbm_bandwidth_utilization": 5.0},
            },
        ],
        "selected_kernel_summary": {
            "mode": "mixed",
            "primary_bottleneck": "memory",
            "primary_bottleneck_pct_of_selected": 60.0,
            "bottleneck_mix": [
                {"bottleneck": "memory", "pct_of_selected": 60.0, "duration_us": 60.0, "kernels": []},
                {"bottleneck": "compute", "pct_of_selected": 40.0, "duration_us": 40.0, "kernels": []},
            ],
        },
    }

    guidance = _build_workload_guidance(kernel, baseline_metrics)

    assert "Mixed bottlenecks: memory 60.0%, compute 40.0% (primary: memory)" in guidance
    assert "Memory-access rewrites inside the kernel body" in guidance
    assert "Instruction-count reduction and control-flow simplification" in guidance


def test_build_workload_guidance_empty_when_no_backend_and_no_metrics():
    kernel = {
        "file_path": "/workspace/kernel.txt",
        "kernel_name": "mystery",
        "kernel_type": "unknown",
    }

    assert _build_workload_guidance(kernel, {}) == ""


def test_system_prompt_deprioritizes_dispatch_path_work():
    assert "- 0: Novel algorithmic kernel rewrites" in _SYSTEM_PROMPT
    assert "- 15: Wrapper/launch-config/dispatch-only changes (lowest priority)" in _SYSTEM_PROMPT
    assert 'Generate at least 3 tasks from the "Prefer First" families' in _SYSTEM_PROMPT
    assert "leave some gpus idle" in _SYSTEM_PROMPT.lower()
    assert "Generate at least one priority-0 task that specifically checks the dispatch path" not in _SYSTEM_PROMPT
