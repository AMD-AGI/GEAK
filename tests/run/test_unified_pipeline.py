"""Tests for ``run/unified.py`` — the single entry point for run_pipeline(ctx, mode)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from minisweagent.run.unified import PipelineContext, _resolve_tools, run_pipeline


def _make_ctx(**overrides) -> PipelineContext:
    base = {
        "preprocess_ctx": {"kernel_path": "/tmp/k.py"},
        "user_prompt": "Optimize kernel X",
    }
    base.update(overrides)
    return PipelineContext(**base)


def test_pipeline_context_defaults():
    ctx = _make_ctx()
    assert ctx.preprocess_ctx == {"kernel_path": "/tmp/k.py"}
    assert ctx.user_prompt == "Optimize kernel X"
    assert ctx.gpu_ids == [0]
    assert ctx.rag_enabled is False
    assert ctx.extra_addenda == []


def test_resolve_tools_disables_rag_when_flag_off():
    ctx = _make_ctx(rag_enabled=False)

    class _FakeRuntime:
        def __init__(self, **kwargs):
            self.disabled: list[list[str]] = []

        def disable_tools(self, names):
            self.disabled.append(list(names))

        def wrap_rag_tools_with_postprocessor(self):  # pragma: no cover - not hit
            raise AssertionError("should not be called when rag is off")

    with patch("minisweagent.tools.tools_runtime.ToolRuntime", _FakeRuntime):
        runtime = _resolve_tools(ctx, mode="fixed")
    assert runtime.disabled == [["query", "optimize"]]


def test_resolve_tools_wraps_rag_when_enabled():
    ctx = _make_ctx(rag_enabled=True)

    class _FakeRuntime:
        def __init__(self, **kwargs):
            self.wrapped = False
            self.disabled: list[list[str]] = []

        def disable_tools(self, names):  # pragma: no cover - not hit
            self.disabled.append(list(names))

        def wrap_rag_tools_with_postprocessor(self):
            self.wrapped = True

    with patch("minisweagent.tools.tools_runtime.ToolRuntime", _FakeRuntime):
        runtime = _resolve_tools(ctx, mode="planned")
    assert runtime.wrapped is True
    assert runtime.disabled == []


def test_pipeline_unknown_mode_raises():
    ctx = _make_ctx()
    with pytest.raises(ValueError, match="Unknown pipeline mode"):
        run_pipeline(ctx, mode="nonsense")  # type: ignore[arg-type]


def test_pipeline_rejects_translate_mode():
    """Translation is a preprocess phase, not a run_pipeline mode.

    ``run_pipeline(mode="translate")`` MUST raise ValueError with a
    pointer to the correct entry point.  If someone ever resurrects
    translate as a mode, this test catches the regression.
    """
    ctx = _make_ctx()
    with pytest.raises(ValueError, match="not a run_pipeline mode"):
        run_pipeline(ctx, mode="translate")  # type: ignore[arg-type]


def test_pipeline_translate_mode_error_points_to_preprocess_phase():
    """The rejection message must tell callers where translation actually lives."""
    ctx = _make_ctx()
    with pytest.raises(ValueError) as excinfo:
        run_pipeline(ctx, mode="translate")  # type: ignore[arg-type]
    msg = str(excinfo.value)
    # The error message is the architectural signpost — assert it says
    # translation is a preprocess phase + points at the file that will
    # host it.
    assert "preprocess phase" in msg
    assert "preprocess/phases/translation.py" in msg


def test_pipeline_fixed_composes_body_and_delegates():
    ctx = _make_ctx(
        config={"agent": {"step_limit": 10}},
        test_command="python test_kernel.py",
        metric="latency",
    )

    captured: dict = {}

    def _fake_run_fixed_mode(**kwargs):
        captured.update(kwargs)
        return "FAKE_RESULT"

    with patch(
        "minisweagent.agents.homogeneous.homogeneous_agent.run_homogeneous_agent",
        _fake_run_fixed_mode,
    ), patch(
        "minisweagent.memory.integration.assemble_memory_context",
        return_value="",
    ), patch(
        "minisweagent.tools.tools_runtime.ToolRuntime",
    ):
        result = run_pipeline(ctx, mode="fixed")

    assert result == "FAKE_RESULT"
    assert captured["task_content"] == "Optimize kernel X"
    assert captured["agent_config"]["test_command"] == "python test_kernel.py"
    assert captured["agent_config"]["metric"] == "latency"
    assert captured["agent_config"]["save_patch"] is True


def test_pipeline_planned_merges_addenda_into_commandment():
    ctx = _make_ctx(
        preprocess_ctx={
            "kernel_path": "/tmp/k.py",
            "commandment": "BASE COMMANDMENT",
        },
        rag_enabled=True,
        extra_addenda=["## DIRECTIVES\n- try ILP"],
    )

    captured: dict = {}

    def _fake_run_orchestrator(**kwargs):
        captured.update(kwargs)
        return "PLANNED_RESULT"

    with patch(
        "minisweagent.run.orchestrator.run_orchestrator",
        _fake_run_orchestrator,
    ), patch("minisweagent.tools.tools_runtime.ToolRuntime"):
        result = run_pipeline(ctx, mode="planned")

    assert result == "PLANNED_RESULT"
    assert captured["heterogeneous"] is True
    pctx = captured["preprocess_ctx"]
    assert pctx["rag_enabled"] is True
    assert pctx["user_instructions"] == "Optimize kernel X"
    assert "BASE COMMANDMENT" in pctx["commandment"]
    assert "## DIRECTIVES" in pctx["commandment"]


def test_pipeline_mixed_routes_to_orchestrator():
    ctx = _make_ctx(
        preprocess_ctx={
            "kernel_path": "/tmp/k.py",
            "discovery": {"kernel": {"type": "triton"}},
            "commandment": "BASE",
        },
    )

    captured: dict = {}

    def _fake_run_orchestrator(**kwargs):
        captured.update(kwargs)
        return "MIXED_RESULT"

    with patch(
        "minisweagent.run.orchestrator.run_orchestrator",
        _fake_run_orchestrator,
    ), patch("minisweagent.tools.tools_runtime.ToolRuntime"):
        result = run_pipeline(ctx, mode="mixed")

    assert result == "MIXED_RESULT"
    assert captured["heterogeneous"] is True
