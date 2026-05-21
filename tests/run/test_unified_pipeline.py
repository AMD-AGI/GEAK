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


def test_pipeline_delegates_to_unified_loop():
    """All valid modes delegate to _run_unified_loop."""
    ctx = _make_ctx()
    sentinel = object()

    with patch(
        "minisweagent.run.unified._run_unified_loop",
        return_value=sentinel,
    ) as mock_loop, patch("minisweagent.tools.tools_runtime.ToolRuntime"):
        result = run_pipeline(ctx, mode="fixed")

    assert result is sentinel
    mock_loop.assert_called_once_with(ctx, "fixed")


def test_pipeline_planned_delegates_to_unified_loop():
    ctx = _make_ctx()
    sentinel = object()

    with patch(
        "minisweagent.run.unified._run_unified_loop",
        return_value=sentinel,
    ) as mock_loop, patch("minisweagent.tools.tools_runtime.ToolRuntime"):
        result = run_pipeline(ctx, mode="planned")

    assert result is sentinel
    mock_loop.assert_called_once_with(ctx, "planned")


def test_pipeline_mixed_delegates_to_unified_loop():
    ctx = _make_ctx()
    sentinel = object()

    with patch(
        "minisweagent.run.unified._run_unified_loop",
        return_value=sentinel,
    ) as mock_loop, patch("minisweagent.tools.tools_runtime.ToolRuntime"):
        result = run_pipeline(ctx, mode="mixed")

    assert result is sentinel
    mock_loop.assert_called_once_with(ctx, "mixed")
