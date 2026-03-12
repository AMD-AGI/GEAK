"""Phase 3: Verify ToolRuntime registers all expected tools and dispatch works.

Tests:
1. All expected tool names are present in _tool_table after construction
2. Dispatch of native tools returns the expected {output, returncode} format
3. Unknown tool names raise ValueError
4. MCP tool entries exist (as _BoundTool instances) even without calling them
"""

from __future__ import annotations

import pytest

from minisweagent.tools.tools_runtime import ToolRuntime

# ---------------------------------------------------------------------------
# Expected tool names
# ---------------------------------------------------------------------------

EXPECTED_NATIVE_TOOLS = {
    "bash",
    "str_replace_editor",
    "save_and_test",
    "submit",
    "resolve_kernel_url",
    "baseline_metrics",
    "check_kernel_compatibility",
    "sub_agent",
}

EXPECTED_MCP_TOOLS = {
    "profile_kernel",
    "generate_optimization",
    "evaluate_kernel_quality",
    "reflect_on_kernel_result",
}


# ---------------------------------------------------------------------------
# Tests: tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Verify all tools are registered in _tool_table after construction."""

    def test_all_native_tools_registered(self):
        rt = ToolRuntime()
        registered = set(rt._tool_table.keys())
        missing = EXPECTED_NATIVE_TOOLS - registered
        assert not missing, f"Missing native tools: {missing}"

    def test_all_mcp_tools_registered(self):
        rt = ToolRuntime()
        registered = set(rt._tool_table.keys())
        missing = EXPECTED_MCP_TOOLS - registered
        assert not missing, f"Missing MCP tools: {missing}"

    def test_mcp_tools_are_bound_tool_instances(self):
        """MCP tools should be _BoundTool instances (from mcp_bridge)."""
        from minisweagent.tools.mcp_bridge import _BoundTool

        rt = ToolRuntime()
        for name in EXPECTED_MCP_TOOLS:
            tool = rt._tool_table[name]
            assert isinstance(tool, _BoundTool), f"Tool '{name}' should be _BoundTool, got {type(tool).__name__}"

    def test_strategy_manager_excluded_by_default(self):
        rt = ToolRuntime(use_strategy_manager=False)
        assert "strategy_manager" not in rt._tool_table

    def test_strategy_manager_included_when_enabled(self):
        rt = ToolRuntime(use_strategy_manager=True)
        assert "strategy_manager" in rt._tool_table


# ---------------------------------------------------------------------------
# Tests: dispatch
# ---------------------------------------------------------------------------


class TestDispatch:
    """Verify ToolRuntime.dispatch routes correctly and returns proper format."""

    def test_dispatch_bash_echo(self):
        rt = ToolRuntime()
        result = rt.dispatch(
            {
                "name": "bash",
                "arguments": {"command": "echo hello_dispatch_test"},
            }
        )
        assert isinstance(result, dict)
        assert "output" in result
        assert "returncode" in result
        assert result["returncode"] == 0
        assert "hello_dispatch_test" in result["output"]

    def test_dispatch_bash_missing_command(self):
        """Dispatch should handle missing 'command' gracefully (not crash)."""
        rt = ToolRuntime()
        result = rt.dispatch(
            {
                "name": "bash",
                "arguments": {},
            }
        )
        # Should not raise -- the runtime injects an empty command
        assert isinstance(result, dict)

    def test_dispatch_check_kernel_compatibility(self):
        rt = ToolRuntime()
        result = rt.dispatch(
            {
                "name": "check_kernel_compatibility",
                "arguments": {
                    "kernel_code": "import triton\n@triton.jit\ndef k(): pass",
                },
            }
        )
        assert isinstance(result, dict)
        assert result["returncode"] == 0

    def test_dispatch_unknown_tool_raises(self):
        rt = ToolRuntime()
        with pytest.raises(ValueError, match="Unknown tool"):
            rt.dispatch({"name": "nonexistent_tool_xyz", "arguments": {}})

    def test_dispatch_baseline_metrics_bad_json(self):
        """baseline_metrics with bad JSON should return error, not crash."""
        rt = ToolRuntime()
        result = rt.dispatch(
            {
                "name": "baseline_metrics",
                "arguments": {"profiler_output": "not valid json"},
            }
        )
        assert isinstance(result, dict)
        assert result["returncode"] == 1

    def test_dispatch_resolve_kernel_url_empty(self):
        """resolve_kernel_url with empty URL should return error."""
        rt = ToolRuntime()
        result = rt.dispatch(
            {
                "name": "resolve_kernel_url",
                "arguments": {"url": ""},
            }
        )
        assert isinstance(result, dict)
        # Should fail gracefully
        assert result["returncode"] == 1 or "error" in result["output"].lower()


# ---------------------------------------------------------------------------
# Tests: tools_list API
# ---------------------------------------------------------------------------


class TestToolsList:
    """Verify the get_tools_list() API returns correct tool definitions."""

    def test_tools_list_returns_list_of_dicts(self):
        rt = ToolRuntime()
        tools = rt.get_tools_list()
        assert isinstance(tools, list)
        assert len(tools) > 0
        for tool in tools:
            assert "name" in tool, f"Tool definition missing 'name': {tool}"

    def test_tools_list_excludes_strategy_manager_by_default(self):
        rt = ToolRuntime(use_strategy_manager=False)
        names = {t["name"] for t in rt.get_tools_list()}
        assert "strategy_manager" not in names

    def test_tools_list_includes_strategy_manager_when_enabled(self):
        rt = ToolRuntime(use_strategy_manager=True)
        names = {t["name"] for t in rt.get_tools_list()}
        assert "strategy_manager" in names
