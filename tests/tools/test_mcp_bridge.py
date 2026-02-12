"""Tests for MCPToolBridge -- sync wrapper around MCP servers."""

from unittest.mock import MagicMock

import pytest

from minisweagent.tools.mcp_bridge import MCPToolBridge, _BoundTool


class TestMCPToolBridge:
    def test_invalid_server_name(self):
        """Verify graceful error when server dir doesn't exist."""
        with pytest.raises(FileNotFoundError):
            MCPToolBridge("nonexistent-server-xyz")

    def test_tool_factory_returns_bound_tool(self):
        bridge = MCPToolBridge("test-server", server_config={"command": ["echo"], "cwd": "/tmp"})
        tool = bridge.tool("my_tool")
        assert isinstance(tool, _BoundTool)
        assert repr(tool) == "MCPTool(test-server::my_tool)"

    def test_bound_tool_calls_bridge(self):
        bridge = MCPToolBridge("test-server", server_config={"command": ["echo"], "cwd": "/tmp"})
        bridge.call_tool = MagicMock(return_value={"output": "ok", "returncode": 0})
        tool = bridge.tool("my_tool")
        result = tool(arg1="val1", arg2="val2")
        bridge.call_tool.assert_called_once_with("my_tool", {"arg1": "val1", "arg2": "val2"})
        assert result["returncode"] == 0

    def test_format_result_success(self):
        raw = {"content": [{"text": "hello world"}], "isError": False}
        result = MCPToolBridge._format_result(raw)
        assert result["returncode"] == 0
        assert "hello world" in result["output"]

    def test_format_result_error(self):
        raw = {"content": [{"text": "something broke"}], "isError": True}
        result = MCPToolBridge._format_result(raw)
        assert result["returncode"] == 1
        assert "something broke" in result["output"]

    def test_format_result_empty(self):
        raw = {"content": [], "isError": False}
        result = MCPToolBridge._format_result(raw)
        assert result["returncode"] == 0

    def test_default_config(self):
        # Use profiler-mcp which we know exists
        config = MCPToolBridge._default_config("profiler-mcp")
        assert "command" in config
        assert "profiler_mcp.server" in config["command"][-1]
        assert "cwd" in config
        assert "PYTHONPATH" in config.get("env", {})

    def test_default_config_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            MCPToolBridge._default_config("nonexistent-server-xyz")
