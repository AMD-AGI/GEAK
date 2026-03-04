# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Tests for MCPToolBridge -- sync wrapper around MCP servers."""

import json
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


class TestPersistentEventLoop:
    """Tests for the persistent background event loop in MCPToolBridge.

    The old implementation used asyncio.run() which creates and destroys a
    loop per call. This caused "Future attached to a different loop" when
    the cached MCPClient's subprocess pipes survived across calls.
    """

    def test_loop_created_lazily(self):
        bridge = MCPToolBridge("test-server", server_config={"command": ["echo"], "cwd": "/tmp"})
        assert bridge._loop is None
        loop = bridge._get_loop()
        assert loop is not None
        assert loop.is_running()
        bridge._shutdown_loop()

    def test_same_loop_reused_across_calls(self):
        bridge = MCPToolBridge("test-server", server_config={"command": ["echo"], "cwd": "/tmp"})
        loop1 = bridge._get_loop()
        loop2 = bridge._get_loop()
        assert loop1 is loop2
        bridge._shutdown_loop()

    def test_loop_runs_on_daemon_thread(self):
        bridge = MCPToolBridge("test-server", server_config={"command": ["echo"], "cwd": "/tmp"})
        bridge._get_loop()
        assert bridge._loop_thread is not None
        assert bridge._loop_thread.is_alive()
        assert bridge._loop_thread.daemon
        bridge._shutdown_loop()

    def test_shutdown_stops_loop(self):
        bridge = MCPToolBridge("test-server", server_config={"command": ["echo"], "cwd": "/tmp"})
        loop = bridge._get_loop()
        bridge._shutdown_loop()
        assert loop.is_closed() or not loop.is_running()

    def test_run_async_uses_persistent_loop(self):
        """Verify _run_async schedules on the persistent loop, not a throwaway one."""
        bridge = MCPToolBridge("test-server", server_config={"command": ["echo"], "cwd": "/tmp"})
        loops_seen = []

        async def capture_loop():
            import asyncio

            loops_seen.append(asyncio.get_running_loop())
            return "ok"

        # Two calls should use the same loop
        bridge._run_async(capture_loop())
        bridge._run_async(capture_loop())

        assert len(loops_seen) == 2
        assert loops_seen[0] is loops_seen[1], (
            "Both calls must run on the same event loop to avoid 'Future attached to a different loop' errors"
        )
        bridge._shutdown_loop()


class TestMultiCallMCPServer:
    """Live test: multiple calls to the same MCP server bridge.

    This is the exact scenario that triggered the 'Future attached to a
    different loop' error in the rope optimization run. The kernel-ercs
    server is used because it has a non-LLM tool (get_amd_gpu_specs).
    """

    def test_consecutive_calls_same_bridge(self):
        """Two consecutive calls to the same bridge must both succeed.

        Before the fix, the second call would fail with:
            Task ... got Future <Future pending> attached to a different loop
        """
        bridge = MCPToolBridge("kernel-ercs", timeout=60)

        # First call -- starts subprocess + makes RPC call
        result1 = bridge.call_tool("get_amd_gpu_specs", {})
        assert result1["returncode"] == 0, f"First call failed: {result1['output']}"

        # Second call -- reuses the same subprocess (same loop)
        result2 = bridge.call_tool("get_amd_gpu_specs", {})
        assert result2["returncode"] == 0, f"Second call failed: {result2['output']}"

        # Both should return valid JSON
        data1 = json.loads(result1["output"])
        data2 = json.loads(result2["output"])
        assert data1 == data2  # same server, same tool, same result

        bridge._shutdown_loop()

    def test_consecutive_calls_different_tools_same_bridge(self):
        """Call two different tools on the same bridge sequentially."""
        bridge = MCPToolBridge("kernel-ercs", timeout=60)

        r1 = bridge.call_tool("get_amd_gpu_specs", {})
        assert r1["returncode"] == 0, f"get_amd_gpu_specs failed: {r1['output']}"

        triton_code = "@triton.jit\ndef add(x, y, out, n, B: tl.constexpr): pass"
        r2 = bridge.call_tool("check_kernel_compatibility", {"kernel_code": triton_code})
        assert r2["returncode"] == 0, f"check_kernel_compatibility failed: {r2['output']}"

        bridge._shutdown_loop()

    def test_works_from_async_context(self):
        """Bridge must work when called from inside a running event loop.

        This simulates the GEAK agent scenario where the agent's main loop
        is running and tools are called synchronously from within it.
        """
        import asyncio

        results = []

        async def agent_main():
            # Simulate the agent calling MCP tools from an async context
            bridge = MCPToolBridge("kernel-ercs", timeout=60)

            r1 = bridge.call_tool("get_amd_gpu_specs", {})
            results.append(r1)

            r2 = bridge.call_tool("get_amd_gpu_specs", {})
            results.append(r2)

            bridge._shutdown_loop()

        asyncio.run(agent_main())

        assert len(results) == 2
        assert results[0]["returncode"] == 0, f"Call 1 from async failed: {results[0]['output']}"
        assert results[1]["returncode"] == 0, f"Call 2 from async failed: {results[1]['output']}"
