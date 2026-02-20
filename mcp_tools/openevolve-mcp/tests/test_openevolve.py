"""Tests for openevolve-mcp server."""

import sys
from pathlib import Path

# Add openevolve-mcp to path
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from openevolve_mcp.server import mcp


class TestOpenEvolveMCPServer:
    def test_server_has_optimize_kernel(self):
        tools = mcp._tool_manager._tools
        assert "optimize_kernel" in tools

    def test_server_has_expected_tools(self):
        tools = mcp._tool_manager._tools
        tool_names = list(tools.keys())
        assert len(tool_names) >= 1
        assert "optimize_kernel" in tool_names
