# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Tests for openevolve-mcp server."""

import asyncio
import sys
from pathlib import Path

# Add openevolve-mcp to path
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from openevolve_mcp.server import mcp


def _list_tool_names():
    tools = asyncio.run(mcp.list_tools())
    return [t.name for t in tools]


class TestOpenEvolveMCPServer:
    def test_server_has_optimize_kernel(self):
        assert "optimize_kernel" in _list_tool_names()

    def test_server_has_expected_tools(self):
        tool_names = _list_tool_names()
        assert len(tool_names) >= 1
        assert "optimize_kernel" in tool_names
