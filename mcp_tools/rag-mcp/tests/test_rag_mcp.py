"""Tests for rag-mcp server."""

import asyncio
import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from rag_mcp.server import mcp


def _get_tool_names() -> set[str]:
    tools = asyncio.run(mcp.list_tools())
    return {t.name for t in tools}


class TestRAGMCPServer:
    def test_server_has_query_tool(self):
        assert "query" in _get_tool_names()

    def test_server_has_optimize_tool(self):
        assert "optimize" in _get_tool_names()

    def test_server_has_exactly_two_tools(self):
        assert len(_get_tool_names()) == 2

    def test_server_tool_names(self):
        assert _get_tool_names() == {"query", "optimize"}
