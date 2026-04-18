"""End-to-end tests for rag-mcp via FastMCP Client.

These tests connect to the server through the MCP protocol and invoke
real tool calls against the semantic index.  They are skipped automatically
when the index is not available.
"""

import json
import sys
from pathlib import Path

import pytest

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from fastmcp import Client
from rag_mcp.server import mcp

INDEX_PATH = Path.home() / ".cache" / "amd-ai-devtool" / "semantic-index"
skip_no_index = pytest.mark.skipif(
    not (INDEX_PATH / "index.faiss").exists(),
    reason="Semantic index not available",
)


@pytest.fixture
def client():
    return Client(mcp)


def _parse_content(result) -> dict:
    """Extract the dict payload from a CallToolResult."""
    text = result.content[0].text
    return json.loads(text)


# ------------------------------------------------------------------
# Tool listing
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_tools(client):
    async with client:
        tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == {"query", "optimize"}


# ------------------------------------------------------------------
# query tool
# ------------------------------------------------------------------

@skip_no_index
@pytest.mark.asyncio
async def test_query_basic(client):
    async with client:
        result = await client.call_tool("query", {"topic": "HIP programming"})
        data = _parse_content(result)
        assert "results" in data
        assert "count" in data
        assert "query" in data
        assert data["query"] == "HIP programming"
        assert data["count"] > 0
        assert "## Result 1:" in data["results"]


@skip_no_index
@pytest.mark.asyncio
async def test_query_with_layer(client):
    async with client:
        result = await client.call_tool("query", {"topic": "ROCm installation", "layer": "rocm"})
        data = _parse_content(result)
        assert "results" in data
        if data.get("count", 0) > 0:
            assert data["query"] == "ROCm installation"
        else:
            assert "message" in data


@skip_no_index
@pytest.mark.asyncio
async def test_query_with_top_k(client):
    async with client:
        result = await client.call_tool("query", {"topic": "GPU memory management", "top_k": 3})
        data = _parse_content(result)
        assert data["count"] <= 3


@skip_no_index
@pytest.mark.asyncio
async def test_query_no_results(client):
    async with client:
        result = await client.call_tool("query", {"topic": "xyzzy_nonexistent_topic_42"})
        data = _parse_content(result)
        if data["count"] == 0:
            assert "message" in data
            assert data["results"] == []


# ------------------------------------------------------------------
# optimize tool
# ------------------------------------------------------------------

@skip_no_index
@pytest.mark.asyncio
async def test_optimize_basic(client):
    async with client:
        result = await client.call_tool("optimize", {"code_type": "matrix multiplication"})
        data = _parse_content(result)
        assert "results" in data
        assert "count" in data
        assert "query" in data
        assert data["count"] > 0
        assert "## Suggestion 1:" in data["results"]


@skip_no_index
@pytest.mark.asyncio
async def test_optimize_with_gpu_model(client):
    async with client:
        result = await client.call_tool("optimize", {
            "code_type": "convolution",
            "gpu_model": "MI300X",
        })
        data = _parse_content(result)
        assert data["count"] > 0
        assert "MI300X" in data["query"]


@skip_no_index
@pytest.mark.asyncio
async def test_optimize_with_context(client):
    async with client:
        result = await client.call_tool("optimize", {
            "code_type": "GEMM",
            "context": "large batch inference",
            "gpu_model": "MI300X",
        })
        data = _parse_content(result)
        assert data["count"] > 0
        assert "optimization" in data["query"]
