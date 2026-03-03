"""
Manual test script for rag-mcp server.

Usage:
    python tests/test_manual.py

Modify the variables below to test different queries.
"""

import asyncio
import json
import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from fastmcp import Client
from rag_mcp.server import mcp

# ====================================================================
# Modify these variables to test different queries
# ====================================================================

QUERY_TOPIC = "HIP programming"
QUERY_LAYER = None          # e.g. "hip", "rocm", "ai_frameworks", or None
QUERY_TOP_K = None          # e.g. 3, 5, or None (use server default)

OPTIMIZE_CODE_TYPE = "matrix multiplication"
OPTIMIZE_GPU_MODEL = None   # e.g. "MI300X", or None
OPTIMIZE_CONTEXT = None     # e.g. "large batch inference", or None
OPTIMIZE_TOP_K = None       # e.g. 3, 5, or None (use server default)

# ====================================================================


def _print_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _pretty_print(data: dict):
    print(json.dumps(data, indent=2, ensure_ascii=False))


async def main():
    client = Client(mcp)

    async with client:
        # ---- 1. list_tools ----
        _print_separator("1. list_tools")
        tools = await client.list_tools()
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        # ---- 2. query ----
        _print_separator(f"2. query(topic={QUERY_TOPIC!r}, layer={QUERY_LAYER!r}, top_k={QUERY_TOP_K!r})")
        query_args = {"topic": QUERY_TOPIC}
        if QUERY_LAYER is not None:
            query_args["layer"] = QUERY_LAYER
        if QUERY_TOP_K is not None:
            query_args["top_k"] = QUERY_TOP_K

        result = await client.call_tool("query", query_args)
        data = json.loads(result.content[0].text)
        _pretty_print(data)

        # ---- 3. optimize ----
        _print_separator(f"3. optimize(code_type={OPTIMIZE_CODE_TYPE!r}, gpu_model={OPTIMIZE_GPU_MODEL!r}, context={OPTIMIZE_CONTEXT!r}, top_k={OPTIMIZE_TOP_K!r})")
        opt_args = {"code_type": OPTIMIZE_CODE_TYPE}
        if OPTIMIZE_GPU_MODEL is not None:
            opt_args["gpu_model"] = OPTIMIZE_GPU_MODEL
        if OPTIMIZE_CONTEXT is not None:
            opt_args["context"] = OPTIMIZE_CONTEXT
        if OPTIMIZE_TOP_K is not None:
            opt_args["top_k"] = OPTIMIZE_TOP_K

        result = await client.call_tool("optimize", opt_args)
        data = json.loads(result.content[0].text)
        _pretty_print(data)


if __name__ == "__main__":
    asyncio.run(main())
