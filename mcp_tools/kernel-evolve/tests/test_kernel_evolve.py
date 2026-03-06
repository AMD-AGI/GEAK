# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Tests for kernel-evolve MCP server."""

import asyncio
import sys
from pathlib import Path

import pytest

# Add kernel-evolve to path
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from kernel_evolve.server import get_optimization_strategies, mcp


def _list_tool_names():
    tools = asyncio.run(mcp.list_tools())
    return [t.name for t in tools]


class TestKernelEvolveServer:
    def test_server_has_expected_tools(self):
        tool_names = _list_tool_names()
        expected = ["generate_optimization", "mutate_kernel", "crossover_kernels", "get_optimization_strategies"]
        for name in expected:
            assert name in tool_names, f"Missing tool: {name}"

    @pytest.mark.parametrize("bottleneck", ["compute", "memory", "latency", "lds", "balanced"])
    def test_get_strategies(self, bottleneck):
        result = get_optimization_strategies(bottleneck=bottleneck)
        assert "strategies" in result
        assert len(result["strategies"]) > 0
        assert result["bottleneck"] == bottleneck

    def test_get_strategies_invalid_type(self):
        result = get_optimization_strategies(bottleneck="invalid_type")
        assert isinstance(result, dict)
