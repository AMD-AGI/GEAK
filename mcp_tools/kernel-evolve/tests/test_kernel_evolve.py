"""Tests for kernel-evolve MCP server."""

import sys
from pathlib import Path

import pytest

# Add kernel-evolve to path
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from kernel_evolve.server import get_optimization_strategies, mcp


class TestKernelEvolveServer:
    def test_server_has_expected_tools(self):
        tools = mcp._tool_manager._tools
        expected = ["generate_optimization", "mutate_kernel", "crossover_kernels", "get_optimization_strategies"]
        for name in expected:
            assert name in tools, f"Missing tool: {name}"

    @pytest.mark.parametrize("bottleneck", ["compute", "memory", "latency", "lds", "balanced"])
    def test_get_strategies(self, bottleneck):
        result = get_optimization_strategies.fn(bottleneck=bottleneck)
        assert "strategies" in result
        assert len(result["strategies"]) > 0
        assert result["bottleneck"] == bottleneck

    def test_get_strategies_invalid_type(self):
        result = get_optimization_strategies.fn(bottleneck="invalid_type")
        # Should still return something (possibly empty or default)
        assert isinstance(result, dict)
