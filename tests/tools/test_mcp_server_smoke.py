"""Phase 2: Live smoke tests for MCP servers -- no mocking.

These tests start real MCP server subprocesses via MCPToolBridge and verify:
1. The subprocess starts without crashing
2. Tools respond with the expected {output, returncode} format
3. Non-LLM tools return valid results with synthetic inputs

MCP servers that require an LLM (kernel-evolve, kernel-ercs evaluate/reflect)
are tested with a connectivity check only -- we verify the subprocess starts
and tools are listed, but don't invoke tools that would make paid API calls
unless AMD_LLM_API_KEY is available.

Must be run inside the geak-agent container where all MCP deps are installed.
"""

from __future__ import annotations

import json
import os

import pytest

from minisweagent.tools.mcp_bridge import MCPToolBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_amd_llm_key() -> bool:
    return bool(os.environ.get("AMD_LLM_API_KEY") or os.environ.get("LLM_GATEWAY_KEY"))


# Skip marker for tests requiring the LLM key
requires_llm_key = pytest.mark.skipif(
    not _has_amd_llm_key(),
    reason="AMD_LLM_API_KEY not set -- skipping LLM-dependent MCP tool test",
)


# ---------------------------------------------------------------------------
# kernel-ercs: non-LLM tools work without API key
# ---------------------------------------------------------------------------


class TestKernelErcsSmoke:
    """Live smoke tests for kernel-ercs MCP server."""

    def _bridge(self) -> MCPToolBridge:
        return MCPToolBridge("kernel-ercs", timeout=60)

    def test_get_amd_gpu_specs(self):
        """get_amd_gpu_specs requires no LLM -- pure data lookup."""
        bridge = self._bridge()
        result = bridge.call_tool("get_amd_gpu_specs", {})

        assert result["returncode"] == 0, f"Unexpected error: {result['output']}"
        # Output should be parseable JSON with GPU spec fields
        data = json.loads(result["output"])
        assert "gpu" in data or "specs" in data or "architecture" in data, (
            f"Unexpected GPU specs structure: {list(data.keys())}"
        )

    def test_check_kernel_compatibility(self):
        """check_kernel_compatibility scans code -- no LLM needed."""
        bridge = self._bridge()
        triton_code = """
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
"""
        result = bridge.call_tool("check_kernel_compatibility", {"kernel_code": triton_code})

        assert result["returncode"] == 0, f"Unexpected error: {result['output']}"
        # Clean Triton code should report as compatible
        data = json.loads(result["output"])
        assert "compatible" in data or "issues" in data or "status" in data, (
            f"Unexpected compatibility structure: {list(data.keys())}"
        )

    @requires_llm_key
    def test_evaluate_kernel_quality(self):
        """evaluate_kernel_quality requires LLM -- only run if key available."""
        bridge = self._bridge()
        simple_kernel = "import triton\nimport triton.language as tl\n\n@triton.jit\ndef add(x_ptr, y_ptr, out, n, BLOCK: tl.constexpr):\n    pid = tl.program_id(0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n    tl.store(out + offs, tl.load(x_ptr + offs) + tl.load(y_ptr + offs))\n"
        result = bridge.call_tool(
            "evaluate_kernel_quality",
            {
                "kernel_code": simple_kernel,
                "model": "claude-sonnet-4.5",
            },
        )
        assert result["returncode"] == 0, f"evaluate_kernel_quality failed: {result['output']}"

    @requires_llm_key
    def test_reflect_on_kernel_result(self):
        """reflect_on_kernel_result requires LLM."""
        bridge = self._bridge()
        result = bridge.call_tool(
            "reflect_on_kernel_result",
            {
                "kernel_code": "@triton.jit\ndef add(x, y, out, n, B: tl.constexpr): pass",
                "test_output": "PASSED correctness, latency 120us",
                "speedup": 1.05,
                "correctness_status": "passed",
                "history": "",
                "tried_strategies": "",
                "model": "claude-sonnet-4.5",
            },
        )
        assert result["returncode"] == 0, f"reflect_on_kernel_result failed: {result['output']}"


# ---------------------------------------------------------------------------
# kernel-evolve: all tools need LLM except get_optimization_strategies
#                and suggest_kernel_params
# ---------------------------------------------------------------------------


class TestKernelEvolveSmoke:
    """Live smoke tests for kernel-evolve MCP server."""

    def _bridge(self) -> MCPToolBridge:
        return MCPToolBridge("kernel-evolve", timeout=60)

    def test_get_optimization_strategies(self):
        """get_optimization_strategies is a data lookup -- no LLM needed."""
        bridge = self._bridge()
        result = bridge.call_tool("get_optimization_strategies", {"bottleneck": "memory"})

        assert result["returncode"] == 0, f"Unexpected error: {result['output']}"
        data = json.loads(result["output"])
        assert "strategies" in data, f"Expected 'strategies' key, got: {list(data.keys())}"
        assert len(data["strategies"]) > 0, "Expected at least one strategy"

    def test_suggest_kernel_params(self):
        """suggest_kernel_params is a data lookup -- no LLM needed."""
        bridge = self._bridge()
        result = bridge.call_tool(
            "suggest_kernel_params",
            {
                "kernel_type": "elementwise",
                "problem_size": 1048576,
            },
        )

        assert result["returncode"] == 0, f"Unexpected error: {result['output']}"
        data = json.loads(result["output"])
        assert "block_size" in data or "params" in data, (
            f"Expected kernel params (block_size, num_warps, ...), got: {list(data.keys())}"
        )

    @requires_llm_key
    def test_generate_optimization(self):
        """generate_optimization requires LLM."""
        bridge = self._bridge()
        kernel = "@triton.jit\ndef add(x, y, out, n, B: tl.constexpr):\n    pid = tl.program_id(0)\n    offs = pid * B + tl.arange(0, B)\n    tl.store(out + offs, tl.load(x + offs) + tl.load(y + offs))\n"
        result = bridge.call_tool(
            "generate_optimization",
            {
                "kernel_code": kernel,
                "bottleneck": "memory",
                "strategy": "vectorize_loads",
                "model": "claude-sonnet-4.5",
            },
        )
        assert result["returncode"] == 0, f"generate_optimization failed: {result['output']}"


# ---------------------------------------------------------------------------
# profiler-mcp: requires GPU for actual profiling, but we can test
#               that the server starts and the tool is reachable
# ---------------------------------------------------------------------------


class TestProfilerMcpSmoke:
    """Live smoke tests for profiler-mcp MCP server."""

    def _bridge(self) -> MCPToolBridge:
        return MCPToolBridge("profiler-mcp", timeout=60)

    def test_server_starts_and_reachable(self):
        """Verify the profiler-mcp subprocess starts without crashing.

        We call profile_kernel with an intentionally bad command to confirm
        the server is alive and returns a proper error (not a crash).
        """
        bridge = self._bridge()
        result = bridge.call_tool(
            "profile_kernel",
            {
                "command": "echo 'smoke test -- not a real kernel'",
                "backend": "metrix",
                "quick": True,
            },
        )
        # We expect either success (unlikely with echo) or a clean error
        # from the server -- NOT a crash / MCP transport error.
        assert isinstance(result, dict)
        assert "output" in result
        assert "returncode" in result
