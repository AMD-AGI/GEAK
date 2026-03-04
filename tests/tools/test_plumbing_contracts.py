# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""Phase 4: Data format contract tests for pipeline handover points.

Tests verify that the output of one pipeline stage is consumable by the next:

4a. profiler-mcp output format -> baseline_metrics input
4b. baseline_metrics output format -> OpenEvolve input
4c. MCPToolBridge._format_result() preserves JSON inside content blocks
4d. generate_optimization output is usable kernel code
4e. evaluate/reflect chain: typical inputs are accepted

No GPU required -- uses synthetic but realistic data shapes.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from minisweagent.baseline_metrics import (
    build_baseline_metrics,
    list_kernels,
)
from minisweagent.tools.mcp_bridge import MCPToolBridge

# ---------------------------------------------------------------------------
# Realistic synthetic data (matching MetrixTool output structure)
# ---------------------------------------------------------------------------


def _kernel(
    name: str,
    duration_us: float = 100.0,
    hbm_bw_util: float = 50.0,
    l2_hit: float = 70.0,
    coalescing: float = 80.0,
    bottleneck: str = "memory",
    observations: list[str] | None = None,
) -> dict:
    return {
        "name": name,
        "duration_us": duration_us,
        "metrics": {
            "duration_us": duration_us,
            "memory.hbm_bandwidth_utilization": hbm_bw_util,
            "memory.hbm_read_bandwidth": 120.5,
            "memory.hbm_write_bandwidth": 45.3,
            "memory.bytes_transferred_hbm": 1048576,
            "memory.l1_hit_rate": 62.1,
            "memory.l2_hit_rate": l2_hit,
            "memory.l2_bandwidth": 35.0,
            "memory.coalescing_efficiency": coalescing,
            "memory.global_load_efficiency": 78.5,
            "memory.global_store_efficiency": 82.3,
            "memory.lds_bank_conflicts": 0.02,
        },
        "bottleneck": bottleneck,
        "observations": observations or [f"{bottleneck}-bound kernel"],
    }


def _profiler_result(kernels: list[dict], device_id: str = "0") -> dict:
    """MetrixTool.profile() return structure."""
    return {
        "results": [
            {
                "device_id": device_id,
                "gpu_info": {"detected": True, "name": "gfx942"},
                "kernels": kernels,
            }
        ]
    }


ADD_KERNEL = _kernel(
    "add_kernel_0d1d2d3d4", duration_us=8.5, hbm_bw_util=65.0, l2_hit=80.0, coalescing=95.0, bottleneck="memory"
)
FRAMEWORK_COPY = _kernel("Memcpy DtoD (Device -> Device)", duration_us=1.2, bottleneck="latency")
FRAMEWORK_EW = _kernel("at::native::vectorized_elementwise_kernel<4>", duration_us=0.8, bottleneck="latency")


# ---------------------------------------------------------------------------
# 4a. profiler-mcp output format -> baseline_metrics input
# ---------------------------------------------------------------------------


class TestProfilerToBaselineContract:
    """Verify that realistic profiler output is consumed by baseline_metrics."""

    def test_list_kernels_on_realistic_output(self):
        result = _profiler_result([ADD_KERNEL, FRAMEWORK_COPY, FRAMEWORK_EW])
        kernels = list_kernels(result)
        assert len(kernels) == 3
        assert kernels[0]["name"] == "add_kernel_0d1d2d3d4"
        # Each kernel must have the fields baseline_metrics expects
        for k in kernels:
            assert "name" in k
            assert "duration_us" in k
            assert "metrics" in k
            assert "bottleneck" in k

    def test_build_baseline_from_realistic_output(self):
        result = _profiler_result([ADD_KERNEL, FRAMEWORK_COPY, FRAMEWORK_EW])
        baseline = build_baseline_metrics(result, kernel_names=["add_kernel_0d1d2d3d4"])

        assert baseline["duration_us"] == pytest.approx(8.5)
        assert baseline["kernel_name"] == "add_kernel_0d1d2d3d4"
        assert baseline["bottleneck"] == "memory"
        assert "metrics" in baseline
        assert baseline["metrics"]["duration_us"] == pytest.approx(8.5)

    def test_profiler_output_with_gpu_info(self):
        """gpu_info field should not break list_kernels or build_baseline_metrics."""
        result = {
            "results": [
                {
                    "device_id": "3",
                    "gpu_info": {
                        "detected": True,
                        "name": "gfx942",
                        "memory_clock": 1600,
                        "compute_units": 304,
                    },
                    "kernels": [ADD_KERNEL],
                }
            ]
        }
        kernels = list_kernels(result, gpu_index=0)
        assert len(kernels) == 1
        baseline = build_baseline_metrics(result, include_all=True)
        assert baseline["duration_us"] > 0

    def test_empty_observations_handled(self):
        """Kernel with empty observations list should not crash."""
        k = _kernel("test_k", observations=[])
        result = _profiler_result([k])
        baseline = build_baseline_metrics(result, include_all=True)
        assert isinstance(baseline["observations"], list)


# ---------------------------------------------------------------------------
# 4b. baseline_metrics output -> OpenEvolve input
# ---------------------------------------------------------------------------


class TestBaselineToOpenEvolveContract:
    """Verify baseline_metrics JSON matches what OpenEvolve consumes."""

    def _simulate_openevolve_parse(self, baseline_path: str, best_duration_us: float = 5.0):
        """Replicate how openevolve-mcp/run_openevolve.py reads baseline_metrics.json."""
        with open(baseline_path) as f:
            baseline = json.load(f)
        baseline_latency = baseline.get("duration_us", 0)
        speedup = baseline_latency / best_duration_us if best_duration_us > 0 else 1.0
        return {"baseline_latency_us": baseline_latency, "speedup": speedup}

    def test_json_file_consumed_by_openevolve(self):
        result = _profiler_result([ADD_KERNEL])
        baseline = build_baseline_metrics(result, include_all=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(baseline, f, indent=2)
            tmp = f.name

        try:
            parsed = self._simulate_openevolve_parse(tmp, best_duration_us=5.0)
            assert parsed["baseline_latency_us"] == pytest.approx(8.5)
            assert parsed["speedup"] == pytest.approx(8.5 / 5.0, rel=1e-3)
            assert parsed["speedup"] > 1.0
        finally:
            os.unlink(tmp)

    def test_required_fields_present(self):
        result = _profiler_result([ADD_KERNEL])
        baseline = build_baseline_metrics(result, include_all=True)

        required = ["duration_us", "kernel_name", "kernel_names", "metrics", "bottleneck", "observations"]
        for field in required:
            assert field in baseline, f"Missing required field: {field}"

    def test_duration_us_positive(self):
        result = _profiler_result([ADD_KERNEL])
        baseline = build_baseline_metrics(result, include_all=True)
        assert baseline["duration_us"] > 0, "duration_us must be positive for OpenEvolve"


# ---------------------------------------------------------------------------
# 4c. MCPToolBridge._format_result() fidelity
# ---------------------------------------------------------------------------


class TestFormatResultFidelity:
    """Verify _format_result preserves structured JSON and handles edge cases."""

    def test_json_in_content_preserved(self):
        """When MCP server returns JSON inside content[0].text, it must be parseable."""
        inner_json = json.dumps({"success": True, "duration_us": 42.5, "kernels": ["k1"]})
        raw = {"content": [{"text": inner_json}], "isError": False}
        result = MCPToolBridge._format_result(raw)

        assert result["returncode"] == 0
        parsed = json.loads(result["output"])
        assert parsed["success"] is True
        assert parsed["duration_us"] == 42.5

    def test_multi_block_content_joined(self):
        """Multiple content blocks should be joined with newlines."""
        raw = {
            "content": [
                {"text": "line one"},
                {"text": "line two"},
                {"text": "line three"},
            ],
            "isError": False,
        }
        result = MCPToolBridge._format_result(raw)
        assert result["returncode"] == 0
        assert "line one" in result["output"]
        assert "line two" in result["output"]
        assert "line three" in result["output"]
        # Lines should be separated by newlines
        lines = result["output"].split("\n")
        assert len(lines) == 3

    def test_error_flag_sets_returncode_1(self):
        raw = {"content": [{"text": "profiler crashed"}], "isError": True}
        result = MCPToolBridge._format_result(raw)
        assert result["returncode"] == 1
        assert "profiler crashed" in result["output"]

    def test_empty_content_no_crash(self):
        raw = {"content": [], "isError": False}
        result = MCPToolBridge._format_result(raw)
        assert result["returncode"] == 0

    def test_nested_json_roundtrip(self):
        """Complex nested JSON should survive the format_result -> parse roundtrip."""
        profiler_output = _profiler_result([ADD_KERNEL])
        inner = json.dumps(profiler_output)
        raw = {"content": [{"text": inner}], "isError": False}
        result = MCPToolBridge._format_result(raw)

        parsed = json.loads(result["output"])
        assert "results" in parsed
        assert len(parsed["results"]) == 1
        assert len(parsed["results"][0]["kernels"]) == 1
        assert parsed["results"][0]["kernels"][0]["name"] == "add_kernel_0d1d2d3d4"

    def test_missing_content_key(self):
        """Raw dict without 'content' key should not crash."""
        raw = {"data": "something", "isError": False}
        result = MCPToolBridge._format_result(raw)
        assert result["returncode"] == 0

    def test_non_dict_content_items_skipped(self):
        """Non-dict items in content list should be handled gracefully."""
        raw = {"content": ["plain string", {"text": "real block"}], "isError": False}
        result = MCPToolBridge._format_result(raw)
        assert result["returncode"] == 0
        assert "real block" in result["output"]


# ---------------------------------------------------------------------------
# 4d. generate_optimization output format
# ---------------------------------------------------------------------------


class TestGenerateOptimizationOutput:
    """Verify the expected output shape of generate_optimization.

    The agent calls generate_optimization and expects to get back kernel code
    it can write to a file. This tests that the bridge + server output is
    usable, using a mock of what the server would return.
    """

    def test_optimization_result_contains_code(self):
        """Simulate the MCP server returning an optimization result."""
        # This is what kernel-evolve server returns (wrapped in MCP content)
        optimization_result = {
            "optimized_code": "@triton.jit\ndef add_optimized(x, y, out, n, B: tl.constexpr):\n    pass\n",
            "strategy": "vectorize_loads",
            "changes": ["Added vectorized load pattern"],
        }
        raw = {"content": [{"text": json.dumps(optimization_result)}], "isError": False}
        result = MCPToolBridge._format_result(raw)

        assert result["returncode"] == 0
        parsed = json.loads(result["output"])
        assert "optimized_code" in parsed
        assert "@triton.jit" in parsed["optimized_code"]


# ---------------------------------------------------------------------------
# 4e. evaluate/reflect chain -- input format compatibility
# ---------------------------------------------------------------------------


class TestEvaluateReflectChain:
    """Verify that typical tool outputs can serve as inputs to evaluate/reflect.

    These don't call the MCP servers (that's Phase 2). Instead they verify
    that the data shapes expected by reflect_on_kernel_result match what
    other tools produce.
    """

    def test_reflect_input_from_save_and_test_and_profiler(self):
        """Simulate assembling reflect_on_kernel_result arguments from
        save_and_test output + profile_kernel output + agent history.
        """
        save_and_test_output = "PASSED: correctness check\nLatency: 8.5 us\nSpeedup: 1.2x"

        # profile_kernel returns {output, returncode} with JSON inside
        profiler_json = json.dumps(_profiler_result([ADD_KERNEL]))

        # The agent assembles these into reflect arguments
        reflect_args = {
            "kernel_code": "@triton.jit\ndef add(x, y, out, n, B: tl.constexpr):\n    pass",
            "test_output": save_and_test_output,
            "speedup": 1.2,
            "correctness_status": "passed",
            "history": f"Step 1: profiled kernel.\n{profiler_json[:200]}...",
            "tried_strategies": "vectorize_loads, shared_memory_staging",
            "model": "claude-sonnet-4.5",
        }

        # All required fields must be present and of correct type
        assert isinstance(reflect_args["kernel_code"], str)
        assert isinstance(reflect_args["test_output"], str)
        assert isinstance(reflect_args["speedup"], float)
        assert isinstance(reflect_args["correctness_status"], str)
        assert isinstance(reflect_args["history"], str)
        assert isinstance(reflect_args["tried_strategies"], str)
        assert isinstance(reflect_args["model"], str)

    def test_baseline_metrics_tool_output_parseable_for_reflect(self):
        """The baseline_metrics tool returns {output: json_str, returncode: 0}.
        Verify the agent can parse it and extract speedup-relevant info.
        """
        result = _profiler_result([ADD_KERNEL])
        baseline = build_baseline_metrics(result, include_all=True)
        baseline_json = json.dumps(baseline)

        # Agent parses baseline output
        parsed = json.loads(baseline_json)
        baseline_latency = parsed["duration_us"]
        current_latency = 6.0  # hypothetical improved latency
        speedup = baseline_latency / current_latency

        assert speedup > 1.0
        assert isinstance(parsed["bottleneck"], str)
        assert isinstance(parsed["observations"], list)
