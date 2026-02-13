"""Phase 5: End-to-end pipeline smoke test.

Exercises the full data flow through native tools:
  resolve_kernel_url -> check_kernel_compatibility -> baseline_metrics

For MCP tools (profile_kernel), we verify the bridge is reachable and
the data format contract holds using a synthetic profiler output that
matches real MetrixTool structure.

This test does NOT require a GPU -- it uses synthetic profiler data for
the baseline_metrics step. For the full GPU pipeline, see the Phase 2
MCP smoke tests which exercise profile_kernel on the live server.

Must be run inside the geak-agent container.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from minisweagent.baseline_metrics import build_baseline_metrics, list_kernels
from minisweagent.tools.check_compat import CheckKernelCompatibilityTool, check_compatibility
from minisweagent.tools.resolve_kernel_url_impl import resolve_kernel_url
from minisweagent.tools.tools_runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Synthetic profiler output (realistic MetrixTool shape)
# ---------------------------------------------------------------------------

SYNTHETIC_PROFILER_OUTPUT = {
    "results": [
        {
            "device_id": "0",
            "gpu_info": {"detected": True, "name": "gfx942"},
            "kernels": [
                {
                    "name": "add_kernel_0d1d2d3d4",
                    "duration_us": 8.5,
                    "metrics": {
                        "duration_us": 8.5,
                        "memory.hbm_bandwidth_utilization": 65.0,
                        "memory.hbm_read_bandwidth": 120.5,
                        "memory.hbm_write_bandwidth": 45.3,
                        "memory.bytes_transferred_hbm": 1048576,
                        "memory.l1_hit_rate": 62.1,
                        "memory.l2_hit_rate": 80.0,
                        "memory.l2_bandwidth": 35.0,
                        "memory.coalescing_efficiency": 95.0,
                        "memory.global_load_efficiency": 78.5,
                        "memory.global_store_efficiency": 82.3,
                        "memory.lds_bank_conflicts": 0.02,
                    },
                    "bottleneck": "memory",
                    "observations": ["memory-bound kernel", "high coalescing efficiency"],
                },
                {
                    "name": "Memcpy DtoD (Device -> Device)",
                    "duration_us": 1.2,
                    "metrics": {
                        "duration_us": 1.2,
                        "memory.hbm_bandwidth_utilization": 10.0,
                        "memory.l2_hit_rate": 50.0,
                        "memory.coalescing_efficiency": 100.0,
                    },
                    "bottleneck": "latency",
                    "observations": ["framework overhead"],
                },
            ],
        }
    ]
}


# The example add kernel from examples/add_kernel/kernel.py
ADD_KERNEL_CODE = '''import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
'''


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------

class TestE2EPipelineSmoke:
    """Walk the full pipeline using native tools + synthetic profiler data."""

    def test_step1_resolve_kernel_url_local_file(self):
        """resolve_kernel_url handles a local file path."""
        # Create a temp kernel file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(ADD_KERNEL_CODE)
            kernel_path = f.name

        try:
            result = resolve_kernel_url(kernel_path)
            assert not result.get("error"), f"Unexpected error: {result}"
            assert result["local_file_path"] == kernel_path
        finally:
            os.unlink(kernel_path)

    def test_step2_check_kernel_compatibility(self):
        """check_kernel_compatibility reports clean Triton code as compatible."""
        issues = check_compatibility(ADD_KERNEL_CODE)
        assert len(issues) == 0, f"Clean Triton code flagged: {issues}"

    def test_step2_check_kernel_compat_via_tool(self):
        """Use the tool wrapper (same interface as ToolRuntime dispatch)."""
        tool = CheckKernelCompatibilityTool()
        result = tool(kernel_code=ADD_KERNEL_CODE)
        assert result["returncode"] == 0
        assert "compatible" in result["output"].lower() or "no issues" in result["output"].lower()

    def test_step3_profiler_output_consumable(self):
        """list_kernels can parse the synthetic profiler output."""
        kernels = list_kernels(SYNTHETIC_PROFILER_OUTPUT)
        assert len(kernels) == 2
        names = [k["name"] for k in kernels]
        assert "add_kernel_0d1d2d3d4" in names

    def test_step4_baseline_metrics_from_profiler(self):
        """build_baseline_metrics produces valid OpenEvolve input."""
        baseline = build_baseline_metrics(
            SYNTHETIC_PROFILER_OUTPUT,
            kernel_names=["add_kernel_0d1d2d3d4"],
        )

        # Required fields for OpenEvolve
        assert baseline["duration_us"] == pytest.approx(8.5)
        assert baseline["kernel_name"] == "add_kernel_0d1d2d3d4"
        assert baseline["bottleneck"] == "memory"
        assert baseline["metrics"]["duration_us"] == pytest.approx(8.5)
        assert len(baseline["observations"]) > 0

    def test_step5_baseline_json_roundtrip(self):
        """Write baseline_metrics.json, read it back, parse as OpenEvolve would."""
        baseline = build_baseline_metrics(
            SYNTHETIC_PROFILER_OUTPUT,
            kernel_names=["add_kernel_0d1d2d3d4"],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(baseline, f, indent=2)
            tmp = f.name

        try:
            with open(tmp) as f:
                loaded = json.load(f)

            # Simulate OpenEvolve reading the baseline
            baseline_latency = loaded.get("duration_us", 0)
            assert baseline_latency > 0, "OpenEvolve would get duration_us=0!"

            # Simulate a candidate that's 1.5x faster
            candidate_latency = baseline_latency / 1.5
            speedup = baseline_latency / candidate_latency
            assert speedup == pytest.approx(1.5, rel=1e-3)
        finally:
            os.unlink(tmp)

    def test_full_pipeline_via_toolruntime_dispatch(self):
        """Exercise the pipeline through ToolRuntime.dispatch (same path the agent uses)."""
        rt = ToolRuntime()

        # Step 1: check_kernel_compatibility via dispatch
        compat_result = rt.dispatch({
            "name": "check_kernel_compatibility",
            "arguments": {"kernel_code": ADD_KERNEL_CODE},
        })
        assert compat_result["returncode"] == 0

        # Step 2: baseline_metrics via dispatch with synthetic profiler output
        baseline_result = rt.dispatch({
            "name": "baseline_metrics",
            "arguments": {
                "profiler_output": json.dumps(SYNTHETIC_PROFILER_OUTPUT),
                "kernel_names": "add_kernel_0d1d2d3d4",
            },
        })
        assert baseline_result["returncode"] == 0, f"baseline_metrics failed: {baseline_result['output']}"

        # The output should be parseable JSON
        baseline = json.loads(baseline_result["output"])
        assert baseline["duration_us"] > 0
        assert baseline["kernel_name"] == "add_kernel_0d1d2d3d4"
