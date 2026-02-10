"""
Test the data handover from kernel-profile (MetrixTool) → baseline_metrics.json → OpenEvolve.

Design principle: kernel selection is the LLM agent's job.  The
``build_baseline_metrics`` utility is a *formatting* layer — it takes the
agent's explicit choice (by name, index, or "all") and produces the JSON
that OpenEvolve expects.

These tests validate:
1. list_kernels surfaces all profiled kernels for the agent to inspect
2. build_baseline_metrics formats the agent's choice correctly
3. Aggregation across multiple agent-chosen kernels is correct
4. The output is compatible with what openevolve-mcp reads
5. Explicit selection prevents silent wrong-kernel bugs
6. JSON round-trip preserves all fields

No GPU required — uses synthetic MetrixTool-shaped dicts.
"""

import json
import os
import tempfile

import pytest

from geakagent.baseline_metrics import (
    aggregate_metrics,
    build_baseline_metrics,
    list_kernels,
)


# ---------------------------------------------------------------------------
# Fixtures: realistic profiler output
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
    """Wrap kernels into MetrixTool.profile() return structure."""
    return {"results": [{"device_id": device_id, "gpu_info": {"detected": False}, "kernels": kernels}]}


# Reusable kernel fixtures
TOPK_1 = _kernel("topk_stage1", duration_us=114.55, hbm_bw_util=2.34, l2_hit=49.72, coalescing=25.0, bottleneck="latency")
TOPK_2 = _kernel("topk_stage2", duration_us=38.20, hbm_bw_util=18.5, l2_hit=72.0, coalescing=60.0, bottleneck="memory")
ROPE   = _kernel("fused_qkv_rope_triton", duration_us=220.0, hbm_bw_util=45.0, l2_hit=55.0, bottleneck="memory")
FW_EW  = _kernel("at::native::vectorized_elementwise_kernel", duration_us=5.2, bottleneck="latency")
FW_RED = _kernel("reduce_kernel<float>", duration_us=3.1, bottleneck="latency")
FW_CP  = _kernel("copyBuffer_impl", duration_us=1.0, bottleneck="latency")


# ---------------------------------------------------------------------------
# Tests: list_kernels — all kernels visible to the agent
# ---------------------------------------------------------------------------

class TestListKernels:
    """The agent sees ALL kernels and makes its own decisions."""

    def test_returns_all_kernels(self):
        """Every kernel — user and framework — is listed."""
        result = _profiler_result([TOPK_1, FW_EW, TOPK_2, FW_RED])
        kernels = list_kernels(result)
        assert len(kernels) == 4
        names = [k["name"] for k in kernels]
        assert "topk_stage1" in names
        assert "at::native::vectorized_elementwise_kernel" in names

    def test_preserves_order(self):
        """Kernel order matches the profiler output (no resorting)."""
        result = _profiler_result([FW_EW, TOPK_2, TOPK_1])
        kernels = list_kernels(result)
        assert [k["name"] for k in kernels] == [
            "at::native::vectorized_elementwise_kernel",
            "topk_stage2",
            "topk_stage1",
        ]

    def test_gpu_index(self):
        result = {
            "results": [
                {"device_id": "0", "gpu_info": {}, "kernels": [TOPK_1]},
                {"device_id": "1", "gpu_info": {}, "kernels": [ROPE]},
            ]
        }
        assert list_kernels(result, gpu_index=1)[0]["name"] == "fused_qkv_rope_triton"

    def test_bad_structure_raises(self):
        with pytest.raises(ValueError, match="missing 'results'"):
            list_kernels({"bad": "data"})


# ---------------------------------------------------------------------------
# Tests: build_baseline_metrics — explicit agent selection
# ---------------------------------------------------------------------------

class TestBuildBaselineMetrics:
    """Agent explicitly chooses kernels; the utility only formats."""

    # --- Selection by name ---

    def test_select_by_name_single(self):
        result = _profiler_result([TOPK_1, TOPK_2, FW_EW])
        baseline = build_baseline_metrics(result, kernel_names=["topk_stage1"])

        assert baseline["kernel_name"] == "topk_stage1"
        assert baseline["kernel_names"] == ["topk_stage1"]
        assert baseline["duration_us"] == pytest.approx(114.55)
        assert baseline["bottleneck"] == "latency"

    def test_select_by_name_multiple(self):
        result = _profiler_result([TOPK_1, TOPK_2, FW_EW])
        baseline = build_baseline_metrics(result, kernel_names=["topk_stage1", "topk_stage2"])

        assert set(baseline["kernel_names"]) == {"topk_stage1", "topk_stage2"}
        assert baseline["duration_us"] == pytest.approx(114.55 + 38.20, rel=1e-3)

    def test_select_framework_kernel_by_name(self):
        """Agent CAN choose framework kernels if the task calls for it."""
        result = _profiler_result([TOPK_1, FW_EW])
        baseline = build_baseline_metrics(
            result,
            kernel_names=["at::native::vectorized_elementwise_kernel"],
        )
        assert baseline["kernel_name"] == "at::native::vectorized_elementwise_kernel"
        assert baseline["duration_us"] == pytest.approx(5.2)

    def test_missing_name_raises(self):
        result = _profiler_result([TOPK_1])
        with pytest.raises(ValueError, match="not found.*nonexistent"):
            build_baseline_metrics(result, kernel_names=["nonexistent"])

    # --- Selection by index ---

    def test_select_by_index(self):
        result = _profiler_result([TOPK_1, TOPK_2, ROPE])
        baseline = build_baseline_metrics(result, kernel_indices=[2])

        assert baseline["kernel_name"] == "fused_qkv_rope_triton"
        assert baseline["duration_us"] == pytest.approx(220.0)

    def test_select_by_multiple_indices(self):
        result = _profiler_result([TOPK_1, TOPK_2, ROPE])
        baseline = build_baseline_metrics(result, kernel_indices=[0, 1])

        assert set(baseline["kernel_names"]) == {"topk_stage1", "topk_stage2"}
        assert baseline["duration_us"] == pytest.approx(114.55 + 38.20, rel=1e-3)

    def test_bad_index_raises(self):
        result = _profiler_result([TOPK_1])
        with pytest.raises(ValueError, match="out of range"):
            build_baseline_metrics(result, kernel_indices=[5])

    # --- include_all ---

    def test_include_all(self):
        result = _profiler_result([TOPK_1, TOPK_2])
        baseline = build_baseline_metrics(result, include_all=True)

        assert len(baseline["kernel_names"]) == 2
        assert baseline["duration_us"] == pytest.approx(114.55 + 38.20, rel=1e-3)

    # --- Validation: must choose exactly one mode ---

    def test_no_selection_raises(self):
        result = _profiler_result([TOPK_1])
        with pytest.raises(ValueError, match="Specify how to select"):
            build_baseline_metrics(result)

    def test_ambiguous_selection_raises(self):
        result = _profiler_result([TOPK_1])
        with pytest.raises(ValueError, match="only one of"):
            build_baseline_metrics(result, kernel_names=["topk_stage1"], include_all=True)

    def test_no_kernels_in_result_raises(self):
        result = {"results": [{"device_id": "0", "gpu_info": {}, "kernels": []}]}
        with pytest.raises(ValueError, match="No kernels found"):
            build_baseline_metrics(result, include_all=True)


# ---------------------------------------------------------------------------
# Tests: aggregation
# ---------------------------------------------------------------------------

class TestAggregateMetrics:

    def test_single_kernel_passthrough(self):
        agg = aggregate_metrics([TOPK_1])
        assert agg["duration_us"] == pytest.approx(114.55)
        assert agg["memory.hbm_bandwidth_utilization"] == pytest.approx(2.34)

    def test_duration_summed(self):
        agg = aggregate_metrics([TOPK_1, TOPK_2])
        assert agg["duration_us"] == pytest.approx(114.55 + 38.20, rel=1e-3)

    def test_bandwidth_weighted_average(self):
        agg = aggregate_metrics([TOPK_1, TOPK_2])
        total_dur = 114.55 + 38.20
        expected = (2.34 * 114.55 + 18.5 * 38.20) / total_dur
        assert agg["memory.hbm_bandwidth_utilization"] == pytest.approx(expected, rel=1e-3)

    def test_l2_hit_weighted(self):
        agg = aggregate_metrics([TOPK_1, TOPK_2])
        total_dur = 114.55 + 38.20
        expected = (49.72 * 114.55 + 72.0 * 38.20) / total_dur
        assert agg["memory.l2_hit_rate"] == pytest.approx(expected, rel=1e-3)

    def test_empty_returns_empty(self):
        assert aggregate_metrics([]) == {}


# ---------------------------------------------------------------------------
# Tests: output structure — what OpenEvolve needs
# ---------------------------------------------------------------------------

class TestOutputStructure:
    """The baseline dict must match what openevolve-mcp/run_openevolve.py reads."""

    def test_duration_us_at_top_level(self):
        """openevolve-mcp does: baseline_metrics.get('duration_us', 0)"""
        result = _profiler_result([TOPK_1])
        baseline = build_baseline_metrics(result, kernel_names=["topk_stage1"])

        assert "duration_us" in baseline
        assert isinstance(baseline["duration_us"], (int, float))
        assert baseline["duration_us"] > 0

    def test_has_all_required_fields(self):
        result = _profiler_result([TOPK_1])
        baseline = build_baseline_metrics(result, kernel_names=["topk_stage1"])

        for field in ["duration_us", "kernel_name", "kernel_names", "metrics", "bottleneck", "observations"]:
            assert field in baseline, f"Missing field: {field}"

    def test_metrics_has_hardware_counters(self):
        result = _profiler_result([TOPK_1])
        baseline = build_baseline_metrics(result, kernel_names=["topk_stage1"])

        for key in ["duration_us", "memory.hbm_bandwidth_utilization", "memory.l2_hit_rate", "memory.coalescing_efficiency"]:
            assert key in baseline["metrics"], f"Missing metric: {key}"

    def test_observations_merged_and_deduplicated(self):
        k1 = _kernel("k1", observations=["memory-bound kernel", "high L2 miss"])
        k2 = _kernel("k2", observations=["memory-bound kernel", "poor coalescing"])
        result = _profiler_result([k1, k2])
        baseline = build_baseline_metrics(result, include_all=True)

        assert "memory-bound kernel" in baseline["observations"]
        assert "high L2 miss" in baseline["observations"]
        assert "poor coalescing" in baseline["observations"]
        assert baseline["observations"].count("memory-bound kernel") == 1

    def test_multi_kernel_name_format(self):
        """Name shows dominant+count for multi-kernel baselines."""
        result = _profiler_result([TOPK_1, TOPK_2])
        baseline = build_baseline_metrics(result, include_all=True)

        # topk_stage1 is longer → dominant
        assert baseline["kernel_name"] == "topk_stage1+1"

    def test_bottleneck_from_dominant_kernel(self):
        result = _profiler_result([TOPK_1, TOPK_2])
        baseline = build_baseline_metrics(result, include_all=True)

        # TOPK_1 (114.55µs, "latency") dominates over TOPK_2 (38.20µs, "memory")
        assert baseline["bottleneck"] == "latency"


# ---------------------------------------------------------------------------
# Tests: OpenEvolve compatibility (simulate server.py parsing)
# ---------------------------------------------------------------------------

class TestOpenEvolveCompatibility:

    def _simulate_openevolve_parse(self, baseline_dict, best_duration_us=80.0):
        """Replicate openevolve-mcp/server.py result parsing."""
        baseline_latency = baseline_dict.get("duration_us", 0)
        speedup = baseline_latency / best_duration_us if best_duration_us > 0 else 1.0
        return {"baseline_latency_us": baseline_latency, "speedup": speedup}

    def test_single_kernel_speedup(self):
        result = _profiler_result([TOPK_1])
        baseline = build_baseline_metrics(result, kernel_names=["topk_stage1"])
        parsed = self._simulate_openevolve_parse(baseline, best_duration_us=80.0)

        assert parsed["baseline_latency_us"] == pytest.approx(114.55, rel=1e-3)
        assert parsed["speedup"] == pytest.approx(114.55 / 80.0, rel=1e-3)
        assert parsed["speedup"] > 1.0

    def test_multi_kernel_speedup(self):
        result = _profiler_result([TOPK_1, TOPK_2, FW_EW])
        baseline = build_baseline_metrics(result, kernel_names=["topk_stage1", "topk_stage2"])
        parsed = self._simulate_openevolve_parse(baseline, best_duration_us=100.0)

        total = 114.55 + 38.20
        assert parsed["baseline_latency_us"] == pytest.approx(total, rel=1e-3)
        assert parsed["speedup"] == pytest.approx(total / 100.0, rel=1e-3)

    def test_duration_us_never_zero(self):
        """Regression: old code produced duration_us=0 from wrong extraction."""
        result = _profiler_result([TOPK_1])
        baseline = build_baseline_metrics(result, kernel_names=["topk_stage1"])
        parsed = self._simulate_openevolve_parse(baseline)

        assert parsed["baseline_latency_us"] != 0

    def test_json_roundtrip(self):
        result = _profiler_result([TOPK_1, TOPK_2])
        baseline = build_baseline_metrics(result, include_all=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(baseline, f, indent=2)
            tmp_path = f.name
        try:
            with open(tmp_path) as f:
                loaded = json.load(f)

            assert loaded["duration_us"] == pytest.approx(baseline["duration_us"])
            assert loaded["kernel_name"] == baseline["kernel_name"]
            assert loaded["kernel_names"] == baseline["kernel_names"]
            assert loaded["metrics"]["duration_us"] == pytest.approx(baseline["metrics"]["duration_us"])

            parsed = self._simulate_openevolve_parse(loaded)
            assert parsed["baseline_latency_us"] > 0
        finally:
            os.unlink(tmp_path)

    def test_old_extraction_code_was_broken(self):
        """The old INSTRUCTIONS.md code (result.get('metrics', result)) fails."""
        result = _profiler_result([TOPK_1])

        old_baseline = result.get("metrics", result)
        assert old_baseline.get("duration_us", 0) == 0, (
            "Old code accidentally works — but it shouldn't since "
            "MetrixTool returns {'results': [...]}, not a flat metrics dict."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
