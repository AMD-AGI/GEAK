"""GPU integration tests for the unified profiler MCP.

These tests require a real AMD GPU. They are skipped automatically
when no GPU is detected (via rocm-smi).

Override the test kernel command with:
    PROFILER_TEST_COMMAND="python3 /path/to/kernel.py --profile" pytest
    PROFILER_TEST_WORKDIR="/path/to/kernel/dir" pytest
"""

from conftest import requires_gpu
from profiler_mcp.server import profile_kernel


# Helper to call the wrapped MCP tool function
def _call(**kwargs):
    return profile_kernel.fn(**kwargs)


VALID_BOTTLENECKS = {
    "memory",
    "compute",
    "latency",
    "lds",
    "balanced",
    "memory-bound",
    "compute-bound",
    "latency-bound",
    "lds-bound",
}


@requires_gpu
class TestMetrixRealKernel:
    def test_success_and_kernels(self, test_command):
        result = _call(command=test_command, backend="metrix", quick=True)

        assert result["success"] is True, f"Profiling failed: {result.get('error')}"
        assert result["backend"] == "metrix"
        assert len(result.get("results", [])) >= 1, "Expected at least 1 GPU result"

        device_result = result["results"][0]
        kernels = device_result.get("kernels", [])
        assert len(kernels) >= 1, "Expected at least 1 kernel"

        # Check that the rope kernel is present
        kernel_names = [k["name"] for k in kernels]
        rope_kernels = [n for n in kernel_names if "rope" in n.lower()]
        assert len(rope_kernels) >= 1, f"Expected rope kernel, got: {kernel_names}"

    def test_kernel_fields(self, test_command):
        result = _call(command=test_command, backend="metrix", quick=True)
        assert result["success"] is True

        kernel = result["results"][0]["kernels"][0]
        assert "name" in kernel
        assert "duration_us" in kernel
        assert kernel["duration_us"] > 0
        assert "bottleneck" in kernel
        assert kernel["bottleneck"] in VALID_BOTTLENECKS
        assert "metrics" in kernel


@requires_gpu
class TestRocprofRealKernel:
    def test_roofline_analysis(self, test_command, test_workdir):
        result = _call(
            command=test_command,
            backend="rocprof-compute",
            workdir=test_workdir,
            profiling_type="roofline",
        )

        assert result["success"] is True, f"Profiling failed: {result.get('error')}"
        assert result["backend"] == "rocprof-compute"
        assert result["profiling_type"] == "roofline"

        analysis = result.get("analysis", "")
        assert len(analysis) > 100, f"Analysis too short ({len(analysis)} chars)"

        # rocprof-compute roofline should mention system info and kernels
        assert "gpu" in analysis.lower() or "GPU" in analysis


@requires_gpu
class TestMetrixQuickVsFull:
    def test_quick_has_fewer_metrics(self, test_command):
        quick_result = _call(command=test_command, backend="metrix", quick=True)
        full_result = _call(command=test_command, backend="metrix", quick=False)

        assert quick_result["success"] is True
        assert full_result["success"] is True

        # Compare metric counts on the first kernel
        quick_kernel = quick_result["results"][0]["kernels"][0]
        full_kernel = full_result["results"][0]["kernels"][0]

        quick_metrics = len(quick_kernel.get("metrics", {}))
        full_metrics = len(full_kernel.get("metrics", {}))

        assert full_metrics >= quick_metrics, (
            f"Full profile ({full_metrics} metrics) should have >= quick profile ({quick_metrics} metrics)"
        )
