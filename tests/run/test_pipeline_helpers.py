"""Tests for workload-aware prompt guidance in pipeline helpers."""

from minisweagent.run.pipeline_helpers import _bottleneck_guidance


def test_bottleneck_guidance_adds_search_specific_hip_hints() -> None:
    metrics = {
        "kernel_name": "rocprim::detail::binary_search lower_bound",
        "bottleneck": "latency",
        "metrics": {
            "memory.hbm_bandwidth_utilization": 0.3,
            "memory.l2_hit_rate": 70.6,
        },
        "top_kernels": [
            {
                "name": "transform_kernel<binary_search<lower_bound>>",
                "bottleneck": "latency",
            }
        ],
    }

    text = "\n".join(_bottleneck_guidance("latency", metrics))

    assert "Workload Guidance (HIP search / pointer-chasing)" in text
    assert "branchless search logic" in text
    assert "Deprioritize generic vectorization" in text
