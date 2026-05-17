"""Tests for workload-aware prompt guidance in pipeline helpers.

Note: this file previously also contained
``test_create_validated_harness_materializes_harness_into_log_dir``, but
``create_validated_harness`` (and its private helpers
``_preferred_harness_path`` / ``_materialize_validated_harness``)
were removed from :mod:`~minisweagent.run.pipeline_helpers` as dead
duplicates of the canonical chain in
:mod:`~minisweagent.run.preprocess.harness_utils`. The harness-utils
copy is what ``HarnessPhase._layer6_unit_test_agent`` actually calls,
and is covered by ``tests/run/test_preprocess_unit.py`` and the new
``tests/run/test_preprocess_phases.py::TestUserTaskPlumbing``.
"""

from __future__ import annotations

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

    assert "Optimization Guidance (bottleneck: latency-bound)" in text
    assert "Workload Guidance (HIP search / pointer-chasing)" in text
    assert "branchless search logic" in text
    assert "Deprioritize generic vectorization" in text
