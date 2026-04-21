from __future__ import annotations

from minisweagent.run.postprocess.evaluation import _build_per_kernel_deltas


def test_per_kernel_deltas_marks_not_selected_when_kernel_still_exists() -> None:
    baseline = {
        "top_kernels": [
            {"name": "kernel_a", "duration_us": 10.0, "bottleneck": "memory", "metrics": {}},
            {"name": "kernel_b", "duration_us": 5.0, "bottleneck": "compute", "metrics": {}},
        ]
    }
    optimized = {
        "top_kernels": [
            {"name": "kernel_a", "duration_us": 8.0, "bottleneck": "memory", "metrics": {}},
        ]
    }
    optimized_all = [
        {"name": "kernel_a", "duration_us": 8.0, "bottleneck": "memory", "metrics": {}},
        {"name": "kernel_b", "duration_us": 4.5, "bottleneck": "compute", "metrics": {}},
    ]

    deltas = _build_per_kernel_deltas(baseline, optimized, optimized_all)

    by_name = {entry["name"]: entry for entry in deltas}
    assert by_name["kernel_b"]["status"] == "not_selected_on_optimized"
    assert by_name["kernel_b"]["optimized_duration_us"] == 4.5


def test_per_kernel_deltas_marks_missing_by_name_when_kernel_disappears() -> None:
    baseline = {
        "top_kernels": [
            {"name": "kernel_a", "duration_us": 10.0, "bottleneck": "memory", "metrics": {}},
        ]
    }
    optimized = {"top_kernels": []}
    optimized_all = [{"name": "kernel_c", "duration_us": 2.0, "bottleneck": "latency", "metrics": {}}]

    deltas = _build_per_kernel_deltas(baseline, optimized, optimized_all)

    assert deltas == [
        {
            "name": "kernel_a",
            "status": "missing_by_name",
            "baseline_duration_us": 10.0,
        }
    ]
