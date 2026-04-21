from __future__ import annotations

from minisweagent.run.preprocess.baseline import build_baseline_metrics
from minisweagent.run.utils.selected_kernel_summary import (
    build_selected_kernel_summary,
    derive_primary_bottleneck,
)


def test_selected_kernel_summary_mode_threshold_boundary() -> None:
    single = build_selected_kernel_summary(
        [
            {"name": "mem_hot", "duration_us": 80.0, "bottleneck": "memory"},
            {"name": "compute_tail", "duration_us": 20.0, "bottleneck": "compute"},
        ]
    )
    mixed = build_selected_kernel_summary(
        [
            {"name": "mem_hot", "duration_us": 79.9, "bottleneck": "memory"},
            {"name": "compute_tail", "duration_us": 20.1, "bottleneck": "compute"},
        ]
    )

    assert single["mode"] == "single"
    assert single["primary_bottleneck"] == "memory"
    assert single["primary_bottleneck_pct_of_selected"] == 80.0

    assert mixed["mode"] == "mixed"
    assert mixed["primary_bottleneck"] == "memory"
    assert mixed["primary_bottleneck_pct_of_selected"] == 79.9


def test_selected_kernel_summary_is_invariant_under_kernel_reordering() -> None:
    kernels_a = [
        {"name": "slow_mem", "duration_us": 30.0, "bottleneck": "memory"},
        {"name": "fast_compute", "duration_us": 10.0, "bottleneck": "compute"},
        {"name": "mid_mem", "duration_us": 20.0, "bottleneck": "memory"},
    ]
    kernels_b = [kernels_a[1], kernels_a[2], kernels_a[0]]

    assert build_selected_kernel_summary(kernels_a) == build_selected_kernel_summary(kernels_b)


def test_build_baseline_metrics_includes_selected_kernel_summary_and_alias() -> None:
    profiler_result = {
        "results": [
            {
                "kernels": [
                    {
                        "name": "kernel_a",
                        "duration_us": 10.0,
                        "bottleneck": "memory-bound",
                        "observations": ["HBM limited"],
                        "metrics": {"duration_us": 10.0},
                    },
                    {
                        "name": "kernel_b",
                        "duration_us": 2.0,
                        "bottleneck": "compute",
                        "observations": ["high ALU"],
                        "metrics": {"duration_us": 2.0},
                    },
                ]
            }
        ]
    }

    baseline = build_baseline_metrics(
        profiler_result,
        kernel_names=["kernel_b", "kernel_a"],
        preserve_order=True,
    )

    assert baseline["primary_bottleneck"] == "memory"
    assert baseline["bottleneck"] == "memory"
    assert baseline["selected_kernel_summary"]["primary_bottleneck"] == "memory"
    assert baseline["selected_kernel_summary"]["mode"] == "single"
    assert [entry["bottleneck"] for entry in baseline["selected_kernel_summary"]["bottleneck_mix"]] == [
        "memory",
        "compute",
    ]


def test_derive_primary_bottleneck_falls_back_to_top_kernels() -> None:
    profiling_metrics = {
        "top_kernels": [
            {"name": "kernel_a", "duration_us": 60.0, "bottleneck": "compute"},
            {"name": "kernel_b", "duration_us": 40.0, "bottleneck": "memory"},
        ]
    }

    assert derive_primary_bottleneck(profiling_metrics) == "compute"
