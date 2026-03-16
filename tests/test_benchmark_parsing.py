from pathlib import Path

from minisweagent.benchmark_parsing import compute_best_patch, parse_shape_latencies_ms


def test_parse_shape_latencies_ms_extracts_each_shape() -> None:
    output = "\n".join(
        [
            "Benchmark mode: 3 shapes, 10 iterations each",
            "  (32,4096): 0.0503 ms",
            "  (64,4096): 0.0525 ms",
            "  (256,8192): 0.0626 ms",
            "Geomean latency: 0.0548 ms",
            "GEAK_RESULT_LATENCY_MS=0.054772",
        ]
    )

    assert parse_shape_latencies_ms(output) == {
        "(32,4096)": 0.0503,
        "(64,4096)": 0.0525,
        "(256,8192)": 0.0626,
    }


def test_compute_best_patch_includes_per_shape_speedups(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "fused_rms_fp8"
    patch_dir = kernel_dir / "results" / "round_1" / "dispatch-path-check"
    patch_dir.mkdir(parents=True)

    (kernel_dir / "benchmark_baseline.txt").write_text(
        "\n".join(
            [
                "Benchmark mode: 2 shapes, 10 iterations each",
                "  (32,4096): 0.0500 ms",
                "  (64,4096): 0.0600 ms",
                "Geomean latency: 0.054772 ms",
                "GEAK_RESULT_LATENCY_MS=0.054772",
            ]
        )
    )
    (patch_dir / "patch_1.patch").write_text("diff --git a/kernel.py b/kernel.py\n+pass\n")
    (patch_dir / "patch_1_test.txt").write_text(
        "\n".join(
            [
                "Benchmark mode: 2 shapes, 10 iterations each",
                "  (32,4096): 0.0400 ms",
                "  (64,4096): 0.0600 ms",
                "Geomean latency: 0.048990 ms",
                "GEAK_RESULT_LATENCY_MS=0.048990",
            ]
        )
    )

    result = compute_best_patch(patch_dir)

    assert result is not None
    assert result["baseline_source"] == "benchmark_baseline.txt"
    assert result["best_patch_id"] == "patch_1"
    assert result["baseline_shape_latency_ms"] == {
        "(32,4096)": 0.05,
        "(64,4096)": 0.06,
    }
    assert result["candidate_shape_latency_ms"] == {
        "(32,4096)": 0.04,
        "(64,4096)": 0.06,
    }
    assert result["per_shape_speedups"] == {
        "(32,4096)": {
            "baseline_ms": 0.05,
            "candidate_ms": 0.04,
            "speedup": 1.25,
        },
        "(64,4096)": {
            "baseline_ms": 0.06,
            "candidate_ms": 0.06,
            "speedup": 1.0,
        },
    }
