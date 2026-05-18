import math
from pathlib import Path

from minisweagent.run.postprocess.benchmark_parsing import (
    compute_best_patch,
    extract_latency_ms,
    parse_labeled_latencies_ms,
    parse_shape_latencies_ms,
    _labeled_latencies_geomean_ms,
)


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


# ── Labeled latency parsing ────────────────────────────────────────


class TestParseLabeledLatencies:
    def test_perf_format(self) -> None:
        output = (
            "Perf: 0.0122 ms (shape_0_forward)\n"
            "Perf: 0.1170 ms (shape_0_backward)\n"
        )
        result = parse_labeled_latencies_ms(output)
        assert result == {"shape_0_forward": 0.0122, "shape_0_backward": 0.117}

    def test_various_labels(self) -> None:
        output = (
            "Latency: 0.5 ms (kernel_a)\n"
            "Time: 1.2 ms (kernel_b)\n"
        )
        result = parse_labeled_latencies_ms(output)
        assert result == {"kernel_a": 0.5, "kernel_b": 1.2}

    def test_no_parenthesized_label(self) -> None:
        output = "Duration: 0.42 ms\n"
        result = parse_labeled_latencies_ms(output)
        assert result == {"entry_0": 0.42}

    def test_empty_output(self) -> None:
        assert parse_labeled_latencies_ms("") == {}

    def test_non_matching_lines_ignored(self) -> None:
        output = (
            "Status: OK\n"
            "Some random text 0.5 ms\n"
            "Perf: 0.01 ms (real)\n"
        )
        result = parse_labeled_latencies_ms(output)
        assert result == {"real": 0.01}


class TestLabeledGeomean:
    def test_single_entry(self) -> None:
        assert _labeled_latencies_geomean_ms("Perf: 0.5 ms (x)\n") == 0.5

    def test_multiple_entries(self) -> None:
        output = "Perf: 1.0 ms (a)\nPerf: 4.0 ms (b)\n"
        result = _labeled_latencies_geomean_ms(output)
        assert result is not None
        assert abs(result - 2.0) < 0.001

    def test_no_entries(self) -> None:
        assert _labeled_latencies_geomean_ms("no data here") is None


class TestExtractLatencyMsWithLabeled:
    def test_hip_perf_format(self) -> None:
        output = (
            "Perf: 0.0122 ms (shape_0_forward)\n"
            "Perf: 0.1170 ms (shape_0_backward)\n"
            "Perf: 0.0129 ms (shape_1_forward)\n"
        )
        result = extract_latency_ms(output)
        assert result is not None
        expected = math.exp(
            (math.log(0.0122) + math.log(0.117) + math.log(0.0129)) / 3
        )
        assert abs(result - expected) < 1e-6

    def test_geak_marker_takes_priority(self) -> None:
        output = "Perf: 0.5 ms (test)\nGEAK_RESULT_LATENCY_MS=0.123\n"
        assert extract_latency_ms(output) == 0.123

    def test_median_takes_priority_over_labeled(self) -> None:
        output = "Perf: 0.5 ms (x)\nMedian latency: 0.3 ms\n"
        assert extract_latency_ms(output) == 0.3


class TestParseShapeLatenciesFallback:
    def test_tuple_format_preferred(self) -> None:
        output = (
            "(32,4096): 0.05 ms\n"
            "Perf: 0.99 ms (ignored)\n"
        )
        result = parse_shape_latencies_ms(output)
        assert result == {"(32,4096)": 0.05}

    def test_labeled_fallback(self) -> None:
        output = "Perf: 0.01 ms (shape_0)\nPerf: 0.02 ms (shape_1)\n"
        result = parse_shape_latencies_ms(output)
        assert result == {"shape_0": 0.01, "shape_1": 0.02}
