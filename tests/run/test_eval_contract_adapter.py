"""Tests for eval_contract_adapter: kernel-level timing extraction & harness generation."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from minisweagent.run.preprocess.eval_contract_adapter import (
    _extract_kernel_latency_ms,
    materialize_shell_contract_harness,
    resolve_shell_eval_commands,
)


# ---------------------------------------------------------------------------
# _extract_kernel_latency_ms — Layer 1: pass-through
# ---------------------------------------------------------------------------


class TestPassThrough:
    def test_existing_marker_returns_kernel(self):
        text = "Perf: 0.05 ms (s0)\nGEAK_RESULT_LATENCY_MS=6.6\n"
        ms, src = _extract_kernel_latency_ms(text)
        assert ms is None
        assert src == "kernel"

    def test_marker_with_speedup(self):
        text = "GEAK_RESULT_LATENCY_MS=123.456\nGEAK_RESULT_SPEEDUP=1.0\n"
        ms, src = _extract_kernel_latency_ms(text)
        assert ms is None
        assert src == "kernel"


# ---------------------------------------------------------------------------
# _extract_kernel_latency_ms — Layer 2: summary extraction
# ---------------------------------------------------------------------------


class TestSummaryExtraction:
    def test_total_kernel_time_ms(self):
        ms, src = _extract_kernel_latency_ms("TOTAL_KERNEL_TIME_MS: 6.6\n")
        assert ms == pytest.approx(6.6)
        assert src == "subprocess_parsed"

    def test_benchmark_latency_ms(self):
        ms, src = _extract_kernel_latency_ms("BENCHMARK_LATENCY_MS: 445.12\n")
        assert ms == pytest.approx(445.12)
        assert src == "subprocess_parsed"

    def test_benchmark_metric(self):
        ms, src = _extract_kernel_latency_ms("BENCHMARK_METRIC: median_latency_ms=123.456\n")
        assert ms == pytest.approx(123.456)
        assert src == "subprocess_parsed"

    def test_median_latency_ms_standalone(self):
        ms, src = _extract_kernel_latency_ms("median_latency_ms: 0.98\n")
        assert ms == pytest.approx(0.98)
        assert src == "subprocess_parsed"

    def test_geomean_ms(self):
        ms, src = _extract_kernel_latency_ms("Geomean (ms): 0.054772\n")
        assert ms == pytest.approx(0.054772)
        assert src == "subprocess_parsed"

    def test_median_latency_keyword(self):
        ms, src = _extract_kernel_latency_ms("Median latency: 0.052 ms\n")
        assert ms == pytest.approx(0.052)
        assert src == "subprocess_parsed"

    def test_total_median_time(self):
        ms, src = _extract_kernel_latency_ms("total median time: 1.234 ms\n")
        assert ms == pytest.approx(1.234)
        assert src == "subprocess_parsed"

    def test_google_benchmark_format(self):
        ms, src = _extract_kernel_latency_ms("BM_MatMul/1024 1000 445.12 ms\n")
        assert ms == pytest.approx(445.12)
        assert src == "subprocess_parsed"

    def test_universal_keyword_fallback(self):
        ms, src = _extract_kernel_latency_ms("Overall latency: 0.052ms\n")
        assert ms == pytest.approx(0.052)
        assert src == "subprocess_parsed"

    def test_summary_wins_over_per_line(self):
        text = "Perf: 0.03 ms (s0)\nPerf: 0.02 ms (s1)\nMedian latency: 0.052 ms\n"
        ms, src = _extract_kernel_latency_ms(text)
        assert ms == pytest.approx(0.052)
        assert src == "subprocess_parsed"


# ---------------------------------------------------------------------------
# _extract_kernel_latency_ms — Layer 3: per-line timing aggregation
# ---------------------------------------------------------------------------


class TestPerLineAggregation:
    def test_perf_ms_lines(self):
        text = "Perf: 0.0325 ms (shape_0)\nPerf: 0.1777 ms (shape_1)\n"
        ms, src = _extract_kernel_latency_ms(text)
        assert ms == pytest.approx(0.0325 + 0.1777)
        assert src == "subprocess_parsed"

    def test_performance_ms_lines(self):
        text = "Performance: 12.3456 ms (shape_0)\n"
        ms, src = _extract_kernel_latency_ms(text)
        assert ms == pytest.approx(12.3456)
        assert src == "subprocess_parsed"

    def test_microseconds_conversion(self):
        text = "Perf: 32.5 us/launch\n"
        ms, src = _extract_kernel_latency_ms(text)
        assert ms == pytest.approx(0.0325)
        assert src == "subprocess_parsed"

    def test_mixed_units(self):
        text = "Perf: 0.0325 ms (s0)\nPerf: 32.5 us (s1)\n"
        ms, src = _extract_kernel_latency_ms(text)
        assert ms == pytest.approx(0.065)
        assert src == "subprocess_parsed"

    def test_real_knn_output(self):
        text = (
            "/data/knn/src/knn_cuda.hip -> /data/knn/src/knn_hip.hip [skipped, already hipified]\n"
            "/data/knn/src/knn.cpp -> /data/knn/src/knn_hip.cpp [skipped, already hipified]\n"
            "Total number of unsupported CUDA function calls: 0\n"
            "\n"
            "Total number of replaced kernel launches: 1\n"
            "ninja: no work to do.\n"
            "Perf: 0.0325 ms (shape_0_standard)\n"
            "Perf: 0.0425 ms (shape_0_transposed)\n"
            "Perf: 0.0420 ms (shape_0_self_query)\n"
            "Perf: 0.1777 ms (shape_1_standard)\n"
            "Perf: 0.1910 ms (shape_1_transposed)\n"
            "Perf: 0.2207 ms (shape_1_self_query)\n"
            "Perf: 0.4469 ms (shape_2_standard)\n"
            "Perf: 0.4630 ms (shape_2_transposed)\n"
            "Perf: 0.5190 ms (shape_2_self_query)\n"
            "Perf: 1.2712 ms (shape_3_standard)\n"
            "Perf: 1.2765 ms (shape_3_transposed)\n"
            "Perf: 1.3132 ms (shape_3_self_query)\n"
            "Perf: 0.1862 ms (shape_4_standard)\n"
            "Perf: 0.1936 ms (shape_4_transposed)\n"
            "Perf: 0.2007 ms (shape_4_self_query)\n"
        )
        ms, src = _extract_kernel_latency_ms(text)
        assert ms == pytest.approx(6.5767, abs=0.001)
        assert src == "subprocess_parsed"

    def test_skips_hipify_lines(self):
        text = (
            "knn.hip -> knn_hip.hip [ok]\n"
            "Perf: 0.05 ms (s0)\n"
        )
        ms, src = _extract_kernel_latency_ms(text)
        assert ms == pytest.approx(0.05)
        assert src == "subprocess_parsed"

    def test_skips_geak_marker_lines(self):
        # Shouldn't match GEAK_ lines as per-line timing
        text = "Perf: 0.05 ms (s0)\n"
        ms, src = _extract_kernel_latency_ms(text)
        assert ms == pytest.approx(0.05)

    def test_skips_compilation_lines(self):
        text = (
            "[1/3] /opt/rocm/bin/hipcc -o knn.so\n"
            "[2/3] c++ -o knn_hip.o\n"
            "[3/3] c++ knn_hip.cuda.o knn_hip.o -shared -o knn.so\n"
            "Perf: 0.05 ms (s0)\n"
        )
        ms, src = _extract_kernel_latency_ms(text)
        assert ms == pytest.approx(0.05)
        assert src == "subprocess_parsed"

    def test_ansi_escape_codes_stripped(self):
        text = (
            "\x1b[92mSuccessfully preprocessed all matching files.\x1b[0m\n"
            "Total number of unsupported CUDA function calls: 0\n"
            "Total number of replaced kernel launches: 1\n"
            "Perf: 0.1642 ms (shape_0)\n"
            "Perf: 0.1096 ms (shape_1)\n"
        )
        ms, src = _extract_kernel_latency_ms(text)
        # Must NOT include phantom 92ms from \x1b[92m
        assert ms == pytest.approx(0.1642 + 0.1096)
        assert src == "subprocess_parsed"

    def test_ansi_bold_escape_ignored(self):
        text = "\x1b[1;31mError in build\x1b[0m\nPerf: 5.0 ms (s0)\n"
        ms, src = _extract_kernel_latency_ms(text)
        assert ms == pytest.approx(5.0)
        assert src == "subprocess_parsed"


# ---------------------------------------------------------------------------
# _extract_kernel_latency_ms — Fallback: wall-clock
# ---------------------------------------------------------------------------


class TestWallClockFallback:
    def test_no_timing_at_all(self):
        ms, src = _extract_kernel_latency_ms("Test completed successfully.\nAll checks passed.\n")
        assert ms is None
        assert src == "wall_clock"

    def test_empty_output(self):
        ms, src = _extract_kernel_latency_ms("")
        assert ms is None
        assert src == "wall_clock"


# ---------------------------------------------------------------------------
# resolve_shell_eval_commands
# ---------------------------------------------------------------------------


class TestResolveShellEvalCommands:
    def test_structured_commands(self):
        ctx = SimpleNamespace(
            correctness_command="python3 task_runner.py correctness",
            performance_command="python3 task_runner.py performance",
            eval_command=None,
        )
        cc, pc = resolve_shell_eval_commands(ctx)
        assert cc == "python3 task_runner.py correctness"
        assert pc == "python3 task_runner.py performance"

    def test_eval_command_split(self):
        ctx = SimpleNamespace(
            correctness_command=None,
            performance_command=None,
            eval_command="make check && make bench",
        )
        cc, pc = resolve_shell_eval_commands(ctx)
        assert cc == "make check"
        assert pc == "make bench"

    def test_no_commands(self):
        ctx = SimpleNamespace(
            correctness_command=None,
            performance_command=None,
            eval_command=None,
        )
        cc, pc = resolve_shell_eval_commands(ctx)
        assert cc is None
        assert pc is None


# ---------------------------------------------------------------------------
# materialize_shell_contract_harness
# ---------------------------------------------------------------------------


class TestMaterializeHarness:
    def test_generates_valid_python(self):
        with tempfile.TemporaryDirectory() as td:
            out = materialize_shell_contract_harness(
                output_dir=Path(td),
                repo_root="/tmp/repo",
                correctness_shell="echo ok",
                performance_shell="echo 'Perf: 1.0 ms (test)'",
            )
            source = out.read_text()
            ast.parse(source)

    def test_contains_required_components(self):
        with tempfile.TemporaryDirectory() as td:
            out = materialize_shell_contract_harness(
                output_dir=Path(td),
                repo_root="/tmp/repo",
                correctness_shell="echo ok",
                performance_shell="echo perf",
            )
            source = out.read_text()
            assert "def _run_shell" in source
            assert "def _extract_kernel_latency_ms" in source
            assert "def main" in source
            assert "import re" in source
            assert "subprocess.PIPE" in source
            assert "GEAK_RESULT_TIMING_SOURCE" in source
            assert "GEAK_RESULT_LATENCY_MS" in source

    def test_embeds_commands(self):
        with tempfile.TemporaryDirectory() as td:
            out = materialize_shell_contract_harness(
                output_dir=Path(td),
                repo_root="/tmp/my_repo",
                correctness_shell="python3 check.py",
                performance_shell="python3 bench.py",
            )
            source = out.read_text()
            assert "python3 check.py" in source
            assert "python3 bench.py" in source
            assert "/tmp/my_repo" in source

    def test_is_executable(self):
        with tempfile.TemporaryDirectory() as td:
            out = materialize_shell_contract_harness(
                output_dir=Path(td),
                repo_root="/tmp/repo",
                correctness_shell="echo ok",
                performance_shell="echo perf",
            )
            import os
            assert os.access(out, os.X_OK)

    def test_captures_stdout_not_call(self):
        with tempfile.TemporaryDirectory() as td:
            out = materialize_shell_contract_harness(
                output_dir=Path(td),
                repo_root="/tmp/repo",
                correctness_shell="echo ok",
                performance_shell="echo perf",
            )
            source = out.read_text()
            assert "subprocess.call" not in source
            assert "subprocess.run" in source
            assert "subprocess.PIPE" in source
