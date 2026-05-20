"""Unit tests for :mod:`minisweagent.run.section_builders`."""

from __future__ import annotations

from minisweagent.run.section_builders import (
    build_benchmark_body,
    build_correctness_body,
    build_full_benchmark_body,
    build_profile_body,
    strip_mode_flags,
    warmup_block,
)


# ---------------------------------------------------------------------------
# strip_mode_flags
# ---------------------------------------------------------------------------


class TestStripModeFlags:
    def test_strips_correctness(self):
        assert strip_mode_flags("cmd --correctness") == "cmd"

    def test_strips_benchmark(self):
        assert strip_mode_flags("cmd --benchmark") == "cmd"

    def test_strips_full_benchmark_without_corruption(self):
        result = strip_mode_flags("cmd --full-benchmark")
        assert result == "cmd"
        assert "--full-" not in result

    def test_strips_profile(self):
        assert strip_mode_flags("cmd --profile") == "cmd"

    def test_no_flags_unchanged(self):
        assert strip_mode_flags("python3 harness.py") == "python3 harness.py"

    def test_multiple_flags(self):
        result = strip_mode_flags("cmd --correctness --benchmark")
        assert result == "cmd"

    def test_preserves_other_flags(self):
        result = strip_mode_flags("cmd --correctness --verbose")
        assert result == "cmd --verbose"

    def test_full_benchmark_with_extra_args(self):
        result = strip_mode_flags("run.sh /path/harness.py --full-benchmark extra")
        assert result == "run.sh /path/harness.py extra"


# ---------------------------------------------------------------------------
# warmup_block
# ---------------------------------------------------------------------------


class TestWarmupBlock:
    def test_zero_runs_empty(self):
        assert warmup_block("cmd", 0) == ""

    def test_negative_runs_empty(self):
        assert warmup_block("cmd", -1) == ""

    def test_one_run_single_command(self):
        assert warmup_block("cmd", 1) == "cmd"

    def test_multiple_runs_for_loop(self):
        result = warmup_block("cmd", 3)
        assert "for _i in $(seq 1 3)" in result
        assert "cmd" in result
        assert result == "for _i in $(seq 1 3); do cmd; done"


# ---------------------------------------------------------------------------
# build_correctness_body
# ---------------------------------------------------------------------------


class TestBuildCorrectness:
    def test_appends_correctness_flag(self):
        result = build_correctness_body("${GEAK_WORK_DIR}/run.sh /path/harness.py")
        assert result == "${GEAK_WORK_DIR}/run.sh /path/harness.py --correctness"

    def test_run_harness_shape(self):
        result = build_correctness_body("${GEAK_WORK_DIR}/run_harness.sh")
        assert result == "${GEAK_WORK_DIR}/run_harness.sh --correctness"


# ---------------------------------------------------------------------------
# build_profile_body
# ---------------------------------------------------------------------------


class TestBuildProfileBody:
    def test_default_warmup_and_replays(self):
        result = build_profile_body("cmd")
        assert "--replays 3" in result
        assert "kernel-profile" in result
        assert "for _i in $(seq 1 2)" in result

    def test_custom_warmup_and_replays(self):
        result = build_profile_body("cmd", warmup_runs=5, profile_replays=10)
        assert "--replays 10" in result
        assert "for _i in $(seq 1 5)" in result

    def test_zero_warmup_no_warmup_line(self):
        result = build_profile_body("cmd", warmup_runs=0)
        assert result.startswith("kernel-profile")
        assert "> /dev/null" not in result.split("\n")[0]

    def test_contains_profile_flag(self):
        result = build_profile_body("cmd")
        assert "--profile" in result

    def test_kernel_profile_wrapping(self):
        result = build_profile_body("${GEAK_WORK_DIR}/run.sh /harness.py")
        assert 'kernel-profile "${GEAK_WORK_DIR}/run.sh /harness.py --profile"' in result

    def test_output_path(self):
        result = build_profile_body("cmd")
        assert "${GEAK_WORK_DIR}/profile.json" in result

    def test_gpu_device_variable(self):
        result = build_profile_body("cmd")
        assert "${GEAK_GPU_DEVICE}" in result

    def test_run_harness_shape(self):
        result = build_profile_body("${GEAK_WORK_DIR}/run_harness.sh")
        assert 'kernel-profile "${GEAK_WORK_DIR}/run_harness.sh --profile"' in result


# ---------------------------------------------------------------------------
# build_benchmark_body / build_full_benchmark_body
# ---------------------------------------------------------------------------


class TestBuildBenchmarkBodies:
    def test_benchmark_uses_full_benchmark_flag(self):
        result = build_benchmark_body("cmd")
        assert "--full-benchmark" in result

    def test_benchmark_includes_extra_args(self):
        result = build_benchmark_body("cmd")
        assert "${GEAK_BENCHMARK_EXTRA_ARGS:-}" in result

    def test_full_benchmark_uses_full_benchmark_flag(self):
        result = build_full_benchmark_body("cmd")
        assert "--full-benchmark" in result

    def test_full_benchmark_includes_extra_args(self):
        result = build_full_benchmark_body("cmd")
        assert "${GEAK_BENCHMARK_EXTRA_ARGS:-}" in result

    def test_both_produce_identical_output(self):
        base = "${GEAK_WORK_DIR}/run.sh /path/harness.py"
        assert build_benchmark_body(base) == build_full_benchmark_body(base)


# ---------------------------------------------------------------------------
# Builder base command shapes
# ---------------------------------------------------------------------------


class TestBuilderBaseCommandShapes:
    """Verify all builders produce valid output for both base_cmd shapes."""

    _SIMPLE_CMD = "${GEAK_WORK_DIR}/run.sh /abs/path/test_harness.py"
    _INNER_CMD = "${GEAK_WORK_DIR}/run_harness.sh"

    def test_correctness_simple(self):
        result = build_correctness_body(self._SIMPLE_CMD)
        assert "/test_harness.py --correctness" in result

    def test_correctness_inner(self):
        result = build_correctness_body(self._INNER_CMD)
        assert "run_harness.sh --correctness" in result

    def test_profile_simple(self):
        result = build_profile_body(self._SIMPLE_CMD)
        assert "/test_harness.py --profile" in result

    def test_profile_inner(self):
        result = build_profile_body(self._INNER_CMD)
        assert "run_harness.sh --profile" in result

    def test_benchmark_simple(self):
        result = build_benchmark_body(self._SIMPLE_CMD)
        assert "/test_harness.py --full-benchmark" in result

    def test_benchmark_inner(self):
        result = build_benchmark_body(self._INNER_CMD)
        assert "run_harness.sh --full-benchmark" in result
