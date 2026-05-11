"""Unit tests for the working memory module."""

from __future__ import annotations

import pytest

from minisweagent.memory.working_memory import (
    MAX_INSIGHTS,
    Insight,
    WorkingMemory,
    classify_change,
    extract_insight_from_tool_result,
    summarize_change,
)

# ---------------------------------------------------------------------------
# Insight dataclass
# ---------------------------------------------------------------------------

class TestInsight:
    def test_format(self):
        ins = Insight(step=3, tag="WIN", message="Speedup geomean: 1.20x")
        assert ins.format() == "[WIN] step 3: Speedup geomean: 1.20x"

    def test_default_timestamp(self):
        ins = Insight(step=0, tag="OK", message="test")
        assert ins.timestamp > 0


# ---------------------------------------------------------------------------
# WorkingMemory core methods
# ---------------------------------------------------------------------------

class TestUpdateSpeedup:
    def test_records_speedup_history(self):
        wm = WorkingMemory()
        wm.current_step = 5
        wm.update_speedup(1.2)
        assert wm.speedup_history == [(5, 1.2)]
        assert wm.best_speedup == 1.2
        assert wm.best_speedup_step == 5
        assert wm.steps_since_improvement == 0

    def test_tracks_best_speedup(self):
        wm = WorkingMemory()
        wm.current_step = 1
        wm.update_speedup(1.1)
        wm.current_step = 2
        wm.update_speedup(1.3)
        wm.current_step = 3
        wm.update_speedup(1.2)  # regression
        assert wm.best_speedup == 1.3
        assert wm.best_speedup_step == 2
        assert wm.steps_since_improvement == 1

    def test_increments_steps_since_improvement(self):
        wm = WorkingMemory()
        wm.update_speedup(1.0)
        wm.update_speedup(0.9)
        wm.update_speedup(0.8)
        assert wm.steps_since_improvement == 2


class TestUpdateLatency:
    def test_first_latency_becomes_baseline(self):
        wm = WorkingMemory()
        wm.update_latency(10.0)
        assert wm.baseline_latency_ms == 10.0
        assert wm.best_latency_ms == 10.0

    def test_computes_speedup_from_baseline(self):
        wm = WorkingMemory()
        wm.update_latency(10.0)  # baseline
        wm.update_latency(5.0)   # 2x speedup
        assert wm.best_speedup == pytest.approx(2.0)
        assert wm.best_latency_ms == 5.0

    def test_updates_best_latency(self):
        wm = WorkingMemory()
        wm.update_latency(10.0)
        wm.update_latency(8.0)
        wm.update_latency(9.0)  # worse than best
        assert wm.best_latency_ms == 8.0


class TestIsDiminishingReturns:
    def test_returns_false_with_few_measurements(self):
        wm = WorkingMemory()
        wm.latency_history = [1.0, 1.0]
        assert wm.is_diminishing_returns() is False

    def test_returns_true_when_flat(self):
        wm = WorkingMemory()
        wm.latency_history = [10.0, 10.05, 9.95]
        assert wm.is_diminishing_returns() is True

    def test_returns_false_when_improving(self):
        wm = WorkingMemory()
        wm.latency_history = [10.0, 8.0, 6.0]
        assert wm.is_diminishing_returns() is False


class TestAddInsight:
    def test_adds_and_caps_at_max(self):
        wm = WorkingMemory()
        for i in range(MAX_INSIGHTS + 5):
            wm.add_insight("OK", f"msg {i}")
        assert len(wm.insights) == MAX_INSIGHTS
        # oldest dropped, newest kept
        assert wm.insights[-1].message == f"msg {MAX_INSIGHTS + 4}"
        assert wm.insights[0].message == "msg 5"

    def test_truncates_long_messages(self):
        wm = WorkingMemory()
        wm.add_insight("OK", "x" * 200)
        assert len(wm.insights[0].message) == 120


# ---------------------------------------------------------------------------
# ingest_insight (centralized handler)
# ---------------------------------------------------------------------------

class TestIngestInsight:
    def test_sets_step_and_adds_to_buffer(self):
        wm = WorkingMemory()
        wm.current_step = 7
        insight = Insight(step=0, tag="OK", message="test msg")
        wm.ingest_insight(insight)
        assert insight.step == 7
        assert len(wm.insights) == 1
        assert wm.insights[0].tag == "OK"

    def test_extracts_bottleneck_type(self):
        wm = WorkingMemory()
        insight = Insight(step=0, tag="OK", message="Profiling: bottleneck=memory")
        wm.ingest_insight(insight)
        assert wm.bottleneck_type == "memory"

    def test_extracts_compute_bottleneck(self):
        wm = WorkingMemory()
        insight = Insight(step=0, tag="OK", message="Profiling: bottleneck=compute")
        wm.ingest_insight(insight)
        assert wm.bottleneck_type == "compute"

    def test_extracts_latency_bottleneck(self):
        wm = WorkingMemory()
        insight = Insight(step=0, tag="OK", message="Profiling: bottleneck=latency")
        wm.ingest_insight(insight)
        assert wm.bottleneck_type == "latency"

    def test_extracts_latency_and_updates(self):
        wm = WorkingMemory()
        wm.update_latency(10.0)  # set baseline
        insight = Insight(step=0, tag="OK", message="latency: 5.0000 ms after optimization")
        wm.ingest_insight(insight)
        assert wm.best_speedup == pytest.approx(2.0)

    def test_extracts_speedup_when_no_latency(self):
        wm = WorkingMemory()
        insight = Insight(step=0, tag="WIN", message="Speedup geomean: 1.50x improvement")
        wm.ingest_insight(insight)
        assert wm.best_speedup == pytest.approx(1.5)

    def test_latency_takes_precedence_over_speedup(self):
        wm = WorkingMemory()
        wm.update_latency(10.0)  # baseline
        # Message has both latency and a speedup-like pattern
        insight = Insight(step=0, tag="OK", message="latency: 8.0000 ms (was 1.25x)")
        wm.ingest_insight(insight)
        # Should use latency (10/8 = 1.25) not the 1.25x in text
        assert wm.best_speedup == pytest.approx(1.25)

    def test_no_bottleneck_update_without_keyword(self):
        wm = WorkingMemory()
        insight = Insight(step=0, tag="OK", message="All tests pass")
        wm.ingest_insight(insight)
        assert wm.bottleneck_type == ""


# ---------------------------------------------------------------------------
# Bottleneck hint keys match profiler output
# ---------------------------------------------------------------------------

class TestBottleneckHintKeys:
    """Ensure hint dict keys match the values produced by the profiler."""

    PROFILER_VALUES = ["memory", "compute", "latency", "balanced"]

    def test_all_profiler_values_have_hints(self):
        wm = WorkingMemory()
        wm.tuning_steps = 3  # hints only appear after 2 tuning steps
        for bn_type in self.PROFILER_VALUES:
            wm.bottleneck_type = bn_type
            text = wm.format_for_injection()
            assert f"Bottleneck: {bn_type}" in text, (
                f"No hint produced for bottleneck_type='{bn_type}'"
            )

    def test_hyphenated_keys_do_not_match(self):
        """Regression: old code used 'memory-bound' etc. which never matched."""
        wm = WorkingMemory()
        wm.tuning_steps = 3
        for bad_key in ["memory-bound", "compute-bound", "latency-bound"]:
            wm.bottleneck_type = bad_key
            text = wm.format_for_injection()
            assert "Bottleneck:" not in text or "balanced" in text, (
                f"Hyphenated key '{bad_key}' should NOT produce a hint"
            )


# ---------------------------------------------------------------------------
# extract_insight_from_tool_result
# ---------------------------------------------------------------------------

class TestExtractInsight:
    def test_returns_none_for_empty(self):
        assert extract_insight_from_tool_result("bash", "", 0) is None

    def test_profiling_bottleneck(self):
        output = '{"bottleneck": "memory", "compute_pct": 30}'
        ins = extract_insight_from_tool_result("profile", output, 0)
        assert ins is not None
        assert ins.tag == "OK"
        assert "bottleneck=memory" in ins.message

    def test_geak_latency(self):
        output = "GEAK_RESULT_LATENCY_MS=0.1234\nDone."
        ins = extract_insight_from_tool_result("bash", output, 0)
        assert ins is not None
        assert "0.1234" in ins.message

    def test_speedup_geomean_win(self):
        output = "Speedup (geomean): 1.35x over baseline"
        ins = extract_insight_from_tool_result("bash", output, 0)
        assert ins is not None
        assert ins.tag == "WIN"
        assert "1.35" in ins.message

    def test_speedup_geomean_fail(self):
        output = "Speedup (geomean): 0.40x over baseline"
        ins = extract_insight_from_tool_result("bash", output, 0)
        assert ins is not None
        assert ins.tag == "FAIL"

    def test_correctness_all_pass(self):
        output = "Running tests...\nAll pass\n"
        ins = extract_insight_from_tool_result("bash", output, 0)
        assert ins is not None
        assert ins.tag == "OK"
        assert "ALL PASS" in ins.message

    def test_correctness_failure(self):
        output = "FAIL: shape mismatch at index 5"
        ins = extract_insight_from_tool_result("bash", output, 1)
        assert ins is not None
        assert ins.tag == "FAIL"

    def test_commandment_ok(self):
        output = "COMMANDMENT.md validation: OK"
        ins = extract_insight_from_tool_result("bash", output, 0)
        assert ins is not None
        assert ins.tag == "OK"
        assert "COMMANDMENT" in ins.message

    def test_commandment_fail(self):
        output = "COMMANDMENT.md validation error: missing entry"
        ins = extract_insight_from_tool_result("bash", output, 1)
        assert ins is not None
        assert ins.tag == "FAIL"

    def test_geo_mean_latency(self):
        output = "Geo mean: 0.024120ms"
        ins = extract_insight_from_tool_result("bash", output, 0)
        assert ins is not None
        assert "0.0241" in ins.message

    def test_shape_specific_latencies(self):
        output = "hd=128: 0.0200ms\nhd=256: 0.0300ms"
        ins = extract_insight_from_tool_result("bash", output, 0)
        assert ins is not None
        assert "0.0300" in ins.message  # takes last

    def test_generic_error(self):
        output = "Something went wrong\nSegmentation fault"
        ins = extract_insight_from_tool_result("bash", output, 139)
        assert ins is not None
        assert ins.tag == "FAIL"
        assert "139" in ins.message

    def test_broken_pipe(self):
        output = "BrokenPipeError in subprocess"
        ins = extract_insight_from_tool_result("bash", output, 1)
        assert ins is not None
        assert ins.tag == "FAIL"
        assert "BrokenPipeError" in ins.message


# ---------------------------------------------------------------------------
# record_strategy
# ---------------------------------------------------------------------------

class TestRecordStrategy:
    def test_records_tried(self):
        wm = WorkingMemory()
        wm.record_strategy("vectorize", success=True)
        assert "vectorize" in wm.strategies_tried
        assert "vectorize" not in wm.strategies_failed

    def test_records_failure(self):
        wm = WorkingMemory()
        wm.record_strategy("fusion", success=False)
        assert "fusion" in wm.strategies_tried
        assert "fusion" in wm.strategies_failed

    def test_no_duplicates(self):
        wm = WorkingMemory()
        wm.record_strategy("tune", success=True)
        wm.record_strategy("tune", success=True)
        assert wm.strategies_tried.count("tune") == 1


# ---------------------------------------------------------------------------
# classify_change / summarize_change
# ---------------------------------------------------------------------------

class TestClassifyChange:
    def test_algorithmic(self):
        assert classify_change("def optimized_kernel(x):") == "algorithmic"

    def test_fusion(self):
        assert classify_change("fused_add_mul kernel") == "fusion"

    def test_tuning(self):
        assert classify_change("BLOCK_SIZE = 256") == "tuning"
        assert classify_change("num_warps = 4") == "tuning"
        assert classify_change("@triton.autotune") == "tuning"

    def test_wrapper_default(self):
        assert classify_change("import torch") == "wrapper"


class TestSummarizeChange:
    def test_returns_string(self):
        result = summarize_change("num_warps = 8")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# record_change_category
# ---------------------------------------------------------------------------

class TestRecordChangeCategory:
    def test_increments_consecutive(self):
        wm = WorkingMemory()
        wm.record_change_category("tuning")
        wm.record_change_category("tuning")
        assert wm.consecutive_same_category == 2
        assert wm.tuning_steps == 2

    def test_resets_on_different(self):
        wm = WorkingMemory()
        wm.record_change_category("tuning")
        wm.record_change_category("tuning")
        wm.record_change_category("algorithmic")
        assert wm.consecutive_same_category == 1
        assert wm.algo_steps == 1


# ---------------------------------------------------------------------------
# Budget / progress signals
# ---------------------------------------------------------------------------

class TestBudgetSignal:
    def test_no_signal_early(self):
        wm = WorkingMemory(max_steps=100, max_cost=1.0)
        wm.update_step(10, 0.1)
        assert wm.get_budget_signal() == ""

    def test_warn_at_70pct(self):
        wm = WorkingMemory(max_steps=100, max_cost=1.0)
        wm.update_step(75, 0.2)
        signal = wm.get_budget_signal()
        assert "BUDGET_WARN" in signal

    def test_critical_at_85pct(self):
        wm = WorkingMemory(max_steps=100, max_cost=1.0)
        wm.update_step(90, 0.2)
        signal = wm.get_budget_signal()
        assert "BUDGET_CRITICAL" in signal

    def test_force_at_95pct(self):
        wm = WorkingMemory(max_steps=100, max_cost=1.0)
        wm.update_step(96, 0.2)
        signal = wm.get_budget_signal()
        assert "BUDGET_FORCE" in signal


class TestProgressSignal:
    def test_stalled(self):
        wm = WorkingMemory()
        wm.best_speedup = 1.1
        wm.steps_since_improvement = 15
        assert "STALLED" in wm.get_progress_signal()

    def test_early_stop(self):
        wm = WorkingMemory()
        wm.best_speedup = 1.1
        wm.steps_since_improvement = 25
        assert "EARLY_STOP" in wm.get_progress_signal()


# ---------------------------------------------------------------------------
# format_for_injection
# ---------------------------------------------------------------------------

class TestFormatForInjection:
    def test_returns_string(self):
        wm = WorkingMemory()
        result = wm.format_for_injection()
        assert isinstance(result, str)

    def test_includes_best_speedup(self):
        wm = WorkingMemory()
        wm.baseline_latency_ms = 10.0
        wm.best_latency_ms = 6.67
        wm.best_speedup = 1.5
        wm.best_speedup_step = 3
        text = wm.format_for_injection()
        assert "1.5" in text or "6.67" in text

    def test_bottleneck_memory_priorities(self):
        wm = WorkingMemory()
        wm.bottleneck_type = "memory"
        text = wm.format_for_injection()
        assert "coalescing" in text.lower() or "memory" in text.lower()

    def test_bottleneck_compute_priorities(self):
        wm = WorkingMemory()
        wm.bottleneck_type = "compute"
        text = wm.format_for_injection()
        assert "algorithmic" in text.lower()
