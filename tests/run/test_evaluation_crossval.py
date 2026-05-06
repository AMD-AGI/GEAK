"""Tests for profiler cross-validation in evaluation.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from minisweagent.run.postprocess.evaluation import (
    _compute_verified_speedup,
    _cross_validate_with_profiler,
    _get_speedup_cap,
    _load_baseline_metrics,
)


@pytest.fixture()
def pp_dir_with_metrics(tmp_path):
    """Create a pp_dir with baseline_metrics.json."""
    def _make(profiler_us: float, benchmark_us: float):
        metrics = {
            "duration_us": benchmark_us,
            "profiler_duration_us": profiler_us,
            "benchmark_duration_us": benchmark_us,
            "benchmark_profiler_ratio": round(benchmark_us / profiler_us, 2) if profiler_us > 0 else None,
        }
        path = tmp_path / "baseline_metrics.json"
        path.write_text(json.dumps(metrics))
        return tmp_path
    return _make


class TestLoadBaselineMetrics:
    def test_returns_none_for_none_dir(self):
        assert _load_baseline_metrics(None) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _load_baseline_metrics(tmp_path) is None

    def test_loads_valid_metrics(self, tmp_path):
        (tmp_path / "baseline_metrics.json").write_text('{"duration_us": 100.0}')
        result = _load_baseline_metrics(tmp_path)
        assert result == {"duration_us": 100.0}

    def test_returns_none_for_invalid_json(self, tmp_path):
        (tmp_path / "baseline_metrics.json").write_text("not json")
        assert _load_baseline_metrics(tmp_path) is None


class TestGetSpeedupCap:
    def test_returns_none_when_no_pp_dir(self):
        assert _get_speedup_cap(None) is None

    def test_returns_none_when_ratio_low(self, pp_dir_with_metrics):
        pp = pp_dir_with_metrics(profiler_us=35.0, benchmark_us=38.0)
        assert _get_speedup_cap(pp) is None

    def test_returns_ratio_when_high(self, pp_dir_with_metrics):
        pp = pp_dir_with_metrics(profiler_us=100.0, benchmark_us=2000.0)
        cap = _get_speedup_cap(pp)
        assert cap == pytest.approx(20.0)

    def test_returns_none_when_profiler_missing(self, tmp_path):
        (tmp_path / "baseline_metrics.json").write_text(
            json.dumps({"duration_us": 2000.0, "benchmark_duration_us": 2000.0})
        )
        assert _get_speedup_cap(tmp_path) is None

    def test_threshold_boundary(self, pp_dir_with_metrics):
        pp = pp_dir_with_metrics(profiler_us=100.0, benchmark_us=1000.0)
        assert _get_speedup_cap(pp) is None  # ratio=10.0, not > 10.0

        pp2 = pp_dir_with_metrics(profiler_us=100.0, benchmark_us=1001.0)
        cap = _get_speedup_cap(pp2)
        assert cap == pytest.approx(10.01)


class TestComputeVerifiedSpeedup:
    @staticmethod
    def _make_round_eval():
        return {"full_benchmark": {}}

    def test_normal_speedup_no_cap(self, pp_dir_with_metrics):
        pp = pp_dir_with_metrics(profiler_us=35.0, benchmark_us=38.0)
        round_eval = self._make_round_eval()
        _compute_verified_speedup(
            "GEAK_RESULT_LATENCY_MS=20.0",
            "GEAK_RESULT_LATENCY_MS=38.0",
            round_eval, "full_benchmark", pp,
        )
        assert round_eval["full_benchmark"]["verified_speedup"] == pytest.approx(1.9, abs=0.01)
        assert "speedup_capped" not in round_eval["full_benchmark"]

    def test_caps_when_exceeds_overhead_ratio(self, pp_dir_with_metrics):
        pp = pp_dir_with_metrics(profiler_us=30.0, benchmark_us=2353.0)
        round_eval = self._make_round_eval()
        _compute_verified_speedup(
            "GEAK_RESULT_LATENCY_MS=31.8",
            "GEAK_RESULT_LATENCY_MS=2353.0",
            round_eval, "full_benchmark", pp,
        )
        # Raw speedup = 2353/31.8 = 74x, overhead ratio = 2353/30 = 78.4
        # 74x < 78.4x so cap doesn't trigger (raw < cap)
        assert round_eval["full_benchmark"]["verified_speedup"] == pytest.approx(74.0, abs=0.5)

    def test_caps_when_speedup_exceeds_cap(self, pp_dir_with_metrics):
        pp = pp_dir_with_metrics(profiler_us=100.0, benchmark_us=2000.0)
        round_eval = self._make_round_eval()
        _compute_verified_speedup(
            "GEAK_RESULT_LATENCY_MS=50.0",
            "GEAK_RESULT_LATENCY_MS=2000.0",
            round_eval, "full_benchmark", pp,
        )
        # Raw speedup = 2000/50 = 40x, overhead ratio = 2000/100 = 20x
        # 40x > 20x so cap triggers
        fb = round_eval["full_benchmark"]
        assert fb["speedup_capped"] is True
        assert fb["benchmark_speedup_raw"] == pytest.approx(40.0, abs=0.1)
        assert fb["verified_speedup"] == pytest.approx(20.0, abs=0.1)

    def test_no_cap_without_pp_dir(self):
        round_eval = self._make_round_eval()
        _compute_verified_speedup(
            "GEAK_RESULT_LATENCY_MS=1.0",
            "GEAK_RESULT_LATENCY_MS=100.0",
            round_eval, "full_benchmark", None,
        )
        assert round_eval["full_benchmark"]["verified_speedup"] == pytest.approx(100.0)
        assert "speedup_capped" not in round_eval["full_benchmark"]

    def test_no_latency_returns_early(self, tmp_path):
        round_eval = self._make_round_eval()
        _compute_verified_speedup(
            "no latency here", "also nothing",
            round_eval, "full_benchmark", tmp_path,
        )
        assert round_eval["full_benchmark"].get("verified_speedup") is None


class TestCrossValidateWithProfiler:
    def test_replaces_capped_speedup(self):
        baseline = {"profiler_duration_us": 100.0, "duration_us": 2000.0}
        optimized = {"duration_us": 90.0}
        round_eval = {
            "full_benchmark": {
                "verified_speedup": 20.0,
                "speedup_capped": True,
            }
        }
        _cross_validate_with_profiler(baseline, optimized, round_eval)
        fb = round_eval["full_benchmark"]
        assert fb["profiler_speedup"] == pytest.approx(100.0 / 90.0, abs=0.01)
        assert fb["verified_speedup"] == pytest.approx(100.0 / 90.0, abs=0.01)

    def test_skips_uncapped_speedup(self):
        baseline = {"profiler_duration_us": 100.0}
        optimized = {"duration_us": 50.0}
        round_eval = {
            "full_benchmark": {
                "verified_speedup": 1.5,
            }
        }
        _cross_validate_with_profiler(baseline, optimized, round_eval)
        assert round_eval["full_benchmark"]["verified_speedup"] == 1.5

    def test_handles_missing_profiler_data(self):
        baseline = {}
        optimized = {"duration_us": 50.0}
        round_eval = {
            "full_benchmark": {
                "verified_speedup": 20.0,
                "speedup_capped": True,
            }
        }
        _cross_validate_with_profiler(baseline, optimized, round_eval)
        # No profiler_duration_us or duration_us in baseline → should not change
        assert round_eval["full_benchmark"]["verified_speedup"] == 20.0

    def test_handles_zero_optimized_duration(self):
        baseline = {"profiler_duration_us": 100.0}
        optimized = {"duration_us": 0.0}
        round_eval = {
            "full_benchmark": {
                "verified_speedup": 20.0,
                "speedup_capped": True,
            }
        }
        _cross_validate_with_profiler(baseline, optimized, round_eval)
        assert round_eval["full_benchmark"]["verified_speedup"] == 20.0
