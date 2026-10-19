# Standalone adaptation, 2026-10-12: test import paths are package-local.
"""Observed UT weighting works across model features, phase labels and workload kinds."""
import copy
import importlib.util
from pathlib import Path

import unittest
import math

_case = unittest.TestCase()

spec = importlib.util.spec_from_file_location(
    "agentx_harness", (Path(__file__).resolve().parents[1] / "scripts") / "harness_lib.py"
)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def experiment():
    # A small-M win, a slower KV-tail case at the SAME M, an unchanged tier,
    # and prefill. Largest-M/analytic weighting would hide the common-case win.
    rows = [
        {"sig": "verify-p50", "regime": "decode", "m": 12, "baseline_ms": 2., "optimized_ms": 1.},
        {"sig": "verify-p90", "regime": "decode", "m": 12, "baseline_ms": 4., "optimized_ms": 5.},
        {"sig": "verify-tier32", "regime": "decode", "m": 18, "baseline_ms": 3., "optimized_ms": 3.},
        {"sig": "hot-prefill", "regime": "prefill", "m": 1993, "baseline_ms": 10., "optimized_ms": 11.},
    ]
    meta = {"workload": {
        "kind": "agentx_trace_replay",
        "serving_weight_model": {
            "source": "agentx_log_fold", "evidence_path": "profile/agentx_workload.json",
            "analytic_calls": {"decode": 10000, "prefill": 10000},  # must be ignored
        },
        "cases": [{"sig": r["sig"], "regime": r["regime"], "m": r["m"], "effective_calls": n}
                  for r, n in zip(rows, [80.5, 10, 9.5, 2])],
    }}
    return rows, meta


def test_log_fold_keeps_kv_cases_and_unchanged_tiers(experiment):
    rows, meta = experiment
    before = copy.deepcopy(meta)
    result = h.serving_weighted_speedup(rows, meta)
    assert math.isclose(result["weighted"], (161 + 40 + 28.5 + 20) / (80.5 + 50 + 28.5 + 22), rel_tol=1e-6, abs_tol=1e-12)
    assert result["included"] == 4
    assert [r["calls"] for r in result["per_case"]] == [80.5, 10, 9.5, 2]
    assert h.served_buckets(meta) == [("decode", 12, 90.5), ("decode", 18, 9.5), ("prefill", 1993, 2)]
    assert meta == before


def test_unserved_phase_and_zero_mass_tail(experiment):
    rows, meta = experiment
    meta["served_regimes"] = ["decode"]
    tail = {"sig": "extrapolated", "regime": "decode", "m": 42, "effective_calls": 0}
    meta["workload"]["cases"].append(tail)
    result = h.serving_weighted_speedup(rows, meta)
    assert math.isclose(result["weighted"], 229.5 / 159, rel_tol=1e-6, abs_tol=1e-12)
    assert result["dropped_unserved"] == ["hot-prefill"]
    assert len(h.served_buckets(meta)) == 2


def test_invalid_call_mass_fails_closed(experiment, bad_mass):
    rows, meta = experiment
    meta["workload"]["cases"][0]["effective_calls"] = bad_mass
    with _case.assertRaises(h.HarnessIncompleteError):
        h.serving_weighted_speedup(rows, meta)


def test_incomplete_extraction_cannot_report_weighted_win(experiment, problem):
    rows, meta = experiment
    workload = meta["workload"]
    if problem == "missing_model":
        workload["serving_weight_model"].pop("source")
    elif problem == "missing_evidence":
        workload["serving_weight_model"].pop("evidence_path")
    elif problem == "missing_cases":
        workload.pop("cases")
    elif problem == "duplicate_model":
        workload["cases"].append(dict(workload["cases"][0]))
    elif problem == "missing_timed":
        rows.pop(0)
    elif problem == "duplicate_timed":
        rows.append(dict(rows[0]))
    elif problem == "unknown_sig":
        rows[0]["sig"] = "unobserved"
    elif problem == "wrong_regime":
        rows[0]["regime"] = "prefill"
    elif problem == "wrong_m":
        rows[0]["m"] = 10
    elif problem == "nan_latency":
        rows[0]["optimized_ms"] = float("nan")
    elif problem == "missing_latency":
        rows[0].pop("optimized_ms")
    with _case.assertRaises(h.HarnessIncompleteError):
        h.serving_weighted_speedup(rows, meta)


def test_empty_measurement_rejected(experiment):
    _, meta = experiment
    with _case.assertRaisesRegex(h.HarnessIncompleteError, "omitted"):
        h.serving_weighted_speedup([], meta)


def test_fixed_load_retains_analytic_weighting():
    rows = [{"sig": "m1", "regime": "decode", "m": 1, "baseline_ms": 2., "optimized_ms": 1.},
            {"sig": "m64", "regime": "decode", "m": 64, "baseline_ms": 4., "optimized_ms": 1.}]
    meta = {"workload": {"serving_weight_model": {"analytic_calls": {"decode": 100}}}}
    result = h.serving_weighted_speedup(rows, meta)
    assert math.isclose(result["weighted"], 402 / 101, rel_tol=1e-6, abs_tol=1e-12)
    assert [r["calls"] for r in result["per_case"]] == [1, 100]


def test_general_weighting_without_model_or_phase_assumptions(kind, phases, ms):
    rows, cases = [], []
    for i, (phase, m, b, o, calls) in enumerate(zip(phases, ms, [1., 4., 2.], [.5, 5., 2.], [100, 20, 5])):
        case = {"sig": f"case-{i}", "regime": phase, "effective_calls": calls}
        if m is not None:
            case["m"] = m
        cases.append(case)
        rows.append({**case, "baseline_ms": b, "optimized_ms": o})
    meta = {"workload": {"kind": "agentx_trace_replay", "scenario": kind, "weighting": "observed", "cases": cases,
                         "serving_weight_model": {"source": "observed_workload",
                                                  "evidence_path": "profile/kernel_workload.json"}}}
    result = h.serving_weighted_speedup(rows, meta)
    assert math.isclose(result["weighted"], 190 / 160, rel_tol=1e-6, abs_tol=1e-12)
    assert result["included"] == 3  # unchanged work still costs time
    assert [row["calls"] for row in result["per_case"]] == [100, 20, 5]


def test_general_observed_mode_cannot_fall_back_to_analytic_model(experiment):
    rows, meta = experiment
    meta["workload"].update(kind="agentx_trace_replay", weighting="observed")
    meta["workload"]["serving_weight_model"] = {"analytic_calls": {"decode": 100}}
    with _case.assertRaisesRegex(h.HarnessIncompleteError, "no analytic fallback"):
        h.serving_weighted_speedup(rows, meta)


def test_custom_phase_buckets_preserve_phase_at_same_m():
    meta = {"workload": {"kind": "agentx_trace_replay", "weighting": "observed", "serving_weight_model": {
        "source": "observed_workload", "evidence_path": "evidence.json"}, "cases": [
        {"sig": "d", "regime": "draft", "m": 4, "effective_calls": 8},
        {"sig": "v", "regime": "verify", "m": 4, "effective_calls": 2},
    ]}}
    assert h.served_buckets(meta) == [("draft", 4, 8), ("verify", 4, 2)]


def test_non_agentx_retains_analytic_weighting_even_with_observed_fields(kind):
    rows = [{"sig": "m1", "regime": "decode", "m": 1, "baseline_ms": 2., "optimized_ms": 1.},
            {"sig": "m64", "regime": "decode", "m": 64, "baseline_ms": 4., "optimized_ms": 1.}]
    legacy = {"workload": {"serving_weight_model": {"analytic_calls": {"decode": 100}}}}
    meta = copy.deepcopy(legacy)
    meta["workload"].update(kind=kind, weighting="observed", cases=[])
    meta["workload"]["serving_weight_model"].update(source="observed_workload")
    assert h.serving_weighted_speedup(rows, meta) == h.serving_weighted_speedup(rows, legacy)
    assert h.served_buckets(meta) == h.served_buckets(legacy)


class ObservedWeightsTest(unittest.TestCase):
    pass

def _test_log_fold_keeps_kv_cases_and_unchanged_tiers(self):
    test_log_fold_keeps_kv_cases_and_unchanged_tiers(experiment())
ObservedWeightsTest.test_log_fold_keeps_kv_cases_and_unchanged_tiers = _test_log_fold_keeps_kv_cases_and_unchanged_tiers

def _test_unserved_phase_and_zero_mass_tail(self):
    test_unserved_phase_and_zero_mass_tail(experiment())
ObservedWeightsTest.test_unserved_phase_and_zero_mass_tail = _test_unserved_phase_and_zero_mass_tail

def _test_invalid_call_mass_fails_closed(self):
    for values in [-1, float("nan"), float("inf"), None, True, "bad"]:
        values = (values,)
        with self.subTest(values=values):
            test_invalid_call_mass_fails_closed(experiment(), *values)
ObservedWeightsTest.test_invalid_call_mass_fails_closed = _test_invalid_call_mass_fails_closed

def _test_incomplete_extraction_cannot_report_weighted_win(self):
    for values in ["missing_model", "missing_evidence", "missing_cases", "duplicate_model",
                                      "missing_timed", "duplicate_timed", "unknown_sig", "wrong_regime",
                                      "wrong_m", "nan_latency", "missing_latency"]:
        values = (values,)
        with self.subTest(values=values):
            test_incomplete_extraction_cannot_report_weighted_win(experiment(), *values)
ObservedWeightsTest.test_incomplete_extraction_cannot_report_weighted_win = _test_incomplete_extraction_cannot_report_weighted_win

def _test_empty_measurement_rejected(self):
    test_empty_measurement_rejected(experiment())
ObservedWeightsTest.test_empty_measurement_rejected = _test_empty_measurement_rejected

def _test_fixed_load_retains_analytic_weighting(self):
    test_fixed_load_retains_analytic_weighting()
ObservedWeightsTest.test_fixed_load_retains_analytic_weighting = _test_fixed_load_retains_analytic_weighting

def _test_general_weighting_without_model_or_phase_assumptions(self):
    for values in [
    ("dynamic_dense", ["prefill", "decode", "decode"], [64, 5, 5]),
    ("speculative_moe", ["draft", "verify", "extend"], [2, 10, 2]),
    ("agentx_trace_replay", ["prefill", "decode", "decode"], [31, 3, 7]),
    ("embedding", ["encode", "pool", "pool"], [None, None, None]),
    ("offline_vision", ["forward", "forward", "forward"], [None, None, None]),
    ("hybrid_model", ["attention", "state_update", "projection"], [4, None, 4]),
    ("custom_workload", ["load", "compute", "reduce"], [None, None, None]),
]:
        with self.subTest(values=values):
            test_general_weighting_without_model_or_phase_assumptions(*values)
ObservedWeightsTest.test_general_weighting_without_model_or_phase_assumptions = _test_general_weighting_without_model_or_phase_assumptions

def _test_general_observed_mode_cannot_fall_back_to_analytic_model(self):
    test_general_observed_mode_cannot_fall_back_to_analytic_model(experiment())
ObservedWeightsTest.test_general_observed_mode_cannot_fall_back_to_analytic_model = _test_general_observed_mode_cannot_fall_back_to_analytic_model

def _test_custom_phase_buckets_preserve_phase_at_same_m(self):
    test_custom_phase_buckets_preserve_phase_at_same_m()
ObservedWeightsTest.test_custom_phase_buckets_preserve_phase_at_same_m = _test_custom_phase_buckets_preserve_phase_at_same_m

def _test_non_agentx_retains_analytic_weighting_even_with_observed_fields(self):
    for values in ["synthetic_isl_osl", "dynamic_serving", "custom_workload"]:
        values = (values,)
        with self.subTest(values=values):
            test_non_agentx_retains_analytic_weighting_even_with_observed_fields(*values)
ObservedWeightsTest.test_non_agentx_retains_analytic_weighting_even_with_observed_fields = _test_non_agentx_retains_analytic_weighting_even_with_observed_fields

if __name__ == "__main__":
    unittest.main()
