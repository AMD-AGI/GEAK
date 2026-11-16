#!/usr/bin/env python3
"""Multi-config observed weighting: each config (InferenceX search-space point) is normalized
to its share before the cases are pooled into one UT metric.

Run:  python3 -I tests/test_multi_config_weights.py   (stdlib only, no torch, no GPU)
"""
import copy
import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "harness_lib_mc", Path(__file__).resolve().parents[1] / "harness_lib.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def _meta(cases, **model):
    m = {"source": "observed_workload", "evidence_path": "evidence/observed_workload.json",
         "config_weighting": "time"}
    m.update(model)
    return {"workload": {"kind": "agentx_trace_replay", "serving_weight_model": m, "cases": cases}}


def _harmonic(pairs):
    """Share-weighted harmonic mean of (share, speedup) pairs."""
    return sum(s for s, _ in pairs) / sum(s / x for s, x in pairs)


# c1: one decode case, 2x faster.  c64: a 1000x larger launch count, 1.25x slower overall.
ROWS = [
    {"sig": "c1/de_1", "regime": "decode", "m": 1, "baseline_ms": 1.0, "optimized_ms": 0.5},
    {"sig": "c64/de_64", "regime": "decode", "m": 64, "baseline_ms": 2.0, "optimized_ms": 2.5},
    {"sig": "c64/pf_8k", "regime": "prefill", "m": 8192, "baseline_ms": 10.0, "optimized_ms": 12.5},
]
CASES = [
    {"sig": "c1/de_1", "regime": "decode", "m": 1, "effective_calls": 100, "config_id": "c1"},
    {"sig": "c64/de_64", "regime": "decode", "m": 64, "effective_calls": 100000, "config_id": "c64"},
    {"sig": "c64/pf_8k", "regime": "prefill", "m": 8192, "effective_calls": 2000, "config_id": "c64"},
]


def _calls_pooled(rows, cases, alpha):
    """sum_M w(M) t(M) with w = sum_c alpha_c freq_c(M): the 2026-11-02 merge's weighting."""
    tot = {}
    for c in cases:
        tot[c["config_id"]] = tot.get(c["config_id"], 0) + c["effective_calls"]
    calls = {c["sig"]: c for c in cases}
    W = D = 0.0
    for r in rows:
        c = calls[r["sig"]]
        w = alpha[c["config_id"]] * c["effective_calls"] / tot[c["config_id"]] * r["baseline_ms"]
        W += w
        D += w * r["optimized_ms"] / r["baseline_ms"]
    return W / D


class CallsWeighting(unittest.TestCase):
    """Default: each config's call distribution normalized, then averaged (freq-weighted)."""

    def _m(self, cases, **kw):
        m = _meta(cases, **kw)
        if "config_weighting" not in kw:
            del m["workload"]["serving_weight_model"]["config_weighting"]
        return m

    def test_default_is_calls_and_matches_the_merge_formula(self):
        r = h.serving_weighted_speedup(copy.deepcopy(ROWS), self._m(copy.deepcopy(CASES)))
        self.assertEqual(r["config_weighting"], "calls")
        self.assertAlmostEqual(r["weighted"], _calls_pooled(ROWS, CASES, {"c1": .5, "c64": .5}))
        # c64's launches cost more per call, so it counts for more than in "time" mode
        t = h.serving_weighted_speedup(copy.deepcopy(ROWS), _meta(copy.deepcopy(CASES)))
        self.assertLess(r["weighted"], t["weighted"])
        self.assertAlmostEqual(r["per_config"]["c64"]["weighted"], 0.8)

    def test_call_count_scale_does_not_matter(self):
        big = copy.deepcopy(CASES)
        for c in big:
            if c["config_id"] == "c64":
                c["effective_calls"] *= 1000
        a = h.serving_weighted_speedup(copy.deepcopy(ROWS), self._m(copy.deepcopy(CASES)))
        b = h.serving_weighted_speedup(copy.deepcopy(ROWS), self._m(big))
        self.assertAlmostEqual(a["weighted"], b["weighted"])

    def test_untimed_mass_stays_in_the_denominator(self):
        # a zero-latency-change case that is not timed still dilutes nothing: every served
        # positive-mass case must be timed (fail closed), as in the single-config path
        with self.assertRaises(h.HarnessIncompleteError):
            h.serving_weighted_speedup(copy.deepcopy(ROWS[:2]), self._m(copy.deepcopy(CASES)))

    def test_unknown_mode_fails_closed(self):
        with self.assertRaises(h.HarnessIncompleteError):
            h.serving_weighted_speedup(copy.deepcopy(ROWS), self._m(copy.deepcopy(CASES),
                                                                    config_weighting="steps"))


class MultiConfigWeights(unittest.TestCase):
    def test_configs_are_equal_weight_regardless_of_call_mass(self):
        meta = _meta(copy.deepcopy(CASES))
        before = copy.deepcopy(meta)
        r = h.serving_weighted_speedup(copy.deepcopy(ROWS), meta)
        self.assertEqual(meta, before)
        self.assertAlmostEqual(r["per_config"]["c1"]["weighted"], 2.0)
        self.assertAlmostEqual(r["per_config"]["c64"]["weighted"], 0.8)
        self.assertAlmostEqual(r["per_config"]["c1"]["share"], 0.5)
        # equal shares -> harmonic mean of 2.0 and 0.8, not the c64-dominated pool (~0.80)
        self.assertAlmostEqual(r["weighted"], _harmonic([(0.5, 2.0), (0.5, 0.8)]))
        self.assertAlmostEqual(sum(c["weight"] for c in r["per_case"] if c["included"]), 1.0)
        raw = sum(c["weight_raw"] for c in r["per_case"]) / sum(
            c["weight_raw"] / c["speedup"] for c in r["per_case"])
        self.assertLess(raw, 0.81)   # what the unnormalized pool would have reported

    def test_single_config_matches_untagged_path(self):
        rows = [dict(r) for r in ROWS if r["sig"].startswith("c64")]
        tagged = _meta([dict(c) for c in CASES if c["config_id"] == "c64"])
        untagged = _meta([{k: v for k, v in c.items() if k != "config_id"}
                          for c in CASES if c["config_id"] == "c64"])
        a = h.serving_weighted_speedup(copy.deepcopy(rows), tagged)
        b = h.serving_weighted_speedup(copy.deepcopy(rows), untagged)
        self.assertAlmostEqual(a["weighted"], b["weighted"])
        self.assertNotIn("per_config", b)

    def test_explicit_config_weights(self):
        meta = _meta(copy.deepcopy(CASES), config_weights={"c1": 1, "c64": 3})
        r = h.serving_weighted_speedup(copy.deepcopy(ROWS), meta)
        self.assertAlmostEqual(r["weighted"], _harmonic([(0.25, 2.0), (0.75, 0.8)]))

    def test_config_where_task_does_not_run_is_left_out(self):
        cases = copy.deepcopy(CASES) + [
            {"sig": "ep1/none", "regime": "decode", "m": 8, "effective_calls": 0, "config_id": "ep1"}]
        r = h.serving_weighted_speedup(copy.deepcopy(ROWS), _meta(cases))
        self.assertNotIn("ep1", r["per_config"])
        self.assertAlmostEqual(r["weighted"], _harmonic([(0.5, 2.0), (0.5, 0.8)]))

    def test_mixed_tagging_fails_closed(self):
        cases = copy.deepcopy(CASES)
        del cases[0]["config_id"]
        with self.assertRaises(h.HarnessIncompleteError):
            h.serving_weighted_speedup(copy.deepcopy(ROWS), _meta(cases))

    def test_config_weights_must_name_every_config(self):
        meta = _meta(copy.deepcopy(CASES), config_weights={"c1": 1})
        with self.assertRaises(h.HarnessIncompleteError):
            h.serving_weighted_speedup(copy.deepcopy(ROWS), meta)
        meta = _meta(copy.deepcopy(CASES), config_weights={"c1": 1, "c64": math.nan})
        with self.assertRaises(h.HarnessIncompleteError):
            h.serving_weighted_speedup(copy.deepcopy(ROWS), meta)

    def test_missing_positive_mass_case_still_fails_closed(self):
        with self.assertRaises(h.HarnessIncompleteError):
            h.serving_weighted_speedup(copy.deepcopy(ROWS[:2]), _meta(copy.deepcopy(CASES)))


if __name__ == "__main__":
    unittest.main()
