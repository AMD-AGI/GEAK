#!/usr/bin/env python3
"""Pure-CPU unit tests for the deterministic roofline policy."""

import importlib.util
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_PATH = os.path.join(ROOT, "scripts", "roofline_policy.py")
SPEC = importlib.util.spec_from_file_location("roofline_policy", POLICY_PATH)
roofline_policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roofline_policy)


class TheoreticalBoundTests(unittest.TestCase):
    def test_ai_below_and_above_empirical_ridge(self):
        self.assertEqual(
            roofline_policy.classify_theoretical_bound(99.9, 100.0),
            "memory_side",
        )
        self.assertEqual(
            roofline_policy.classify_theoretical_bound(100.0, 100.0),
            "compute_side",
        )

    def test_missing_or_invalid_ridge_is_unknown(self):
        self.assertEqual(
            roofline_policy.classify_theoretical_bound(None, 100.0), "unknown"
        )
        self.assertEqual(
            roofline_policy.classify_theoretical_bound(10.0, 0.0), "unknown"
        )

    def test_low_utilization_does_not_override_theoretical_side(self):
        result = roofline_policy.build_classification(
            {
                "ai_hbm": 10.0,
                "compute_empirical_peak_gflops": 1000.0,
                "hbm_empirical_peak_gbps": 10.0,
                "compute_utilization_pct": 5.0,
                "hbm_utilization_pct": 4.0,
            }
        )
        self.assertEqual(result["theoretical_bound"], "memory_side")
        self.assertEqual(result["observed_limit"], "latency_occupancy")
        self.assertEqual(
            result["recommended_specialties"], ["algorithm", "compute"]
        )


class EfficiencyTests(unittest.TestCase):
    def test_memory_side_efficiency_uses_empirical_hbm_peak(self):
        result = roofline_policy.compute_roofline_efficiency(
            {
                "ai_hbm": 10.0,
                "performance_gflops": 20000.0,
                "compute_empirical_peak_gflops": 100000.0,
                "hbm_empirical_peak_gbps": 4000.0,
                "hbm_spec_peak_gbps": 5000.0,
            }
        )
        self.assertAlmostEqual(result["ai_ridge_empirical"], 25.0)
        self.assertAlmostEqual(result["roofline_empirical_ceiling_gflops"], 40000.0)
        self.assertAlmostEqual(result["roofline_efficiency_pct"], 50.0)
        self.assertAlmostEqual(result["headroom_ratio"], 2.0)
        self.assertAlmostEqual(result["roofline_spec_hbm_efficiency_pct"], 40.0)
        self.assertEqual(result["peak_basis"], "empirical")

    def test_compute_side_efficiency_is_capped_by_compute_peak(self):
        result = roofline_policy.compute_roofline_efficiency(
            {
                "ai_hbm": 100.0,
                "performance_gflops": 75000.0,
                "compute_empirical_peak_gflops": 100000.0,
                "hbm_empirical_peak_gbps": 4000.0,
            }
        )
        self.assertAlmostEqual(result["roofline_efficiency_pct"], 75.0)

    def test_spec_peak_never_replaces_missing_empirical_peak(self):
        result = roofline_policy.compute_roofline_efficiency(
            {
                "ai_hbm": 10.0,
                "performance_gflops": 20000.0,
                "compute_empirical_peak_gflops": 100000.0,
                "hbm_spec_peak_gbps": 5000.0,
            }
        )
        self.assertIsNone(result["roofline_efficiency_pct"])
        self.assertEqual(result["peak_basis"], "unavailable")
        self.assertAlmostEqual(result["roofline_spec_hbm_efficiency_pct"], 40.0)


class ObservedLimitTests(unittest.TestCase):
    def test_policy_version_is_numeric(self):
        self.assertIsInstance(roofline_policy.POLICY_VERSION, int)

    def test_priority_and_each_saturation_class(self):
        classify = roofline_policy.classify_observed_limit
        self.assertEqual(classify(90, 90, lds_util_pct=95), "lds")
        self.assertEqual(classify(80, 70), "balanced")
        self.assertEqual(classify(80, 20), "compute")
        self.assertEqual(classify(20, 80), "hbm")
        self.assertEqual(classify(20, 20, l2_util_pct=80), "cache")
        self.assertEqual(classify(20, 20), "latency_occupancy")

    def test_overhead_has_absolute_priority(self):
        self.assertEqual(
            roofline_policy.classify_observed_limit(
                99, 99, lds_util_pct=99, overhead_bound=True
            ),
            "overhead",
        )

    def test_all_missing_is_unknown(self):
        self.assertEqual(
            roofline_policy.classify_observed_limit(None, None), "unknown"
        )

    def test_explicit_no_fp_work(self):
        result = roofline_policy.build_classification(
            {"no_fp_work": True, "hbm_utilization_pct": 20.0}
        )
        self.assertEqual(result["observed_limit"], "no_fp_work")

    def test_recommendations_use_only_supported_specialties(self):
        for observed in (
            "hbm",
            "compute",
            "cache",
            "lds",
            "balanced",
            "latency_occupancy",
            "overhead",
            "no_fp_work",
            "unknown",
        ):
            result = roofline_policy.recommend_optimization(
                "memory_side", observed, 50.0
            )
            self.assertTrue(
                set(result["recommended_specialties"]).issubset(
                    set(roofline_policy.SPECIALTIES)
                )
            )
            self.assertTrue(result["recommended_levers"])

    def test_lds_routes_to_memory_and_algorithm(self):
        result = roofline_policy.recommend_optimization(
            "memory_side", "lds", 50.0
        )
        self.assertEqual(
            result["recommended_specialties"], ["memory", "algorithm"]
        )


class ComparisonTests(unittest.TestCase):
    def _case(self, performance):
        metrics = {
            "performance_gflops": performance,
            "compute_actual_gflops": performance,
            "compute_empirical_peak_gflops": 1000.0,
            "hbm_empirical_peak_gbps": 10.0,
            "ai_hbm": 200.0,
            "compute_utilization_pct": performance / 10.0,
            "hbm_utilization_pct": 20.0,
            "peak_basis": "empirical",
            "compute_metric": "MFMA FLOPs (F16)",
        }
        return {
            "case_id": "m1024",
            "shape": [1024, 1024],
            "dtypes": ["fp16"],
            "kernel": "gemm_kernel",
            "peak_basis": "empirical",
            "compute_metric": "MFMA FLOPs (F16)",
            "metrics": metrics,
        }

    def test_before_after_delta(self):
        result = roofline_policy.compare_cases(self._case(500.0), self._case(750.0))
        self.assertTrue(result["compatible"])
        self.assertAlmostEqual(result["performance_ratio"], 1.5)
        self.assertTrue(result["improved"])
        self.assertAlmostEqual(result["deltas"]["compute_utilization_pct"], 25.0)

    def test_identity_and_peak_basis_are_strict(self):
        before = self._case(500.0)
        after = self._case(750.0)
        after["shape"] = [2048, 1024]
        with self.assertRaises(ValueError):
            roofline_policy.compare_cases(before, after)

        after = self._case(750.0)
        after["metrics"]["hbm_empirical_peak_gbps"] = 11.0
        with self.assertRaises(ValueError):
            roofline_policy.compare_cases(before, after)

    def test_empirical_peak_allows_normal_microbenchmark_noise(self):
        before = self._case(500.0)
        after = self._case(750.0)
        after["metrics"]["compute_empirical_peak_gflops"] = 1030.0
        after["metrics"]["hbm_empirical_peak_gbps"] = 10.3
        result = roofline_policy.compare_cases(before, after)
        self.assertTrue(result["compatible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
