# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the fixed stopping bound against analytic and adversarial cases."""

import json
import math
import sys
import unittest
from decimal import Decimal, localcontext

from interface.native_cost_controls.quality_stop_stats import (
    CRITICAL_T,
    LOOK_ALPHA,
    assess_scores,
)


class QualityStopStatisticsTests(unittest.TestCase):
    def check_json(self, result):
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertIs(result["execution_authority_established"], False)

    def test_critical_value_matches_exact_df2_tail_conservatively(self):
        with localcontext() as context:
            context.prec = 70
            exact = (Decimal(1682) / Decimal(59)).sqrt()
            actual = Decimal.from_float(CRITICAL_T)
            self.assertGreaterEqual(actual, exact)
            self.assertLess(actual - exact, Decimal("2e-12"))
            tail = Decimal("0.5") - actual / (2 * (actual * actual + 2).sqrt())
            self.assertLessEqual(tail, Decimal(1) / Decimal(60))
        self.assertAlmostEqual(LOOK_ALPHA * 3, 0.05)

    def test_logarithmic_analytic_sample_uses_sample_variance(self):
        result = assess_scores([1.0, math.e, math.e**2], look_index=1)
        self.assertIs(result["valid"], True)
        self.assertAlmostEqual(result["mean_log"], 1.0, places=14)
        self.assertAlmostEqual(result["sample_log_sd"], 1.0, places=14)
        self.assertAlmostEqual(result["geometric_mean"], math.e, places=14)
        expected = math.exp(1 - math.sqrt(1682.0 / 177.0))
        self.assertLess(result["lower_speedup"], expected)
        self.assertAlmostEqual(result["lower_speedup"], expected, places=11)
        self.assertIs(result["certified"], False)
        self.check_json(result)

    def test_stable_improvement_passes_each_allocated_look(self):
        for look in (1, 2, 3):
            with self.subTest(look=look):
                result = assess_scores([1.08, 1.081, 1.082], look_index=look)
                self.assertIs(result["valid"], True)
                self.assertIs(result["certified"], True)
                self.assertGreater(result["lower_speedup"], 1.05)
                self.assertEqual(result["reason"], "floor_passed")
                self.check_json(result)

    def test_high_mean_with_large_uncertainty_does_not_pass(self):
        result = assess_scores([1.0, 1.2, 1.4], look_index=1)
        self.assertGreater(result["geometric_mean"], 1.05)
        self.assertLess(result["lower_speedup"], 1.05)
        self.assertIs(result["valid"], True)
        self.assertIs(result["certified"], False)

    def test_equal_reported_values_supply_no_certificate(self):
        for score in (1, 1.05, 1.2, sys.float_info.max):
            with self.subTest(score=score):
                result = assess_scores([score] * 3, look_index=1)
                self.assertIs(result["valid"], False)
                self.assertIs(result["certified"], False)
                self.assertEqual(result["reason"], "zero_log_score_variance")
                self.check_json(result)

    def test_invalid_or_exhausted_look_never_passes(self):
        for look in (0, 4, -1, True, False, 1.0, "1", None, 10**100):
            with self.subTest(look=look):
                result = assess_scores([1.08, 1.081, 1.082], look_index=look)
                self.assertIs(result["valid"], False)
                self.assertEqual(result["reason"], "invalid_or_exhausted_look")
                self.check_json(result)

    def test_invalid_floors_are_not_coerced(self):
        for floor in (0, -1, True, False, "1.05", None, math.nan, math.inf, 10**1000):
            with self.subTest(floor=repr(floor)):
                result = assess_scores([1.08, 1.081, 1.082], look_index=1, floor=floor)
                self.assertIs(result["valid"], False)
                self.assertEqual(result["reason"], "invalid_floor")
                self.check_json(result)

    def test_unserializable_invalid_integer_is_not_echoed(self):
        result = assess_scores([1.08, 1.081, 1.082], look_index=10**10000)
        self.assertIs(result["valid"], False)
        self.assertIsNone(result["look_index"])
        self.check_json(result)

    def test_only_three_materialized_scores_are_allowed(self):
        invalid = (None, "1.08", {0: 1.08, 1: 1.09, 2: 1.1}, [], [1.08], [1.08] * 2,
                   [1.08] * 4, iter([1.08, 1.09, 1.1]))
        for scores in invalid:
            with self.subTest(kind=type(scores).__name__):
                result = assess_scores(scores, look_index=1)
                self.assertIs(result["valid"], False)
                self.assertEqual(result["reason"], "exactly_three_scores_required")
                self.check_json(result)
        self.assertTrue(assess_scores((1.08, 1.081, 1.082), look_index=1)["valid"])

    def test_bad_score_types_and_values_cannot_certify(self):
        for bad in (True, False, "1.08", None, {}, [], 0, -1, math.nan, math.inf,
                    -math.inf, 10**1000):
            for position in range(3):
                with self.subTest(bad=repr(bad), position=position):
                    scores = [1.08, 1.081, 1.082]
                    scores[position] = bad
                    result = assess_scores(scores, look_index=1)
                    self.assertIs(result["valid"], False)
                    self.assertIs(result["certified"], False)
                    self.assertEqual(result["reason"], "scores_must_be_finite_positive_numbers")
                    self.check_json(result)

    def test_bound_is_strict_at_the_reported_threshold(self):
        scores = [1.08, 1.081, 1.082]
        initial = assess_scores(scores, look_index=1)
        boundary = initial["lower_speedup"]
        self.assertFalse(assess_scores(scores, look_index=1, floor=boundary)["certified"])
        self.assertFalse(assess_scores(scores, look_index=1, floor=boundary + 1e-10)["certified"])
        self.assertTrue(assess_scores(scores, look_index=1, floor=boundary - 1e-10)["certified"])

    def test_extreme_finite_scores_keep_json_finite(self):
        for scores in ([1e-300, 1e-299, 1e-298],
                       [sys.float_info.max / 3, sys.float_info.max / 2, sys.float_info.max],
                       [5e-324, 1.0, sys.float_info.max]):
            with self.subTest(scores=scores):
                result = assess_scores(scores, look_index=1)
                self.assertIs(result["valid"], True)
                self.check_json(result)

    def test_order_and_input_storage_do_not_change_assessment(self):
        scores = [1.08, 1.081, 1.082]
        original = list(scores)
        first = assess_scores(scores, look_index=1)
        other = assess_scores(list(reversed(scores)), look_index=1)
        self.assertEqual(scores, original)
        for key in ("mean_log", "sample_log_sd", "lower_speedup", "certified"):
            self.assertEqual(first[key], other[key])


if __name__ == "__main__":
    unittest.main()
