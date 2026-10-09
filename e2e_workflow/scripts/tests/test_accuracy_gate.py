import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import accuracy_gate  # noqa: E402


def _legs(n, lost, gained, base_right):
    """Paired outcomes: `lost` base-right/cand-wrong, `gained` the reverse."""
    base = {i: i < base_right for i in range(n)}
    cand = dict(base)
    wrong_in_base = [i for i in range(n) if not base[i]]
    for i in range(lost):
        cand[i] = False
    for i in wrong_in_base[:gained]:
        cand[i] = True
    return base, cand


class AccuracyGateTest(unittest.TestCase):
    def test_drop_within_tolerance_passes(self):
        base, cand = _legs(200, lost=3, gained=2, base_right=180)
        result = accuracy_gate.gate(base, cand, tol=0.01)
        self.assertEqual(result["verdict"], "pass")
        self.assertFalse(result["warning"])

    def test_small_sample_material_drop_is_inconclusive_not_pass(self):
        # 0.895 -> 0.870 at n=200 (16 lost / 11 gained): the measured e14 case.
        base, cand = _legs(200, lost=16, gained=11, base_right=179)
        result = accuracy_gate.gate(base, cand, tol=0.01)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertAlmostEqual(result["drop"], 0.025)

    def test_significant_material_drop_fails(self):
        base, cand = _legs(200, lost=20, gained=4, base_right=180)
        result = accuracy_gate.gate(base, cand, tol=0.01)
        self.assertEqual(result["verdict"], "fail")
        self.assertLess(result["mcnemar_one_sided_p"], 0.05)

    def test_unresolvable_drop_on_full_set_passes_with_warning(self):
        # net 14 lost (1.06pt > tol) spread over 126 discordant questions: p~0.12
        base, cand = _legs(1319, lost=70, gained=56, base_right=1180)
        result = accuracy_gate.gate(base, cand, tol=0.01)
        self.assertEqual(result["verdict"], "pass")
        self.assertTrue(result["warning"])

    def test_legs_must_answer_the_same_questions(self):
        with self.assertRaises(ValueError):
            accuracy_gate.gate({0: True, 1: True}, {0: True, 2: True}, tol=0.01)


if __name__ == "__main__":
    unittest.main()
