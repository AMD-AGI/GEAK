"""The apply-back A/B stop rule: stop early only when the answer is already clear."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fusion_ab_decide as ab  # noqa: E402


class FusionAbDecideTest(unittest.TestCase):
    def test_one_pair_is_never_enough(self):
        self.assertEqual(ab.decide([191.1], [202.7])["decision"], "continue")

    def test_clear_win_stops_after_two_pairs(self):
        # Measured on Qwen3.5-397B (router fusion e02): +6.0%, spread ~0.15%.
        out = ab.decide([191.154, 191.096], [202.704, 202.411])
        self.assertEqual(out["decision"], "accept")
        self.assertTrue(out["nonoverlap"])
        self.assertGreater(out["delta_pct"], 5.9)

    def test_small_real_win_needs_the_third_pair(self):
        ref, cand = [200.0, 200.4], [202.0, 202.3]      # +0.97%, spread 0.2% -> not 10x
        self.assertEqual(ab.decide(ref, cand)["decision"], "continue")
        self.assertEqual(ab.decide(ref + [200.2], cand + [202.1])["decision"], "accept")

    def test_overlap_at_max_pairs_rejects(self):
        out = ab.decide([200.0, 201.5, 200.8], [201.0, 202.0, 201.2])
        self.assertFalse(out["nonoverlap"])
        self.assertEqual(out["decision"], "reject")

    def test_inside_noise_band_rejects_at_max_pairs(self):
        out = ab.decide([200.0, 200.02, 200.01], [200.5, 200.52, 200.51])   # +0.25%
        self.assertTrue(out["nonoverlap"])
        self.assertEqual(out["decision"], "reject")

    def test_slower_candidate_rejects_immediately(self):
        self.assertEqual(ab.decide([200.0, 200.2], [199.0, 199.1])["decision"], "reject")

    def test_identical_legs_do_not_make_any_delta_clear(self):
        # Zero spread would make 10 x spread = 0; the floor keeps a 0.4% delta from "clearing".
        out = ab.decide([200.0, 200.0], [200.8, 200.8], noise_band_pct=0.5)
        self.assertEqual(out["decision"], "continue")

    def test_non_positive_leg_rejects(self):
        self.assertEqual(ab.decide([200.0, 0.0], [210.0, 211.0])["decision"], "reject")

    def test_cli_prints_the_decision_line(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ab.main(["--ref", "191.1", "191.0", "--cand", "202.7", "202.4"])
        self.assertIn("AB_DECISION=accept", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
