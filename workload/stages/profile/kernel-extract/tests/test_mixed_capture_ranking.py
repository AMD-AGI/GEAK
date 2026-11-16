"""Mixed graph/eager sampling retains the native graph's collective wait mass."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_ranking


class MixedCaptureRankingTests(unittest.TestCase):
    def setUp(self):
        self.cov = {"phases": {"DECODE": {
            "total_mass_us": 1000,
            "kernels": [{"kernel": "compute", "mass_us": 400},
                        {"kernel": "__collective_wait__", "mass_us": 600}],
            "points": [{"point_id": "graph", "collective_wait_mass_us": 100},
                       {"point_id": "eager", "collective_wait_mass_us": 500}]}}}

    def test_mixed_points_exclude_only_eager_wait(self):
        r = build_ranking.build(self.cov, eager_points=["eager"])
        self.assertEqual(r["denominators"]["total_device_us"], 500)
        self.assertEqual(r["ranking"][0]["selection_pct_gpu_time"], 80)
        self.assertEqual(r["non_selectable"][0]["mass_us"], 100)
        self.assertEqual(r["denominators"]["excluded_from_denominator"]["DECODE"]
                         ["point_ids"], ["eager"])

    def test_all_graph_keeps_wait(self):
        r = build_ranking.build(self.cov)
        self.assertEqual(r["denominators"]["total_device_us"], 1000)
        self.assertEqual(r["ranking"][0]["selection_pct_gpu_time"], 40)

    def test_legacy_phase_option_still_excludes_all_wait(self):
        r = build_ranking.build(self.cov, eager_captured=["DECODE"])
        self.assertEqual(r["denominators"]["total_device_us"], 400)
        self.assertFalse(r["non_selectable"])

    def test_unknown_or_unmeasured_eager_point_fails(self):
        with self.assertRaisesRegex(ValueError, "absent from ledger"):
            build_ranking.build(self.cov, eager_points=["typo"])
        old = copy.deepcopy(self.cov)
        del old["phases"]["DECODE"]["points"][1]["collective_wait_mass_us"]
        with self.assertRaisesRegex(ValueError, "regenerate coverage_ledger"):
            build_ranking.build(old, eager_points=["eager"])


if __name__ == "__main__":
    unittest.main()
