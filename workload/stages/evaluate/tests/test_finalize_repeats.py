"""A comparison is graded only when the candidate ran the configured number of repeats."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FINALIZE = Path(__file__).resolve().parents[1] / "finalize.py"


def arm(throughput, repeats=None, legs=None):
    doc = {"throughput": throughput, "metric_basis": "tok/s"}
    if repeats is not None:
        doc["repeats"] = repeats
    if legs is not None:
        doc["legs"] = [{"throughput": throughput}] * legs
    return doc


class CompleteComparisonTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "applyback" / "out"
        (self.out / "e2e").mkdir(parents=True)
        reference = self.root / "baseline" / "out"
        reference.mkdir(parents=True)
        (reference / "baseline.json").write_text(json.dumps({"measured": True, "throughput": 100.0}))

    def finalize(self, candidate, stock, params):
        summary = dict(candidate, stock_control=dict(stock, admissible=True))
        (self.out / "e2e" / "bench_summary.json").write_text(json.dumps(summary))
        (self.out / "params.json").write_text(json.dumps(params))
        env = dict(os.environ, OUT=str(self.out), RUN_ROOT=str(self.root),
                   PARAMS=str(self.out / "params.json"), STAGE="applyback",
                   STAGE_SKILL="overlay-ab")
        subprocess.run([sys.executable, str(FINALIZE)], env=env, capture_output=True, text=True)
        return json.loads((self.out / "result.json").read_text())

    def test_complete_comparison_is_graded(self):
        res = self.finalize(arm(99.0, repeats=2), arm(100.0, repeats=2), {"repeats": 2})
        self.assertEqual(res["status"], "ok")
        self.assertTrue((self.out / "ab.json").exists())

    def test_candidate_still_measuring_blocks(self):
        res = self.finalize(arm(99.0, repeats=1), arm(100.0, repeats=2), {"repeats": 2})
        self.assertEqual(res["status"], "blocked")
        self.assertIn("candidate has 1 of 2", res["notes"])
        self.assertFalse((self.out / "ab.json").exists())

    def test_a_single_stock_control_run_is_enough(self):
        # overlay-ab asks for one admissible unchanged arm; it need not repeat `repeats` times
        res = self.finalize(arm(99.0, repeats=3), arm(100.0), {"repeats": 3})
        self.assertEqual(res["status"], "ok")
        res = self.finalize(arm(99.0, repeats=3), arm(100.0, repeats=1), {"repeats": 3})
        self.assertEqual(res["status"], "ok")

    def test_legs_count_stands_in_for_a_missing_repeats_field(self):
        done = self.finalize(arm(99.0, legs=2), arm(100.0, legs=2), {"repeats": 2})
        self.assertEqual(done["status"], "ok")
        short = self.finalize(arm(99.0, legs=1), arm(100.0, legs=2), {"repeats": 2})
        self.assertEqual(short["status"], "blocked")

    def test_arm_that_records_no_repeats_blocks(self):
        res = self.finalize(arm(99.0), arm(100.0), {"repeats": 2})
        self.assertEqual(res["status"], "blocked")
        self.assertIn("candidate has 0 of 2", res["notes"])

    def test_no_configured_repeats_leaves_the_comparison_unchecked(self):
        res = self.finalize(arm(99.0, repeats=1), arm(100.0, repeats=1), {})
        self.assertEqual(res["status"], "ok")


if __name__ == "__main__":
    unittest.main()
