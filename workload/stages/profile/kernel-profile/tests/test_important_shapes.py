#!/usr/bin/env python3
"""Prefill points spread over the observed range, important cells ranked by measured time
with their launch, and --phases forwarded with its values.

Run:  python3 -I workload/stages/profile/kernel-profile/tests/test_important_shapes.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import important_shapes  # noqa: E402
import observed_workload  # noqa: E402
import plan_points  # noqa: E402
import run_points  # noqa: E402

SERVING = {"model": "org/model", "tp": 4, "args": []}


def observed():
    # prefill: new tokens 100..2000, cached prefix grows with them; decode: bs 1..12
    prefill = [{"new_token": n, "new_seq": 1, "cached_token": 50 * n, "cuda_graph": False}
               for n in range(100, 2001, 100)]
    decode = [{"running": bs, "token": bs * (80000 + 1000 * bs), "iterations": 40,
               "cuda_graph": bs <= 8} for bs in range(1, 13)]
    return {"decode_log_interval": 40, "raw_prefill_rows": prefill, "raw_decode_rows": decode}


class PlanTest(unittest.TestCase):
    def test_prefill_points_span_the_range(self):
        doc = plan_points.plan(SERVING, {}, observed(), 8, prefill_points=3)
        pre = sorted(c["partition"][1] for c in doc["captures"] if c["partition"][0] == "PREFILL")
        self.assertEqual(len(pre), 3)
        self.assertLess(pre[0], 1000)
        self.assertGreater(pre[-1], 1500)
        self.assertEqual(doc["capture_count"], 8)          # decode takes the rest of the budget

    def test_default_keeps_the_greedy_order(self):
        doc = plan_points.plan(SERVING, {}, observed(), 2)
        self.assertEqual([c["partition"][1] for c in doc["captures"]], [2000, 1900])


class ImportantTest(unittest.TestCase):
    def setUp(self):
        self.plan = plan_points.plan(SERVING, {}, observed(), 0)
        self.plan["boundaries"] = {"graph_bs": [1, 2, 4, 8]}

    def coverage(self, decode_points):
        return {"phases": {"DECODE": {"points": [
            {"point_id": "p%d" % bs, "shape": bs, "context": 80000, "us_per_instance": us}
            for bs, us in decode_points]}}}

    def test_measured_time_ranks_cells_and_records_launch(self):
        # the bs=12 point costs 10x the bs=1 point: by proxy (iterations) every cell weighs
        # the same; by time the large batches dominate
        doc = important_shapes.rank(self.plan, self.coverage([(1, 100.0), (12, 1000.0)]), 50)
        dec = {c["shape"]: c for c in doc["cells"] if c["phase"] == "DECODE"}
        self.assertEqual(dec[12]["launch"], {"mode": "eager"})          # above the capture sizes
        self.assertEqual(dec[1]["why"], ["boundary"])                    # boundary, not heavy
        self.assertEqual(dec[1]["launch"], {"mode": "graph", "capture_bs": 1})
        self.assertNotIn(2, dec)                                         # cheap and not a boundary
        self.assertTrue(all("time" in c["why"] for s, c in dec.items() if 4 <= s < 12))
        self.assertEqual(doc["calibration"]["DECODE"]["basis"], "measured")
        self.assertGreaterEqual(doc["calibration"]["DECODE"]["time_covered_pct"], 50)

    def test_phase_without_measured_points_falls_back_to_proxy(self):
        doc = important_shapes.rank(self.plan, self.coverage([(1, 100.0)]), 90)
        self.assertTrue(doc["calibration"]["PREFILL"]["basis"].startswith("proxy"))
        pre = [c for c in doc["cells"] if c["phase"] == "PREFILL"]
        self.assertTrue(all(c["launch"] == {"mode": "eager"} for c in pre))

    def test_graph_cell_pads_to_the_next_capture_size(self):
        self.assertEqual(important_shapes.capture_bs(3, [1, 2, 4, 8]), 4)
        self.assertIsNone(important_shapes.capture_bs(9, [1, 2, 4, 8]))

    def test_a_long_tail_is_capped_and_says_so(self):
        doc = important_shapes.rank(self.plan, self.coverage([(1, 100.0)]), 90, max_cells=3)
        pre = [c for c in doc["cells"] if c["phase"] == "PREFILL" and "time" in c["why"]]
        self.assertEqual(len(pre), 3)
        self.assertTrue(doc["calibration"]["PREFILL"]["capped"])

    def test_prefill_bands_merge_neighbours_but_not_across_cuts(self):
        doc = important_shapes.rank(self.plan, self.coverage([(1, 100.0)]), 100)
        pre = {c["shape"]: c for c in doc["cells"] if c["phase"] == "PREFILL"}
        # 100..2000 by 100 with context 51x new tokens: fewer bands than shapes, none
        # spanning a power of two
        self.assertLess(len(pre), 20)
        self.assertEqual(sum(c.get("members", 1) for c in pre.values()), 20)
        for c in pre.values():
            lo, hi = c.get("shape_range", [c["shape"]] * 2)
            self.assertEqual((lo - 1).bit_length(), (hi - 1).bit_length(), c)
        variants = {"variants": [
            {"fn": "k", "frames": [{"args": {"x": {"__tensor__": 1, "shape": [n, 8]}}}]}
            for n in (128, 1100)]}
        self.assertEqual(important_shapes.variant_cuts(variants), [128, 1100])
        cut = important_shapes.rank(self.plan, self.coverage([(1, 100.0)]), 100, variants=variants)
        ranges = [c.get("shape_range", [c["shape"]] * 2) for c in cut["cells"] if c["phase"] == "PREFILL"]
        self.assertFalse(any(lo < 1100 <= hi for lo, hi in ranges), ranges)

    def test_capture_sizes_come_from_the_server_log(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        log = Path(tmp.name) / "server.log"
        log.write_text("[x TP0] Capture draft CUDA graph begin. bs=[1, 2]\n"
                       "[x TP0] Capture target verify CUDA graph begin. backend=full, "
                       "num_tokens_per_req=8, bs=[1, 2, 3, 4, 8], avail=1\n")
        self.assertEqual(observed_workload.graph_capture_bs([str(log)]), [1, 2, 3, 4, 8])


class PhasesArgTest(unittest.TestCase):
    def test_values_follow_the_flag(self):
        self.assertEqual(run_points.phases_arg(["--top-k", "5", "--phases", "PREFILL", "DECODE",
                                                "--overall-pct-gt", "3"]),
                         ["--phases", "PREFILL", "DECODE"])
        self.assertEqual(run_points.phases_arg(["--phases=DECODE"]), ["--phases=DECODE"])
        self.assertEqual(run_points.phases_arg(["--top-k", "5"]), [])


if __name__ == "__main__":
    unittest.main()
