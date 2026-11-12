#!/usr/bin/env python3
"""applyback measures every config a UT covered; leaving one out is `incomplete`, never `ok`.

Run:  python3 -I workload/stages/applyback/tests/test_points.py   (stdlib, CPU only)
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SLOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("points", SLOT / "points.py")
points = importlib.util.module_from_spec(spec)
spec.loader.exec_module(points)

PIDS = ["tp4-ep4-c%d-x" % c for c in (1, 4, 8, 16)]


def baseline(measure_order=None):
    top = {
        "multi_config": True,
        "points": [{"point_id": p, "config": {"concurrency": int(p.split("-c")[1].split("-")[0])}}
                   for p in PIDS],
        "groups": {"ep4": list(PIDS)},
        "representative": [PIDS[0], PIDS[-1]],
    }
    if measure_order:
        top["measure_order"] = measure_order
    return top


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


class PointsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tasks = self.tmp / "tasks"
        # two tasks; the UT covered c1, c8 and c16, plus a config the baseline does not have
        write(self.tasks / "moe" / "meta.json", {"workload": {"cases": [
            {"config_id": PIDS[0]}, {"config_id": PIDS[2]}, {"config_id": PIDS[3]}]}})
        write(self.tasks / "norm" / "meta.json", {"workload": {"cases": [
            {"config_id": PIDS[3]}, {"config_id": "tp8-ep8-c13-x"}]}})

    def test_default_is_every_covered_point_low_mid_high_first(self):
        sel = points.select(baseline(), self.tasks)
        self.assertEqual(sel["points"], [PIDS[0], PIDS[2], PIDS[3]])
        self.assertEqual(sel["unmeasured_covered"], [])
        self.assertEqual(sel["ut_configs_without_baseline"], ["tp8-ep8-c13-x"])

    def test_a_baseline_without_measure_order_is_not_narrowed_to_its_two_ends(self):
        # the 2026-11-06 run: points="all", no measure_order -> silently only c10 and c40
        sel = points.select(baseline(), self.tasks, "all")
        self.assertEqual(sel["points"], [PIDS[0], PIDS[2], PIDS[3], PIDS[1]])

    def test_representative_reports_the_covered_points_it_leaves_out(self):
        sel = points.select(baseline(), self.tasks, "representative")
        self.assertEqual(sel["points"], [PIDS[0], PIDS[3]])
        self.assertEqual(sel["unmeasured_covered"], [PIDS[2]])

    def test_explicit_measure_order_is_kept(self):
        order = [PIDS[3], PIDS[0], PIDS[1], PIDS[2]]
        sel = points.select(baseline(order), self.tasks)
        self.assertEqual(sel["points"], [PIDS[3], PIDS[0], PIDS[2]])

    def test_without_ut_cases_every_point_is_measured(self):
        sel = points.select(baseline(), self.tmp / "no_tasks")
        self.assertEqual(sel["points"], [PIDS[0], PIDS[2], PIDS[3], PIDS[1]])

    def test_finalize_calls_an_unmeasured_covered_point_incomplete(self):
        run = self.tmp / "run_musespark"
        write(run / "baseline" / "out" / "baseline.json", baseline())
        out = run / "applyback" / "out"
        (out / "overlay").mkdir(parents=True)
        (out / "applied.json").write_text(json.dumps({"applied": [{"name": "moe"}]}))
        params = out / "params.json"
        params.write_text(json.dumps({"points": "representative", "reference": "ci"}))
        env = dict(os.environ, OUT=str(out), RUN_ROOT=str(run), PARAMS=str(params),
                   TASKS_DIR=str(self.tasks), STAGE="applyback", STAGE_SKILL="overlay-ab",
                   WORKLOAD_ROOT=str(SLOT.parents[1]))
        subprocess.run([sys.executable, str(SLOT / "finalize.py")], env=env,
                       capture_output=True, text=True)
        result = json.loads((out / "result.json").read_text())
        ab = json.loads((out / "ab.json").read_text())
        self.assertNotEqual(result["status"], "ok")
        self.assertEqual(ab["verdict"], "incomplete")
        self.assertEqual(ab["point_selection"]["unmeasured_covered"], [PIDS[2]])
        left_out = json.loads((out / "points" / PIDS[2] / "result.json").read_text())
        self.assertIn("left it out", left_out["notes"])
        never = json.loads((out / "points" / PIDS[0] / "result.json").read_text())
        self.assertIn("never measured", never["notes"])
        self.assertIn("tp8-ep8-c13-x", result["notes"])


if __name__ == "__main__":
    unittest.main()
