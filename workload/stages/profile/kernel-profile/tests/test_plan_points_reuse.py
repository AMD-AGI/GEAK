#!/usr/bin/env python3
"""plan_points --reuse-capture: an identical deployment's signed censuses are planned verbatim
and resumed; anything else is captured, and a different deployment is refused.

Run:  python3 -I workload/stages/profile/kernel-profile/tests/test_plan_points_reuse.py
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import plan_points  # noqa: E402

IDENTITY = {"image": "img@sha256:1", "model_revision": "m1", "source_revision": "s1", "gpu_arch": "gfx950"}
SERVING = {"model": "org/model", "tp": 8, "gpus": "0,1,2,3,4,5,6,7",
           "args": ["--port", "30000", "--max-running-requests", "64"]}


def observed(decode_rows):
    return {"decode_log_interval": 10, "raw_prefill_rows": [],
            "raw_decode_rows": [{"running": bs, "token": bs * ctx, "iterations": n}
                                for bs, ctx, n in decode_rows]}


def config(root, rows, serving=SERVING, identity=IDENTITY):
    root.mkdir(parents=True)
    (root / "evidence").mkdir()
    (root / "serving.json").write_text(json.dumps(serving))
    (root / "runtime_identity.json").write_text(json.dumps(identity))
    (root / "evidence/observed_workload.json").write_text(json.dumps(observed(rows)))
    return root


def plan(eval_dir, *extra):
    return subprocess.run([sys.executable, str(SCRIPTS / "plan_points.py"), "--eval-dir", str(eval_dir),
                           "--num-points", "8", "--target-pct", "100", *map(str, extra)],
                          capture_output=True, text=True)


class ReuseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # c8 captured bs=8 and bs=16; its censuses are signed in profile_run.json
        self.c8 = config(self.tmp / "c8", [(8, 1000, 50), (16, 1000, 50)])
        self.assertEqual(plan(self.c8).returncode, 0)
        pts = json.loads((self.c8 / "sampling_points.json").read_text())["points"]
        (self.c8 / "evidence/census").mkdir()
        run = {"points": {}}
        for p in pts:
            (self.c8 / "evidence/census" / (p["point_id"] + ".json")).write_text("{}")
            run["points"][p["point_id"]] = {"phase": p["phase"], "capture_mode": "eager",
                                            "measurement_signature": p["measurement_signature"]}
        (self.c8 / "profile_run.json").write_text(json.dumps(run))
        self.c8_points = {p["point_id"]: p for p in pts}

    def test_identical_deployment_reuses_and_captures_only_the_rest(self):
        # c16 serves bs=16 (captured by c8) and bs=32 (new); only the port differs
        c16 = config(self.tmp / "c16", [(16, 1000, 80), (32, 1000, 20)],
                     serving=dict(SERVING, args=["--port", "30001", "--max-running-requests", "64"]))
        p = plan(c16, "--reuse-capture", self.c8)
        self.assertEqual(p.returncode, 0, p.stderr)
        doc = json.loads((c16 / "sampling_plan.json").read_text())
        reused = [c["point"] for c in doc["reused"]]
        self.assertEqual([r["batch_size"] for r in reused], [16])
        self.assertIn(reused[0]["point_id"], self.c8_points)            # planned verbatim
        self.assertEqual([c["point"]["batch_size"] for c in doc["captures"]], [32])
        census = c16 / "evidence/census" / (reused[0]["point_id"] + ".json")
        self.assertTrue(census.is_symlink())
        run = json.loads((c16 / "profile_run.json").read_text())["points"][reused[0]["point_id"]]
        self.assertEqual(run["measurement_signature"], reused[0]["measurement_signature"])
        self.assertEqual(run["reused_from"], str(self.c8.resolve()))

    def test_a_different_deployment_is_refused(self):
        c16 = config(self.tmp / "c16", [(16, 1000, 80)],
                     serving=dict(SERVING, args=["--max-running-requests", "128"]))
        p = plan(c16, "--reuse-capture", self.c8)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("different deployment", p.stderr)
        self.assertFalse((c16 / "sampling_points.json").exists())

    def test_reuse_needs_a_runtime_identity(self):
        c16 = config(self.tmp / "c16", [(16, 1000, 80)], identity={})
        p = plan(c16, "--reuse-capture", self.c8)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("runtime identity", p.stderr)

    def test_a_row_without_context_is_counted_not_guessed(self):
        obs = observed([(8, 1000, 50), (16, 1000, 50)])
        obs["raw_decode_rows"].append({"running": 2, "token": 0, "iterations": 40})   # `#token: 0`
        doc = plan_points.plan(SERVING, IDENTITY, obs, 8, target_pct=100)
        self.assertEqual(doc["rows_without_context"], {"DECODE": {"rows": 1, "instances": 40, "proxy_mass": 40}})
        self.assertEqual(sorted(c["point"]["batch_size"] for c in doc["captures"]), [8, 16])

    def test_without_reuse_the_plan_is_unchanged(self):
        doc = plan_points.plan(SERVING, IDENTITY, observed([(8, 1000, 50), (16, 1000, 50)]), 8, target_pct=100)
        self.assertEqual(doc["reused"], [])
        self.assertEqual(sorted(c["point"]["batch_size"] for c in doc["captures"]), [8, 16])


if __name__ == "__main__":
    unittest.main()
