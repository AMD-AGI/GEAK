#!/usr/bin/env python3
"""applyback after a partial (salvage) profile delivery: same verdict, stated scope.

Run:  python3 -m pytest -q workload/stages/applyback/tests/test_partial_profile_scope.py
(stdlib, CPU only)
"""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SLOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("applyback_finalize", SLOT / "finalize.py")
applyback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(applyback)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class PartialProfileScopeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="applyback_partial_")
        self.addCleanup(shutil.rmtree, tmp)
        self.root = Path(tmp)
        self.out = self.root / "applyback" / "out"
        self.out.mkdir(parents=True)
        write(self.root / "profile" / "out" / "tasks" / "tA" / "meta.json",
              {"workload": {"cases": [{"config_id": "c4"}, {"config_id": "c16"}]}})
        self.profile({"delivery": "partial", "metrics": {
            "delivered_tasks": ["tA"], "uncaptured_configs": ["c64"],
            "blocked_identity_ids": ["kB", "kC"]}})

    def profile(self, extra):
        doc = {"stage": "profile", "skill": "kernel-extract", "status": "ok",
               "artifacts": {}, "metrics": {}}
        doc.update(extra)
        write(self.root / "profile" / "out" / "result.json", doc)

    def fake_judge(self):
        write(self.out / "ab.json", {"verdict": "gain", "e2e_delta_pct": 3.1})
        write(self.out / "result.json", {"stage": "applyback", "skill": "overlay-ab",
                                         "status": "ok", "artifacts": {"ab": "ab.json"},
                                         "metrics": {"verdict": "gain"},
                                         "notes": "2 points: c4 +3.1% (gain)."})
        (self.out / "REPORT.md").write_text("## End to end\n")
        return 0

    def run_main(self, **env):
        environ = dict(os.environ, OUT=str(self.out), RUN_ROOT=str(self.root), **env)
        with mock.patch.dict(os.environ, environ, clear=True), \
                mock.patch.object(applyback, "judge", self.fake_judge):
            return applyback.main()

    def test_partial_profile_labels_without_changing_the_verdict(self):
        self.assertEqual(self.run_main(), 0)
        ab = json.loads((self.out / "ab.json").read_text())
        self.assertEqual((ab["verdict"], ab["e2e_delta_pct"]), ("gain", 3.1))
        scope = ab["profile_delivery"]
        self.assertEqual(scope["profile_delivery"], "partial")
        self.assertEqual(scope["covered_configs"], ["c16", "c4"])
        self.assertEqual(scope["uncaptured_configs"], ["c64"])
        self.assertEqual(scope["blocked_identities"], ["kB", "kC"])
        self.assertIn("not a full-campaign verdict", scope["verdict_scope"])
        res = json.loads((self.out / "result.json").read_text())
        self.assertEqual((res["status"], res["metrics"]["verdict"]), ("ok", "gain"))
        self.assertEqual(res["delivery"], "partial")
        self.assertEqual(res["metrics"]["profile_delivery"], "partial")
        self.assertTrue(res["notes"].startswith("PARTIAL COVERAGE"))
        self.assertIn("not a full-campaign verdict", res["notes"])
        self.assertIn("2 points: c4 +3.1% (gain).", res["notes"])
        self.assertTrue((self.out / "REPORT.md").read_text().startswith(
            "> **PARTIAL COVERAGE"))

    def test_full_profile_is_untouched(self):
        self.profile({})
        self.run_main()
        res = json.loads((self.out / "result.json").read_text())
        self.assertNotIn("delivery", res)
        self.assertNotIn("profile_delivery", json.loads((self.out / "ab.json").read_text()))
        self.assertEqual((self.out / "REPORT.md").read_text(), "## End to end\n")

    def test_per_point_leg_is_not_labelled(self):
        self.run_main(APPLYBACK_POINT="c4")
        self.assertNotIn("delivery", json.loads((self.out / "result.json").read_text()))


if __name__ == "__main__":
    unittest.main()
