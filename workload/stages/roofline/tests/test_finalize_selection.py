"""Roofline ranks on the profile's GPU shares, an explicit input: a UT request does not
carry them (PR #619 review). Missing share evidence is a missing precondition."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FINALIZE = Path(__file__).resolve().parents[1] / "finalize.py"


class SelectionInputTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out, self.inp = self.root / "roofline/out", self.root / "ut/out"
        self.out.mkdir(parents=True)
        self.inp.mkdir(parents=True)
        (self.out / "rows.json").write_text(json.dumps({"rows": []}))
        (self.out / "folded.json").write_text(json.dumps({"k": {}}))

    def finalize(self, params):
        (self.out / "params.json").write_text(json.dumps(params))
        env = dict(os.environ, OUT=str(self.out), IN=str(self.inp), RUN_ROOT=str(self.root),
                   PARAMS=str(self.out / "params.json"), STAGE="roofline",
                   STAGE_SKILL="kernel-roofline")
        subprocess.run([sys.executable, str(FINALIZE)], env=env, capture_output=True, text=True)
        return json.loads((self.out / "result.json").read_text())

    def test_missing_share_evidence_blocks(self):
        res = self.finalize({})
        self.assertEqual(res["status"], "blocked")
        self.assertIn("params.selection", res["notes"])

    def test_selection_param_resolves_against_run_root(self):
        sel = self.root / "profile/out/selection_targets.json"
        sel.parent.mkdir(parents=True)
        sel.write_text(json.dumps({"targets": []}))
        res = self.finalize({"selection": "profile/out/selection_targets.json"})
        self.assertEqual(res["status"], "fail")           # read, but it names no targets
        self.assertIn(str(sel), res["notes"])


if __name__ == "__main__":
    unittest.main()
