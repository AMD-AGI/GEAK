#!/usr/bin/env python3
"""ci_config: a recipe flag with several values launches and aligns as several values.

Run:  python3 -I workload/stages/applyback/tests/test_ci_config.py   (stdlib + PyYAML, CPU only)
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SLOT = Path(__file__).resolve().parents[1]
CI_CONFIG = SLOT / "ci_config.py"

RECIPE = """model:
  path: hf:org/model
  container: repo/image@sha256:abc
roles:
  agg:
    args:
      tensor-parallel-size: 8
      cuda-graph-bs: [1, 2, 4, 8]
      enable-metrics: true
      disable-radix-cache: false
    env:
      SGLANG_USE_AITER: 1
benchmark:
  env:
    CONC: 4
"""


class CiConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pd = self.tmp / "tp8-ep8-c4-x"
        (self.pd / "server_logs" / "logs").mkdir(parents=True)
        (self.pd / "server_logs" / "logs" / "config.yaml").write_text(RECIPE)
        (self.pd / "baseline.json").write_text(json.dumps({"config": {"concurrency": 4}}))

    def launch(self):
        r = subprocess.run([sys.executable, str(CI_CONFIG), "launch", str(self.pd)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def align(self, argv, env=None, image="repo/image@sha256:abc"):
        inspect = self.tmp / "inspect.json"
        inspect.write_text(json.dumps([{"Config": {
            "Image": image, "Env": env or ["SGLANG_USE_AITER=1"],
            "Cmd": ["-m", "sglang.launch_server", "--model-path", "org/model", "--port", "8000"] + argv}}]))
        out = self.tmp / "alignment.json"
        r = subprocess.run([sys.executable, str(CI_CONFIG), "align", str(self.pd), str(inspect), str(out)],
                           capture_output=True, text=True)
        return r.returncode, json.loads(out.read_text())

    def test_a_list_value_is_passed_as_separate_values(self):
        argv = self.launch()["server_argv"]
        i = argv.index("--cuda-graph-bs")
        self.assertEqual(argv[i + 1:i + 5], ["1", "2", "4", "8"])
        self.assertNotIn("--disable-radix-cache", argv)

    def test_the_launch_it_writes_aligns(self):
        rc, doc = self.align(self.launch()["server_argv"])
        self.assertEqual(rc, 0, doc)
        self.assertTrue(doc["aligned"])

    def test_a_dropped_value_of_a_list_flag_is_a_difference(self):
        argv = self.launch()["server_argv"]
        i = argv.index("--cuda-graph-bs")
        rc, doc = self.align(argv[:i + 4] + argv[i + 5:])     # drops the trailing 8
        self.assertEqual(rc, 1)
        self.assertEqual([d[0] for d in doc["args"]["differ"]], ["cuda-graph-bs"])


if __name__ == "__main__":
    unittest.main()
