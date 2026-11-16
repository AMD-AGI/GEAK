#!/usr/bin/env python3
"""ci_pins client: the leg replays the CI's recorded client argv, and `prove` accepts it.

Run:  python3 -I workload/stages/applyback/tests/test_ci_pins.py   (stdlib + PyYAML, CPU only)
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

SLOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ci_pins", SLOT.parent / "_shared" / "ci_pins.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

SHA = "c7bc74fd64371315a091bc15f9723882eead0dc1"
# the CI's `CLI Command:` as aiperf records it (2026-11 Muse Spark tp4-ep4-c10), plus a flag the
# old hardcoded client argv did not carry
CI_CLI = ("aiperf profile --url http://10.24.112.103:8000 --model org/model --tokenizer org/model "
          "--concurrency 10 --benchmark-duration 3600 --tokenizer-trust-remote-code --server-metrics "
          "http://10.24.112.103:8000/metrics http://10.24.112.103:9400/metrics --output-artifact-dir "
          "/logs/agentic/aiperf_artifacts --public-dataset semianalysis_cc_traces_weka_062126 "
          "--request-timeout-seconds 1800")


class ClientArgvTest(unittest.TestCase):
    def setUp(self):
        self.pd = Path(tempfile.mkdtemp()) / "tp4-ep4-c10-x"
        logs = self.pd / "server_logs" / "logs"
        (logs / "agentic").mkdir(parents=True)
        (logs / "config.yaml").write_text(
            "model:\n  path: hf:org/model\n  container: repo/image@sha256:abc\nroles:\n  agg: {}\n")
        (logs / "x_agg_w0.out").write_text("loading /hub/models--org--model/snapshots/%s/x\n" % SHA)
        (logs / "agentic" / "benchmark.log").write_text(
            "CLI Command: %s\nBenchmark Duration: 3600\n" % CI_CLI)

    def test_the_replayed_argv_is_the_ci_argv_and_proves(self):
        out = self.pd.parent / "leg"
        argv = P.client_argv(str(self.pd), "http://127.0.0.1:8000", str(out / "aiperf_artifacts"))
        self.assertEqual(argv[0], "profile")
        self.assertIn("--request-timeout-seconds", argv)
        self.assertEqual(argv[argv.index("--tokenizer-revision") + 1], SHA)
        (out).mkdir()
        (out / "server.log").write_text("models--org--model/snapshots/%s/\n" % SHA)
        (out / "benchmark.log").write_text("CLI Command: aiperf %s\nBenchmark Duration: 3600\n"
                                           % " ".join(argv))
        self.assertEqual(P.prove(str(self.pd), str(out / "server.log"), str(out / "benchmark.log"),
                                 str(out / "pins.json")), 0)


if __name__ == "__main__":
    unittest.main()
