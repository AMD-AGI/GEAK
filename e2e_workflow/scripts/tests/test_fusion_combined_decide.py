"""Combined apply-back A/B: one base server vs one server with every fusion stacked."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import fusion_combined_decide as cd  # noqa: E402


class DecideTest(unittest.TestCase):
    # Qwen3.5-397B 20261008: base ~190.8, all six fusions stacked ~247.3, gsm8k flat.
    BASE = [190.576, 191.084, 190.777]
    CAND = [247.096, 247.256, 247.31]

    def test_clear_combined_win_accepts(self):
        out = cd.decide(self.BASE, self.CAND, (0.970, 200), (0.965, 200))
        self.assertEqual(out["decision"], "accept")
        self.assertTrue(out["nonoverlap"])
        self.assertGreater(out["delta_pct"], 29.0)
        self.assertLess(out["gsm8k_z"], 2.0)

    def test_overlap_rejects(self):
        out = cd.decide([200.0, 201.5, 200.8], [201.0, 202.0, 201.2], (0.97, 200), (0.97, 200))
        self.assertFalse(out["nonoverlap"])
        self.assertEqual(out["decision"], "reject")

    def test_inside_noise_band_rejects(self):
        out = cd.decide([200.0, 200.02, 200.01], [200.5, 200.52, 200.51], (0.97, 200), (0.97, 200))
        self.assertTrue(out["nonoverlap"])
        self.assertEqual(out["decision"], "reject")

    def test_significant_accuracy_drop_rejects_a_fast_stack(self):
        out = cd.decide(self.BASE, self.CAND, (0.970, 200), (0.900, 200))
        self.assertTrue(out["perf_pass"])
        self.assertFalse(out["accuracy_pass"])
        self.assertEqual(out["decision"], "reject")

    def test_accuracy_drop_within_noise_passes(self):
        # 3pt at n=200 is ~1 sigma: noise, not a regression.
        out = cd.decide(self.BASE, self.CAND, (0.970, 200), (0.940, 200))
        self.assertTrue(out["accuracy_pass"])

    def test_missing_gsm8k_rejects(self):
        out = cd.decide(self.BASE, self.CAND, None, (0.965, 200))
        self.assertEqual(out["decision"], "reject")
        self.assertIn("base", out["reason"])

    def test_missing_rounds_reject(self):
        self.assertEqual(cd.decide([], self.CAND, (0.97, 200), (0.97, 200))["decision"], "reject")

    def test_unengaged_overlay_is_listed_but_does_not_flip_the_decision(self):
        out = cd.decide(self.BASE, self.CAND, (0.97, 200), (0.97, 200),
                        engaged={"router_shared_topk": 8, "moe_act_quant": 3}, tp=8)
        self.assertEqual(out["decision"], "accept")
        self.assertEqual(out["not_engaged"], ["moe_act_quant"])


class CliTest(unittest.TestCase):
    def _write(self, d, name, obj):
        path = os.path.join(d, name)
        with open(path, "w") as fh:
            json.dump(obj, fh)
        return path

    def test_cli_reads_bench_gsm8k_and_server_log(self):
        with tempfile.TemporaryDirectory() as d:
            base = self._write(d, "base.json", {"all_throughput": DecideTest.BASE})
            cand = self._write(d, "cand.json", {"all_throughput": DecideTest.CAND})
            gb = self._write(d, "gb.json", {"summary": {"exact_match": 0.97, "n": 200}})
            gc = self._write(d, "gc.json", {"summary": {"exact_match": 0.965, "n": 200}})
            log = os.path.join(d, "server.log")
            with open(log, "w") as fh:
                for rank in range(8):
                    fh.write("[TP%d] [overlay-router_shared_topk] ENGAGED via seam\n" % rank)
            out = os.path.join(d, "decision.json")
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, "fusion_combined_decide.py"),
                 "--base-summary", base, "--cand-summary", cand,
                 "--base-gsm8k", gb, "--cand-gsm8k", gc, "--cand-server-log", log,
                 "--expect-banner", "router_shared_topk", "--expect-banner", "moe_act_quant",
                 "--tp", "8", "--out", out],
                stdout=subprocess.PIPE, universal_newlines=True, check=True)
            self.assertTrue(proc.stdout.strip().endswith("COMBINED_DECISION=accept"))
            with open(out) as fh:
                doc = json.load(fh)
            self.assertEqual(doc["engaged"], {"router_shared_topk": 8, "moe_act_quant": 0})
            self.assertEqual(doc["not_engaged"], ["moe_act_quant"])


if __name__ == "__main__":
    unittest.main()
