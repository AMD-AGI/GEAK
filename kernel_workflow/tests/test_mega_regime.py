"""Unit tests for tools/mega_regime.py (shape -> regime map)."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import mega_regime as mr  # noqa: E402

BASE = ["--experts-per-rank", "48", "--topk", "6", "--world", "8", "--hidden", "7168",
        "--inter", "3072", "--weight-bits", "4.25", "--hbm-tbps", "6.9",
        "--math-tflops", "5000"]


def run(*extra):
    return mr.analyse(mr.build_parser().parse_args(BASE + list(extra)))


class RegimeTest(unittest.TestCase):
    def test_parse_guard(self):
        self.assertEqual(mr.parse_guard("64_uniform_mtpr64"),
                         {"guard": "64_uniform_mtpr64", "tokens": 64, "route": "uniform", "mtpr": 64})
        self.assertEqual(mr.parse_guard("8192_rank-mixed-skew")["mtpr"], None)
        with self.assertRaises(ValueError):
            mr.parse_guard("uniform_64")

    def test_small_is_weight_stream_large_is_compute(self):
        rows = {r["guard"]: r for r in run("--guards", "32_uniform", "16384_uniform")["guards"]}
        self.assertEqual(rows["32_uniform"]["binding_resource"], "weight_stream")
        self.assertEqual(rows["16384_uniform"]["binding_resource"], "compute")
        self.assertAlmostEqual(rows["32_uniform"]["rows_per_expert"], 4.0)

    def test_unexplained_share_marks_fixed_cost(self):
        out = run("--guards", "64_uniform_mtpr64", "--baseline-ms", "64_uniform_mtpr64=0.354")
        row = out["guards"][0]
        self.assertTrue(row["regime"].startswith("fixed_cost+"))
        self.assertGreater(row["unexplained_share"], 0.30)
        self.assertEqual(out["buckets"][0]["guards"], ["64_uniform_mtpr64"])

    def test_link_bound_only_when_given(self):
        self.assertNotIn("communication", run("--guards", "64_uniform")["guards"][0]["bounds_us"])
        bounds = run("--guards", "64_uniform", "--link-gbps", "400")["guards"][0]["bounds_us"]
        self.assertIn("communication", bounds)

    def test_tile_efficiency(self):
        self.assertEqual(mr.tile_efficiency(8, 1, 32), 0.25)
        self.assertEqual(mr.tile_efficiency(0, 1, 32), None)

    def test_no_kernel_specific_defaults(self):
        with self.assertRaises(SystemExit):
            mr.main(["--guards", "64_uniform", "--experts-per-rank", "48", "--topk", "6",
                     "--world", "8", "--hidden", "7168", "--inter", "3072",
                     "--hbm-tbps", "0", "--math-tflops", "1"])


if __name__ == "__main__":
    unittest.main()
