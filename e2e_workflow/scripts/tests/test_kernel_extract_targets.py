"""Behavioral checks for the extraction scope, independent of optimizer budgets."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = (Path(__file__).resolve().parents[2] / "knowledge/kernel_extraction_skills"
          / "kernel-extract/scripts/select_targets.py")
spec = importlib.util.spec_from_file_location("extract_targets", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
select_targets = module.select_targets


def row(name, overall, prefill=0, decode=0):
    return {"name": name, "selection_pct_gpu_time": overall,
            "share_of_phase_pct": {"PREFILL": prefill, "DECODE": decode}}


class ExtractionTargetsTest(unittest.TestCase):
    def test_phase_top_five_then_strict_threshold_without_backfill(self):
        result = select_targets({"ranking": [
            row("p1", 8, prefill=30), row("p2", 3, prefill=20),
            row("p3", 2, prefill=15), row("p4", 6, prefill=12),
            row("shared", 7, prefill=10, decode=30),
            row("p6", 9, prefill=5),
            row("d2", 6, decode=20), row("d3", 5, decode=15),
            row("d4", 4, decode=10), row("d5", 3.01, decode=8),
            row("d6", 10, decode=7),
        ]})
        targets = {r["id"]: r for r in result["targets"]}
        self.assertEqual(set(targets), {"p1", "p4", "shared", "d2", "d3", "d4", "d5"})
        self.assertEqual(targets["shared"]["selected_phases"], ["PREFILL", "DECODE"])
        self.assertEqual(targets["shared"]["phase_ranks"], {"PREFILL": 5, "DECODE": 1})
        self.assertEqual(result["target_count"], 7)
        self.assertFalse(result["per_phase"]["PREFILL"]["top_k"][1]["selected"])

    def test_less_than_five_and_no_qualifying_kernels_are_valid(self):
        result = select_targets({"ranking": [row("tiny", 3, prefill=100)]})
        self.assertEqual(result["target_count"], 0)
        self.assertEqual(len(result["per_phase"]["PREFILL"]["top_k"]), 1)
        self.assertEqual(result["per_phase"]["DECODE"]["top_k"], [])

    def test_do_not_hide_selected_collectives_or_optimization_drops(self):
        collective = {**row("collective", 8, decode=30), "editable": False}
        low_headroom = {**row("compute", 4, decode=20), "expected_e2e_gain": 0}
        result = select_targets({"ranking": [collective, low_headroom], "budget": 1})
        self.assertEqual({r["id"] for r in result["targets"]}, {"collective", "compute"})

    def test_ties_use_identity_and_ignore_old_global_rank(self):
        rows = [{**row(name, 4, decode=10), "rank": 99-i}
                for i, name in enumerate(["f", "e", "d", "c", "b", "a"])]
        self.assertEqual([r["id"] for r in select_targets({"ranking": rows})["targets"]],
                         ["a", "b", "c", "d", "e"])

    def test_triton_specializations_share_one_slot(self):
        gemm = "_gemm_a8w8_blockscale_kernel_GROUP_K_128_SPLITK_BLOCK_SIZE_{}_GRID_MN_{}"
        triton = {"classification": "triton"}
        rows = [{**row(gemm.format(4096, 12), 9.9, decode=19.7), **triton},
                {**row(gemm.format(4096, 8), 9.8, decode=19.6), **triton},
                {**row(gemm.format(2048, 32), 5.1, decode=10.2), **triton},
                {**row("_gemm_a8w8_blockscale_kernel_v2_GROUP_K_128", 4.5, decode=9), **triton},
                {**row("Cijk_Alik_MT32x16_GRVW_8_1", 4.4, decode=8.8), "classification": "library_gemm"},
                row("quant", 3.6, decode=7.2), row("moe", 3.5, decode=7.0), row("attn", 3.2, decode=6.4)]
        result = select_targets({"ranking": rows})
        top = result["per_phase"]["DECODE"]["top_k"]
        self.assertEqual([r["id"] for r in top], [
            gemm.format(4096, 12), "_gemm_a8w8_blockscale_kernel_v2_GROUP_K_128",
            "Cijk_Alik_MT32x16_GRVW_8_1", "quant", "moe"])
        self.assertEqual(top[0]["specializations"],
                         [gemm.format(4096, 12), gemm.format(4096, 8), gemm.format(2048, 32)])
        self.assertAlmostEqual(top[0]["phase_pct"], 49.5)
        self.assertAlmostEqual(top[0]["overall_pct"], 24.8)
        self.assertNotIn("specializations", top[1])

    def test_invalid_or_missing_costs_and_duplicate_identity_fail(self):
        valid = row("a", 4, decode=10)
        for bad in [None, "4", True, float("nan"), float("inf"), -1, 101]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                select_targets({"ranking": [{**valid, "selection_pct_gpu_time": bad}]})
        for invalid in [[], [valid, valid], [{**valid, "share_of_phase_pct": {}}],
                        [{**valid, "share_of_phase_pct": {"DECODE": float("nan")}}]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_targets({"ranking": invalid})


if __name__ == "__main__":
    unittest.main()
