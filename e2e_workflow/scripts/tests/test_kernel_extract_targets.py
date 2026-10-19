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


def row(name, prefill=0, decode=0, window_pct=None):
    """A ranking row. `window_pct` is the capture-window figure, never the threshold input."""
    entry = {"name": name, "share_of_phase_pct": {"PREFILL": prefill, "DECODE": decode}}
    if window_pct is not None:
        entry["selection_pct_gpu_time"] = window_pct
    return entry


def doc(rows, prefill=0.5, decode=0.5, basis="log_instance_counts_x_measured_point_cost",
        points=None):
    return {"ranking": rows, "phase_share": {
        "basis": basis, "PREFILL": prefill, "DECODE": decode,
        "points_measured": points if points is not None
        else {"PREFILL": 3, "DECODE": 3}}}


class ExtractionTargetsTest(unittest.TestCase):
    def test_threshold_only_selection_without_a_rank_cut(self):
        # p_lead leads PREFILL but PREFILL is only 12% of the workload, so it misses.
        # d7 is DECODE rank 7 and still clears 3%, which a Top-5 rule used to bury.
        rows = [row("p_lead", prefill=20), row("p_small", prefill=8)]
        rows += [row(f"d{i}", decode=share) for i, share in
                 enumerate([30, 12, 10, 8, 7, 6, 5], start=1)]
        result = select_targets(doc(rows, prefill=0.1217, decode=0.8783))
        selected = {t["id"] for t in result["targets"]}

        self.assertIn("d7", selected, "a rank-7 decode kernel above 3% must be selected")
        self.assertNotIn("p_lead", selected,
                         "a phase leader below the overall threshold is not selected")
        # 20% of a 12.17% phase is 2.43% of the workload.
        p_lead = result["per_phase"]["PREFILL"]["ranked"][0]
        self.assertEqual(p_lead["phase_rank"], 1)
        self.assertFalse(p_lead["selected"])
        self.assertAlmostEqual(p_lead["overall_pct"], 20 * 0.1217, places=6)
        self.assertEqual(result["target_count"], len(selected))

    def test_phase_split_decides_which_phase_is_selectable(self):
        rows = [row("p", prefill=10), row("d", decode=10)]
        prefill_heavy = select_targets(doc(rows, prefill=0.5848, decode=0.4152))
        decode_heavy = select_targets(doc(rows, prefill=0.1217, decode=0.8783))
        self.assertEqual({t["id"] for t in prefill_heavy["targets"]}, {"p", "d"})
        # The same ranking under the real decode-dominated split keeps only decode work.
        self.assertEqual({t["id"] for t in decode_heavy["targets"]}, {"d"})

    def test_overall_share_ignores_the_window_percentage_in_the_ranking(self):
        # A window figure that would have passed the threshold must not rescue the row.
        rows = [row("window_inflated", prefill=10, window_pct=17.7)]
        result = select_targets(doc(rows, prefill=0.1217, decode=0.8783))
        self.assertEqual(result["targets"], [])
        reported = result["per_phase"]["PREFILL"]["ranked"][0]
        self.assertAlmostEqual(reported["overall_pct"], 1.217, places=6)
        self.assertEqual(result["per_phase"]["PREFILL"]["phase_share_of_workload"],
                         0.1217 / (0.1217 + 0.8783))

    def test_percentages_and_fractions_are_both_accepted(self):
        rows = [row("d", decode=10)]
        as_fraction = select_targets(doc(rows, prefill=0.1217, decode=0.8783))
        as_percent = select_targets(doc(rows, prefill=12.17, decode=87.83))
        self.assertEqual(as_fraction["targets"][0]["overall_pct"],
                         as_percent["targets"][0]["overall_pct"])

    def test_missing_or_single_window_phase_share_fails(self):
        rows = [row("d", decode=10)]
        with self.assertRaises(ValueError):
            select_targets({"ranking": rows})
        for basis in ["single_window", "capture_window", "trace_window"]:
            with self.subTest(basis=basis), self.assertRaises(ValueError):
                select_targets(doc(rows, basis=basis))

    def test_a_phase_with_mass_measured_at_one_point_fails(self):
        rows = [row("d", decode=10)]
        with self.assertRaises(ValueError):
            select_targets(doc(rows, points={"PREFILL": 3, "DECODE": 1}))
        # A phase with no mass needs no points.
        ok = select_targets(doc(rows, prefill=0, decode=1.0,
                                points={"PREFILL": 0, "DECODE": 3}))
        self.assertEqual({t["id"] for t in ok["targets"]}, {"d"})

    def test_window_denominator_is_allowed_only_when_declared(self):
        rows = [row("d", decode=10)]
        result = select_targets(doc(rows, basis="single_window",
                                    points={"PREFILL": 1, "DECODE": 1}),
                                allow_window_denominator=True)
        self.assertTrue(result["policy"]["window_denominator_allowed"])
        self.assertEqual({t["id"] for t in result["targets"]}, {"d"})

    def test_top_k_is_an_execution_cap_that_keeps_the_remainder(self):
        rows = [row(f"d{i}", decode=share) for i, share in enumerate([30, 20, 10], start=1)]
        result = select_targets(doc(rows, prefill=0, decode=1.0,
                                    points={"PREFILL": 0, "DECODE": 3}), top_k=2)
        self.assertEqual([t["id"] for t in result["targets"]], ["d1", "d2"])
        self.assertEqual([d["id"] for d in result["dropped_by_cap"]], ["d3"])
        self.assertEqual(result["dropped_by_cap"][0]["reason"],
                         "above_threshold_but_outside_top_k")

    def test_no_qualifying_kernels_is_valid(self):
        result = select_targets(doc([row("tiny", prefill=5)], prefill=0.1, decode=0.9,
                                    points={"PREFILL": 2, "DECODE": 2}))
        self.assertEqual(result["target_count"], 0)
        self.assertEqual(len(result["per_phase"]["PREFILL"]["ranked"]), 1)
        self.assertEqual(result["per_phase"]["DECODE"]["ranked"], [])

    def test_do_not_hide_selected_collectives_or_optimization_drops(self):
        collective = {**row("collective", decode=30), "editable": False}
        low_headroom = {**row("compute", decode=20), "expected_e2e_gain": 0}
        result = select_targets(doc([collective, low_headroom], prefill=0, decode=1.0,
                                    points={"PREFILL": 0, "DECODE": 3}))
        self.assertEqual({r["id"] for r in result["targets"]}, {"collective", "compute"})

    def test_ties_use_identity_and_ignore_old_global_rank(self):
        rows = [{**row(name, decode=10), "rank": 99 - i}
                for i, name in enumerate(["f", "e", "d", "c", "b", "a"])]
        result = select_targets(doc(rows, prefill=0, decode=1.0,
                                    points={"PREFILL": 0, "DECODE": 3}))
        self.assertEqual([r["id"] for r in result["targets"]],
                         ["a", "b", "c", "d", "e", "f"])

    def test_invalid_or_missing_costs_and_duplicate_identity_fail(self):
        for bad in [None, "4", True, float("nan"), float("inf"), -1, 101]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                select_targets(doc([{"name": "a",
                                     "share_of_phase_pct": {"DECODE": bad}}]))
        with self.assertRaises(ValueError):
            select_targets(doc([row("a", decode=10), row("a", decode=5)]))
        with self.assertRaises(ValueError):
            select_targets(doc([{"name": "a"}]))
        with self.assertRaises(ValueError):
            select_targets(doc([]))


if __name__ == "__main__":
    unittest.main()
