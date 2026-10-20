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


def verified_points(counts):
    """`counts` per phase -> that many verified, counted sampling points."""
    return [{"point_id": "%s_%d" % (phase[:2].lower(), i), "phase": phase,
             "verified": True, "counted": True}
            for phase, n in counts.items() for i in range(n)]


def doc(rows, prefill=0.5, decode=0.5, basis="log_instance_counts_x_measured_point_cost",
        points=None, point_records=None):
    counts = points if points is not None else {"PREFILL": 3, "DECODE": 3}
    return {"ranking": rows, "phase_share": {
        "basis": basis, "PREFILL": prefill, "DECODE": decode,
        "points": point_records if point_records is not None else verified_points(counts)}}


def group_doc(rows, groups):
    """A decode-only workload, so a row's decode share IS its whole-workload share."""
    return {**doc(rows, prefill=0, decode=1.0,
                  points={"PREFILL": 0, "DECODE": 3}), "fusion_groups": groups}


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

    def test_fusion_group_recalls_members_that_miss_the_threshold_alone(self):
        # rope 1.5% + cache 1.2% + norm 1.0% = 3.7% of the workload: no stage clears 3%,
        # the chain does, and a later fusion needs all three.
        rows = [row("rope", decode=1.5), row("cache", decode=1.2), row("norm", decode=1.0),
                row("big", decode=30)]
        result = select_targets(group_doc(rows, [{
            "id": "qk_rope_cache_norm", "members": ["rope", "cache", "norm"],
            "adjacency_evidence": {"kind": "observed_consecutive_launches",
                                   "observed_sequences": 30, "intervening_kernels": 0},
            "rationale": "consecutive stages of the QK path"}]))
        by_id = {t["id"]: t for t in result["targets"]}
        self.assertEqual(set(by_id), {"rope", "cache", "norm", "big"})
        self.assertEqual(by_id["rope"]["reason"], "recalled_by_fusion_group")
        self.assertEqual(by_id["big"]["reason"], "overall_share_above_threshold")
        self.assertEqual(by_id["cache"]["fusion_groups"], ["qk_rope_cache_norm"])
        self.assertEqual(result["recalled_count"], 3)
        self.assertTrue(result["fusion_groups"][0]["recalled"])
        self.assertAlmostEqual(result["fusion_groups"][0]["group_overall_pct"], 3.7, places=6)

    def test_a_group_below_the_threshold_recalls_nothing(self):
        rows = [row("a", decode=1.0), row("b", decode=1.2)]
        result = select_targets(group_doc(rows, [{
            "id": "tiny", "members": ["a", "b"],
            "adjacency_evidence": {"kind": "observed_dispatch_chain"}}]))
        self.assertEqual(result["targets"], [])
        self.assertFalse(result["fusion_groups"][0]["recalled"])
        self.assertEqual(result["fusion_groups"][0]["reason"],
                         "fusion_group_share_not_above_threshold")

    def test_a_member_is_counted_once_across_overlapping_groups(self):
        rows = [row("shared", decode=2.0), row("x", decode=2.0), row("y", decode=2.0)]
        evidence = {"kind": "observed_consecutive_launches"}
        result = select_targets(group_doc(rows, [
            {"id": "g1", "members": ["shared", "x"], "adjacency_evidence": evidence},
            {"id": "g2", "members": ["shared", "y"], "adjacency_evidence": evidence}]))
        by_id = {t["id"]: t for t in result["targets"]}
        self.assertEqual(by_id["shared"]["fusion_groups"], ["g1", "g2"])
        # 2 + 2 + 2, not 2 + 2 + 2 + 2 with `shared` booked twice.
        self.assertAlmostEqual(result["selected_overall_pct"], 6.0, places=6)
        self.assertEqual(result["target_count"], 3)

    def test_a_grouping_without_adjacency_evidence_is_rejected(self):
        rows = [row("parent", decode=5), row("child", decode=5)]
        for evidence in [None, {}, {"kind": "name_similarity"},
                         {"kind": "shared_callable_name"}, "observed_consecutive_launches"]:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                select_targets(group_doc(rows, [{"id": "g", "members": ["parent", "child"],
                                                 "adjacency_evidence": evidence}]))

    def test_malformed_groups_are_rejected(self):
        rows = [row("a", decode=2), row("b", decode=2)]
        evidence = {"kind": "observed_consecutive_launches"}
        bad = [
            {"id": "g", "members": ["a"], "adjacency_evidence": evidence},          # one stage
            {"id": "g", "members": ["a", "a"], "adjacency_evidence": evidence},     # repeat
            {"id": "g", "members": ["a", "ghost"], "adjacency_evidence": evidence}, # absent
            {"members": ["a", "b"], "adjacency_evidence": evidence},                # no id
        ]
        for group in bad:
            with self.subTest(group=group), self.assertRaises(ValueError):
                select_targets(group_doc(rows, [group]))
        with self.assertRaises(ValueError):  # duplicate group id
            select_targets(group_doc(rows, [
                {"id": "g", "members": ["a", "b"], "adjacency_evidence": evidence},
                {"id": "g", "members": ["a", "b"], "adjacency_evidence": evidence}]))

    def test_group_threshold_can_differ_from_the_kernel_threshold(self):
        rows = [row("a", decode=1.0), row("b", decode=1.2)]
        groups = [{"id": "g", "members": ["a", "b"],
                   "adjacency_evidence": {"kind": "observed_dispatch_chain"}}]
        self.assertEqual(select_targets(group_doc(rows, groups))["targets"], [])
        relaxed = select_targets(group_doc(rows, groups), group_pct_gt=2)
        self.assertEqual({t["id"] for t in relaxed["targets"]}, {"a", "b"})
        self.assertEqual(relaxed["policy"]["fusion_group_pct_gt"], 2)

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


class SamplingPointVerificationTest(unittest.TestCase):
    """A split is only as good as the captures under it."""

    ROWS = [row("d", decode=10)]

    def share(self, point_records):
        return select_targets(doc(self.ROWS, prefill=0.4, decode=0.6,
                                  point_records=point_records))

    def pts(self, **over):
        base = [{"point_id": "pr_1", "phase": "PREFILL", "verified": True, "counted": True},
                {"point_id": "pr_2", "phase": "PREFILL", "verified": True, "counted": True},
                {"point_id": "de_1", "phase": "DECODE", "verified": True, "counted": True},
                {"point_id": "de_2", "phase": "DECODE", "verified": True, "counted": True}]
        for rec in base:
            if rec["point_id"] in over:
                rec.update(over[rec["point_id"]])
        return base

    def test_all_verified_points_pass(self):
        self.assertEqual({t["id"] for t in self.share(self.pts())["targets"]}, {"d"})

    def test_an_unverified_point_cannot_be_counted(self):
        # The retest's four DECODE captures held no decode step span and were counted anyway.
        with self.assertRaises(ValueError) as cm:
            self.share(self.pts(de_1={"verified": False}))
        self.assertIn("could not verify", str(cm.exception))

    def test_excluding_an_unverified_point_is_allowed_if_enough_remain(self):
        pts = self.pts(de_1={"verified": False, "counted": False,
                             "why_excluded": "no DECODE step spans in the capture"})
        pts.append({"point_id": "de_3", "phase": "DECODE", "verified": True, "counted": True})
        self.assertEqual({t["id"] for t in self.share(pts)["targets"]}, {"d"})

    def test_dropping_below_the_minimum_fails_the_phase(self):
        with self.assertRaises(ValueError) as cm:
            self.share(self.pts(de_1={"verified": False, "counted": False},
                                de_2={"verified": False, "counted": False}))
        self.assertIn("DECODE", str(cm.exception))
        self.assertIn("verified", str(cm.exception))

    def test_a_partial_point_needs_relabelling_before_it_counts(self):
        # 'PREFILL captured, but not at new_tokens=2660' -- real, but about another shape.
        with self.assertRaises(ValueError) as cm:
            self.share(self.pts(pr_1={"verified": "partial"}))
        self.assertIn("relabelled_to_achieved_shape", str(cm.exception))
        ok = self.share(self.pts(pr_1={"verified": "partial",
                                       "relabelled_to_achieved_shape": True}))
        self.assertEqual({t["id"] for t in ok["targets"]}, {"d"})

    def test_points_list_is_required_and_well_formed(self):
        for bad in [None, [], {}, [{"phase": "DECODE", "verified": True, "counted": True}],
                    [{"point_id": "x", "phase": "DECODE", "verified": "yes", "counted": True}]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                select_targets({"ranking": self.ROWS, "phase_share": {
                    "basis": "log_instance_counts_x_measured_point_cost",
                    "PREFILL": 0.4, "DECODE": 0.6, "points": bad}})

    def test_points_measured_must_agree_with_what_was_counted(self):
        d = doc(self.ROWS, prefill=0.4, decode=0.6, point_records=self.pts())
        d["phase_share"]["points_measured"] = {"PREFILL": 6, "DECODE": 2}
        with self.assertRaises(ValueError) as cm:
            select_targets(d)
        self.assertIn("disagrees", str(cm.exception))


class CommunicationOperatorScopeTest(unittest.TestCase):
    """Collectives are selected for mass accounting but are not given UTs."""

    def build(self, **kw):
        rows = [{**row("allreduce", decode=20), "entity_kind": "collective"},
                {**row("allreduce_pf", prefill=30), "entity_kind": "collective"},
                row("gemm", decode=10)]
        return select_targets(doc(rows, prefill=0.4, decode=0.6), **kw)

    def test_collectives_are_selected_and_keep_their_mass(self):
        out = self.build()
        ids = {t["id"] for t in out["targets"]}
        self.assertEqual(ids, {"allreduce", "allreduce_pf", "gemm"},
                         "a collective stays selected; it holds real device time")
        self.assertAlmostEqual(out["selected_overall_pct"],
                               sum(t["overall_pct"] for t in out["targets"]), places=9)

    def test_collectives_carry_no_ut_and_are_not_deferrals(self):
        out = self.build()
        scopes = {t["id"]: t["ut_scope"] for t in out["targets"]}
        self.assertEqual(scopes, {"allreduce": "excluded", "allreduce_pf": "excluded",
                                  "gemm": "ut"})
        self.assertEqual(out["ut_target_count"], 1)
        self.assertEqual({e["id"] for e in out["ut_excluded"]},
                         {"allreduce", "allreduce_pf"})
        for entry in out["ut_excluded"]:
            self.assertIn("out of optimization scope", entry["why"])
        # Nothing is parked as pending work: these are scope exclusions, not deferrals.
        self.assertEqual(out["dropped_by_cap"], [])

    def test_the_excluded_kinds_are_configurable_and_recorded(self):
        out = self.build(ut_exclude_kinds=())
        self.assertEqual(out["ut_target_count"], 3)
        self.assertEqual(out["ut_excluded"], [])
        self.assertEqual(out["policy"]["ut_excluded_entity_kinds"], [])
        self.assertEqual(self.build()["policy"]["ut_excluded_entity_kinds"], ["collective"])

    def test_an_unlabelled_row_defaults_to_a_gpu_kernel_owing_a_ut(self):
        out = select_targets(doc([row("x", decode=10)], prefill=0.4, decode=0.6))
        self.assertEqual(out["targets"][0]["entity_kind"], "gpu_kernel")
        self.assertEqual(out["targets"][0]["ut_scope"], "ut")

    def test_a_malformed_entity_kind_fails(self):
        for bad in ["", "   ", 7, []]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                select_targets(doc([{**row("x", decode=10), "entity_kind": bad}],
                                   prefill=0.4, decode=0.6))
