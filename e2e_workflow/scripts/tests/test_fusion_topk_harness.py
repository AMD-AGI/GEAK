import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_topk_harness as topk


class FusionTopkTest(unittest.TestCase):
    def _write(self, root, name, value):
        path = os.path.join(root, name)
        with open(path, "w") as fh:
            json.dump(value, fh)
        return path

    def _table(self):
        # providers: triton region (author C1) vs aiter region (author C2)
        return {"tables": [
            {"phase": "prefill", "pattern_id": "P0",
             "rows": [{"row_id": "pn", "provider": "aiter"},
                      {"row_id": "pt", "provider": "triton"}]},
            {"phase": "decode", "pattern_id": "P0",
             "rows": [{"row_id": "dn", "provider": "aiter"}]}]}

    def _candidates(self):
        return {"candidates": [
            # B collective: 现成算子=有 (kernel exists) but size-guard blocks the
            # fused path in prefill -> NOT actionable (via collective_guard_checks)
            {"candidate_id": "pf_ar", "phase": "prefill", "pattern_id": "P0",
             "family": "collective_norm", "implementation_class":
             "existing_api_needs_adapter", "readiness":
             "needs_source_dependency_proof", "exact_kernel_status": "yes",
             "removable_row_ids": ["pn"],
             "existing_apis": [{"name": "fused_allreduce_rmsnorm"}]},
            # same recipe, decode: flag engages it (A) + exact=yes -> actionable
            {"candidate_id": "dc_ar", "phase": "decode", "pattern_id": "P0",
             "family": "collective_norm", "implementation_class":
             "existing_flag_or_env", "readiness":
             "ready_for_api_validation", "exact_kernel_status": "yes",
             "removable_row_ids": ["dn"],
             "existing_apis": [{"name": "fused_allreduce_rmsnorm"}]},
            # author-track, triton region -> C1, actionable via authoring
            {"candidate_id": "pf_layout", "phase": "prefill", "pattern_id": "P0",
             "family": "layout", "implementation_class": "new_helper_kernel",
             "readiness": "research_only", "exact_kernel_status": "no",
             "removable_row_ids": ["pt"], "existing_apis": []},
        ]}

    def _validation(self):
        return {"metrics": {
            "phase_total_forward_us": {"prefill": 10000.0, "decode": 1000.0},
            # prefill collective fused path exceeds the size guard -> blocked here
            "collective_guard_checks": [
                {"candidate_id": "pf_ar", "verdict": "exceeds"}],
            "candidate_savings": [
                {"candidate_id": "pf_ar", "estimate_us": 40.0,
                 "stack_estimate_us": 400.0, "basis": "roofline"},
                {"candidate_id": "dc_ar", "estimate_us": 4.0,
                 "stack_estimate_us": 40.0, "basis": "roofline"},
                {"candidate_id": "pf_layout", "estimate_us": 30.0,
                 "stack_estimate_us": 300.0, "basis": "roofline"}]}}

    def _run(self, tmp, top_k=10):
        return topk.rank(
            self._write(tmp, "c.json", self._candidates()),
            self._write(tmp, "v.json", self._validation()),
            self._write(tmp, "t.json", self._table()), top_k)

    def test_tiers_recipes_actionability_and_boards(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, actions, recipes = self._run(tmp)
            # recipe_key includes tier-group, so the A (flag) collective and the
            # B (adapter) collective are SEPARATE recipes (not merged).
            col = [r for r in recipes if "collective_norm" in r["recipe_key"]]
            self.assertEqual(len(col), 2)
            a_rec = next(r for r in col if r["tier"] == "A")
            b_rec = next(r for r in col if r["tier"] == "B")
            # A recipe: decode occ, exact=yes -> actionable, tier A
            self.assertEqual(a_rec["per_phase"]["decode"]["tier"], "A")
            self.assertEqual(
                a_rec["per_phase"]["decode"]["actionable_us"], 40.0)
            # B collective, prefill size-guard blocked -> not actionable
            self.assertEqual(
                b_rec["per_phase"]["prefill"]["actionable_us"], 0.0)
            self.assertEqual(b_rec["per_phase"]["prefill"]["full_us"], 400.0)
            # author-track triton region -> C1
            layout = next(r for r in recipes if r["family"] == "layout")
            self.assertEqual(layout["tier"], "C1")
            self.assertTrue(layout["per_phase"]["prefill"]["actionable_us"] > 0)

    def test_merged_actions_only_A_B_sorted_and_C_deferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, actions, _ = self._run(tmp)
            # one merged list; only A/B shown; decode A collective present
            self.assertTrue(all(a["tier"] in ("A", "B") for a in actions))
            self.assertEqual(actions[0]["tier"], "A")
            self.assertEqual(actions[0]["phase"], "decode")
            self.assertIn("collective_norm", actions[0]["recipe_key"])
            self.assertIn("KernelFusion", actions[0]["route"])
            # prefill AR (exact=no B) is not actionable -> not in the list
            self.assertFalse(any(
                a["phase"] == "prefill" and "collective_norm" in a["recipe_key"]
                for a in actions))
            # C1 author-track deferred, not ranked
            self.assertEqual(result["deferred_author_count"], 1)

    def _cross_phase_rank(self, tmp, workload=None):
        table = {"tables": [
            {"phase": "prefill", "pattern_id": "P0",
             "rows": [{"row_id": "p", "provider": "aiter"}]},
            {"phase": "decode", "pattern_id": "P0",
             "rows": [{"row_id": "d", "provider": "aiter"}]},
        ]}
        candidates = {"candidates": [
            {"candidate_id": "prefill_win", "phase": "prefill",
             "pattern_id": "P0", "family": "prefill_family",
             "implementation_class": "existing_api_needs_adapter",
             "readiness": "ready_for_api_validation",
             "exact_kernel_status": "yes", "removable_row_ids": ["p"],
             "existing_apis": [{"name": "prefill_kernel"}]},
            {"candidate_id": "decode_win", "phase": "decode",
             "pattern_id": "P0", "family": "decode_family",
             "implementation_class": "existing_api_needs_adapter",
             "readiness": "ready_for_api_validation",
             "exact_kernel_status": "yes", "removable_row_ids": ["d"],
             "existing_apis": [{"name": "decode_kernel"}]},
        ]}
        validation = {"metrics": {
            "phase_total_forward_us": {"prefill": 1000.0, "decode": 100.0},
            "candidate_savings": [
                {"candidate_id": "prefill_win", "estimate_us": 10.0,
                 "stack_estimate_us": 100.0},
                {"candidate_id": "decode_win", "estimate_us": 1.0,
                 "stack_estimate_us": 5.0},
            ]}}
        return topk.rank(
            self._write(tmp, "cross-c.json", candidates),
            self._write(tmp, "cross-v.json", validation),
            self._write(tmp, "cross-t.json", table), 10,
            workload=workload)

    def test_real_workload_moves_repeated_decode_ahead_of_prefill(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, actions, _ = self._cross_phase_rank(
                tmp, {"isl": 8192, "osl": 1000, "conc": 4})
            # Locally prefill is 10% and decode is only 5%, but the request has
            # 999 decode forwards, so decode is the larger throughput target.
            self.assertEqual([a["phase"] for a in actions],
                             ["decode", "prefill"])
            decode = actions[0]
            self.assertEqual(result["workload"]["decode_forwards"], 999)
            self.assertEqual(result["workload_total_forward_us"], 100900.0)
            self.assertEqual(decode["forward_pct"], 5.0)
            self.assertEqual(decode["workload_weight"], 999)
            self.assertEqual(decode["workload_savings_us"], 4995.0)
            self.assertEqual(decode["workload_forward_pct"], 4.9504)
            self.assertEqual(
                result["execution_list"][0]["workload_forward_pct"], 4.9504)

    def test_osl_one_assigns_no_decode_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, actions, _ = self._cross_phase_rank(
                tmp, {"isl": 8192, "osl": 1, "conc": 4})
            self.assertEqual(result["workload"]["decode_forwards"], 0)
            self.assertEqual([a["phase"] for a in actions],
                             ["prefill", "decode"])
            self.assertEqual(actions[1]["workload_savings_us"], 0.0)
            self.assertEqual(actions[1]["workload_forward_pct"], 0.0)

    def test_missing_workload_preserves_phase_local_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, actions, _ = self._cross_phase_rank(tmp)
            self.assertFalse(result["workload"]["available"])
            self.assertEqual(result["workload"]["ranking_metric"],
                             "phase_forward_pct")
            self.assertEqual([a["phase"] for a in actions],
                             ["prefill", "decode"])
            self.assertIsNone(actions[0]["workload_forward_pct"])

    def test_mutually_exclusive_cross_tier_both_listed(self):
        # Two decode candidates sharing a removable row (mutually exclusive):
        # a high-benefit A and a low-benefit B. Different tiers -> BOTH are
        # listed and cross-flagged mutually_exclusive; the human/3.2 picks.
        with tempfile.TemporaryDirectory() as tmp:
            table = {"tables": [{"phase": "decode", "pattern_id": "P0",
                                 "rows": [{"row_id": "q", "provider": "aiter"}]}]}
            cands = {"candidates": [
                {"candidate_id": "big", "phase": "decode", "pattern_id": "P0",
                 "family": "collective_norm_quant", "implementation_class":
                 "existing_flag_or_env", "readiness": "ready_for_api_validation",
                 "exact_kernel_status": "yes", "removable_row_ids": ["q"],
                 "live_call_seam": "--enable-aiter-allreduce-fusion",
                 "existing_apis": [{"name": "fused_ar_rmsnorm_quant"}]},
                {"candidate_id": "small", "phase": "decode", "pattern_id": "P0",
                 "family": "norm_quant", "implementation_class":
                 "existing_api_needs_adapter", "readiness":
                 "ready_for_api_validation", "exact_kernel_status": "yes",
                 "removable_row_ids": ["q"],
                 "live_call_seam": "rmsnorm.py:1",
                 "existing_apis": [{"name": "add_rmsnorm_quant"}]}]}
            val = {"metrics": {
                "phase_total_forward_us": {"decode": 1000.0},
                "candidate_savings": [
                    {"candidate_id": "big", "estimate_us": 50.0,
                     "stack_estimate_us": 500.0, "ceiling_count": 10,
                     "basis": "roofline"},
                    {"candidate_id": "small", "estimate_us": 5.0,
                     "stack_estimate_us": 50.0, "ceiling_count": 10,
                     "basis": "roofline"}]}}
            result, actions, _ = topk.rank(
                self._write(tmp, "c.json", cands),
                self._write(tmp, "v.json", val),
                self._write(tmp, "t.json", table), 10)
            tiers = sorted(a["tier"] for a in actions)
            # different-tier mutually-exclusive options are BOTH listed (a
            # partial-cheap A and a fuller-costlier B are a tradeoff, not a dedup)
            self.assertEqual(len(actions), 2)
            self.assertEqual(tiers, ["A", "B"])
            # and both are flagged mutually exclusive (share the removable row q)
            self.assertTrue(all(a["mutually_exclusive_with"] for a in actions))

    def test_renders_action_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_md = os.path.join(tmp, "topk.md")
            out_json = os.path.join(tmp, "topk.json")
            topk.run(
                self._write(tmp, "c.json", self._candidates()),
                self._write(tmp, "v.json", self._validation()),
                self._write(tmp, "t.json", self._table()),
                out_md, out_json, 10)
            with open(out_md) as fh:
                report = fh.read()
            self.assertIn("优先行动（集成什么）", report)
            self.assertIn("对应 Kernel / API", report)
            self.assertIn("现成算子", report)
            self.assertIn("C 类（无现成算子", report)

    # ---- the board is a binding execution list, not advice ---------------

    def test_execution_list_names_the_candidates_behind_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, actions, _ = self._run(tmp)
            self.assertEqual(
                len(result["execution_list"]), len(actions))
            for entry, action in zip(result["execution_list"], actions):
                self.assertTrue(entry["candidate_ids"])
                self.assertEqual(entry["candidate_ids"],
                                 action["candidate_ids"])
                self.assertEqual(
                    entry["required_disposition"],
                    ["applied", "blocked", "deferred_with_reason"])

    def test_truncation_is_recorded_not_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates = self._candidates()
            candidates["candidates"].append({
                "candidate_id": "dc_nq", "phase": "decode", "pattern_id": "P0",
                "family": "norm_quant",
                "implementation_class": "existing_api_needs_adapter",
                "readiness": "ready_for_api_validation",
                "exact_kernel_status": "yes", "removable_row_ids": ["dn2"],
                "existing_apis": [{"name": "rmsnorm_quant"}]})
            validation = self._validation()
            validation["metrics"]["candidate_savings"].append(
                {"candidate_id": "dc_nq", "estimate_us": 1.0,
                 "stack_estimate_us": 10.0, "basis": "roofline"})
            result, actions, _ = topk.rank(
                self._write(tmp, "c.json", candidates),
                self._write(tmp, "v.json", validation),
                self._write(tmp, "t.json", self._table()), 1)
            self.assertEqual(len(actions), 1)
            self.assertEqual(result["truncated_count"], 1)
            self.assertTrue(result["truncated_actions"])
            self.assertEqual(result["candidate_total"], 4)
            self.assertIn("截断 1 条", topk.render_markdown(result, actions))

    def test_board_carries_the_coverage_it_was_built_on(self):
        # A ranking of an incomplete surface has to say so on the same page.
        with tempfile.TemporaryDirectory() as tmp:
            validation = self._validation()
            validation["phase_coverage"] = {
                "available": True,
                "shape_resolution_by_phase": {
                    "decode": {"rows": 64, "resolved": 0,
                               "resolved_fraction": 0.0}},
                "decode_evidence": "sequence_only_shapes_unresolved",
                "decode_requires_graph_capture": True,
                "problems": ["phase 'decode' resolved 0/64 row shapes"],
                "waiver": "known gap", "ok": True}
            validation["region_coverage"] = {
                "regions_total": 22, "covered": 20, "deferred": 0,
                "uncovered": 2}
            result, actions, _ = topk.rank(
                self._write(tmp, "c.json", self._candidates()),
                self._write(tmp, "v.json", validation),
                self._write(tmp, "t.json", self._table()), 10)
            self.assertEqual(result["region_coverage"]["uncovered"], 2)
            md = topk.render_markdown(result, actions)
            self.assertIn("未覆盖 2", md)
            self.assertIn("resolved 0/64", md)
            self.assertIn("known gap", md)

    def test_pairwise_conflicts_do_not_collapse_into_choose_one(self):
        # A conflicts with B and B with C, but A and C are compatible. Calling
        # that a "pick exactly one" group would forbid a legal combination --
        # the same harm as dropping a candidate, dressed as caution.
        with tempfile.TemporaryDirectory() as tmp:
            table = {"tables": [{"phase": "decode", "pattern_id": "P0",
                                 "rows": [{"row_id": r, "provider": "aiter"}
                                          for r in ("r1", "r2", "r3")]}]}
            def cand(cid, rows):
                return {"candidate_id": cid, "phase": "decode",
                        "pattern_id": "P0", "family": cid,
                        "implementation_class": "existing_api_needs_adapter",
                        "readiness": "ready_for_api_validation",
                        "exact_kernel_status": "yes",
                        "removable_row_ids": rows,
                        "existing_apis": [{"name": "k_" + cid}]}
            candidates = {"candidates": [
                cand("a", ["r1"]), cand("b", ["r1", "r2"]),
                cand("c", ["r2"])]}
            validation = {"metrics": {
                "phase_total_forward_us": {"decode": 1000.0},
                "collective_guard_checks": [],
                "candidate_savings": [
                    {"candidate_id": "a", "estimate_us": 3.0,
                     "stack_estimate_us": 30.0, "basis": "roofline"},
                    {"candidate_id": "b", "estimate_us": 2.0,
                     "stack_estimate_us": 20.0, "basis": "roofline"},
                    {"candidate_id": "c", "estimate_us": 1.0,
                     "stack_estimate_us": 10.0, "basis": "roofline"}]}}
            result, actions, _ = topk.rank(
                self._write(tmp, "c.json", candidates),
                self._write(tmp, "v.json", validation),
                self._write(tmp, "t.json", table), 10)
            groups = result["exclusive_groups"]
            self.assertEqual(len(groups), 1)
            group = groups[0]
            self.assertEqual(group["choose"], "compatible_subset")
            self.assertEqual(len(group["member_exec_ids"]), 3)
            edges = {tuple(e) for e in group["conflict_edges"]}
            self.assertEqual(len(edges), 2)
            md = topk.render_markdown(result, actions)
            self.assertIn("两两部分冲突", md)

    def test_a_true_clique_is_still_choose_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = {"tables": [{"phase": "decode", "pattern_id": "P0",
                                 "rows": [{"row_id": r, "provider": "aiter"}
                                          for r in ("r1", "r2")]}]}
            def cand(cid, rows):
                return {"candidate_id": cid, "phase": "decode",
                        "pattern_id": "P0", "family": cid,
                        "implementation_class": "existing_api_needs_adapter",
                        "readiness": "ready_for_api_validation",
                        "exact_kernel_status": "yes",
                        "removable_row_ids": rows,
                        "existing_apis": [{"name": "k_" + cid}]}
            # narrow vs broad at the same position: a真子集 of b, so the single
            # pair IS the whole group -> a genuine choose-one.
            candidates = {"candidates": [cand("a", ["r1"]),
                                         cand("b", ["r1", "r2"])]}
            validation = {"metrics": {
                "phase_total_forward_us": {"decode": 1000.0},
                "collective_guard_checks": [],
                "candidate_savings": [
                    {"candidate_id": "a", "estimate_us": 3.0,
                     "stack_estimate_us": 30.0, "basis": "roofline"},
                    {"candidate_id": "b", "estimate_us": 2.0,
                     "stack_estimate_us": 20.0, "basis": "roofline"}]}}
            result, actions, _ = topk.rank(
                self._write(tmp, "c.json", candidates),
                self._write(tmp, "v.json", validation),
                self._write(tmp, "t.json", table), 10)
            self.assertEqual(result["exclusive_groups"][0]["choose"], 1)
            self.assertIn("只能落一条", topk.render_markdown(result, actions))

    def test_same_family_different_seams_do_not_conflict(self):
        """e01/e02-style rows in different pattern positions may coexist.

        Family equality is not conflict evidence; only same phase+pattern with
        intersecting removable row ids can create an edge.
        """
        with tempfile.TemporaryDirectory() as tmp:
            table = {"tables": [
                {"phase": "decode", "pattern_id": "P_LEFT",
                 "rows": [{"row_id": "left_q", "provider": "aiter"}]},
                {"phase": "decode", "pattern_id": "P_RIGHT",
                 "rows": [{"row_id": "right_q", "provider": "aiter"}]},
            ]}
            def cand(cid, pattern, row):
                return {
                    "candidate_id": cid, "phase": "decode",
                    "pattern_id": pattern, "family": "norm_quant",
                    "implementation_class": "existing_api_needs_adapter",
                    "readiness": "ready_for_api_validation",
                    "exact_kernel_status": "yes",
                    "removable_row_ids": [row],
                    "existing_apis": [{"name": "kernel_for_" + cid}],
                }
            candidates = {"candidates": [
                cand("left", "P_LEFT", "left_q"),
                cand("right", "P_RIGHT", "right_q"),
            ]}
            validation = {"metrics": {
                "phase_total_forward_us": {"decode": 1000.0},
                "candidate_savings": [
                    {"candidate_id": "left", "estimate_us": 2.0,
                     "stack_estimate_us": 20.0},
                    {"candidate_id": "right", "estimate_us": 1.0,
                     "stack_estimate_us": 10.0},
                ]}}
            result, actions, _ = topk.rank(
                self._write(tmp, "c.json", candidates),
                self._write(tmp, "v.json", validation),
                self._write(tmp, "t.json", table), 10)
            self.assertEqual(len(actions), 2)
            self.assertEqual(result["exclusive_groups"], [])
            self.assertTrue(all(
                not action["mutually_exclusive_with"] for action in actions))


class SubsumptionLadderTest(unittest.TestCase):
    """A rung whose removable rows are a strict SUBSET of another surviving
    row's is covered by that row's unit-side microbench, and must say so on the
    execution list. On DSR1 2026-09-03 two rungs of one ladder each paid for
    their own slots (8 of 10) and the ★★★-prior fusion below them never got
    benched at all."""

    def _write(self, root, name, value):
        path = os.path.join(root, name)
        with open(path, "w") as fh:
            json.dump(value, fh)
        return path

    def _ladder(self, tmp, top_k=10):
        table = {"tables": [{"phase": "decode", "pattern_id": "P0",
                             "rows": [{"row_id": "r1", "provider": "aiter"},
                                      {"row_id": "r2", "provider": "aiter"},
                                      {"row_id": "r3", "provider": "aiter"},
                                      {"row_id": "z1", "provider": "aiter"}]}]}
        def cand(cid, family, rows, api):
            return {"candidate_id": cid, "phase": "decode", "pattern_id": "P0",
                    "family": family,
                    "implementation_class": "existing_api_needs_adapter",
                    "readiness": "ready_for_api_validation",
                    "exact_kernel_status": "yes", "removable_row_ids": rows,
                    "live_call_seam": "x.py:1",
                    "existing_apis": [{"name": api}]}
        cands = {"candidates": [
            # AR+norm+quant (widest) > AR+norm > AR : one ladder, three rungs
            cand("wide", "collective_norm_quant", ["r1", "r2", "r3"],
                 "fused_ar_rmsnorm_quant"),
            cand("mid", "collective_norm", ["r1", "r2"], "fused_ar_rmsnorm"),
            cand("narrow", "collective", ["r1"], "fused_allreduce"),
            # unrelated rows -> its own ladder top, never subsumed
            cand("alone", "layout", ["z1"], "flatten_quant"),
        ]}
        val = {"metrics": {
            "phase_total_forward_us": {"decode": 1000.0},
            "candidate_savings": [
                {"candidate_id": "wide", "estimate_us": 50.0,
                 "stack_estimate_us": 500.0, "basis": "roofline"},
                {"candidate_id": "mid", "estimate_us": 30.0,
                 "stack_estimate_us": 300.0, "basis": "roofline"},
                {"candidate_id": "narrow", "estimate_us": 20.0,
                 "stack_estimate_us": 200.0, "basis": "roofline"},
                {"candidate_id": "alone", "estimate_us": 10.0,
                 "stack_estimate_us": 100.0, "basis": "roofline"}]}}
        return topk.rank(
            self._write(tmp, "c.json", cands),
            self._write(tmp, "v.json", val),
            self._write(tmp, "t.json", table), top_k)

    def _by_candidate(self, result):
        out = {}
        for entry in result["execution_list"]:
            for cid in entry["candidate_ids"]:
                out[cid] = entry
        return out

    def test_narrower_rungs_point_at_the_widest_and_cost_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _actions, _ = self._ladder(tmp)
            rows = self._by_candidate(result)
            wide, mid, narrow = rows["wide"], rows["mid"], rows["narrow"]
            # one rung at a time: narrow -> mid -> wide, so apply-back descends
            # a single step instead of jumping the whole ladder.
            self.assertEqual(narrow["subsumed_by"], mid["exec_id"])
            self.assertEqual(mid["subsumed_by"], wide["exec_id"])
            self.assertIsNone(wide["subsumed_by"])
            # ...but cost is charged to the TOP of the chain, not the next step.
            self.assertEqual(narrow["ladder_top"], wide["exec_id"])
            self.assertEqual(mid["ladder_top"], wide["exec_id"])
            self.assertIsNone(wide["ladder_top"])
            self.assertEqual([wide["unit_cost"], mid["unit_cost"],
                              narrow["unit_cost"]], [1, 0, 0])

    def test_an_unrelated_row_is_its_own_ladder_top(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _actions, _ = self._ladder(tmp)
            alone = self._by_candidate(result)["alone"]
            self.assertIsNone(alone["subsumed_by"])
            self.assertIsNone(alone["ladder_top"])
            self.assertEqual(alone["unit_cost"], 1)
            self.assertEqual(alone["subsumes"], [])

    def test_same_recipe_uses_one_unit_representative(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = {"tables": [{
                "phase": "decode", "pattern_id": "P0",
                "rows": [
                    {"row_id": "a1", "provider": "aiter"},
                    {"row_id": "a2", "provider": "aiter"},
                    {"row_id": "b1", "provider": "aiter"},
                    {"row_id": "b2", "provider": "aiter"},
                ]}]}
            def cand(cid, rows):
                return {
                    "candidate_id": cid, "phase": "decode",
                    "pattern_id": "P0", "family": "norm_quant",
                    "implementation_class": "existing_api_needs_adapter",
                    "readiness": "ready_for_api_validation",
                    "exact_kernel_status": "yes",
                    "members": [{"row_id": rid} for rid in rows],
                    "removable_row_ids": [rows[-1]],
                    "existing_apis": [{"name": "fused_norm_quant"}],
                }
            cands = {"candidates": [
                cand("nq_z", ["a1", "a2"]),
                cand("nq_a", ["b1", "b2"]),
            ]}
            val = {"metrics": {
                "phase_total_forward_us": {"decode": 1000.0},
                "candidate_savings": [
                    {"candidate_id": "nq_z", "estimate_us": 5.0,
                     "stack_estimate_us": 50.0, "basis": "roofline"},
                    {"candidate_id": "nq_a", "estimate_us": 4.0,
                     "stack_estimate_us": 40.0, "basis": "roofline"},
                ]}}
            result, _actions, _ = topk.rank(
                self._write(tmp, "c.json", cands),
                self._write(tmp, "v.json", val),
                self._write(tmp, "t.json", table), 10)
            self.assertEqual(len(result["execution_list"]), 1)
            entry = result["execution_list"][0]
            # The highest-benefit occurrence is representative even though its
            # id sorts after the equivalent occurrence.
            self.assertEqual(entry["unit_representative_candidate_id"], "nq_z")
            self.assertEqual(entry["unit_equivalent_candidate_ids"], ["nq_a"])

    def test_child_and_standalone_occurrences_of_one_recipe_are_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = {"tables": [{
                "phase": "decode", "pattern_id": "P0",
                "rows": [{"row_id": rid, "provider": "aiter"}
                         for rid in ("ar", "n1", "q1", "n2", "q2")]}]}
            def cand(cid, family, rows, removable, api):
                return {
                    "candidate_id": cid, "phase": "decode",
                    "pattern_id": "P0", "family": family,
                    "implementation_class": "existing_api_needs_adapter",
                    "readiness": "ready_for_api_validation",
                    "exact_kernel_status": "yes",
                    "members": [{"row_id": rid} for rid in rows],
                    "removable_row_ids": removable,
                    "existing_apis": [{"name": api}],
                }
            cands = {"candidates": [
                cand("wide", "collective_norm_quant", ["ar", "n1", "q1"],
                     ["n1", "q1"], "fused_ar_nq"),
                cand("nq_child", "norm_quant", ["n1", "q1"],
                     ["q1"], "fused_nq"),
                cand("nq_standalone", "norm_quant", ["n2", "q2"],
                     ["q2"], "fused_nq"),
            ]}
            val = {"metrics": {
                "phase_total_forward_us": {"decode": 1000.0},
                "candidate_savings": [
                    {"candidate_id": "wide", "estimate_us": 9.0,
                     "stack_estimate_us": 90.0, "basis": "roofline"},
                    {"candidate_id": "nq_child", "estimate_us": 5.0,
                     "stack_estimate_us": 50.0, "basis": "roofline"},
                    {"candidate_id": "nq_standalone", "estimate_us": 4.0,
                     "stack_estimate_us": 40.0, "basis": "roofline"},
                ]}}
            result, _actions, _ = topk.rank(
                self._write(tmp, "c.json", cands),
                self._write(tmp, "v.json", val),
                self._write(tmp, "t.json", table), 10)
            rows = self._by_candidate(result)
            self.assertNotEqual(rows["nq_child"]["exec_id"],
                                rows["nq_standalone"]["exec_id"])
            self.assertEqual(rows["nq_child"]["ladder_top"],
                             rows["wide"]["exec_id"])
            self.assertIsNone(rows["nq_standalone"]["ladder_top"])

    def test_every_rung_stays_on_the_board(self):
        # Subsumption changes who PAYS for the microbench. It must never drop a
        # row: the cheap/partial vs costly/fuller tradeoff is the reader's call.
        with tempfile.TemporaryDirectory() as tmp:
            result, actions, _ = self._ladder(tmp)
            self.assertEqual(len(result["execution_list"]), len(actions))
            self.assertEqual(
                set(self._by_candidate(result)),
                {"wide", "mid", "narrow", "alone"})

    def test_subsumes_is_the_inverse_of_subsumed_by(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _actions, _ = self._ladder(tmp)
            rows = self._by_candidate(result)
            self.assertEqual(rows["wide"]["subsumes"], [rows["mid"]["exec_id"]])
            self.assertEqual(rows["mid"]["subsumes"],
                             [rows["narrow"]["exec_id"]])
            self.assertEqual(rows["narrow"]["subsumes"], [])

    def test_board_renders_the_ladder(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_md = os.path.join(tmp, "topk.md")
            result, _actions, _ = self._ladder(tmp)
            with open(out_md, "w") as fh:
                fh.write(topk.render_markdown(result, _actions))
            with open(out_md) as fh:
                report = fh.read()
            self.assertIn("阶梯", report)
            self.assertIn("单侧由其覆盖", report)
            # and the asymmetry is stated where the reader sees it
            self.assertIn("apply-back 不做这种剪枝", report)


if __name__ == "__main__":
    unittest.main()
