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
            # prefill AR (exact=no B) is not actionable -> not in the list
            self.assertFalse(any(
                a["phase"] == "prefill" and "collective_norm" in a["recipe_key"]
                for a in actions))
            # C1 author-track deferred, not ranked
            self.assertEqual(result["deferred_author_count"], 1)

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


if __name__ == "__main__":
    unittest.main()
