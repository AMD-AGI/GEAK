import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_unitside_harness as uh


class _UnitsideFixture(object):
    def _candidates(self):
        # one collective (AR+norm) with a captured [4,7168] decode member shape, and
        # one single-GPU B (norm+quant).
        return {"candidates": [
            {"candidate_id": "dc_ar", "family": "collective_norm",
             "phase": "decode", "implementation_class": "existing_flag_or_env",
             "existing_apis": [{"name": "aiter fused_allreduce_rmsnorm "
                               "(--enable-aiter-allreduce-fusion)"}],
             "members": [
                 {"stage": "communication", "shape": {"input_dims": [[4, 7168]]}},
                 {"stage": "norm", "shape": {"input_dims": [[4, 7168], [7168]]}}]},
            {"candidate_id": "dc_nq", "family": "norm_quant",
             "phase": "decode", "implementation_class": "existing_api_needs_adapter",
             "existing_apis": [{"name": "aiter add_rmsnorm_quant"}],
             "members": [
                 {"stage": "norm", "shape": {"input_dims": [[4, 7168]]}}]},
        ]}

    def _verdict(self, **over):
        v = {"candidate_id": "dc_ar", "family": "collective_norm",
             "fused_fn": "aiter fused_allreduce_rmsnorm", "tested_shape": [4, 7168],
             "dtypes": ["bf16"], "tol": 0.02, "parity": "pass",
             "ref_ms": 0.20, "cand_ms": 0.12, "isolated_speedup": 1.67,
             "engaged": True, "tp": 8,
             "ref_ops": ["all_reduce", "rms_norm"], "outside_work_removed": []}
        v.update(over)
        return v

    def _run(self, candidates, verdicts, min_speedup=1.0,
             require_coverage=False, waivers=None, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            cpath = os.path.join(tmp, "c.json")
            with open(cpath, "w") as fh:
                json.dump(candidates, fh)
            vdir = os.path.join(tmp, "verdicts")
            os.makedirs(vdir)
            for i, v in enumerate(verdicts):
                with open(os.path.join(vdir, "v%d.json" % i), "w") as fh:
                    json.dump(v, fh)
            return uh.validate(cpath, vdir, min_speedup,
                               require_coverage=require_coverage,
                               waivers=waivers, **kw)

    def _status(self, result, cid):
        for r in result["results"]:
            if r["candidate_id"] == cid:
                return r["unit_side_status"]
        return None

    def _candidates_decode_bucket(self):
        # decode collective with NO captured member dims (runtime_probe_wrapper),
        # only selected_bucket.batch_size -> provenance falls back to the token count.
        return {"candidates": [
            {"candidate_id": "dc_probe", "family": "collective_norm",
             "phase": "decode", "implementation_class": "existing_flag_or_env",
             "selected_bucket": {"phase": "decode", "batch_size": 4,
                                 "input_tokens": 0},
             "existing_apis": [{"name": "aiter fused_allreduce_rmsnorm"}],
             "members": [{"stage": "communication", "shape": {"input_dims": []}},
                         {"stage": "norm", "shape": {"input_dims": []}}]}]}


class FusionUnitsideTest(_UnitsideFixture, unittest.TestCase):
    def test_clean_pass(self):
        res = self._run(self._candidates(), [self._verdict()])
        self.assertEqual(res["status"], "pass")           # no errors
        self.assertEqual(self._status(res, "dc_ar"), "pass")
        self.assertEqual(res["counts"]["pass"], 1)

    def test_parity_fail_is_needs_diagnosis_not_fail(self):
        # A mis-fed kernel takes a wrong branch and mis-times, so `fail` (terminal
        # on the board) would close a candidate that was never actually measured.
        res = self._run(self._candidates(), [self._verdict(parity="fail")])
        self.assertEqual(self._status(res, "dc_ar"), "needs_diagnosis")
        self.assertEqual(res["status"], "fail")           # undiagnosed -> gate fails
        self.assertTrue(any("elimination record" in e for e in res["errors"]))

    def test_parity_fail_with_a_diagnosis_does_not_fail_the_gate(self):
        res = self._run(self._candidates(),
                        [self._verdict(parity="fail",
                                       parity_diagnosis="ruled out layout, fnuz/fn, "
                                                        "group_size, scale orient")])
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self._status(res, "dc_ar"), "needs_diagnosis")

    def test_no_speedup_fails(self):
        res = self._run(self._candidates(), [self._verdict(isolated_speedup=0.98)])
        self.assertEqual(self._status(res, "dc_ar"), "fail")

    def test_not_engaged_is_blocked(self):
        # collective whose fused path fell back to split -> blocked, not fail
        res = self._run(self._candidates(),
                        [self._verdict(engaged=False)])
        self.assertEqual(self._status(res, "dc_ar"), "blocked")

    def test_shape_mismatch_is_error(self):
        # tested a shape the candidate never captured -> untrustworthy -> harness error
        res = self._run(self._candidates(),
                        [self._verdict(tested_shape=[8, 8192])])
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("tested_shape" in e for e in res["errors"]))
        self.assertIsNone(self._status(res, "dc_ar"))   # no row from a bad verdict

    def test_fused_fn_not_in_apis_is_error(self):
        res = self._run(self._candidates(),
                        [self._verdict(fused_fn="some_unrelated_kernel")])
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("fused_fn" in e for e in res["errors"]))

    def test_unknown_candidate_is_error(self):
        res = self._run(self._candidates(),
                        [self._verdict(candidate_id="nope")])
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("unknown candidate" in e for e in res["errors"]))

    def test_missing_fields_is_error(self):
        res = self._run(self._candidates(), [{"candidate_id": "dc_ar"}])
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("missing fields" in e for e in res["errors"]))

    def test_decode_bucket_provenance_pass(self):
        v = self._verdict(candidate_id="dc_probe", tested_shape=[4, 7168])
        res = self._run(self._candidates_decode_bucket(), [v],
                        allow_shapeless_candidate="runtime_probe_wrapper decode")
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self._status(res, "dc_probe"), "pass")

    def test_decode_bucket_wrong_token_count_is_error(self):
        v = self._verdict(candidate_id="dc_probe", tested_shape=[16, 7168])
        res = self._run(self._candidates_decode_bucket(), [v],
                        allow_shapeless_candidate="runtime_probe_wrapper decode")
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("token count" in e for e in res["errors"]))

    def test_single_gpu_b_pass_and_render(self):
        v = {"candidate_id": "dc_nq", "family": "norm_quant",
             "fused_fn": "add_rmsnorm_quant", "tested_shape": [4, 7168],
             "parity": "pass", "isolated_speedup": 1.3, "ref_ms": 0.1,
             "cand_ms": 0.077, "engaged": True, "tp": 1,
             "ref_ops": ["rms_norm", "quant"], "outside_work_removed": []}
        res = self._run(self._candidates(), [v])
        self.assertEqual(self._status(res, "dc_nq"), "pass")
        md = uh.render_markdown(res)
        self.assertIn("单侧 Gate", md)
        self.assertIn("dc_nq", md)


class ShapeProvenanceTest(_UnitsideFixture, unittest.TestCase):
    """A gate that vanishes when its input is missing is worse than no gate.

    DSR1 2026-08-31: the candidates carried no member shapes, so `cand_shapes` was
    empty and the strict check was skipped. A microbench on [4,512] (rebuilt from
    config as input_tokens x kv_lora_rank -- no batch axis, no rope columns) was
    accepted, the kernel took a degenerate path, and the resulting parity fail plus
    distorted timing blocked two fusions worth +20.6% and +14.0% e2e."""

    def test_shapeless_candidate_is_an_error_by_default(self):
        v = self._verdict(candidate_id="dc_probe", tested_shape=[4, 7168])
        res = self._run(self._candidates_decode_bucket(), [v])
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("no captured member shape" in e for e in res["errors"]))

    def test_opt_in_restores_the_token_count_check(self):
        v = self._verdict(candidate_id="dc_probe", tested_shape=[4, 7168])
        res = self._run(self._candidates_decode_bucket(), [v],
                        allow_shapeless_candidate="probe wrapper")
        self.assertEqual(res["status"], "pass")


class StatusAxesTest(_UnitsideFixture, unittest.TestCase):
    """correctness and perf are independent; collapsing them double-counts a fault."""

    def test_win_is_pass_on_both_axes(self):
        r = self._run(self._candidates(), [self._verdict()])["results"][0]
        self.assertEqual((r["correctness_status"], r["perf_status"]), ("pass", "win"))
        self.assertEqual(r["failure_kind"], "none")

    def test_slow_but_correct_is_a_performance_failure(self):
        r = self._run(self._candidates(),
                      [self._verdict(isolated_speedup=0.98)])["results"][0]
        self.assertEqual(r["unit_side_status"], "fail")
        self.assertEqual((r["correctness_status"], r["perf_status"]),
                         ("pass", "no_win"))
        self.assertEqual(r["failure_kind"], "performance")

    def test_divergence_without_a_diagnosis_is_unmeasured_not_functional(self):
        r = self._run(self._candidates(), [self._verdict(parity="fail")])["results"][0]
        self.assertEqual((r["correctness_status"], r["perf_status"]),
                         ("fail", "undefined"))
        self.assertEqual(r["failure_kind"], "unmeasured")
        self.assertIsNone(r["isolated_speedup"])

    def test_divergence_with_a_diagnosis_is_a_functional_failure(self):
        r = self._run(self._candidates(),
                      [self._verdict(parity="fail",
                                     parity_diagnosis="ruled out fnuz/fn, layout, "
                                                      "group_size, scale orientation")
                       ])["results"][0]
        self.assertEqual(r["failure_kind"], "functional")
        self.assertEqual(r["perf_status"], "undefined")

    def test_collective_not_engaged_is_neither_kind_of_failure(self):
        r = self._run(self._candidates(), [self._verdict(engaged=False)])["results"][0]
        self.assertEqual(r["unit_side_status"], "blocked")
        self.assertEqual(r["failure_kind"], "not_engaged")

    def test_both_axes_are_rendered(self):
        res = self._run(self._candidates(),
                        [self._verdict(isolated_speedup=0.98)])
        md = uh.render_markdown(res)
        self.assertIn("correctness", md)
        self.assertIn("performance", md)


class PhaseGeneralizationTest(unittest.TestCase):
    """A kernel that wins in one phase must be ANSWERED FOR in every other phase.

    DSR1 2026-08-31: fused_qk_rope_concat_and_cache_mla passed 单侧 at 2.76x
    parity-exact -- in prefill only, because the candidate that claimed it was a
    prefill candidate. prefill is 2.48% of wall clock, so both rows were correctly
    deferred. The same kernel in decode was worth +20.61% e2e. Coverage was counted
    per CANDIDATE, every candidate had a verdict, so no cell ever went red."""

    def _cands(self):
        common = {"family": "norm_quant",
                  "implementation_class": "existing_api_needs_adapter",
                  "existing_apis": [{"name": "aiter add_rmsnorm_quant"}],
                  "members": [{"stage": "norm",
                               "shape": {"input_dims": [[4, 7168]]}}]}
        return {"candidates": [dict(common, candidate_id="d0", phase="decode"),
                               dict(common, candidate_id="p0", phase="prefill")]}

    def _verdict(self, cid, **over):
        v = {"candidate_id": cid, "family": "norm_quant",
             "fused_fn": "aiter add_rmsnorm_quant", "tested_shape": [4, 7168],
             "parity": "pass", "isolated_speedup": 1.3, "ref_ms": 0.1,
             "cand_ms": 0.077, "engaged": True, "tp": 1,
             "ref_ops": ["rms_norm", "quant"], "outside_work_removed": []}
        v.update(over)
        return v

    def _run(self, verdicts, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            cpath = os.path.join(tmp, "c.json")
            with open(cpath, "w") as fh:
                json.dump(self._cands(), fh)
            vdir = os.path.join(tmp, "verdicts")
            os.makedirs(vdir)
            for i, v in enumerate(verdicts):
                with open(os.path.join(vdir, "v%d.json" % i), "w") as fh:
                    json.dump(v, fh)
            kw.setdefault("require_coverage", False)
            return uh.validate(cpath, vdir, 1.0, **kw)

    def test_a_winner_unbenched_in_the_other_phase_is_a_gap(self):
        res = self._run([self._verdict("d0")])
        self.assertEqual(res["status"], "fail")
        gaps = res["phase_generalization"]["open_gaps"]
        self.assertEqual([(g["fused_fn"], g["phase"]) for g in gaps],
                         [("aiter add_rmsnorm_quant", "prefill")])

    def test_benching_it_in_both_phases_closes_the_gap(self):
        res = self._run([self._verdict("d0"), self._verdict("p0")])
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["phase_generalization"]["open_gaps"], [])

    def test_a_waiver_closes_the_gap_with_a_reason_on_the_record(self):
        res = self._run(
            [self._verdict("d0")],
            phase_waivers={"aiter add_rmsnorm_quant@prefill": "no norm+quant seam"})
        self.assertEqual(res["status"], "pass")

    def test_a_loser_generates_no_gap(self):
        # only a WINNER has to be answered for elsewhere
        res = self._run([self._verdict("d0", isolated_speedup=0.9)])
        self.assertEqual(res["phase_generalization"]["open_gaps"], [])

    def test_unrelated_candidate_in_other_phase_is_not_a_gap(self):
        cands = self._cands()
        cands["candidates"][1]["existing_apis"] = [
            {"name": "aiter unrelated_prefill_fusion"}]
        with tempfile.TemporaryDirectory() as tmp:
            cpath = os.path.join(tmp, "c.json")
            with open(cpath, "w") as fh:
                json.dump(cands, fh)
            vdir = os.path.join(tmp, "verdicts")
            os.makedirs(vdir)
            with open(os.path.join(vdir, "v.json"), "w") as fh:
                json.dump(self._verdict("d0"), fh)
            res = uh.validate(cpath, vdir, 1.0, require_coverage=False)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["phase_generalization"]["open_gaps"], [])


class CoverageGateTest(unittest.TestCase):
    """Recall, not precision: a candidate nobody benched must not render as done.

    Regression for DSR1 2026-08-26, where 16 of 42 candidates were microbenched and
    the report read "总计 16 条 verdict：pass 16 / fail 0 ... status=pass"."""

    def _cands(self, n_scope=3, n_tierc=0):
        cands = []
        for i in range(n_scope):
            cands.append({
                "candidate_id": "c%d" % i, "family": "norm_quant",
                "phase": "decode" if i % 2 == 0 else "prefill",
                "implementation_class": "existing_api_needs_adapter",
                "existing_apis": [{"name": "aiter add_rmsnorm_quant"}],
                "members": [{"stage": "norm",
                             "shape": {"input_dims": [[4, 7168]]}}]})
        for i in range(n_tierc):
            cands.append({
                "candidate_id": "t%d" % i, "family": "topk_shared_append",
                "phase": "decode", "implementation_class": "new_helper_kernel",
                "existing_apis": [],
                "members": [{"stage": "topk", "shape": {"input_dims": [[4, 7168]]}}]})
        return {"candidates": cands}

    def _verdict(self, cid):
        return {"candidate_id": cid, "family": "norm_quant",
                "fused_fn": "aiter add_rmsnorm_quant", "tested_shape": [4, 7168],
                "parity": "pass", "isolated_speedup": 1.3, "ref_ms": 0.1,
                "cand_ms": 0.077, "engaged": True, "tp": 1,
                "ref_ops": ["rms_norm", "quant"], "outside_work_removed": []}

    def _run(self, cands, verdicts, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            cpath = os.path.join(tmp, "c.json")
            with open(cpath, "w") as fh:
                json.dump(cands, fh)
            vdir = os.path.join(tmp, "verdicts")
            os.makedirs(vdir)
            for i, v in enumerate(verdicts):
                with open(os.path.join(vdir, "v%d.json" % i), "w") as fh:
                    json.dump(v, fh)
            return uh.validate(cpath, vdir, 1.0, **kw)

    def _status(self, res, cid):
        for r in res["results"]:
            if r["candidate_id"] == cid:
                return r["unit_side_status"]
        return None

    def test_full_coverage_passes(self):
        res = self._run(self._cands(3), [self._verdict("c%d" % i) for i in range(3)])
        self.assertEqual(res["status"], "pass")
        self.assertTrue(res["coverage"]["complete"])
        self.assertEqual(res["coverage"]["not_validated"], 0)

    def test_missing_verdict_is_not_validated_and_fails(self):
        # THE regression: 1 of 3 benched used to render as a clean pass.
        res = self._run(self._cands(3), [self._verdict("c0")])
        self.assertEqual(res["status"], "fail")
        self.assertEqual(self._status(res, "c1"), "not_validated")
        self.assertEqual(self._status(res, "c2"), "not_validated")
        self.assertEqual(res["coverage"]["in_scope"], 3)
        self.assertEqual(res["coverage"]["validated"], 1)
        self.assertEqual(res["coverage"]["not_validated"], 2)
        self.assertEqual(sorted(res["coverage"]["not_validated_ids"]), ["c1", "c2"])
        self.assertFalse(res["coverage_ok"])

    def test_gap_is_visible_in_the_markdown(self):
        res = self._run(self._cands(3), [self._verdict("c0")])
        md = uh.render_markdown(res)
        self.assertIn("覆盖率", md)
        self.assertIn("覆盖率缺口", md)
        self.assertIn("c1", md)
        self.assertIn("c2", md)

    def test_per_phase_breakdown(self):
        # c0,c2 decode / c1 prefill; bench only decode -> prefill 0/1 must show.
        res = self._run(self._cands(3),
                        [self._verdict("c0"), self._verdict("c2")])
        by_phase = res["coverage"]["by_phase"]
        self.assertEqual(by_phase["decode"]["validated"], 2)
        self.assertEqual(by_phase["prefill"]["validated"], 0)
        self.assertEqual(by_phase["prefill"]["not_validated"], 1)

    def test_waiver_with_reason_is_accepted(self):
        res = self._run(self._cands(2), [self._verdict("c0")],
                        waivers={"c1": "needs paged-KV state, deferred to round 2"},
                        require_phase_generalization=False)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self._status(res, "c1"), "waived")
        self.assertIn("paged-KV", self._reason(res, "c1"))
        self.assertEqual(res["coverage"]["waived"], 1)

    def _reason(self, res, cid):
        for r in res["results"]:
            if r["candidate_id"] == cid:
                return r["reason"]
        return ""

    def test_waiver_for_unknown_candidate_is_an_error(self):
        res = self._run(self._cands(1), [self._verdict("c0")],
                        waivers={"ghost": "typo"})
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("unknown candidate_id" in e for e in res["errors"]))

    def test_waiver_without_reason_is_rejected(self):
        with self.assertRaises(SystemExit):
            uh._parse_waivers(["c1"])
        with self.assertRaises(SystemExit):
            uh._parse_waivers(["c1="])

    def test_tier_c_is_deferred_not_missing(self):
        # no existing kernel -> nothing to microbench; counted + shown, never a gap.
        res = self._run(self._cands(1, n_tierc=2), [self._verdict("c0")])
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self._status(res, "t0"), "deferred_author")
        self.assertEqual(res["coverage"]["deferred_author"], 2)
        self.assertEqual(res["coverage"]["in_scope"], 1)
        self.assertEqual(res["coverage"]["not_validated"], 0)

    def test_allow_partial_reports_gap_without_failing(self):
        res = self._run(self._cands(3), [self._verdict("c0")],
                        require_coverage=False,
                        require_phase_generalization=False)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(res["coverage"]["not_validated"], 2)   # still reported
        self.assertIn("覆盖率缺口", uh.render_markdown(res))     # still loud

    def test_untrustworthy_verdict_is_not_also_a_coverage_gap(self):
        # a submitted-but-rejected verdict is already a loud error; it must not be
        # double-reported as a silent absence.
        bad = self._verdict("c0")
        bad["tested_shape"] = [999, 7168]
        res = self._run(self._cands(1), [bad])
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("tested_shape" in e for e in res["errors"]))
        self.assertEqual(res["coverage"]["not_validated"], 0)
        self.assertIsNone(self._status(res, "c0"))

    def test_dsr1_shape_regression(self):
        # the exact shape of the bug: 42 candidates (40 in scope + 2 tier-C),
        # 16 verdicts -> must be a LOUD fail, not "pass 16 / fail 0".
        cands = self._cands(40, n_tierc=2)
        res = self._run(cands, [self._verdict("c%d" % i) for i in range(16)])
        self.assertEqual(res["status"], "fail")
        self.assertEqual(res["coverage"]["candidates_total"], 42)
        self.assertEqual(res["coverage"]["in_scope"], 40)
        self.assertEqual(res["coverage"]["validated"], 16)
        self.assertEqual(res["coverage"]["not_validated"], 24)
        self.assertEqual(res["coverage"]["deferred_author"], 2)

    def test_topk_limits_coverage_to_execution_list_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpath = os.path.join(tmp, "c.json")
            with open(cpath, "w") as fh:
                json.dump(self._cands(3), fh)
            topk_path = os.path.join(tmp, "topk.json")
            with open(topk_path, "w") as fh:
                json.dump({"execution_list": [{
                    "exec_id": "e01", "candidate_ids": ["c0"]}]}, fh)
            vdir = os.path.join(tmp, "verdicts")
            os.makedirs(vdir)
            with open(os.path.join(vdir, "c0.json"), "w") as fh:
                json.dump(self._verdict("c0"), fh)
            res = uh.validate(
                cpath, vdir, 1.0, topk_path=topk_path,
                require_phase_generalization=False)
            self.assertEqual(res["status"], "pass")
            self.assertEqual(res["coverage"]["in_scope"], 1)
            self.assertEqual(res["coverage"]["not_validated"], 0)
            self.assertEqual(res["coverage"]["deferred_rank_budget"], 2)
            self.assertEqual(res["counts"]["deferred_rank_budget"], 2)


class BudgetVsSubsumptionTest(unittest.TestCase):
    """Running out of budget and being covered by a ladder top are OPPOSITES.

    Regression for DSR1 2026-09-03: 17 budget overruns were passed in as
    `--waive`, rendered `waived`, dropped out of the unvalidated set, and
    coverage reported complete over a board where the ★★★-prior oproj candidate
    had never been measured. 「没测」不得渲染成「测过了」."""

    _cands = CoverageGateTest._cands
    _verdict = CoverageGateTest._verdict
    _run = CoverageGateTest._run
    _status = CoverageGateTest._status
    _reason = CoverageGateTest._reason

    def test_budget_skipped_counts_as_not_covered_and_fails(self):
        res = self._run(self._cands(2), [self._verdict("c0")],
                        budget_skipped={"c1": "unit-side budget 1 exhausted"},
                        require_phase_generalization=False)
        self.assertEqual(res["status"], "fail")
        self.assertEqual(self._status(res, "c1"), "budget_skipped")
        self.assertFalse(res["coverage"]["complete"])
        self.assertEqual(res["coverage"]["budget_skipped"], 1)
        self.assertEqual(res["coverage"]["not_validated"], 1)
        self.assertIn("c1", res["coverage"]["not_validated_ids"])
        # and it is NOT quietly counted as a waiver
        self.assertEqual(res["coverage"]["waived"], 0)

    def test_a_waiver_cannot_launder_a_budget_skip(self):
        # Same candidate presented BOTH ways: the skip wins, loudly.
        res = self._run(self._cands(2), [self._verdict("c0")],
                        waivers={"c1": "past budget"},
                        budget_skipped={"c1": "budget exhausted"},
                        require_phase_generalization=False)
        self.assertEqual(self._status(res, "c1"), "budget_skipped")
        self.assertEqual(res["status"], "fail")

    def test_subsumed_pass_is_covered_without_its_own_microbench(self):
        res = self._run(self._cands(2), [self._verdict("c0")],
                        subsumed={"c1": "e01"},
                        require_phase_generalization=False)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self._status(res, "c1"), "subsumed_pass")
        self.assertIn("e01", self._reason(res, "c1"))
        self.assertTrue(res["coverage"]["complete"])
        self.assertEqual(res["coverage"]["subsumed_pass"], 1)
        self.assertEqual(res["coverage"]["not_validated"], 0)

    def test_equivalent_pass_is_covered_by_one_representative_microbench(self):
        res = self._run(self._cands(2), [self._verdict("c0")],
                        equivalent={"c1": "c0"},
                        require_phase_generalization=False)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self._status(res, "c1"), "equivalent_pass")
        self.assertEqual(res["coverage"]["validated"], 1)
        self.assertEqual(res["coverage"]["equivalent_pass"], 1)
        self.assertEqual(res["coverage"]["not_validated"], 0)

    def test_equivalent_inherits_a_failing_representative_once(self):
        losing = self._verdict("c0")
        losing["isolated_speedup"] = 0.9
        res = self._run(self._cands(2), [losing],
                        equivalent={"c1": "c0"},
                        require_phase_generalization=False)
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self._status(res, "c1"), "equivalent_fail")
        self.assertTrue(res["coverage"]["complete"])
        self.assertEqual(res["coverage"]["equivalent_covered"], 1)
        self.assertEqual(res["coverage"]["equivalent_pass"], 0)

    def test_a_row_cannot_be_both_covered_and_never_measured(self):
        res = self._run(self._cands(2), [self._verdict("c0")],
                        subsumed={"c1": "e01"},
                        budget_skipped={"c1": "budget exhausted"},
                        require_phase_generalization=False)
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("covered and --budget-skipped" in e
                            for e in res["errors"]))

    def test_unknown_ids_in_the_new_maps_are_errors(self):
        res = self._run(self._cands(1), [self._verdict("c0")],
                        subsumed={"ghost": "e01"},
                        budget_skipped={"phantom": "budget"})
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("--subsumed references unknown" in e
                            for e in res["errors"]))
        self.assertTrue(any("--budget-skipped references unknown" in e
                            for e in res["errors"]))

    def test_the_markdown_says_never_measured_not_waived(self):
        res = self._run(self._cands(2), [self._verdict("c0")],
                        budget_skipped={"c1": "unit-side budget exhausted"},
                        require_phase_generalization=False)
        md = uh.render_markdown(res)
        self.assertIn("从未测过", md)
        self.assertIn("覆盖率缺口", md)
        self.assertIn("c1", md)

    def test_subsumed_and_skipped_are_separate_columns_in_coverage(self):
        res = self._run(self._cands(3), [self._verdict("c0")],
                        subsumed={"c1": "e01"},
                        budget_skipped={"c2": "budget exhausted"},
                        require_phase_generalization=False)
        cov = res["coverage"]
        self.assertEqual((cov["validated"], cov["subsumed_pass"],
                          cov["budget_skipped"], cov["waived"]), (1, 1, 1, 0))
        self.assertEqual(cov["not_validated"], 1)


if __name__ == "__main__":
    unittest.main()
