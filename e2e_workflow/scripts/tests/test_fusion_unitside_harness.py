import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_unitside_harness as uh


class FusionUnitsideTest(unittest.TestCase):
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
             "engaged": True, "tp": 8}
        v.update(over)
        return v

    def _run(self, candidates, verdicts, min_speedup=1.0):
        with tempfile.TemporaryDirectory() as tmp:
            cpath = os.path.join(tmp, "c.json")
            with open(cpath, "w") as fh:
                json.dump(candidates, fh)
            vdir = os.path.join(tmp, "verdicts")
            os.makedirs(vdir)
            for i, v in enumerate(verdicts):
                with open(os.path.join(vdir, "v%d.json" % i), "w") as fh:
                    json.dump(v, fh)
            return uh.validate(cpath, vdir, min_speedup)

    def _status(self, result, cid):
        for r in result["results"]:
            if r["candidate_id"] == cid:
                return r["unit_side_status"]
        return None

    def test_clean_pass(self):
        res = self._run(self._candidates(), [self._verdict()])
        self.assertEqual(res["status"], "pass")           # no errors
        self.assertEqual(self._status(res, "dc_ar"), "pass")
        self.assertEqual(res["counts"]["pass"], 1)

    def test_parity_fail(self):
        res = self._run(self._candidates(), [self._verdict(parity="fail")])
        self.assertEqual(res["status"], "pass")           # verdict trustworthy
        self.assertEqual(self._status(res, "dc_ar"), "fail")

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
        self.assertEqual(len(res["results"]), 0)

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

    def test_decode_bucket_provenance_pass(self):
        v = self._verdict(candidate_id="dc_probe", tested_shape=[4, 7168])
        res = self._run(self._candidates_decode_bucket(), [v])
        self.assertEqual(res["status"], "pass")
        self.assertEqual(self._status(res, "dc_probe"), "pass")

    def test_decode_bucket_wrong_token_count_is_error(self):
        v = self._verdict(candidate_id="dc_probe", tested_shape=[16, 7168])
        res = self._run(self._candidates_decode_bucket(), [v])
        self.assertEqual(res["status"], "fail")
        self.assertTrue(any("token count" in e for e in res["errors"]))

    def test_single_gpu_b_pass_and_render(self):
        v = {"candidate_id": "dc_nq", "family": "norm_quant",
             "fused_fn": "add_rmsnorm_quant", "tested_shape": [4, 7168],
             "parity": "pass", "isolated_speedup": 1.3, "ref_ms": 0.1,
             "cand_ms": 0.077, "engaged": True, "tp": 1}
        res = self._run(self._candidates(), [v])
        self.assertEqual(self._status(res, "dc_nq"), "pass")
        md = uh.render_markdown(res)
        self.assertIn("单侧 Gate", md)
        self.assertIn("dc_nq", md)


if __name__ == "__main__":
    unittest.main()
