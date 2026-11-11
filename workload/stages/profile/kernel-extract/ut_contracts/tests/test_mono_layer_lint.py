"""CPU tests for the mono_layer contract: oracle references in the family schema, the reduced set
of gated outputs, and compare_one's handling of a subject that left a poisoned (non-finite)
value where the compared-against output is finite.

Run: python3 workload/stages/profile/kernel-extract/ut_contracts/tests/test_mono_layer_lint.py
"""

import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = os.path.dirname(HERE)
sys.path.insert(0, UTC)
import ut_contract as U  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

needs_torch = unittest.skipIf(torch is None, "torch is not installed")

TASK_AXES = {"hidden": 16, "world": 2, "launches": 2, "pools": 2, "kv_width": 4, "topk": 2}


def _family():
    return U.load_family("mono_layer")


def _broken(mutate):
    doc = json.loads(json.dumps(_family()))
    mutate(doc)
    return doc


class MonoLayerFamilyTest(unittest.TestCase):
    def test_reference_is_the_split_oracle(self):
        doc = _family()
        self.assertEqual(doc["reference"], "oracle:split")
        self.assertEqual(U.reference_oracle(doc), "split")
        self.assertIsNone(U.f64_reference(doc))
        self.assertIn("native", doc["oracles"])
        native = doc["oracles"]["native"]
        self.assertIs(native["controls"], False)
        self.assertEqual(native["cases"], "per_width_smallest")
        self.assertTrue(all(b["kind"] == "sqnr" for b in native["bounds"].values()))

    def test_gated_outputs_and_their_bounds(self):
        out = _family()["outputs"]
        self.assertEqual(set(out), {"stream_out", "attn_out", "layer_out", "kv_rows", "kv_stray", "ids"})
        for name in ("stream_out", "attn_out", "layer_out"):
            self.assertEqual(out[name]["bound"], {"kind": "tolerance", "atol": 1 / 128, "rtol": 1 / 128})
        kv = out["kv_rows"]
        self.assertEqual((kv["bound"]["atol"], kv["bound"]["rtol"]), (1 / 64, 1 / 8))
        self.assertTrue(kv["bound"]["per_row"])
        self.assertEqual(kv["max_violation_fraction"], 0.0)
        self.assertEqual(kv["shape"][2], "pools")
        self.assertEqual(out["kv_stray"]["bound"], {"kind": "exact", "value": 0})
        self.assertEqual(out["ids"]["bound"]["kind"], "exact")
        self.assertEqual(out["ids"]["optional_if_not_produced"], {"poison": -7})

    def test_family_carries_no_task_specific_fields(self):
        doc = _family()
        for key in ("replay_bound", "task_split", "widths"):
            self.assertNotIn(key, doc)
        for o in doc["oracles"].values():
            self.assertNotIn("env", o)

    def test_ids_present_only_for_routed_moe(self):
        doc = _family()
        moe = U.freeze_task_contract(doc, attrs={"layer_kind": "routed_moe"}, const_axes=TASK_AXES)
        sym = U.symbols_for(moe, {"rows": 3, "context": 8})
        self.assertTrue(U.present(moe["outputs"]["ids"], sym))
        self.assertEqual(U.shape_of(moe["outputs"]["ids"], sym), (2, 2, 3, 2))
        self.assertEqual(U.shape_of(moe["outputs"]["kv_rows"], sym), (2, 2, 2, 2, 3, 4))
        dense_axes = {k: v for k, v in TASK_AXES.items() if k != "topk"}
        dense = U.freeze_task_contract(doc, attrs={"layer_kind": "dense"}, const_axes=dense_axes)
        sym = U.symbols_for(dense, {"rows": 3, "context": 8})
        self.assertFalse(U.present(dense["outputs"]["ids"], sym))
        names = [n for n, s in dense["outputs"].items() if U.present(s, sym)]
        self.assertNotIn("ids", names)

    def test_control_categories_follow_layer_kind(self):
        doc = _family()
        self.assertIn("routing_ids", U.control_categories(doc, {"layer_kind": "routed_moe"}))
        self.assertNotIn("routing_ids", U.control_categories(doc, {"layer_kind": "dense"}))
        self.assertIn("norm_weight", U.control_categories(doc, {"layer_kind": "dense"}))
        for kind in ("routed_moe", "dense"):
            self.assertTrue({"softmax_scale", "peer_partial", "kv_slot"}
                            <= set(U.control_categories(doc, {"layer_kind": kind})))

    def test_oracle_doc_narrows_outputs_and_swaps_bounds(self):
        doc = U.freeze_task_contract(_family(), attrs={}, const_axes=TASK_AXES)
        nd = U.oracle_doc(doc, "native")
        self.assertEqual(set(nd["outputs"]), {"stream_out", "attn_out", "layer_out", "kv_rows"})
        self.assertEqual(nd["outputs"]["kv_rows"]["bound"], {"kind": "sqnr", "min_db": 20})
        sd = U.oracle_doc(doc, "split")
        self.assertEqual(sd["outputs"], doc["outputs"])
        self.assertEqual(doc["outputs"]["kv_rows"]["bound"]["kind"], "tolerance")   # not mutated


class OracleSchemaTest(unittest.TestCase):
    def assertRejects(self, mutate, needle):
        with self.assertRaises(U.ContractError) as cm:
            U.validate_family(_broken(mutate))
        self.assertIn(needle, str(cm.exception))

    def test_unknown_oracle_name(self):
        self.assertRejects(lambda d: d.__setitem__("reference", "oracle:nope"), "names no oracle")

    def test_illegal_bound_kind(self):
        self.assertRejects(lambda d: d["oracles"]["native"]["bounds"].__setitem__(
            "stream_out", {"kind": "close_enough"}), "bound kind")

    def test_outputs_not_a_subset(self):
        self.assertRejects(lambda d: d["oracles"]["native"]["outputs"].append("scores"),
                           "not outputs of the contract")

    def test_bound_for_an_output_the_oracle_does_not_produce(self):
        self.assertRejects(lambda d: d["oracles"]["native"]["bounds"].__setitem__(
            "ids", {"kind": "exact"}), "does not produce")

    def test_oracle_needs_a_path(self):
        self.assertRejects(lambda d: d["oracles"]["split"].pop("path"), "needs a path")

    def test_oracle_cases_and_controls(self):
        self.assertRejects(lambda d: d["oracles"]["native"].__setitem__("cases", "some"), "cases")
        self.assertRejects(lambda d: d["oracles"]["native"].__setitem__("controls", "no"), "controls")

    def test_oracle_reference_satisfies_bounds_needing_a_reference(self):
        def mutate(d):
            d["outputs"]["stream_out"]["bound"] = {"kind": "output_rounding"}
        U.validate_family(_broken(mutate))
        def mutate_none(d):
            mutate(d)
            d["reference"] = None
        with self.assertRaises(U.ContractError):
            U.validate_family(_broken(mutate_none))

    def test_optional_if_not_produced_shape(self):
        self.assertRejects(lambda d: d["outputs"]["ids"].__setitem__("optional_if_not_produced", {}),
                           "poison")
        self.assertRejects(lambda d: d["outputs"]["stream_out"].__setitem__(
            "optional_if_not_produced", {"poison": 0}), "exact outputs")

    def test_control_categories_keyed_by_attr_values(self):
        self.assertRejects(lambda d: d["required_control_categories"].__setitem__("by", "nope"), "not an attr")
        self.assertRejects(lambda d: d["required_control_categories"]["cases"].__setitem__("sparse", ["x"]),
                           "not in")


@needs_torch
class NonFiniteSubjectTest(unittest.TestCase):
    """A poisoned buffer the subject never wrote reads NaN; it must fail wherever the
    compared-against value is finite, and NaN on both sides stays agreement."""

    def setUp(self):
        self.y = torch.linspace(-2, 2, 64, dtype=torch.float32).reshape(4, 16)
        self.bad = self.y.clone()
        self.bad[1, 3] = float("nan")

    def test_tolerance_fails_with_inf_error(self):
        b = {"kind": "tolerance", "atol": 1 / 128, "rtol": 1 / 128}
        r = U.compare_one(b, self.bad, self.y)
        self.assertFalse(r["ok"])
        self.assertEqual(r["nonfinite_mismatches"], 1)
        self.assertTrue(math.isinf(r["max_abs_err"]))
        self.assertGreater(r["violation_fraction"], 0)
        # against a float64 reference too, where the bias check used to read rms=NaN as unbiased
        r = U.compare_one(b, self.bad, self.y, ref=self.y.double())
        self.assertFalse(r["ok"])
        self.assertFalse(r["unbiased"])
        # a budget does not let it through: the non-finite count is not budgeted
        r = U.compare_one(b, self.bad, self.y, max_violation_fraction=0.5)
        self.assertFalse(r["ok"])

    def test_inf_subject_fails(self):
        bad = self.y.clone()
        bad[0, 0] = float("inf")
        self.assertFalse(U.compare_one({"kind": "tolerance", "atol": 1.0, "rtol": 1.0}, bad, self.y)["ok"])

    def test_noise_relative_fails(self):
        b = {"kind": "noise_relative", "m": 4.0, "k": 4.0}
        self.assertFalse(U.compare_one(b, self.bad, self.y, base_noise=self.y)["ok"])

    def test_sqnr_fails(self):
        r = U.compare_one({"kind": "sqnr", "min_db": 20}, self.bad, self.y)
        self.assertFalse(r["ok"])
        self.assertEqual(r["sqnr_db"], -math.inf)
        r = U.compare_one({"kind": "sqnr", "min_db": 20}, self.bad, self.y, ref=self.y.double())
        self.assertFalse(r["ok"])
        self.assertEqual(r["sqnr_db"], -math.inf)

    def test_exact_fails(self):
        self.assertFalse(U.compare_one({"kind": "exact"}, self.bad, self.y)["ok"])

    def test_nan_on_both_sides_is_agreement(self):
        both = self.bad.clone()
        for b in ({"kind": "tolerance", "atol": 1 / 128, "rtol": 1 / 128}, {"kind": "sqnr", "min_db": 20},
                  {"kind": "exact"}):
            r = U.compare_one(b, both, self.bad)
            self.assertTrue(r["ok"], (b, r))
        r = U.compare_one({"kind": "tolerance", "atol": 1 / 128, "rtol": 1 / 128}, both, self.bad)
        self.assertEqual(r["nonfinite_mismatches"], 0)

    def test_finite_subject_against_nan_still_fails(self):
        self.assertFalse(U.compare_one({"kind": "tolerance", "atol": 1.0, "rtol": 1.0}, self.y, self.bad)["ok"])

    def test_clean_compare_unchanged(self):
        near = self.y + 1e-4
        r = U.compare_one({"kind": "tolerance", "atol": 1 / 128, "rtol": 1 / 128}, near, self.y)
        self.assertTrue(r["ok"])
        self.assertEqual(r["nonfinite_mismatches"], 0)
        self.assertTrue(U.compare_one({"kind": "sqnr", "min_db": 20}, near, self.y)["ok"])


@needs_torch
class NoiseRelativeKeysTest(unittest.TestCase):
    def test_m_and_k_are_read(self):
        b = torch.ones(1, 8, dtype=torch.float32)
        noise = b.clone()
        noise[0, 0] += 0.01                      # run-to-run noise of 0.01 on one element
        c = b.clone()
        c[0, 0] += 0.03                          # 3x the noise
        loose = U.compare_one({"kind": "noise_relative", "m": 4.0, "k": 0.0}, c, b, base_noise=noise)
        tight = U.compare_one({"kind": "noise_relative", "m": 2.0, "k": 0.0}, c, b, base_noise=noise)
        self.assertTrue(loose["ok"])
        self.assertFalse(tight["ok"])
        # the long names still work when the short ones are absent
        old = U.compare_one({"kind": "noise_relative", "multiple": 2.0, "rounding_multiple": 0.0}, c, b,
                            base_noise=noise)
        self.assertFalse(old["ok"])
        # k is the rounding floor: a large k admits the deviation even with m tiny
        floor = U.compare_one({"kind": "noise_relative", "m": 0.0, "k": 2.0 ** 20}, c, b, base_noise=noise)
        self.assertTrue(floor["ok"])


@needs_torch
class MonoOutputsCompareTest(unittest.TestCase):
    """compare_outputs over a frozen mono_layer contract (the subject against the split oracle)."""

    def setUp(self):
        self.doc = U.freeze_task_contract(_family(), attrs={"layer_kind": "routed_moe"},
                                          const_axes=TASK_AXES)
        self.sym = U.symbols_for(self.doc, {"rows": 3, "context": 8})
        g = torch.Generator().manual_seed(0)
        self.ref = {}
        for name, s in self.doc["outputs"].items():
            shape = U.shape_of(s, self.sym)
            dt = U.dtype_of(s, self.sym)
            if name == "ids":
                t = torch.randint(0, 32, shape, generator=g, dtype=torch.int32)
            elif name == "kv_stray":
                t = torch.zeros(shape, dtype=torch.int64)
            else:
                t = torch.randn(shape, generator=g).to(U.torch_dtype(dt))
            self.ref[name] = t

    def _cmp(self, **changes):
        sub = {k: v.clone() for k, v in self.ref.items()}
        for k, fn in changes.items():
            fn(sub[k])
        return U.compare_outputs(self.doc, sub, self.ref, sym=self.sym, refs=self.ref)

    def test_identical_passes(self):
        res = self._cmp()
        self.assertTrue(res["ok"], res)
        self.assertEqual(set(res["outputs"]), set(self.ref))

    def test_one_kv_row_out_of_bound_fails(self):
        def bump(t):
            t[1, 0, 1, 0, 2, :] += 1.0               # one token's V row of one pool on one rank
        res = self._cmp(kv_rows=bump)
        self.assertFalse(res["ok"])
        kv = res["outputs"]["kv_rows"]
        self.assertFalse(kv["ok"])
        self.assertEqual(kv["violating_rows"], 1)
        self.assertEqual(kv["rows"], 2 * 2 * 2 * 2 * 3)

    def test_one_kv_element_one_code_off_passes(self):
        def nudge(t):
            t[0, 0, 0, 0, 0, 0] = t[0, 0, 0, 0, 0, 0] * (1 + 1 / 16)
        self.assertTrue(self._cmp(kv_rows=nudge)["ok"])

    def test_unwritten_kv_row_fails(self):
        def poison(t):
            t[0, 1, 0, 1, 1, :] = float("nan")
        res = self._cmp(kv_rows=poison)
        self.assertFalse(res["outputs"]["kv_rows"]["ok"])
        self.assertEqual(res["outputs"]["kv_rows"]["violating_rows"], 1)

    def test_stray_write_fails(self):
        def stray(t):
            t[1, 0, 1] = 16
        self.assertFalse(self._cmp(kv_stray=stray)["outputs"]["kv_stray"]["ok"])

    def test_stray_count_must_be_zero_even_if_the_oracle_agrees(self):
        sub = {k: v.clone() for k, v in self.ref.items()}
        sub["kv_stray"][0, 0, 0] = 3
        ref = dict(self.ref, kv_stray=sub["kv_stray"].clone())
        res = U.compare_outputs(self.doc, sub, ref, sym=self.sym, refs=ref)
        self.assertFalse(res["outputs"]["kv_stray"]["ok"])

    def test_ids_all_poison_is_not_produced(self):
        res = self._cmp(ids=lambda t: t.fill_(-7))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["outputs"]["ids"]["status"], "not_produced")

    def test_ids_partly_poison_fails(self):
        def half(t):
            t[0].fill_(-7)
        res = self._cmp(ids=half)
        self.assertFalse(res["ok"])
        self.assertEqual(res["outputs"]["ids"]["status"], "partially_written")

    def test_ids_permuted_fails(self):
        def perm(t):
            t.copy_((t + 1) % 32)
        res = self._cmp(ids=perm)
        self.assertFalse(res["outputs"]["ids"]["ok"])

    def test_oracle_that_produced_no_ids_fails(self):
        ref = dict(self.ref, ids=torch.full_like(self.ref["ids"], -7))
        res = U.compare_outputs(self.doc, dict(self.ref), ref, sym=self.sym, refs=ref)
        self.assertFalse(res["outputs"]["ids"]["ok"])
        self.assertEqual(res["outputs"]["ids"]["status"], "reference_not_produced")

    def test_stream_one_ulp_passes_and_nan_fails(self):
        self.assertTrue(self._cmp(stream_out=lambda t: t.mul_(1 + 2 ** -8))["ok"])
        def nan(t):
            t[0, 0, 0] = float("nan")
        self.assertFalse(self._cmp(stream_out=nan)["outputs"]["stream_out"]["ok"])

    def test_dense_has_no_ids(self):
        axes = {k: v for k, v in TASK_AXES.items() if k != "topk"}
        doc = U.freeze_task_contract(_family(), attrs={"layer_kind": "dense"}, const_axes=axes)
        sym = U.symbols_for(doc, {"rows": 3, "context": 8})
        ref = {k: v for k, v in self.ref.items() if k != "ids"}
        res = U.compare_outputs(doc, ref, ref, sym=sym, refs=ref)
        self.assertTrue(res["ok"])
        self.assertNotIn("ids", res["outputs"])


if __name__ == "__main__":
    unittest.main()
