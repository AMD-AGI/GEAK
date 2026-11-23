"""CPU tests for the mono_layer contract: oracle references in the family schema, the reduced set
of gated outputs, compare_one's handling of a subject that left a poisoned (non-finite) value where
the compared-against output is finite, the negative-control gate on such outputs
(ut_driver.numerical_control_outputs), and `scaffold_ut.py new --family mono_layer` (generic rank
harness, mono templates, control stubs).

Run: python3 workload/stages/ut/kernel-ut/ut_contracts/tests/test_mono_layer_lint.py
"""

import ast
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = os.path.dirname(HERE)
SKILL = os.path.dirname(UTC)
sys.path.insert(0, UTC)
import scaffold_ut  # noqa: E402
import ut_contract as U  # noqa: E402
import ut_driver as D  # noqa: E402

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
        self.assertEqual(set(out), {"stream_out", "attn_out", "layer_out", "stream_in", "kv_rows",
                                    "kv_stray", "ids"})
        tol = {"kind": "tolerance", "atol": 1 / 128, "rtol": 1 / 128}
        self.assertEqual(out["attn_out"]["bound"], tol)
        self.assertEqual(out["stream_out"]["bound"], dict(tol, delta={"minus": "stream_in"}))
        self.assertEqual(out["layer_out"]["bound"], dict(tol, delta={"minus": "stream_in", "along": "launches"}))
        for name in ("stream_in", "stream_out", "attn_out", "layer_out", "kv_rows"):
            self.assertIs(out[name]["require_finite"], True, name)
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
        self.assertEqual(set(nd["outputs"]), {"stream_in", "stream_out", "attn_out", "layer_out", "kv_rows"})
        self.assertEqual(nd["outputs"]["kv_rows"]["bound"], {"kind": "sqnr", "min_db": 20})
        # the native leg's stream checks are on the step's change, not the residual-dominated stream
        self.assertEqual(nd["outputs"]["stream_out"]["bound"]["delta"], {"minus": "stream_in"})
        self.assertEqual(nd["outputs"]["layer_out"]["bound"]["delta"]["along"], "launches")
        self.assertTrue(nd["outputs"]["kv_rows"]["require_finite"])     # survives the bound swap
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

    def test_oracle_reference_does_not_satisfy_bounds_needing_float64(self):
        # an oracle leg gives outputs, not float64 values / magnitudes: dot_product & co. and
        # topk_equivalent's "ref:<output>" scores need a float64 reference
        self.assertRejects(lambda d: d["outputs"]["attn_out"].__setitem__("bound", {"kind": "output_rounding"}),
                           "float64 reference")
        def topk(d):
            d["outputs"]["ids"]["bound"] = {"kind": "topk_equivalent", "scores": "ref:attn_out"}
            d["outputs"]["ids"].pop("optional_if_not_produced")
        self.assertRejects(topk, "float64 reference")
        self.assertRejects(lambda d: d["oracles"]["native"]["bounds"].__setitem__(
            "attn_out", {"kind": "baseline_relative"}), "float64 reference")

    def test_delta_schema(self):
        self.assertRejects(lambda d: d["outputs"]["stream_out"]["bound"].__setitem__(
            "delta", {"minus": "nope"}), "delta.minus")
        self.assertRejects(lambda d: d["outputs"]["stream_out"]["bound"].__setitem__(
            "delta", {"minus": "stream_out"}), "delta.minus")
        self.assertRejects(lambda d: d["outputs"]["layer_out"]["bound"]["delta"].__setitem__(
            "along", "pools"), "not an axis")
        self.assertRejects(lambda d: d["outputs"]["kv_stray"]["bound"].__setitem__(
            "delta", {"minus": "stream_in"}), "tolerance or sqnr")
        self.assertRejects(lambda d: d["oracles"]["native"]["outputs"].remove("stream_in"),
                           "not an output of the oracle")

    def test_require_finite_is_a_bool(self):
        self.assertRejects(lambda d: d["outputs"]["kv_rows"].__setitem__("require_finite", "yes"),
                           "require_finite")

    def test_optional_if_not_produced_shape(self):
        self.assertRejects(lambda d: d["outputs"]["ids"].__setitem__("optional_if_not_produced", {}),
                           "poison")
        self.assertRejects(lambda d: d["outputs"]["stream_out"].__setitem__(
            "optional_if_not_produced", {"poison": 0}), "exact outputs")

    def test_control_categories_keyed_by_attr_values(self):
        self.assertRejects(lambda d: d["required_control_categories"].__setitem__("by", "nope"), "not an attr")
        self.assertRejects(lambda d: d["required_control_categories"]["cases"].__setitem__("sparse", ["x"]),
                           "not in")
        with self.assertRaises(U.ContractError):          # a ContractError, not a KeyError
            U.validate_family(_broken(lambda d: d["attrs"]["layer_kind"].pop("values")))

    def test_control_categories_never_silently_empty(self):
        doc = _family()
        with self.assertRaises(U.ContractError):
            U.control_categories(doc, {})                              # the task pins no layer_kind
        with self.assertRaises(U.ContractError):
            U.control_categories(doc, {"layer_kind": "sparse"})        # a value with no case

    def test_min_controls_per_category_shape(self):
        self.assertEqual(_family()["min_controls_per_category"], {"kv_slot": 3})
        self.assertRejects(lambda d: d.__setitem__("min_controls_per_category", {"kv_slot": 0}), ">= 1")
        self.assertRejects(lambda d: d.__setitem__("min_controls_per_category", ["kv_slot"]), ">= 1")
        self.assertRejects(lambda d: d.__setitem__("min_controls_per_category", {"kv_slot": True}), ">= 1")


def _frozen(layer_kind="routed_moe"):
    return U.freeze_task_contract(_family(), attrs={"layer_kind": layer_kind, "stream_dtype": "float32"},
                                  const_axes=TASK_AXES)


def _ctl(category, i=0):
    return {"leaf": "kernel_src/x.py", "find": "f%d" % i, "replace": "r", "expect": "e", "category": category}


class ControlCategoryLintTest(unittest.TestCase):
    """DESIGN 3 / W6: every control names its category; the task's required categories (selected
    by its pinned layer_kind) are covered, kv_slot three times."""

    FULL = ["softmax_scale", "peer_partial", "kv_slot", "kv_slot", "kv_slot", "routing_ids"]

    def lint(self, cats, layer_kind="routed_moe"):
        return U._lint_control_categories(_frozen(layer_kind), [_ctl(c, i) for i, c in enumerate(cats)])

    def test_full_routed_moe_set_passes(self):
        self.assertEqual(self.lint(self.FULL), [])

    def test_extra_categories_are_allowed(self):
        self.assertEqual(self.lint(self.FULL + ["kv_write_dropped", "residual_gate"]), [])

    def test_missing_category_is_named(self):
        errs = self.lint([c for c in self.FULL if c != "routing_ids"])
        self.assertEqual(len(errs), 1)
        self.assertIn("routing_ids has 0", errs[0])

    def test_kv_slot_needs_three(self):
        errs = self.lint(["softmax_scale", "peer_partial", "kv_slot", "kv_slot", "routing_ids"])
        self.assertEqual(len(errs), 1)
        self.assertIn("kv_slot has 2 control(s), the contract requires 3", errs[0])

    def test_a_control_without_category_is_an_error(self):
        controls = [_ctl(c, i) for i, c in enumerate(self.FULL)] + [{"leaf": "kernel_src/y.py", "find": "z"}]
        errs = U._lint_control_categories(_frozen(), controls)
        self.assertEqual(len(errs), 1)
        self.assertIn("has no category", errs[0])

    def test_dense_needs_norm_weight_not_routing_ids(self):
        errs = self.lint(self.FULL, layer_kind="dense")
        self.assertEqual(len(errs), 1)
        self.assertIn("norm_weight has 0", errs[0])

    def test_a_family_without_required_categories_is_not_checked(self):
        doc = _frozen()
        doc.pop("required_control_categories")
        self.assertEqual(U._lint_control_categories(doc, [{"leaf": "a", "find": "b"}]), [])


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
        # compare_one alone; outputs marked require_finite fail it in compare_outputs (below)
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


STEP_RMS = 0.05     # a layer's change to an N(0,1) residual stream


def _mono_outputs(doc, sym, seed=0):
    """Outputs of one step: stream_in ~ N(0,1), each launch adds a change of rms STEP_RMS,
    stream_out = the last launch's stream."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, s in doc["outputs"].items():
        if not U.present(s, sym):
            continue
        shape = U.shape_of(s, sym)
        dt = U.torch_dtype(U.dtype_of(s, sym))
        if name == "ids":
            t = torch.randint(0, 32, shape, generator=g, dtype=torch.int32)
        elif name == "kv_stray":
            t = torch.zeros(shape, dtype=torch.int64)
        else:
            t = torch.randn(shape, generator=g).to(dt)
        out[name] = t
    stream = out["stream_in"].to(torch.float32)
    layers = []
    for _ in range(out["layer_out"].shape[1]):
        stream = stream + STEP_RMS * torch.randn(stream.shape, generator=g)
        layers.append(stream)
    out["layer_out"] = torch.stack(layers, dim=1)
    out["stream_out"] = stream.to(out["stream_out"].dtype)
    return out


@needs_torch
class MonoOutputsCompareTest(unittest.TestCase):
    """compare_outputs over a frozen mono_layer contract (the subject against the split oracle)."""

    def setUp(self):
        self.doc = U.freeze_task_contract(_family(), attrs={"layer_kind": "routed_moe"},
                                          const_axes=TASK_AXES)
        self.sym = U.symbols_for(self.doc, {"rows": 3, "context": 8})
        self.ref = _mono_outputs(self.doc, self.sym)

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
        self.assertEqual(res["outputs"]["kv_rows"]["status"], "subject_nonfinite")
        self.assertEqual(res["outputs"]["kv_rows"]["nonfinite"], 4)
        # compare_one alone (no require_finite) judges the same row a violating row
        r = U.compare_one(self.doc["outputs"]["kv_rows"]["bound"], self._poisoned(poison), self.ref["kv_rows"])
        self.assertEqual(r["violating_rows"], 1)

    def _poisoned(self, fn):
        t = self.ref["kv_rows"].clone()
        fn(t)
        return t

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

    def test_stream_one_ulp_of_the_change_passes_and_nan_fails(self):
        def ulp(t):
            t.add_((t - self.ref["stream_in"]) * 2 ** -8)
        self.assertTrue(self._cmp(stream_out=ulp)["ok"])
        def nan(t):
            t[0, 0, 0] = float("nan")
        self.assertFalse(self._cmp(stream_out=nan)["outputs"]["stream_out"]["ok"])

    def test_bf16_stream_rounding_passes(self):
        doc = U.freeze_task_contract(_family(), attrs={"layer_kind": "routed_moe", "stream_dtype": "bfloat16"},
                                     const_axes=TASK_AXES)
        sym = U.symbols_for(doc, {"rows": 3, "context": 8})
        ref = _mono_outputs(doc, sym)
        self.assertEqual(ref["stream_out"].dtype, torch.bfloat16)
        sub = dict(ref, stream_out=(ref["stream_out"].float() * (1 + 2 ** -9)).to(torch.bfloat16))
        self.assertTrue(U.compare_outputs(doc, sub, ref, sym=sym, refs=ref)["ok"])
        # the same perturbation is a real error on a float32 stream (the change is ~5% of it)
        sub32 = dict(self.ref, stream_out=self.ref["stream_out"] * (1 + 2 ** -6))
        self.assertFalse(U.compare_outputs(self.doc, sub32, self.ref, sym=self.sym, refs=self.ref)["ok"])

    def test_a_launch_that_changes_nothing_fails(self):
        # launch 1 skipped: its layer_out repeats launch 0's (the layer's change is lost)
        def skip(t):
            t[:, 1] = t[:, 0]
        self.assertFalse(self._cmp(layer_out=skip)["outputs"]["layer_out"]["ok"])

    def test_a_step_that_changes_nothing_on_the_reference_fails(self):
        ref = dict(self.ref, stream_out=self.ref["stream_in"].clone())
        res = U.compare_outputs(self.doc, dict(ref), ref, sym=self.sym, refs=ref)
        self.assertEqual(res["outputs"]["stream_out"]["status"], "reference_delta_zero")

    def test_nan_on_both_sides_fails_for_poisoned_outputs(self):
        # the legs share the harness's poisoning / gather: a mistake there leaves NaN at the same
        # place on every leg, which compare_one counts as agreement -- require_finite does not
        for name, idx in (("kv_rows", (0, 1, 0, 1, 1)), ("stream_out", (0, 0, 0)), ("layer_out", (1, 0, 2, 3))):
            ref = {k: v.clone() for k, v in self.ref.items()}
            ref[name][idx] = float("nan")
            res = U.compare_outputs(self.doc, {k: v.clone() for k, v in ref.items()}, ref, sym=self.sym, refs=ref)
            self.assertFalse(res["ok"], name)
            self.assertFalse(res["outputs"][name]["ok"], name)
            self.assertEqual(res["outputs"][name]["status"], "reference_nonfinite", name)
            # also when the oracle is passed as the baseline (no refs)
            res = U.compare_outputs(self.doc, {k: v.clone() for k, v in ref.items()}, ref, sym=self.sym)
            self.assertEqual(res["outputs"][name]["status"], "baseline_nonfinite", name)

    def test_ids_partly_poison_on_the_reference_fails_even_where_they_match(self):
        ref = dict(self.ref, ids=self.ref["ids"].clone())
        ref["ids"][0, 1].fill_(-7)
        res = U.compare_outputs(self.doc, dict(ref), ref, sym=self.sym, refs=ref)
        self.assertFalse(res["ok"])
        self.assertEqual(res["outputs"]["ids"]["status"], "reference_partially_written")

    def test_native_sqnr_is_on_the_layer_change(self):
        # a wholly wrong contribution (another random change of the same rms) keeps the stream at
        # ~26 dB of itself; judged on the change it is ~-3 dB and fails
        nd = U.oracle_doc(self.doc, "native")
        other = _mono_outputs(self.doc, self.sym, seed=1)
        sub = {k: v.clone() for k, v in self.ref.items()}
        d_out = other["stream_out"] - other["stream_in"]
        sub["stream_out"] = (self.ref["stream_in"] + d_out).to(sub["stream_out"].dtype)
        dl = other["layer_out"] - torch.cat([other["stream_in"].unsqueeze(1), other["layer_out"][:, :-1]], dim=1)
        sub["layer_out"] = self.ref["stream_in"].unsqueeze(1) + dl.cumsum(dim=1)
        whole = U.compare_one({"kind": "sqnr", "min_db": 20}, sub["stream_out"], self.ref["stream_out"])
        self.assertTrue(whole["ok"], whole)                  # the old check on the stream: passes
        res = U.compare_outputs(nd, sub, self.ref, sym=self.sym)
        self.assertFalse(res["outputs"]["stream_out"]["ok"])
        self.assertFalse(res["outputs"]["layer_out"]["ok"])
        self.assertLess(res["outputs"]["stream_out"]["sqnr_db"], 3)
        # the right change passes the native leg
        self.assertTrue(U.compare_outputs(nd, dict(self.ref), self.ref, sym=self.sym)["ok"])

    def test_dense_has_no_ids(self):
        axes = {k: v for k, v in TASK_AXES.items() if k != "topk"}
        doc = U.freeze_task_contract(_family(), attrs={"layer_kind": "dense"}, const_axes=axes)
        sym = U.symbols_for(doc, {"rows": 3, "context": 8})
        ref = {k: v for k, v in self.ref.items() if k != "ids"}
        res = U.compare_outputs(doc, ref, ref, sym=sym, refs=ref)
        self.assertTrue(res["ok"])
        self.assertNotIn("ids", res["outputs"])


# --- negative controls on non-finite outputs (ut_driver.numerical_control_outputs) ---

NF_DOC = {"outputs": {
    "stream_out": {"shape": ["n"], "dtype": "float32", "require_finite": True,
                   "bound": {"kind": "tolerance", "atol": 1e-3, "rtol": 1e-3}},
    "kv_rows": {"shape": ["n"], "dtype": "float32", "require_finite": True,
                "bound": {"kind": "tolerance", "atol": 1e-3, "rtol": 1e-3}},
    "ids": {"shape": ["n"], "dtype": "int32", "bound": {"kind": "exact"}}},
       "compare": {"determinism": "bounded", "max_violation_fraction": 0.0, "element_order": "fixed"}}
NF_T = types.SimpleNamespace(doc=NF_DOC)


@unittest.skipIf(torch is None, "needs torch")
class NonfiniteControls(unittest.TestCase):
    def compare(self, sub, ref):
        return U.compare_outputs(NF_DOC, sub, ref, sym={"n": 4}, refs=ref)

    def base(self):
        return {"stream_out": torch.ones(4), "kv_rows": torch.ones(4), "ids": torch.arange(4, dtype=torch.int32)}

    def test_a_subject_left_at_the_poison_is_a_rejection_at_infinite_margin(self):
        sub = self.base()
        sub["kv_rows"] = torch.tensor([1.0, float("nan"), 1.0, 1.0])
        res = self.compare(sub, self.base())
        self.assertEqual(res["outputs"]["kv_rows"]["status"], "subject_nonfinite")
        counted = D.numerical_control_outputs(NF_T, res["outputs"])
        self.assertEqual(sorted(counted), ["kv_rows"])
        self.assertEqual(U.control_margin({"outputs": counted}), float("inf"))

    def test_a_nonfinite_reference_is_not_a_detection(self):
        ref = self.base()
        ref["stream_out"] = torch.tensor([float("nan"), 1.0, 1.0, 1.0])
        res = self.compare(self.base(), ref)
        self.assertFalse(res["outputs"]["stream_out"]["ok"])
        self.assertNotEqual(res["outputs"]["stream_out"]["status"], "subject_nonfinite")
        self.assertEqual(D.numerical_control_outputs(NF_T, res["outputs"]), {})

    def test_numerical_and_exact_rejections_still_count(self):
        sub = self.base()
        sub["stream_out"] = torch.tensor([2.0, 1.0, 1.0, 1.0])
        sub["ids"] = torch.tensor([0, 1, 2, 0], dtype=torch.int32)
        counted = D.numerical_control_outputs(NF_T, self.compare(sub, self.base())["outputs"])
        self.assertEqual(sorted(counted), ["ids", "stream_out"])

    def test_a_passing_output_is_never_counted(self):
        self.assertEqual(D.numerical_control_outputs(NF_T, self.compare(self.base(), self.base())["outputs"]), {})


# --- scaffold_ut.py new --family mono_layer ---

AXES = ["hidden=8192", "world=8", "launches=2", "pools=2", "kv_width=128", "topk=8"]


def _new(task, *extra):
    argv = ["new", "--family", "mono_layer", "--task", task, "--target", "pkg.runner:Runner.launch",
            "--bind-module", "pkg", "--bind-file", "kernel_src/pkg/__init__.py"]
    for a in AXES:
        argv += ["--axis", a]
    with contextlib.redirect_stdout(io.StringIO()):
        scaffold_ut.main(argv + list(extra))


def _classdict(src, cls):
    """Literal class attributes of `cls` in source `src`."""
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for st in node.body:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    try:
                        out[st.targets[0].id] = ast.literal_eval(st.value)
                    except ValueError:
                        out[st.targets[0].id] = ast.unparse(st.value)
    return out


class MonoScaffold(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mono_scaffold_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.task = os.path.join(self.root, "mono")

    def read(self, name):
        with open(os.path.join(self.task, name)) as f:
            return f.read()

    def test_the_generic_harness_is_copied_byte_for_byte(self):
        _new(self.task)
        with open(os.path.join(UTC, "mono_harness.py")) as f:
            self.assertEqual(self.read("rank_harness.py"), f.read())
        self.assertFalse(os.path.exists(os.path.join(self.task, "rank_chain.py")))

    def test_rank_model_carries_the_frozen_constants(self):
        _new(self.task, "--attr", "layer_kind=routed_moe")
        attrs = _classdict(self.read("rank_model.py"), "RankModel")
        self.assertEqual((attrs["WORLD"], attrs["LAUNCHES"], attrs["POOLS"], attrs["HIDDEN"], attrs["TOP_K"],
                          attrs["KV_WIDTH"]), (8, 2, 2, 8192, 8, 128))
        self.assertEqual(attrs["STREAM_DTYPE"], "torch.float32")
        self.assertEqual(attrs["PACKAGE"], "pkg")
        doc = U.load_task_contract(self.task)
        self.assertEqual(attrs["PATHS"][0], "deployed")
        self.assertEqual(set(attrs["PATHS"][1:]), {o["path"] for o in doc["oracles"].values()})

    def test_a_dense_layer_has_no_top_k(self):
        _new(self.task, "--attr", "layer_kind=dense")
        self.assertIsNone(_classdict(self.read("rank_model.py"), "RankModel")["TOP_K"])

    def test_cases_wire_every_oracle_leg_and_compile(self):
        _new(self.task)
        src = self.read("cases.py")
        ast.parse(src)
        doc = U.load_task_contract(self.task)
        for name, o in doc["oracles"].items():
            self.assertIn("def oracle_%s(args):" % name, src)
            self.assertIn("return _step(args, %r)" % o["path"], src)
            self.assertIn("%r: oracle_%s" % (name, name), src)
        self.assertIn("CHAIN_LENGTH = %d" % doc["timing"]["chain_length"], src)
        self.assertIn("GRAPH_PHASES = ['DECODE']", src)
        self.assertNotIn("raise NotImplementedError", src)   # nothing to fill but the call-site docstring

    def test_an_override_that_drops_an_oracle_drops_its_leg(self):
        ov = os.path.join(self.root, "ov.json")
        fam = U.load_family("mono_layer")
        with open(ov, "w") as f:
            json.dump([{"path": "oracles", "value": {"split": fam["oracles"]["split"]}, "why": "native not yet"}], f)
        _new(self.task, "--override", ov)
        src = self.read("cases.py")
        self.assertIn("ORACLES = {'split': oracle_split}", src)
        self.assertNotIn("oracle_native", src)
        self.assertEqual(_classdict(self.read("rank_model.py"), "RankModel")["PATHS"], ("deployed", "split"))

    def test_graph_phases_follow_the_phase_regimes(self):
        pr = os.path.join(self.root, "pr.json")
        with open(pr, "w") as f:
            json.dump({"DECODE": {"cuda_graph": True}, "PREFILL": {"cuda_graph": False}}, f)
        _new(self.task, "--phase-regimes", pr)
        self.assertIn("GRAPH_PHASES = ['DECODE']", self.read("cases.py"))

    def test_meta_has_the_leg_env_and_rank_chain_timing(self):
        _new(self.task)
        with open(os.path.join(self.task, "meta.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["leg_env"], "rank_harness:leg_env")
        self.assertEqual(meta["timing"]["kind"], "rank_chain")

    def test_control_stubs_cover_every_required_category(self):
        _new(self.task, "--attr", "layer_kind=routed_moe")
        with open(os.path.join(self.task, "meta.json")) as f:
            controls = json.load(f)["ut_contract"]["negative_controls"]
        doc = U.load_task_contract(self.task)
        want = {c: int((doc.get("min_controls_per_category") or {}).get(c, 1))
                for c in U.control_categories(doc, doc["task"]["attrs"])}
        got = {}
        for c in controls:
            got[c["category"]] = got.get(c["category"], 0) + 1
        self.assertEqual(got, want)
        self.assertEqual(got["kv_slot"], 3)

    def test_the_lint_refuses_the_unfilled_stubs(self):
        _new(self.task)
        errs = U.lint_task(self.task)["errors"]
        self.assertTrue(any("FILL" in e or "editable leaf" in e or "not an editable leaf" in e
                            or "find" in e for e in errs), errs)

    def test_the_worked_example_implements_the_template_interface(self):
        _new(self.task)
        tmpl = ast.parse(self.read("rank_model.py"))
        with open(os.path.join(SKILL, "references", "examples", "muse_spark_target_rank_model.py")) as f:
            ex = ast.parse(f.read())

        def members(tree):
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == "RankModel":
                    return {st.name for st in node.body if isinstance(st, ast.FunctionDef)} | \
                           {st.targets[0].id for st in node.body if isinstance(st, ast.Assign)}
            return set()

        self.assertEqual(members(tmpl) - members(ex), set())


if __name__ == "__main__":
    unittest.main()
