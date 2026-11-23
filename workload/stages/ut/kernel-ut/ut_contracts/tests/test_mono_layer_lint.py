"""CPU tests for the mono_layer contract: oracle references in the family schema, the reduced set
of gated outputs, compare_one's handling of a subject that left a poisoned (non-finite) value where
the compared-against output is finite, the negative-control gate on such outputs
(ut_driver.numerical_control_outputs), and `scaffold_ut.py new --family mono_layer` (generic rank
harness, mono templates, control stubs), the stream layouts / world axis / harness hooks beyond
Muse Spark, and the DeepSeek-V4.1-Flash models (references/mono_models.json, launch waivers and
negative controls as scaffold_tasks.py stamps them, the worked examples against the template).

Run: python3 workload/stages/ut/kernel-ut/ut_contracts/tests/test_mono_layer_lint.py
"""

import ast
import contextlib
import io
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
import types
import textwrap
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
HAVE_TORCH = torch is not None

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
                                    "kv_stray", "ids", "stream_aux_in", "stream_aux_out"})
        # the mhc stream's mixes (DeepSeek V4.1) exist only for stream_layout mhc
        for name in ("stream_aux_in", "stream_aux_out"):
            self.assertEqual(out[name]["present_when"], "stream_layout == 'mhc'", name)
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


# --- stream layouts, world axis and harness hooks beyond Muse Spark (DeepSeek-V4.1 mhc) ---

MUSE_AXES = {"hidden": 8192, "world": 8, "launches": 2, "pools": 2, "kv_width": 128, "topk": 8}
MHC_AXES = {"hidden": 5120, "world": 2, "launches": 2, "pools": 2, "kv_width": 512, "topk": 6, "hc": 4}
DS_ATTRS = {"layer_kind": "routed_moe", "stream_dtype": "bfloat16", "stream_layout": "mhc"}


class StreamLayoutFamilyTest(unittest.TestCase):
    def setUp(self):
        self.fam = U.load_family("mono_layer")

    def test_match_covers_the_dsv41_kernels_and_ops(self):
        import re
        k = re.compile(self.fam["match"]["kernel"][0])
        op = re.compile(self.fam["match"]["op"][0])
        for name in ("aiter::dsv41_mono_k1_s6_tp2_r2_tl0_f1_i0", "aiter::dsv41_mono_k2_s48_tp2_r1_tl0_i0",
                     "aiter::dsv41_mono_ffn_s6_tp2_tl0", "aiter::muse_spark_layer_s8_q8_w8_n1_x"):
            self.assertTrue(k.search(name), name)
        for name in ("DSV41MonoLayer", "MonoDecodeLayer", "MuseMonoRunner"):
            self.assertTrue(op.search(name), name)

    def test_a_flat_task_keeps_the_muse_shapes(self):
        doc = U.freeze_task_contract(self.fam, attrs={"layer_kind": "routed_moe", "stream_dtype": "float32"},
                                     const_axes=MUSE_AXES)
        self.assertEqual(doc["task"]["attrs"]["stream_layout"], "flat")
        sym = U.symbols_for(doc, {"rows": 8, "context": 100})
        got = {n: U.shape_of(o, sym) for n, o in doc["outputs"].items() if U.present(o, sym)}
        self.assertEqual(got["stream_out"], (8, 8, 8192))
        self.assertEqual(got["layer_out"], (8, 2, 8, 8192))
        self.assertNotIn("stream_aux_out", got)
        ins = {n: U.shape_of(o, sym) for n, o in doc["inputs"].items() if U.present(o, sym)}
        self.assertEqual(ins, {"stream": (8, 8192), "attn_input": (8, 8192)})

    def test_an_mhc_task_carries_the_hc_stream_and_its_mixes(self):
        doc = U.freeze_task_contract(self.fam, attrs=DS_ATTRS, const_axes=MHC_AXES)
        sym = U.symbols_for(doc, {"rows": 6, "context": 4096})
        got = {n: U.shape_of(o, sym) for n, o in doc["outputs"].items() if U.present(o, sym)}
        self.assertEqual(got["stream_out"], (2, 6, 4, 5120))
        self.assertEqual(got["stream_in"], (2, 6, 4, 5120))
        self.assertEqual(got["layer_out"], (2, 2, 6, 4, 5120))
        self.assertEqual(got["attn_out"], (2, 6, 5120))
        self.assertEqual(got["stream_aux_in"], (2, 6, 24))
        self.assertEqual(got["stream_aux_out"], (2, 2, 6, 24))
        self.assertEqual(got["kv_rows"], (2, 2, 2, 2, 6, 512))
        ins = {n: U.shape_of(o, sym) for n, o in doc["inputs"].items() if U.present(o, sym)}
        self.assertEqual(ins, {"stream": (6, 4, 5120), "stream_aux": (6, 24), "attn_input": (6, 5120)})

    def test_delta_along_reads_a_by_cases_shape(self):
        bad = json.loads(json.dumps(self.fam))
        bad["outputs"]["layer_out"]["shape"]["cases"]["mhc"] = ["world", "rows", "launches", "hc", "hidden"]
        with self.assertRaises(U.ContractError):
            U.validate_family(bad)          # launches at another position in one case
        self.assertEqual(U.shape_cases(["a", "b"]), [["a", "b"]])
        self.assertEqual(len(U.shape_cases(self.fam["outputs"]["layer_out"]["shape"])), 2)

    def test_timing_world_is_not_hard_coded(self):
        self.assertNotIn("world", self.fam["timing"])


@unittest.skipUnless(HAVE_TORCH, "torch")
class DeltaRuntimeTest(unittest.TestCase):
    def test_delta_along_launches_of_an_mhc_layer_out(self):
        import torch
        fam = U.load_family("mono_layer")
        spec = fam["outputs"]["layer_out"]
        s_in = torch.zeros(2, 3, 4, 5)
        lo = torch.stack([s_in + 1, s_in + 3], dim=1)             # [world, launches, rows, hc, hidden]
        c, b, _, fail = U._delta(spec, spec["bound"]["delta"], lo, lo, None, {"stream_in": s_in},
                                 {"stream_in": s_in}, None, None)
        self.assertIsNone(fail)
        self.assertTrue(bool((c[:, 0] == 1).all() and (c[:, 1] == 2).all()))


def _scaffold_mono(task, axes, attrs=None, extra=()):
    argv = ["new", "--family", "mono_layer", "--task", task, "--target",
            "vllm.models.deepseek_v41.amd.mono.runner:DSV41MonoLayer.forward",
            "--bind-module", "vllm.models.deepseek_v41.amd.mono",
            "--bind-file", "kernel_src/vllm/models/deepseek_v41/amd/mono/__init__.py"]
    for k, v in axes.items():
        argv += ["--axis", "%s=%d" % (k, v)]
    for k, v in (attrs or {}).items():
        argv += ["--attr", "%s=%s" % (k, v)]
    with contextlib.redirect_stdout(io.StringIO()):
        return scaffold_ut.main(argv + list(extra))


class StreamScaffoldTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mono_dsv41_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.task = os.path.join(self.root, "dsv41_mono_target_layer")

    def read(self, name):
        with open(os.path.join(self.task, name)) as f:
            return f.read()

    def test_mhc_scaffold_declares_the_stream_hooks_and_world_2(self):
        _scaffold_mono(self.task, MHC_AXES, DS_ATTRS)
        meta = json.loads(self.read("meta.json"))
        self.assertEqual(meta["timing"], {"kind": "rank_chain", "world": 2, "chain_length": 8})
        src = self.read("rank_model.py")
        compile(src, "rank_model.py", "exec")
        self.assertIn("STREAM_SHAPE = (4, 5120)", src)
        self.assertIn("STREAM_AUX = (24, torch.float32)", src)
        self.assertIn("WORLD = 2 ", src)
        cases = self.read("cases.py")
        compile(cases, "cases.py", "exec")
        self.assertIn('aux=args.get("stream_aux")', cases)

    def test_a_flat_scaffold_declares_no_stream_hooks(self):
        _scaffold_mono(self.task, MUSE_AXES)
        src = self.read("rank_model.py")
        self.assertNotIn("STREAM_SHAPE =", src)
        self.assertNotIn("STREAM_AUX =", src)

    def test_rank_chain_needs_the_world_axis(self):
        axes = dict(MHC_AXES)
        axes.pop("world")
        with self.assertRaises(SystemExit):
            _scaffold_mono(self.task, axes, DS_ATTRS)

    def test_mhc_needs_hc(self):
        axes = dict(MHC_AXES)
        axes.pop("hc")
        with self.assertRaises(SystemExit):
            _scaffold_mono(self.task, axes, DS_ATTRS)


# ----------------------------------------------------------------------------- the harness hooks
FAKE_MODEL = textwrap.dedent('''
    import torch

    BLOCK, DATA, REC = 4, 6, 8          # a page: BLOCK data rows of DATA bytes, then BLOCK 2-B scales


    class RankModel:
        WORLD = 1
        LAUNCHES = 1
        POOLS = 1
        HIDDEN = 8
        TOP_K = None
        KV_WIDTH = 3
        STREAM_DTYPE = torch.bfloat16
        INPUT_DTYPE = torch.bfloat16
        STREAM_SHAPE = (2, 8)
        STREAM_AUX = (3, torch.float32)
        PATHS = ("deployed",)
        PACKAGE = "fakepkg.sub.mono"
        NOTES = []

        def __init__(self, rank, dev, group, n_layers=None):
            self.dev = dev
            self.caches = [(torch.zeros(3, 64, dtype=torch.uint8),)]   # 3 pages of 64 B

        def pool_of(self, layer):
            return 0

        def writes(self, path, i):
            return (0,)

        def kv_targets(self, m):
            return (m["slots"],)

        def kv_bytes(self, layer, t, idx):
            page, off = idx // BLOCK, idx % BLOCK
            d0 = page * t.stride(0) + off * DATA
            s0 = page * t.stride(0) + BLOCK * DATA + off * 2
            return torch.cat([d0[:, None] + torch.arange(DATA), s0[:, None] + torch.arange(2)], 1)

        def kv_decode(self, layer, raw):
            return raw[:, :3].float() + raw[:, 6:7].float()

        def buffers(self):
            return {"ids": None, "produced": {}, "inputs": {}, "diag": {}}

        def launch(self, i, rows, m, fb, stream, x, aux):
            t = self.caches[0][0]
            flat = self.kv_bytes(0, t, m["slots"])
            t.view(-1)[flat.reshape(-1)] = 5        # K1's insert: every byte of the record
            return stream + 1, x * 2, aux + 0.5      # new tensors, as DSV41MonoLayer.forward

        def end_step(self):
            pass
''')


@unittest.skipUnless(HAVE_TORCH, "torch")
class HarnessHooksTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="mono_harness_hooks_")
        self.addCleanup(shutil.rmtree, self.dir, True)
        shutil.copy2(os.path.join(UTC, "mono_harness.py"), os.path.join(self.dir, "rank_harness.py"))
        with open(os.path.join(self.dir, "rank_model.py"), "w") as f:
            f.write(FAKE_MODEL)
        for name in ("rank_model", "rank_harness_dsv41_test"):
            sys.modules.pop(name, None)
        spec = importlib.util.spec_from_file_location("rank_harness_dsv41_test", os.path.join(self.dir, "rank_harness.py"))
        self.H = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.H)
        self.addCleanup(sys.modules.pop, "rank_model", None)

    def rank(self):
        import torch
        R = object.__new__(self.H.Rank)
        R.rank, R.dev = 0, torch.device("cpu")
        R.model = self.H.Model(0, R.dev, None)
        R.caches = R.model.caches
        R.static, R._shared = None, {}
        return R

    def test_stream_shape_and_aux_inputs(self):
        st, x, aux = self.H.make_inputs(3, 7, with_aux=True)
        self.assertEqual((tuple(st.shape), tuple(x.shape), tuple(aux.shape)), ((3, 2, 8), (3, 8), (3, 3)))
        st2, x2 = self.H.make_inputs(3, 7)
        self.assertTrue(bool((st2 == st).all() and (x2 == x).all()))   # the aux is drawn last

    def test_kv_bytes_poison_collect_and_aux_capture(self):
        import torch
        R = self.rank()
        m = {"slots": torch.tensor([1, 6])}   # page 0 row 1, page 1 row 2
        cap = self.H._Capture(R.model, 2, R.dev)
        snap = R._snapshot()
        R._poison_launch(0, "deployed", m, 2)
        t = R.caches[0][0]
        flat = R.model.kv_bytes(0, t, m["slots"])
        self.assertTrue(bool((t.view(-1)[flat.reshape(-1)] == self.H.KV_POISON).all()))
        self.assertEqual(int((t != 0).sum()), flat.numel())               # only the records' bytes
        st, x, aux = self.H.make_inputs(2, 1, with_aux=True)
        self.H.Rank._launches(R, 2, "deployed", m, None, st, x, cap, aux)
        out = R._collect(2, m, cap, snap)
        self.assertEqual(tuple(out["kv_rows"].shape), (1, 1, 2, 2, 3))
        self.assertTrue(bool((out["kv_rows"][0, 0, 0] == 10).all()))      # decoded: 5 + 5
        self.assertEqual(int(out["kv_stray"].sum()), 0)
        self.assertTrue(torch.equal(out["layer_out"][0], (st + 1).to(out["layer_out"].dtype)))   # the returned stream
        self.assertTrue(torch.equal(out["stream_aux_in"], aux))
        self.assertTrue(torch.allclose(out["stream_aux_out"][0], aux + 0.5))
        self.assertEqual(tuple(out["stream_in"].shape), (2, 2, 8))

    def test_a_write_outside_the_record_is_stray(self):
        import torch
        R = self.rank()
        m = {"slots": torch.tensor([1])}
        cap = self.H._Capture(R.model, 1, R.dev)
        snap = R._snapshot()
        st, x, aux = self.H.make_inputs(1, 1, with_aux=True)
        self.H.Rank._launches(R, 1, "deployed", m, None, st, x, cap, aux)
        R.caches[0][0].view(-1)[63] = 9                                     # a page's unused tail
        self.assertEqual(int(R._collect(1, m, cap, snap)["kv_stray"].sum()), 1)

    def test_byte_level_decode_without_kv_decode(self):
        import torch
        R = self.rank()
        R.model.kv_decode = None                 # no decode hook: every byte one e4m3 code
        self.H.Model.KV_WIDTH = 8
        self.addCleanup(setattr, self.H.Model, "KV_WIDTH", 3)
        m = {"slots": torch.tensor([0])}
        cap = self.H._Capture(R.model, 1, R.dev)
        snap = R._snapshot()
        st, x, aux = self.H.make_inputs(1, 1, with_aux=True)
        self.H.Rank._launches(R, 1, "deployed", m, None, st, x, cap, aux)
        out = R._collect(1, m, cap, snap)
        want = torch.tensor([5], dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        self.assertTrue(bool((out["kv_rows"][0, 0, 0, 0] == want).all()))

    def test_a_missing_aux_operand_is_refused(self):
        with self.assertRaises(ValueError):
            self.H._aux_on(None, "cpu")

    def test_dotted_package_resolves_without_importing_its_parents(self):
        root = os.path.join(self.dir, "site")
        pkg = os.path.join(root, "fakepkg", "sub", "mono")
        os.makedirs(pkg)
        for d in (os.path.join(root, "fakepkg"), os.path.join(root, "fakepkg", "sub"), pkg):
            with open(os.path.join(d, "__init__.py"), "w") as f:
                f.write("raise RuntimeError('imported')\n" if d != pkg else "")
        sys.path.insert(0, root)
        self.addCleanup(sys.path.remove, root)
        self.assertEqual(self.H._package_root(), pkg)
        self.assertNotIn("fakepkg", sys.modules)
        # an overlay finder that claims the dotted name wins
        other = os.path.join(self.dir, "overlay", "mono")
        os.makedirs(other)
        open(os.path.join(other, "__init__.py"), "w").close()

        class Finder:
            def find_spec(self, name, path=None, target=None):
                if name != "fakepkg.sub.mono":
                    return None
                return importlib.util.spec_from_file_location(name, os.path.join(other, "__init__.py"),
                                                              submodule_search_locations=[other])

        f = Finder()
        sys.meta_path.insert(0, f)
        self.addCleanup(sys.meta_path.remove, f)
        self.assertEqual(self.H._package_root(), other)

# --- DeepSeek-V4.1-Flash models (references/mono_models.json dsv41_flash_target / _draft) ---

sys.path.insert(0, os.path.join(SKILL, "scripts"))
import scaffold_tasks as ST  # noqa: E402

DS_AXES = {"hidden": 5120, "world": 2, "launches": 5, "pools": 2, "kv_width": 512, "topk": 6, "hc": 4}
DRAFT_AXES = {"hidden": 5120, "world": 2, "launches": 3, "pools": 1, "kv_width": 512, "topk": 3, "hc": 4}
EXAMPLE = os.path.join(SKILL, "references", "examples", "dsv41_flash_target_rank_model.py")
DRAFT_EXAMPLE = os.path.join(SKILL, "references", "examples", "dsv41_flash_draft_rank_model.py")


def _models():
    with open(os.path.join(SKILL, "references", "mono_models.json")) as f:
        return {m["name"]: m for m in json.load(f)["models"]}


def _not_exposed_overrides(model):
    """The contract overrides a request writes for a model's `not_exposed` outputs
    (mono-layer-ut.md section 0)."""
    return [{"path": "outputs.%s.present_when" % o, "value": "False",
             "why": "%s: the runner does not expose %s (%s)" % (model["name"], o, why)}
            for o, why in sorted((model.get("not_exposed") or {}).items())]


ENTRY_OP = "vllm.v1.attention.ops.flydsl_kernels:_run_compiled@DeepseekV4DecoderLayer"
# the 56 dsv41 mono symbols of the 2026-11-13 TP2 profile (profile/out/mono_classification.json)
_T_K1 = ("r0_tl0_f0_i0", "r1_tl0_f1_i0", "r1_tl0_f1_i1", "r2_tl0_f0_i1", "r2_tl0_f1_i0", "r2_tl0_f1_i1")
_T_K2 = ("r0_tl0_i0", "r1_tl0_i0", "r1_tl0_i1", "r2_tl0_i0", "r2_tl0_i1")
PROFILED = ["_ZN5aiter%ddsv41_mono_k1_s%d_tp2_%sE.kd" % (0, s, v) for s in (6, 12, 24, 48) for v in _T_K1] + \
           ["_ZN5aiter%ddsv41_mono_k2_s%d_tp2_%sE.kd" % (0, s, v) for s in (6, 12, 24, 48) for v in _T_K2] + \
           ["_ZN5aiter%ddsv41_mono_k1_s%d_tp2_r0_tl0_f%d_i0_e128_k3_g1_w192E.kd" % (0, s, f)
            for s in (5, 10, 20, 40) for f in (0, 1)] + \
           ["_ZN5aiter%ddsv41_mono_k2_s%d_tp2_r0_tl0_i0_e128_k3_g1_w192E.kd" % (0, s) for s in (5, 10, 20, 40)]


def _layer_types(path):
    """A worked example's layer-type constants (module-level tuples of literals / names), read
    without importing it (it imports torch)."""
    env = {}
    with open(path) as f:
        tree = ast.parse(f.read())

    def val(node):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Tuple):
            return tuple(val(e) for e in node.elts)
        return ast.literal_eval(node)

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            try:
                if isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for n, v in zip(tgt.elts, node.value.elts):
                        env[n.id] = val(v)
                elif isinstance(tgt, ast.Name):
                    env[tgt.id] = val(node.value)
            except (ValueError, KeyError, AttributeError):
                continue
    return env



class DsModelEntryTest(unittest.TestCase):
    def test_the_entry(self):
        m = _models()["dsv41_flash_target"]
        self.assertEqual(m["attrs"], DS_ATTRS)
        self.assertEqual(sorted(m["oracles_ready"]), ["native", "split"])   # GPU-verified 2026-11-17
        self.assertNotIn("oracles_unverified", m)
        self.assertIn("native oracle floor", m["oracles_verified"])
        self.assertNotIn("oracles_pending", m)
        self.assertEqual(m["bind_module"], "vllm.models.deepseek_v41.amd.mono")
        self.assertEqual(m["entry_ops"], [ENTRY_OP])

    def test_the_entry_matches_the_example(self):
        for name, path, attrs in (("dsv41_flash_target", EXAMPLE, DS_ATTRS), ("dsv41_flash_draft", DRAFT_EXAMPLE, DS_ATTRS)):
            m = _models()[name]
            with open(path) as f:
                ex = _classdict(f.read(), "RankModel")
            self.assertEqual(ex["WORLD"], 2)
            for k, attr in (("hidden", "HIDDEN"), ("launches", "LAUNCHES"), ("pools", "POOLS"),
                            ("kv_width", "KV_WIDTH"), ("topk", "TOP_K")):
                self.assertEqual(m["axes"][k], ex[attr], (name, k))
            self.assertEqual(ex["STREAM_SHAPE"], (m["axes"]["hc"], m["axes"]["hidden"]))
            self.assertEqual(ex["PACKAGE"], m["bind_module"])
            self.assertEqual(m["attrs"], attrs)
            self.assertEqual(ex["PATHS"], ("deployed", "split", "native"))
            self.assertEqual(set(ex["PATHS"][1:]), set(m["oracles_ready"]) | set(m.get("oracles_unverified") or []))
            self.assertEqual(m["worked_example"], os.path.relpath(path, SKILL))

    def test_the_config_constants(self):
        cfg_path = "/shared_nfs/models/DeepSeek-V4.1-Flash/config.json"
        if not os.path.exists(cfg_path):
            self.skipTest("model config not mounted")
        with open(cfg_path) as f:
            c = json.load(f)["text_config"]
        m = _models()["dsv41_flash_target"]["axes"]
        self.assertEqual((m["hidden"], m["topk"], m["hc"]), (c["hidden_size"], c["num_experts_per_tok"], c["hc_mult"]))
        self.assertEqual(m["kv_width"], c["head_dim"])
        d = _models()["dsv41_flash_draft"]["axes"]
        self.assertEqual(d["topk"], c["dspark_num_experts_per_tok"])
        self.assertEqual(d["launches"], c["num_nextn_predict_layers"])
        self.assertEqual(c["compress_ratios"][c["num_hidden_layers"]:], [0] * d["launches"])
        # the target example's layer types stand for served layers of their type
        types = _layer_types(EXAMPLE)
        for ratio, fold, index, served in types["LAYER_TYPES"]:
            self.assertEqual(c["compress_ratios"][served], ratio, served)
            self.assertEqual(served in c["index_source_layer_ids"], index, served)
            self.assertEqual(served in c["engram_layer_ids"] or served == 0, not fold, served)

    def test_the_models_name_only_family_oracles_and_outputs(self):
        fam = _family()
        for name, m in _models().items():
            self.assertLessEqual(set(m["oracles_ready"]) | set(m.get("oracles_unverified") or []),
                                 set(fam["oracles"]), name)
            self.assertLessEqual(set(m.get("not_exposed") or {}), set(fam["outputs"]), name)
            for w in m.get("launch_waivers") or []:
                self.assertTrue(w["why"].strip(), name)
                re.compile(w["pattern"])

    def test_the_case_axes_evaluate_with_the_serving_args(self):
        args = {"num_speculative_tokens": "5"}
        for name, rows in (("dsv41_flash_target", 48), ("dsv41_flash_draft", 40)):
            expr = _models()[name]["case_axes"]["DECODE"]
            self.assertEqual(eval(expr["rows"].format(**args), {}, {"shape": 8}), rows, name)
            self.assertEqual(expr["context"], "context")

    def test_the_patterns_partition_the_profiled_symbols(self):
        ms = {n: re.compile(m["kernel"]) for n, m in _models().items() if n.startswith("dsv41")}
        for k in PROFILED:
            hits = [n for n, r in ms.items() if r.search(k)]
            self.assertEqual(hits, ["dsv41_flash_draft" if "_e128_k3_" in k else "dsv41_flash_target"], k)


class DsOverrideScaffoldTest(unittest.TestCase):
    def test_the_verified_oracles_stay_in_the_frozen_contract(self):
        root = tempfile.mkdtemp(prefix="mono_dsv41_")
        self.addCleanup(shutil.rmtree, root, True)
        task = os.path.join(root, "dsv41_mono_target_layer")
        path = os.path.join(root, "ov.json")
        with open(path, "w") as f:
            json.dump(_not_exposed_overrides(_models()["dsv41_flash_target"]), f)
        _scaffold_mono(task, DS_AXES, DS_ATTRS, ("--override", path))
        doc = U.load_task_contract(task)
        self.assertEqual(sorted(doc["oracles"]), ["native", "split"])
        self.assertEqual(doc["reference"], "oracle:split")
        self.assertEqual([o["path"] for o in doc["task"]["overrides"]], ["outputs.ids.present_when"])
        sym = U.symbols_for(doc, {"rows": 6, "context": 911})
        self.assertFalse(U.present(doc["outputs"]["ids"], sym))     # not exposed by the runner
        self.assertTrue(U.present(doc["outputs"]["stream_aux_out"], sym))
        with open(os.path.join(task, "rank_model.py")) as f:
            self.assertEqual(_classdict(f.read(), "RankModel")["PATHS"], ("deployed", "split", "native"))


class DsLaunchWaiversTest(unittest.TestCase):
    def test_the_rules_waive_exactly_the_unreached_build(self):
        rules = _models()["dsv41_flash_target"]["launch_waivers"]
        target = [k for k in PROFILED if "_e128_k3_" not in k]
        out = ST.expand_launch_waivers(rules, {"DECODE": target})
        self.assertEqual(sorted(w["kernel"] for w in out),
                         sorted(k for k in target if "_k1_" in k and "_r2_tl0_f0_i1" in k))
        self.assertEqual(len(out), 4)                       # widths 6, 12, 24, 48
        self.assertTrue(all(w["phase"] == "DECODE" and w["why"] for w in out))
        self.assertIsNone(ST.expand_launch_waivers(None, {"DECODE": target}))

    def test_the_example_reaches_every_other_target_build(self):
        types = _layer_types(EXAMPLE)
        reach = set()
        for ratio, fold, index, _ in types["LAYER_TYPES"]:
            reach.add(("k1", ratio, int(fold), int(index)))
            reach.add(("k2", ratio, None, int(index)))
        waived = [re.compile(w["pattern"]) for w in _models()["dsv41_flash_target"]["launch_waivers"]]
        for k in PROFILED:
            if "_e128_k3_" in k:
                continue
            m = re.search(r"dsv41_mono_(k[12])_s\d+_tp2_r(\d)_tl0_(?:f(\d)_)?i(\d)", k)
            key = (m.group(1), int(m.group(2)), None if m.group(3) is None else int(m.group(3)), int(m.group(4)))
            self.assertTrue((key in reach) != any(w.search(k) for w in waived), k)
        # the chain times only fold-1 layers of the served types, with the call()'s first layer first
        self.assertTrue(all(t[1] for t in types["CHAIN_PATTERN"]))
        self.assertTrue(set(types["CHAIN_PATTERN"]) <= set(types["LAYER_TYPES"]))

    def test_the_draft_example_reaches_every_draft_build(self):
        types = _layer_types(DRAFT_EXAMPLE)
        folds = {int(t[1]) for t in types["LAYER_TYPES"]}
        self.assertEqual(folds, {0, 1})
        self.assertEqual({t[0] for t in types["LAYER_TYPES"]}, {0})
        for k in PROFILED:
            if "_e128_k3_" in k:
                m = re.search(r"_r(\d)_tl0_(?:f(\d)_)?i(\d)_e128_k3_g1_w192", k)
                self.assertEqual((m.group(1), m.group(3)), ("0", "0"), k)
                self.assertTrue(m.group(2) is None or int(m.group(2)) in folds, k)

    def test_the_lint_refuses_a_stale_or_unexplained_waiver(self):
        meta = {"expected_launches": {"DECODE": ["a", "b"]},
                "ut_contract": {"launch_waivers": [{"kernel": "a", "phase": "DECODE", "why": "x"}]}}
        self.assertEqual(U._lint_launch_waivers(meta), [])
        meta["ut_contract"]["launch_waivers"].append({"kernel": "c", "phase": "DECODE", "why": "x"})
        meta["ut_contract"]["launch_waivers"].append({"kernel": "b", "phase": "DECODE", "why": ""})
        errs = U._lint_launch_waivers(meta)
        self.assertEqual(len(errs), 2)
        self.assertIn("waives nothing", errs[0])
        self.assertIn("need kernel, phase and why", errs[1])

    def test_stamp_meta_keeps_hand_written_waivers(self):
        root = tempfile.mkdtemp(prefix="mono_dsv41_meta_")
        self.addCleanup(shutil.rmtree, root, True)
        with open(os.path.join(root, "meta.json"), "w") as f:
            json.dump({"ut_contract": {"launch_waivers": [{"kernel": "h", "phase": "DECODE", "why": "hand"},
                                                          {"kernel": "a", "phase": "DECODE", "why": "old"}]}}, f)
        ST.stamp_meta(root, {"expected_launches": {"DECODE": ["a", "h"]},
                             "launch_waivers": [{"pattern": "^a$", "why": "rule"}]})
        with open(os.path.join(root, "meta.json")) as f:
            got = json.load(f)["ut_contract"]["launch_waivers"]
        self.assertEqual(got, [{"kernel": "h", "phase": "DECODE", "why": "hand"},
                               {"kernel": "a", "phase": "DECODE", "why": "rule"}])


class DsExampleTest(unittest.TestCase):
    def setUp(self):
        with open(EXAMPLE) as f:
            self.src = f.read()
        with open(DRAFT_EXAMPLE) as f:
            self.draft = f.read()

    def test_they_implement_the_template_interface_and_the_hooks(self):
        root = tempfile.mkdtemp(prefix="mono_dsv41_tmpl_")
        self.addCleanup(shutil.rmtree, root, True)
        task = os.path.join(root, "t")
        _scaffold_mono(task, DS_AXES, DS_ATTRS)
        with open(os.path.join(task, "rank_model.py")) as f:
            tmpl = ast.parse(f.read())

        def members(tree):
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == "RankModel":
                    return {st.name for st in node.body if isinstance(st, ast.FunctionDef)} | \
                           {st.targets[0].id for st in node.body if isinstance(st, ast.Assign)}
            return set()

        for src in (self.src, self.draft):
            mine = members(ast.parse(src))
            self.assertEqual(members(tmpl) - mine, set())
            self.assertTrue({"kv_bytes", "kv_decode", "STREAM_SHAPE", "STREAM_AUX"} <= mine)

    def test_the_subject_is_imported_inside_methods_only(self):
        for src in (self.src, self.draft):
            for node in ast.parse(src).body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    self.assertFalse(any(n.startswith("vllm") for n in names), names)

    def test_the_draft_shares_the_target_helpers_verbatim(self):
        def block(src):
            return src[src.index("def _mono():"):src.index("class RankModel:")]
        self.assertEqual(block(self.draft), block(self.src))

    def test_no_attribute_shadows_a_method(self):
        """An instance attribute named like a method of its class replaces the method on the instance
        (VllmNative once kept its DeepseekV4MoE as self.moe next to its moe() method)."""
        for src in (self.src, self.draft):
            for node in ast.parse(src).body:
                if not isinstance(node, ast.ClassDef):
                    continue
                methods = {f.name for f in node.body if isinstance(f, ast.FunctionDef)}
                attrs = {t.attr for n in ast.walk(node) if isinstance(n, ast.Assign) for tt in n.targets
                         for t in (tt.elts if isinstance(tt, ast.Tuple) else [tt])
                         if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"}
                self.assertEqual(methods & attrs, set(), node.name)

    def test_native_is_vllms_chain(self):
        """The native path names vLLM's ops (the numerics test's chain) and never the mono package's
        stages; it is teacher-forced and aligns near ties."""
        b = self.src[self.src.index("class VllmNative:"):self.src.index("class RankModel:")]
        for op in ("MHCPreDelayedOp", "rocm_mxfp8_block32_gemm", "fused_q_kv_rmsnorm",
                   "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert", "triton_sparse_mla_fwd",
                   "rocm_inverse_rope_mxfp8_rows", "rocm_dsv41_mxfp8_wo_a_bmm", "DeepseekV4MoE",
                   "process_weights_after_loading", "init_distributed_environment"):
            self.assertIn(op, b)
        self.assertNotIn("mono.stages", b.replace("from vllm.models.deepseek_v41.amd.mono.stages import dims", ""))
        for src in (self.src, self.draft):
            self.assertIn("self._teacher = (d_res", src)
            self.assertIn("normed_k = self._kernel_ffn_input(i, rows)", src)


class DsModelControlsTest(unittest.TestCase):
    """The models' GPU-verified negative controls (mono_models.json) and how scaffold_tasks stamps them."""

    def test_every_ds_model_carries_its_verified_controls(self):
        for name in ("dsv41_flash_target", "dsv41_flash_draft"):
            cs = _models()[name]["negative_controls"]
            self.assertEqual(len(cs), 6, name)
            for c in cs:
                for k in ("leaf", "category", "find", "replace", "expect"):
                    self.assertTrue(c[k].strip(), (name, k))
                self.assertFalse(c["find"].startswith("FILL"), name)
                self.assertNotEqual(c["find"], c["replace"], name)
                self.assertTrue(c["leaf"].startswith("kernel_src/vllm/models/deepseek_v41/amd/mono/"), name)
            self.assertEqual(len({(c["leaf"], c["find"]) for c in cs}), len(cs), name)

    def test_the_draft_controls_reach_its_generic_builds(self):
        # no tuned build runs in the draft (MonoBuild.tuned: target, ratio 2): its peer_partial must not
        # sit in the tuned-only reduce_slice_quad, and its row swap must clamp the odd width's last row
        cs = {c["category"]: c for c in _models()["dsv41_flash_draft"]["negative_controls"]}
        self.assertNotIn("range_constexpr(1, len(ws))", cs["peer_partial"]["find"])
        self.assertIn("attn_region_pair", cs["peer_partial"]["find"])
        swap = [c for c in _models()["dsv41_flash_draft"]["negative_controls"] if "t % 2" in c["replace"]]
        self.assertEqual(len(swap), 1)
        self.assertIn("fx.min(", swap[0]["replace"])

    def test_stamp_meta_replaces_the_fill_stubs_and_keeps_hand_written_ones(self):
        root = tempfile.mkdtemp(prefix="stamp_ctl_")
        self.addCleanup(shutil.rmtree, root, True)
        stub = {"leaf": "kernel_src/x.py", "category": "kv_slot", "find": "FILL: one exact, unique source fragment",
                "replace": "FILL: r", "expect": "FILL: e"}
        hand = {"leaf": "kernel_src/x.py", "category": "softmax_scale", "find": "a = 1", "replace": "a = 2",
                "expect": "hand-written"}
        old = {"leaf": "kernel_src/x.py", "category": "kv_slot", "find": "b = 1", "replace": "b = 3", "expect": "old"}
        with open(os.path.join(root, "meta.json"), "w") as f:
            json.dump({"ut_contract": {"negative_controls": [stub, hand, old]}}, f)
        new = [dict(old, replace="b = 2", expect="new"), {"leaf": "kernel_src/y.py", "category": "peer_partial",
                                                          "find": "c = 1", "replace": "c = 0", "expect": "e"}]
        ST.stamp_meta(root, {"negative_controls": new})
        with open(os.path.join(root, "meta.json")) as f:
            got = json.load(f)["ut_contract"]["negative_controls"]
        self.assertEqual(got, [hand] + new)


if __name__ == "__main__":
    unittest.main()
