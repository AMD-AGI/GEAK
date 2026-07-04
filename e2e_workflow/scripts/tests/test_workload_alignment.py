#!/usr/bin/env python3
"""Unit tests for the workload-alignment scripts (stdlib only; no pytest needed).

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_workload_alignment.py

Covers the three deterministic, pure-stdlib pieces the workload-aligned harness relies on:
  - parse_regime.py        : launch-flag/model-config -> regime descriptor
  - attribute_weights.py   : meta shapes JOIN profile weight signal (op_kind-aware) + regime guards
  - parse_profile.build_workload : trace agg -> per-(shape,dtype) weighted workload model
"""
import importlib.util
import json
import os
import tempfile
import unittest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(mod_name, filename):
    path = os.path.join(SCRIPTS_DIR, filename)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


parse_regime = _load("parse_regime", "parse_regime.py")
attribute_weights = _load("attribute_weights", "attribute_weights.py")
parse_profile = _load("parse_profile", "parse_profile.py")
harness_lib = _load("harness_lib", "harness_lib.py")


def _write_json(obj):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh)
    return path


# --------------------------------------------------------------------------- #
# parse_regime.py
# --------------------------------------------------------------------------- #
class TestParseRegime(unittest.TestCase):
    def test_empty_defaults(self):
        r = parse_regime.parse_regime("")
        self.assertEqual(r["quant"]["method"], "none")
        self.assertEqual(r["quant"]["source"], "none")
        self.assertEqual(r["kv_cache_dtype"], "auto")
        self.assertEqual(r["compile"], "eager")
        self.assertTrue(r["cuda_graph"])
        self.assertFalse(r["enforce_eager"])   # default baseline keeps graph replay ON

    def test_enforce_eager_flags(self):
        for flag in ("--enforce-eager", "--disable-cuda-graph"):
            r = parse_regime.parse_regime(flag)
            self.assertTrue(r["enforce_eager"], flag)
            self.assertFalse(r["cuda_graph"], flag)

    def test_fp8_quant_flag(self):
        r = parse_regime.parse_regime("--quantization fp8")
        self.assertEqual(r["quant"]["method"], "fp8")
        self.assertEqual(r["quant"]["act_dtype"], "fp8")
        self.assertEqual(r["quant"]["weight_dtype"], "fp8_e4m3")
        self.assertEqual(r["quant"]["source"], "flag")

    def test_equals_form_and_full_serving_flags(self):
        r = parse_regime.parse_regime(
            "--quantization=fp8 --kv-cache-dtype=fp8 --enable-torch-compile --disable-cuda-graph")
        self.assertEqual(r["quant"]["method"], "fp8")
        self.assertEqual(r["kv_cache_dtype"], "fp8")
        self.assertEqual(r["compile"], "torch_compile")
        self.assertFalse(r["cuda_graph"])

    def test_awq_int4(self):
        r = parse_regime.parse_regime("--quantization awq")
        self.assertEqual(r["quant"]["weight_dtype"], "int4")
        self.assertEqual(r["quant"]["act_dtype"], "bf16")

    def test_model_config_fp8_blockscale(self):
        cfg = _write_json({"quantization_config": {"quant_method": "fp8",
                                                   "weight_block_size": [128, 128]}})
        try:
            r = parse_regime.parse_regime("", cfg)   # no flag -> model config wins
            self.assertEqual(r["quant"]["source"], "model_config")
            self.assertEqual(r["quant"]["method"], "fp8_blockscale")
            self.assertEqual(r["quant"]["block_size"], [128, 128])
            self.assertEqual(r["quant"]["act_dtype"], "fp8")
        finally:
            os.unlink(cfg)

    def test_flag_overrides_but_notes_model_mismatch(self):
        cfg = _write_json({"quantization_config": {"quant_method": "fp8"}})
        try:
            r = parse_regime.parse_regime("--quantization none", cfg)
            self.assertEqual(r["quant"]["source"], "flag")
            self.assertIn("model config says fp8", r["notes"])
        finally:
            os.unlink(cfg)


# --------------------------------------------------------------------------- #
# attribute_weights.py
# --------------------------------------------------------------------------- #
class TestAttributeGemm(unittest.TestCase):
    def _meta(self, **kw):
        m = {
            "op_kind": "gemm", "short_name": "_gemm_a8w8",
            "a_shape": ["M", 512], "b_shape": [1024, 512], "dtype": "fp8_e4m3",
            "decode_m_buckets": [1, 128], "prefill_m_buckets": [2048],
        }
        m.update(kw)
        return m

    def _entries(self, decode_us, prefill_us):
        # GRID_MN small -> decode (M_blocks<=1.5); large -> prefill. N=1024, BLOCK_SIZE_N=128 -> nblk=8.
        return [
            {"name": "_gemm_a8w8 GRID_MN_8 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
             "pct_gpu_time": 5.0, "cases": [{"dims": [], "weight": decode_us}]},
            {"name": "_gemm_a8w8 GRID_MN_512 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
             "pct_gpu_time": 20.0, "cases": [{"dims": [], "weight": prefill_us}]},
        ]

    def test_gemm_regime_split_and_shapes(self):
        notes = []
        cases = attribute_weights.attribute_gemm(self._meta(), self._entries(1000.0, 5000.0), notes)
        regimes = {c["regime"] for c in cases}
        self.assertEqual(regimes, {"decode", "prefill"})
        # shapes always come from meta: [[M,K],[N,K]] with K=512, N=1024
        for c in cases:
            self.assertEqual(c["dims"][0][1], 512)
            self.assertEqual(c["dims"][1], [1024, 512])
            self.assertEqual(c["dtypes"], ["fp8_e4m3", "fp8_e4m3"])
            self.assertEqual(c["weight_source"], "regime")
        # decode within-regime prior: 80% on the largest decode bucket (128)
        decode = {c["m"]: c["weight"] for c in cases if c["regime"] == "decode"}
        self.assertAlmostEqual(decode[128], 1000.0 * 0.8, places=3)
        self.assertAlmostEqual(decode[1], 1000.0 * 0.2, places=3)

    def test_gemm_trace_weight_when_profile_exposes_shape(self):
        notes = []
        entries = [{"name": "_gemm_a8w8 GRID_MN_8 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
                    "pct_gpu_time": 5.0,
                    "cases": [{"dims": [[128, 512], [1024, 512]], "weight": 777.0}]}]
        cases = attribute_weights.attribute_gemm(self._meta(), entries, notes)
        traced = [c for c in cases if c["weight_source"] == "trace"]
        self.assertTrue(traced)
        self.assertAlmostEqual(traced[0]["weight"], 777.0, places=3)

    def test_zero_decode_time_warns(self):
        notes = []
        cases = attribute_weights.attribute_gemm(self._meta(), self._entries(0.0, 5000.0), notes)
        decode = [c for c in cases if c["regime"] == "decode"]
        self.assertTrue(all(c["weight"] == 0.0 for c in decode))
        self.assertTrue(all(c["weight_source"] == "prior" for c in decode))
        self.assertTrue(any("ZERO profiled" in n for n in notes))

    def test_regime_floor_protects_decode(self):
        cases = attribute_weights.attribute_gemm(self._meta(), self._entries(0.0, 5000.0), [])
        notes = []
        attribute_weights._apply_regime_floor(cases, 0.3, notes)
        total = sum(c["weight"] for c in cases)
        decode_share = sum(c["weight"] for c in cases if c["regime"] == "decode") / total
        self.assertAlmostEqual(decode_share, 0.3, places=2)
        self.assertTrue(any(c["weight_source"] == "regime_floor"
                            for c in cases if c["regime"] == "decode"))


class TestAttributeGeneric(unittest.TestCase):
    def test_shape_match_trace_vs_prior(self):
        # q1 trailing dims [8,128] match the profiled shape (exact/fuzzy -> trace); q3 has DIFFERENT
        # trailing dims [16,128] so the fuzzy matcher can't attribute it -> prior (weight 0).
        meta = {"op_kind": "attn", "short_name": "_attn_fwd",
                "cases": [{"sig": "q1", "input_shapes": [[1, 8, 128]], "input_dtypes": ["bf16"]},
                          {"sig": "q2", "input_shapes": [[2048, 16, 128]], "input_dtypes": ["bf16"]}]}
        entries = [{"name": "_attn_fwd_kernel", "short_name": "_attn_fwd", "pct_gpu_time": 10.0,
                    "cases": [{"dims": [[1, 8, 128]], "dtypes": ["bf16"], "weight": 100.0, "count": 5}]}]
        notes = []
        cases = attribute_weights.attribute_generic(meta, entries, notes)
        by = {c["name"]: c for c in cases}
        self.assertEqual(by["q1"]["weight_source"], "trace")
        self.assertAlmostEqual(by["q1"]["weight"], 100.0, places=3)
        self.assertEqual(by["q2"]["weight_source"], "prior")
        self.assertEqual(by["q2"]["weight"], 0.0)


class TestServingCountCorrection(unittest.TestCase):
    """The short profiling window (~40 steps) under-counts decode at large OSL, so the raw decode:prefill
    weight split is wrong. --isl/--osl rescale each regime from window counts to lifecycle counts using
    the measured per-call latency."""
    def test_estimate_calls_basic(self):
        est = attribute_weights.estimate_serving_regime_calls(1000, 1000)
        self.assertEqual(est, {"prefill": 1, "decode": 1000})

    def test_estimate_calls_chunked_prefill(self):
        est = attribute_weights.estimate_serving_regime_calls(1000, 500, prefill_chunk=256)
        self.assertEqual(est, {"prefill": 4, "decode": 500})   # ceil(1000/256)=4

    def test_estimate_calls_missing_params_noop(self):
        self.assertEqual(attribute_weights.estimate_serving_regime_calls(None, None), {})

    def _meta(self):
        return {"op_kind": "gemm", "short_name": "_gemm_a8w8",
                "a_shape": ["M", 512], "b_shape": [1024, 512], "dtype": "fp8_e4m3",
                "decode_m_buckets": [128], "prefill_m_buckets": [2048]}

    def _entries_with_counts(self):
        # decode kernel: cheap per call but 40 window steps; prefill: 1 pass, expensive.
        # window weights: decode 1000us(count40 -> 25us/call), prefill 5000us(count1). Raw: prefill wins.
        return [
            {"name": "_gemm_a8w8 GRID_MN_8 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
             "pct_gpu_time": 5.0, "cases": [{"dims": [], "weight": 1000.0, "count": 40}]},
            {"name": "_gemm_a8w8 GRID_MN_512 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
             "pct_gpu_time": 20.0, "cases": [{"dims": [], "weight": 5000.0, "count": 1}]},
        ]

    def test_gemm_decode_reweighted_to_lifecycle(self):
        notes = []
        workload = {"isl": 1000, "osl": 1000, "prefill_chunk": None}  # decode=1000, prefill=1
        cases = attribute_weights.attribute_gemm(self._meta(), self._entries_with_counts(), notes,
                                                 workload=workload)
        w = {c["regime"]: c["weight"] for c in cases}
        # per-call latency: decode 25us x 1000 = 25000; prefill 5000us x 1 = 5000. Decode now dominates.
        self.assertGreater(w["decode"], w["prefill"])
        # scale normalized to min: prefill x1 (5000), decode x25 (1000->25000)
        self.assertAlmostEqual(w["prefill"], 5000.0, places=1)
        self.assertAlmostEqual(w["decode"], 25000.0, places=1)
        self.assertTrue(any("serving-count correction" in n for n in notes))

    def test_gemm_no_workload_is_unchanged(self):
        base = attribute_weights.attribute_gemm(self._meta(), self._entries_with_counts(), [])
        corr = attribute_weights.attribute_gemm(self._meta(), self._entries_with_counts(), [],
                                                workload=None)
        self.assertEqual({c["regime"]: c["weight"] for c in base},
                         {c["regime"]: c["weight"] for c in corr})

    def test_missing_counts_skips_correction(self):
        notes = []
        # entries without a `count` field -> can't recover per-call latency -> no-op + note.
        entries = [{"name": "_gemm_a8w8 GRID_MN_8 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
                    "cases": [{"dims": [], "weight": 1000.0}]},
                   {"name": "_gemm_a8w8 GRID_MN_512 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
                    "cases": [{"dims": [], "weight": 5000.0}]}]
        cases = attribute_weights.attribute_gemm(self._meta(), entries, notes,
                                                 workload={"isl": 1000, "osl": 1000})
        w = {c["regime"]: c["weight"] for c in cases}
        self.assertAlmostEqual(w["prefill"], 5000.0, places=1)   # unchanged
        self.assertAlmostEqual(w["decode"], 1000.0, places=1)
        self.assertTrue(any("no observed per-regime call counts" in n for n in notes))


class TestQuantStamping(unittest.TestCase):
    """The regime's job: stamp per-operand dtype/quant so the harness builds in-regime operands (fp8 +
    scales, not bf16), AND the restored _regime_warnings live-seam guard (isolated-win/e2e-loss)."""
    def test_quant_block_fp8_blockscale(self):
        meta = {"dtype": "fp8_e4m3", "out_dtype": "bf16", "weight_block_size": [128, 128]}
        regime = {"quant": {"method": "fp8", "act_dtype": "fp8", "block_size": [128, 128]},
                  "kv_cache_dtype": "fp8"}
        q = attribute_weights._quant_block(meta, regime)
        self.assertEqual(q["act_dtype"], "fp8")
        self.assertEqual(q["weight_dtype"], "fp8_e4m3")
        self.assertEqual(q["out_dtype"], "bf16")
        self.assertEqual(q["weight_block_size"], [128, 128])
        self.assertEqual(q["scale_dtype"], "float32")
        self.assertEqual(q["kv_cache_dtype"], "fp8")

    def test_regime_warnings_present(self):
        # the live-seam guard machinery must exist (restored after being wrongly deleted)
        self.assertTrue(hasattr(attribute_weights, "_regime_warnings"))

    def test_live_seam_guard_flags_low_pct(self):
        # a seam carrying near-zero %GPU under the online regime is probably NOT the live kernel
        notes = []
        entries = [{"pct_gpu_time": 0.4}]
        w = attribute_weights._regime_warnings(
            {"quant": {"method": "fp8"}}, "gemm", entries, live_pct=0.4, live_pct_min=2.0, notes=notes)
        self.assertIn("probably NOT the live kernel", w)
        # a healthy seam (>= min) produces no live-seam warning
        w2 = attribute_weights._regime_warnings(
            {"quant": {"method": "fp8"}}, "gemm", [{"pct_gpu_time": 30.0}], 30.0, 2.0, [])
        self.assertNotIn("probably NOT the live kernel", w2)

    def test_enforce_eager_strawman_flagged(self):
        notes = []
        w = attribute_weights._regime_warnings(
            {"enforce_eager": True}, "gemm", [{"pct_gpu_time": 30.0}], 30.0, 2.0, notes)
        self.assertIn("strawman", w)

    def test_compile_strawman_flagged_for_norm(self):
        w = attribute_weights._regime_warnings(
            {"compile": "torch_compile"}, "norm", [{"pct_gpu_time": 30.0}], 30.0, 2.0, [])
        self.assertIn("strawman", w)


class TestAttributeWeightsEndToEnd(unittest.TestCase):
    """Drive main() through the filesystem like the extractor does."""
    def test_main_stamps_quant_and_normalizes(self):
        meta = {"op_kind": "gemm", "short_name": "_gemm_a8w8",
                "a_shape": ["M", 512], "b_shape": [1024, 512], "dtype": "fp8_e4m3",
                "decode_m_buckets": [1, 128], "prefill_m_buckets": [2048],
                "weight_block_size": [128, 128],
                "regime": {"quant": {"method": "fp8", "act_dtype": "fp8", "block_size": [128, 128]},
                           "kv_cache_dtype": "auto", "compile": "eager"}}
        prof = {"schema": "workload-v1", "kernels": [
            {"name": "_gemm_a8w8 GRID_MN_8 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
             "pct_gpu_time": 5.0, "cases": [{"dims": [], "weight": 1000.0}]},
            {"name": "_gemm_a8w8 GRID_MN_512 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
             "pct_gpu_time": 20.0, "cases": [{"dims": [], "weight": 5000.0}]}]}
        meta_p, prof_p = _write_json(meta), _write_json(prof)
        out_p = tempfile.mkstemp(suffix=".json")[1]
        try:
            import sys
            argv = sys.argv
            sys.argv = ["attribute_weights.py", "--meta", meta_p, "--profile-weights", prof_p,
                        "--name-match", "_gemm_a8w8", "--min-regime-share", "0.3", "--out", out_p]
            try:
                attribute_weights.main()
            finally:
                sys.argv = argv
            with open(out_p) as fh:
                out = json.load(fh)
            self.assertEqual(out["schema"], "workload-v1")
            self.assertEqual(out["op_kind"], "gemm")
            self.assertTrue(out["cases"])
            self.assertAlmostEqual(sum(c["weight_norm"] for c in out["cases"]), 1.0, places=4)
            for c in out["cases"]:                       # quant stamped from meta/regime
                self.assertEqual(c["quant"]["act_dtype"], "fp8")
                self.assertEqual(c["quant"]["weight_block_size"], [128, 128])
            # decode floor honored end-to-end
            dshare = sum(c["weight_norm"] for c in out["cases"] if c["regime"] == "decode")
            self.assertGreaterEqual(dshare, 0.29)
        finally:
            for p in (meta_p, prof_p, out_p):
                if os.path.exists(p):
                    os.unlink(p)


# --------------------------------------------------------------------------- #
# parse_profile.build_workload
# --------------------------------------------------------------------------- #
class TestBuildWorkload(unittest.TestCase):
    def test_per_case_weights_and_provenance(self):
        agg = {"some_gemm_kernel": {
            "calls": 15, "total_us": 1500.0, "shapes": set(), "dtypes": set(),
            "by_case": {
                ('[[1, 512]]', '["c10::BFloat16"]'): {"count": 10, "total_us": 1000.0},
                ('', ''): {"count": 5, "total_us": 500.0},     # shape hidden (graph replay)
            }}}
        wl = parse_profile.build_workload(agg, 1500.0, top_n=25)
        self.assertEqual(wl["schema"], "workload-v1")
        self.assertEqual(wl["num_kernels"], 1)
        k = wl["kernels"][0]
        self.assertEqual(k["pct_gpu_time"], 100.0)
        self.assertEqual(len(k["cases"]), 2)
        # sorted by weight desc -> the traced (1000us) case first
        self.assertEqual(k["cases"][0]["weight_source"], "trace")
        self.assertEqual(k["cases"][0]["dims"], [[1, 512]])
        self.assertAlmostEqual(k["cases"][0]["baseline_latency_ms"], 0.1, places=6)
        self.assertEqual(k["cases"][1]["weight_source"], "regime_prior")
        self.assertEqual(k["cases"][1]["dims"], [])
        self.assertAlmostEqual(sum(c["weight_norm"] for c in k["cases"]), 1.0, places=4)

    def test_target_filter(self):
        agg = {"foo_kernel": {"calls": 1, "total_us": 10.0, "shapes": set(), "dtypes": set(),
                              "by_case": {('', ''): {"count": 1, "total_us": 10.0}}},
               "bar_kernel": {"calls": 1, "total_us": 20.0, "shapes": set(), "dtypes": set(),
                              "by_case": {('', ''): {"count": 1, "total_us": 20.0}}}}
        wl = parse_profile.build_workload(agg, 30.0, top_n=25, target="foo")
        self.assertEqual(wl["num_kernels"], 1)
        self.assertEqual(wl["kernels"][0]["name"], "foo_kernel")


# --------------------------------------------------------------------------- #
# op_kind-aware attribution beyond GEMM (attn / moe / recurrent) — the unified engine
# --------------------------------------------------------------------------- #
class TestAttributeAttn(unittest.TestCase):
    """Attention: the regime is discriminated by the KERNEL NAME (prefill FMHA vs paged decode), and
    the extractor tags each meta case with its regime. Decode usually hides its shape behind a graph,
    so its time must still be attributed from the kernel total, not dropped."""
    def _meta(self):
        return {"op_kind": "attn", "short_name": "attn",
                "cases": [{"sig": "prefill_q2048", "dims": [[2048, 24, 128]], "dtypes": ["bf16"],
                           "regime": "prefill"},
                          {"sig": "decode_q1", "dims": [[64, 24, 128]], "dtypes": ["bf16"],
                           "regime": "decode"}]}

    def test_name_based_regime_split_when_shapes_hidden(self):
        # both launches are graph-hidden (dims=[]); only the NAME says which regime.
        entries = [
            {"name": "fmha_prefill_kernel", "short_name": "attn",
             "cases": [{"dims": [], "weight": 800.0}]},
            {"name": "paged_attention_decode_kernel", "short_name": "attn",
             "cases": [{"dims": [], "weight": 200.0}]},
        ]
        notes = []
        cases = attribute_weights.attribute_attn(self._meta(), entries, notes)
        by = {c["name"]: c for c in cases}
        self.assertEqual(by["prefill_q2048"]["regime"], "prefill")
        self.assertEqual(by["decode_q1"]["regime"], "decode")
        # prefill got the 800us, decode the 200us — NOT collapsed to 0 prior
        self.assertAlmostEqual(by["prefill_q2048"]["weight"], 800.0, places=1)
        self.assertAlmostEqual(by["decode_q1"]["weight"], 200.0, places=1)
        self.assertTrue(all(c["weight_source"] == "regime" for c in cases))

    def test_shape_matched_decode_uses_trace(self):
        # profile exposed the decode shape -> trace weight; prefill stays name-classified.
        entries = [
            {"name": "fmha_prefill_kernel", "short_name": "attn",
             "cases": [{"dims": [], "weight": 500.0}]},
            {"name": "paged_attention_decode_kernel", "short_name": "attn",
             "cases": [{"dims": [[64, 24, 128]], "dtypes": ["bf16"], "weight": 300.0, "count": 9}]},
        ]
        notes = []
        cases = attribute_weights.attribute_attn(self._meta(), entries, notes)
        by = {c["name"]: c for c in cases}
        self.assertEqual(by["decode_q1"]["weight_source"], "trace")
        self.assertAlmostEqual(by["decode_q1"]["weight"], 300.0, places=1)


class TestAttributeRecurrent(unittest.TestCase):
    """A pure-decode recurrent kernel runs only under a HIP/CUDA graph: shapes hidden, one regime.
    Its total time must be distributed across the meta cases by the size prior (larger batch dominates),
    NOT dropped to weight-0 prior (which would collapse the weighted metric to a geomean)."""
    def test_size_prior_batch_dominates(self):
        meta = {"op_kind": "linear_attn_recurrent", "short_name": "gdn_decode",
                "cases": [{"sig": "decode_B64", "dims": [[64, 10240], [64, 48]], "regime": "decode"},
                          {"sig": "decode_B1", "dims": [[1, 10240], [1, 48]], "regime": "decode"}]}
        entries = [{"name": "gdn_decode_kernel", "short_name": "gdn_decode",
                    "cases": [{"dims": [], "weight": 200000.0, "count": 1824}]}]
        notes = []
        cases = attribute_weights.attribute_generic(meta, entries, notes)
        by = {c["name"]: c for c in cases}
        self.assertEqual(by["decode_B64"]["weight_source"], "regime_prior")
        total = sum(c["weight"] for c in cases)
        # B64 element count 64*10240 >> B1 1*10240 -> ~0.985 share
        self.assertGreater(by["decode_B64"]["weight"] / total, 0.95)
        self.assertGreater(by["decode_B1"]["weight"], 0.0)   # tail is present, not zero

    def test_no_total_no_shape_stays_prior_zero(self):
        # no profiled time at all -> honest weight-0 prior (nothing to distribute)
        meta = {"op_kind": "editable", "short_name": "k",
                "cases": [{"sig": "c0", "dims": [[8, 8]], "regime": ""}]}
        cases = attribute_weights.attribute_generic(meta, [], [])
        self.assertEqual(cases[0]["weight"], 0.0)
        self.assertEqual(cases[0]["weight_source"], "prior")


class TestPassthrough(unittest.TestCase):
    """When meta has no explicit cases, _passthrough emits the profile's own per-(shape,dtype)
    weights verbatim — the fallback for kernels the extractor didn't tag with cases."""
    def test_passthrough_emits_profile_shapes(self):
        entries = [{"name": "k", "short_name": "k", "cases": [
            {"dims": [[8, 128]], "dtypes": ["bf16"], "weight": 100.0, "count": 3},
            {"dims": [[16, 128]], "dtypes": ["bf16"], "weight": 200.0, "count": 7},
        ]}]
        notes = []
        cases = attribute_weights._passthrough(entries, notes)
        self.assertEqual(len(cases), 2)
        self.assertTrue(all(c["weight_source"] == "trace" for c in cases))
        self.assertAlmostEqual(cases[0]["weight"], 100.0)
        self.assertAlmostEqual(cases[1]["weight"], 200.0)

    def test_passthrough_empty_profile(self):
        notes = []
        cases = attribute_weights._passthrough([], notes)
        self.assertEqual(cases, [])
        self.assertTrue(any("nothing to weight" in n for n in notes))


class TestRegimeFloorEdgeCases(unittest.TestCase):
    """Edge cases in _apply_regime_floor: overflow guard, and non-GEMM (no per-case M) even split."""
    def test_floor_overflow_skips(self):
        cases = [
            {"regime": "decode", "weight": 0.0, "weight_source": "prior"},
            {"regime": "prefill", "weight": 0.0, "weight_source": "prior"},
            {"regime": "other", "weight": 100.0, "weight_source": "regime"},
        ]
        notes = []
        attribute_weights._apply_regime_floor(cases, 0.6, notes)
        self.assertTrue(any("skipped" in n for n in notes))

    def test_non_gemm_even_split(self):
        cases = [
            {"name": "d1", "regime": "decode", "weight": 0.0, "weight_source": "prior"},
            {"name": "d2", "regime": "decode", "weight": 0.0, "weight_source": "prior"},
            {"name": "p1", "regime": "prefill", "weight": 100.0, "weight_source": "regime"},
        ]
        notes = []
        attribute_weights._apply_regime_floor(cases, 0.3, notes)
        decode_cases = [c for c in cases if c["regime"] == "decode"]
        self.assertTrue(all(c["weight_source"] == "regime_floor" for c in decode_cases))
        self.assertAlmostEqual(decode_cases[0]["weight"], decode_cases[1]["weight"])
        total = sum(c["weight"] for c in cases)
        decode_share = sum(c["weight"] for c in decode_cases) / total
        self.assertAlmostEqual(decode_share, 0.3, places=2)


class TestAttnUnnamedSpreading(unittest.TestCase):
    """When attention launches can't be name-classified (no decode/prefill/paged keywords), the
    unnamed time is spread across meta regimes by size."""
    def test_unnamed_spread_by_size(self):
        meta = {"op_kind": "attn", "short_name": "attn",
                "cases": [{"sig": "prefill_q2048", "dims": [[2048, 24, 128]], "dtypes": ["bf16"],
                           "regime": "prefill"},
                          {"sig": "decode_q1", "dims": [[64, 24, 128]], "dtypes": ["bf16"],
                           "regime": "decode"}]}
        entries = [{"name": "some_unknown_attn_op", "short_name": "attn",
                    "cases": [{"dims": [], "weight": 1000.0}]}]
        notes = []
        cases = attribute_weights.attribute_attn(meta, entries, notes)
        self.assertTrue(any("unnamed launches" in n for n in notes))
        by = {c["name"]: c for c in cases}
        self.assertGreater(by["prefill_q2048"]["weight"], by["decode_q1"]["weight"])
        self.assertGreater(by["decode_q1"]["weight"], 0.0)


class TestBaseToken(unittest.TestCase):
    """_base_token should keep embedded digits (a8w8) and only strip trailing _NNN suffixes."""
    def test_keeps_embedded_digits(self):
        self.assertEqual(attribute_weights._base_token("_gemm_a8w8"), "_gemm_a8w8")

    def test_strips_trailing_numeric_suffix(self):
        self.assertEqual(attribute_weights._base_token("_gemm_a8w8_128"), "_gemm_a8w8")

    def test_drops_whitespace_params(self):
        self.assertEqual(attribute_weights._base_token("_gemm_a8w8 GRID_MN_8"), "_gemm_a8w8")


class TestAttributeMoe(unittest.TestCase):
    """MoE grouped-GEMM reuses the precise bucket/grid GEMM engine (effective-M from routing) and adds
    a low-confidence note."""
    def test_moe_delegates_to_gemm_engine(self):
        meta = {"op_kind": "moe", "short_name": "fused_moe",
                "a_shape": ["M", 512], "b_shape": [1024, 512], "dtype": "fp8_e4m3",
                "decode_m_buckets": [8], "prefill_m_buckets": [2048]}
        entries = [{"name": "fused_moe GRID_MN_8 BLOCK_SIZE_N_128", "short_name": "fused_moe",
                    "cases": [{"dims": [], "weight": 1000.0}]},
                   {"name": "fused_moe GRID_MN_512 BLOCK_SIZE_N_128", "short_name": "fused_moe",
                    "cases": [{"dims": [], "weight": 5000.0}]}]
        notes = []
        cases = attribute_weights.attribute_moe(meta, entries, notes)
        regimes = {c["regime"] for c in cases}
        self.assertEqual(regimes, {"decode", "prefill"})
        self.assertTrue(any("routing-dependent" in n for n in notes))


class TestHarnessRegime(unittest.TestCase):
    """harness_lib regime-driven synthesis derivations — pure (no torch): a unittest that synthesizes
    inputs in the LIVE regime can never key the paged-KV `x`/dtype/scales off the wrong (compute) dtype.
    These are GENERAL over dtype/quant — not an fp8 special-case (int8 -> x=16 too, fp32 -> x=4)."""

    def test_deployment_graph_mode(self):
        # default / graphed baseline -> time under a graph
        self.assertTrue(harness_lib.deployment_graph_mode({}))
        self.assertTrue(harness_lib.deployment_graph_mode({"cuda_graph": True}))
        # enforce-eager / disabled graph -> eager timing (regime genuinely runs eager)
        self.assertFalse(harness_lib.deployment_graph_mode({"enforce_eager": True}))
        self.assertFalse(harness_lib.deployment_graph_mode({"cuda_graph": False}))

    def test_pack_x_across_dtypes(self):
        self.assertEqual(harness_lib.pack_x("fp8"), 16)
        self.assertEqual(harness_lib.pack_x("fp8_e4m3fnuz"), 16)
        self.assertEqual(harness_lib.pack_x("int8"), 16)
        self.assertEqual(harness_lib.pack_x("fp16"), 8)
        self.assertEqual(harness_lib.pack_x("bf16"), 8)
        self.assertEqual(harness_lib.pack_x("fp32"), 4)

    def test_regime_spec_fp8_kv(self):
        spec = harness_lib.regime_spec({"kv_cache_dtype": "fp8", "quant": {"method": "none"}})
        self.assertEqual(spec["kv_x"], 16)
        self.assertTrue(spec["kv_quant"])
        self.assertTrue(spec["needs_scales"])

    def test_regime_spec_auto_kv(self):
        spec = harness_lib.regime_spec({"kv_cache_dtype": "auto", "quant": {"method": "none"}})
        self.assertEqual(spec["kv_dtype"], "bf16")
        self.assertEqual(spec["kv_x"], 8)
        self.assertFalse(spec["kv_quant"])
        self.assertFalse(spec["needs_scales"])

    def test_regime_spec_int8_kv(self):
        spec = harness_lib.regime_spec({"kv_cache_dtype": "int8", "quant": {"method": "none"}})
        self.assertEqual(spec["kv_x"], 16)
        self.assertTrue(spec["needs_scales"])

    def test_regime_spec_quant_needs_scales(self):
        spec = harness_lib.regime_spec({"kv_cache_dtype": "auto",
                                        "quant": {"method": "fp8", "weight_dtype": "fp8_e4m3"}})
        self.assertEqual(spec["quant_method"], "fp8")
        self.assertEqual(spec["operand_dtype"], "fp8_e4m3")
        self.assertTrue(spec["needs_scales"])

    def test_parser_to_spec_coherence(self):
        """The missing seam: parse_regime output must plug straight into regime_spec with no glue."""
        r = parse_regime.parse_regime("--quantization fp8 --kv-cache-dtype fp8")
        spec = harness_lib.regime_spec(r)
        self.assertEqual(spec["kv_x"], 16)
        self.assertTrue(spec["needs_scales"])

        r0 = parse_regime.parse_regime("")
        spec0 = harness_lib.regime_spec(r0)
        self.assertEqual(spec0["kv_x"], 8)
        self.assertFalse(spec0["needs_scales"])

    def test_fp8_is_fnuz_by_arch(self):
        """The ONE hardware-specific axis: MI300 (gfx942/CDNA3) = fnuz fp8; MI355 (gfx950/CDNA4) = OCP fn."""
        self.assertTrue(harness_lib.fp8_is_fnuz("gfx942"))
        self.assertTrue(harness_lib.fp8_is_fnuz("gfx942:sramecc+:xnack-"))
        self.assertTrue(harness_lib.fp8_is_fnuz("gfx90a"))
        self.assertFalse(harness_lib.fp8_is_fnuz("gfx950"))
        self.assertFalse(harness_lib.fp8_is_fnuz(""))

    def test_pack_x_arch_independent(self):
        """Layout math is arch-independent: every fp8 variant is 1 byte -> x=16 on MI300 AND MI355."""
        for name in ("fp8", "fp8_e4m3", "fp8_e4m3fnuz", "fp8_e4m3fn", "fp8_e5m2"):
            self.assertEqual(harness_lib.pack_x(name), 16, name)

    def test_regime_dtype_arch_driven_fp8(self):
        """A bare fp8 name resolves to the arch's variant; an explicit fnuz/fn wins. Guarded for no-torch."""
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")
        if not hasattr(torch, "float8_e4m3fnuz") or not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("torch build lacks both fp8 variants")
        self.assertEqual(harness_lib.regime_dtype("fp8", arch="gfx942"), torch.float8_e4m3fnuz)
        self.assertEqual(harness_lib.regime_dtype("fp8", arch="gfx950"), torch.float8_e4m3fn)
        self.assertEqual(harness_lib.regime_dtype("fp8_e4m3", arch="gfx950"), torch.float8_e4m3fn)
        # explicit checkpoint-declared format wins over arch:
        self.assertEqual(harness_lib.regime_dtype("fp8_e4m3fnuz", arch="gfx950"), torch.float8_e4m3fnuz)


if __name__ == "__main__":
    unittest.main(verbosity=2)
