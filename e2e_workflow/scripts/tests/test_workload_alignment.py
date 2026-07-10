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


class TestServingCallModel(unittest.TestCase):
    """--isl/--osl EXPOSE the analytic serving call model (serving_weight_model.analytic_calls) for the
    immutable unittest to SELF-WEIGHT with its own MEASURED latency (weight_i = baseline_ms_i ×
    analytic_calls[regime_i]). They must NOT rescale the profile `weight` here — the intra-kernel
    prefill/decode split is reconstructed from measured latency in the unittest, not patched onto the
    biased short profiling window (that used to fight the self-weight)."""
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
                "decode_m_buckets": [128], "prefill_m_buckets": [2048], "regime": {}}

    def _prof(self):
        # window weights: decode 1000us, prefill 5000us. The raw profiled TIME split, verbatim.
        return {"schema": "workload-v1", "kernels": [
            {"name": "_gemm_a8w8 GRID_MN_8 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
             "pct_gpu_time": 5.0, "cases": [{"dims": [], "weight": 1000.0, "count": 40}]},
            {"name": "_gemm_a8w8 GRID_MN_512 BLOCK_SIZE_N_128", "short_name": "_gemm_a8w8",
             "pct_gpu_time": 20.0, "cases": [{"dims": [], "weight": 5000.0, "count": 1}]}]}

    def test_attribute_gemm_takes_no_workload(self):
        # the intra-kernel weight rescale is gone: attribute_gemm must NOT accept a workload kwarg.
        import inspect
        self.assertNotIn("workload", inspect.signature(attribute_weights.attribute_gemm).parameters)
        self.assertFalse(hasattr(attribute_weights, "_apply_serving_scale"))
        self.assertFalse(hasattr(attribute_weights, "serving_regime_scale"))

    def _run_main(self, extra):
        import sys
        meta_p = _write_json(self._meta())
        prof_p = _write_json(self._prof())
        out_p = tempfile.mkstemp(suffix=".json")[1]
        argv = sys.argv
        sys.argv = ["attribute_weights.py", "--meta", meta_p, "--profile-weights", prof_p,
                    "--name-match", "_gemm_a8w8", "--out", out_p] + extra
        try:
            attribute_weights.main()
            with open(out_p) as fh:
                return json.load(fh)
        finally:
            sys.argv = argv
            for p in (meta_p, prof_p, out_p):
                if os.path.exists(p):
                    os.unlink(p)

    def test_main_surfaces_self_weight_and_floors_graph_hidden_decode(self):
        # The AUTHORITATIVE decode:prefill split is NOT the static `weight` — it is the unittest
        # self-weight (measured ms x analytic_calls), surfaced verbatim under serving_weight_model for the
        # UT to consume. --isl/--osl NEVER lifecycle-rescale the static `weight`; the ONLY way they touch
        # it is the documented auto decode-floor, which protects a graph-hidden decode regime (profiled
        # decode share here = 1000/6000 = 0.167 < 0.20) from being under-weighted in the coarse prior.
        got = self._run_main(["--isl", "1000", "--osl", "1000"])
        self.assertIsNotNone(got["serving_weight_model"])
        self.assertEqual(got["serving_weight_model"]["analytic_calls"], {"prefill": 1, "decode": 1000})
        w_with = {c["regime"]: c["weight"] for c in got["cases"]}
        norm_with = {c["regime"]: c["weight_norm"] for c in got["cases"]}
        # prefill still dominates; the floor only lifts decode to its 0.20 floor (does not invert the order).
        self.assertGreater(w_with["prefill"], w_with["decode"])
        self.assertIn("auto decode-floor", got["notes"])
        self.assertAlmostEqual(norm_with["decode"], attribute_weights._DECODE_AUTOFLOOR, places=6)   # floored up to 0.20
        self.assertAlmostEqual(norm_with["prefill"], 1.0 - attribute_weights._DECODE_AUTOFLOOR, places=6)

        # without the flags: no serving model, NO analytic calls -> the auto decode-floor cannot fire, so
        # the static `weight` is the RAW profiled TIME split (decode 1000/6000 = 0.167), NOT floored.
        base = self._run_main([])
        self.assertIsNone(base["serving_weight_model"])
        self.assertNotIn("auto decode-floor", base["notes"])
        norm_base = {c["regime"]: c["weight_norm"] for c in base["cases"]}
        self.assertAlmostEqual(norm_base["decode"], 1000.0 / 6000.0, places=6)     # raw, unfloored
        # the flags moved decode's prior share UP from the raw 0.167 to the 0.20 floor — and only that.
        self.assertGreater(norm_with["decode"], norm_base["decode"])


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

    def test_enforce_eager_not_flagged_as_strawman(self):
        # eager is used ONLY when the online regime is EXPLICITLY enforce-eager — and then eager IS the
        # faithful deployment context, NOT a strawman. So an enforce-eager regime must produce no warning
        # (consistent with deployment_graph_mode returning eager for it). The strawman is the OPPOSITE
        # case (eager baseline while the deployment replays under a graph), which is structurally
        # prevented by deployment_graph_mode, so there is nothing to warn about here.
        notes = []
        w = attribute_weights._regime_warnings(
            {"enforce_eager": True}, "gemm", [{"pct_gpu_time": 30.0}], 30.0, 2.0, notes)
        self.assertNotIn("strawman", w)
        self.assertEqual(w, "")

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


class TestRandomVsBaseline(unittest.TestCase):
    """harness_lib.check_random_vs_baseline — value-parity vs the live frozen baseline on random input
    DRAWS at FIXED online shapes. Correctness is a hard gate; speedup is report-only. Torch-guarded."""

    def _shapes(self, torch):
        def mk(rng):
            # build on the generator's device: check_random_vs_baseline seeds rng on the run device
            # (cuda when available), and torch.randn requires the tensor and generator share a device.
            return torch.randn(64, 128, generator=rng, device=rng.device)
        return [{"sig": "M=64", "make_inputs": mk}]

    def test_identical_all_correct(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")
        ok, pc = harness_lib.check_random_vs_baseline(
            lambda a: a * 2.0, lambda a: a * 2.0, self._shapes(torch), tol=2e-2,
            draws=3, warmup=1, repeats=3)
        self.assertTrue(ok)
        self.assertEqual(len(pc), 3)
        self.assertTrue(all(p["correct"] for p in pc))
        # dims fixed per sig — NOT random shapes; only values vary across draws
        self.assertEqual({p["case"].split(":", 1)[1] for p in pc}, {"M=64"})

    def test_wrong_draw_fails_gate(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")
        # Wrong on exactly ONE draw. Key on the INPUT tensor identity (each draw builds its args once
        # and reuses the same object for the correctness check AND the timing calls), so the injected
        # error is deterministic per draw and NOT perturbed by how many times time_op invokes cur.
        seen = []

        def cur(a):
            key = id(a)
            if key not in seen:
                seen.append(key)
            return a * 2.0 + (5.0 if seen.index(key) == 1 else 0.0)   # diverge on the 2nd draw

        ok, pc = harness_lib.check_random_vs_baseline(
            lambda a: a * 2.0, cur, self._shapes(torch), tol=2e-2,
            draws=3, warmup=0, repeats=1)
        self.assertFalse(ok)
        self.assertFalse(all(p["correct"] for p in pc))
        self.assertTrue(any(p["correct"] for p in pc))   # a single bad draw fails the whole gate

    def test_shared_buffer_baseline_snapshotted(self):
        """A baseline that returns a persistent buffer must still be compared against its SNAPSHOT, so a
        divergent candidate is caught (not masked by aliasing)."""
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")
        static = torch.zeros(64, 128)

        def base_static(a):
            static.copy_(a * 2.0)
            return static

        ok, pc = harness_lib.check_random_vs_baseline(
            base_static, lambda a: a * 3.0, self._shapes(torch), tol=2e-2,
            draws=1, warmup=0, repeats=1)
        self.assertFalse(ok)


class TestConcAndServedCalls(unittest.TestCase):
    """P1 fix: CONC enters prefill through the launch COUNT (×CONC) and decode through the SHAPE (M),
    so it must NOT cancel; and the served-regimes gate zeroes a regime the kernel does not run in."""
    def test_conc_scales_prefill_only(self):
        est = attribute_weights.estimate_serving_regime_calls(8192, 1024, conc=32)
        self.assertEqual(est, {"prefill": 32, "decode": 1024})   # prefill = 32*ceil(8192/8192)

    def test_conc_with_chunked_prefill(self):
        est = attribute_weights.estimate_serving_regime_calls(8192, 1024, conc=32, prefill_chunk=4096)
        self.assertEqual(est, {"prefill": 64, "decode": 1024})   # 32 * ceil(8192/4096)=32*2

    def test_conc_default_is_backward_compatible(self):
        self.assertEqual(attribute_weights.estimate_serving_regime_calls(1000, 1000),
                         {"prefill": 1, "decode": 1000})

    def test_served_gate_zeroes_unserved_regime(self):
        est = attribute_weights.estimate_serving_regime_calls(8192, 1024, conc=32, served=["prefill"])
        self.assertEqual(est, {"prefill": 32, "decode": 0})      # decode never runs on this kernel
        est2 = attribute_weights.estimate_serving_regime_calls(8192, 1024, conc=32, served=["decode"])
        self.assertEqual(est2, {"prefill": 0, "decode": 1024})


class TestServingWeightedSpeedup(unittest.TestCase):
    """The centralized PRIMARY metric: served gate + analytic calls (never profile counts) + identity guard."""
    def _meta(self, served=None, calls=None):
        return {"served_regimes": served,
                "workload": {"serving_weight_model": {"analytic_calls": calls or {"prefill": 32, "decode": 1024}}}}

    def test_served_drops_decode_on_prefill_only_kernel(self):
        # a decode bucket that leaked into a prefill-only kernel must NOT be weighted (the gqa bug).
        per_case = [
            {"sig": "prefill_m8192", "regime": "prefill", "m": 8192, "baseline_ms": 10.0, "optimized_ms": 9.0},
            {"sig": "decode_m32", "regime": "decode", "m": 32, "baseline_ms": 2.0, "optimized_ms": 1.0},  # 2x
        ]
        r = harness_lib.serving_weighted_speedup(per_case, self._meta(served=["prefill"]))
        self.assertIn("decode_m32", r["dropped_unserved"])
        # only prefill survives -> weighted ~= prefill speedup (10/9), NOT dominated by the decode 2x.
        self.assertAlmostEqual(r["weighted"], 10.0 / 9.0, places=6)
        self.assertEqual(r["included"], 1)

    def test_identity_bucket_excluded(self):
        per_case = [
            {"sig": "prefill_m8192", "regime": "prefill", "m": 8192, "baseline_ms": 10.0, "optimized_ms": 9.0},
            {"sig": "decode_m32", "regime": "decode", "m": 32, "baseline_ms": 2.0, "optimized_ms": 2.0},  # identity
        ]
        r = harness_lib.serving_weighted_speedup(per_case, self._meta())
        self.assertIn("decode_m32", r["suspect_identity"])
        self.assertAlmostEqual(r["weighted"], 10.0 / 9.0, places=6)   # decode identity excluded

    def test_all_identity_returns_none_untrusted(self):
        per_case = [
            {"sig": "prefill_m8192", "regime": "prefill", "m": 8192, "baseline_ms": 10.0, "optimized_ms": 10.0},
            {"sig": "decode_m32", "regime": "decode", "m": 32, "baseline_ms": 2.0, "optimized_ms": 2.0},
        ]
        r = harness_lib.serving_weighted_speedup(per_case, self._meta())
        self.assertIsNone(r["weighted"])
        self.assertTrue(r["reason"])

    def test_conc_dominant_bucket_carries_passes(self):
        # decode passes (1024) land on the largest-M decode bucket; smaller decode bucket gets calls=1.
        per_case = [
            {"sig": "decode_m32", "regime": "decode", "m": 32, "baseline_ms": 2.0, "optimized_ms": 1.0},   # 2x, calls 1024
            {"sig": "decode_m1", "regime": "decode", "m": 1, "baseline_ms": 0.5, "optimized_ms": 0.5},     # identity -> excluded
        ]
        r = harness_lib.serving_weighted_speedup(per_case, self._meta(calls={"decode": 1024}))
        dom = [c for c in r["per_case"] if c["sig"] == "decode_m32"][0]
        self.assertEqual(dom["calls"], 1024)


# --------------------------------------------------------------------------- #
# parse_profile.py — serving-phase (prefill/decode) accounting from the trace's
# gpu_user_annotation step spans (skill kernel-phase-accounting integration).
# --------------------------------------------------------------------------- #
def _synthetic_trace():
    """Tiny Kineto-style trace: 1 prefill(mixed) step (M=8192) + 2 pure-decode steps (batch 8).
    fused_moe fires in all 3 (both), _gqa_sparse_fwd only in prefill, reduce only in decode."""
    ev = [
        # step spans on the GPU timeline (perfskills dialect)
        {"cat": "gpu_user_annotation", "name": "execute_context_2(8192)_generation_0(0)",
         "ph": "X", "ts": 0.0, "dur": 100.0},
        {"cat": "gpu_user_annotation", "name": "execute_context_0(0)_generation_8(8)",
         "ph": "X", "ts": 100.0, "dur": 10.0},
        {"cat": "gpu_user_annotation", "name": "execute_context_0(0)_generation_8(8)",
         "ph": "X", "ts": 110.0, "dur": 10.0},
        # kernels
        {"cat": "kernel", "ph": "X", "ts": 5.0, "dur": 40.0, "name": "fused_moe_kernel", "args": {}},
        {"cat": "kernel", "ph": "X", "ts": 6.0, "dur": 30.0, "name": "_gqa_sparse_fwd_kernel", "args": {}},
        {"cat": "kernel", "ph": "X", "ts": 101.0, "dur": 2.0, "name": "fused_moe_kernel", "args": {}},
        {"cat": "kernel", "ph": "X", "ts": 102.0, "dur": 1.0, "name": "cross_device_reduce_1stage", "args": {}},
        {"cat": "kernel", "ph": "X", "ts": 111.0, "dur": 2.0, "name": "fused_moe_kernel", "args": {}},
        {"cat": "kernel", "ph": "X", "ts": 112.0, "dur": 1.0, "name": "cross_device_reduce_1stage", "args": {}},
    ]
    return {"traceEvents": ev}


class TestPhaseAccounting(unittest.TestCase):
    def setUp(self):
        self.trace = _write_json(_synthetic_trace())
        self.agg, self.total, self.launch, self.pmeta = parse_profile.parse_torch_trace(self.trace)

    def tearDown(self):
        os.unlink(self.trace)

    def test_step_counts_and_phase_meta(self):
        self.assertTrue(self.pmeta["has_annotations"])
        self.assertEqual(self.pmeta["n_prefill_steps"], 1)
        self.assertEqual(self.pmeta["n_decode_steps"], 2)
        self.assertEqual(self.pmeta["prefill_tokens"], 8192)
        self.assertEqual(self.pmeta["decode_batches"], [8, 8])

    def test_kernel_phase_attribution(self):
        self.assertEqual(self.agg["fused_moe_kernel"]["by_phase"]["prefill"]["count"], 1)
        self.assertEqual(self.agg["fused_moe_kernel"]["by_phase"]["decode"]["count"], 2)
        # prefill-only and decode-only kernels stay in their single phase
        self.assertEqual(set(self.agg["_gqa_sparse_fwd_kernel"]["by_phase"]), {"prefill"})
        self.assertEqual(set(self.agg["cross_device_reduce_1stage"]["by_phase"]), {"decode"})

    def test_summary_steady_and_est_calls(self):
        # conc==8 == captured decode batch -> steady; est_calls decode==OSL, prefill==CONC*ceil(ISL/chunk)
        s = parse_profile.build_summary(self.agg, self.total, self.launch, "torch-trace", 10,
                                        conc=8, isl=8192, osl=1024, chunk=8192,
                                        capture_sizes=[8, 16, 32], phase_meta=self.pmeta)
        self.assertTrue(s["serving"]["steady"])
        self.assertEqual(s["serving"]["analytic_calls"], {"prefill": 8, "decode": 1024})
        by = {k["short_name"]: k for k in s["top_kernels"]}
        moe = by["fused_moe_kernel"]
        self.assertEqual(moe["phase"], "both")
        self.assertEqual(sorted(moe["served_regimes"]), ["decode", "prefill"])
        self.assertEqual(moe["est_calls"], {"prefill": 8, "decode": 1024})
        self.assertEqual(moe["est_shape"]["prefill"]["M"], 8192)
        self.assertEqual(moe["est_shape"]["decode"]["M"], 8)           # conc snapped to capture size 8
        self.assertEqual(by["_gqa_sparse_fwd_kernel"]["phase"], "prefill")
        self.assertEqual(by["cross_device_reduce_1stage"]["phase"], "decode")

    def test_summary_non_steady_warns(self):
        # captured decode batch 8 << concurrency 32 -> not steady; counts still valid
        s = parse_profile.build_summary(self.agg, self.total, self.launch, "torch-trace", 10,
                                        conc=32, isl=8192, osl=1024, chunk=8192,
                                        capture_sizes=[8, 16, 32], phase_meta=self.pmeta)
        self.assertFalse(s["serving"]["steady"])
        self.assertEqual(s["serving"]["decode_batch_captured"], 8)
        self.assertEqual(s["serving"]["decode_batch_steady"], 32)
        # decode est_shape M snaps concurrency 32 up to capture size 32
        moe = {k["short_name"]: k for k in s["top_kernels"]}["fused_moe_kernel"]
        self.assertEqual(moe["est_shape"]["decode"]["M"], 32)

    def test_workload_case_regime_and_served_from_profile(self):
        wl = parse_profile.build_workload(self.agg, self.total, 10)
        by = {k["short_name"]: k for k in wl["kernels"]}
        self.assertEqual(sorted(by["fused_moe_kernel"]["served_regimes"]), ["decode", "prefill"])
        # attribute_weights derives the gate from the profile's measured phase
        self.assertEqual(attribute_weights.served_from_profile([by["fused_moe_kernel"]]),
                         {"prefill", "decode"})
        self.assertEqual(attribute_weights.served_from_profile([by["_gqa_sparse_fwd_kernel"]]),
                         {"prefill"})

    def test_analytic_calls_matches_shared_impl(self):
        self.assertEqual(parse_profile.analytic_regime_calls(8192, 1024, 32, 8192),
                         attribute_weights.estimate_serving_regime_calls(8192, 1024, 32, prefill_chunk=8192))

    def test_no_annotations_backward_compatible(self):
        # a trace without execute_* spans yields no phase fields (old behavior preserved)
        ev = {"traceEvents": [{"cat": "kernel", "ph": "X", "ts": 1.0, "dur": 5.0,
                               "name": "fused_moe_kernel", "args": {}}]}
        p = _write_json(ev)
        try:
            agg, tot, lau, pm = parse_profile.parse_torch_trace(p)
            self.assertEqual(pm, {})
            s = parse_profile.build_summary(agg, tot, lau, "torch-trace", 5)
            self.assertNotIn("serving", s)
            self.assertNotIn("phase", s["top_kernels"][0])
        finally:
            os.unlink(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
