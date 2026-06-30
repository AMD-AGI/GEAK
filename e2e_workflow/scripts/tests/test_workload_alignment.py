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


class TestQuantStamping(unittest.TestCase):
    """The regime's job that REMAINS: stamp per-operand dtype/quant so the harness builds in-regime
    operands (fp8 + scales, not bf16). The old _regime_warnings gate was removed by design."""
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

    def test_no_regime_warnings_attr(self):
        # the gate machinery must be gone
        self.assertFalse(hasattr(attribute_weights, "_regime_warnings"))


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
                os.path.exists(p) and os.unlink(p)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
