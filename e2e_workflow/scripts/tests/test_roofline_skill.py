"""Unit tests for the `roofline` analysis skill (stdlib only; no pytest, no GPU, no model needed).

Two things are locked here.

1. **Calibration against a real run.** The numbers below are the measured profile of
   Qwen3.5-35B-A3B-FP8 on gfx950 / vLLM / TP1 / isl-osl 1k / conc 64 — a run whose outcome we know.
   The skill must reproduce the call it would have made:
     fused_moe   26.45% GPU, ~88% of roofline -> saturated, ~1.02x attainable  (measured: 1.047x
                 isolated, -0.064% e2e -> the budget spent there was wasted)
     paged_attn   8.86% GPU, ~18% of roofline -> underperforming, >1.4x        (measured: 1.56x isolated)
   The load-bearing assertion is `test_rankings_disagree`: ranking by pct_gpu_time puts MoE first,
   ranking by roofline headroom puts attention first. That inversion is the entire point of the skill.

2. **Degradation is non-fatal.** Every level of the SKILL.md ladder returns a value instead of raising,
   so a bad peak table / unmodellable op / impossible result / missing counter cannot fail a run.

3. **One data source.** Peaks come from perf_knowledge/hardware/data/sku.json (peaks.md is its generated
   rendering) and bound thresholds from thresholds.json; the tests below pin both ties.
"""
import json
import importlib.util
import os
import sys
import unittest

E2E_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(E2E_DIR, "knowledge", "analysis_skills", "roofline"))
import roofline_tools as rt  # noqa: E402

PEAKS_MD = os.path.join(os.path.dirname(os.path.abspath(rt.__file__)), "peaks.md")
SKU_JSON = rt.SKU_JSON
REPO = os.path.dirname(E2E_DIR)
KT_DIR = os.path.join(REPO, "kernel_workflow", "scripts", "kernel_tools")

_IDENTITY_SPEC = importlib.util.spec_from_file_location(
    "gpu_identity", os.path.join(os.path.dirname(E2E_DIR), "scripts", "gpu_identity.py"))
gpu_identity = importlib.util.module_from_spec(_IDENTITY_SPEC)
_IDENTITY_SPEC.loader.exec_module(gpu_identity)


def _identity(gfx, marketing_name, cu):
    """Structured identity exactly as the workflow's rocminfo probe reports it."""
    return gpu_identity.parse_rocminfo(
        "*******\nAgent 1\n*******\n"
        "  Name:                    %s\n"
        "  Marketing Name:          %s\n"
        "  Compute Unit:            %d\n" % (gfx, marketing_name, cu))

# --- measured profile of the reference run (see module docstring) ---------------------------------
MOE = dict(E=256, M=64, top_k=8, hidden=2048, inter=512, layers=40,
           t_launch_s=49.471e-6, launches_per_step=80, pct_gpu=26.45)
ATTN = dict(batch=64, isl=1024, osl=1024, kv_heads=2, q_heads=16, head_dim=256,
            t_launch_s=141.112e-6, pct_gpu=8.86)


def _peaks():
    return rt.load_peaks(SKU_JSON, "gfx950")


def _moe_metrics(all_experts=False):
    p = _peaks()
    per_expert_elems = 2 * MOE["inter"] * MOE["hidden"] + MOE["hidden"] * MOE["inter"]
    n = MOE["E"] if all_experts else rt.experts_hit(MOE["E"], MOE["M"] * MOE["top_k"])
    # 80 launches/step over 40 layers => 2 launches per logical layer (SKILL.md "per-launch, not aggregate")
    t_layer = (MOE["launches_per_step"] / MOE["layers"]) * MOE["t_launch_s"]
    return rt.roofline_metrics(
        n * per_expert_elems * rt.dtype_bytes("fp8"),
        2 * MOE["M"] * MOE["top_k"] * per_expert_elems,
        t_layer, p["hbm_bw_bytes_s"], rt.peak_flops_for(p, "fp8"),
        rt.TARGET_EFF["moe"], pct_gpu_time=MOE["pct_gpu"])


def _attn_metrics():
    p = _peaks()
    seq = ATTN["isl"] + ATTN["osl"] // 2          # mid-run average context
    kv_bytes = ATTN["batch"] * seq * ATTN["kv_heads"] * ATTN["head_dim"] * 2 * rt.dtype_bytes("bf16")
    flops = 2 * ATTN["batch"] * ATTN["q_heads"] * seq * ATTN["head_dim"] * 2
    return rt.roofline_metrics(kv_bytes, flops, ATTN["t_launch_s"], p["hbm_bw_bytes_s"],
                               rt.peak_flops_for(p, "bf16"), rt.TARGET_EFF["attn"],
                               pct_gpu_time=ATTN["pct_gpu"])


class TestPeaks(unittest.TestCase):
    def test_known_gfx_from_table(self):
        p = _peaks()
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p["hbm_bw_bytes_s"], 8.0e12, delta=1e9)
        self.assertAlmostEqual(rt.peak_flops_for(p, "fp8"), 5.0e15, delta=1e12)
        self.assertEqual(p["source"], "table")
        self.assertEqual(p["confidence"], "high")

    def test_gfx950_profile_dtype_aliases_keep_their_compute_peak(self):
        p = _peaks()
        expected = {
            "fp8_e4m3": 5.0e15,
            "fp8_w8a8": 5.0e15,
            "float8_e4m3fn": 5.0e15,
            "torch.float8_e5m2fnuz": 5.0e15,
            "half": 2.5e15,
            "torch.half": 2.5e15,
            "bfloat16": 2.5e15,
            "float16": 2.5e15,
            "float32": 1.573e14,
        }
        for dtype, peak in expected.items():
            with self.subTest(dtype=dtype):
                self.assertAlmostEqual(
                    rt.peak_flops_for(p, dtype), peak, delta=peak * 1e-6
                )
        self.assertEqual(rt.dtype_bytes("torch.float8_e4m3fnuz"), 1)
        self.assertEqual(rt.dtype_bytes("half"), 2)

    def test_peaks_md_path_resolves_to_the_same_sku_row(self):
        """peaks.md is generated from sku.json; pointing at it must give the json's row."""
        for gfx, product in (("gfx950", None), ("gfx942", None), ("gfx1151", None),
                             ("gfx1201", "r9700"), ("gfx950", "mi350x")):
            with self.subTest(gfx=gfx, product=product):
                self.assertEqual(rt.load_peaks(PEAKS_MD, gfx, product=product),
                                 rt.load_peaks(SKU_JSON, gfx, product=product))

    def test_gfx942_also_tabulated(self):
        p = rt.load_peaks(SKU_JSON, "gfx942")
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p["hbm_bw_bytes_s"], 5.3e12, delta=1e9)

    def test_gfx1151_row(self):
        """RDNA3.5 APU: memory axis, CU count, and the theoretical WMMA compute peaks."""
        p = rt.load_peaks(SKU_JSON, "gfx1151")
        self.assertIsNotNone(p)
        # the e2e roofline ranks on the DATASHEET pin rate; sku.json stores the measured 0.212
        # TB/s as the row's ceiling (peak_hbm_basis=measured) and 0.256 as datasheet_hbm_tb_s
        self.assertAlmostEqual(p["hbm_bw_bytes_s"], 256.0e9, delta=1e9)
        self.assertEqual(p["denominator_basis"], "datasheet")
        self.assertEqual(p["cu"], 40)
        self.assertEqual(p["source"], "table")
        # 512 ops/clk/CU x 40 CU x 2.9 GHz; dual-issue fp32 256 FLOP/clk/CU x 40 x 2.9 GHz
        self.assertAlmostEqual(rt.peak_flops_for(p, "bf16"), 59.4e12, delta=1e10)
        self.assertAlmostEqual(rt.peak_flops_for(p, "fp32"), 29.7e12, delta=1e10)
        self.assertEqual((p["l2_bytes"], p["mall_bytes"]), (2 << 20, 32 << 20))

    def test_gfx1151_metrics_see_both_axes(self):
        """Both roofs must be live, not just the memory one. A high-AI kernel at 67% of the WMMA
        peak has to come back compute-bound with ~1.35x attainable — if the compute peak were
        missing, compute_util would read a hard 0.0 and the same kernel would be sold as
        latency-bound with 9x headroom off the 10% memory utilization."""
        p = rt.load_peaks(SKU_JSON, "gfx1151")
        m = rt.roofline_metrics(2.56e6, 4e9, 100e-6, p["hbm_bw_bytes_s"],
                                rt.peak_flops_for(p, "bf16"), 0.90, pct_gpu_time=20.0)
        self.assertEqual(m["bound_type"], "compute")
        self.assertAlmostEqual(m["compute_util"], 40.0 / 59.4, places=3)
        self.assertAlmostEqual(m["hbm_util"], 0.1, places=3)
        self.assertIsNotNone(m["ridge_point"])
        self.assertLess(m["attainable_speedup"], 2.0)

    def test_bf16_equals_fp16_on_every_arch_that_tabulates_flops(self):
        """peaks.md's own load-bearing cross-check: the matrix core runs both at the same rate
        (MFMA on CDNA, WMMA on RDNA), so an inequality means the compute axis is inflated."""
        for gfx in ("gfx942", "gfx950", "gfx1151"):
            p = rt.load_peaks(SKU_JSON, gfx)
            self.assertEqual(p["flops"]["bf16"], p["flops"]["fp16"], gfx)
        for product, gfx in (("mi350x", "gfx950"), ("mi325x", "gfx942"), ("mi308x", "gfx942"),
                             ("r9700", "gfx1201")):
            p = rt.load_peaks(SKU_JSON, gfx, product=product)
            self.assertEqual(p["flops"]["bf16"], p["flops"]["fp16"], product)
            
    def test_r9700_peaks_require_product_identity(self):
        self.assertIsNone(rt.resolve_peaks(SKU_JSON, "gfx1201"))
        self.assertIsNone(rt.resolve_peaks(PEAKS_MD, "gfx1201"))
        p = rt.resolve_peaks(SKU_JSON, "gfx1201", product="r9700")
        self.assertIsNotNone(p)
        self.assertEqual(p["source"], "table")
        self.assertEqual(p["confidence"], "high")
        self.assertEqual(p.get("product"), "r9700")
        self.assertEqual(p["cu"], 64)
        self.assertAlmostEqual(p["hbm_bw_bytes_s"], 6.4e11, delta=1e8)
        self.assertAlmostEqual(rt.peak_flops_for(p, "bf16"), 1.91e14, delta=1e11)
        self.assertAlmostEqual(rt.peak_flops_for(p, "fp16"), 1.91e14, delta=1e11)
        self.assertAlmostEqual(rt.peak_flops_for(p, "fp32"), 4.78e13, delta=1e11)
        self.assertAlmostEqual(rt.peak_flops_for(p, "fp8"), 3.83e14, delta=1e11)
        for alias in (
            "fp8_e4m3",
            "fp8_e5m2",
            "fp8_w8a8",
            "float8_e4m3fn",
            "torch.float8_e5m2fnuz",
        ):
            self.assertAlmostEqual(
                rt.peak_flops_for(p, alias), 3.83e14, delta=1e11
            )
        self.assertAlmostEqual(rt.peak_flops_for(p, "int8"), 3.83e14, delta=1e11)
        self.assertIsNone(rt.peak_flops_for(p, "fp4"))
        self.assertNotIn("fp4", p["flops"])

    def test_instinct_identity_selects_its_product_row(self):
        """gpu_identity now names Instinct products. The product row must be found (never derived),
        and where the product IS the arch default its numbers equal the bare-gfx lookup."""
        original = rt.derive_peaks_from_props

        def no_derive(device=0):
            raise AssertionError("tabulated CDNA peaks fell back to device properties")

        rt.derive_peaks_from_props = no_derive
        try:
            for gfx, name, cu, key, sku in (
                    ("gfx950", "AMD Instinct MI355X", 256, "mi355x", "MI355X"),
                    ("gfx950", "AMD Instinct MI350X", 256, "mi350x", "MI350X"),
                    ("gfx942", "AMD Instinct MI300X", 304, "mi300x", "MI300X"),
                    ("gfx942", "AMD Instinct MI325X", 304, "mi325x", "MI325X")):
                with self.subTest(name=name):
                    identity = _identity(gfx, name, cu)
                    # the product-pin contract is untouched; the row name rides in `sku`
                    self.assertEqual(identity["target"], "unknown")
                    self.assertEqual(identity["sku"], key)
                    self.assertEqual(rt.product_from_identity(identity), key)
                    p = rt.resolve_peaks(SKU_JSON, identity["gfx"], identity=identity)
                    self.assertEqual(p["sku"], sku)
                    self.assertEqual(p.get("product"), key)
                    # passing only `target` (the old call) still lands on the arch default
                    self.assertEqual(rt.resolve_peaks(SKU_JSON, gfx, product=identity["target"]),
                                     rt.load_peaks(SKU_JSON, gfx))
                    if key not in ("mi355x", "mi300x"):
                        continue
                    bare = rt.load_peaks(SKU_JSON, gfx)
                    self.assertEqual({k: v for k, v in p.items() if k != "product"}, bare)
                    self.assertEqual(p["source"], "table")
                    self.assertEqual(p["confidence"], "high")
        finally:
            rt.derive_peaks_from_props = original

    def test_unknown_product_identity_keeps_instinct_tables(self):
        """A partitioned or unlisted Instinct card reports target=unknown. Passing that straight
        through as `product` must still find the ISA-keyed default rows, not derive."""
        original = rt.derive_peaks_from_props

        def no_derive(device=0):
            raise AssertionError("tabulated CDNA peaks fell back to device properties")

        rt.derive_peaks_from_props = no_derive
        try:
            for gfx, name, cu in (("gfx950", "AMD Instinct MI355X", 32),     # CPX partition
                                  ("gfx942", "AMD Instinct MI300X VF", 304)):
                with self.subTest(gfx=gfx):
                    identity = _identity(gfx, name, cu)
                    self.assertEqual((identity["target"], identity["sku"]), ("unknown", "unknown"))
                    p = rt.resolve_peaks(SKU_JSON, identity["gfx"], product=identity["target"])
                    self.assertEqual(p, rt.load_peaks(SKU_JSON, gfx))
                    self.assertEqual(rt.resolve_peaks(SKU_JSON, gfx, identity=identity), p)
                    self.assertNotIn("product", p)
        finally:
            rt.derive_peaks_from_props = original

    def test_product_rows_carry_their_own_peaks(self):
        """MI350X is not MI355X and MI325X is not MI300X: a product identity must reach the row
        with its own numbers (the bare-gfx default would quote the sibling's)."""
        mi350 = rt.load_peaks(SKU_JSON, "gfx950", product="mi350x")
        self.assertAlmostEqual(rt.peak_flops_for(mi350, "bf16"), 2.3e15, delta=1e9)
        self.assertAlmostEqual(rt.peak_flops_for(mi350, "fp4"), 9.2e15, delta=1e9)
        self.assertAlmostEqual(rt.peak_flops_for(mi350, "fp32"), 1.442e14, delta=1e9)
        mi355 = rt.load_peaks(SKU_JSON, "gfx950")
        self.assertAlmostEqual(rt.peak_flops_for(mi355, "fp4"), 1.0e16, delta=1e9)
        self.assertAlmostEqual(rt.peak_flops_for(mi355, "fp64"), 7.86e13, delta=1e9)
        mi325 = rt.load_peaks(SKU_JSON, "gfx942", product="mi325x")
        self.assertAlmostEqual(mi325["hbm_bw_bytes_s"], 6.0e12, delta=1e9)
        self.assertEqual(mi325["flops"], rt.load_peaks(SKU_JSON, "gfx942")["flops"])
        # a product whose arch disagrees with the gfx is ignored, not trusted
        self.assertEqual(rt.load_peaks(SKU_JSON, "gfx942", product="mi355x"),
                         rt.load_peaks(SKU_JSON, "gfx942"))
        # MI308X is ORC-derived: tabulated, but not high confidence
        self.assertEqual(rt.load_peaks(SKU_JSON, "gfx942", product="mi308x")["confidence"], "medium")

    def test_unsupported_rows_never_resolve(self):
        """gfx1100 / gfx1200 rows are kept in sku.json for reference only."""
        for gfx in ("gfx1100", "gfx1200"):
            self.assertIsNone(rt.load_peaks(SKU_JSON, gfx))
        self.assertIsNone(rt.resolve_peaks(SKU_JSON, "gfx1200"))

    def test_identity_product_still_scopes_the_r9700_table(self):
        r9700 = _identity("gfx1201", gpu_identity.R9700_MARKETING_NAME, 64)
        p = rt.resolve_peaks(SKU_JSON, r9700["gfx"], product=r9700["target"])
        self.assertIsNotNone(p)
        self.assertEqual(p.get("product"), "r9700")
        self.assertEqual(rt.resolve_peaks(SKU_JSON, r9700["gfx"], identity=r9700), p)
        other = _identity("gfx1201", "Another gfx1201 Product", 64)
        self.assertEqual(other["target"], "unknown")
        self.assertIsNone(rt.resolve_peaks(SKU_JSON, other["gfx"], product=other["target"]))
        self.assertIsNone(rt.resolve_peaks(SKU_JSON, other["gfx"], identity=other))

    def test_product_key_precedence_is_sku_then_target(self):
        self.assertEqual(rt.product_from_identity({"sku": "mi350x", "target": "unknown"}), "mi350x")
        self.assertEqual(rt.product_from_identity({"sku": "unknown", "target": "r9700"}), "r9700")
        self.assertEqual(rt.product_from_identity({"target": "r9700"}), "r9700")   # pre-sku record
        self.assertEqual(rt.product_from_identity({"sku": "unknown", "target": "unknown"}), "")
        self.assertEqual(rt.product_from_identity(sku=None, target="unknown"), "")
        self.assertEqual(rt.product_from_identity(sku="mi325x", target="unknown"), "mi325x")

    def test_peak_flops_for_unknown_dtype_is_none_not_table_max(self):
        p = _peaks()
        self.assertIsNone(rt.peak_flops_for(p, "mystery_dtype"))
        self.assertIsNone(rt.peak_flops_for(None, "bf16"))

    def test_client_rdna4_family_is_gfx120x_not_gfx1250(self):
        self.assertTrue(rt.is_client_rdna4("gfx1201"))
        self.assertTrue(rt.is_client_rdna4("gfx1200"))
        self.assertFalse(rt.is_client_rdna4("gfx1250"))
        self.assertFalse(rt.is_client_rdna4("gfx950"))

    def test_unmeasured_gfx120x_is_hard_unknown_not_derived(self):
        self.assertIsNone(rt.resolve_peaks("/nonexistent/peaks.md", "gfx1209"))

    def test_gfx1250_missing_table_may_derive(self):
        called = []
        original = rt.derive_peaks_from_props

        def fake_derive(device=0):
            called.append(device)
            return None

        rt.derive_peaks_from_props = fake_derive
        try:
            self.assertIsNone(rt.resolve_peaks("/nonexistent/peaks.md", "gfx1250"))
            self.assertEqual(called, [0])
            called.clear()
            self.assertIsNone(rt.resolve_peaks("/nonexistent/peaks.md", "gfx1209"))
            self.assertEqual(called, [])
        finally:
            rt.derive_peaks_from_props = original

    def test_L1_unknown_gfx_returns_none(self):
        """Unknown gfx -> None, so the caller falls back to derived peaks at confidence=low."""
        self.assertIsNone(rt.load_peaks(SKU_JSON, "gfx-does-not-exist"))
        self.assertIsNone(rt.load_peaks(PEAKS_MD, "gfx-does-not-exist"))

    def test_L5_missing_peaks_file_does_not_raise(self):
        self.assertIsNone(rt.load_peaks("/nonexistent/peaks.md", "gfx950"))
        self.assertIsNone(rt.load_peaks("/nonexistent/sku.json", "gfx950"))

    def test_peaks_md_fallback_parser_matches_sku_json(self):
        """If sku.json is unreachable the generated peaks.md blocks are parsed instead; they must
        give the same answer as the json for every reachable row (minus json-only provenance)."""
        with open(PEAKS_MD, encoding="utf-8") as fh:
            text = fh.read()
        with open(SKU_JSON, encoding="utf-8") as fh:
            skus = json.load(fh)["skus"]
        drop = ("basis", "measured_ceilings", "confidence")
        n = 0
        for name, row in skus.items():
            if row.get("geak_support") != "supported":
                continue
            cases = []
            if row.get("identity_target"):
                cases.append(row["identity_target"])
            if row.get("roofline_default_for_arch"):
                cases.append(None)
            for product in cases:
                with self.subTest(sku=name, product=product):
                    want = rt.load_peaks(SKU_JSON, row["arch"], product=product)
                    got = rt._peaks_from_md(text, row["arch"], product=product)
                    self.assertIsNotNone(got)
                    for k in drop:
                        want.pop(k, None)
                        got.pop(k, None)
                    self.assertEqual(got, want)
                    n += 1
        self.assertGreaterEqual(n, 8)

    def test_peaks_md_generated_block_is_current(self):
        """peaks.md is DERIVED documentation: its generated block must equal a fresh render."""
        sys.path.insert(0, KT_DIR)
        try:
            import extract_sku  # noqa: PLC0415
        finally:
            sys.path.remove(KT_DIR)
        self.assertEqual(extract_sku.check_docs(extract_sku.load(SKU_JSON), [PEAKS_MD]), [])


class TestCalibrationMoE(unittest.TestCase):
    """The 26%-of-GPU head that was, in fact, already at the bandwidth wall."""

    def test_memory_bound(self):
        m = _moe_metrics()
        self.assertEqual(m["bound_type"], "memory")
        self.assertLess(m["arithmetic_intensity"], m["ridge_point"] / 10)

    def test_near_roofline(self):
        m = _moe_metrics()
        self.assertGreaterEqual(m["roofline_pct"], 0.85)
        self.assertLessEqual(m["roofline_pct"], 0.92)

    def test_saturated_and_no_meaningful_headroom(self):
        m = _moe_metrics()
        self.assertEqual(m["headroom_class"], "saturated")
        self.assertLess(m["attainable_speedup"], 1.10)
        # measured e2e was -0.064%, i.e. inside the 0.5% noise band -> prediction must agree
        self.assertLess(m["expected_e2e_gain_pct"], 1.0)

    def test_88pct_against_a_90pct_target_is_saturated_not_moderate(self):
        """Regression: banding on target_eff, not on the raw roofline.

        Classifying 0.88 against a 0.90 target as `moderate` would route this head back to the
        kernel track — exactly the wasted budget this skill exists to prevent.
        """
        self.assertEqual(rt.classify_headroom(0.88, 0.90), "saturated")
        self.assertEqual(rt.classify_headroom(0.60, 0.90), "moderate")
        self.assertEqual(rt.classify_headroom(0.20, 0.90), "underperforming")


class TestCalibrationAttention(unittest.TestCase):
    """The 8.86%-of-GPU kernel that actually had the headroom (measured 1.56x isolated)."""

    def test_underperforming_with_real_headroom(self):
        a = _attn_metrics()
        self.assertEqual(a["headroom_class"], "underperforming")
        self.assertGreater(a["attainable_speedup"], 1.4)

    def test_prediction_brackets_the_measured_speedup(self):
        """target_eff=0.50 predicts ~1.7-2.8x; the squad measured 1.56x. A prior that predicted
        below the measured value would be the dangerous direction (it under-ranks real work)."""
        self.assertGreaterEqual(_attn_metrics()["attainable_speedup"], 1.56)


class TestBoundTypeByUtilization(unittest.TestCase):
    """bound_type is decided by measured utilization on BOTH axes, not by AI-vs-ridge alone."""

    def test_attention_is_latency_bound_but_keeps_its_verdict(self):
        """Small AI + low HBM util is NOT memory-bound — it is latency/occupancy-bound. The load-bearing
        part: the verdict is KEPT (real headroom), so a low-utilization head still ranks by headroom
        rather than falling back to raw %GPU. Only the *lever class* changes (occupancy/fusion)."""
        a = _attn_metrics()
        self.assertEqual(a["bound_type"], "latency")           # was mislabeled "memory" pre-fix
        self.assertLess(a["hbm_util"], 0.60)
        self.assertLess(a["compute_util"], 0.60)
        self.assertEqual(a["headroom_class"], "underperforming")  # verdict survives
        self.assertGreater(a["attainable_speedup"], 1.4)

    def test_saturated_memory_head_stays_memory_bound(self):
        """A genuinely bandwidth-bound head (high hbm_util) keeps bound_type='memory' so it routes to
        the byte-reduction track, not the occupancy track."""
        m = _moe_metrics()
        self.assertEqual(m["bound_type"], "memory")
        self.assertGreaterEqual(m["hbm_util"], 0.60)
        self.assertEqual(m["headroom_class"], "saturated")

    def test_latency_verdict_kept_is_distinct_from_dispatch_no_verdict(self):
        """Two latency kinds must not collapse: a real-work low-util launch keeps a verdict; a launch
        timed by the dispatch floor does not."""
        p = _peaks()
        # low util, but well above the ~5us dispatch floor -> latency WITH a verdict
        real = rt.roofline_metrics(2e8, 1e8, 100e-6, p["hbm_bw_bytes_s"],
                                   rt.peak_flops_for(p, "bf16"), 0.90, pct_gpu_time=10.0)
        self.assertEqual(real["bound_type"], "latency")
        self.assertNotEqual(real["headroom_class"], "unknown")
        self.assertGreater(real["expected_e2e_gain_pct"], 0.0)
        # same shape, timed at the dispatch floor -> latency with NO verdict
        floor = rt.roofline_metrics(2e8, 1e8, 2e-6, p["hbm_bw_bytes_s"],
                                    rt.peak_flops_for(p, "bf16"), 0.90, pct_gpu_time=10.0)
        self.assertEqual(floor["bound_type"], "latency")
        self.assertEqual(floor["headroom_class"], "unknown")
        self.assertEqual(floor["expected_e2e_gain_pct"], 0.0)


class TestRankingInversion(unittest.TestCase):
    def test_rankings_disagree(self):
        """THE point of the skill: %GPU says MoE, headroom says attention — and reality agreed
        with headroom (MoE -0.064% e2e vs attention 1.56x isolated)."""
        moe, attn = _moe_metrics(), _attn_metrics()
        self.assertGreater(MOE["pct_gpu"], ATTN["pct_gpu"])                       # by %GPU: MoE first
        self.assertGreater(attn["expected_e2e_gain_pct"],
                           moe["expected_e2e_gain_pct"])                          # by headroom: attn first


class TestDegradation(unittest.TestCase):
    def test_L3_impossible_result_is_clamped_and_flagged(self):
        """Assuming ALL experts stream overshoots the roofline. That must clamp + flag `suspect`
        (and preserve the raw value), never be silently emitted or discarded."""
        m = _moe_metrics(all_experts=True)
        self.assertTrue(m["suspect"])
        self.assertGreater(m["roofline_pct_raw"], 1.0)
        self.assertLessEqual(m["roofline_pct"], 1.0)

    def test_L3_infeasible_never_yields_a_saturation_verdict(self):
        """Regression: a clamped 100% is a MODELLING FAILURE, not evidence the kernel is at the wall.

        Reading it as `saturated` would let a wrong byte model silently make a routing decision —
        observed in a real run, where fused_moe came back roofline_pct=1.00 + suspect and would
        otherwise have been classified saturated on the strength of the clamp alone.
        """
        m = _moe_metrics(all_experts=True)
        self.assertEqual(m["headroom_class"], "unknown")
        self.assertEqual(m["attainable_speedup"], 1.0)
        self.assertEqual(m["expected_e2e_gain_pct"], 0.0)
        self.assertIn("bytes_upper_bound", m)          # what the model violated, for stage C
        self.assertLess(m["bytes_upper_bound"], m["bytes_est"])

    def test_dispatch_bound_launch_gets_no_verdict(self):
        """A launch timed by dispatch overhead (tiny, high call count) is not evidence about
        bandwidth or math. Emit bound_type='latency' and no verdict — the lever is fusion."""
        p = _peaks()
        m = rt.roofline_metrics(1e5, 1e5, 2e-6, p["hbm_bw_bytes_s"],
                                rt.peak_flops_for(p, "bf16"), 0.875, pct_gpu_time=4.9)
        self.assertEqual(m["bound_type"], "latency")
        self.assertEqual(m["headroom_class"], "unknown")
        self.assertEqual(m["expected_e2e_gain_pct"], 0.0)
        self.assertIn("fusion", m["note"])

    def test_bound_type_is_a_closed_set(self):
        """Regression: a real run emitted an invented 'dispatch' bound_type the consumer had no
        routing rule for. Every path must land inside BOUND_TYPES."""
        p = _peaks()
        cases = [_moe_metrics(), _attn_metrics(), _moe_metrics(all_experts=True),
                 rt.roofline_metrics(1e5, 1e5, 2e-6, p["hbm_bw_bytes_s"],
                                     rt.peak_flops_for(p, "bf16"), 0.875)]
        for m in cases:
            self.assertIn(m["bound_type"], rt.BOUND_TYPES, m.get("note", ""))


class TestHeadScoping(unittest.TestCase):
    """Only kernels big enough to change a decision are analysed at all."""

    ENTRIES = [{"short_name": "big", "pct_gpu_time": 26.4}, {"short_name": "mid", "pct_gpu_time": 8.9},
               {"short_name": "bar", "pct_gpu_time": 5.0}, {"short_name": "small", "pct_gpu_time": 1.7},
               {"short_name": "tiny", "pct_gpu_time": 0.2}]

    def test_below_bar_is_skipped_not_degraded(self):
        sel = [e["short_name"] for e in rt.select_entries(self.ENTRIES, min_pct_gpu=5.0)]
        self.assertEqual(sel, ["big", "mid", "bar"])   # sorted desc, sub-bar absent entirely

    def test_top_n_cap(self):
        self.assertEqual(len(rt.select_entries(self.ENTRIES, min_pct_gpu=0.0, top_n=2)), 2)

    def test_malformed_input_is_safe(self):
        self.assertEqual(rt.select_entries(None), [])
        self.assertEqual(rt.select_entries([]), [])
        self.assertEqual(rt.select_entries([{"short_name": "no-pct"}], min_pct_gpu=5.0), [])

    def test_L2_unusable_input_returns_none(self):
        for bad in [(0, 0, 1e-6, 1e12, 1e15, 0.9),      # no bytes and no flops
                    (1e6, 1e6, 0, 1e12, 1e15, 0.9),     # no time
                    (1e6, 1e6, 1e-6, 0, 1e15, 0.9),     # no peak bandwidth
                    (None, None, None, None, None, None)]:
            self.assertIsNone(rt.roofline_metrics(*bad), bad)

    def test_L2_unknown_dtype_falls_back(self):
        self.assertEqual(rt.dtype_bytes("some-future-dtype"), 2)
        self.assertEqual(rt.dtype_bytes(None), 2)
        self.assertEqual(rt.dtype_bytes("torch.float8_e4m3fnuz"), 1)
        self.assertEqual(rt.dtype_bytes("c10::BFloat16"), 2)

    def test_L4_missing_counters_return_none_not_zero(self):
        """None (= 'not measured', keep the analytic estimate), never 0.0 (= 'measured nothing')."""
        self.assertIsNone(rt.bytes_from_counters({}))
        self.assertIsNone(rt.bytes_from_counters(None))
        self.assertIsNone(rt.flops_from_counters({"MemUnitStalled": 5.0}))
        self.assertEqual(rt.parse_counter_csv("/nonexistent/counters.csv"), {})

    def test_counter_conversion(self):
        self.assertEqual(rt.bytes_from_counters({"FETCH_SIZE": 1024.0, "WRITE_SIZE": 512.0}),
                         1536.0 * 1024)
        self.assertEqual(rt.flops_from_counters({"MfmaFlopsBF16": 42.0}), 42.0)
        self.assertEqual(rt.flops_from_counters({"SQ_INSTS_VALU_MFMA_MOPS_F8": 2.0},
                                                mfma_flops_per_mop_f8=512.0), 1024.0)

    def test_classify_never_raises(self):
        for bad in [(None, 0.9), ("x", 0.9), (0.5, None), (0, 0), (-1, 0.9)]:
            self.assertEqual(rt.classify_headroom(*bad), "unknown", bad)

    def test_experts_hit_is_bounded_and_safe(self):
        self.assertEqual(rt.experts_hit(0, 100), 0.0)
        self.assertEqual(rt.experts_hit(256, 0), 0.0)
        self.assertEqual(rt.experts_hit(None, None), 0.0)
        self.assertLessEqual(rt.experts_hit(256, 10**6), 256.0)
        self.assertAlmostEqual(rt.experts_hit(256, 512), 221.0, delta=3.0)


class TestMissingComputePeak(unittest.TestCase):
    """A dtype with no tabulated compute peak is UNKNOWN on the compute axis, never 0.0."""

    def test_missing_compute_peak_is_none_not_zero(self):
        p = _peaks()
        m = rt.roofline_metrics(2.56e6, 4e9, 100e-6, p["hbm_bw_bytes_s"], None, 0.90,
                                pct_gpu_time=20.0)
        self.assertIsNone(m["compute_util"])
        self.assertIsNone(m["ridge_point"])
        # memory far below its roof and the compute axis unknown -> no latency verdict
        self.assertEqual(m["bound_type"], "unknown")
        self.assertEqual(m["headroom_class"], "unknown")
        self.assertEqual(m["attainable_speedup"], 1.0)
        self.assertEqual(m["expected_e2e_gain_pct"], 0.0)
        self.assertIn("compute_util unknown", m["note"])
        self.assertIn(m["bound_type"], rt.BOUND_TYPES)

    def test_missing_compute_peak_keeps_a_memory_wall_verdict(self):
        """Near the memory roof the bound is memory whatever the compute axis does."""
        p = _peaks()
        m = rt.roofline_metrics(0.85 * p["hbm_bw_bytes_s"] * 1e-3, 1e9, 1e-3,
                                p["hbm_bw_bytes_s"], None, 0.90)
        self.assertIsNone(m["compute_util"])
        self.assertEqual(m["bound_type"], "memory")
        self.assertNotEqual(m["headroom_class"], "unknown")

    def test_no_flops_is_memory_only_and_still_classified(self):
        """0 FLOPs: the compute axis cannot bind, so a low memory util is latency, as before."""
        p = _peaks()
        m = rt.roofline_metrics(1e7, 0, 100e-6, p["hbm_bw_bytes_s"],
                                rt.peak_flops_for(p, "bf16"), 0.875)
        self.assertEqual(m["compute_util"], 0.0)
        self.assertEqual(m["bound_type"], "latency")

    def test_untabulated_dtype_reaches_the_metric_as_none(self):
        """R9700 has no fp4 rate -> peak_flops_for None -> compute_util None (end to end)."""
        p = rt.resolve_peaks(SKU_JSON, "gfx1201", product="r9700")
        pk = rt.peak_flops_for(p, "fp4")
        self.assertIsNone(pk)
        m = rt.roofline_metrics(1e8, 1e11, 1e-3, p["hbm_bw_bytes_s"], pk, 0.90)
        self.assertIsNone(m["compute_util"])
        self.assertEqual(m["bound_type"], "unknown")


class TestRooflineBasis(unittest.TestCase):
    """Every % of roofline names its numerator and denominator basis; datasheet ranks only."""

    def test_default_basis_is_model_over_datasheet_and_rank_only(self):
        for m in (_moe_metrics(), _attn_metrics(), _moe_metrics(all_experts=True)):
            self.assertEqual(m["numerator_basis"], "model")
            self.assertEqual(m["denominator_basis"], "datasheet")
            self.assertFalse(m["may_gate"])

    def test_only_counters_over_a_probed_denominator_may_gate(self):
        p = _peaks()
        args = (2e8, 1e8, 100e-6, p["hbm_bw_bytes_s"], rt.peak_flops_for(p, "bf16"), 0.90)
        cases = {("counters", "in-shape probe"): True,
                 ("counters", "empirical@mem_bw_probe.py-torch2.9"): True,
                 ("counters", "datasheet"): False,
                 ("model", "in-shape probe"): False,
                 ("counters", "derived"): False}
        for (num, den), want in cases.items():
            with self.subTest(num=num, den=den):
                m = rt.roofline_metrics(*args, numerator_basis=num, denominator_basis=den)
                self.assertEqual((m["numerator_basis"], m["denominator_basis"], m["may_gate"]),
                                 (num, den, want))

    def test_peak_tables_carry_their_denominator_basis(self):
        self.assertEqual(_peaks()["denominator_basis"], "datasheet")
        self.assertEqual(rt.resolve_peaks(SKU_JSON, "gfx1201", product="r9700")["denominator_basis"],
                         "datasheet")


class TestUnifiedThresholds(unittest.TestCase):
    """The module's cut points ARE thresholds.json bound_classification (one table, GEAK-wide)."""

    @classmethod
    def setUpClass(cls):
        with open(rt.THRESHOLDS_JSON, encoding="utf-8") as fh:
            cls.bc = json.load(fh)["bound_classification"]

    def test_module_constants_come_from_thresholds_json(self):
        bc = self.bc
        self.assertEqual(rt.UTIL_BOUND_THRESHOLD, bc["roof_util"]["bound_min"])
        self.assertEqual(rt.UTIL_SATURATED_THRESHOLD, bc["roof_util"]["saturated_min"])
        self.assertEqual(rt.HEADROOM_SATURATED, bc["headroom_vs_target_eff"]["saturated_min"])
        self.assertEqual(rt.HEADROOM_MODERATE, bc["headroom_vs_target_eff"]["moderate_min"])
        self.assertEqual(rt.FEASIBLE_MIN, bc["feasibility"]["min"])
        self.assertEqual(rt.FEASIBLE_MAX, bc["feasibility"]["max"])
        self.assertEqual(rt.LAUNCH_OVERHEAD_S, bc["dispatch"]["launch_overhead_s"])
        self.assertEqual(rt.LATENCY_BOUND_FACTOR, bc["dispatch"]["latency_bound_factor"])

    def test_one_ladder_for_every_metric(self):
        """0.60 roof util, SoL 60/40 + balanced band and the 80% saturation are ONE ladder."""
        bc, lad = self.bc, self.bc["ladder_pct"]
        self.assertEqual((lad["saturated"], lad["bound"], lad["low"]), (80, 60, 40))
        self.assertEqual(100 * bc["roof_util"]["bound_min"], lad["bound"])
        self.assertEqual(100 * bc["roof_util"]["saturated_min"], lad["saturated"])
        self.assertEqual(bc["sol_pipe_pct"]["high"], lad["bound"])
        self.assertEqual(bc["sol_pipe_pct"]["low"], lad["low"])
        self.assertEqual(bc["sol_pipe_pct"]["balanced_band"], [lad["low"], lad["bound"]])
        self.assertEqual(bc["vmem_pct_of_peak"]["saturated"], lad["saturated"])
        with open(rt.THRESHOLDS_JSON, encoding="utf-8") as fh:
            whole = json.load(fh)
        self.assertEqual(whole["confirm"]["hbm_bw_saturated_pct"]["value"], lad["saturated"])
        self.assertEqual(whole["fallback"]["balanced_band_pct"]["value"], [lad["low"], lad["bound"]])
        self.assertEqual(whole["fallback"]["mfma_util_compute_pct"]["value"], lad["bound"])
        for block, body in bc.items():
            if isinstance(body, dict) and block != "ladder_pct":
                self.assertIn("metric", body, block)
                self.assertIn("rule", body, block)

    def test_missing_thresholds_file_keeps_builtin_defaults(self):
        self.assertEqual(rt._load_bound_classification("/nonexistent/thresholds.json"), {})


class TestSkillDocConsistency(unittest.TestCase):
    """The helper's priors must not drift from the SKILL.md table that documents them."""

    def test_target_eff_matches_skill_md(self):
        skill_md = os.path.join(os.path.dirname(PEAKS_MD), "SKILL.md")
        with open(skill_md, encoding="utf-8") as fh:
            text = fh.read()
        self.assertEqual(rt.TARGET_EFF["gemm"], 0.90)
        self.assertEqual(rt.TARGET_EFF["moe"], 0.90)
        self.assertEqual(rt.TARGET_EFF["attn"], 0.50)
        self.assertEqual(rt.target_eff_for("moe", product="r9700"), 0.76)
        self.assertEqual(rt.target_eff_for("elementwise", product="r9700"), 0.76)
        self.assertEqual(rt.target_eff_for("gemm", product="r9700"), 0.90)
        self.assertEqual(rt.target_eff_for("moe", product="unknown"), 0.90)
        for frag in ("dense GEMM | **0.90**", "MoE / grouped GEMM | **0.90**",
                     "attention decode (paged) | **0.50**",
                     "R9700 (`product=r9700`) MoE weight streaming / elementwise | **0.76**"):
            self.assertIn(frag, text, "SKILL.md target_eff table drifted from roofline_tools.TARGET_EFF")

    def test_workload_contract_is_stated(self):
        """The hard constraint the byte-reduction track must not violate."""
        with open(os.path.join(os.path.dirname(PEAKS_MD), "SKILL.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("must not be changed", text)
        self.assertIn("speculative decoding", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
