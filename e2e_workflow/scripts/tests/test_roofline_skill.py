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
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "knowledge", "analysis_skills", "roofline"))
import roofline_tools as rt  # noqa: E402

PEAKS_MD = os.path.join(os.path.dirname(os.path.abspath(rt.__file__)), "peaks.md")

# --- measured profile of the reference run (see module docstring) ---------------------------------
MOE = dict(E=256, M=64, top_k=8, hidden=2048, inter=512, layers=40,
           t_launch_s=49.471e-6, launches_per_step=80, pct_gpu=26.45)
ATTN = dict(batch=64, isl=1024, osl=1024, kv_heads=2, q_heads=16, head_dim=256,
            t_launch_s=141.112e-6, pct_gpu=8.86)


def _peaks():
    return rt.load_peaks(PEAKS_MD, "gfx950")


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

    def test_gfx942_also_tabulated(self):
        p = rt.load_peaks(PEAKS_MD, "gfx942")
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p["hbm_bw_bytes_s"], 5.3e12, delta=1e9)

    def test_L1_unknown_gfx_returns_none(self):
        """Unknown gfx -> None, so the caller falls back to derived peaks at confidence=low."""
        self.assertIsNone(rt.load_peaks(PEAKS_MD, "gfx-does-not-exist"))

    def test_L5_missing_peaks_file_does_not_raise(self):
        self.assertIsNone(rt.load_peaks("/nonexistent/peaks.md", "gfx950"))


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


class TestSkillDocConsistency(unittest.TestCase):
    """The helper's priors must not drift from the SKILL.md table that documents them."""

    def test_target_eff_matches_skill_md(self):
        skill_md = os.path.join(os.path.dirname(PEAKS_MD), "SKILL.md")
        with open(skill_md, encoding="utf-8") as fh:
            text = fh.read()
        self.assertEqual(rt.TARGET_EFF["gemm"], 0.90)
        self.assertEqual(rt.TARGET_EFF["moe"], 0.90)
        self.assertEqual(rt.TARGET_EFF["attn"], 0.50)
        for frag in ("dense GEMM | **0.90**", "MoE / grouped GEMM | **0.90**",
                     "attention decode (paged) | **0.50**"):
            self.assertIn(frag, text, "SKILL.md target_eff table drifted from roofline_tools.TARGET_EFF")

    def test_workload_contract_is_stated(self):
        """The hard constraint the byte-reduction track must not violate."""
        with open(os.path.join(os.path.dirname(PEAKS_MD), "SKILL.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("must not be changed", text)
        self.assertIn("speculative decoding", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
