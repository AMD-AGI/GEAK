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


class TestMeasurementBasis(unittest.TestCase):
    """An estimate may annotate; only a measurement may rank.

    `2*M*N*K` is the cost of the arithmetic we ASSUME the kernel does, and `experts_hit` is an
    expected value over a router nobody observed. Both are priors. Before this contract existed the
    artifact could not tell them apart from counted traffic, so a modelled row could be ranked on.
    """

    def test_analytic_rows_are_not_rankable(self):
        for name, m in (("moe", _moe_metrics()), ("attn", _attn_metrics())):
            self.assertEqual(m["measurement_basis"], "model", name)
            self.assertEqual(m["confidence"], "low", name)
            self.assertFalse(m["rankable"], name)
            self.assertIn("do not rank", m["basis_note"], name)

    def test_analytic_rows_keep_their_verdict(self):
        """Not rankable is not "no answer": SKILL.md section 5 keeps stage A as display+annotate,
        and the ranking inversion above is computed from exactly these rows."""
        m = _moe_metrics()
        self.assertEqual(m["headroom_class"], "saturated")
        self.assertGreater(m["roofline_pct"], 0.0)

    def test_counted_traffic_makes_a_memory_verdict_rankable(self):
        p = _peaks()
        # The reference MoE layer: ~697 MB of fp8 weights streamed in 98.9 us (FETCH_SIZE is KiB).
        counters = {"FETCH_SIZE": 697e6 / 1024.0, "WRITE_SIZE": 0.0}
        per_expert_elems = 2 * MOE["inter"] * MOE["hidden"] + MOE["hidden"] * MOE["inter"]
        m = rt.roofline_metrics_from_counters(
            counters, 98.9e-6, p["hbm_bw_bytes_s"], rt.peak_flops_for(p, "fp8"),
            rt.TARGET_EFF["moe"], pct_gpu_time=26.45,
            flops_est=2 * MOE["M"] * MOE["top_k"] * per_expert_elems)

        self.assertTrue(m["bytes_measured"])
        self.assertFalse(m["flops_measured"])
        self.assertEqual(m["bound_type"], "memory")
        # The verdict sits on the memory roof and the bytes under it were counted -> rankable.
        self.assertEqual(m["measurement_basis"], "mixed")
        self.assertEqual(m["confidence"], "medium")
        self.assertTrue(m["rankable"])
        self.assertNotIn("basis_note", m)

    def test_rankability_follows_the_axis_the_verdict_sits_on(self):
        """Counting bytes does not license a COMPUTE-side verdict, and vice versa.

        This is the sharp edge of the rule: `roofline_pct` is achieved/peak on one roof, so the
        quantity that roof is made of is the one that has to be real.
        """
        p = _peaks()
        # High AI -> compute side. Bytes counted, FLOPs modelled => the verdict axis is modelled.
        m = rt.roofline_metrics(1e6, 1e12, 1e-3, p["hbm_bw_bytes_s"],
                                rt.peak_flops_for(p, "fp8"), 0.90,
                                pct_gpu_time=10.0, bytes_measured=True, flops_measured=False)
        self.assertGreater(m["arithmetic_intensity"], m["ridge_point"])
        self.assertEqual(m["measurement_basis"], "mixed")
        self.assertEqual(m["confidence"], "low")
        self.assertFalse(m["rankable"])
        self.assertIn("achieved FLOPs", m["basis_note"])

    def test_both_axes_counted_is_high_confidence(self):
        p = _peaks()
        # 6.1 GB moved and 5e11 FLOPs executed in 1 ms -> 6.1 TB/s, AI 81 (left of the 312 ridge).
        m = rt.roofline_metrics_from_counters(
            {"FETCH_SIZE": 4e6, "WRITE_SIZE": 2e6, "MfmaFlopsBF16": 5e11},
            1e-3, p["hbm_bw_bytes_s"], rt.peak_flops_for(p, "bf16"), rt.TARGET_EFF["gemm"])

        self.assertEqual(m["bound_type"], "memory")

        self.assertEqual(m["measurement_basis"], "counters")
        self.assertEqual(m["confidence"], "high")
        self.assertTrue(m["rankable"])

    def test_default_is_estimate_not_measurement(self):
        """An un-migrated caller says nothing about provenance, and nothing must read as model."""
        p = _peaks()
        m = rt.roofline_metrics(2e8, 1e8, 100e-6, p["hbm_bw_bytes_s"],
                                rt.peak_flops_for(p, "bf16"), 0.90, pct_gpu_time=10.0)
        self.assertEqual(m["measurement_basis"], "model")
        self.assertFalse(m["rankable"])

    def test_measured_but_infeasible_is_still_not_rankable(self):
        """suspect and no-verdict rows stay unrankable however they were obtained -- counting the
        bytes does not rescue a row whose ratio is impossible."""
        p = _peaks()
        # Well above the dispatch floor, so this is the infeasible branch and not the latency one.
        over = rt.roofline_metrics(1e12, 1e6, 100e-6, p["hbm_bw_bytes_s"],
                                   rt.peak_flops_for(p, "fp8"), 0.90,
                                   bytes_measured=True, flops_measured=True)
        self.assertTrue(over["suspect"])
        self.assertEqual(over["headroom_class"], "unknown")
        self.assertFalse(over["rankable"])

    def test_no_usable_counters_returns_none(self):
        """None = "stay at stage A/B", visible to the caller. Never a silent zero."""
        p = _peaks()
        self.assertIsNone(rt.roofline_metrics_from_counters(
            {}, 1e-3, p["hbm_bw_bytes_s"], rt.peak_flops_for(p, "bf16"), 0.90))
        self.assertIsNone(rt.roofline_metrics_from_counters(
            {"MemUnitStalled": 5.0}, 1e-3, p["hbm_bw_bytes_s"],
            rt.peak_flops_for(p, "bf16"), 0.90))

    def test_basis_is_a_closed_set(self):
        p = _peaks()
        rows = [_moe_metrics(), _attn_metrics(), _moe_metrics(all_experts=True),
                rt.roofline_metrics_from_counters(
                    {"FETCH_SIZE": 2e6, "MfmaFlopsBF16": 5e12}, 1e-3, p["hbm_bw_bytes_s"],
                    rt.peak_flops_for(p, "bf16"), 0.90)]
        for m in rows:
            self.assertIn(m["measurement_basis"], rt.MEASUREMENT_BASES)
            self.assertIn(m["confidence"], ("low", "medium", "high"))


#: SKILL.md 9.1 — the MoE stage-1 head of run 20260907T031112Z-58636ae3, verbatim from its
#: profile_roofline.json round 0. bf16 peak because that is the ridge the run recorded (312.5).
Q38_MOE1 = dict(bytes_est=815504879, flops_est=5368709120, t_launch_s=69.857e-6,
                pct_gpu=20.615, peak_bw=8.0e12, peak_flops=2.5e15)
#: The same seam with its traffic COUNTED: rocprofv3 FETCH_SIZE/WRITE_SIZE on the task package's
#: decode M=64 case, median of 4 dispatches (see roofline-autit/moe_measured_0915/MEASUREMENT.md).
#: Different operating point than the run above (E=256/inter=2048, CK dispatch) -- what is locked
#: here is feasible-and-rankable vs impossible-and-not, not a bytes-to-bytes delta.
Q38_MOE1_MEASURED = dict(fetch_kib=334391.0, write_kib=276.0, t_dispatch_s=142.28e-6)


class TestAssumedDistributionCounterExample(unittest.TestCase):
    """SKILL.md 9.1: the run where an assumed router cost the biggest kernel its verdict.

    Locked because it is the evidence for the whole measurement-basis contract. The modelled bytes
    imply 1.46x of peak HBM -- not imprecise, impossible -- so the L3 ladder refuses a verdict, and
    20.6% of GPU time yields no routing signal. Counting the traffic makes the same kernel routable.
    """

    def _modelled(self):
        return rt.roofline_metrics(
            Q38_MOE1["bytes_est"], Q38_MOE1["flops_est"], Q38_MOE1["t_launch_s"],
            Q38_MOE1["peak_bw"], Q38_MOE1["peak_flops"], rt.TARGET_EFF["moe"],
            pct_gpu_time=Q38_MOE1["pct_gpu"])

    def _measured(self):
        return rt.roofline_metrics_from_counters(
            {"FETCH_SIZE": Q38_MOE1_MEASURED["fetch_kib"],
             "WRITE_SIZE": Q38_MOE1_MEASURED["write_kib"]},
            Q38_MOE1_MEASURED["t_dispatch_s"],
            Q38_MOE1["peak_bw"], Q38_MOE1["peak_flops"], rt.TARGET_EFF["moe"],
            pct_gpu_time=Q38_MOE1["pct_gpu"], flops_est=Q38_MOE1["flops_est"])

    def test_modelled_bytes_are_physically_impossible(self):
        m = self._modelled()
        self.assertAlmostEqual(m["hbm_util"], 1.459, places=2)
        self.assertTrue(m["suspect"])
        # >=1.46x over-count, straight from the artifact: the model claims more bytes than the
        # kernel could have moved at peak in its own measured time.
        self.assertGreater(m["bytes_est"], m["bytes_upper_bound"])
        self.assertAlmostEqual(m["bytes_est"] / m["bytes_upper_bound"], 1.459, places=2)

    def test_the_biggest_kernel_in_the_run_gets_no_verdict(self):
        m = self._modelled()
        self.assertEqual(m["headroom_class"], "unknown")
        self.assertEqual(m["measurement_basis"], "model")
        self.assertFalse(m["rankable"])

    def test_counting_the_traffic_makes_it_routable(self):
        c = self._measured()
        self.assertAlmostEqual(c["hbm_util"], 0.301, places=2)
        self.assertFalse(c["suspect"])
        self.assertEqual(c["headroom_class"], "underperforming")
        self.assertAlmostEqual(c["attainable_speedup"], 2.99, places=1)
        self.assertTrue(c["rankable"])

    def test_the_two_rows_route_to_opposite_tracks(self):
        """Not a precision difference -- a different lever.

        The modelled row's prose concluded 'AT or very near the memory roof, so the lever is BYTE
        REDUCTION', inferred from the direction of an infeasible number. Counted, both utilisations
        are under 0.60, so it is latency/occupancy-bound and section 7 sends it to occupancy and
        dependency chains instead.
        """
        m, c = self._modelled(), self._measured()
        self.assertEqual(m["bound_type"], "memory")
        self.assertEqual(c["bound_type"], "latency")
        self.assertLess(c["hbm_util"], 0.60)
        self.assertLess(c["compute_util"], 0.60)

    def test_the_contrast_is_the_point(self):
        """Modelled: impossible and unusable. Counted: feasible, and it routes."""
        m, c = self._modelled(), self._measured()
        self.assertGreater(m["roofline_pct_raw"], 1.0)
        self.assertLessEqual(c["roofline_pct"], 1.0)
        self.assertFalse(m["rankable"])
        self.assertTrue(c["rankable"])

    def test_worked_example_is_written_down(self):
        with open(os.path.join(os.path.dirname(PEAKS_MD), "SKILL.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("9.1 Worked counter-example", text)
        self.assertIn("mfma_moe1_silu_mul_afp4_wfp4_bf16", text)
        self.assertIn("1.459", text)
        # The caveat is load-bearing: the two columns are different operating points.
        self.assertIn("Do not read the two columns as the same measurement", text)


class TestPerShapeFold(unittest.TestCase):
    """SKILL.md 3a: one row per operating point, folded by deployment time.

    The defect this closes is arithmetic, not stylistic. `base_latency_ms` is `total_us / count`
    over a whole phase (`parse_profile.py:199`) while the byte side is modelled from one
    representative shape (`_est_shape`, `parse_profile.py:203`), so numerator and denominator
    described different operating points. The resulting bias has a FIXED SIGN -- the modal shape is
    small, the phase mean is dragged up by the large chunks, so `roofline_pct` reads low and
    `attainable_speedup = target_eff / roofline_pct` reads high. Phantom headroom, worst on exactly
    the kernels whose shape spread is widest, and `expected_e2e_gain_pct` then spends budget on it.
    """

    #: A prefill distribution with the shape that makes the single-row treatment wrong: the modal
    #: chunk is small and frequent, the large chunks are rare and set the phase mean.
    SHAPES = ((1936, 90), (8192, 14), (32768, 10))
    H = 6144

    def _rows(self, effs):
        """One memory-bound row per shape, each at its own efficiency `eff` of the HBM roof.

        Deliberately an elementwise residual+norm, not a GEMM: at these N/K a bf16 GEMM has an
        arithmetic intensity far above the ridge, so it walks toward the COMPUTE roof and a
        bytes-derived time would be rejected as infeasible by L3. The memory-axis fold needs an op
        that is genuinely on the memory roof.
        """
        p = _peaks()
        pk, pf = p["hbm_bw_bytes_s"], rt.peak_flops_for(p, "bf16")
        out = []
        for (m, calls), eff in zip(self.SHAPES, effs):
            by = 3.0 * m * self.H * rt.dtype_bytes("bf16")     # read x, read residual, write y
            out.append({"name": "M%d" % m, "m": m, "calls": calls, "row": rt.roofline_metrics(
                by, 4.0 * m * self.H, by / (eff * pk), pk, pf, rt.TARGET_EFF["elementwise"],
                pct_gpu_time=6.98, bytes_measured=True, flops_measured=True)})
        return out

    @staticmethod
    def _axis(fold, name="memory"):
        return [a for a in fold["axes"] if a["roof_axis"] == name][0]

    def test_time_weighting_is_the_aggregate_not_an_approximation(self):
        """Sum(w_i*pct_i)/sum(w_i) == total bytes / total time / peak, exactly.

        This is the identity SKILL.md 3a claims. If it ever stops holding, the weight is wrong --
        weighting by call count alone would let a shape that runs often but briefly outvote the one
        actually consuming the wall clock.
        """
        rows = self._rows((0.55, 0.55, 0.55))
        fold = rt.fold_cases(rows, rt.TARGET_EFF["elementwise"], pct_gpu_time=6.98)
        pk = _peaks()["hbm_bw_bytes_s"]
        agg = (sum(c["calls"] * c["row"]["bytes_est"] for c in rows)
               / (sum(c["calls"] * c["row"]["t_ms"] * 1e-3 for c in rows) * pk))
        mem = self._axis(fold)
        self.assertAlmostEqual(mem["roofline_pct"], agg, places=12)
        self.assertAlmostEqual(mem["roofline_pct"], 0.55, places=12)
        self.assertEqual(fold["excluded"]["n"], 0)
        self.assertAlmostEqual(mem["weight_share"], 1.0, places=12)

    def test_the_fold_moves_the_answer_off_the_modal_shape(self):
        """A kernel that is poor when small and good when large: 30% modal vs 52% folded.

        The point of the section is that this gap is not noise -- it is a 1.74x overstatement of
        headroom, and it lands on whichever kernels have the widest shape distribution.
        """
        rows = self._rows((0.30, 0.60, 0.80))
        mem = self._axis(rt.fold_cases(rows, rt.TARGET_EFF["elementwise"], pct_gpu_time=6.98))
        modal = rows[0]["row"]["roofline_pct"]
        self.assertAlmostEqual(modal, 0.30, places=6)
        self.assertGreater(mem["roofline_pct"], modal * 1.3)
        overstatement = (rt.TARGET_EFF["elementwise"] / modal) / mem["attainable_speedup"]
        self.assertGreater(overstatement, 1.5)

    def test_rows_without_a_verdict_are_excluded_and_their_share_reported(self):
        """Folding a dispatch-bound row at its clamped value manufactures saturation.

        A zero-`calls` row is excluded rather than defaulted to 1: a silent default would flatten
        the very distribution this section exists to respect.
        """
        rows = self._rows((0.55, 0.55, 0.55))
        p = _peaks()
        bad = rows + [
            {"name": "dispatch_bound", "calls": 500, "row": rt.roofline_metrics(
                1024, 1024, 1e-6, p["hbm_bw_bytes_s"], rt.peak_flops_for(p, "bf16"),
                rt.TARGET_EFF["elementwise"], bytes_measured=True, flops_measured=True)},
            {"name": "no_calls", "calls": 0, "row": rows[0]["row"]},
        ]
        fold = rt.fold_cases(bad, rt.TARGET_EFF["elementwise"], pct_gpu_time=6.98)
        self.assertEqual(fold["excluded"]["n"], 2)
        self.assertEqual({e["case"]["name"] for e in fold["excluded"]["cases"]},
                         {"dispatch_bound", "no_calls"})
        mem = self._axis(fold)
        self.assertAlmostEqual(mem["roofline_pct"], 0.55, places=12)   # bad rows did not move it
        self.assertLess(mem["weight_share"], 1.0)
        self.assertGreater(fold["excluded"]["weight_share"], 0.0)

    def test_axes_are_never_combined_and_the_gpu_budget_is_conserved(self):
        """roofline_pct is achieved/peak on ONE roof; averaging across roofs divides by two peaks.

        Each axis may also claim only the slice of the kernel's GPU time its weight accounts for,
        or every axis would separately promise the whole kernel's e2e gain.
        """
        p = _peaks()
        rows = self._rows((0.55, 0.55, 0.55))[:1] + [
            {"name": "compute_heavy", "calls": 50, "row": rt.roofline_metrics(
                1e6, 8e11, 1e-3, p["hbm_bw_bytes_s"], rt.peak_flops_for(p, "bf16"),
                rt.TARGET_EFF["elementwise"], pct_gpu_time=6.98,
                bytes_measured=True, flops_measured=True)}]
        fold = rt.fold_cases(rows, rt.TARGET_EFF["elementwise"], pct_gpu_time=6.98)
        self.assertEqual({a["roof_axis"] for a in fold["axes"]}, {"memory", "compute"})
        self.assertAlmostEqual(sum(a["pct_gpu_time_share"] for a in fold["axes"]), 6.98, places=9)

    def test_roof_axis_is_recorded_separately_from_bound_type(self):
        """`bound_type` can read 'latency' while the ratio still sits on the memory roof.

        `fold_cases` groups on `roof_axis` precisely because `bound_type` is not a safe proxy for
        which denominator the ratio was taken against.
        """
        row = _attn_metrics()
        self.assertEqual(row["bound_type"], "latency")
        self.assertEqual(row["roof_axis"], "memory")

    def test_degradation_is_non_fatal(self):
        self.assertIsNone(rt.fold_cases([], rt.TARGET_EFF["elementwise"]))
        self.assertIsNone(rt.fold_cases(None, 0))
        self.assertIsNone(rt.fold_cases(self._rows((0.55, 0.55, 0.55)), 0))
        # A row that is not a dict, and a case that is not a dict, are skipped rather than raising.
        self.assertIsNone(rt.fold_cases(["not a dict", {"calls": 3, "row": None}],
                                        rt.TARGET_EFF["elementwise"]))

    def test_section_3a_is_written_down(self):
        with open(os.path.join(os.path.dirname(PEAKS_MD), "SKILL.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("## 3a.", text)
        self.assertIn("profile_workload.json", text)
        # The three constraints that make the identity hold must survive future edits.
        self.assertIn("Never fold across `roof_axis`", text)
        self.assertIn("must come from the deployment trace", text)
        self.assertIn("Fold only rows that have a verdict", text)
        # And the reason counters alone do not close it.
        self.assertIn("Counters do not fix this", text)


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

    def test_measurement_basis_contract_is_documented(self):
        """The basis fields are only useful if the consumer is told what they mean."""
        with open(os.path.join(os.path.dirname(PEAKS_MD), "SKILL.md"), encoding="utf-8") as fh:
            text = fh.read()
        for frag in ("measurement_basis", "bytes_measured", "rankable",
                     "roofline_metrics_from_counters"):
            self.assertIn(frag, text, "SKILL.md does not document the measurement-basis contract")
        self.assertEqual(rt.MEASUREMENT_BASES, ("counters", "mixed", "model"))
        self.assertIn("counters|mixed|model", text)

    def test_workload_contract_is_stated(self):
        """The hard constraint the byte-reduction track must not violate."""
        with open(os.path.join(os.path.dirname(PEAKS_MD), "SKILL.md"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("must not be changed", text)
        self.assertIn("speculative decoding", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
