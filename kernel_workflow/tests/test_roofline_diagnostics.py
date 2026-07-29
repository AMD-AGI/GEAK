#!/usr/bin/env python3
"""Phase 2 tests: raw-counter diagnostics, latency dep/issue split, red flags,
efficiency-artifact rejection, and parser coverage of blocks 7/10/11/16.

Pure-CPU: loads modules by path like the sibling policy/parser tests so no package
install or GPU is required.
"""

import importlib.util
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_PATH = os.path.join(ROOT, "scripts", "roofline_policy.py")
KERNEL_PATH = os.path.join(ROOT, "scripts", "roofline_kernel.py")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "analyze_full_blocks_gfx942.txt")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


roofline_policy = _load("roofline_policy", POLICY_PATH)
roofline_kernel = _load("roofline_kernel", KERNEL_PATH)


class RefineLatencyTests(unittest.TestCase):
    def test_dependency_wait_dominant(self):
        limit, evidence = roofline_policy.refine_latency_limit(
            {"dependency_wait_pct": 60.0, "issue_wait_pct": 10.0}
        )
        self.assertEqual(limit, "latency_dep")
        self.assertTrue(any("dependency_wait_pct" in e for e in evidence))

    def test_issue_wait_dominant(self):
        limit, _ = roofline_policy.refine_latency_limit(
            {"dependency_wait_pct": 10.0, "issue_wait_pct": 60.0}
        )
        self.assertEqual(limit, "latency_issue")

    def test_comparable_buckets_stay_generic(self):
        limit, _ = roofline_policy.refine_latency_limit(
            {"dependency_wait_pct": 20.0, "issue_wait_pct": 22.0}
        )
        self.assertEqual(limit, "latency_occupancy")

    def test_below_threshold_stays_generic(self):
        # dep dominates issue by ratio, but neither reaches the material floor.
        limit, _ = roofline_policy.refine_latency_limit(
            {"dependency_wait_pct": 8.0, "issue_wait_pct": 1.0}
        )
        self.assertEqual(limit, "latency_occupancy")

    def test_missing_diagnostics_are_fail_soft(self):
        self.assertEqual(
            roofline_policy.refine_latency_limit({}), ("latency_occupancy", [])
        )
        self.assertEqual(
            roofline_policy.refine_latency_limit(None), ("latency_occupancy", [])
        )

    def test_opposite_levers_for_dep_vs_issue(self):
        dep = roofline_policy.recommend_optimization("compute_side", "latency_dep")
        issue = roofline_policy.recommend_optimization("compute_side", "latency_issue")
        dep_levers = " ".join(dep["recommended_levers"]).lower()
        issue_levers = " ".join(issue["recommended_levers"]).lower()
        self.assertIn("increase registers", dep_levers)
        self.assertIn("reduce registers", issue_levers)


class RedFlagTests(unittest.TestCase):
    def _flags(self, diagnostics, extra=None):
        metrics = {"diagnostics": diagnostics}
        if extra:
            metrics.update(extra)
        return {f["flag"] for f in roofline_policy.detect_red_flags(metrics)}

    def test_gpu_underfilled(self):
        self.assertIn("gpu_underfilled", self._flags({"ctas": 100.0, "num_cus": 304.0}))

    def test_no_gpu_flag_when_filled(self):
        self.assertNotIn(
            "gpu_underfilled", self._flags({"ctas": 304.0, "num_cus": 304.0})
        )

    def test_register_spill(self):
        self.assertIn("register_spill", self._flags({"scratch_per_workitem": 16.0}))
        self.assertNotIn(
            "register_spill", self._flags({"scratch_per_workitem": 0.0})
        )

    def test_low_occupancy(self):
        self.assertIn("low_occupancy", self._flags({"achieved_occupancy_pct": 13.0}))
        self.assertNotIn(
            "low_occupancy", self._flags({"achieved_occupancy_pct": 80.0})
        )

    def test_poor_coalescing(self):
        self.assertIn("poor_coalescing", self._flags({"coalescing_pct": 25.0}))
        self.assertNotIn("poor_coalescing", self._flags({"coalescing_pct": 90.0}))

    def test_lds_bank_conflicts(self):
        self.assertIn("lds_bank_conflicts", self._flags({"lds_bank_conflict_pct": 40.0}))
        self.assertNotIn(
            "lds_bank_conflicts", self._flags({"lds_bank_conflict_pct": 0.0})
        )

    def test_efficiency_artifact(self):
        flags = self._flags({}, extra={"roofline_efficiency_pct": 140.0})
        self.assertIn("efficiency_artifact", flags)
        self.assertNotIn(
            "efficiency_artifact",
            self._flags({}, extra={"roofline_efficiency_pct": 57.0}),
        )

    def test_missing_inputs_produce_no_flags(self):
        self.assertEqual(roofline_policy.detect_red_flags({"diagnostics": {}}), [])
        self.assertEqual(roofline_policy.detect_red_flags(None), [])


class EfficiencyArtifactGateTests(unittest.TestCase):
    def _case(self, efficiency):
        metrics = {
            "roofline_efficiency_pct": efficiency,
            "headroom_ratio": 2.0,
            # AI + ridge so theoretical_bound is known (else confidence falls to low).
            "ai_hbm": 200.0,
            "ai_ridge_empirical": 100.0,
            "compute_utilization_pct": 30.0,
            "hbm_utilization_pct": 20.0,
            "diagnostics": {},
        }
        classification = roofline_policy.build_classification(metrics)
        return {
            "case_id": "c0",
            "status": "matched",
            "weight": 1.0,
            "metrics": metrics,
            "classification": classification,
        }

    def test_artifact_case_is_invalidated(self):
        result = {"status": "ok", "cases": [self._case(140.0)]}
        guidance = roofline_policy.assess_guidance(result)
        self.assertFalse(guidance["valid"])
        self.assertEqual(guidance["cases"][0]["reason"], "efficiency_artifact")
        self.assertIn("c0", guidance["invalid_case_ids"])
        flag_names = {f["flag"] for f in guidance["red_flags"]}
        self.assertIn("efficiency_artifact", flag_names)

    def test_normal_efficiency_case_is_valid(self):
        result = {"status": "ok", "cases": [self._case(57.0)]}
        guidance = roofline_policy.assess_guidance(result)
        self.assertTrue(guidance["valid"])
        self.assertNotEqual(guidance["cases"][0]["reason"], "efficiency_artifact")


class ParserFixtureTests(unittest.TestCase):
    """Guard the block 7/10/11/16 parser against a real gfx942 analyze capture."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(FIXTURE):
            raise unittest.SkipTest("full-profile fixture not present")
        text = open(FIXTURE).read()
        kernels = roofline_kernel.parse_rocprof_compute(text, dtypes=["FP16"])
        valid = [k for k in kernels if roofline_kernel._has_valid_kernel_metrics(k)]
        cls.kernel = valid[0]
        cls.diag = cls.kernel["metrics"]["diagnostics"]

    def test_latency_split_fields_populated(self):
        for key in ("dependency_wait_pct", "issue_wait_pct", "active_pct"):
            self.assertIsNotNone(self.diag[key], key)
        # dep + issue + active should roughly partition Wave Cycles (<= ~100%+slack).
        total = self.diag["dependency_wait_pct"] + self.diag["issue_wait_pct"] + self.diag["active_pct"]
        self.assertLess(total, 130.0)

    def test_pipeline_and_occupancy_fields(self):
        self.assertIsNotNone(self.diag["mfma_util_pct"])
        self.assertIsNotNone(self.diag["valu_util_pct"])
        self.assertIsNotNone(self.diag["achieved_occupancy_pct"])
        self.assertIsNotNone(self.diag["ctas"])
        self.assertIsNotNone(self.diag["num_cus"])

    def test_memory_quality_fields(self):
        self.assertIsNotNone(self.diag["coalescing_pct"])
        self.assertIsNotNone(self.diag["lds_bank_conflict_pct"])

    def test_classification_uses_diagnostics(self):
        classification = self.kernel["classification"]
        # This FP16 GEMM is latency/issue-bound with poor coalescing; occupancy is either
        # flagged plainly or (with VGPR+AGPR present) as a register ceiling -- both acceptable.
        self.assertIn(
            classification["observed_limit"],
            ("latency_issue", "latency_dep", "latency_occupancy", "compute"),
        )
        flag_names = {f["flag"] for f in classification.get("red_flags", [])}
        self.assertIn("poor_coalescing", flag_names)
        self.assertTrue(
            flag_names & {"low_occupancy", "register_occupancy_ceiling", "occupancy_not_register_limited"},
            flag_names,
        )

    def test_register_ceiling_parsed_from_registers(self):
        # Fixture VGPR 128 + AGPR 352 = 480 regs/thread -> waves/SIMD = min(8, 512/480) = 1,
        # well under the ceiling-of-2 threshold, so the register occupancy flag fires.
        self.assertIsNotNone(self.diag["waves_per_simd_ceiling"])
        self.assertLessEqual(
            self.diag["waves_per_simd_ceiling"], roofline_policy.REGISTER_CEILING_WAVES
        )
        flag_names = {f["flag"] for f in roofline_policy.detect_red_flags(self.kernel["metrics"])}
        self.assertIn("register_occupancy_ceiling", flag_names)


class RegisterCeilingTests(unittest.TestCase):
    def _flags(self, diag):
        return {f["flag"] for f in roofline_policy.detect_red_flags({"diagnostics": diag})}

    def test_register_ceiling_at_or_below_two(self):
        # ceiling 2 -> register-limited occupancy is 25%; achieved 20% is pinned at it.
        flags = self._flags({"waves_per_simd_ceiling": 2.0, "achieved_occupancy_pct": 20.0})
        self.assertIn("register_occupancy_ceiling", flags)
        self.assertNotIn("low_occupancy", flags)
        self.assertNotIn("occupancy_not_register_limited", flags)

    def test_low_ceiling_but_well_below_is_not_register_limited(self):
        # ceiling 2 -> 25% register ceiling, but achieved 8% sits far below it: registers are
        # NOT the constraint, so cutting VGPR would do nothing (the key false-positive fix).
        flags = self._flags({"waves_per_simd_ceiling": 2.0, "achieved_occupancy_pct": 8.0})
        self.assertNotIn("register_occupancy_ceiling", flags)
        self.assertIn("occupancy_not_register_limited", flags)

    def test_low_occupancy_but_not_register_limited(self):
        # Registers leave headroom (ceiling 8) yet occupancy is low -> limiter is elsewhere.
        flags = self._flags({"waves_per_simd_ceiling": 8.0, "achieved_occupancy_pct": 20.0})
        self.assertIn("occupancy_not_register_limited", flags)
        self.assertNotIn("register_occupancy_ceiling", flags)
        self.assertNotIn("low_occupancy", flags)

    def test_low_occupancy_when_ceiling_unknown(self):
        flags = self._flags({"achieved_occupancy_pct": 20.0})
        self.assertIn("low_occupancy", flags)
        self.assertNotIn("occupancy_not_register_limited", flags)


class AmdahlWorthinessTests(unittest.TestCase):
    def test_ceiling_math(self):
        # 40% of e2e time, 4x kernel headroom -> 40 * (1 - 1/4) = 30% reclaimable.
        self.assertAlmostEqual(
            roofline_policy.amdahl_ceiling_pct(0.4, 4.0), 30.0, places=6
        )

    def test_ceiling_fail_soft(self):
        self.assertIsNone(roofline_policy.amdahl_ceiling_pct(None, 4.0))
        self.assertIsNone(roofline_policy.amdahl_ceiling_pct(0.4, None))
        self.assertIsNone(roofline_policy.amdahl_ceiling_pct(0.0, 4.0))

    def _case(self, case_id, weight, headroom):
        metrics = {
            "roofline_efficiency_pct": 50.0,
            "headroom_ratio": headroom,
            "ai_hbm": 200.0,
            "ai_ridge_empirical": 100.0,
            "compute_utilization_pct": 30.0,
            "hbm_utilization_pct": 20.0,
            "diagnostics": {},
        }
        classification = roofline_policy.build_classification(metrics)
        return {
            "case_id": case_id,
            "status": "matched",
            "weight": weight,
            "metrics": metrics,
            "classification": classification,
        }

    def test_tiny_share_case_is_valid_but_not_dominant(self):
        # A big-headroom but tiny-time-share case must not become dominant; a modest
        # but heavy-share case wins instead.
        big = self._case("big_share", 99.0, 1.5)
        tiny = self._case("tiny_share", 0.01, 5.0)
        result = {"status": "ok", "cases": [big, tiny]}
        guidance = roofline_policy.assess_guidance(result)
        by_id = {c["case_id"]: c for c in guidance["cases"]}
        self.assertTrue(by_id["tiny_share"]["below_amdahl_floor"])
        self.assertTrue(by_id["tiny_share"]["valid"])
        self.assertEqual(by_id["tiny_share"]["reason"], "below_amdahl_floor")
        self.assertNotEqual(guidance["dominant_case_id"], "tiny_share")

    def test_time_shares_sum_to_one(self):
        result = {
            "status": "ok",
            "cases": [self._case("a", 3.0, 2.0), self._case("b", 1.0, 2.0)],
        }
        guidance = roofline_policy.assess_guidance(result)
        shares = [c["time_share"] for c in guidance["cases"]]
        self.assertAlmostEqual(sum(shares), 1.0, places=6)
        by_id = {c["case_id"]: c for c in guidance["cases"]}
        self.assertAlmostEqual(by_id["a"]["time_share"], 0.75, places=6)


class NoiseFloorTests(unittest.TestCase):
    def _cmp(self, before_gf, after_gf, before_eff=50.0, after_eff=50.0):
        # Performance ratio is derived from performance_gflops (higher = faster). Equal
        # empirical peaks keep the compare compatible.
        peaks = {"compute_empirical_peak_gflops": 1000.0, "hbm_empirical_peak_gbps": 5000.0}
        before = {"metrics": dict(peaks, performance_gflops=before_gf, roofline_efficiency_pct=before_eff)}
        after = {"metrics": dict(peaks, performance_gflops=after_gf, roofline_efficiency_pct=after_eff)}
        return roofline_policy.compare_cases(before, after)

    def test_within_noise_not_counted_as_improved(self):
        # 2% faster is inside the +/-3.4% floor -> noise, not a real win.
        cmp = self._cmp(100.0, 102.0)
        self.assertTrue(cmp["within_noise"])
        self.assertFalse(cmp["improved"])

    def test_real_improvement_beyond_floor(self):
        cmp = self._cmp(100.0, 110.0)
        self.assertFalse(cmp["within_noise"])
        self.assertTrue(cmp["improved"])

    def test_utilization_moved_but_perf_did_not(self):
        # Performance flat (noise) yet efficiency climbed -> the classic misleading signal.
        cmp = self._cmp(100.0, 101.0, before_eff=40.0, after_eff=55.0)
        self.assertTrue(cmp["within_noise"])
        self.assertTrue(cmp["utilization_moved_perf_did_not"])


if __name__ == "__main__":
    unittest.main()
