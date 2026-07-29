#!/usr/bin/env python3
"""Pure-CPU unit tests for the roofline guidance validity gate (assess_guidance).

The gate decides whether roofline evidence is trustworthy enough to STEER the
optimizer before it reaches the planner/engineers, in both default and deep mode.
"""

import importlib.util
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_PATH = os.path.join(ROOT, "scripts", "roofline_policy.py")
SPEC = importlib.util.spec_from_file_location("roofline_policy", POLICY_PATH)
roofline_policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roofline_policy)


def _case(case_id, *, status="matched", observed="hbm", confidence="high",
          specialties=("memory",), levers=("reduce HBM bytes",),
          headroom=3.0, measured=True, weight=1.0):
    metrics = {}
    classification = {
        "observed_limit": observed,
        "confidence": confidence,
        "recommended_specialties": list(specialties),
        "recommended_levers": list(levers),
        "headroom_ratio": headroom,
    }
    if measured:
        classification["roofline_efficiency_pct"] = 20.0
        metrics["hbm_utilization_pct"] = 65.0
    return {
        "case_id": case_id,
        "status": status,
        "weight": weight,
        "metrics": metrics,
        "classification": classification,
    }


def _result(cases, status="ok", priority_order=None):
    summary = {}
    if priority_order is not None:
        summary["priority_order"] = [{"case_id": c} for c in priority_order]
    return {"status": status, "cases": cases, "summary": summary}


class AssessGuidanceTests(unittest.TestCase):
    def test_valid_matched_cases_aggregate_recommendations(self):
        result = _result([
            _case("decode_m64", observed="hbm", specialties=("memory",),
                  levers=("reduce HBM bytes",), weight=3.0),
            _case("prefill_m1024", observed="compute", specialties=("compute",),
                  levers=("improve MFMA issue",), weight=1.0),
        ], priority_order=["decode_m64", "prefill_m1024"])
        g = roofline_policy.assess_guidance(result)
        self.assertTrue(g["valid"])
        self.assertEqual(g["reason"], "actionable_cases_present")
        self.assertEqual(g["dominant_case_id"], "decode_m64")
        self.assertEqual(g["recommended_specialties"], ["memory", "compute"])
        self.assertIn("reduce HBM bytes", g["recommended_levers"])
        self.assertEqual(g["invalid_case_ids"], [])

    def test_skipped_result_is_invalid(self):
        result = _result([_case("c0")], status="skipped")
        g = roofline_policy.assess_guidance(result)
        self.assertFalse(g["valid"])
        self.assertTrue(g["reason"].startswith("roofline_status_"))
        self.assertEqual(g["invalid_case_ids"], ["c0"])

    def test_low_confidence_case_is_rejected(self):
        result = _result([_case("c0", confidence="low")])
        g = roofline_policy.assess_guidance(result)
        self.assertFalse(g["valid"])
        self.assertEqual(g["cases"][0]["reason"], "low_confidence")

    def test_unknown_limit_is_rejected(self):
        result = _result([_case("c0", observed="unknown", specialties=())])
        g = roofline_policy.assess_guidance(result)
        self.assertFalse(g["valid"])
        self.assertEqual(g["cases"][0]["reason"], "observed_limit_unknown")

    def test_matched_but_no_measured_signal_is_rejected(self):
        # The regression this gate exists for: a matched case with no real
        # numeric evidence must NOT be treated as actionable steering.
        result = _result([_case("c0", measured=False, headroom=None)])
        g = roofline_policy.assess_guidance(result)
        self.assertFalse(g["valid"])
        self.assertEqual(g["cases"][0]["reason"], "no_measured_signal")

    def test_missing_specialty_is_rejected(self):
        result = _result([_case("c0", specialties=(), levers=())])
        g = roofline_policy.assess_guidance(result)
        self.assertFalse(g["valid"])
        self.assertEqual(g["cases"][0]["reason"], "no_recommended_specialty")

    def test_mixed_valid_and_invalid(self):
        result = _result([
            _case("good", observed="hbm"),
            _case("bad", confidence="low"),
        ], priority_order=["good", "bad"])
        g = roofline_policy.assess_guidance(result)
        self.assertTrue(g["valid"])
        self.assertEqual(g["dominant_case_id"], "good")
        self.assertEqual(g["invalid_case_ids"], ["bad"])

    def test_low_headroom_is_valid_but_flagged_and_deprioritized(self):
        result = _result([
            _case("tiny", observed="hbm", headroom=1.02, weight=5.0),
            _case("roomy", observed="compute", specialties=("compute",),
                  headroom=4.0, weight=1.0),
        ], priority_order=["tiny", "roomy"])
        g = roofline_policy.assess_guidance(result)
        self.assertTrue(g["valid"])
        tiny = next(c for c in g["cases"] if c["case_id"] == "tiny")
        self.assertTrue(tiny["valid"])
        self.assertTrue(tiny["low_headroom"])
        self.assertEqual(tiny["reason"], "low_headroom")
        # a case with real headroom leads even though tiny has higher priority
        self.assertEqual(g["dominant_case_id"], "roomy")

    def test_no_cases(self):
        g = roofline_policy.assess_guidance(_result([]))
        self.assertFalse(g["valid"])
        self.assertEqual(g["reason"], "no_cases")

    def test_non_dict_input_is_fail_soft(self):
        g = roofline_policy.assess_guidance(None)
        self.assertFalse(g["valid"])
        self.assertEqual(g["reason"], "result_not_object")


if __name__ == "__main__":
    unittest.main()
