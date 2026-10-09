#!/usr/bin/env python3
"""route_review.py on a synthetic routed run: attribution, savings, overspend, proposals.

    python3 -m pytest e2e_workflow/routing/test_route_review.py -q
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import route_review as RR  # noqa: E402

HAIKU, SONNET, OPUS55 = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"
LANES = [HAIKU, SONNET, "claude-opus-4-6", OPUS55]


def call(label, model, inp=0, read=0, w5=0, out=0, role="engineer"):
    row = {"agent_label": label, "model": model, "role": role, "input_tokens": inp,
           "cache_read_input_tokens": read, "cache_creation_input_tokens": w5,
           "cache_write_5m_tokens": w5, "cache_write_1h_tokens": 0, "output_tokens": out}
    row["cost_usd"] = RR.L.cost_of(row, RR.L.DEFAULT_RATES)
    row["cost_breakdown"] = RR.L.cost_breakdown(row, RR.L.DEFAULT_RATES)
    return row


# eng:memory fails once on Haiku, escalates, succeeds on Sonnet. verify succeeds on Haiku first time.
CALLS = [
    call("tech_lead:plan r1", OPUS55, inp=1000, w5=200_000, out=20_000, role="tech_lead"),
    call("route:classify eng:memory", SONNET, w5=20_000, out=500, role="route_classifier"),
    call("eng d1:memory", HAIKU, read=1_000_000, w5=100_000, out=40_000),
    call("route:classify eng:memory", SONNET, w5=20_000, out=500, role="route_classifier"),
    call("eng d4:memory", SONNET, read=1_000_000, w5=100_000, out=40_000),
    call("verify d4", HAIKU, read=200_000, w5=50_000, out=5_000),
]
ROUTING = {"lanes": LANES, "decider": SONNET, "thresholds": {"conf_escalate": 0.7, "max_retries_per_lane": 3,
           "max_top_escalations": 1, "max_output_tokens": 1_000_000, "floors": {}},
           "killed": False, "spent_output_tokens": 106_000, "audit": [
    {"event": "decide", "scope": "eng:memory", "lane": 0, "why": "classified high; small-lane confidence 0.9 >= 0.7 -> SMALL"},
    {"event": "dispatch", "scope": "eng:memory", "label": "eng d1:memory", "model": HAIKU, "capped": False},
    {"event": "fail", "scope": "eng:memory", "model": HAIKU, "fails": 1, "evidence": "verify correctness=fail"},
    {"event": "decide", "scope": "eng:memory", "lane": 1, "why": "escalate: confidence 0.3 < 0.7"},
    {"event": "dispatch", "scope": "eng:memory", "label": "eng d4:memory", "model": SONNET, "capped": False},
    {"event": "decide", "scope": "verify", "lane": 0, "why": "classified small; small-lane confidence 0.95 >= 0.7 -> SMALL"},
    {"event": "dispatch", "scope": "verify", "label": "verify d4", "model": HAIKU, "capped": False},
]}


class TestRouteReview(unittest.TestCase):
    def setUp(self):
        self.R = RR.review(CALLS, ROUTING, None, RR.L.DEFAULT_RATES)

    def test_scope_twin_matches_the_js_ladder(self):
        self.assertEqual(RR.scope_of("eng d3:memory"), "eng:memory")
        self.assertEqual(RR.scope_of("verify d3 (recovered)"), "verify")
        self.assertEqual(RR.scope_of("researcher:q q1"), "researcher:q")

    def test_totals_and_router_reconcile(self):
        self.assertAlmostEqual(self.R["total"], sum(c["cost_usd"] for c in CALLS), places=9)
        router = sum(c["cost_usd"] for c in CALLS if c["role"] == "route_classifier")
        self.assertAlmostEqual(self.R["router"], router, places=9)

    def test_failed_haiku_attempt_is_the_overspend(self):
        haiku_eng = CALLS[2]["cost_usd"]
        self.assertAlmostEqual(self.R["wasted"], haiku_eng, places=9)
        self.assertEqual([(sc, m) for sc, m, *_ in self.R["wasted_rows"]], [("eng:memory", HAIKU)])

    def test_proposes_a_floor_only_where_a_lane_was_skipped_over(self):
        self.assertEqual(self.R["proposals"], {"eng:memory": 1})

    def test_without_control_savings_are_labelled_counterfactual(self):
        md = RR.render(self.R, ROUTING)
        self.assertIn("no measured saving", md)
        self.assertIn("WHERE DID I OVERSPEND?", md)
        self.assertTrue(md.rstrip().endswith("```"))   # floors block to paste, nothing applied

    def test_repriced_baseline_uses_the_brain_card(self):
        # Every row repriced at Opus 5.5 must cost at least as much as the routed cheap rows.
        self.assertGreater(self.R["repriced"], self.R["total"])

    def test_control_run_gives_a_measured_saving_and_flags_a_loss(self):
        cheaper_ctl = [call("eng d1:memory", OPUS55, read=10_000)]
        R = RR.review(CALLS, ROUTING, cheaper_ctl, RR.L.DEFAULT_RATES)
        md = RR.render(R, ROUTING)
        self.assertIn("**Measured**", md)
        self.assertIn("Routing cost more than it saved", md)

    def test_review_names_the_runs_own_decider_and_its_old_tokenizer_models(self):
        md = RR.render(self.R, ROUTING)
        self.assertIn("(`claude-sonnet-5-5` decider)", md)
        self.assertNotIn("older tokenizer", md)          # every lane in this run is on the current tokenizer
        # A run from before the 2026-10-09 lane change: Haiku 4.5 small lane, Sonnet 5 decider.
        old_routing = dict(ROUTING, decider="claude-sonnet-5")
        old_calls = [call("eng d1:memory", "claude-haiku-4-5-20251001", read=1_000, out=100),
                     call("tech_lead:plan r1", OPUS55, inp=1000, out=100, role="tech_lead")]
        md_old = RR.render(RR.review(old_calls, old_routing, None, RR.L.DEFAULT_RATES), old_routing)
        self.assertIn("(`claude-sonnet-5` decider)", md_old)
        self.assertIn("`claude-haiku-4-5-20251001` uses an older tokenizer", md_old)

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            cp, rp = os.path.join(d, "calls.jsonl"), os.path.join(d, "ret.json")
            with open(cp, "w") as fh:
                fh.write("\n".join(json.dumps(c) for c in CALLS) + "\n")
            with open(rp, "w") as fh:
                json.dump({"lanes_result": [{"routing": ROUTING}]}, fh)   # nested, as a workflow return nests it
            self.assertEqual(RR.main(["--calls", cp, "--routing", rp, "--out", d]), 0)
            with open(os.path.join(d, "proposed_floors.json")) as fh:
                self.assertEqual(json.load(fh), {"eng:memory": 1})


if __name__ == "__main__":
    unittest.main()
