"""final_outcome on real horizon seals and real validated recommendations.

selected = a selected choice; no_improvement = an applied clear (keep ORIGINAL);
everything else is an error with best None, never a substitute program.
measured_outcome reports improved only for a measured speedup above 1.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_selection_outcome.py
"""

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_selection_fixtures as fx  # noqa: E402
from geak_evo.selection.outcome import final_outcome, measured_outcome  # noqa: E402


def error(detail):
    return {"outcome": "error", "best": None, "detail": detail}


class MeasuredOutcomeTest(unittest.TestCase):
    def test_only_a_faster_selected_program_is_improved(self):
        selected = {"outcome": "selected", "best": {"choice_handle": "h"}, "detail": "selected"}
        self.assertEqual(measured_outcome(selected, 1.2),
                         {"outcome": "improved", "best": {"choice_handle": "h"}, "detail": "selected", "speedup": 1.2})
        self.assertEqual(measured_outcome(selected, 1.0),
                         {"outcome": "no_improvement", "best": None, "detail": "selected_not_faster",
                          "selected": {"choice_handle": "h"}, "speedup": 1.0})
        for bad in (None, 0, -1.0, float("nan"), float("inf"), True, "1.2"):
            with self.assertRaisesRegex(ValueError, "measured speedup"):
                measured_outcome(selected, bad)
        for other in (error("unresolved"), {"outcome": "no_improvement", "best": None, "detail": "clear"}):
            self.assertIs(measured_outcome(other, None), other)


class OutcomeFromSealsTest(fx.PortScratch, unittest.TestCase):
    def test_selected_then_clear(self):
        campaign, ticker, frontier = self.frontier_campaign("main")
        campaign.plan(fx.set_directive(campaign.choices[0]), proposal={"action": "stop"})
        ticker.now = fx.H4 + 10
        campaign.plan(fx.CLEAR, proposal=None)
        self.assertEqual(final_outcome(self.seal(campaign, frontier, "H4")),
                         {"outcome": "selected", "best": campaign.choices[0], "detail": "selected"})
        ticker.now = fx.H8
        self.assertEqual(final_outcome(self.seal(campaign, frontier, "H8")),
                         {"outcome": "no_improvement", "best": None, "detail": "clear"})

    def test_no_set_before_horizon(self):
        campaign, ticker, frontier = self.frontier_campaign("keep", horizons=("H4",))
        campaign.plan(None, proposal={"action": "probe"}, wrapped=False)
        ticker.now = fx.H4
        self.assertEqual(final_outcome(self.seal(campaign, frontier, "H4")), error("no_applied_recommendation"))

    def test_no_campaign_revision_before_horizon(self):
        campaign = fx.Campaign(self.api, self.root / "empty")
        ticker = fx.Ticker(fx.START_NS + 10)
        frontier = self.api["register"](campaign.store, ticker, expected_binding=fx.EXPECTED,
                                        clock_ref=fx.CLOCK_REF, clock_raw=fx.CLOCK_RAW, horizons=("H4",))
        campaign.store.checkpoint("other-direction", campaign.base, note=dict(fx.NOTE), expected_revision=0)
        ticker.now = fx.H4 + 5
        campaign.begin()
        self.assertEqual(final_outcome(self.seal(campaign, frontier, "H4")),
                         error("no_campaign_revision_before_horizon"))

    def test_unresolved(self):
        campaign, ticker, frontier = self.frontier_campaign("refused", horizons=("H4",))
        campaign.plan(fx.set_directive(campaign.choices[1]), proposal={"action": "stop"})
        ticker.now = fx.H4
        seal, _ = frontier.seal_horizon(campaign.store, "H4", read_ref=campaign.read_ref,
                                        authenticate_choice=lambda frozen, context, choice: dict(choice, x=1))
        self.assertEqual(final_outcome(seal), error("native_choice_differs"))


class OutcomeShapesTest(unittest.TestCase):
    def test_other_inputs_are_errors(self):
        self.assertEqual(final_outcome(None), error("no_recommendation"))
        self.assertEqual(final_outcome({"status": "selected", "choice": None}), error("unknown_status"))
        self.assertEqual(final_outcome({"status": "other"}), error("unknown_status"))
        self.assertEqual(final_outcome({"schema": "geak-committed-horizon-endpoint-v1", "status": "unresolved"}),
                         error("unresolved"))


if __name__ == "__main__":
    unittest.main()
