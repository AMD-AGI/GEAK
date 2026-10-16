"""Drift test 10: horizon cutoffs use the injected clock.

Every "now" (registration, commit receipts, the sealing instant) comes from the
``clock`` passed to ``CommitFrontier`` as ``(boot_id, monotonic_ns)``. A sample
from another boot is never compared with the campaign's monotonic times.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_selection_frontier.py
"""

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_selection_fixtures as fx  # noqa: E402
from geak_evo.selection import frontier as frontier_module  # noqa: E402
from geak_evo.selection.frontier import FrontierError  # noqa: E402


class InjectedClockTest(fx.PortScratch, unittest.TestCase):
    def test_default_clock_is_boot_and_monotonic_pair(self):
        boot, monotonic_ns = frontier_module.sample_clock()
        self.assertEqual(boot, frontier_module.BOOT_ID_PATH.read_text().strip())
        self.assertIs(type(monotonic_ns), int)

    def test_cutoffs_come_from_the_campaign_clock(self):
        cutoffs = frontier_module.horizon_cutoffs(fx.CLOCK, ["H4", "H8", "END"])
        self.assertEqual(cutoffs, {"H4": fx.H4, "H8": fx.H8, "END": fx.CLOCK["deadline_ns"]})

    def test_commit_after_cutoff_is_excluded(self):
        campaign, ticker, frontier = self.frontier_campaign("timeline")
        campaign.plan(fx.set_directive(campaign.choices[0]), proposal={"action": "stop"})
        ticker.now = fx.H4 - 100
        with self.assertRaisesRegex(FrontierError, "horizon_not_observed"):
            self.seal(campaign, frontier, "H4")
        ticker.now = fx.H4 + 10
        campaign.plan(fx.set_directive(campaign.choices[1]), proposal={"action": "stop"})
        h4 = self.seal(campaign, frontier, "H4")
        self.assertEqual(h4["status"], "selected")
        self.assertEqual(h4["recommendation"]["choice"], campaign.choices[0])
        self.assertEqual(h4["cutoff_monotonic_ns"], fx.H4)
        self.assertGreaterEqual(h4["frontier"]["observed_monotonic_ns"], fx.H4)
        self.assertEqual(h4["frontier"]["boot_id"], fx.CLOCK["boot_id"])
        ticker.now = fx.H8
        h8 = self.seal(campaign, frontier, "H8")
        self.assertEqual(h8["recommendation"]["choice"], campaign.choices[1])
        self.assertIn(h8["frontier"]["observed_monotonic_ns"], range(fx.H8, ticker.now))  # fake-clock samples

    def test_commit_straddling_cutoff_is_unresolved(self):
        campaign, ticker, frontier = self.frontier_campaign("straddle", horizons=("H4",))
        ticker.now = fx.H4 - 1
        campaign.commit(campaign.state)
        seal = self.seal(campaign, frontier, "H4")
        self.assertEqual((seal["status"], seal["error_reason"], seal["recommendation"]),
                         ("unresolved", "commit_straddles_horizon", None))

    def test_registration_after_cutoff_or_bad_sample_is_refused(self):
        late = fx.Campaign(self.api, self.root / "late")
        with self.assertRaisesRegex(FrontierError, "registration_clock_or_horizon_differs"):
            self.api["register"](late.store, fx.Ticker(fx.H4), expected_binding=fx.EXPECTED,
                                 clock_ref=fx.CLOCK_REF, clock_raw=fx.CLOCK_RAW, horizons=("H4",))
        bad = fx.Campaign(self.api, self.root / "bad")
        with self.assertRaisesRegex(FrontierError, "clock_sample_differs"):
            frontier_module.CommitFrontier.register(
                bad.store, expected_binding=fx.EXPECTED, clock_ref=fx.CLOCK_REF, clock_raw=fx.CLOCK_RAW,
                horizons=("H4",), clock=lambda: (fx.CLOCK["boot_id"], float(fx.START_NS + 10)))
        other = fx.Campaign(self.api, self.root / "other-boot")
        with self.assertRaisesRegex(FrontierError, "registration_clock_or_horizon_differs"):
            frontier_module.CommitFrontier.register(
                other.store, expected_binding=fx.EXPECTED, clock_ref=fx.CLOCK_REF, clock_raw=fx.CLOCK_RAW,
                horizons=("H4",), clock=lambda: ("other-boot", fx.START_NS + 10))

    def test_samples_from_another_boot_are_refused(self):
        campaign, ticker, frontier = self.frontier_campaign("reboot", horizons=("H4",))
        campaign.plan(fx.set_directive(campaign.choices[0]), proposal={"action": "stop"})
        frontier.clock = lambda: ("other-boot", ticker())
        ticker.now = fx.H4
        with self.assertRaisesRegex(FrontierError, "horizon_not_observed"):
            self.seal(campaign, frontier, "H4")
        campaign.commit(campaign.state)  # The journal write succeeds but gets no receipt.
        self.assertEqual(frontier.last_witness_error, "FrontierError")
        frontier.clock = lambda: (fx.CLOCK["boot_id"], ticker())
        seal = self.seal(campaign, frontier, "H4")
        self.assertEqual((seal["status"], seal["error_reason"], seal["recommendation"]),
                         ("unresolved", "receipt_inventory_differs", None))


if __name__ == "__main__":
    unittest.main()
