"""Drift test 3: a choice is eligible only with a complete, clean, matching witness.

Partial trial coverage, a nonzero exit, or a witness whose IDs belong to another
episode makes the choice ineligible. ``partition_choices`` lists such entries as
rejected; ``eligible_choices`` refuses the whole list; an I/O error is not a
rejection and propagates.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_selection_choices.py
"""

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_selection_fixtures as fx  # noqa: E402
from geak_evo.selection.choices import RecommendationError  # noqa: E402

BAD = {"partial_coverage": "differs from exact checked native passes",
       "nonzero_exit": "differs from exact checked native passes",
       "id_mismatch": "belongs to another campaign episode"}


class ChoiceEligibilityTest(fx.PortScratch, unittest.TestCase):
    def fixture(self, variants):
        episodes = fx.EpisodeFixture(self.api, self.root / "episodes", variants)
        return episodes, episodes.campaign.store, episodes.campaign.snapshot()

    def test_valid_episode_is_eligible(self):
        episodes, store, snapshot = self.fixture(["valid"])
        listed = self.api["eligible_choices"](store, snapshot, episodes.episode)
        self.assertEqual([choice["attempt_id"] for choice in listed], ["attempt-0"])

    def test_each_defect_is_rejected(self):
        for kind, reason in BAD.items():
            with self.subTest(kind=kind):
                episodes, store, snapshot = self.fixture(["valid", kind])
                eligible, rejected = self.api["partition_choices"](store, snapshot, episodes.episode)
                self.assertEqual([choice["attempt_id"] for choice in eligible], ["attempt-0"])
                self.assertEqual(len(rejected), 1)
                self.assertEqual((rejected[0]["attempt_id"], rejected[0]["status"], rejected[0]["error_class"]),
                                 ("attempt-1", "rejected_native_eligibility", "RecommendationError"))
                self.assertIn(reason, rejected[0]["reason"])
                with self.assertRaisesRegex(RecommendationError, reason):
                    self.api["eligible_choices"](store, snapshot, episodes.episode)
                fx.remove(self.root / "episodes")

    def test_io_error_propagates(self):
        episodes, store, snapshot = self.fixture(["valid", "episode_oserror"])
        with self.assertRaises(OSError):
            self.api["partition_choices"](store, snapshot, episodes.episode)


if __name__ == "__main__":
    unittest.main()
