"""Drift test 2: final selection never falls back to another program.

The final recommendation is whatever the planner's committed chain says, replayed
from the store. A clear gives no program; a broken or missing chain is an error,
never the fastest or most recent candidate.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_selection_final.py
"""

import os
import sys
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_selection_fixtures as fx  # noqa: E402
from geak_evo.selection import final  # noqa: E402
from geak_evo.selection.final import RecommendationValidationError  # noqa: E402
from geak_evo.selection.outcome import final_outcome  # noqa: E402


class FinalSelectionTest(fx.PortScratch, unittest.TestCase):
    def campaigns(self):
        return fx.campaign_scenarios(self.api, self.root)

    def validate(self, campaign, **overrides):
        arguments = dict(expected_binding=fx.EXPECTED, clock_ref=fx.CLOCK_REF, read_ref=campaign.read_ref,
                         authenticate_choice=campaign.authenticate)
        arguments.update(overrides)
        return self.api["validate"](campaign.store, campaign.snapshot(), **arguments)

    def test_latest_set_is_selected(self):
        campaigns = self.campaigns()
        for name, index in (("set", 0), ("set_set", 1), ("set_keep", 1), ("set_rejected", 0)):
            with self.subTest(name=name):
                result = self.validate(campaigns[name])
                self.assertEqual(result["status"], "selected")
                self.assertEqual(result["choice"], campaigns[name].choices[index])
                self.assertEqual(final_outcome(result),
                                 {"outcome": "selected", "best": result["choice"], "detail": "selected"})

    def test_clear_keeps_original(self):
        campaigns = self.campaigns()
        for name in ("set_clear", "clear_only"):
            with self.subTest(name=name):
                result = self.validate(campaigns[name])
                self.assertEqual((result["status"], result["choice"]), ("missing", None))
                self.assertEqual(result["record"]["disposition"], "clear")
                self.assertEqual(final_outcome(result)["outcome"], "no_improvement")
                self.assertIsNone(final_outcome(result)["best"])

    def test_no_recommendation_is_an_error_not_a_substitute(self):
        result = self.validate(self.campaigns()["empty"])
        self.assertEqual((result["status"], result["head"], result["choice"]), ("missing", None, None))
        self.assertEqual(final_outcome(result),
                         {"outcome": "error", "best": None, "detail": "no_applied_recommendation"})
        self.assertEqual(final_outcome(None), {"outcome": "error", "best": None, "detail": "no_recommendation"})

    def test_broken_chain_raises(self):
        campaigns = self.campaigns()
        cases = {"dropped_events": "missing its unique committed event",
                 "revived_head": "head differs from complete replay"}
        for name, message in cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(RecommendationValidationError, message):
                    self.validate(campaigns[name])

    def test_unreadable_or_refused_choice_raises(self):
        campaign = self.campaigns()["set"]

        def gone(ref):
            raise FileNotFoundError("gone")
        with self.assertRaises(FileNotFoundError):
            self.validate(campaign, read_ref=gone)
        with self.assertRaisesRegex(RecommendationValidationError, "authenticated native SELECTED witness"):
            self.validate(campaign, authenticate_choice=lambda context, choice: dict(choice, extra=1))

    def test_reply_from_another_model_raises(self):
        campaign = self.campaigns()["set"]
        with mock.patch.object(final, "LEGACY_UNROUTED_CLAUDE_MODEL", "claude-other"):
            with self.assertRaisesRegex(RecommendationValidationError, "receipt identity differs"):
                self.validate(campaign)
        config = {"workflow_backend": "claude"}
        observed = {"reported_models": ["claude-other"], "reported_tools": ["Bash"]}
        self.assertFalse(final._model_observation(config, {}, observed, "claude-opus-5", None, None))
        self.assertFalse(final._model_observation(config, {}, dict(observed, reported_models=["claude-opus-5"],
                                                                   reported_tools=["Bash", "Write"]),
                                                  "claude-opus-5", None, None))

    def test_admitted_route_must_name_one_reported_model(self):
        route = {"schema": "geak-native-model-route-v1", "workflow_backend": "claude",
                 "wire_protocol": "anthropic_messages", "requested_model": "claude-opus-5",
                 "accepted_reported_models": ["claude-opus-5", "claude-other"]}
        raw = fx.canonical(route)
        config = {"workflow_backend": "claude", "model_route_ref": {"path": "/fixture/route.json",
                                                                    "sha256": fx.sha(raw)}}
        with self.assertRaisesRegex(RecommendationValidationError, "one exact requested and reported model"):
            final._model_identity(config, lambda ref: raw, None)
        one = fx.canonical(dict(route, accepted_reported_models=["claude-opus-5"]))
        config["model_route_ref"]["sha256"] = fx.sha(one)
        self.assertEqual(final._model_identity(config, lambda ref: one, None),
                         ("claude-opus-5", config["model_route_ref"]))
        with self.assertRaisesRegex(RecommendationValidationError, "requires its admitted route"):
            final._model_identity({"workflow_backend": "codex"}, None, None)
        codex = {"workflow_backend": "codex", "model_route_ref": config["model_route_ref"]}
        with self.assertRaisesRegex(ValueError, "explicit model route services required"):
            final._model_identity(codex, lambda ref: one, None)


if __name__ == "__main__":
    unittest.main()
