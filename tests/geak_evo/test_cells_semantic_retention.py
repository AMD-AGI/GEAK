"""Semantic membership survives missing/failed timing without becoming an elite."""

import copy
import unittest

from geak_evo.archive import cells


class SemanticRetentionTest(unittest.TestCase):
    def setUp(self):
        self.spec = {"schema": "native-categorical-cells-v1", "descriptor_id": "method",
                     "revision": 1, "axes": [{"name": "method", "meaning": "source dataflow",
                                               "values": ["staged", "direct"]}]}
        self.charter_ref = {"evidence_id": "charter", "artifact": "charter.json"}
        self.membership_ref = {"evidence_id": "member", "artifact": "membership.json"}
        self.member = {"candidate_id": "candidate", "island_id": "island",
                       "charter_ref": self.charter_ref,
                       "assignment": {"status": "supported", "descriptors": {"method": "staged"},
                                      "evidence_refs": [{"evidence_id": "source", "artifact": "code"}],
                                      "reason": "source loads a staged tile"}}
        class Store:
            def get_candidate(self, candidate):
                return {"source_id": "source"}
        self.projector = cells.NativeCellProjector(
            Store(), episode_material=lambda *args: None, episode_review=lambda *args: None)

    def place(self, failures=None, retired=False, measured=False, reviewed=True):
        found = {"failures": failures or {}, "roles": {}, "reviewed": {"candidate": 0} if reviewed else {},
                 "observations": {}, "unplaced": []}
        if measured:
            context = {"benchmark_case_ids": ["small", "large"], "statistic": "fixture-geomean"}
            found["observations"]["source", cells._hash(context), "operation"] = {
                "source_id": "source", "operation_id": "operation", "comparison": context,
                "per_case_times_ms": {"small": 1.0, "large": 4.0}}
        identity = ("direction", cells._hash(self.membership_ref))
        members = {identity: {"record": copy.deepcopy(self.member), "ref": self.membership_ref,
                             "direction_id": "direction", "position": 1}}
        eligible, historical = self.projector._place_members(
            {"research_refs": {"charters": {"island": self.charter_ref}}},
            {cells._hash(self.charter_ref): {"descriptor_spec": self.spec}}, members,
            {identity} if retired else set(), found)
        return eligible, historical, found["unplaced"]

    def assert_unscored_semantics(self, unplaced, status):
        self.assertEqual(len(unplaced), 1)
        row = unplaced[0]
        self.assertEqual(row["coordinates"], ["staged"])
        self.assertEqual(row["charter_ref"], self.charter_ref)
        self.assertNotIn("descriptor_spec", row)
        self.assertEqual(row["descriptor_status"], "supported")
        self.assertEqual(row["quality_status"], status)
        self.assertEqual(row["membership_ref"], self.membership_ref)
        self.assertNotIn("quality", row)
        self.assertNotIn("cell_id", row)

    def test_unmeasured_source_keeps_supported_coordinates_without_fitness(self):
        eligible, historical, unplaced = self.place()
        self.assertEqual((eligible, historical), ([], []))
        self.assert_unscored_semantics(unplaced, "no_comparable_timing")
        self.assertEqual(cells.reduce_cells(eligible), [])

    def test_failed_source_keeps_semantics_and_exact_failure_without_fitness(self):
        failure = {"feedback_ref": {"path": "/record", "sha256": "f" * 64},
                   "error": "compile failed"}
        eligible, historical, unplaced = self.place({"source": {"failed-operation": failure}}, measured=True)
        self.assertEqual((eligible, historical), ([], []))
        self.assert_unscored_semantics(unplaced, "native_trial_error")
        self.assertEqual(unplaced[0]["failed_trials"], [failure])
        self.assertEqual(cells.reduce_cells(eligible), [])

    def test_timed_member_keeps_existing_quality_and_case_vector(self):
        eligible, historical, unplaced = self.place(measured=True)
        self.assertEqual((historical, unplaced), ([], []))
        reduced = cells.reduce_cells(eligible)
        elite = reduced[0]["development_elite"]
        self.assertEqual(elite["quality"]["value_ms"], 2.0)
        self.assertEqual(elite["quality"]["observations"][0]["per_case_times_ms"],
                         {"small": 1.0, "large": 4.0})
        self.assertEqual(cells.select_cell({"cells": reduced}, reduced[0]["cell_id"], "source"),
                         {"parent_candidate_id": "candidate", "island_id": "island",
                          "membership_ref": self.membership_ref})

    def test_unsupported_assignment_does_not_acquire_coordinates(self):
        self.member["assignment"]["status"] = "unknown"
        eligible, historical, unplaced = self.place()
        self.assertEqual((eligible, historical), ([], []))
        self.assertIsNone(unplaced[0]["coordinates"])
        self.assertEqual(unplaced[0]["descriptor_status"], "unknown")
        self.assertEqual(unplaced[0]["quality_status"], "descriptor_unsupported")

    def test_supported_but_unreviewed_membership_is_explicit(self):
        eligible, historical, unplaced = self.place(reviewed=False)
        self.assertEqual((eligible, historical), ([], []))
        self.assert_unscored_semantics(unplaced, "unreviewed")

    def test_retired_membership_remains_historical(self):
        eligible, historical, unplaced = self.place(retired=True)
        self.assertEqual((eligible, unplaced), ([], []))
        self.assertEqual(len(historical), 1)


if __name__ == "__main__":
    unittest.main()
