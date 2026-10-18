"""Development observations: the latest complete saved proof per direction and cell.

Run from the repo root:  python3 -B tests/geak_evo/test_advice_observations.py
"""

import copy
import hashlib
import json
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from geak_evo.archive.store import StateError  # noqa: E402
from geak_evo.selection import advice  # noqa: E402

JOB = {"job_id": "1" * 64, "measurement_epoch_id": "epoch-1"}
KIND = advice.OBSERVED_GAIN_KIND


def _id(*parts):
    return hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()


def _ref(name, artifact):
    return {"evidence_id": _id(name), "artifact": artifact}


class _Store:
    def __init__(self):
        self.objects, self.revisions = {}, {}

    def put(self, name, candidate_id, files):
        self.objects[_id(name)] = ({"candidate_id": candidate_id},
                                   {key: json.dumps(value) for key, value in files.items()})

    def evidence(self, identity):
        if identity not in self.objects:
            raise ValueError("absent evidence")
        envelope, files = self.objects[identity]
        return copy.deepcopy(envelope), dict(files)

    def get_revision(self, name, revision):
        return copy.deepcopy(self.revisions[(name, revision)])


class Campaign:
    """A committed snapshot with saved per-episode proofs, built episode by episode."""

    def __init__(self):
        self.store = _Store()
        self.state = {"binding": dict(JOB), "opportunities": [], "history": [], "latest_receipts": {}}
        self.loop = type("Loop", (), {"store": self.store})()

    def episode(self, direction, outcome="observed_comparison", selected="sel", terminal="term", signed=0.25,
                reviewed=True, proof=True, membership=None, island="unassigned", donors=None):
        number = len(self.state["opportunities"])
        key = "op-%03d" % number
        packet, collection = _ref(key + "/packet", "assignment.json"), _ref(key + "/collection", "c.json")
        transport = _ref(key + "/transport", "transport.json")
        decision = {"selection": {"donor_source_ids": list(donors or [])}}
        self.store.put(key + "/packet", terminal, {"assignment.json": {"decision": decision}})
        opportunity = dict(JOB, opportunity_id=key, direction_id=direction, island_id=island, engineer_id="e",
                           settled=True, completed_sequence=number, membership_ref=membership,
                           progress_complete=False)
        event = dict(attempt_id=key, settled=True, candidate_id=terminal, packet_ref=packet,
                     transport_ref=transport, collection_ref=collection)
        self.state["history"].append(event)
        if reviewed:
            self.state["history"].append({"review_kind": "research_review", "proposal": {
                "direction_id": direction, "transport_ref": transport, "collection_ref": collection}})
        if proof:
            complete = outcome != "unknown"
            review = {"kind": "evolution_progress_review", "binding": dict(JOB), "opportunity_id": key,
                      "complete": complete, "gain_kind": KIND, "outcome_kind": outcome}
            observed = outcome == "observed_comparison"
            saved = {"gain_kind": KIND, "outcome_kind": outcome, "binding": dict(JOB), "opportunity_id": key,
                     "candidate_id": terminal, "assignment_ref": packet, "progress_review": review,
                     "selected_source_id": selected if complete else None,
                     "selected_quality": {"value_ms": 2.0, "observation_count": 3} if complete else None,
                     "baseline": {"source_id": "base"} if observed else None,
                     "signed_frontier_log_ratio": signed if observed else None,
                     "nonnegative_frontier_log_ratio": max(0.0, signed) if observed else None,
                     "credit": None, "unknown_reasons": [] if complete else ["timing missing"]}
            self.store.put(key + "/proof", terminal, {advice.REVIEW_ARTIFACT: review,
                                                      advice.PROOF_ARTIFACT: saved})
            opportunity.update(progress_review_refs=[_ref(key + "/proof", advice.REVIEW_ARTIFACT)],
                               progress_gain_kind=KIND, progress_complete=complete,
                               progress_outcome_kind=outcome)
        self.state["opportunities"].append(opportunity)
        return key

    def snapshot(self):
        revision = len(self.store.revisions) + 1
        self.store.revisions[("campaign", revision)] = {"note": {"evolution": copy.deepcopy(self.state)}}
        return {"campaign_id": "campaign", "revision": revision, "state": copy.deepcopy(self.state)}

    def view(self, rows=advice.OBSERVATION_ROWS):
        development = advice.NativeDevelopmentAdvice(self.loop, None)
        return advice.observation_view(development, self.snapshot(), rows)


def _row(view, direction):
    return next(row for row in view["directions"] if row["direction_id"] == direction)


class ObservationTest(unittest.TestCase):
    def test_latest_complete_is_selected_outcome_with_proof_ref(self):
        campaign = Campaign()
        first = campaign.episode("dir-a", selected="sel-1", terminal="term-1", signed=0.25)
        unknown = campaign.episode("dir-a", outcome="unknown", terminal="term-2")
        unreviewed = campaign.episode("dir-a", reviewed=False, proof=False, terminal="term-3")
        full, inline = campaign.view()
        row = _row(full, "dir-a")
        self.assertEqual(row["last_complete"]["opportunity_id"], first)
        self.assertEqual(row["last_complete"]["selected_source_id"], "sel-1")  # selected, not terminal
        self.assertEqual(row["last_complete"]["signed_frontier_log_ratio"], 0.25)
        self.assertEqual(row["last_complete"]["ref"], {"evidence_id": _id(first + "/proof"),
                                                       "artifact": advice.PROOF_ARTIFACT})
        self.assertEqual((row["incomplete_count"], row["missing_count"], row["episode_count"]), (1, 1, 3))
        self.assertEqual(row["incomplete_refs"][0]["opportunity_id"], unknown)
        self.assertEqual(row["missing_refs"], [{"opportunity_id": unreviewed,
                                                "ref": _ref(unreviewed + "/collection", "c.json")}])
        self.assertEqual((full["cumulative_rate"], full["donor_marginal_credit"]), ("unknown", "unknown"))
        self.assertEqual(inline["directions"], full["directions"])  # short lists are not cut

    def test_later_episodes_do_not_rewrite_earlier_observations(self):
        campaign = Campaign()
        campaign.episode("dir-b", selected="sel-b", signed=0.5)
        campaign.episode("dir-a", selected="sel-1")
        before = _row(campaign.view()[0], "dir-b")
        latest = campaign.episode("dir-a", outcome="baseline_initialization", selected="sel-2")
        campaign.episode("dir-a", outcome="unknown")
        full = campaign.view()[0]
        self.assertEqual(_row(full, "dir-b"), before)
        self.assertEqual(_row(full, "dir-a")["last_complete"]["opportunity_id"], latest)
        self.assertIsNone(_row(full, "dir-a")["last_complete"]["signed_frontier_log_ratio"])
        self.assertEqual([row["direction_id"] for row in full["directions"]], ["dir-a", "dir-b"])  # recency

    def test_every_assigned_membership_keeps_its_own_cell_row(self):
        campaign = Campaign()
        one, two = _ref("membership-1", "m.json"), _ref("membership-2", "m.json")
        campaign.episode("dir-a", membership=one, island="isle", selected="s1")
        campaign.episode("dir-a", membership=two, island="isle", selected="s2")
        campaign.episode("dir-b", membership=one, island="isle", outcome="unknown")
        campaign.episode("dir-c")  # unassigned: direction row only
        full = campaign.view()[0]
        cells = {json.dumps(row["parent_membership_ref"], sort_keys=True): row for row in full["admitted_memberships"]}
        self.assertEqual(len(cells), 2)
        first = cells[json.dumps(one, sort_keys=True)]
        self.assertEqual(first["direction_ids"], ["dir-a", "dir-b"])
        self.assertEqual((first["episode_count"], first["incomplete_count"]), (2, 1))
        self.assertEqual(first["last_complete"]["selected_source_id"], "s1")
        self.assertEqual(cells[json.dumps(two, sort_keys=True)]["last_complete"]["selected_source_id"], "s2")
        self.assertEqual(len(full["directions"]), 3)

    def test_combined_source_context_has_no_marginal_credit(self):
        campaign = Campaign()
        campaign.episode("dir-a")
        combined = campaign.episode("dir-a", donors=["donor-1", "donor-2"], signed=-0.1)
        full = campaign.view()[0]
        self.assertEqual(len(full["combined_source_outcomes"]), 1)
        row = full["combined_source_outcomes"][0]
        self.assertEqual((row["opportunity_id"], row["status"], row["donor_marginal_credit"]),
                         (combined, "complete", "unknown"))
        self.assertEqual(row["donor_source_ids"], ["donor-1", "donor-2"])
        self.assertEqual(_row(full, "dir-a")["last_complete"]["donor_source_ids"], ["donor-1", "donor-2"])

    def test_inline_view_is_bounded_and_deterministic(self):
        campaign = Campaign()
        for number in range(40):
            campaign.episode("direction-%02d" % number, selected=_id("source", number), terminal=_id("t", number),
                             membership=_ref("membership", number), island="isle",
                             donors=[_id("donor", number)] if number % 4 == 0 else None)
            campaign.episode("direction-%02d" % number, outcome="unknown", terminal=_id("u", number))
        full, inline = campaign.view()
        again = campaign.view()[1]
        self.assertEqual(dict(inline, campaign_revision=0), dict(again, campaign_revision=0))
        self.assertEqual(len(inline["directions"]), advice.OBSERVATION_ROWS["directions"])
        self.assertEqual(inline["omitted"], {"directions": 34, "admitted_memberships": 37,
                                             "combined_source_outcomes": 8})
        self.assertNotIn("omitted_direction_ids", inline)  # count only; the full file lists every direction
        self.assertEqual(inline["directions"][0]["direction_id"], "direction-39")  # most recent first
        self.assertEqual(len(full["directions"]), 40)
        size = len(json.dumps(inline, sort_keys=True, separators=(",", ":")).encode())
        self.assertLess(size, 8 * 1024)

    def test_long_nested_lists_are_cut_inline_and_kept_in_full(self):
        campaign, membership = Campaign(), _ref("membership", 0)
        sizes = []
        for rounds in (1, 4):
            while len(campaign.state["opportunities"]) < rounds * 30:
                number = len(campaign.state["opportunities"])
                campaign.episode("direction-%03d" % number, outcome="unknown", membership=membership, island="isle",
                                 donors=[_id("donor", number, item) for item in range(10)])
                campaign.episode("direction-%03d" % number, reviewed=False, proof=False, membership=membership,
                                 island="isle")
                campaign.episode("direction-%03d" % number, membership=membership, island="isle")
            full, inline = campaign.view()
            sizes.append(len(json.dumps(inline, sort_keys=True, separators=(",", ":")).encode()))
        row, cut = full["admitted_memberships"][0], inline["admitted_memberships"][0]
        self.assertEqual(len(row["direction_ids"]), 40)  # the full file keeps every direction
        self.assertEqual((cut["direction_ids"], cut["direction_ids_total"], cut["direction_ids_omitted"]),
                         (row["direction_ids"][-1:], 40, 39))
        self.assertEqual((cut["incomplete_refs"], cut["incomplete_count"]), (row["incomplete_refs"][-1:], 40))
        donors = inline["combined_source_outcomes"][0]
        self.assertEqual((len(donors["donor_source_ids"]), donors["donor_source_ids_total"]), (1, 10))
        self.assertLess(abs(sizes[1] - sizes[0]), 64)  # bounded by the caps, not by campaign length
        self.assertLess(sizes[1], 8 * 1024)

    def test_failed_record_check_is_unavailable_and_store_errors_propagate(self):
        campaign = Campaign()
        campaign.episode("dir-a")
        campaign.state["opportunities"][0]["progress_outcome_kind"] = "unknown"
        full, inline = campaign.view()
        self.assertEqual((full["status"], full["error_type"]), ("unavailable", "ValueError"))
        self.assertIn("differs", full["reason"])
        self.assertEqual(inline, {"schema": advice.OBSERVATIONS_SCHEMA, "status": "unavailable",
                                  "error_type": "ValueError"})  # no reason text in the model-visible view

        class Broken:
            def observations(self, snapshot):
                raise StateError("journal")

        with self.assertRaises(StateError):
            advice.observation_view(Broken(), {})


if __name__ == "__main__":
    unittest.main()
