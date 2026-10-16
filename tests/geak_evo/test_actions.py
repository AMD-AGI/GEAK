"""Research actions called directly on a CPU fixture store (no loop, model or GPU).

The old test_research_actions.py drives these actions through EvolutionLoop.run;
these tests call perform_action / preview_action directly with the same kind of
payloads, so they run the same on the Oct-15 original and on the port.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from geak_evo.archive.actions import index_result, perform_action, preview_action, without_timestamp
from geak_evo.archive.records import archive_checkpoint, read_record
from geak_evo.archive.store import ResearchStore

CASES = {"job": ["short", "long"]}
NOTE = {"claim": "CPU fixture", "scope": "No performance measurement", "next_test": "Inspect records"}


def _verifier(artifacts, expected):
    return {key: True for key in ("independent", "settled", "correct", "eligible")}


class ResearchActionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.work = root / "work"
        self.work.mkdir()
        self.store = ResearchStore(root / "archive", allowed_epochs=[("job", "epoch")],
                                   source_paths={"job": ["kernel.py"]}, verify_evidence=_verifier)
        self.a = self.candidate("baseline\n")
        self.donor = self.candidate("donor_value = 7\n", parents=[self.a])
        for direction, tip in (("D1", self.a), ("D2", self.donor)):
            archive_checkpoint(self.store, direction, tip, expected_revision=0, frozen_cases=CASES, note=NOTE,
                               card={"idea": {"idea_id": direction, "statement": "Synthetic idea",
                                              "status": "hypothesis"}})
        self.records = {}
        self.operations = 0

    def candidate(self, text, **lineage):
        (self.work / "kernel.py").write_text(text)
        source = self.store.capture_source(self.work, ["kernel.py"])
        return self.store.candidate("job", "epoch", source, **lineage)

    def proposal(self, action, payload, direction_id="D1", **changes):
        direction = self.store.get(direction_id)
        result = {"kind": "research_action", "action": action, "direction_id": direction_id,
                  "expected_revision": direction["revision"], "candidate_id": direction["tip"], "payload": payload}
        result.update(changes)
        return result

    def perform(self, proposal):
        self.operations += 1
        refs = perform_action(self.store, proposal, "op-%d" % self.operations, accounting={},
                              frozen_cases=CASES, records=self.records)
        index_result(self.records, proposal, refs)
        return refs

    def preview(self, proposal):
        return preview_action(self.store, proposal, "op-preview", accounting={}, frozen_cases=CASES,
                              records=self.records)

    @staticmethod
    def charter(island, previous=None):
        return {"island_id": island, "question": "Which source organization supports this hypothesis?",
                "descriptor_spec": {"representation": "agent claim, not measured novelty"},
                "previous_ref": previous, "reason": "Separate inspectable implementation hypotheses."}

    @staticmethod
    def membership(island, charter, *, retain=(), previous=None, **member):
        value = {"island_id": island, "charter_ref": charter, "previous_ref": previous,
                 "assignment": {"status": "tentative", "descriptors": {"representation": island},
                                "evidence_refs": [], "reason": "Synthetic source grouping."}}
        value.update(member)
        return {"retain_refs": list(retain), "reason": "The agent proposes this grouping.", "membership": value}

    def members(self, direction_id):
        card = read_record(self.store, self.store.get(direction_id)["note"]["archive_card_ref"])
        return card.get("membership_refs", []), card.get("source_membership_refs", [])

    def islands(self):
        self.perform(self.proposal("charter", self.charter("A")))
        self.perform(self.proposal("membership", self.membership("A", self.records["charters"]["A"])))
        self.perform(self.proposal("charter", self.charter("B"), "D2"))
        self.perform(self.proposal("membership", self.membership("B", self.records["charters"]["B"]), "D2"))
        own, _ = self.members("D2")
        self.perform(self.proposal("membership", self.membership("A", self.records["charters"]["A"], retain=own),
                                   "D2"))

    @staticmethod
    def concept(island):
        return {"concept_id": "concept-" + island, "origin_island_id": "B", "island_id": island,
                "derivation": "Synthetic source-free summary law", "scope": "CPU fixture hypothesis",
                "evidence_refs": []}

    def exchange_request(self):
        direction = self.store.get("D1")
        donor_members, _ = self.members("D2")
        request = {"schema_version": 1, "exchange_id": "agent-combination", "job_id": "job",
                   "measurement_epoch_id": "epoch", "question": "Inspect both source and concept carriers.",
                   "recipient": {"direction_id": "D1", "revision": direction["revision"],
                                 "candidate_id": direction["tip"],
                                 "source_id": self.store.get_candidate(direction["tip"])["source_id"],
                                 "island_id": "A", "membership_ref": self.members("D1")[0][0]},
                   "offers": []}
        for island, member in zip(("B", "A"), donor_members):
            common = {"origin_island_id": "B", "selected_island_id": island, "claim": "A contribution to inspect",
                      "scope": "CPU fixture", "obligations": [], "carrier_refs": []}
            request["offers"].append(dict(common, offer_id="source-" + island, kind="source",
                                          candidate_id=self.donor,
                                          source_id=self.store.get_candidate(self.donor)["source_id"],
                                          origin_ref=donor_members[0], membership_ref=member))
            request["offers"].append(dict(common, offer_id="concept-" + island, kind="concept",
                                          concept_ref=self.records["concepts"]["concept-" + island]))
        return request

    def test_charter_membership_concept_and_exchange_records(self):
        self.islands()
        self.assertEqual(set(self.records["charters"]), {"A", "B"})
        own, sources = self.members("D2")
        self.assertEqual(len(own), 2)
        self.assertEqual(sources, [])
        self.assertEqual({read_record(self.store, ref)["island_id"] for ref in own}, {"A", "B"})
        self.assertEqual(self.store.get("D2")["note"]["research_action_id"], "op-5")
        for island in ("A", "B"):
            refs = self.perform(self.proposal("concept", self.concept(island)))
            _, artifacts = self.store.evidence(refs["concept_ref"]["evidence_id"])
            self.assertEqual(json.loads(artifacts["concept.json"])["kind"], "exchange_concept")
        refs = self.perform(self.proposal("exchange", self.exchange_request()))
        _, artifacts = self.store.evidence(refs["exchange_ref"]["evidence_id"])
        self.assertEqual(json.loads(artifacts["exchange.json"])["exchange_id"], "agent-combination")
        self.assertEqual(self.records["exchanges"]["agent-combination"], refs["exchange_ref"])
        with self.assertRaisesRegex(ValueError, "exchange ID already exists"):
            self.perform(self.proposal("exchange", self.exchange_request()))
        with self.assertRaisesRegex(ValueError, "concept ID already exists"):
            self.perform(self.proposal("concept", self.concept("A")))

    def test_membership_for_other_candidate_goes_to_source_refs(self):
        self.perform(self.proposal("charter", self.charter("A")))
        payload = self.membership("A", self.records["charters"]["A"], candidate_id=self.donor)
        refs = self.perform(self.proposal("membership", payload))
        own, sources = self.members("D1")
        self.assertEqual((own, sources), ([], [refs["membership_ref"]]))
        self.assertEqual(read_record(self.store, refs["membership_ref"])["candidate_id"], self.donor)

    def test_preview_predicts_refs_and_state_without_writing(self):
        self.perform(self.proposal("charter", self.charter("A")))
        proposal = self.proposal("membership", self.membership("A", self.records["charters"]["A"]))
        before = self.store.get("D1")
        preview = self.preview(proposal)
        self.assertEqual(self.store.get("D1"), before)
        with self.assertRaises(Exception):
            self.store.evidence(preview["result_refs"]["membership_ref"]["evidence_id"])
        refs = perform_action(self.store, proposal, "op-preview", accounting={}, frozen_cases=CASES,
                              records=self.records)
        self.assertEqual(refs, preview["result_refs"])
        self.assertEqual(without_timestamp(self.store.get("D1")), without_timestamp(preview["checkpoint"]))
        record_only = self.preview(self.proposal("concept", self.concept("A")))
        self.assertIsNone(record_only["checkpoint"])

    def test_charter_and_membership_need_current_heads(self):
        first = self.perform(self.proposal("charter", self.charter("A")))["charter_ref"]
        with self.assertRaisesRegex(ValueError, "current exact head"):
            self.perform(self.proposal("charter", self.charter("A")))
        second = self.perform(self.proposal("charter", self.charter("A", previous=first)))["charter_ref"]
        with self.assertRaisesRegex(ValueError, "stale charter"):
            self.perform(self.proposal("membership", self.membership("A", first)))
        member = self.perform(self.proposal("membership", self.membership("A", second)))["membership_ref"]
        with self.assertRaisesRegex(ValueError, "exact predecessor"):
            self.perform(self.proposal("membership", self.membership("A", second, retain=[member])))
        with self.assertRaisesRegex(ValueError, "duplicate retained"):
            self.perform(self.proposal("membership", dict(self.membership("A", second), retain_refs=[member, member],
                                                          membership=None)))
        with self.assertRaisesRegex(ValueError, "in the current card"):
            self.perform(self.proposal("membership", dict(self.membership("A", second), retain_refs=[first],
                                                          membership=None)))
        self.perform(self.proposal("membership", self.membership("A", second, retain=[], previous=member)))

    def test_malformed_or_stale_proposals_are_rejected_without_change(self):
        before = self.store.get()
        charter = self.charter("A")
        cases = [(self.proposal("charter", charter, expected_revision=7), "direction changed"),
                 (self.proposal("charter", charter, candidate_id=self.donor), "direction changed"),
                 (self.proposal("charter", dict(charter, extra=1)), "payload fields"),
                 (self.proposal("charter", dict(charter, reason=" ")), "nonempty scientific text"),
                 (self.proposal("unknown", {}), "unknown research action"),
                 (self.proposal("nominate", {"action": "revive"}), "maturation action/payload mismatch"),
                 (self.proposal("accept_anchor", {"nominee_id": self.a, "reason": "x",
                                                  "evidence_ref": {"evidence_id": "0" * 64, "artifact": "a.json"}}),
                  "exact native evaluation")]
        for proposal, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.perform(proposal)
        self.assertEqual(self.store.get(), before)
        self.assertEqual(self.records, {})

    def test_membership_requires_a_current_card(self):
        self.store.checkpoint("legacy", self.a, note=NOTE, expected_revision=0)
        with self.assertRaisesRegex(ValueError, "current archive card"):
            self.perform(self.proposal("membership", {"retain_refs": [], "membership": None, "reason": "r"},
                                       "legacy"))

    def test_accept_anchor_uses_host_verifier_and_preview_does_not_commit(self):
        evidence = self.store.archive_evidence(self.a, {"native_evaluation.json": b"{}"})
        payload = {"nominee_id": self.a, "reason": "Host evaluation passed",
                   "evidence_ref": {"evidence_id": evidence, "artifact": "native_evaluation.json"}}
        preview = self.preview(self.proposal("accept_anchor", payload))
        self.assertIsNone(self.store.get("D1")["anchor"])
        self.assertEqual(preview["checkpoint"]["anchor"]["evidence_id"], evidence)
        refs = self.perform(self.proposal("accept_anchor", copy.deepcopy(payload)))
        self.assertEqual(refs, {"anchor_ref": payload["evidence_ref"]})
        anchor = self.store.get("D1")["anchor"]
        self.assertEqual(anchor, preview["checkpoint"]["anchor"])
        self.assertEqual(self.records, {})

    def test_index_result_and_without_timestamp(self):
        records = {}
        index_result(records, {"action": "membership", "payload": {}}, {"membership_ref": "x"})
        self.assertEqual(records, {})
        index_result(records, {"action": "concept", "payload": {"concept_id": "c"}}, {"concept_ref": "ref"})
        self.assertEqual(records, {"concepts": {"c": "ref"}})
        self.assertEqual(without_timestamp({"updated_at": 1, "tip": "t"}), {"tip": "t"})


if __name__ == "__main__":
    unittest.main()
