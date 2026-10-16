"""Synthetic CPU archive checks; no runtime, hardware, or scientific qualification."""

import json
from pathlib import Path
import tempfile
import unittest

from geak_evo.archive.records import (ArchiveError, archive_checkpoint, archive_record,
                                format_archive, read_archive, read_record)
from geak_evo.archive.store import ResearchStore, StateError


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.options = dict(allowed_epochs=[("job", "epoch"), ("job", "older")],
                            source_paths={"job": ["kernel.py"]}, verify_evidence=self.verifier)
        self.store = ResearchStore(self.root / "archive", **self.options)
        self.cases = {"job": ["short", "long"]}
        self.a = self.candidate("baseline")
        self.note = {"claim": "CPU fixture only", "scope": "synthetic job", "next_test": "inspect exact evidence"}

    @staticmethod
    def verifier(artifacts, expected):
        receipt = json.loads(artifacts["validation.json"])
        if receipt["binding"] != expected:
            raise StateError("wrong synthetic validation binding")
        return {key: True for key in ("independent", "settled", "correct", "eligible")}

    def candidate(self, text, *, parents=(), donors=(), epoch="epoch"):
        (self.work / "kernel.py").write_text(text)
        source = self.store.capture_source(self.work, ["kernel.py"])
        return self.store.candidate("job", epoch, source, parents=parents, donors=donors)

    def raw(self, candidate, data, name="observation.json"):
        data = data if isinstance(data, bytes) else json.dumps(data).encode()
        return {"evidence_id": self.store.archive_evidence(candidate, {name: data}), "artifact": name}

    def record(self, candidate, kind, **fields):
        return archive_record(self.store, candidate, dict(kind=kind, **fields), frozen_cases=self.cases)

    def charter(self, candidate, island="I1"):
        return self.record(candidate, "evo_island_charter", island_id=island,
                           question="Can a different dataflow help?", descriptor_spec={"ownership": ["row", "split"]})

    def membership(self, candidate, island="I1", *, status="unknown", evidence=None, charter=None):
        return self.record(candidate, "evo_membership", island_id=island,
                           charter_ref=charter or self.charter(candidate, island),
                           assignment={"status": status, "descriptors": {"ownership": None},
                                       "evidence_refs": evidence or [], "reason": "Case evidence pending"})

    def card(self, **fields):
        return dict(idea={"idea_id": "Q1", "statement": "Test streaming summary", "status": "proposed"}, **fields)

    def checkpoint(self, candidate=None, *, direction="D1", revision=0, card=None, **fields):
        return archive_checkpoint(self.store, direction, candidate or self.a, card=card or self.card(),
                                  note=self.note, expected_revision=revision, frozen_cases=self.cases, **fields)

    def selection(self, recipient, parent, donor, parent_membership, donor_membership):
        def offer(candidate, role, island, ref):
            return {"role": role, "candidate_id": candidate,
                    "source_id": self.store.get_candidate(candidate)["source_id"],
                    "selected_from_island_id": island, "membership_ref": ref}
        return self.record(recipient, "evo_decision", decision_type="selection", rationale="Compare complementary mechanisms",
                           recipient={"direction_id": "D1", "island_id": "I1", "membership_ref": parent_membership},
                           offers=[offer(parent, "parent", "I1", parent_membership),
                                   offer(donor, "donor", "I2", donor_membership)])

    def test_unknown_startup_and_legacy_direction(self):
        self.assertEqual(read_archive(self.store)["distinct_tip_sources"], 0)
        self.checkpoint()
        self.store.checkpoint("legacy", self.a, note=self.note, expected_revision=0)
        view = read_archive(self.store)
        self.assertIsNone(view["directions"]["D1"]["anchor"])
        self.assertIsNone(view["directions"]["legacy"]["card"])
        self.assertEqual(view["distinct_tip_sources"], 1)
        output = format_archive(view)
        self.assertIn("research metadata unknown", output)
        self.assertIn("not established", output)
        self.assertNotIn("0.0", output)

    def test_two_islands_and_exact_crossover_selection(self):
        donor = self.candidate("donor")
        first = self.membership(self.a, "I1")
        second = self.membership(donor, "I2")
        selected = self.selection(self.a, self.a, donor, first, second)
        child = self.candidate("combined but unvalidated", parents=[self.a], donors=[donor])
        child_membership = self.membership(child, "I1")
        self.checkpoint(child, card=self.card(selection_ref=selected, membership_refs=[child_membership]))
        self.checkpoint(donor, direction="D2", card=self.card(membership_refs=[second]))
        view = read_archive(self.store)
        self.assertEqual(set(view["islands"]), {"I1", "I2"})
        self.assertEqual(view["distinct_tip_sources"], 2)
        self.assertIsNone(view["directions"]["D1"]["anchor"])
        self.assertIsNone(view["islands"]["I1"][0]["validated_elite"])
        self.assertEqual(view["directions"]["D1"]["card"]["selection_ref"], selected)
        with self.assertRaises(ArchiveError):
            self.checkpoint(self.a, direction="D3", card=self.card(selection_ref=selected))

    def test_selected_donor_must_exist_in_child_ancestry(self):
        donor = self.candidate("donor")
        selected = self.selection(self.a, self.a, donor, self.membership(self.a), self.membership(donor, "I2"))
        child = self.candidate("parent only", parents=[self.a])
        with self.assertRaisesRegex(ArchiveError, "absent from ancestry"):
            self.checkpoint(child, card=self.card(selection_ref=selected))

    def test_selection_rejects_wrong_source_or_membership_snapshot(self):
        donor = self.candidate("donor")
        first, second = self.membership(self.a), self.membership(donor, "I2")
        ref = self.selection(self.a, self.a, donor, first, second)
        record = read_record(self.store, ref)
        record["offers"][1]["source_id"] = self.store.get_candidate(self.a)["source_id"]
        with self.assertRaisesRegex(ArchiveError, "source mismatch"):
            archive_record(self.store, self.a, record)
        record["offers"][1]["source_id"] = self.store.get_candidate(donor)["source_id"]
        record["offers"][1]["selected_from_island_id"] = "I1"
        with self.assertRaisesRegex(ArchiveError, "island mismatch"):
            archive_record(self.store, self.a, record)

    def test_migration_keeps_source_and_uncertain_assignments(self):
        origin, destination = self.membership(self.a), self.membership(self.a, "I2", status="tentative")
        migration = self.record(self.a, "evo_decision", decision_type="migration", rationale="Explore ownership question",
                                evidence_refs=[origin, destination], details={"origin_ref": origin, "destination_ref": destination})
        self.checkpoint(card=self.card(membership_refs=[destination], decision_refs=[migration]))
        view = read_archive(self.store)
        self.assertEqual(view["directions"]["D1"]["tip"], self.a)
        self.assertIsNone(view["islands"]["I2"][0]["validated_elite"])
        self.assertEqual(read_record(self.store, origin)["island_id"], "I1")
        with self.assertRaisesRegex(ArchiveError, "source-bound evidence"):
            self.membership(self.a, status="supported")

    def test_anchor_failed_protected_candidate_and_later_fallback_survive(self):
        self.checkpoint(resources={"gpu_seconds": 0})
        candidate = self.store.get_candidate(self.a)
        validation = self.raw(self.a, {"binding": {"job_id": "job", "measurement_epoch_id": "epoch",
                                                   "candidate_source_id": candidate["source_id"]}}, "validation.json")
        self.store.promote("D1", self.a, validation["evidence_id"], expected_revision=1)
        failed = self.candidate("unfinished syntax !!!", parents=[self.a])
        fidelity = self.raw(failed, b"Intended streaming invariant remains; compile is failing", "source-audit.txt")
        opportunity = self.raw(failed, {"requested": {"seconds": 10}, "reserved": {"seconds": 10},
                                       "receipts": [], "settled": False})
        card = self.card(last_faithful={"candidate_id": failed, "claim": "Provisional fidelity; not numerical validity",
                                       "evidence_refs": [fidelity]},
                         protection={"candidate_id": failed, "opportunity_ref": opportunity, "rationale": "Repair lowering"})
        self.checkpoint(failed, revision=2, card=card, status="unfinished", resources={"gpu_seconds": 0})
        fallback = self.candidate("fallback omits mechanism", parents=[failed])
        self.checkpoint(fallback, revision=3, card=card)
        reopened = ResearchStore(self.store.root, **self.options)
        view = read_archive(reopened)
        row = view["directions"]["D1"]
        self.assertEqual(row["anchor"]["candidate_id"], self.a)
        self.assertEqual(row["tip"], fallback)
        self.assertEqual(row["card"]["last_faithful"]["candidate_id"], failed)
        self.assertEqual(row["resources"], {"gpu_seconds": 0})
        self.assertEqual(row["opportunity_verification"], "not_verified_by_archive")
        self.assertIn("reported actual: unknown", format_archive(view))
        self.assertEqual(reopened.get_candidate(failed)["source_id"], self.store.get_candidate(failed)["source_id"])

    def test_faithfulness_needs_evidence_for_that_exact_candidate(self):
        other = self.candidate("different")
        evidence = self.raw(self.a, b"audit", "audit.txt")
        for refs in ([], [evidence]):
            with self.subTest(refs=refs), self.assertRaises(ArchiveError):
                self.checkpoint(card=self.card(last_faithful={"candidate_id": other, "claim": "Faithful", "evidence_refs": refs}))

    def test_replacement_card_omission_preserves_last_faithful_after_fallback_and_reopen(self):
        faithful = self.candidate("unfinished streaming summary")
        evidence = self.raw(faithful, b"Scoped provisional source witness", "witness.txt")
        witness = {"candidate_id": faithful, "claim": "Source-level partial fidelity", "evidence_refs": [evidence]}
        first = self.checkpoint(faithful, card=self.card(last_faithful=witness))
        fallback = self.candidate("fallback without streaming summary", parents=[faithful])

        self.checkpoint(fallback, revision=1, card=self.card())

        reopened = ResearchStore(self.store.root, **self.options)
        row = read_archive(reopened)["directions"]["D1"]
        self.assertEqual(row["tip"], fallback)
        self.assertEqual(row["card"]["last_faithful"], witness)
        self.assertEqual(read_record(reopened, first["note"]["archive_card_ref"])["last_faithful"], witness)
        self.assertEqual(reopened.evidence(evidence["evidence_id"])[1]["witness.txt"], b"Scoped provisional source witness")

    def test_explicit_fidelity_withdrawal_requires_correction(self):
        evidence = self.raw(self.a, b"Initial provisional witness", "witness.txt")
        witness = {"candidate_id": self.a, "claim": "Mechanism appears present", "evidence_refs": [evidence]}
        first = self.checkpoint(card=self.card(last_faithful=witness))

        with self.assertRaises(ArchiveError):
            self.checkpoint(revision=1, card=self.card(last_faithful=None))

        self.assertEqual(self.store.get("D1"), first)
        self.assertEqual(read_archive(self.store)["directions"]["D1"]["card"]["last_faithful"], witness)

    def test_fidelity_withdrawal_preserves_prior_witness_source_and_evidence(self):
        source_text = "if False: streaming_summary()"
        alleged_faithful = self.candidate(source_text)
        initial = self.raw(alleged_faithful, b"Initial source inspection claimed a streaming path", "witness.txt")
        witness = {"candidate_id": alleged_faithful, "claim": "Streaming path appears present", "evidence_refs": [initial]}
        first = self.checkpoint(alleged_faithful, card=self.card(last_faithful=witness))
        first_ref = first["note"]["archive_card_ref"]
        counterinspection = self.raw(alleged_faithful, b"The streaming path is disabled; no faithful implementation exists", "counterinspection.txt")
        correction = {"card_ref": first_ref, "field": "last_faithful", "reason": "Withdraw the mistaken source-level fidelity claim",
                      "evidence_refs": [counterinspection]}
        fallback = self.candidate("ordinary fallback", parents=[alleged_faithful])

        self.checkpoint(fallback, revision=1, card=self.card(last_faithful=None, corrections=[correction]))

        reopened = ResearchStore(self.store.root, **self.options)
        row = read_archive(reopened)["directions"]["D1"]
        self.assertEqual(row["tip"], fallback)
        self.assertIn("last_faithful", row["card"])
        self.assertIsNone(row["card"]["last_faithful"])
        self.assertEqual(row["card"]["corrections"], [correction])
        self.assertEqual(read_record(reopened, first_ref)["last_faithful"], witness)
        self.assertEqual(reopened.evidence(initial["evidence_id"])[1]["witness.txt"], b"Initial source inspection claimed a streaming path")
        self.assertEqual(reopened.evidence(counterinspection["evidence_id"])[1]["counterinspection.txt"],
                         b"The streaming path is disabled; no faithful implementation exists")
        restored = self.root / "withdrawn_source"
        reopened.materialize(reopened.get_candidate(alleged_faithful)["source_id"], restored)
        self.assertEqual((restored / "kernel.py").read_text(), source_text)

    def test_fidelity_withdrawal_rejects_stale_empty_and_wrong_epoch_corrections(self):
        evidence = self.raw(self.a, b"Initial provisional witness", "witness.txt")
        witness = {"candidate_id": self.a, "claim": "Mechanism appears present", "evidence_refs": [evidence]}
        first = self.checkpoint(card=self.card(last_faithful=witness))
        first_ref = first["note"]["archive_card_ref"]
        current = self.checkpoint(revision=1, card=self.card(last_faithful=witness, previous_ref=first_ref))
        current_ref = current["note"]["archive_card_ref"]
        counterinspection = self.raw(self.a, b"The claimed mechanism is absent", "counterinspection.txt")
        older = self.candidate("historical source", epoch="older")
        old_evidence = self.raw(older, b"Historical counterinspection", "counterinspection.txt")
        correction = {"card_ref": current_ref, "field": "last_faithful", "reason": "Retract a mistaken fidelity witness",
                      "evidence_refs": [counterinspection]}
        defects = {"stale_target": {"card_ref": first_ref}, "empty_evidence": {"evidence_refs": []},
                   "wrong_epoch": {"evidence_refs": [old_evidence]}, "empty_reason": {"reason": " "}}

        for name, changes in defects.items():
            with self.subTest(defect=name), self.assertRaises(ArchiveError):
                self.checkpoint(revision=2, card=self.card(last_faithful=None, corrections=[dict(correction, **changes)]))
            self.assertEqual(self.store.get("D1"), current)
            self.assertEqual(read_archive(self.store)["directions"]["D1"]["card"]["last_faithful"], witness)

    def test_protection_receipt_is_readable_scoped_metadata_before_commit(self):
        audit = self.raw(self.a, b"provisional fidelity audit", "audit.txt")
        for payload in (b"ordinary text", [], {"direction_id": "wrong"}):
            opportunity = self.raw(self.a, payload)
            card = self.card(last_faithful={"candidate_id": self.a, "claim": "Provisionally faithful", "evidence_refs": [audit]},
                             protection={"candidate_id": self.a, "opportunity_ref": opportunity, "rationale": "Test prerequisite"})
            with self.subTest(payload=payload), self.assertRaises(ArchiveError):
                self.checkpoint(card=card)
        self.assertEqual(self.store.get(), {})

    def test_typed_decision_refs_are_validated_before_checkpoint(self):
        bad = self.raw(self.a, {"kind": "evo_decision", "schema_version": 2})
        with self.assertRaises(ArchiveError):
            self.checkpoint(card=self.card(decision_refs=[bad]))

    def test_correction_retains_prior_claim_and_rejects_other_direction(self):
        first = self.checkpoint()
        first_ref = first["note"]["archive_card_ref"]
        revised = self.card(previous_ref=first_ref, corrections=[{"card_ref": first_ref, "field": "idea.statement",
                               "reason": "Compiler did not realize intended overlap", "evidence_refs": []}])
        revised["idea"]["statement"] = "Overlap remains unestablished"
        self.checkpoint(revision=1, card=revised)
        self.assertEqual(read_record(self.store, first_ref)["idea"]["statement"], "Test streaming summary")
        self.assertEqual(read_archive(self.store)["directions"]["D1"]["card"]["idea"]["statement"], "Overlap remains unestablished")
        with self.assertRaises(ArchiveError):
            self.checkpoint(direction="D2", card=revised)
        revised["corrections"][0]["field"] = "idea.made_up_field"
        with self.assertRaisesRegex(ArchiveError, "field does not exist"):
            self.checkpoint(revision=2, card=revised)

    def test_compilation_case_and_output_joins(self):
        binary1 = self.raw(self.a, b"synthetic binary one", "kernel.bin")
        binary2 = self.raw(self.a, b"synthetic binary two", "kernel.bin")
        compilation = self.record(self.a, "evo_compilation", case_ids=["short"], recipe_id="1" * 64, outputs={"ISA": binary1})
        diagnostic = dict(case_ids=["short"], compilation_ref=compilation, artifacts={"ISA": binary1},
                          claim="Synthetic emitted instruction observation", unknown_reason=None)
        valid = self.record(self.a, "evo_diagnostic", **diagnostic)
        counter = self.raw(self.a, b"synthetic post-build counter output", "counters.txt")
        self.record(self.a, "evo_diagnostic", **dict(diagnostic, supplemental_refs=[counter]))
        self.checkpoint(card=self.card(observation_refs=[valid]))
        read_archive(self.store, frozen_cases=self.cases)
        with self.assertRaisesRegex(ArchiveError, "frozen case registry"):
            read_archive(self.store)
        for changes, message in (({"artifacts": {"ISA": binary2}}, "this compilation"),
                                 ({"artifacts": {"renamed_code": binary2}}, "this compilation"),
                                 ({"case_ids": ["long"]}, "compilation case"),
                                 ({"case_ids": ["fabricated"]}, "unregistered")):
            with self.subTest(changes=changes), self.assertRaisesRegex(ArchiveError, message):
                self.record(self.a, "evo_diagnostic", **dict(diagnostic, **changes))
        other = self.candidate("other source")
        with self.assertRaisesRegex(ArchiveError, "candidate mismatch"):
            self.record(other, "evo_diagnostic", **diagnostic)
        with self.assertRaisesRegex(ArchiveError, "one case/specialization"):
            self.record(self.a, "evo_compilation", case_ids=["short", "long"], recipe_id="1" * 64, outputs={"ISA": binary1})

    def test_long_history_is_iterative_and_every_committed_card_reopens(self):
        prior = None
        for revision in range(80):
            card = self.card(**({"previous_ref": prior} if prior else {}))
            state = self.checkpoint(revision=revision, card=card)
            prior = state["note"]["archive_card_ref"]
        reopened = ResearchStore(self.store.root, **self.options)
        row = read_archive(reopened)["directions"]["D1"]
        self.assertEqual(row["revision"], 80)
        self.assertEqual(row["card_ref"], prior)

    def test_unknown_compilation_keeps_failure_raw_and_reason(self):
        failure = self.raw(self.a, b"compile failed before ISA", "compile.log")
        ref = self.record(self.a, "evo_diagnostic", case_ids=["short"], compilation_ref=None,
                          artifacts={"compile_log": failure}, claim="Compilation failed", unknown_reason="No code object emitted")
        self.assertIsNone(read_record(self.store, ref, frozen_cases=self.cases)["compilation_ref"])
        with self.assertRaises(ArchiveError):
            self.record(self.a, "evo_diagnostic", case_ids=["short"], compilation_ref=None,
                        artifacts={}, claim="Unknown", unknown_reason=None)

    def test_stale_write_and_unreferenced_objects_cannot_change_current_card(self):
        first = self.checkpoint()
        other = self.candidate("new source")
        with self.assertRaises(StateError):
            self.checkpoint(other, revision=0)
        reopened = ResearchStore(self.store.root, **self.options)
        row = read_archive(reopened)["directions"]["D1"]
        self.assertEqual(row["card_ref"], first["note"]["archive_card_ref"])
        self.assertEqual(row["tip"], self.a)

    def test_wrong_epoch_source_and_unsafe_artifact_refs(self):
        older = self.candidate("same mathematical job", epoch="older")
        for ref in (self.raw(older, b"old", "old.log"),
                    {"evidence_id": "0" * 64, "artifact": "../outside"}):
            with self.subTest(ref=ref), self.assertRaises(ArchiveError):
                self.checkpoint(card=self.card(observation_refs=[ref]))
        record = {"kind": "evo_archive_card", "direction_id": "D1", **self.card(),
                  "binding": {"job_id": "job", "measurement_epoch_id": "epoch", "candidate_source_id": "0" * 64}}
        with self.assertRaisesRegex(ArchiveError, "binding mismatch"):
            archive_record(self.store, self.a, record)

    def test_metadata_cannot_claim_anchor_and_unsupported_version_fails(self):
        with self.assertRaises(ArchiveError):
            self.checkpoint(card=self.card(anchor={"candidate_id": self.a, "eligible": True}))
        with self.assertRaisesRegex(ArchiveError, "unsupported metadata version"):
            self.checkpoint(card=self.card(schema_version=2))
        state = self.checkpoint()
        changed = self.candidate("changed outside adapter")
        self.store.checkpoint("D1", changed, note=state["note"], expected_revision=1, evidence=state["evidence"])
        with self.assertRaisesRegex(ArchiveError, "misbound current card"):
            read_archive(self.store)

    def test_missing_or_changed_referenced_bytes_are_rejected(self):
        raw = self.raw(self.a, b"source audit", "audit.txt")
        self.checkpoint(card=self.card(observation_refs=[raw]))
        envelope, _ = self.store.evidence(raw["evidence_id"])
        blob = self.store.root / "objects" / envelope["artifacts"]["audit.txt"]
        blob.chmod(0o600)
        blob.write_bytes(b"tampered")
        with self.assertRaises(StateError):
            read_archive(self.store)


if __name__ == "__main__":
    unittest.main()
