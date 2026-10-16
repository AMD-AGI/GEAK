"""Tests for geak_evo.archive.maturation (ported from tests/test_maturation.py).

CPU lifecycle fixtures; the receipt verifier is synthetic, never native proof.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_maturation.py
Parity run on the original modules: set EVO_PARITY_ORIGINAL=1 and put the
original ``evolution/src`` and ``bridge`` on PYTHONPATH instead.
"""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

if os.environ.get("EVO_PARITY_ORIGINAL"):
    from geak_evo.maturation import (MaturationError, checkpoint_maturation, read_maturation,
                                    pending_maturation_reviews, validate_protection)
    from geak_evo.research_state import ResearchStore
    from geak_evo.selection import account_opportunities
    from native_maturation import COMMON_SCOPE, opportunity_view
else:
    from geak_evo.archive.maturation import (COMMON_SCOPE, MaturationError, checkpoint_maturation,
                                             opportunity_view, pending_maturation_reviews,
                                             read_maturation, validate_protection)
    from geak_evo.archive.store import ResearchStore
    from geak_evo.selection.work import account_opportunities


JOB = "1" * 64
BINDING = {"job_id": JOB, "measurement_epoch_id": "epoch"}
LIMITS = {"engineer_active_seconds": 100, "model_usd": 10}


class MaturationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.options = {"allowed_epochs": [(JOB, "epoch"), (JOB, "old")],
                        "source_paths": {JOB: ["kernel.py"]},
                        "verify_evidence": lambda raw, expected: {k: True for k in ("independent", "settled", "correct", "eligible")}}
        self.store = ResearchStore(self.root / "archive", **self.options)
        self.anchor = self.candidate("baseline fixture")
        self.note = {"claim": "fixture only", "scope": "synthetic job", "next_test": "inspect exact source", "external_note": "preserve"}
        self.store.checkpoint("D", self.anchor, note=self.note, expected_revision=0)
        proof = self.raw(self.anchor, {"synthetic": True})
        self.store.promote("D", self.anchor, proof["evidence_id"], expected_revision=1)
        self.faithful = self.candidate("streaming summary present; syntax !!!", parents=[self.anchor])
        self.set_tip(self.faithful)
        self.witness = self.raw(self.faithful, b"Scoped streaming invariant survives; compile/correctness unqualified", "witness.txt")
        self.authorized = set()

    def candidate(self, source, *, parents=(), epoch="epoch"):
        (self.work / "kernel.py").write_text(source)
        source_id = self.store.capture_source(self.work, ["kernel.py"])
        return self.store.candidate(JOB, epoch, source_id, parents=parents)

    def raw(self, candidate, value, artifact="receipt.json"):
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        return {"evidence_id": self.store.archive_evidence(candidate, {artifact: data}), "artifact": artifact}

    def set_tip(self, candidate):
        state = self.store.get("D")
        return self.store.checkpoint("D", candidate, note=state["note"], expected_revision=state["revision"],
                                     status=state["status"], resources=state["resources"], evidence=state["evidence"])

    def decision(self, action, **fields):
        return dict(action=action, reason="Resolve the stated implementation obligation", next_test="Inspect the emitted dependency", evidence_refs=[], **fields)

    def commit(self, decision, **kwargs):
        return checkpoint_maturation(self.store, "D", decision=decision,
                                     expected_revision=self.store.get("D")["revision"], **kwargs)

    def nominate(self):
        return self.commit(self.decision("nominate", proposition="Stream the sufficient summary without materialization",
                           scope="Complete operation under the registered epoch",
                           last_faithful={"candidate_id": self.faithful, "claim": "Source-level partial fidelity, currently invalid",
                                          "evidence_refs": [self.witness]}))

    def opportunity(self, key="O1", *, settled=False, cost=0, sequence=0):
        return dict(BINDING, opportunity_id=key, direction_id="D", engineer_id="E", island_id="I",
                    question="Repair the faithful summary", opened_at=0, review_at=30,
                    requested={"engineer_active_seconds": 20, "model_usd": 2},
                    reserved={"engineer_active_seconds": 20, "model_usd": 2},
                    receipts=[{"receipt_id": "cost-" + key, "costs": {"engineer_active_seconds": cost, "model_usd": cost / 10}}],
                    settled=settled, completed_sequence=sequence if settled else None, progress_complete=settled)

    def accounting(self, records=None, now=10):
        return account_opportunities(records or [self.opportunity()], binding=BINDING, limits=LIMITS, deadline=100, now=now)

    def grant(self, accounting=None, key="O1"):
        return self.commit(self.decision("grant", opportunity_id=key, release_condition="Review the unresolved compiler obligation",
                                        displaced_work="Incumbent tuning receives less of the same budget"),
                           accounting=accounting or self.accounting())

    def observation(self, *, candidate=None, delivered=True, work="implementation_or_diagnosis", settled=True, interrupted=False, key="O1", authorize=True):
        value = dict(kind="maturation_observation", binding=BINDING, direction_id="D", opportunity_id=key,
                     assigned_candidate_id=self.faithful, candidate_id=candidate or self.faithful,
                     delivered=delivered, work_kind=work, settled=settled, interrupted=interrupted)
        ref = self.raw(candidate or self.faithful, value)
        if authorize:
            self.authorized.add((ref["evidence_id"], ref["artifact"]))
        return ref

    def verifier(self, ref, data, expected):
        if (ref["evidence_id"], ref["artifact"]) not in self.authorized:
            return False
        value = json.loads(data)
        if "expected" in value:
            return value["expected"] == expected
        return all(value.get(key) == item for key, item in expected.items())

    def observe(self, ref=None, accounting=None):
        return self.commit(self.decision("observe"), observation_ref=ref or self.observation(),
                           accounting=accounting or self.accounting([self.opportunity(settled=True, cost=10)]),
                           verify_host=self.verifier)

    def review(self, outcome="continue", **fields):
        return self.decision("review", outcome=outcome, changed_since_prior="The compiler dependency is now localized",
                             revival_condition="A changed lowering prerequisite or scoped replication", **fields)

    def test_invalid_faithful_source_gets_finite_opportunity_without_anchor_promotion(self):
        nomination = self.nominate()
        granted = self.grant()
        view = validate_protection(self.store, granted["ref"], direction_id="D", candidate_id=self.faithful,
                                   opportunity_id="O1", accounting=self.accounting())
        self.assertEqual(view["remaining_reservation"], {"engineer_active_seconds": 20, "model_usd": 2})
        self.assertIsNone(granted["record"]["observation"])
        self.assertFalse(view["native_eligible"])
        self.assertEqual(self.store.get("D")["anchor"]["candidate_id"], self.anchor)
        self.assertEqual(self.store.get("D")["tip"], self.faithful)
        self.assertEqual(self.store.get("D")["note"]["external_note"], "preserve")
        with self.assertRaisesRegex(MaturationError, "stale"):
            validate_protection(self.store, nomination["ref"], direction_id="D", candidate_id=self.faithful,
                                opportunity_id="O1", accounting=self.accounting())

    def test_fallback_and_new_tip_do_not_erase_last_faithful_or_anchor(self):
        self.nominate(); self.grant()
        fallback = self.candidate("fast fallback deletes the streaming summary", parents=[self.faithful])
        self.set_tip(fallback)
        self.observe(self.observation(candidate=fallback))
        reopened = ResearchStore(self.store.root, **self.options)
        record = read_maturation(reopened, "D")
        self.assertEqual(record["last_faithful"]["candidate_id"], self.faithful)
        self.assertEqual(reopened.get("D")["tip"], fallback)
        self.assertEqual(reopened.get("D")["anchor"]["candidate_id"], self.anchor)
        decision = self.review(last_faithful=None)
        with self.assertRaisesRegex(MaturationError, "cannot erase"):
            self.commit(decision, accounting=self.accounting([self.opportunity(settled=True, cost=10)]))

    def test_fidelity_retraction_preserves_history_and_requires_new_evidence_to_reinstate(self):
        nomination = self.nominate()
        witness = nomination["record"]["last_faithful"]
        correction = self.raw(self.faithful, b"Inspection found the claimed mechanism disabled.", "correction.txt")
        decision = self.review("shelve", last_faithful=None, retract_faithful_ref=nomination["ref"])
        decision.update(reason="Withdraw the mistaken fidelity assertion", evidence_refs=[correction])
        retracted = self.commit(decision)
        self.assertIsNone(read_maturation(self.store, "D")["last_faithful"])
        self.assertEqual(retracted["record"]["lifecycle"], "shelved")
        self.assertEqual(self.store.get("D")["anchor"]["candidate_id"], self.anchor)
        self.assertEqual(self.store.get("D")["tip"], self.faithful)
        historical = self.store.evidence(nomination["ref"]["evidence_id"])[1]
        self.assertEqual(json.loads(historical["maturation.json"])["last_faithful"], witness)
        self.store.materialize(self.store.get_candidate(self.faithful)["source_id"], self.root / "retained-source")
        revival = self.decision("revive", trigger_kind="scoped_replication", trigger_reason="Reinspect the disputed path")
        revival["evidence_refs"] = [correction]
        self.commit(revival)
        with self.assertRaisesRegex(MaturationError, "faithful source witness"):
            self.grant()
        grant = self.decision("grant", opportunity_id="O1", release_condition="Review new evidence",
                              displaced_work="Incumbent repair", last_faithful=witness)
        with self.assertRaisesRegex(MaturationError, "withdrawn witness needs new source evidence"):
            self.commit(grant, accounting=self.accounting())
        copied = self.raw(self.faithful, self.store.evidence(self.witness["evidence_id"])[1]["witness.txt"], "rewrapped.txt")
        grant["last_faithful"] = dict(witness, evidence_refs=[copied])
        with self.assertRaisesRegex(MaturationError, "new source evidence"):
            self.commit(grant, accounting=self.accounting())
        new_evidence = self.raw(self.faithful, b"New scoped inspection resolves which source path is active.", "reinspection.txt")
        grant["last_faithful"] = dict(witness, claim="Revised scoped fidelity claim", evidence_refs=[new_evidence])
        restored = self.commit(grant, accounting=self.accounting())
        self.assertEqual(restored["record"]["grant"]["faithful_seed"], grant["last_faithful"])

    def test_correction_only_withdrawal_keeps_inflight_grant_and_settlement_obligations(self):
        self.nominate()
        granted = self.grant()
        correction = self.raw(self.faithful, b"The grant's inspected seed did not implement its claimed path.", "correction.txt")
        decision = self.review("withdraw_fidelity", last_faithful=None, retract_faithful_ref=granted["ref"])
        decision["evidence_refs"] = [correction]
        withdrawn = self.commit(decision)
        self.assertEqual(withdrawn["record"]["lifecycle"], "active")
        self.assertEqual(withdrawn["record"]["grant"], granted["record"]["grant"])
        self.assertEqual(withdrawn["record"]["used_opportunities"], ["O1"])
        with self.assertRaisesRegex(MaturationError, "withdrawn"):
            validate_protection(self.store, withdrawn["ref"], direction_id="D", candidate_id=self.faithful,
                                opportunity_id="O1", accounting=self.accounting())
        settled = self.accounting([self.opportunity(settled=True, cost=10)])
        owed = pending_maturation_reviews(self.store, settled)
        self.assertEqual(len(owed), 1)
        self.assertTrue(owed[0]["needs_observation"])
        self.assertTrue(owed[0]["needs_review"])
        self.observe(accounting=settled)
        self.assertEqual(read_maturation(self.store, "D")["grant"], granted["record"]["grant"])
        self.commit(self.review("continue"), accounting=settled)
        self.assertEqual(pending_maturation_reviews(self.store, settled), [])
        with self.assertRaisesRegex(MaturationError, "faithful source witness"):
            self.grant(accounting=self.accounting([self.opportunity(settled=True, cost=10), self.opportunity("O2")]), key="O2")

    def test_fidelity_retraction_rejects_stale_target_empty_or_wrong_epoch_evidence(self):
        nomination = self.nominate()
        granted = self.grant()
        correction = self.raw(self.faithful, b"Counter-inspection", "correction.txt")
        foreign = self.candidate("older epoch", epoch="old")
        foreign_evidence = self.raw(foreign, b"Counter-inspection from another epoch", "correction.txt")
        decision = self.review("withdraw_fidelity", last_faithful=None, retract_faithful_ref=granted["ref"])
        decision["evidence_refs"] = [correction]
        for changes in ({"retract_faithful_ref": nomination["ref"]}, {"evidence_refs": []},
                        {"evidence_refs": [foreign_evidence]}, {"reason": ""}):
            with self.subTest(changes=changes), self.assertRaises(MaturationError):
                self.commit(dict(decision, **changes))
            self.assertEqual(self.store.get("D")["note"]["maturation_ref"], granted["ref"])

    def test_rewording_or_replacing_current_witness_cannot_restore_a_withdrawn_grant(self):
        self.nominate()
        granted = self.grant()
        witness = granted["record"]["grant"]["faithful_seed"]
        copied = self.raw(self.faithful, self.store.evidence(self.witness["evidence_id"])[1]["witness.txt"], "repacked.txt")
        reworded = dict(witness, claim="Reworded attribution of the same source path", evidence_refs=[copied])
        updated = self.commit(self.decision("observe", last_faithful=reworded),
                              observation_ref=self.observation(settled=False),
                              accounting=self.accounting(), verify_host=self.verifier)
        correction = self.raw(self.faithful, b"The source path attributed to this seed is absent.", "correction.txt")
        decision = self.review("withdraw_fidelity", last_faithful=None, retract_faithful_ref=updated["ref"])
        decision["evidence_refs"] = [correction]
        self.commit(decision)
        replacement = self.candidate("a distinct inspected implementation", parents=[self.faithful])
        replacement_witness = {"candidate_id": replacement, "claim": "A different witnessed source",
                               "evidence_refs": [self.raw(replacement, b"Inspect the distinct path.", "witness.txt")]}
        current = self.commit(self.decision("observe", last_faithful=replacement_witness),
                              observation_ref=self.observation(candidate=replacement, settled=False),
                              accounting=self.accounting(), verify_host=self.verifier)
        self.assertEqual(current["record"]["grant"], granted["record"]["grant"])
        with self.assertRaisesRegex(MaturationError, "withdrawn"):
            validate_protection(self.store, current["ref"], direction_id="D", candidate_id=self.faithful,
                                opportunity_id="O1", accounting=self.accounting())
        settled = self.opportunity(settled=True, cost=10)
        self.observe(accounting=self.accounting([settled]))
        self.commit(self.review(), accounting=self.accounting([settled]))
        fresh = self.raw(self.faithful, b"A new scoped inspection establishes the seed's active path.", "new.txt")
        decision = self.decision("grant", opportunity_id="O2", release_condition="Review the fresh inspection",
                                 displaced_work="Incumbent repair", last_faithful=dict(witness, evidence_refs=[fresh]))
        ledger = self.accounting([settled, self.opportunity("O2")])
        renewed = self.commit(decision, accounting=ledger)
        checked = validate_protection(self.store, renewed["ref"], direction_id="D", candidate_id=self.faithful,
                                      opportunity_id="O2", accounting=ledger)
        self.assertEqual(checked["faithful_seed"], decision["last_faithful"])

    def test_reinstatement_requires_evidence_new_across_all_prior_retractions(self):
        nominated = self.nominate()
        witness = nominated["record"]["last_faithful"]
        correction = self.raw(self.faithful, b"Counter-inspection of the attributed path.", "correction.txt")

        def withdraw(ref):
            decision = self.review("withdraw_fidelity", last_faithful=None, retract_faithful_ref=ref)
            decision["evidence_refs"] = [correction]
            return self.commit(decision)

        withdraw(nominated["ref"])
        second_evidence = self.raw(self.faithful, b"A second independent source inspection.", "second.txt")
        second = self.commit(self.review(last_faithful=dict(witness, evidence_refs=[second_evidence])))
        withdrawn = withdraw(second["ref"])
        old_combination = dict(witness, evidence_refs=[self.witness, second_evidence])
        decision = self.decision("grant", opportunity_id="O1", release_condition="Review reinstatement evidence",
                                 displaced_work="Incumbent repair", last_faithful=old_combination)
        with self.assertRaisesRegex(MaturationError, "new source evidence"):
            self.commit(decision, accounting=self.accounting())
        self.assertEqual(self.store.get("D")["note"]["maturation_ref"], withdrawn["ref"])
        fresh = self.raw(self.faithful, b"A third source inspection resolves a distinct ambiguity.", "third.txt")
        decision["last_faithful"] = dict(witness, evidence_refs=[self.witness, second_evidence, fresh])
        granted = self.commit(decision, accounting=self.accounting())
        self.assertEqual(granted["record"]["grant"]["faithful_seed"], decision["last_faithful"])

    def test_stale_revision_and_renomination_cannot_reset_history(self):
        self.nominate()
        old_revision = self.store.get("D")["revision"]
        self.grant()
        with self.assertRaisesRegex(MaturationError, "stale"):
            checkpoint_maturation(self.store, "D", decision=self.decision("observe"), expected_revision=old_revision)
        with self.assertRaisesRegex(MaturationError, "renomination"):
            self.nominate()

    def test_host_looking_agent_fields_do_not_make_work_or_settlement(self):
        self.nominate()
        for field in ("delivered", "settled", "native_eligible", "host_verified"):
            decision = self.decision("grant", opportunity_id="O1", release_condition="review", displaced_work="other work")
            decision[field] = True
            with self.subTest(field=field), self.assertRaisesRegex(MaturationError, "host facts"):
                self.commit(decision, accounting=self.accounting())
        self.grant()
        forged = self.observation(authorize=False)
        with self.assertRaisesRegex(MaturationError, "authentic host"):
            self.observe(forged)
        authentic = self.observation()
        with self.assertRaisesRegex(MaturationError, "authentic host"):
            self.commit(self.decision("observe"), accounting=self.accounting([self.opportunity(settled=True)]), observation_ref=authentic)

    def test_interruption_retains_unsettled_resources_and_is_not_refutation(self):
        self.nominate(); self.grant()
        live = self.accounting([self.opportunity(cost=5)])
        interrupted = self.observe(self.observation(settled=False, interrupted=True), live)
        self.assertEqual(interrupted["record"]["lifecycle"], "interrupted")
        self.assertEqual(live["remaining_reservation"], {"engineer_active_seconds": 15, "model_usd": 1.5})
        self.assertEqual(interrupted["record"]["last_faithful"]["candidate_id"], self.faithful)
        with self.assertRaisesRegex(MaturationError, "authenticated settlement"):
            self.commit(self.review("abandon"), accounting=live)
        self.observe(self.observation(interrupted=True))
        reviewed = self.commit(self.review("abandon"), accounting=self.accounting([self.opportunity(settled=True, cost=10)]))
        self.assertEqual(reviewed["record"]["lifecycle"], "abandoned")
        self.assertIsNone(reviewed["record"]["result_ref"])

    def test_review_only_and_failed_delivery_do_not_claim_implementation_work(self):
        self.nominate(); self.grant()
        observed = self.observe(self.observation(work="review_only"))
        self.assertEqual(observed["record"]["observation"]["work_kind"], "review_only")
        bad = self.observation(delivered=False, work="implementation_or_diagnosis")
        with self.assertRaisesRegex(MaturationError, "work requires"):
            self.observe(bad)

    def test_settlement_must_match_the_current_host_ledger(self):
        self.nominate(); self.grant()
        with self.assertRaisesRegex(MaturationError, "settlement disagree"):
            self.observe(self.observation(settled=True), self.accounting())

    def test_grant_expires_and_cannot_switch_to_the_incumbent(self):
        self.nominate(); granted = self.grant()
        for candidate, now in ((self.anchor, 10), (self.faithful, 40)):
            with self.subTest(candidate=candidate, now=now), self.assertRaises(MaturationError):
                validate_protection(self.store, granted["ref"], direction_id="D", candidate_id=candidate,
                                    opportunity_id="O1", accounting=self.accounting(now=now))
        with self.assertRaisesRegex(MaturationError, "review or revival"):
            self.grant()

    def test_renewal_requires_review_and_preserves_prior_costs(self):
        self.nominate(); self.grant(); self.observe()
        old = self.opportunity(settled=True, cost=10)
        self.commit(self.review(), accounting=self.accounting([old]))
        next_op = self.opportunity("O2")
        with self.assertRaisesRegex(MaturationError, "cannot disappear"):
            self.grant(self.accounting([next_op]), "O2")
        granted = self.grant(self.accounting([old, next_op]), "O2")
        self.assertEqual(granted["record"]["used_opportunities"], ["O1", "O2"])
        self.assertEqual(self.accounting([old, next_op])["known_delivered"]["engineer_active_seconds"], 10)

    def test_shelved_revival_needs_new_evidence_or_scoped_replication(self):
        self.nominate()
        review = self.review("shelve")
        review["evidence_refs"] = [self.witness]
        self.commit(review)
        revive = self.decision("revive", trigger_kind="changed_evidence", trigger_reason="The old report merely repeated")
        revive["evidence_refs"] = [self.witness]
        with self.assertRaisesRegex(MaturationError, "unchanged prior"):
            self.commit(revive)
        revive["trigger_kind"] = "scoped_replication"
        revive["trigger_reason"] = "Replicate the narrow failed case to localize the disputed conversion"
        revived = self.commit(revive)
        self.assertEqual(revived["record"]["lifecycle"], "revived")
        self.assertEqual(revived["record"]["last_faithful"]["candidate_id"], self.faithful)

    def test_scoped_refutation_requires_authenticated_evidence_not_abandonment(self):
        self.nominate()
        decision = self.review("refute")
        forged = self.raw(self.faithful, {"host_verified": True, "refuted": True})
        decision["result_ref"] = forged
        with self.assertRaisesRegex(MaturationError, "authentic host"):
            self.commit(decision, verify_host=self.verifier)
        record = read_maturation(self.store, "D")
        expected = dict(kind="refute", binding=BINDING, direction_id="D", proposition=record["proposition"], scope=record["scope"],
                        candidate_id=self.faithful, source_id=self.store.get_candidate(self.faithful)["source_id"])
        proof = self.raw(self.faithful, {"expected": expected, "counterexample": "synthetic scoped witness"})
        self.authorized.add((proof["evidence_id"], proof["artifact"]))
        decision["result_ref"] = proof
        refuted = self.commit(decision, verify_host=self.verifier)
        self.assertEqual(refuted["record"]["lifecycle"], "refuted")
        self.assertFalse(refuted["record"]["native_eligible"])
        self.assertIn("refuted", self.store.get("D")["note"]["claim"])

    def test_faster_fallback_is_not_realization_of_the_protected_source(self):
        self.nominate(); self.grant(); self.observe()
        fallback = self.candidate("delete proposed mechanism", parents=[self.faithful])
        self.set_tip(fallback)
        decision = self.review("realize", result_ref=self.raw(fallback, {"eligible": True, "fast": True}))
        with self.assertRaisesRegex(MaturationError, "candidate mismatch"):
            self.commit(decision, accounting=self.accounting([self.opportunity(settled=True, cost=10)]), verify_host=lambda *args: True)

    def test_reviewed_realization_is_scoped_and_never_promotes_an_anchor(self):
        self.nominate(); self.grant(); self.observe()
        record = read_maturation(self.store, "D")
        expected = dict(kind="realize", binding=BINDING, direction_id="D", proposition=record["proposition"],
                        scope=record["scope"], candidate_id=self.faithful,
                        source_id=self.store.get_candidate(self.faithful)["source_id"])
        proof = self.raw(self.faithful, {"expected": expected, "scope": "synthetic mechanism review only"})
        self.authorized.add((proof["evidence_id"], proof["artifact"]))
        result = self.commit(self.review("realize", result_ref=proof), verify_host=self.verifier,
                             accounting=self.accounting([self.opportunity(settled=True, cost=10)]))
        self.assertEqual(result["record"]["lifecycle"], "realized")
        self.assertEqual(result["record"]["scope"], record["scope"])
        self.assertFalse(result["record"]["native_eligible"])
        self.assertEqual(result["state"]["anchor"]["candidate_id"], self.anchor)

    def test_settled_missing_bill_does_not_disappear_when_obligation_is_shelved(self):
        self.nominate(); self.grant()
        missing = self.opportunity(settled=True)
        missing["receipts"] = []
        ledger = self.accounting([missing])
        self.observe(self.observation(work="none", delivered=False), ledger)
        result = self.commit(self.review("shelve"), accounting=ledger)
        self.assertEqual(result["record"]["lifecycle"], "shelved")
        self.assertEqual(ledger["remaining_reservation"], missing["reserved"])
        self.assertTrue(all(value is None for value in ledger["delivered"].values()))

    def test_old_epoch_witness_and_mutated_opportunity_are_rejected(self):
        old = self.candidate("historical faithful source", epoch="old")
        witness = self.raw(old, b"old evidence", "witness.txt")
        decision = self.decision("nominate", proposition="stream", scope="current job",
                                 last_faithful={"candidate_id": old, "claim": "historical", "evidence_refs": [witness]})
        with self.assertRaisesRegex(MaturationError, "job/epoch"):
            self.commit(decision)
        self.nominate(); granted = self.grant()
        changed = self.accounting()
        changed["opportunities"][0]["reserved"]["model_usd"] = 3
        with self.assertRaisesRegex(MaturationError, "envelope changed"):
            validate_protection(self.store, granted["ref"], direction_id="D", candidate_id=self.faithful,
                                opportunity_id="O1", accounting=changed)

    def test_comparison_anchor_stays_fixed_when_current_incumbent_changes(self):
        original = copy.deepcopy(self.store.get("D")["anchor"])
        nomination = self.nominate()
        self.assertEqual(nomination["record"]["comparison_anchor"], original)
        newer = self.candidate("new incumbent fixture", parents=[self.anchor])
        proof = self.raw(newer, {"synthetic": "a different validated incumbent"})
        self.store.promote("D", newer, proof["evidence_id"], expected_revision=self.store.get("D")["revision"])
        self.grant(); self.observe()
        old = self.opportunity(settled=True, cost=10)
        self.commit(self.review(), accounting=self.accounting([old]))
        renewal = self.grant(self.accounting([old, self.opportunity("O2")]), "O2")
        self.assertEqual(renewal["record"]["grant"]["comparison_anchor"], original)
        view = validate_protection(self.store, renewal["ref"], direction_id="D", candidate_id=self.faithful,
                                   opportunity_id="O2", accounting=self.accounting([old, self.opportunity("O2")]))
        self.assertEqual(view["comparison_anchor"], original)
        reopened = ResearchStore(self.store.root, **self.options)
        self.assertEqual(read_maturation(reopened, "D")["comparison_anchor"], original)
        self.assertEqual(reopened.get("D")["anchor"]["candidate_id"], newer)

    def test_missing_comparison_anchor_stays_unknown_after_later_promotion(self):
        self.store.checkpoint("U", self.faithful, note=self.note, expected_revision=0)
        nomination = checkpoint_maturation(self.store, "U", expected_revision=1,
            decision=self.decision("nominate", proposition="A source without a measured comparator", scope="CPU fixture",
                                   last_faithful={"candidate_id": self.faithful, "claim": "Fixture only",
                                                  "evidence_refs": [self.witness]}))
        self.assertIsNone(nomination["record"]["comparison_anchor"])
        proof = self.raw(self.anchor, {"synthetic": "late anchor"})
        self.store.promote("U", self.anchor, proof["evidence_id"], expected_revision=nomination["state"]["revision"])
        opportunity = self.opportunity()
        opportunity["direction_id"] = "U"
        ledger = self.accounting([opportunity])
        grant = checkpoint_maturation(self.store, "U", expected_revision=self.store.get("U")["revision"],
            decision=self.decision("grant", opportunity_id="O1", release_condition="Review source", displaced_work="Other diagnosis"),
            accounting=ledger)
        self.assertIsNone(grant["record"]["grant"]["comparison_anchor"])
        self.assertIsNone(validate_protection(self.store, grant["ref"], direction_id="U", candidate_id=self.faithful,
                                             opportunity_id="O1", accounting=ledger)["comparison_anchor"])

    def test_comparison_snapshot_resolves_evidence_and_cannot_change_in_history(self):
        self.nominate()
        granted = self.grant()
        for corruption in ("wrong_evidence", "changed_comparator"):
            record = copy.deepcopy(granted["record"])
            if corruption == "wrong_evidence":
                record["comparison_anchor"]["evidence_id"] = self.witness["evidence_id"]
                error = "comparison anchor evidence mismatch"
            else:
                record["comparison_anchor"] = None
                error = "fixed comparison anchor changed"
            record["grant"]["comparison_anchor"] = copy.deepcopy(record["comparison_anchor"])
            ref = self.raw(self.faithful, record, "maturation.json")
            state = self.store.get("D")
            self.store.checkpoint("D", state["tip"], note=dict(state["note"], maturation_ref=ref),
                                  expected_revision=state["revision"], evidence=state["evidence"] + [ref["evidence_id"]])
            with self.subTest(corruption=corruption), self.assertRaisesRegex(MaturationError, error):
                read_maturation(self.store, "D")

    def test_settled_protection_requires_actual_observation_then_agent_review(self):
        self.nominate(); self.grant()
        self.assertEqual(pending_maturation_reviews(self.store, self.accounting()), [])
        settled = self.accounting([self.opportunity(settled=True, cost=10)])
        due = pending_maturation_reviews(self.store, settled)
        self.assertEqual(len(due), 1)
        self.assertTrue(due[0]["needs_observation"])
        self.assertTrue(due[0]["needs_review"])
        self.assertEqual(due[0]["assigned_candidate_id"], self.faithful)
        self.observe(self.observation(work="none", delivered=False), settled)
        due = pending_maturation_reviews(self.store, settled)
        self.assertFalse(due[0]["needs_observation"])
        self.assertTrue(due[0]["needs_review"])
        self.commit(self.review("shelve"), accounting=settled)
        self.assertEqual(pending_maturation_reviews(self.store, settled), [])

    def test_obligations_from_another_campaign_ledger_are_not_imported(self):
        self.nominate(); self.grant()
        unrelated = self.opportunity("OTHER", settled=True, cost=10)
        self.assertEqual(pending_maturation_reviews(self.store, self.accounting([unrelated])), [])

    def test_fallback_workspace_can_receive_the_preserved_faithful_seed(self):
        self.nominate()
        fallback = self.candidate("fallback removed the summary", parents=[self.faithful])
        self.set_tip(fallback)
        grant = self.grant()
        self.assertEqual(grant["record"]["grant"]["start_candidate_id"], fallback)
        self.assertEqual(grant["record"]["grant"]["faithful_seed"]["candidate_id"], self.faithful)
        view = validate_protection(self.store, grant["ref"], direction_id="D", candidate_id=fallback,
                                   opportunity_id="O1", accounting=self.accounting())
        self.assertEqual(view["protected_candidate_id"], fallback)
        self.assertEqual(view["faithful_seed"]["evidence_refs"], [self.witness])
        self.assertEqual(view["faithful_seed_source_id"], self.store.get_candidate(self.faithful)["source_id"])
        self.assertEqual(self.store.get("D")["tip"], fallback)
        self.assertEqual(self.store.get("D")["anchor"]["candidate_id"], self.anchor)
        with self.assertRaisesRegex(MaturationError, "starting workspace"):
            validate_protection(self.store, grant["ref"], direction_id="D", candidate_id=self.faithful,
                                opportunity_id="O1", accounting=self.accounting())
        # Merely supplying a seed does not count as restoration or realization.
        observation = self.observation(candidate=fallback)
        payload = json.loads(self.store.evidence(observation["evidence_id"])[1][observation["artifact"]])
        payload["assigned_candidate_id"] = fallback
        observation = self.raw(fallback, payload)
        self.authorized.add((observation["evidence_id"], observation["artifact"]))
        self.observe(observation)
        with self.assertRaisesRegex(MaturationError, "candidate mismatch"):
            self.commit(self.review("realize", result_ref=self.raw(fallback, {"fast": True})),
                        accounting=self.accounting([self.opportunity(settled=True, cost=10)]), verify_host=lambda *args: True)

    def test_legacy_record_without_comparison_stays_unknown_and_bytes_unchanged(self):
        self.nominate()
        old = self.grant()["record"]
        old.pop("comparison_anchor")
        old["grant"].pop("comparison_anchor")
        old["grant"].pop("faithful_seed")
        # An archived schema-1 fixture predating these fields, with no invented
        # history binding to a modern nomination containing a comparator.
        old["previous_ref"] = None
        ref = self.raw(self.faithful, old, "maturation.json")
        before = self.store.evidence(ref["evidence_id"])[1][ref["artifact"]]
        state = self.store.get("D")
        self.store.checkpoint("D", state["tip"], note=dict(state["note"], maturation_ref=ref),
                              expected_revision=state["revision"], evidence=state["evidence"] + [ref["evidence_id"]])
        record = read_maturation(self.store, "D")
        self.assertIsNone(record["comparison_anchor"])
        view = validate_protection(self.store, ref, direction_id="D", candidate_id=self.faithful,
                                   opportunity_id="O1", accounting=self.accounting())
        self.assertIsNone(view["comparison_anchor"])
        self.assertEqual(view["faithful_seed"]["candidate_id"], self.faithful)
        self.assertEqual(self.store.evidence(ref["evidence_id"])[1][ref["artifact"]], before)


class OpportunityViewTests(unittest.TestCase):
    """The planner view merged in from the bridge: pure display, no grants."""

    def context(self):
        witness = {"candidate_id": "c" * 64, "claim": "Partial fidelity", "evidence_refs": []}
        return {"campaign_id": "campaign", "state": {},
                "archive": {"directions": {
                    "campaign": {"revision": 9, "tip": "0" * 64, "card": None},
                    "B": {"revision": 2, "tip": "b" * 64, "card": None},
                    "A": {"revision": 3, "tip": "a" * 64, "card": {"last_faithful": witness}}}},
                "maturation": {"A": {"lifecycle": "nominated", "last_faithful": witness}},
                "owed_maturation_reviews": [{"direction_id": "A"}]}

    def offer(self):
        return {"available": True, "reservation": {"model_usd": 1}, "operation_id": "op-1"}

    def test_named_witnesses_are_listed_without_any_grant(self):
        view = opportunity_view(self.context(), self.offer(),
                                [{"candidate_id": "c" * 64, "source_id": "s" * 64}], enabled=True)
        self.assertEqual(view["schema"], "native-maturation-opportunity-v1")
        self.assertEqual([(row["direction_id"], row["designation_record"], row["source_id"])
                          for row in view["named_faithful_witnesses"]],
                         [("A", "archive_card", "s" * 64), ("A", "maturation_record", "s" * 64)])
        self.assertFalse(view["automatic_grant"] or view["new_resource_credit"])
        self.assertTrue(all(row["grant_eligibility_asserted"] is False and row["faithfulness_verified"] is False
                            for row in view["named_faithful_witnesses"]))
        self.assertEqual(view["pending_maturation_reviews"], [{"direction_id": "A"}])
        self.assertNotIn("program_lease_protected_option", view)
        self.assertIn("faithful", COMMON_SCOPE)

    def test_unknown_witness_candidate_is_unmatched_and_treatment_must_be_explicit(self):
        view = opportunity_view(self.context(), self.offer(), [], enabled=False)
        self.assertEqual({row["source_catalog_match"] for row in view["named_faithful_witnesses"]}, {False})
        with self.assertRaises(ValueError):
            opportunity_view(self.context(), self.offer(), [], enabled=None)


if __name__ == "__main__":
    unittest.main()
