"""CPU-only search-policy checks (arm rules and read scope); no model or sandbox.

Ported from the old test_search_policy.py. The old tests that need CLIPlanner
(engineer/planner side, not part of this port) or EvolutionLoop.run are not
copied here; serial-G admission is tested directly with a synthetic snapshot.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from geak_evo.archive.policy import (ProposalError, lineage_candidates, native_capability_context, native_enabled,
                                     policy_contract, program_lease_bound, proposal_ref, validate_action,
                                     validate_lease, validate_policy, validate_raw_engineer_proposal,
                                     validate_read_scope)
from geak_evo.archive.records import archive_checkpoint
from geak_evo.archive.store import ResearchStore


class SearchPolicyTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.study, self.shared = self.root / "study", self.root / "shared"
        self.campaign, self.sibling = self.study / "g", self.study / "evo"
        for path in (self.study, self.shared, self.campaign, self.sibling):
            path.mkdir(mode=0o700)
        self.policy = {"arm": "G", "study_root": str(self.study), "shared_read_roots": [str(self.shared)]}
        self.actions = self.shared / "actions.md"
        self.actions.write_text("CPU action instructions.\n")
        self.job_id = "job"
        self.store = ResearchStore(self.campaign / "store", allowed_epochs=[(self.job_id, "epoch")],
                                   source_paths={self.job_id: ["implementation.py"]})
        self.source_dir = self.campaign / "source"
        self.source_dir.mkdir()
        self.seed = self.candidate("working baseline\n")
        self.broken = self.candidate("unfinished representation\n", parents=[self.seed])
        self.tip = self.candidate("repaired representation\n", parents=[self.broken])
        archive_checkpoint(self.store, "baseline", self.tip, expected_revision=0,
            card={"idea": {"idea_id": "representation", "statement": "Try different live state", "status": "hypothesis"}},
            note={"claim": "CPU fixture", "scope": "No performance measurement", "next_test": "Inspect history"})

    def candidate(self, text, **lineage):
        (self.source_dir / "implementation.py").write_text(text)
        source = self.store.capture_source(self.source_dir, ["implementation.py"])
        return self.store.candidate(self.job_id, "epoch", source, **lineage)

    def snapshot(self, history=(), notes=None):
        return {"campaign_id": "campaign",
                "archive": {"directions": {"campaign": {"tip": self.seed}, "baseline": {"tip": self.tip}}},
                "state": {"history": list(history), "binding": {"job_id": self.job_id, "measurement_epoch_id": "epoch"},
                          "opportunities": [{"opportunity_id": "op-1", "direction_id": "episode-1"}]},
                "direction_notes": notes or {}}

    def proposal(self, **updates):
        result = {"direction_id": "episode-2", "expected_revision": 0, "parent_candidate_id": self.seed,
                  "island_id": "unassigned", "membership_ref": None, "exchange_ref": None,
                  "protection_ref": None, "question": "Restore or combine earlier scratch experiments"}
        result.update(updates)
        return result

    def reviewed_history(self, *, settled=True, reviewed=True):
        roles = {"selected": self.tip, "unfinished": self.broken}
        collection = {"evidence_id": self.store.archive_evidence(
            self.broken, {"native_source_roles.json": json.dumps(roles).encode()}),
            "artifact": "native_source_roles.json"}
        transport = {"evidence_id": self.store.archive_evidence(self.broken, {"transport.json": b"{}"}),
                     "artifact": "transport.json"}
        row = {"attempt_id": "op-1", "collection_ref": collection, "transport_ref": transport,
               "candidate_id": self.broken, "settled": settled}
        note = {"research_status": "reported" if reviewed else "pending", "research_review_id": "review-1",
                "reviewed_outcome_refs": [transport, collection], "research_update_ref": collection,
                "latest_transport_ref": transport}
        return [row], {"episode-1": {"note": note}}

    def test_scope_rejects_siblings_parent_grants_aliases_and_overlapping_shared_roots(self):
        other = self.sibling / "result.txt"
        other.write_text("Other arm's result")
        link = self.campaign / "linked-result"
        link.symlink_to(other)
        validate_read_scope(self.policy, self.campaign, [self.actions, self.store.root])
        for path in (other, self.sibling, self.study, link, self.root):
            with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                validate_read_scope(self.policy, self.campaign, [path])
        for shared in (self.root, self.study, self.sibling):
            policy = dict(self.policy, shared_read_roots=[str(shared)])
            with self.subTest(shared=shared), self.assertRaisesRegex(ValueError, "disjoint"):
                validate_policy(policy, self.campaign)
        with self.assertRaisesRegex(ValueError, "below"):
            validate_policy(self.policy, self.study)

    def test_lineage_includes_slow_and_broken_history(self):
        self.assertEqual(set(lineage_candidates(self.store, [self.tip])), {self.seed, self.broken, self.tip})
        self.assertIn("slower tip", policy_contract(self.policy)["shared_capabilities"])

    def test_serial_g_first_episode_starts_from_the_seed(self):
        snapshot = self.snapshot()
        validate_action(self.policy, snapshot, self.proposal(), store=self.store)
        validate_action(self.policy, snapshot, {"kind": "research_review", "direction_id": "baseline"})
        validate_action(self.policy, snapshot, None)
        bad = [self.proposal(parent_candidate_id=self.tip), self.proposal(direction_id="baseline"),
               self.proposal(direction_id="campaign"), self.proposal(expected_revision=1),
               self.proposal(exchange_ref={"donor": "other"}), self.proposal(protection_ref={"grant": "portfolio"}),
               self.proposal(island_id="new-island"), self.proposal(semantic_donors=[self.tip]),
               {"kind": "maturation_review", "direction_id": "baseline"}]
        bad += [{"kind": "research_action", "action": action, "direction_id": "baseline"}
                for action in ("accept_anchor", "charter", "membership", "concept", "exchange", "nominate")]
        for proposal in bad:
            with self.subTest(proposal=proposal), self.assertRaises(ProposalError):
                validate_action(self.policy, snapshot, proposal, store=self.store)
        for proposal in bad:
            validate_action(dict(self.policy, arm="EVO"), snapshot, proposal)

    def test_serial_g_later_episode_uses_reviewed_source_roles(self):
        history, notes = self.reviewed_history()
        snapshot = self.snapshot(history, notes)
        for parent in (self.tip, self.broken):
            validate_action(self.policy, snapshot, self.proposal(parent_candidate_id=parent), store=self.store)
        with self.assertRaisesRegex(ProposalError, "exact source role"):
            validate_action(self.policy, snapshot, self.proposal(), store=self.store)
        for history, notes, message in (self.reviewed_history(settled=False) + ("must be settled",),
                                        self.reviewed_history(reviewed=False) + ("scientific review first",)):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message) as caught:
                validate_action(self.policy, self.snapshot(history, notes), self.proposal(), store=self.store)
            self.assertNotIsInstance(caught.exception, ProposalError)
        with self.assertRaisesRegex(ValueError, "exact campaign store") as caught:
            validate_action(self.policy, self.snapshot(), self.proposal())
        self.assertNotIsInstance(caught.exception, ProposalError)

    def test_policy_fields_capabilities_and_program_lease(self):
        for policy, message in ((dict(self.policy, extra=1), "requires arm"), (dict(self.policy, arm="X"), "G or EVO"),
                                (dict(self.policy, native_capabilities=["cells", "cells"]), "invalid enabled"),
                                (dict(self.policy, native_capabilities=["magic"]), "invalid enabled"),
                                (dict(self.policy, native_capabilities=["program_lease"]), "exactly two"),
                                (dict(self.policy, arm="EVO", native_capabilities=["continuation", "program_lease"],
                                      program_lease_max_episodes=3), "exactly two"),
                                (dict(self.policy, shared_read_roots=str(self.shared)), "explicit list"),
                                (dict(self.policy, shared_read_roots=[str(self.shared)] * 2), "duplicate")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_policy(policy)
        with self.assertRaisesRegex(ValueError, "must be explicit"):
            validate_policy(self.policy, require_native_capabilities=True)
        self.assertIsNone(validate_policy(None))
        lease = dict(self.policy, arm="EVO", native_capabilities=["continuation", "program_lease"])
        self.assertEqual(validate_policy(lease), lease)
        self.assertEqual(program_lease_bound(lease), 2)
        self.assertEqual(program_lease_bound(self.policy), 0)
        self.assertTrue(native_enabled(None, "cells"))
        self.assertFalse(native_enabled(None, "program_lease"))
        self.assertTrue(native_enabled(self.policy, "puct"))
        self.assertFalse(native_enabled(dict(self.policy, native_capabilities=[]), "puct"))
        self.assertEqual(native_capability_context(self.policy), "")
        self.assertIn('["continuation", "program_lease"]', native_capability_context(lease))
        terms = {"episodes": 2, "release_condition": "after two", "displaced_work": "other line"}
        validate_lease({"lease": terms}, 2)
        for bound, value in ((3, terms), (2, dict(terms, episodes=3)), (2, dict(terms, displaced_work=" "))):
            with self.subTest(bound=bound, value=value), self.assertRaises(ProposalError):
                validate_lease({"lease": value}, bound)

    def test_island_group_rules(self):
        registry, other = self.study / "registry", self.study / "other"
        registry.mkdir()
        group = {"group_id": "group", "registry_root": str(registry),
                 "members": [str(self.campaign), str(self.sibling)]}
        evo = dict(self.policy, arm="EVO", island_group=group)
        validate_policy(evo, self.campaign)
        contract = policy_contract(evo)
        self.assertIn("concurrent island", contract["treatment"])
        self.assertIn("serial continuing", policy_contract(self.policy)["treatment"])
        self.assertEqual(policy_contract(None)["arm"], "unregistered")
        cases = ((dict(evo, arm="G"), self.campaign, "only EVO"),
                 (dict(evo, island_group=dict(group, members=[str(self.campaign)])), self.campaign, "distinct"),
                 (dict(evo, island_group=dict(group, members=[str(self.campaign), str(self.campaign / "x")])),
                  self.campaign, "nested"),
                 (dict(evo, island_group=dict(group, registry_root=str(self.campaign))), self.campaign, "disjoint"),
                 (evo, other, "not a declared island group member"))
        for policy, campaign, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_policy(policy, campaign)

    def test_raw_engineer_proposal_checks(self):
        claim = {"evidence_id": "a" * 64, "artifact": "claim.json"}
        raw = {"direction_id": "d", "expected_revision": 0, "parent_candidate_id": "p", "island_id": "i",
               "engineer_id": "e", "question": "q", "reason": "r", "evidence_uptake": "u",
               "premises": [{"claim": claim, "use": "hypothesis", "reason": "r", "next_test": "t"}],
               "repeat_delta": {"kind": "new_evidence", "reason": "r", "expected_observation": "o",
                                "reviewed_attempts": ["a1"], "evidence": [claim]},
               "exchange_ref": None, "protection_ref": None, "override_reason": None, "membership_ref": None,
               "semantic_donors": ["donor"]}
        validate_raw_engineer_proposal(raw)
        changes = [({"extra": 1}, "unexpected \\['extra'\\]"), ({"question": " "}, "invalid raw engineer field"),
                   ({"expected_revision": -1}, "revision"), ({"override_reason": ""}, "override"),
                   ({"semantic_donors": ["p"]}, "donor"),
                   ({"premises": [{"claim": dict(claim, artifact="../x"), "use": "fact", "reason": "",
                                   "next_test": ""}]}, "unsafe artifact name"),
                   ({"premises": [dict(raw["premises"][0], next_test=" ")]}, "hypothesis needs"),
                   ({"repeat_delta": dict(raw["repeat_delta"], evidence=[])}, "new evidence needs"),
                   ({"repeat_delta": dict(raw["repeat_delta"], kind="other")}, "invalid repeat delta")]
        for change, message in changes:
            with self.subTest(message=message), self.assertRaisesRegex(ProposalError, message):
                validate_raw_engineer_proposal(dict(copy.deepcopy(raw), **change))
        missing = dict(raw)
        del missing["membership_ref"]
        with self.assertRaisesRegex(ProposalError, "missing \\['membership_ref'\\]"):
            validate_raw_engineer_proposal(missing)
        self.assertEqual(proposal_ref(claim), json.dumps(claim, sort_keys=True, separators=(",", ":")))
        with self.assertRaisesRegex(ProposalError, "invalid evidence ID"):
            proposal_ref(dict(claim, evidence_id="A" * 64))


if __name__ == "__main__":
    unittest.main()
