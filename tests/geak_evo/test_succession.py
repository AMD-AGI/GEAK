"""Tests for geak_evo.archive.succession and geak_evo.archive.episodes.

Synthetic store fixtures only. Parts that need planner or engineer code
(prompt rendering, full episode checks) are tested only where that code is
importable.

Run with PYTHONPATH=<repo root>:  python3 -B tests/geak_evo/test_succession.py
Parity run on the original bridge modules: set EVO_PARITY_ORIGINAL=1 and put
the original ``evolution/src`` and ``bridge`` on PYTHONPATH instead.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

if os.environ.get("EVO_PARITY_ORIGINAL"):
    from geak_evo import research_state as store_module
    import native_evidence as episodes
    import native_succession as succession
    PLANNER_CONTEXT = True
else:
    from geak_evo.archive import episodes, succession
    from geak_evo.archive import store as store_module
    PLANNER_CONTEXT = importlib.util.find_spec("geak_evo.planner") is not None

BASE_NOTE = {"claim": "c", "scope": "s", "next_test": "t"}


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


class Fixture(unittest.TestCase):
    """One seed ``a`` and one engineer turn ``op1`` that produced ``b`` on direction d1."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name).resolve()
        work = base / "work"
        work.mkdir()
        (work / "k.py").write_text("v = 0\n")
        store = self.store = store_module.ResearchStore(
            base / "archive", allowed_epochs=[("job", "epoch-1")], source_paths={"job": ["k.py"]})
        self.a = store.candidate("job", "epoch-1", store.capture_source(work, ["k.py"]))
        (work / "k.py").write_text("v = 1\n")
        self.b_source = store.capture_source(work, ["k.py"])
        self.b = store.candidate("job", "epoch-1", self.b_source, parents=[self.a])
        transport = store.archive_evidence(self.b, {"transport.json": b"{}"})
        self.T = {"evidence_id": transport, "artifact": "transport.json"}
        self.roles = {"selected": self.b, "terminal": self.b}
        self.C = self.collection(self.roles)
        store.checkpoint("campaign", self.a, note=BASE_NOTE, expected_revision=0)
        store.checkpoint("d1", self.a, note=BASE_NOTE, expected_revision=0)
        self.review_note = dict(BASE_NOTE, research_status="reported", research_review_id="r1",
                                reviewed_outcome_refs=[self.T, self.C], research_update_ref=self.C,
                                latest_transport_ref=self.T)
        store.checkpoint("d1", self.b, note=self.review_note, expected_revision=1)
        self.snapshot = {"campaign_id": "campaign", "revision": 1, "state": {
            "binding": {"job_id": "job", "measurement_epoch_id": "epoch-1"},
            "opportunities": [{"opportunity_id": "op1", "direction_id": "d1", "settled": True}],
            "history": [
                {"attempt_id": "op1", "settled": True, "candidate_id": self.b,
                 "transport_ref": self.T, "collection_ref": self.C},
                {"review_kind": "research_review", "direction_revision": 2,
                 "proposal": {"direction_id": "d1", "transport_ref": self.T, "collection_ref": self.C,
                              "expected_revision": 1, "note": {"research_review_id": "r1"}}}]}}

    def collection(self, roles, harvest=None):
        harvest = harvest or {role: {"source_id": self.b_source} for role in roles}
        identity = self.store.archive_evidence(self.b, {
            "collection.json": encoded({"costs": {"wall_seconds": 3}}),
            "native_source_roles.json": encoded(roles),
            "native_idea_harvest.json": encoded(harvest)})
        return {"evidence_id": identity, "artifact": "collection.json"}

    def history(self):
        return self.snapshot["state"]["history"][0]

    def review(self):
        return self.snapshot["state"]["history"][1]


class EpisodeReviewTests(Fixture):
    def test_returns_the_committed_review_revision(self):
        review = succession.episode_review(self.store, self.snapshot, self.history(), "d1")
        self.assertEqual(review, {"revision": 2, "note": self.store.get_revision("d1", 2)["note"]})

    def test_mismatches_fail(self):
        cases = {"missing": lambda s: s["state"]["history"].pop(),
                 "revision": lambda s: s["state"]["history"][1].update(direction_revision=3),
                 "note": lambda s: s["state"]["history"][1]["proposal"]["note"].update(claim="other"),
                 "direction": lambda s: s["state"]["history"][1]["proposal"].update(direction_id="d2")}
        for name, change in cases.items():
            snapshot = copy.deepcopy(self.snapshot)
            change(snapshot)
            with self.subTest(name=name), self.assertRaises(ValueError):
                succession.episode_review(self.store, snapshot, snapshot["state"]["history"][0], "d1")

    def test_tip_must_be_the_reviewed_candidate(self):
        self.store.checkpoint("d1", self.a, note=self.review_note, expected_revision=2)
        self.review()["direction_revision"] = 3
        self.review()["proposal"]["expected_revision"] = 2
        with self.assertRaises(ValueError):
            succession.episode_review(self.store, self.snapshot, self.history(), "d1")


class ParentOutcomeTests(Fixture):
    def test_lists_every_role_with_costs_and_review(self):
        outcomes = succession.parent_outcomes(self.store, self.snapshot, self.b)
        self.assertEqual(len(outcomes), 1)
        outcome = outcomes[0]
        self.assertEqual(outcome["roles"], ["selected", "terminal"])
        self.assertEqual((outcome["direction_id"], outcome["attempt_id"]), ("d1", "op1"))
        self.assertEqual(outcome["collection_costs"], {"wall_seconds": 3})
        self.assertEqual(outcome["review"]["revision"], 2)
        self.assertEqual(succession.parent_outcomes(self.store, self.snapshot, self.a), [])
        unreviewed = succession.parent_outcomes(self.store, self.snapshot, self.b, require_review=False)
        self.assertIsNone(unreviewed[0]["review"])

    def test_unsettled_and_roleless_rows_are_skipped(self):
        self.history()["settled"] = False
        self.assertEqual(succession.parent_outcomes(self.store, self.snapshot, self.b), [])

    def test_wrong_binding_or_role_fails(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["state"]["binding"]["measurement_epoch_id"] = "epoch-2"
        with self.assertRaises(ValueError):
            succession.parent_outcomes(self.store, snapshot, self.b)
        self.history()["collection_ref"] = self.collection({"winner": self.b})
        with self.assertRaises(ValueError):
            succession.parent_outcomes(self.store, self.snapshot, self.b, require_review=False)
        self.history()["collection_ref"] = self.collection(
            self.roles, {"selected": {"source_id": "0" * 64}, "terminal": {"source_id": "0" * 64}})
        with self.assertRaises(ValueError):
            succession.parent_outcomes(self.store, self.snapshot, self.b, require_review=False)


class SelectedParentMaterialTests(Fixture):
    @unittest.skipIf(os.environ.get("EVO_PARITY_ORIGINAL"), "prospective retained-trial continuation")
    def test_failed_and_slower_trials_can_be_selected_only_after_their_episode_review(self):
        from geak_evo.engineer.investigation import retain_trial_sources
        from geak_evo.archive.refs import file_ref
        work = self.store.root.parent / "trial-work"
        work.mkdir()
        trials, expected = [], []
        for index, outcome in enumerate(("correctness failed", "slower")):
            code = ("v = " + repr(outcome) + "\n").encode()
            (work / "k.py").write_bytes(code)
            source_id = self.store.capture_source(work, ["k.py"])
            saved = self.store.root.parent / ("trial-" + str(index))
            self.store.materialize(source_id, saved)
            trials.append({"source_archive": {"source_id": source_id, "source_dir": str(saved),
                "manifest_ref": file_ref(self.store.root / "objects" / source_id)},
                "verified": {"source_id": source_id, "operation_id": "slow-trial",
                             "turn_id": "op1", "per_case_times_ms": {"case": 9.0}} if outcome == "slower" else None,
                "error": None if outcome == "slower" else outcome})
            feedback = self.store.root.parent / ("trial-feedback-" + str(index) + ".json")
            feedback.write_bytes(encoded({"source_id": source_id, "turn_id": "op1",
                "operation_id": "slow-trial" if outcome == "slower" else "failed-trial"}))
            trials[-1]["archive_ref"] = file_ref(feedback)
            expected.append(code)
        harvest = {role: {"source_id": self.b_source} for role in self.roles}
        harvest["trials"] = trials
        inventory = retain_trial_sources(self.store, harvest,
            {"attempt_id": "op1", "request": {"tip": {"candidate_id": self.a}, "donors": []}},
            {"job_id": "job", "measurement_epoch_id": "epoch-1"})
        evidence = self.store.archive_evidence(self.b, {
            "collection.json": encoded({"costs": {"wall_seconds": 3}}),
            "native_source_roles.json": encoded(self.roles),
            "native_idea_harvest.json": encoded(harvest),
            "native_trial_sources.json": encoded(inventory)})
        collection = {"evidence_id": evidence, "artifact": "collection.json"}
        self.history()["collection_ref"] = collection
        proposal = {"parent_candidate_id": inventory["trials"][0]["candidate_id"]}
        with self.assertRaisesRegex(ValueError, "exact committed scientific review"):
            succession.selected_parent_material(self.store, self.snapshot, proposal)
        self.review_note.update(reviewed_outcome_refs=[self.T, collection], research_update_ref=collection)
        self.store.checkpoint("d1", self.b, note=self.review_note, expected_revision=2)
        self.review()["direction_revision"] = 3
        self.review()["proposal"].update(collection_ref=collection, expected_revision=2)
        for index, row in enumerate(inventory["trials"]):
            (material,) = succession.selected_parent_material(self.store, self.snapshot,
                {"parent_candidate_id": row["candidate_id"], "direction_id": "continue-" + str(index),
                 "expected_revision": 0})
            self.assertEqual(material["source_id"], row["source_id"])
            self.assertEqual(material["reviewed_outcomes"][0]["trial_indexes"], [index])
            self.assertEqual(material["reviewed_outcomes"][0]["roles"], [])
            self.assertIn({"evidence_id": evidence, "artifact": "native_trial_sources.json"},
                          [item["evidence_ref"] for item in material["evidence"]])
            self.assertEqual(self.store.get_candidate(row["candidate_id"])["parents"], [self.a])
            restored = self.store.root.parent / ("continued-" + str(index))
            self.store.materialize(material["source_id"], restored)
            self.assertEqual((restored / "k.py").read_bytes(), expected[index])
        self.assertIsNone(self.store.get("d1")["anchor"])

    def test_invalid_donor_lists_fail(self):
        for donors in ([self.a, self.a], [self.b], [""], "x"):
            proposal = {"parent_candidate_id": self.b, "semantic_donors": donors}
            with self.subTest(donors=donors), self.assertRaises(ValueError):
                succession.selected_parent_material(self.store, self.snapshot, proposal)

    def test_unreviewed_parent_must_be_the_campaign_seed(self):
        with self.assertRaises(ValueError):
            succession.selected_parent_material(self.store, self.snapshot,
                                                {"parent_candidate_id": self.a, "semantic_donors": [self.b]})
        self.store.checkpoint("campaign", self.b, note=BASE_NOTE, expected_revision=1)
        with self.assertRaises(ValueError):
            succession.selected_parent_material(self.store, self.snapshot, {"parent_candidate_id": self.a})

    @unittest.skipUnless(PLANNER_CONTEXT, "planner context module not ported yet")
    def test_material_for_parent_and_donor(self):
        proposal = {"parent_candidate_id": self.b, "semantic_donors": []}
        (material,) = succession.selected_parent_material(self.store, self.snapshot, proposal)
        self.assertEqual(material["role"], "working_parent")
        self.assertEqual(material["source_id"], self.b_source)
        self.assertEqual(material["contribution_status"], "unassessed")
        harvest = {"evidence_id": self.C["evidence_id"], "artifact": "native_idea_harvest.json"}
        self.assertEqual([row["evidence_ref"] for row in material["evidence"]], [self.T, self.C, harvest])
        seed = succession.selected_parent_material(self.store, self.snapshot, {"parent_candidate_id": self.a})
        self.assertEqual(seed[0]["reviewed_outcomes"], [])


class LoopStub:
    def __init__(self, store, snapshot):
        self.store, self._snapshot, self.calls = store, snapshot, []

    def snapshot(self):
        return self._snapshot

    def record_progress_review(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class EpisodeTests(Fixture):
    def native(self):
        self.loop = LoopStub(self.store, self.snapshot)
        return episodes.NativeEvidence(self.loop, self.store.root, harvest=None, episode_material=None,
                                       scratch_root=None, runtime_refs={})

    def test_plain_review_needs_no_observation(self):
        native = self.native()
        self.assertEqual(native.review_assignment(self.snapshot, {"decision": {"action": "review"}}), {})
        for outcome in ("realize", "refute"):
            proposal = {"decision": {"action": "review", "outcome": outcome}}
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                native.review_assignment(self.snapshot, proposal)

    def test_progress_skips_unreviewed_and_already_recorded_turns(self):
        self.snapshot["state"]["history"].pop()
        self.assertIs(self.native().record_reviewed_progress(self.snapshot), self.snapshot)
        self.assertEqual(self.loop.calls, [])
        self.snapshot["state"]["opportunities"][0]["progress_review_refs"] = ["done"]
        self.snapshot["state"]["history"].append({"review_kind": "research_review",
                                                  "proposal": {"transport_ref": self.T}})
        self.assertIs(self.native().record_reviewed_progress(self.snapshot), self.snapshot)
        self.assertEqual(self.loop.calls, [])

    def test_episode_requires_committed_snapshot_and_exact_history(self):
        kwargs = dict(harvest=None, episode_material=None, scratch_root=None, runtime_refs={})
        with self.assertRaises((ValueError, KeyError)):  # The campaign note has no "evolution" state.
            episodes.committed_native_episode(self.store, self.store.root, self.snapshot, self.history(), **kwargs)
        note = dict(BASE_NOTE, evolution=self.snapshot["state"])
        self.store.checkpoint("campaign", self.a, note=note, expected_revision=1)
        self.snapshot["revision"] = 2
        changed = dict(self.history(), settled="yes")
        with self.assertRaises(ValueError):
            episodes.committed_native_episode(self.store, self.store.root, self.snapshot, changed, **kwargs)
        self.snapshot["state"]["opportunities"][0]["settled"] = False
        note = dict(BASE_NOTE, evolution=self.snapshot["state"])
        self.store.checkpoint("campaign", self.a, note=note, expected_revision=2)
        self.snapshot["revision"] = 3
        with self.assertRaises(ValueError):
            episodes.committed_native_episode(self.store, self.store.root, self.snapshot, self.history(), **kwargs)


if __name__ == "__main__":
    unittest.main()
