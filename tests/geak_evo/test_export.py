"""Synthetic CPU-only archive export tests; no model, device or live study reads."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from geak_evo.archive.export import export_programs, main, _inventory
from geak_evo.archive.claims import ClaimLog
from geak_evo.archive.records import ArchiveError, archive_checkpoint, archive_record
from geak_evo.archive.store import ResearchStore, StateError
from geak_evo.selection import recommendation


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.options = dict(allowed_epochs=[("job", "epoch")], source_paths={"job": ["main.py", "absent.py"]})
        self.store = ResearchStore(self.root / "store", **self.options)
        self.seed = self.candidate("seed")
        self.child = self.candidate("slower", [self.seed])
        self.failed = self.candidate("failed", [self.seed])
        self.note = dict(claim="Synthetic intent", scope="CPU fixture", next_test="Inspect the next observation")
        self.evo = dict(history=[], binding=dict(job_id="job", measurement_epoch_id="epoch"), deadline=200,
                        pending=dict(operation_id="operation-1"))
        self.store.checkpoint("campaign", self.seed, note=dict(self.note, evolution=self.evo), expected_revision=0)
        self.out = self.root / "export"

    def candidate(self, text, parents=()):
        (self.work / "main.py").write_text(text + "\n")
        (self.work / "main.py").chmod(0o755)
        source = self.store.capture_source(self.work, self.options["source_paths"]["job"])
        return self.store.candidate("job", "epoch", source, parents=parents)

    def evaluate(self, candidate, evaluation_id, *, settled=True, correct=True):
        value = dict(evaluation_id=evaluation_id,
            binding=dict(candidate_id=candidate, **self.store._binding(self.store.get_candidate(candidate))),
            settled=settled, correctness_observed=settled, correct=correct,
            eligible=settled and correct is True, times_ms={"case": 9.0} if correct else {},
            physical_devices=["synthetic-device"], operations=[])
        return self.store.archive_evidence(candidate, {"native_evaluation.json": encoded(value)})

    def record(self):
        evidence = [self.evaluate(self.child, "eval-1"), self.evaluate(self.child, "eval-2"),
                    self.evaluate(self.failed, "eval-failed", correct=False),
                    self.evaluate(self.failed, "eval-incomplete", settled=False, correct=None)]
        archive_checkpoint(self.store, "idea-direction", self.child, expected_revision=0, note=self.note,
            card=dict(idea=dict(idea_id="../idea", statement="Try less live state", status="unassessed"),
                      author=dict(engineer_id="engineer-1", attempt_ref=None)), evidence=evidence)

    def recommend(self, candidate=None):
        state = self.store.read_current("campaign")
        evo = copy.deepcopy(state["note"]["evolution"])
        context = dict(campaign_id="campaign", revision=state["revision"], state=copy.deepcopy(evo))
        clock_ref = dict(path="/historical/clock.json", sha256="1" * 64)
        intent = dict(campaign_id="campaign", binding=evo["binding"], operation_id="operation-1",
            admission=dict(clock_ref=clock_ref), started_monotonic_ns=1_000_000_000,
            deadline_monotonic_ns=10_000_000_000)
        directive = dict(op="set", choice_handle="chosen", reason="Synthetic selection") if candidate else dict(
            op="clear", reason="Withdraw the recommendation")
        record = recommendation.make_record(context=context, intent=intent, clock_ref=clock_ref,
            clock=dict(started_monotonic_ns=1_000_000_000, wall_seconds=100, started_unix=100),
            receipt_ref=None, response_raw=b"{}", proposal_raw=b"null", directive=directive,
            choices=[dict(choice_handle="chosen", candidate_id=candidate)], accepted_ns=2_000_000_000)
        evidence = self.store.archive_evidence(self.seed, {"recommendation-binding.json": encoded(record)})
        recommendation.commit_record(evo, record, planner_ref=dict(evidence_id=evidence), context=context,
                                     commit_revision=state["revision"] + 1, in_budget=True)
        self.store.checkpoint("campaign", self.seed, note=dict(self.note, evolution=evo),
                              expected_revision=state["revision"], evidence=[evidence])

    def test_nonempty_host_selection_timeline_and_all_measurements(self):
        self.record()
        self.recommend(self.child)
        self.recommend()
        export_programs(self.store, self.out)
        timeline = json.loads((self.out / "timeline.json").read_bytes())
        self.assertEqual(timeline["kind"], "host_selected")
        self.assertEqual([row["disposition"] for row in timeline["entries"]], ["set", "clear"])
        self.assertEqual([row["program_id"] for row in timeline["entries"]], [self.child, None])
        self.assertTrue(all(row["accepted_monotonic_ns"] == 2_000_000_000 for row in timeline["entries"]))
        measurements = json.loads((self.out / "measurements.json").read_bytes())
        self.assertEqual({row["evaluation_id"] for row in measurements}, {"eval-1", "eval-2", "eval-failed", "eval-incomplete"})
        self.assertEqual(len((self.out / "measurements.csv").read_text().splitlines()), 5)
        idea = json.loads(next((self.out / "ideas").iterdir()).read_bytes())
        self.assertEqual(idea["statement"], "Try less live state")
        self.assertEqual(idea["revisions"][0]["author"]["engineer_id"], "engineer-1")
        self.assertIn("not discovery or measurement time", next((self.out / "programs").glob("*/program.json")).read_text())

    def test_determinism_modes_tamper_detection_and_relocation(self):
        self.record()
        manifest = export_programs(self.store, self.out)
        before = _inventory(self.out)
        self.assertEqual(export_programs(self.store, self.out), manifest)
        self.assertEqual(_inventory(self.out), before)
        relocated = self.root / "relocated-store"
        shutil.copytree(self.store.root, relocated)
        other = self.root / "relocated-export"
        previous_umask = os.umask(0o077)
        try:
            export_programs(ResearchStore(relocated, **self.options), other)
        finally:
            os.umask(previous_umask)
        self.assertEqual(_inventory(other), before)
        for source_manifest in (other / "programs").glob("*/source.json"):
            program = json.loads((source_manifest.parent / "program.json").read_bytes())
            self.assertEqual(hashlib.sha256(source_manifest.read_bytes()).hexdigest(), program["source_id"])
            self.assertEqual(hashlib.sha256((source_manifest.parent / "candidate.json").read_bytes()).hexdigest(),
                             program["candidate_id"])
        for receipt in (other / "observations").glob("*/native_evaluation.json"):
            identity = receipt.parent.name
            _, original = self.store.evidence_artifact(identity, "native_evaluation.json")
            self.assertEqual(receipt.read_bytes(), original)
        source = next((self.out / "programs").glob("*/source/main.py"))
        self.assertEqual(manifest["files_executable"][source.relative_to(self.out).as_posix()], 0o111)
        source.chmod(0o644)
        with self.assertRaisesRegex(StateError, "existing export differs"):
            export_programs(self.store, self.out)
        source.chmod(0o755)
        source.write_text("tampered")
        with self.assertRaisesRegex(StateError, "existing export differs"):
            export_programs(self.store, self.out)
        self.assertIn("revalidate on a relocated host", (other / "RESUME.json").read_text())
        extra = other / "unexpected-empty"
        extra.mkdir()
        with self.assertRaisesRegex(StateError, "existing export differs"):
            export_programs(self.store, other)
        extra.rmdir()
        (other / "index.json").chmod(0o666)
        with self.assertRaisesRegex(StateError, "existing export differs"):
            export_programs(self.store, other)

    def test_same_checkpoint_mixed_outcomes_never_choose_a_hash_winner(self):
        evidence = [self.evaluate(self.child, "passing", correct=True),
                    self.evaluate(self.child, "failing", correct=False)]
        self.store.checkpoint("direction", self.child, note=self.note, expected_revision=0, evidence=evidence)
        export_programs(self.store, self.out)
        [row] = [json.loads(path.read_bytes()) for path in (self.out / "programs").glob("*/program.json")
                 if json.loads(path.read_bytes())["candidate_id"] == self.child]
        self.assertEqual(row["status"]["value"], "mixed")
        self.assertEqual(row["status"]["outcomes"], ["eligible", "failed"])
        self.assertEqual({item["evaluation_id"] for item in row["status"]["evaluations"]}, {"passing", "failing"})
        indexed = next(item for item in json.loads((self.out / "index.json").read_bytes()) if item["program_id"] == self.child)
        self.assertIsNone(indexed["times_ms"])
        self.assertEqual(indexed["kind"], "native_evaluation")

    def test_distinct_investigations_survive_and_only_linked_copies_merge(self):
        row = dict(trial_index=0, candidate_id=None, source_id=None, operation_id=None, feedback_ref=None,
                   feedback_error=None, source_retention_error="missing")
        inventory = encoded(dict(schema_version=1, trials=[row]))
        harvest = encoded(dict(trials=[dict(archive_ref=None, error="first outcome")]))
        raw = self.store.archive_evidence(self.child, {"native_trial_sources.json": inventory, "archive/manifest.json": harvest})
        first = self.store.archive_evidence(self.child, {"native_trial_sources.json": inventory,
            "native_idea_harvest.json": harvest, "native_evidence_index.json": encoded([
                dict(evidence_ref=dict(evidence_id=raw, artifact="archive/manifest.json"))])})
        second = self.store.archive_evidence(self.failed, {"native_trial_sources.json": inventory,
            "native_idea_harvest.json": encoded(dict(trials=[dict(archive_ref=None, error="second outcome")]))})
        self.store.checkpoint("direction", self.child, note=self.note, expected_revision=0, evidence=[raw, first, second])
        manifest = export_programs(self.store, self.out)
        trials = json.loads((self.out / "trials.json").read_bytes())
        self.assertEqual(manifest["source_retention_failures"], 2)
        self.assertEqual({trial["outcome"]["error"] for trial in trials}, {"first outcome", "second outcome"})
        self.assertEqual({identity for trial in trials for identity in trial["evidence_ids"]}, {raw, first, second})

    def test_diagnostic_cards_accept_explicit_host_case_registry(self):
        cases = {"job": ["case-a"]}
        diagnostic = archive_record(self.store, self.child, dict(kind="evo_diagnostic", case_ids=["case-a"],
            compilation_ref=None, artifacts={}, claim="synthetic diagnostic", unknown_reason="no compilation"), frozen_cases=cases)
        archive_checkpoint(self.store, "direction", self.child, expected_revision=0, note=self.note, frozen_cases=cases,
            card=dict(idea=dict(idea_id="diagnostic", statement="test idea", status="unassessed"), observation_refs=[diagnostic]))
        with self.assertRaisesRegex(ArchiveError, "frozen case registry"):
            export_programs(self.store, self.out)
        self.assertFalse(self.out.exists())
        export_programs(self.store, self.out, frozen_cases=cases)
        registry = self.root / "cases.json"
        registry.write_bytes(encoded(cases))
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--store", str(self.store.root), "--out", str(self.out), "--frozen-cases", str(registry)])

    def test_archived_pair_and_job_are_copied_without_new_ratios(self):
        pair = b'{"eligible": false, "comparison": {"status": "incomplete"}}\n'
        job = b'{"workload": {"device": "synthetic-device"}}\n'
        evidence = self.store.archive_evidence(self.child, {"paired_feedback.json": pair, "native/job.json": job})
        self.store.checkpoint("direction", self.child, note=self.note, expected_revision=0, evidence=[evidence])
        export_programs(self.store, self.out)
        folder = self.out / "observations" / evidence
        self.assertEqual((folder / "paired_feedback.json").read_bytes(), pair)
        self.assertEqual((folder / "native/job.json").read_bytes(), job)
        self.assertNotIn("descriptive_ratio", (self.out / "index.json").read_text())

    def test_trial_inventory_keeps_missing_and_conflicting_feedback_explicit(self):
        source = self.store.get_candidate(self.failed)["source_id"]
        feedback = dict(path="/historical/feedback.json", sha256="2" * 64)
        rows = [dict(trial_index=0, operation_id="trial-1", source_id=source, candidate_id=self.failed,
                     feedback_ref=feedback, source_retention_error=None, feedback_error="feedback/source conflict"),
                dict(trial_index=1, operation_id=None, source_id=None, candidate_id=None,
                     feedback_ref=None, source_retention_error="source missing", feedback_error=None)]
        context = dict(native_observations=[dict(source_id=source, operation_id="trial-1", feedback_ref=feedback,
                                                per_case_times_ms={"case": 0.001})])
        evidence = self.store.archive_evidence(self.child, {"native_trial_sources.json": encoded(dict(trials=rows)),
            "native_idea_context.json": encoded(context), "native_idea_harvest.json": encoded(dict(trials=[
                dict(archive_ref=feedback, error="correctness failed"), dict(archive_ref=None, error="interrupted")]))})
        self.store.checkpoint("direction", self.child, note=self.note, expected_revision=0, evidence=[evidence])
        manifest = export_programs(self.store, self.out)
        measurements = json.loads((self.out / "measurements.json").read_bytes())
        self.assertEqual(len(measurements), 2)
        self.assertTrue(all(row["times_ms"] is None for row in measurements))
        self.assertEqual([row["outcome"]["error"] for row in measurements], ["correctness failed", "interrupted"])
        self.assertEqual(manifest["source_retention_failures"], 1)
        self.assertEqual(manifest["programs"], 3)
        history = json.loads((self.out / "trial-history.json").read_bytes())
        self.assertEqual(history["coverage"]["trial_count"], 2)
        self.assertEqual({row["trial_id"] for row in history["trials"]},
                         {row["trial_id"] for row in measurements})
        self.assertEqual(history["coverage"]["outside_inventory"], "unknown")

    def test_corrupt_source_never_publishes(self):
        evidence = self.evaluate(self.child, "eval")
        self.store.checkpoint("direction", self.child, note=self.note, expected_revision=0, evidence=[evidence])
        blob = self.store.source(self.store.get_candidate(self.child)["source_id"])["files"][1]["blob_id"]
        path = self.store.root / "objects" / blob
        path.chmod(0o600)
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(StateError, "hash mismatch"):
            export_programs(self.store, self.out)
        self.assertFalse(self.out.exists())

    def test_wrong_job_receipt_and_unsafe_attempt_name(self):
        value = dict(evaluation_id="bad", binding=dict(candidate_id=self.child, job_id="wrong",
            measurement_epoch_id="epoch", candidate_source_id=self.store.get_candidate(self.child)["source_id"]))
        evidence = self.store.archive_evidence(self.child, {"native_evaluation.json": encoded(value)})
        self.store.checkpoint("direction", self.child, note=self.note, expected_revision=0, evidence=[evidence])
        with self.assertRaisesRegex(StateError, "binding differs"):
            export_programs(self.store, self.out)
        self.assertFalse(self.out.exists())

    def test_attempt_names_are_hashed_and_author_identity_is_retained(self):
        packet = dict(proposal=dict(parent_candidate_id=self.seed, direction_id="direction", island_id="island",
                                    engineer_id="engineer"), decision={})
        evidence = self.store.archive_evidence(self.child, {"packet.json": encoded(packet),
            "native_trial_sources.json": encoded(dict(trials=[dict(trial_index=0, candidate_id=self.failed,
                source_id=self.store.get_candidate(self.failed)["source_id"], operation_id="trial",
                feedback_ref=None, feedback_error=None, source_retention_error=None)]))})
        self.evo["history"] = [dict(attempt_id="../../escape", packet_ref=dict(evidence_id=evidence, artifact="packet.json"),
                                    collection_ref=dict(evidence_id=evidence, artifact="native_trial_sources.json"),
                                    candidate_id=self.child, settled=True)]
        self.store.checkpoint("campaign", self.seed, note=dict(self.note, evolution=self.evo), expected_revision=1)
        archive_checkpoint(self.store, "direction", self.child, expected_revision=0, note=self.note,
            card=dict(idea=dict(idea_id="idea-1", statement="Develop slower idea", status="unassessed")))
        export_programs(self.store, self.out)
        [attempt] = (self.out / "attempts").iterdir()
        self.assertEqual(len(attempt.stem), 64)
        self.assertEqual(json.loads(attempt.read_bytes())["attempt_id"], "../../escape")
        programs = [json.loads(path.read_bytes()) for path in (self.out / "programs").glob("*/program.json")]
        child = next(row for row in programs if row["candidate_id"] == self.child)
        self.assertEqual(child["authors"][0]["engineer_id"], "engineer")
        intermediate = next(row for row in programs if row["candidate_id"] == self.failed)
        self.assertEqual(intermediate["idea_ids"], [])
        self.assertEqual(intermediate["directions"], ["direction"])
        missing = json.loads(attempt.read_bytes())["implemented_idea"]
        self.assertIsNone(missing["strategy"])
        self.assertIsNone(missing["reported_source_id"])
        self.assertIn("No saved strategy", missing["missing_reason"])

    def test_completed_research_is_readable_with_exact_trial_and_claim_scope(self):
        source = self.store.get_candidate(self.child)["source_id"]
        packet = dict(proposal=dict(parent_candidate_id=self.seed, direction_id="direction", island_id="island",
            engineer_id="engineer", question="Could reducing live state help?"), decision={})
        strategy = "Reuse the loaded tile; the expected traffic change still needs counters.\n" + "µ" * 4200
        harvest = dict(selected=dict(source_id=source), trials=[dict(archive_ref=None, error="failed trial")],
            explanation=dict(strategy=strategy, source_id=source, binding_valid=True, investigation_complete=True,
                             explanation_status="engineer_report_bound", errors=[]))
        evidence = self.store.archive_evidence(self.child, {
            "packet.json": encoded(packet), "native_idea_harvest.json": encoded(harvest),
            "native_source_roles.json": encoded(dict(selected=self.child, terminal=self.child, faithful=self.seed)),
            "report.md": b"BULK NARRATION MUST NOT BE COPIED",
            "native_trial_sources.json": encoded(dict(trials=[dict(trial_index=0, candidate_id=self.failed,
                source_id=self.store.get_candidate(self.failed)["source_id"], operation_id="trial", feedback_ref=None,
                feedback_error=None, source_retention_error=None)]))})
        collection = dict(evidence_id=evidence, artifact="native_idea_harvest.json")
        self.evo["history"] = [dict(attempt_id="engineer-turn", packet_ref=dict(evidence_id=evidence, artifact="packet.json"),
                                    collection_ref=collection, candidate_id=self.child, settled=True)]
        binding = dict(job_id="job", case_id="case", source_id=source, work_id="work",
                       measurement_epoch_id="epoch", hardware_id="CPU fixture", scope="complete_callable")
        claim = dict(kind="scientific_claim", statement="Less traffic might explain the change.", binding=binding,
            evidence=[], depends_on=[], assumptions=["Actual traffic is unknown."], applicability="Fixture only",
            next_test="Collect source-bound traffic counters.")
        log = ClaimLog(self.store)
        original = log.append(self.child, claim, binding)
        log.correct(self.child, [original], scope=binding, reason="Timing alone does not establish traffic.",
                    consequence="Keep the mechanism unconfirmed.")
        self.evo["claim_head"] = log.head
        self.store.checkpoint("campaign", self.seed, note=dict(self.note, evolution=self.evo), expected_revision=1)
        # A later immutable object without a committed journal reference must stay out.
        log.append(self.child, dict(claim, statement="UNCOMMITTED CLAIM"), binding)
        archive_checkpoint(self.store, "direction", self.child, expected_revision=0,
            note=dict(self.note, research_status="reported", next_test="Measure traffic next.",
                      evidence_uptake="Timing was insufficient.", research_update_ref=collection),
            card=dict(idea=dict(idea_id="idea-1", statement="Could reducing live state help?", status="unassessed")))
        export_programs(self.store, self.out)
        programs = [json.loads(path.read_bytes()) for path in (self.out / "programs").glob("*/program.json")]
        child = next(row for row in programs if row["candidate_id"] == self.child)
        trial = next(row for row in programs if row["candidate_id"] == self.failed)
        faithful = next(row for row in programs if row["candidate_id"] == self.seed)
        attempt = json.loads((self.out / child["research_files"]["attempts"][0]).read_bytes())
        self.assertEqual(attempt["question"], packet["proposal"]["question"])
        self.assertEqual(attempt["implemented_idea"]["strategy"], strategy[:4096])
        self.assertEqual(attempt["implemented_idea"]["reported_source_id"], source)
        self.assertTrue(attempt["implemented_idea"]["strategy_truncated"])
        retained_strategy = (self.out / attempt["implemented_idea"]["strategy_file"]).read_bytes()
        self.assertEqual(retained_strategy, strategy.encode("utf-8"))
        self.assertEqual(attempt["implemented_idea"]["strategy_sha256"], hashlib.sha256(retained_strategy).hexdigest())
        self.assertIn("not a description of every trial", attempt["implemented_idea"]["basis"])
        self.assertEqual(trial["research_files"]["attempts"], child["research_files"]["attempts"])
        self.assertEqual(faithful["research_files"]["attempts"], child["research_files"]["attempts"])
        self.assertEqual(faithful["authors"], [])
        self.assertEqual(trial["research_files"]["claims"], [])
        direction = json.loads((self.out / trial["research_files"]["directions"][0]).read_bytes())
        self.assertEqual(direction["program_id"], self.child)
        self.assertEqual(direction["note"]["next_test"], "Measure traffic next.")
        idea = json.loads(next((self.out / "ideas").iterdir()).read_bytes())
        self.assertEqual(idea["statement"], packet["proposal"]["question"])
        self.assertEqual(idea["investigation_files"], child["research_files"]["attempts"])
        claims = json.loads((self.out / "claims.json").read_bytes())
        self.assertEqual([event["kind"] for event in claims["events"]], ["claim", "claim_correction"])
        self.assertEqual(len(child["research_files"]["claims"]), 2)
        for event in claims["events"]:
            _, raw = self.store.evidence_artifact(event["ref"]["evidence_id"], event["ref"]["artifact"])
            self.assertEqual((self.out / event["file"]).read_bytes(), raw)
        exported_text = b"\n".join(path.read_bytes() for path in self.out.rglob("*") if path.is_file())
        self.assertNotIn(b"UNCOMMITTED CLAIM", exported_text)
        self.assertNotIn(b"BULK NARRATION", exported_text)
        before = _inventory(self.out)
        export_programs(self.store, self.out)
        self.assertEqual(_inventory(self.out), before)

    def test_cli_and_symlink_refusal(self):
        self.record()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(["--store", str(self.store.root), "--out", str(self.out)])
        self.assertEqual(json.loads(output.getvalue())["programs"], 3)
        linked = self.root / "linked"
        linked.symlink_to(self.out, target_is_directory=True)
        with self.assertRaisesRegex(StateError, "symlinks"):
            export_programs(self.store, linked)


if __name__ == "__main__":
    unittest.main()
