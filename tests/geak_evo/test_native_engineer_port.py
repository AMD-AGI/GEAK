"""Synthetic native entry/collection fixtures, shared by Oct 15 and the port."""

import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

import test_engineer_port as base

ORIGINAL = os.environ.get("EVO_ENGINEER_PARITY") == "original"
if ORIGINAL:
    import native_investigation as investigation
    import native_assignment as assignment
    import operation_files as operations
else:
    from geak_evo.engineer import investigation, assignment, operations


class NativeEngineerTests(unittest.TestCase):
    capture = base.EngineerTests.capture

    def setUp(self):
        base.EngineerTests.setUp(self)
        self.now = 100.0
        self.clock = SimpleNamespace(monotonic=lambda: self.now,
                                     monotonic_ns=lambda: int(self.now * 1e9),
                                     boot_id=lambda: Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        self.calls = []
        self.settled = True
        self.entry_returncode = 0
        self.operation_id = "fixture-operation-1"
        self.prompt = self.root / "native-prompt.txt"
        self.prompt.write_text("Inspect the exact assigned source and retain all trials.\n")
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.entry = self.runtime / "entry.mjs"
        self.collector = self.runtime / "collector.py"
        self.driver = self.runtime / "driver.py"
        for path in (self.entry, self.collector, self.driver):
            path.write_text("CPU fixture bytes; never executed.\n")
        self.clock_ref = self.save("clock.json", {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "started_monotonic_ns": 0, "started_unix": 0, "wall_seconds": 300,
            "session_id": "cpu-session"})

    def save(self, name, value):
        path = self.runtime / name
        path.write_bytes(investigation.encoded(value))
        return base.file_ref(path)

    def prepare(self, request, prompt_ref, operation_id):
        internal = {"mode": "fresh", "prior_receipt_ref": None}
        shared = {"turn_id": operation_id, "direction_id": request["direction_id"],
                  "workspace": request["workspace"], "job_id": self.job_id,
                  "measurement_epoch_id": self.epoch,
                  "starting_source_id": request["tip"]["source_id"]}
        entry = {"protocol": "geak-one-engineer-v1", "turn_id": operation_id,
                 "idea_id": request["direction_id"], "workspace": request["workspace"],
                 "starting_source_id": request["tip"]["source_id"],
                 "continuation": internal, "direction": {"prompt": request["next_question"]},
                 "insights": [prompt_ref], "timeout_ms": 1000}
        if request["donors"]:
            parents = [{"candidate_id": row["candidate_id"]} for row in
                       [request["tip"], *request["donors"]]]
            entry["parents"] = shared["parents"] = parents
        entry_ref = self.save("entry-input.json", entry)
        config = {"store_root": str(self.store.root), "entry_input_ref": entry_ref,
                  "backend_config": {"direction_id": request["direction_id"],
                    "continuation": internal, "clock_ref": self.clock_ref,
                    "model_config": {"read_roots": [str(self.root)]},
                    "probe_config": dict(shared, turn_deadline_ns=102_000_000_000,
                                         budget={"deadline_ns": 300_000_000_000})}}
        config_ref = self.save("config.json", config)
        return {"entry_runner_ref": base.file_ref(self.entry), "entry_input_ref": entry_ref,
                "backend_binding_ref": self.save("binding.json", {
                    "config_ref": config_ref, "driver_ref": base.file_ref(self.driver)}),
                "collection_config_ref": self.save("collection.json", shared),
                "collector_ref": base.file_ref(self.collector),
                "entry_provenance_ref": self.save("provenance.json", {"fixture": True})}

    def process(self, command, **kwargs):
        self.calls.append((command, kwargs))
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        destination = Path(command[command.index("--output") + 1])
        if "--input" in command:
            self.now += 0.25
            if self.entry_returncode:
                return SimpleNamespace(returncode=self.entry_returncode)
            (self.workspace / self.paths[0]).write_text("answer = 2\n")
            self.terminal_source = self.capture(str(self.workspace))
            native = {"schema": "geak-native-idea-model-receipt-v1",
                "invocation_id": self.operation_id, "direction_id": self.request["direction_id"],
                "workspace": str(self.workspace), "job_id": self.job_id,
                "measurement_epoch_id": self.epoch,
                "initial_source_id": self.request["tip"]["source_id"],
                "settlement": {"settled": self.settled, "scope": "CPU fixture"},
                "terminal_snapshot": {"source_id": self.terminal_source},
                "observed": {"identity_valid": True, "terminal": "turn.completed",
                             "protocol_errors": []}, "thread_id": base.THREAD,
                "status": "completed" if self.settled else "unsettled", "returncode": 0,
                "native_results": [{"request_id": "evaluation-a"}, {"request_id": "evaluation-b"}],
                "usage": {"input_tokens": 20, "output_tokens": 8},
                "monetary_cost": None, "monetary_cost_status": "unknown"}
            native_ref = self.save("model-receipt.json", native)
            (self.runtime / "events.jsonl").write_bytes(b'{"type":"fixture.native"}\n')
            destination.write_bytes(investigation.encoded({"transport_result": {"receipt_ref": native_ref}}))
        else:
            self.now += 0.5
            archive = destination / "archive"
            archive.mkdir(parents=True)
            manifest = {"binding": {"job_id": self.job_id, "measurement_epoch_id": self.epoch},
                "terminal": {"source_id": self.terminal_source, "source_dir": str(self.workspace),
                             "manifest_ref": base.file_ref(self.store.root / "objects" / self.terminal_source)},
                "trials": getattr(self, "trials", []), "selected": None,
                "faithful": getattr(self, "faithful", None), "unfinished": None,
                "explanation": {"explanation_status": "unavailable", "errors": ["fixture"]}}
            (archive / "manifest.json").write_bytes(investigation.encoded(manifest))
            (archive / "report.md").write_text("A changed source without a confirmed improvement.\n")
        return SimpleNamespace(returncode=0)

    def native(self):
        if ORIGINAL:
            for patcher in (mock.patch.object(investigation, "time", self.clock),
                            mock.patch.object(investigation, "subprocess", SimpleNamespace(
                                run=self.process, DEVNULL=subprocess.DEVNULL))):
                patcher.start()
                self.addCleanup(patcher.stop)
            return investigation.NativeInvestigation(self.store, self.prepare)
        return investigation.NativeInvestigation(self.store, self.prepare,
            clock=self.clock, process_runner=self.process, node_executable="fixture-node",
            python_executable="fixture-python")

    def run_native(self, native):
        return native.run(self.request, self.attempts,
                          prompt_path=self.prompt, operation_id=self.operation_id)

    def test_native_edit_collect_retains_raw_cost_source_and_pending_review(self):
        native = self.native()
        result = self.run_native(native)
        candidate = self.store.candidate(self.job_id, self.epoch, self.terminal_source,
                                        parents=[self.request["tip"]["candidate_id"]])
        collected = native.collect(result, candidate)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(collected["costs"], {"wall_seconds": 0.75, "evaluations": 2})
        self.assertEqual(result["receipt"]["captured_tip_source_id"], self.terminal_source)
        self.assertTrue(result["receipt"]["transport_complete"])
        self.assertNotIn("research_update", collected)
        self.assertEqual(collected["card"]["idea"]["status"], "no_confirmed_best" if ORIGINAL else "unassessed")
        if not ORIGINAL:
            self.assertEqual(collected["card"]["idea"]["statement"], self.request["next_question"])
        raw = json.loads(collected["artifacts"]["native_receipt.json"])
        self.assertIsNone(raw["monetary_cost"])
        self.assertEqual(raw["usage"], {"input_tokens": 20, "output_tokens": 8})
        self.assertEqual(json.loads(collected["artifacts"]["native_source_roles.json"]), {"terminal": candidate})
        index = json.loads(collected["artifacts"]["native_evidence_index.json"])
        self.assertEqual({row["evidence_ref"]["artifact"] for row in index},
                         {"archive/manifest.json", "archive/report.md"})
        self.assertEqual((Path(result["attempt_dir"]) / "native_events.jsonl").read_bytes(),
                         b'{"type":"fixture.native"}\n')
        with self.assertRaises(FileExistsError):
            self.run_native(native)
        self.assertEqual(len(self.calls), 2)

    def test_unsettled_native_keeps_unknown_evaluations_and_does_not_collect(self):
        self.settled = False
        native = self.native()
        result = self.run_native(native)
        collected = native.collect(result, self.request["tip"]["candidate_id"])
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(result["receipt"]["captured_tip_source_id"])
        self.assertFalse(result["receipt"]["transport_complete"])
        self.assertIsNone(collected["costs"]["evaluations"])
        self.assertIsNone(collected["card"])
        with self.assertRaises(FileExistsError):
            self.run_native(native)

    def test_faithful_source_retains_selected_parent_and_semantic_donor_ancestry(self):
        donor_dir = self.root / "donor-source"
        donor_dir.mkdir()
        (donor_dir / self.paths[0]).write_text("answer = 41\n")
        donor_source = self.capture(str(donor_dir))
        donor = self.store.candidate(self.job_id, self.epoch, donor_source)
        self.request["donors"] = [base.source_reference(self.store, donor)]
        faithful_dir = self.root / "faithful-source"
        faithful_dir.mkdir()
        (faithful_dir / self.paths[0]).write_text("answer = 42\n")
        faithful_source = self.capture(str(faithful_dir))
        self.faithful = {"source_id": faithful_source, "source_dir": str(faithful_dir),
                         "manifest_ref": base.file_ref(self.store.root / "objects" / faithful_source)}
        native = self.native()
        result = self.run_native(native)
        candidate = self.store.candidate(self.job_id, self.epoch, self.terminal_source,
            parents=[self.request["tip"]["candidate_id"]], donors=[donor])
        collected = native.collect(result, candidate)
        roles = json.loads(collected["artifacts"]["native_source_roles.json"])
        faithful = self.store.get_candidate(roles["faithful"])
        self.assertEqual(faithful["parents"], [self.request["tip"]["candidate_id"]])
        self.assertEqual(faithful["donors"], [donor])
        self.assertEqual(faithful["source_id"], faithful_source)
        self.assertEqual(collected["card"]["last_faithful"]["candidate_id"], roles["faithful"])

    def test_failed_entry_retains_preparation_and_refuses_repeat(self):
        self.entry_returncode = 1
        native = self.native()
        with self.assertRaisesRegex(ValueError, "never retry"):
            self.run_native(native)
        attempt = self.attempts / self.operation_id
        self.assertTrue((attempt / "native_prepared.json").is_file())
        self.assertFalse((attempt / "receipt.json").exists())
        with self.assertRaises(FileExistsError):
            self.run_native(native)
        self.assertEqual(len(self.calls), 1)

    def test_prepared_identity_rejects_changed_operation_before_process(self):
        actual_prepare = self.prepare
        def wrong_operation(request, prompt_ref, operation_id):
            return actual_prepare(request, prompt_ref, "wrong-operation")
        native = self.native()
        native.prepare_native = wrong_operation
        with self.assertRaisesRegex(ValueError, "reserved operation"):
            self.run_native(native)
        self.assertEqual(self.calls, [])

    def trial_source(self, name, *, verdict=None, error=None):
        source_dir = self.root / name
        source_dir.mkdir()
        (source_dir / self.paths[0]).write_text("answer = " + repr(name) + "\n")
        source_id = self.capture(str(source_dir))
        snapshot = {"source_id": source_id, "source_dir": str(source_dir),
                    "manifest_ref": base.file_ref(self.store.root / "objects" / source_id)}
        feedback = {"source_id": source_id, "operation_id": name, "status": verdict,
                    "turn_id": self.operation_id}
        ref = self.save(name + ".feedback.json", feedback)
        return {"source_archive": snapshot, "source_archive_error": None,
                "feedback_ref": ref, "archive_ref": ref, "artifacts": [],
                "verified": dict(feedback) if verdict == "slower" else None,
                "diagnostic": None, "error": error}

    @unittest.skipIf(ORIGINAL, "prospective per-trial candidate inventory")
    def test_intermediate_failed_slower_and_incomplete_sources_remain_recoverable(self):
        donor = self.trial_source("donor")
        donor_id = self.store.candidate(self.job_id, self.epoch, donor["source_archive"]["source_id"])
        self.request["donors"] = [base.source_reference(self.store, donor_id)]
        self.trials = [self.trial_source("slower", verdict="slower"),
                       self.trial_source("failed", error="correctness failed"),
                       self.trial_source("incomplete", error="evaluation interrupted")]
        # The verified fixture still carries the existing observation interface.
        self.trials[0]["verified"].update(per_case_times_ms={"case": 9.0},
            measurement_valid=True, custody_proven=True, reward_eligible=False)
        native = self.native()
        result = self.run_native(native)
        terminal = self.store.candidate(self.job_id, self.epoch, self.terminal_source,
            parents=[self.request["tip"]["candidate_id"]], donors=[donor_id])
        collected = native.collect(result, terminal)
        inventory = json.loads(collected["artifacts"]["native_trial_sources.json"])
        self.assertEqual(len(inventory["trials"]), 3)
        for trial, row in zip(self.trials, inventory["trials"]):
            self.assertIsNone(row["source_retention_error"])
            self.assertIsNone(row["feedback_error"])
            self.assertEqual(row["source_id"], trial["source_archive"]["source_id"])
            self.assertNotEqual(row["candidate_id"], terminal)
            candidate = self.store.get_candidate(row["candidate_id"])
            self.assertEqual(candidate["parents"], [self.request["tip"]["candidate_id"]])
            self.assertEqual(candidate["donors"], [donor_id])
            destination = self.root / (row["operation_id"] + "-restored")
            self.store.materialize(candidate["source_id"], destination)
            self.assertEqual((destination / self.paths[0]).read_bytes(),
                             (Path(trial["source_archive"]["source_dir"]) / self.paths[0]).read_bytes())
        self.assertEqual(json.loads(collected["artifacts"]["native_source_roles.json"]), {"terminal": terminal})
        raw = collected["card"]["observation_refs"][0]
        _, artifacts = self.store.evidence(raw["evidence_id"])
        self.assertEqual(json.loads(artifacts["native_trial_sources.json"]), inventory)
        self.assertEqual(json.loads(artifacts["archive/manifest.json"])["trials"], self.trials)

    @unittest.skipIf(ORIGINAL, "prospective per-trial candidate inventory")
    def test_missing_corrupt_and_conflicting_trial_sources_are_explicit(self):
        good = self.trial_source("good", verdict="slower")
        corrupt = self.trial_source("corrupt")
        (Path(corrupt["source_archive"]["source_dir"]) / self.paths[0]).write_text("changed\n")
        conflicting = self.trial_source("conflicting", verdict="slower")
        conflicting["verified"]["source_id"] = good["source_archive"]["source_id"]
        missing = {"verified": None, "error": "feedback unavailable", "source_archive_error": "source missing"}
        receipt = {"request": self.request, "attempt_id": self.operation_id}
        inventory = investigation.retain_trial_sources(self.store,
            {"trials": [good, missing, corrupt, conflicting]}, receipt,
            {"job_id": self.job_id, "measurement_epoch_id": self.epoch})
        self.assertIsNotNone(inventory["trials"][0]["candidate_id"])
        errors = [row["source_retention_error"] for row in inventory["trials"][1:3]]
        self.assertIn("no authenticated trial source", errors[0])
        self.assertIn("retained trial source changed", errors[1])
        self.assertTrue(all(row["candidate_id"] is None for row in inventory["trials"][1:3]))
        # Preserve the authenticated source, but withhold the conflicting
        # operation association. A source record is not a valid measurement.
        conflict = inventory["trials"][3]
        self.assertIsNotNone(conflict["candidate_id"])
        self.assertIsNone(conflict["operation_id"])
        self.assertIn("source archive differ", conflict["feedback_error"])

    @unittest.skipIf(ORIGINAL, "prospective per-trial candidate inventory")
    def test_checked_legacy_source_is_retained_without_snapshot_but_raw_feedback_is_not(self):
        checked = self.trial_source("checked", verdict="slower")
        checked.pop("source_archive")
        unbound = self.trial_source("unbound")
        unbound.pop("source_archive")
        result = investigation.retain_trial_sources(self.store, {"trials": [checked, unbound]},
            {"request": self.request, "attempt_id": self.operation_id},
            {"job_id": self.job_id, "measurement_epoch_id": self.epoch})
        self.assertIsNotNone(result["trials"][0]["candidate_id"])
        self.assertIsNone(result["trials"][1]["candidate_id"])

    @unittest.skipIf(ORIGINAL, "prospective per-trial candidate inventory")
    def test_corrupt_archive_blob_is_fatal_not_a_missing_snapshot(self):
        from geak_evo.archive.store import StateError
        trial = self.trial_source("corrupt-archive", verdict="slower")
        trial.pop("source_archive")
        source = self.store.source(trial["verified"]["source_id"])
        blob = self.store.root / "objects" / source["files"][0]["blob_id"]
        blob.chmod(0o600)
        blob.write_bytes(b"corrupt")
        with self.assertRaisesRegex(StateError, "hash mismatch"):
            investigation.retain_trial_sources(self.store, {"trials": [trial]},
                {"request": self.request, "attempt_id": self.operation_id},
                {"job_id": self.job_id, "measurement_epoch_id": self.epoch})

    @unittest.skipIf(ORIGINAL, "prospective owning-turn source lineage")
    def test_continuing_harvest_never_reparents_prior_or_unknown_trials(self):
        prior = self.trial_source("prior", verdict="slower")
        unknown = self.trial_source("unknown")
        unknown["archive_ref"] = None
        native = {"job_id": self.job_id, "measurement_epoch_id": self.epoch}
        receipt = {"request": self.request, "attempt_id": self.operation_id}
        first = investigation.retain_trial_sources(self.store, {"trials": [prior]}, receipt, native)
        original = self.store.get_candidate(first["trials"][0]["candidate_id"])
        following = copy.deepcopy(receipt)
        following["attempt_id"] = "next-episode"
        following["request"]["tip"]["candidate_id"] = first["trials"][0]["candidate_id"]
        continued = investigation.retain_trial_sources(self.store, {"trials": [prior, unknown]}, following, native)
        self.assertTrue(all(row["candidate_id"] is None for row in continued["trials"]))
        self.assertTrue(all("owning turn" in row["source_retention_error"] for row in continued["trials"]))
        self.assertEqual(self.store.get_candidate(first["trials"][0]["candidate_id"]), original)

    @unittest.skipIf(ORIGINAL, "prospective separation of idea and implementation")
    def test_idea_question_survives_selected_strategy_and_failed_implementation(self):
        for selected, strategy in ((None, "unfinished experiment"),
                                   ({"source_id": "selected"}, "an implemented fallback")):
            manifest = {"selected": selected, "explanation": {"strategy": strategy}}
            idea = investigation.harvested_idea(manifest, self.request)
            self.assertEqual(idea, {"idea_id": self.request["direction_id"],
                "statement": self.request["next_question"], "status": "unassessed"})
            self.assertEqual(manifest["explanation"]["strategy"], strategy)

    def test_reservation_preflight_never_recredits_or_mutates_snapshot(self):
        snapshot = {"state": {}, "accounting": {"available": {"wall_seconds": 2, "evaluations": 1}}}
        original = copy.deepcopy(snapshot)
        assignment.reservation_preflight(None, snapshot, {"reservation": {"wall_seconds": 2, "evaluations": 1}})
        with self.assertRaisesRegex(ValueError, "remaining campaign"):
            assignment.reservation_preflight(None, snapshot, {"reservation": {"wall_seconds": 3, "evaluations": 1}})
        self.assertEqual(snapshot, original)

    def test_new_opportunity_never_reuses_previous_episode_directory(self):
        first = operations.episode_output(self.root, "idea", "prior", opportunity="one")
        second = operations.episode_output(self.root, "idea", "prior", opportunity="two")
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, self.root / "episodes/idea/prior")
        with self.assertRaises(ValueError):
            operations.episode_output(self.root, "idea", "prior", opportunity="../escape")


if __name__ == "__main__":
    unittest.main()
