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

    def collection_fixture(self):
        native = self.native()
        result = self.run_native(native)
        candidate = self.store.candidate(self.job_id, self.epoch, self.terminal_source,
                                        parents=[self.request["tip"]["candidate_id"]])
        return native, result, candidate

    def committed_episode_fixture(self, result, candidate, collected):
        """Commit real store records; only the external material reader is a CPU fixture."""
        from geak_evo.archive import episodes
        attempt = Path(result["attempt_dir"])
        prepared = investigation.read(base.file_ref(attempt / "native_prepared.json"))
        native_receipt = investigation.read(result["receipt"]["native_receipt_ref"])
        opportunity = {"job_id": self.job_id, "measurement_epoch_id": self.epoch,
            "opportunity_id": self.operation_id, "direction_id": self.request["direction_id"],
            "island_id": "island", "engineer_id": "engineer", "question": self.request["next_question"],
            "opened_at": 100, "review_at": 105, "requested": {"wall_seconds": 5, "evaluations": 2},
            "reserved": {"wall_seconds": 5, "evaluations": 2}, "membership_ref": None,
            "receipts": [], "settled": True, "completed_sequence": 1, "progress_complete": False}
        proposal = {key: opportunity[key] for key in episodes._PROPOSAL_FIELDS}
        proposal.update(parent_candidate_id=self.request["tip"]["candidate_id"],
                        reservation=opportunity["reserved"])
        packet = {"proposal": proposal, "opportunity": dict(opportunity,
            receipts=[], settled=False, completed_sequence=None, progress_complete=False)}
        packet_id = self.store.archive_evidence(self.request["tip"]["candidate_id"], {
            "assignment.json": investigation.encoded(packet), "engineer_prompt.txt": self.prompt.read_bytes()})
        transport = {name: (attempt / name).read_bytes() for name in
                     ("native_prepared.json", "native_entry_result.json", "native_receipt.json", "prompt.txt")}
        transport["transport.json"] = investigation.encoded(result["receipt"])
        transport_id = self.store.archive_evidence(candidate, transport)
        artifacts = dict(collected["artifacts"], **{"collection.json": investigation.encoded({"costs": collected["costs"]})})
        collection_id = self.store.archive_evidence(candidate, artifacts)
        history = {"attempt_id": self.operation_id, "candidate_id": candidate, "settled": True,
            "packet_ref": {"evidence_id": packet_id, "artifact": "assignment.json"},
            "transport_ref": {"evidence_id": transport_id, "artifact": "transport.json"},
            "collection_ref": {"evidence_id": collection_id, "artifact": "collection.json"}}
        note = {"claim": "CPU fixture", "scope": "source retention", "next_test": "continue exact source",
            "research_status": "reported", "research_review_id": "fixture-review",
            "reviewed_outcome_refs": [history["transport_ref"], history["collection_ref"]],
            "research_update_ref": history["collection_ref"], "latest_transport_ref": history["transport_ref"]}
        direction = self.request["direction_id"]
        self.store.checkpoint(direction, candidate, note=note, expected_revision=0)
        review = {"review_kind": "research_review", "direction_revision": 1, "proposal": {
            "direction_id": direction, "transport_ref": history["transport_ref"],
            "collection_ref": history["collection_ref"], "expected_revision": 0,
            "note": {"research_review_id": "fixture-review"}}}
        state = {"binding": {"job_id": self.job_id, "measurement_epoch_id": self.epoch},
                 "history": [history, review], "opportunities": [opportunity]}
        self.store.checkpoint("campaign", candidate, note=dict(note, evolution=state), expected_revision=0)
        snapshot = {"campaign_id": "campaign", "revision": 1, "state": state}
        def material_reader(receipt_ref, manifest_ref, expected, **kwargs):
            self.assertEqual(receipt_ref, result["receipt_ref"])
            self.assertEqual(expected["turn_id"], self.operation_id)
            self.assertEqual(expected["starting_source_id"], self.request["tip"]["source_id"])
            manifest = investigation.read(manifest_ref)
            return result["receipt"], {"harvest": manifest, "native_receipt": native_receipt,
                "refs": {"native_entry_binding": investigation.read(base.file_ref(attempt / "native_entry_binding.json"))}}
        kwargs = dict(harvest=None, episode_material=material_reader, scratch_root=self.root,
            runtime_refs={key: prepared[key] for key in ("entry_runner_ref", "collector_ref", "entry_provenance_ref")})
        return snapshot, history, kwargs

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_recovered_collection_passes_committed_episode_and_parent_continuation(self):
        from geak_evo.archive import episodes, succession
        native, result, candidate = self.collection_fixture()
        def interrupted(command, **kwargs):
            kwargs["stderr"].write(b"interrupted\n")
            raise KeyboardInterrupt
        native.process_runner = interrupted
        with self.assertRaises(KeyboardInterrupt):
            native.collect(result, candidate)
        native.process_runner = self.process
        collected = native.collect(result, candidate)
        snapshot, history, kwargs = self.committed_episode_fixture(result, candidate, collected)
        outer, material = episodes.committed_native_episode(self.store, self.root, snapshot, history, **kwargs)
        self.assertEqual(outer, result["receipt"])
        self.assertEqual(material["harvest"]["terminal"]["source_id"], self.terminal_source)
        # A later bad/stale original output is irrelevant to the successful retry.
        old = Path(result["attempt_dir"]) / "collection/archive"
        old.mkdir(parents=True)
        (old / "manifest.json").write_bytes(b"corrupt abandoned collection")
        self.assertEqual(episodes.committed_native_episode(self.store, self.root, snapshot, history, **kwargs),
                         (outer, material))
        parents = succession.selected_parent_material(self.store, snapshot, {
            "parent_candidate_id": candidate, "semantic_donors": []})
        self.assertEqual(parents[0]["source_id"], self.terminal_source)
        self.assertEqual(parents[0]["candidate_id"], candidate)
        self.assertEqual(sum("--input" in call[0] for call in self.calls), 1)

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_committed_legacy_collection_remains_readable(self):
        from geak_evo.archive import episodes
        native, result, candidate = self.collection_fixture()
        collected = native.collect(result, candidate)
        del collected["artifacts"]["native_collection_attempt.json"]
        snapshot, history, kwargs = self.committed_episode_fixture(result, candidate, collected)
        self.assertEqual(episodes.committed_native_episode(self.store, self.root, snapshot, history, **kwargs)[0],
                         result["receipt"])

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_collection_locator_rejects_wrong_provenance_and_supports_first_retry_records(self):
        from geak_evo.archive import episodes
        native, result, candidate = self.collection_fixture()
        collected = native.collect(result, candidate)
        attempt = Path(result["attempt_dir"])
        prepared = investigation.read(base.file_ref(attempt / "native_prepared.json"))
        clock = investigation.read(self.clock_ref)
        record = json.loads(collected["artifacts"]["native_collection_attempt.json"])
        def locate(value):
            return episodes._collection_manifest(attempt, {"native_collection_attempt.json": investigation.encoded(value)},
                prepared, result["receipt_ref"], candidate, clock)
        self.assertEqual(locate(record), record["manifest_ref"])
        first_version = copy.deepcopy(record)
        del first_version["manifest_ref"]
        self.assertEqual(locate(first_version), record["manifest_ref"])
        changes = [lambda v: v.update(revision=1), lambda v: v.update(revision=True),
            lambda v: v.update(manifest_ref=base.file_ref(self.prompt)),
            lambda v: v["intent"].update(receipt_ref=base.file_ref(self.prompt)),
            lambda v: v["intent"].update(candidate_id=self.request["tip"]["candidate_id"]),
            lambda v: v["status"].update(returncode=7),
            lambda v: v["status"]["command"].extend(["--output", str(self.root)]),
            lambda v: v["status"]["command"].__setitem__(-1, str(self.root)),
            lambda v: v.update(status_ref=base.file_ref(self.prompt))]
        for change in changes:
            value = copy.deepcopy(record)
            change(value)
            with self.subTest(value=value), self.assertRaises((ValueError, FileNotFoundError)):
                locate(value)

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_collection_recovers_interruption_before_output_and_charges_downtime(self):
        native, result, candidate = self.collection_fixture()
        attempt = Path(result["attempt_dir"])
        def interrupted(command, **kwargs):
            self.calls.append((command, kwargs))
            kwargs["stderr"].write(b"collector interrupted before output\n")
            self.now += 1
            raise KeyboardInterrupt
        native.process_runner = interrupted
        with self.assertRaises(KeyboardInterrupt):
            native.collect(result, candidate)
        original_intent = (attempt / "collection_intent.json").read_bytes()
        self.now += 10  # Recovery downtime remains inside the original charge.
        native.process_runner = self.process
        collected = native.collect(result, candidate)
        self.assertEqual(collected["costs"], {"wall_seconds": 11.75, "evaluations": 2})
        self.assertEqual((attempt / "collection_intent.json").read_bytes(), original_intent)
        self.assertEqual((attempt / "collection.stderr").read_bytes(), b"collector interrupted before output\n")
        self.assertTrue((attempt / "collection-0001/archive/manifest.json").is_file())
        self.assertEqual(sum("--input" in call[0] for call in self.calls), 1)
        self.assertEqual(sum("--config" in call[0] for call in self.calls), 2)
        replay = json.loads(collected["artifacts"]["native_collection_attempt.json"])
        self.assertEqual(replay["revision"], 1)
        self.assertEqual(replay["intent"]["receipt_ref"], result["receipt_ref"])
        self.assertEqual(replay["status"]["returncode"], 0)
        self.assertEqual(investigation.read(replay["intent_ref"]), replay["intent"])
        self.assertEqual(investigation.read(replay["status_ref"]), replay["status"])

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_collection_preserves_partial_output_and_rechecks_in_fresh_directory(self):
        native, result, candidate = self.collection_fixture()
        attempt = Path(result["attempt_dir"])
        def partial(command, **kwargs):
            self.calls.append((command, kwargs))
            archive = Path(command[command.index("--output") + 1]) / "archive"
            archive.mkdir(parents=True)
            (archive / "manifest.json").write_bytes(b'{"unfinished":')
            raise KeyboardInterrupt
        native.process_runner = partial
        with self.assertRaises(KeyboardInterrupt):
            native.collect(result, candidate)
        native.process_runner = self.process
        collected = native.collect(result, candidate)
        self.assertEqual((attempt / "collection/archive/manifest.json").read_bytes(), b'{"unfinished":')
        self.assertEqual(json.loads(collected["artifacts"]["native_source_roles.json"]), {"terminal": candidate})
        self.assertTrue((attempt / "collection-0001.status.json").is_file())

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_complete_output_from_failed_collector_is_not_reused_as_success(self):
        native, result, candidate = self.collection_fixture()
        attempt = Path(result["attempt_dir"])
        def failed_after_output(command, **kwargs):
            self.process(command, **kwargs)
            return SimpleNamespace(returncode=7)
        native.process_runner = failed_after_output
        with self.assertRaisesRegex(ValueError, "collection failed"):
            native.collect(result, candidate)
        first_manifest = (attempt / "collection/archive/manifest.json").read_bytes()
        native.process_runner = self.process
        native.collect(result, candidate)
        self.assertEqual((attempt / "collection/archive/manifest.json").read_bytes(), first_manifest)
        self.assertEqual(json.loads((attempt / "collection.status.json").read_bytes())["returncode"], 7)
        self.assertEqual(json.loads((attempt / "collection-0001.status.json").read_bytes())["returncode"], 0)
        self.assertEqual(len(self.calls), 3)  # One engineer; two CPU collections.

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_collection_repeats_validation_after_output_but_before_return(self):
        native, result, candidate = self.collection_fixture()
        attempt = Path(result["attempt_dir"])
        def interrupted_after_output(command, **kwargs):
            self.process(command, **kwargs)
            raise KeyboardInterrupt
        native.process_runner = interrupted_after_output
        with self.assertRaises(KeyboardInterrupt):
            native.collect(result, candidate)
        self.assertTrue((attempt / "collection/archive/manifest.json").is_file())
        self.assertFalse((attempt / "collection.status.json").exists())
        native.process_runner = self.process
        # The newly run collection must still reject changed terminal bytes.
        (self.workspace / self.paths[0]).write_text("corrupted terminal\n")
        with self.assertRaisesRegex(ValueError, "retained source changed"):
            native.collect(result, candidate)
        self.assertEqual(len(self.calls), 3)

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_collection_retry_rejects_changed_binding_before_any_process(self):
        native, result, candidate = self.collection_fixture()
        native.collect(result, candidate)
        attempt = Path(result["attempt_dir"])
        binding = attempt / "native_entry_binding.json"
        binding.chmod(0o600)
        binding.write_bytes(b"different input\n")
        with self.assertRaisesRegex(ValueError, "saved collection input changed"):
            native.collect(result, candidate)
        self.assertEqual(len(self.calls), 2)

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_collection_retry_rejects_new_input_refs_and_new_boot(self):
        native, result, candidate = self.collection_fixture()
        native.collect(result, candidate)
        attempt = Path(result["attempt_dir"])
        path = attempt / "native_prepared.json"
        original = path.read_bytes()
        # Even identical parsed preparation through differently hashed bytes
        # must not silently replace the original collection intent.
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "saved inputs or boot"):
            native.collect(result, candidate)
        path.write_bytes(original)
        with mock.patch.object(self.clock, "boot_id", return_value="another-boot"):
            with self.assertRaisesRegex(ValueError, "saved inputs or boot"):
                native.collect(result, candidate)
        self.assertEqual(len(self.calls), 2)

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_collection_retry_rejects_corrupted_pinned_collector(self):
        native, result, candidate = self.collection_fixture()
        native.collect(result, candidate)
        self.collector.write_text("changed collector\n")
        with self.assertRaises(ValueError):
            native.collect(result, candidate)
        self.assertEqual(len(self.calls), 2)

    @unittest.skipIf(ORIGINAL, "same-operation collection recovery")
    def test_old_partial_collection_without_timing_intent_requires_owner_recovery(self):
        native, result, candidate = self.collection_fixture()
        (Path(result["attempt_dir"]) / "collection.stdout").write_bytes(b"old attempt\n")
        with self.assertRaisesRegex(ValueError, "lacks its original intent"):
            native.collect(result, candidate)
        self.assertEqual(len(self.calls), 1)

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
