"""Replay the pinned external patch and exercise optional report retention on CPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from geak_evo.archive.store import ResearchStore


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReportContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.sources.cleanup)
        root = Path(cls.sources.name)
        package = Path(__file__).resolve().parents[2] / "scripts/native_runtime"
        patch = json.loads((package / "SOURCES.json").read_text())["patches"][0]
        raw = (package / patch["patch"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == patch["patch_sha256"]
        for row in patch["files"]:
            ref = row["original_ref"]
            if ref is None:
                continue
            if not Path(ref["path"]).is_file():
                raise unittest.SkipTest("retained exact external base sources are unavailable")
            raw = Path(ref["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
            target = root / row["native_root_relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        command = ["git", "apply", "-p1", str(package / patch["patch"])]
        subprocess.run(command[:2] + ["--check"] + command[2:], cwd=root, check=True)
        subprocess.run(command, cwd=root, check=True)
        for row in patch["files"]:
            assert hashlib.sha256((root / row["native_root_relative_path"]).read_bytes()).hexdigest() == row["source_ref"]["sha256"]
        from geak_evo.evaluate import artifacts, evaluator
        # The retained provider uses the owner's existing legacy-name bridge.
        with mock.patch.dict(sys.modules, {"geak_evo.arena_evaluator": evaluator,
                                           "geak_evo.artifact_transport": artifacts}):
            cls.host_module = load("report_fixture_host", root / "host/host.py")
        cls.harvest = load("report_fixture_harvest", root / "workspace_helpers/harvest.py")
        cls.context = cls.host_module.report_context

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        work = self.root / "work"
        work.mkdir()
        (work / "kernel.py").write_text("value = 1\n")
        self.store = ResearchStore(self.root / "store", allowed_epochs=[("job", "epoch")],
                                   source_paths={"job": ["kernel.py"]})
        source = self.store.capture_source(work, ["kernel.py"])
        self.snapshot = {"source_id": source, "candidate_id": self.store.candidate("job", "epoch", source),
                         "source": self.store.source(source), "job_id": "job", "measurement_epoch_id": "epoch"}
        self.config = {"report_context": dict(self.context.POLICY), "session_id": "session", "turn_id": "turn",
            "workspace": str(work), "job_id": "job", "measurement_epoch_id": "epoch",
            "native_envelope_seconds": 1, "budget": {"boot_id": "fixture-boot"}}
        self.row = {"request": {"request_id": "trial", "expected_source_id": source, "profile": "native"},
                    "request_sha256": "a" * 64, "operation_id": "evaluation"}
        self.report = work / ".idea-output/turn/report.md"
        self.report.parent.mkdir(parents=True)
        self.report.write_text("Try a different dataflow; check spills before interpreting timing.\n")
        self.folder = self.root / "state/trial"
        self.folder.mkdir(parents=True)
        self.publication = self.root / "publication"
        self.publication.mkdir()

    def capture(self, *, recovery=False):
        self.context.capture(self.config, self.store, self.row, self.snapshot, self.folder,
                             write=self.host_module.atomic, recovery=recovery)

    def feedback(self):
        envelope = self.context.publish(self.config, self.store, self.row, self.snapshot,
            self.folder, self.publication, write=self.host_module.atomic)
        return dict(self.context.binding(self.config, self.row, self.snapshot),
                    snapshot=self.snapshot, request_sha256=self.row["request_sha256"],
                    context_refs=envelope, artifact_index=[], native_pass=True)

    def test_capture_precedes_evaluation_and_recovery_never_recaptures(self):
        host = object.__new__(self.host_module.ProbeHost)
        host.config, host.turn, host.capture_report = self.config, self.folder.parent, True
        host._publish_lifecycle = lambda *args: None
        host.check_turn_clock = lambda: None
        host.budget = SimpleNamespace(check_clock=lambda *args: None)
        before = self.report.read_bytes()
        def evaluate(*args):
            self.assertTrue((self.folder / "report-context.json").is_file())
            self.report.write_text("Later explanation written during evaluation.\n")
            raise ValueError("synthetic native failure")
        host.backend = SimpleNamespace(store=self.store, snapshot=lambda *args: self.snapshot,
                                       evaluate=evaluate, reconcile=evaluate)
        first = host._work(self.row)
        self.assertEqual(first["error"], "ValueError: synthetic native failure")
        pointer = (self.folder / "report-context.json").read_bytes()
        recovered = host._work(dict(self.row, snapshot=self.snapshot), recovery=True)
        self.assertEqual(recovered["error"], first["error"])
        self.assertEqual((self.folder / "report-context.json").read_bytes(), pointer)
        evidence = json.loads(pointer)["evidence_id"]
        self.assertEqual(self.store.evidence(evidence)[1]["report.md"], before)
        feedback = self.feedback()
        retained = self.harvest.report_context.retain(feedback, self.config, self.root / "harvest",
                                                     write=self.harvest.write_bytes)
        self.assertIsNone(retained["error"])
        self.assertEqual((self.root / "harvest/report.md").read_bytes(), before)
        self.assertEqual(feedback["artifact_index"], [])
        self.assertTrue(feedback["native_pass"])

    def test_missing_oversized_invalid_and_unsafe_reports_remain_optional(self):
        cases = [(None, "missing"), (b"x" * (self.context.MAX_REPORT_BYTES + 1), "too_large"),
                 (b"\xff", "invalid_utf8"), (b"", "empty")]
        for index, (body, expected) in enumerate(cases):
            with self.subTest(status=expected):
                self.folder = self.root / ("capture-" + str(index)); self.folder.mkdir()
                self.report.unlink(missing_ok=True)
                if body is not None:
                    self.report.write_bytes(body)
                self.capture()
                pointer = json.loads((self.folder / "report-context.json").read_bytes())
                record = json.loads(self.store.evidence(pointer["evidence_id"])[1]["context.json"])
                self.assertEqual(record["availability"], expected)
                if expected != "empty":
                    self.assertIsNone(record["report"])
                    self.assertIsNone(record["full_content_sha256"])
        self.folder = self.root / "unsafe"; self.folder.mkdir()
        self.report.unlink(); self.report.symlink_to(self.root / "store")
        self.capture()
        self.assertTrue((self.folder / "report-context.json").is_file())

    def test_partial_context_publication_reuses_original_bytes(self):
        self.capture()
        original_report = self.report.read_bytes()
        original_pointer = (self.folder / "report-context.json").read_bytes()
        def interrupted_write(path, raw, **kwargs):
            self.host_module.atomic(path, raw, **kwargs)
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.context.publish(self.config, self.store, self.row, self.snapshot,
                self.folder, self.publication, write=interrupted_write)
        self.report.write_text("A later explanation must not replace the capture.\n")
        first = self.feedback()
        self.assertIsNone(first["context_refs"]["error"])
        self.assertEqual(self.feedback(), first)
        self.assertEqual((self.folder / "report-context.json").read_bytes(), original_pointer)
        self.assertEqual((self.publication / "context/report.md").read_bytes(), original_report)
        retained = self.context.retain(first, self.config, self.root / "recovered",
                                       write=self.harvest.write_bytes)
        self.assertEqual(retained["report_sha256"], hashlib.sha256(original_report).hexdigest())
        self.assertIsInstance(retained["capture_monotonic_ns"], int)
        # A mismatched prior publication is never overwritten or accepted.
        body = self.publication / "context/report.md"
        body.chmod(0o600); body.write_bytes(b"corrupt")
        failed = self.feedback()
        self.assertEqual(failed["context_refs"]["availability"], "capture_unavailable")
        self.assertIn("publication differs", failed["context_refs"]["error"])
        self.assertEqual(body.read_bytes(), b"corrupt")

    def test_missing_recovery_capture_is_explicit_and_storage_failure_is_nonfatal(self):
        self.capture(recovery=True)
        pointer = json.loads((self.folder / "report-context.json").read_bytes())
        artifacts = self.store.evidence(pointer["evidence_id"])[1]
        record = json.loads(artifacts["context.json"])
        self.assertEqual(record["availability"], "capture_unavailable")
        self.assertIsNone(record["observed_monotonic_ns"])
        self.assertNotIn("report.md", artifacts)
        self.folder = self.root / "storage-failure"; self.folder.mkdir()
        with mock.patch.object(self.store, "archive_evidence", side_effect=OSError("fixture storage failure")):
            self.capture()
        self.assertFalse((self.folder / "report-context.json").exists())

    def test_corrupt_optional_body_and_wrong_trial_do_not_change_native_result(self):
        self.capture()
        feedback = self.feedback()
        original = copy.deepcopy(feedback)
        ref = next(row for row in feedback["context_refs"]["artifacts"] if row["name"] == "report.md")
        Path(ref["path"]).chmod(0o600); Path(ref["path"]).write_bytes(b"changed")
        retained = self.context.retain(feedback, self.config, self.root / "bad", write=self.harvest.write_bytes)
        self.assertIsNotNone(retained["error"])
        self.assertEqual(feedback, original)
        feedback["context_refs"]["binding"]["operation_id"] = "other"
        retained = self.context.retain(feedback, self.config, self.root / "wrong", write=self.harvest.write_bytes)
        self.assertIn("another trial", retained["error"])

    def test_disabled_policy_old_feedback_and_private_capture_rejection(self):
        self.assertFalse(self.context.enabled({}))
        self.assertFalse(self.context.enabled({"report_context": {"capture": "off", "delivery": "off"}}))
        with self.assertRaisesRegex(ValueError, "private capture-only"):
            self.context.enabled({"report_context": {"capture": "report-v1", "delivery": "off"}})
        self.assertEqual(self.context.retain({}, {}, self.root / "old", write=self.harvest.write_bytes)["availability"],
                         "unsupported_or_not_supplied")

    def test_collector_retry_preserves_context_ref_and_normal_history_delivery(self):
        from test_native_engineer_port import NativeEngineerTests
        from geak_evo.engineer import assignment, investigation
        from geak_evo.archive.refs import file_ref, read_ref
        fixture = NativeEngineerTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        native = fixture.native()
        context_feedback = {}
        def process(command, **kwargs):
            completed = fixture.process(command, **kwargs)
            destination = Path(command[command.index("--output") + 1])
            if "--input" in command:
                candidate = fixture.store.candidate(fixture.job_id, fixture.epoch, fixture.terminal_source)
                snapshot = {"source_id": fixture.terminal_source, "candidate_id": candidate,
                    "job_id": fixture.job_id, "measurement_epoch_id": fixture.epoch}
                config = dict(self.config, workspace=str(fixture.workspace), turn_id=fixture.operation_id,
                              job_id=fixture.job_id, measurement_epoch_id=fixture.epoch)
                row = copy.deepcopy(self.row)
                row["request"]["expected_source_id"] = fixture.terminal_source
                report = fixture.workspace / ".idea-output" / fixture.operation_id / "report.md"
                report.parent.mkdir(parents=True); report.write_text("Original trial premise.\n")
                folder = fixture.root / "context-state"; folder.mkdir()
                publication = fixture.root / "context-publication"; publication.mkdir()
                self.context.capture(config, fixture.store, row, snapshot, folder, write=self.host_module.atomic)
                envelope = self.context.publish(config, fixture.store, row, snapshot, folder, publication,
                                                write=self.host_module.atomic)
                context_feedback.update(self.context.binding(config, row, snapshot), snapshot=snapshot,
                    request_sha256=row["request_sha256"], context_refs=envelope, artifact_index=[])
                feedback_ref = fixture.save("trial-feedback.json", context_feedback)
                entry = json.loads(destination.read_bytes())
                receipt = investigation.read(entry["transport_result"]["receipt_ref"])
                receipt["native_results"][0]["feedback_ref"] = feedback_ref
                entry["transport_result"]["receipt_ref"] = fixture.save("model-receipt.json", receipt)
                destination.write_bytes(investigation.encoded(entry))
                fixture.context_feedback_ref, fixture.context_config = feedback_ref, config
            else:
                archive = destination / "archive"
                manifest = json.loads((archive / "manifest.json").read_bytes())
                saved_feedback = self.harvest.write_bytes(archive / "trial-feedback.json", read_ref(fixture.context_feedback_ref))
                context = self.context.retain(context_feedback, fixture.context_config, archive / "trial-context",
                                              write=self.harvest.write_bytes)
                manifest["trials"] = [{"feedback_ref": fixture.context_feedback_ref, "archive_ref": saved_feedback,
                    "artifacts": context["artifacts"], "context": context, "verified": None,
                    "error": "synthetic failed trial", "source_archive": manifest["terminal"]}]
                (archive / "manifest.json").write_bytes(investigation.encoded(manifest))
            return completed
        native.process_runner = process
        result = fixture.run_native(native)
        candidate = fixture.store.candidate(fixture.job_id, fixture.epoch, fixture.terminal_source,
                                           parents=[fixture.request["tip"]["candidate_id"]])
        native.process_runner = mock.Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            native.collect(result, candidate)
        intent_before = (Path(result["attempt_dir"]) / "collection_intent.json").read_bytes()
        report = fixture.workspace / ".idea-output" / fixture.operation_id / "report.md"
        report.write_text("Later report after collector interruption.\n")
        with mock.patch.object(self.context, "capture", side_effect=AssertionError("unexpected recapture")):
            native.process_runner = process
            collected = native.collect(result, candidate)
        self.assertEqual((Path(result["attempt_dir"]) / "collection_intent.json").read_bytes(), intent_before)
        receipt = json.loads(collected["artifacts"]["native_receipt.json"])
        self.assertEqual(receipt["native_results"][0]["feedback_ref"], fixture.context_feedback_ref)
        identity = fixture.store.archive_evidence(candidate, collected["artifacts"])
        snapshot = {"archive": {"directions": {}}, "state": {"history": [
            {"collection_ref": {"evidence_id": identity, "artifact": "native_idea_harvest.json"}}]}}
        delivered = assignment.archive_files(fixture.store, snapshot, extra_candidates=[candidate])
        self.assertIn(b"Original trial premise.\n", [read_ref(ref) for ref in delivered])
        self.assertNotIn(b"Later report after collector interruption.\n", [read_ref(ref) for ref in delivered])


if __name__ == "__main__":
    unittest.main()
