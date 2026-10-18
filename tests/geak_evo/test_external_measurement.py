"""CPU checks; native callbacks and admission custody are not exercised."""

import contextlib
import copy
import hashlib
import io
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

_script = Path(__file__).resolve().parents[2] / "scripts/native_runtime/external_measurement.py"
_spec = importlib.util.spec_from_file_location("measurement", _script)
measurement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(measurement)
from geak_evo.archive import store as store_module
from geak_evo import control_result
from geak_evo.archive.refs import file_ref
from geak_evo.evaluate.jobs import job_digest
from geak_evo.evaluate.evaluator import ArenaEvaluationError
import test_external_ut as external_fixtures


class OrdinaryRegistrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.native = self.root / "native"
        self.native.mkdir()
        self.source = self.native / "adapter.py"
        self.source.write_bytes(b"# synthetic ordinary source\n")
        self.wrapper = self.root / "measurement.py"
        self.wrapper.write_bytes(b"# synthetic registered entrypoint\n")
        source_sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.admission = {
            "assets_sha256": {"adapter.py": source_sha},
            "registration_sha256": {
                str(self.source): source_sha,
                str(self.wrapper): hashlib.sha256(self.wrapper.read_bytes()).hexdigest(),
                str(self.root / "inaccessible-owner/owner.py"): "a" * 64,
            },
        }
        # Synthetic files are ordinary-owned; retain the real file/hash reader.
        patch = mock.patch.object(measurement, "protected", side_effect=Path)
        patch.start()
        self.addCleanup(patch.stop)

    def verify(self):
        return measurement.ordinary_registration(self.admission, self.native, self.wrapper)

    def test_unreadable_owner_source_stays_in_full_admission_without_ordinary_read(self):
        before = copy.deepcopy(self.admission)
        checked = self.verify()
        self.assertEqual(set(checked), {str(self.source), str(self.wrapper)})
        self.assertEqual(self.admission, before)

    def test_changed_native_source_refuses_before_import(self):
        self.source.write_bytes(b"# changed reader code\n")
        with self.assertRaisesRegex(ValueError, "registered file changed"):
            self.verify()

    def test_unregistered_entrypoint_refuses(self):
        del self.admission["registration_sha256"][str(self.wrapper)]
        with self.assertRaisesRegex(ValueError, "wrapper is absent"):
            self.verify()

    def test_asset_cannot_escape_native_root(self):
        self.admission["assets_sha256"] = {"../measurement.py":
            self.admission["registration_sha256"][str(self.wrapper)]}
        with self.assertRaisesRegex(ValueError, "native asset differs"):
            self.verify()


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "task"
        contract = external_fixtures.fixture(self.task)
        self.job = job_digest(contract)
        self.store = store_module.ResearchStore(self.root / "store",
            allowed_epochs={(self.job, "epoch")}, source_paths={self.job: contract["source_paths"]})
        self.original_source = self.store.capture_source(self.task, contract["source_paths"])
        self.original = self.store.candidate(self.job, "epoch", self.original_source)
        response = {"receipt": {"settled": True, "native_suite_pass_observed": True,
                                "native_reproducible_gain": False}, "costs": {"evaluations": 1}}
        self.evaluator = NS(contract=contract, evaluate=mock.Mock(return_value=response))
        self.services = NS(store=self.store, evaluator=self.evaluator, store_module=store_module,
                           reader=NS(main=mock.Mock(return_value=0)), job_id=self.job, epoch="epoch")

    def run_measure(self, identity="cpu-evaluation"):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = measurement.measure(self.services, self.task, identity)
        return code, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_capture_uses_actual_workspace_source_and_modes_not_original_candidate(self):
        candidate = self.task / "kernel_src/module.py"
        candidate.write_bytes(b"# synthetic changed candidate\n")
        candidate.chmod(0o755)
        (self.task / "unrelated.py").write_bytes(b"# outside the declared source closure\n")
        code, rows = self.run_measure()
        self.assertEqual(code, 0)  # A valid no-gain native pass is not retried.
        identity, result = rows
        self.assertNotEqual(identity["candidate_id"], self.original)
        saved = self.store.get_candidate(identity["candidate_id"])
        self.assertEqual(saved["source_id"], identity["source_id"])
        files = self.store.source(identity["source_id"])["files"]
        self.assertEqual([row["path"] for row in files], ["kernel_src/module.py"])
        self.assertEqual(files[0]["executable"], 0o111)
        self.assertEqual(self.store._bytes(files[0]["blob_id"]), candidate.read_bytes())
        self.assertEqual(result["external_evaluation"], {
            "candidate_id": identity["candidate_id"], "evaluation_id": "cpu-evaluation"})
        self.evaluator.evaluate.assert_called_once_with(identity["candidate_id"], evaluation_id="cpu-evaluation")

    def test_changed_protected_workspace_file_refuses_before_evaluation(self):
        (self.task / "ut_main.py").write_bytes(b"# changed protected UT\n")
        with self.assertRaisesRegex(ValueError, "protected workspace artifact changed"):
            self.run_measure()
        self.evaluator.evaluate.assert_not_called()

    def test_symlinked_protected_file_refuses_before_evaluation(self):
        target = self.task / "ut_main.py"
        target.unlink()
        target.symlink_to(self.task / "harness_lib.py")
        with self.assertRaises(OSError):
            self.run_measure()
        self.evaluator.evaluate.assert_not_called()

    def test_missing_source_never_falls_back_to_original(self):
        (self.task / "kernel_src/module.py").unlink()
        with self.assertRaisesRegex(ValueError, "candidate source closure is incomplete"):
            self.run_measure()
        self.evaluator.evaluate.assert_not_called()

    def test_read_delegates_exact_existing_result_abi_without_evaluation(self):
        arguments = ["--candidate-id", "candidate", "--evaluation-id", "evaluation",
                     "--workspace", str(self.task), "--patch", "patch.diff"]
        with mock.patch.object(measurement, "bind_services", return_value=self.services) as bind:
            code = measurement.main(["--admission", "admission.json", "a" * 64, "read", *arguments])
        self.assertEqual(code, 0)
        bind.assert_called_once_with({"path": "admission.json", "sha256": "a" * 64})
        self.services.reader.main.assert_called_once_with(arguments, evaluator=self.evaluator)
        self.evaluator.evaluate.assert_not_called()

    def test_read_refuses_changed_protected_artifact_before_reader(self):
        (self.task / "baseline_src/module.py").write_bytes(b"# changed ORIGINAL\n")
        with mock.patch.object(measurement, "bind_services", return_value=self.services):
            with self.assertRaisesRegex(ValueError, "protected workspace artifact changed"):
                measurement.main(["--admission", "admission.json", "a" * 64, "read",
                    "--candidate-id", "candidate", "--evaluation-id", "evaluation", "--workspace", str(self.task)])
        self.services.reader.main.assert_not_called()

    def test_measure_allocates_distinct_actual_evaluation_ids(self):
        outputs = []
        with mock.patch.object(measurement, "bind_services", return_value=self.services):
            for _ in range(2):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(measurement.main(["--admission", "admission.json", "a" * 64,
                        "measure", "--workspace", str(self.task)]), 0)
                outputs.append(json.loads(output.getvalue().splitlines()[0]))
        self.assertEqual(outputs[0]["candidate_id"], outputs[1]["candidate_id"])
        self.assertNotEqual(outputs[0]["evaluation_id"], outputs[1]["evaluation_id"])
        for row in outputs:
            self.assertRegex(row["evaluation_id"], r"^geak-[0-9a-f]{32}$")

    def test_existing_id_reconciles_once_and_rejects_different_candidate(self):
        original = external_fixtures.ExternalTests()
        original.setUp()
        self.addCleanup(original.doCleanups)
        services = NS(store=original.store, evaluator=original.evaluator, store_module=store_module,
                      job_id=original.evaluator.contract_id, epoch=original.evaluator.epoch)
        task = original.root / "workspace"
        shutil.copytree(original.root / "task", task)
        with mock.patch.object(original.evaluator, "launch", wraps=original.evaluator.launch) as launch, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(measurement.measure(services, task, "same-id"), 0)
            self.assertEqual(measurement.measure(services, task, "same-id"), 0)
            (task / "kernel_src/module.py").write_bytes(b"# different candidate\n")
            with self.assertRaisesRegex(ArenaEvaluationError, "evaluation ID belongs to a different"):
                measurement.measure(services, task, "same-id")
        launch.assert_called_once()

    def test_final_capture_does_not_dispatch_and_rejects_later_source_changes(self):
        original = external_fixtures.ExternalTests()
        original.setUp()
        self.addCleanup(original.doCleanups)
        services = NS(store=original.store, evaluator=original.evaluator, store_module=store_module,
                      job_id=original.evaluator.contract_id, epoch=original.evaluator.epoch)
        task = original.root / "workspace"
        shutil.copytree(original.root / "task", task)
        with mock.patch.object(original.evaluator, "launch", wraps=original.evaluator.launch) as launch:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                measurement.measure(services, task, "fresh-final", capture_only=True)
            selected = json.loads(output.getvalue())
            self.assertTrue(selected["capture_only"])
            launch.assert_not_called()
            self.assertFalse((original.evaluator.root / "fresh-final").exists())
            path = task / "kernel_src/module.py"
            before = path.read_bytes()
            path.write_bytes(b"# post-selection edit\n")
            with self.assertRaisesRegex(ValueError, "candidate selected before final validation"):
                measurement.measure(services, task, "fresh-final", candidate_id=selected["candidate_id"])
            launch.assert_not_called()
            path.write_bytes(before)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(measurement.measure(services, task, "fresh-final",
                                 candidate_id=selected["candidate_id"]), 0)
            launch.assert_called_once()
            with self.assertRaisesRegex(ValueError, "identity is already in use"):
                measurement.measure(services, task, "fresh-final", capture_only=True)

    def test_canonical_capture_joins_commit_clock_source_and_only_matching_native_evidence(self):
        fixture = external_fixtures.ExternalTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        task = fixture.root / "workspace"
        shutil.copytree(fixture.root / "task", task)
        def git(*args):
            subprocess.run(["git", "-C", str(task), *args], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "fixture")
        clock_path = fixture.root / "clock.json"
        clock_path.write_text(json.dumps({"boot_id": "fixture", "started_monotonic_ns": 100, "wall_seconds": 28800}))
        services = NS(store=fixture.store, evaluator=fixture.evaluator, store_module=store_module,
            reader=control_result, job_id=fixture.evaluator.contract_id, epoch=fixture.evaluator.epoch,
            clock_ref=file_ref(clock_path))
        fixture.evaluator.evaluate(fixture.original, evaluation_id="existing")
        selected = {"candidate_id": fixture.original, "evaluation_id": "existing"}
        def capture(boot="fixture"):
            output = io.StringIO()
            instants = iter([200, 300])
            with contextlib.redirect_stdout(output):
                measurement.capture_selection(services, task, selected=selected,
                    clock=lambda: (boot, next(instants)))
            row = json.loads(output.getvalue())
            _, raw = fixture.store.evidence_artifact(row["capture_ref"]["evidence_id"], "canonical-selection.json")
            self.assertEqual(json.loads(raw), {k: v for k, v in row.items() if k != "capture_ref"})
            return row
        with mock.patch.object(fixture.evaluator, "launch", side_effect=AssertionError("no new evaluation")):
            original = capture()
            self.assertEqual(original["source_id"], fixture.original_source)
            self.assertEqual(original["canonical_commit"]["status"], "committed")
            self.assertEqual(original["development_evidence"]["status"], "validated")
            self.assertEqual(original["clock_ref"], services.clock_ref)
            self.assertEqual((original["lower_monotonic_ns"], original["upper_monotonic_ns"]), (200, 300))
            (task / "kernel_src/module.py").write_bytes(b"# different hand merge\n")
            dirty = capture()
            self.assertEqual(dirty["canonical_commit"]["status"], "unresolved")
            git("add", ".")
            git("-c", "user.name=fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "hand merge")
            changed = capture()
            self.assertEqual(changed["canonical_commit"]["status"], "committed")
            self.assertNotEqual(changed["source_id"], original["source_id"])
            self.assertEqual(changed["development_evidence"]["status"], "unresolved")
            self.assertNotIn("result", changed["development_evidence"])
            with self.assertRaisesRegex(ValueError, "original clock"):
                capture("other-boot")


if __name__ == "__main__":
    unittest.main()
