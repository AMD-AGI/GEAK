"""CPU checks; native callbacks and admission custody are not exercised."""

import contextlib
import io
import importlib.util
import json
import shutil
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
from geak_evo.evaluate.jobs import job_digest
from geak_evo.evaluate.evaluator import ArenaEvaluationError
import test_external_ut as external_fixtures


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


if __name__ == "__main__":
    unittest.main()
