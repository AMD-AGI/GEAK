"""Common handoff through real command/evaluator/export/Git, with CPU callbacks.

Search and recommendation authenticity use explicit synthetic fixtures here;
their shared implementations have separate contract tests. No model/GPU runs.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest import mock

from geak_evo import command, dispatch, result
from geak_evo.archive.store import _write
from geak_evo.planner.model import encoded
import test_command as fixtures
import test_engineer_port as engineer_fixtures
import test_shared_host as host_fixtures


class DispatchAdapterTests(unittest.TestCase):
    def setUp(self):
        self.owner = fixtures.CommandTests(methodName="runTest")
        self.owner.setUp()
        self.addCleanup(self.owner.doCleanups)
        self.f = self.owner.f
        self.directory = self.f.root / "dispatch" / "run"
        self.directory.mkdir(parents=True)
        self.repo = self.directory / "repo"
        self.workspace = self.repo / "pkg" / "operator"
        self.workspace.mkdir(parents=True)
        (self.repo / "unrelated.txt").write_text("protected sibling\n")
        for name in self.f.contract["artifacts"]:
            target = self.workspace / name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes((self.f.task / name).read_bytes())
        dispatch._git(self.repo, "init", "-q")
        dispatch._git(self.repo, "add", "-A")
        dispatch._git(self.repo, "commit", "-qm", "fixture baseline")
        self.base = dispatch._git(self.repo, "rev-parse", "HEAD").decode().strip()
        self.config = self.owner.config
        clock = command._document(self.config["search"]["clock_ref"])
        clock["started_unix"] = time.time()
        self.config["search"]["clock_ref"] = self.owner.save("clock.json", clock)
        native = command._document(self.config["search"]["templates"]["config"])
        self.binding = {"schema": "geak-evo-dispatch-owner-v1", "dispatch_root": str(self.directory.parent),
            "owner_config_ref": {}, "task_ref": native["contract_ref"], "max_steps": 1, "export_seconds": 30,
            "case_map": {"lane-a": "case-a", "lane-b": "case-b"}, "device": {"arch": "fixture", "gpus": ["0"]}}
        self.task = {"schema_version": 1, "eval_dir": str(self.directory), "workspace": str(self.workspace),
            "base_commit": self.base, "task_ref": native["contract_ref"], "editable_paths": ["kernel.py"],
            "baseline": {"per_case": {"lane-a": 99.0, "lane-b": 77.0}, "geomean_ms": 87.0},
            "objective": {"kind": "geomean"}, "incumbent_speedup": 1, "min_improvement": 0.01,
            "device": self.binding["device"], "budget": {"max_rounds": 1, "deadline_epoch": None,
                "agent_timeout_s": None, "min_improvement": 0.01}}

    def invoke(self, *, cli=False):
        self.binding["owner_config_ref"] = self.owner.save("owner.json", self.config)
        binding_ref = self.owner.save("dispatch-owner.json", self.binding)
        task_path = self.directory / "task.json"
        _write(task_path, encoded(self.task))
        with mock.patch.object(command, "prepare_native_search", return_value=self.owner.search), \
             mock.patch.object(command, "validate_final_recommendation", return_value=self.owner.boundary.selection), \
             mock.patch.object(result, "validate_final_recommendation", return_value=self.owner.boundary.selection):
            services = dict(owner_binding_ref=binding_ref, store=self.f.store, services=self.owner.services,
                            evaluator=self.f.engine, budget=self.owner.budget, episode=lambda *args: None)
            if cli:
                code = dispatch.main(["--task", str(task_path), "--out", str(self.directory / "result.json")], **services)
                return code, json.loads((self.directory / "result.json").read_bytes())
            return dispatch.run_dispatch_task(task_path, self.directory / "result.json", **services)

    def test_measured_pairs_and_exact_selected_bytes_are_committed_in_real_subdirectory(self):
        expected = self.owner.boundary.selected_root / "kernel.py"
        expected.chmod(0o755)
        source = self.f.store.capture_source(self.owner.boundary.selected_root, ["kernel.py"])
        candidate = self.f.store.candidate(self.f.job_id, "epoch", source)
        self.owner.boundary.choice.update(candidate_id=candidate, source_id=source)
        value = self.invoke()
        self.assertEqual(value["status"], "improved", value)
        self.assertEqual(value["best"]["speedup"], 2)
        self.assertEqual(value["best"]["per_case"], {
            "lane-a": {"baseline_ms": 1.0, "optimized_ms": 0.5},
            "lane-b": {"baseline_ms": 1.0, "optimized_ms": 0.5}})
        self.assertEqual((self.workspace / "kernel.py").read_bytes(), expected.read_bytes())
        self.assertEqual((self.workspace / "kernel.py").stat().st_mode & 0o111, 0o111)
        self.assertEqual((self.repo / "unrelated.txt").read_text(), "protected sibling\n")
        dispatch._clean(self.repo, value["best"]["commit"])
        dispatch._git(self.repo, "merge-base", "--is-ancestor", self.base, value["best"]["commit"])
        self.assertEqual(len(self.f.launches), 2)
        self.owner.search.run.assert_called_once_with(max_steps=1)
        self.assertFalse(value["measurement_valid"] or value["framework_final_pass"])

    def test_common_time_share_weights_are_preserved_through_case_mapping(self):
        self.task["objective"] = {"kind": "time_weighted", "weights": {"lane-a": 3.0, "lane-b": 1.0},
                                  "workload_spec_path": "/fixture/workload.json"}
        self.config["objective"] = {**self.task["objective"], "weights": {"case-a": 3.0, "case-b": 1.0}}
        def settle(handle, request, operation):
            times = ({"case-a": 1.0, "case-b": 1.0} if request["binding"]["candidate_id"] == self.f.candidate
                     else {"case-a": 0.5, "case-b": 0.25})
            self.f.mutate = lambda host: host["observations"]["benchmark"].update(times_ms=times)
            return self.owner.boundary.settle(handle, request, operation)
        self.f.engine.settle = settle
        value = self.invoke()
        self.assertEqual(value["status"], "improved", value)
        self.assertAlmostEqual(value["best"]["speedup"], 16 / 7)

    def test_incumbent_threshold_prevents_a_commit_despite_native_improvement(self):
        self.task["incumbent_speedup"] = 2
        value = self.invoke()
        self.assertEqual(value["status"], "no_improvement", value)
        self.assertIsNone(value["best"])
        dispatch._clean(self.repo, self.base)
        self.assertEqual(len(self.f.launches), 2)

    def test_missing_case_map_is_rejected_before_search(self):
        del self.binding["case_map"]["lane-b"]
        value = self.invoke()
        self.assertEqual(value["status"], "error")
        self.assertIn("one-to-one", value["notes"])
        self.owner.search.run.assert_not_called()
        self.assertEqual(self.f.launches, [])

    def test_wrong_admitted_task_is_rejected_before_search(self):
        self.task["task_ref"] = dict(self.task["task_ref"], sha256="b" * 64)
        value = self.invoke()
        self.assertEqual(value["status"], "error")
        self.assertIn("not the admitted", value["notes"])
        self.owner.search.run.assert_not_called()

    def test_clean_but_changed_protected_artifact_is_rejected(self):
        (self.workspace / "metadata.json").write_text("{}\n")
        dispatch._git(self.repo, "add", "-A")
        dispatch._git(self.repo, "commit", "-qm", "changed fixture metadata")
        self.task["base_commit"] = dispatch._git(self.repo, "rev-parse", "HEAD").decode().strip()
        value = self.invoke()
        self.assertEqual(value["status"], "error")
        self.assertIn("protected task artifact", value["notes"])
        self.assertEqual(self.f.launches, [])

    def test_owner_clock_cannot_outlive_dispatcher_command_cutoff(self):
        clock = command._document(self.config["search"]["clock_ref"])
        clock["started_unix"] -= 86400  # An old wall timestamp cannot extend the monotonic budget.
        self.config["search"]["clock_ref"] = self.owner.save("clock.json", clock)
        self.task["budget"]["deadline_epoch"] = time.time() + 100
        value = self.invoke()
        self.assertEqual(value["status"], "error")
        self.assertIn("export margin", value["notes"])
        self.owner.search.run.assert_not_called()

    def test_wall_clock_adjustment_does_not_reject_a_fitting_monotonic_budget(self):
        clock = command._document(self.config["search"]["clock_ref"])
        clock["started_unix"] += 86400
        self.config["search"]["clock_ref"] = self.owner.save("clock.json", clock)
        self.task["budget"]["deadline_epoch"] = time.time() + 4000
        value = self.invoke()
        self.assertEqual(value["status"], "improved", value)

    def test_declared_absent_selected_file_can_already_be_absent(self):
        (self.owner.boundary.selected_root / "kernel.py").unlink()
        source = self.f.store.capture_source(self.owner.boundary.selected_root, ["kernel.py"])
        (self.workspace / "kernel.py").unlink()
        commit = dispatch._commit(self.f.store, source, self.repo, self.workspace, self.base, None)
        self.assertFalse((self.workspace / "kernel.py").exists())
        dispatch._clean(self.repo, commit)

    def interrupted_evaluation(self, phase, *, during_search=False):
        previous = signal.getsignal(signal.SIGTERM)
        callback = getattr(self.f.engine, phase)
        def terminated(*args):
            callback(*args)
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("signal handler should interrupt evaluation")
        setattr(self.f.engine, phase, terminated)
        next_decision = mock.Mock(return_value=self.owner.snapshot)
        if during_search:
            def search(**kwargs):
                try:
                    self.f.engine.evaluate(self.f.candidate, evaluation_id="search-evaluation")
                except command.ArenaEvaluationUnsettled:
                    pass
                return next_decision()
            self.owner.search.run.side_effect = search
        code, value = self.invoke(cli=True)
        self.assertEqual(code, 1)
        self.assertEqual(value["status"], "error")
        self.assertIsNone(value["best"])
        self.assertEqual(len(self.f.launches), 1)
        next_decision.assert_not_called()
        receipts = list(self.f.engine.root.glob("*/collection-*/arena_evaluation.json"))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_bytes())
        self.assertIn({"stage": phase, "error_kind": "CommandInterrupted"}, receipt["collection_errors"])
        self.assertFalse(receipt["native_suite_pass_observed"])
        self.assertEqual(receipt["settled"], phase == "launch")
        self.assertEqual(len(self.f.handles), 1)
        if phase == "launch":
            self.assertIsNone(self.f.handles[0])
            self.assertTrue((receipts[0].parent / "host_receipt.json").is_file())
            self.assertTrue((receipts[0].parent / "raw" / "host" / "cleanup.json").is_file())
        native = json.loads((self.directory / "native-result" / "result.json").read_bytes())
        self.assertEqual((native["validation_status"], native["error_kind"]),
                         ("command_incomplete", "CommandInterrupted"))
        self.assertEqual(native["command"]["stage"], "search" if during_search else "baseline")
        if not during_search:
            row = next(iter(self.owner.budget.operations.values()))
            self.assertEqual(row["status"], "unsettled")
            self.assertGreater(row["elapsed_seconds"], 0)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        dispatch._clean(self.repo, self.base)

    def test_sigterm_unwinds_native_accounting_and_restores_handler(self):
        self.interrupted_evaluation("settle")

    def test_sigterm_during_final_evaluation_launch_preserves_interruption(self):
        self.interrupted_evaluation("launch")

    def test_sigterm_during_search_evaluation_launch_stops_search(self):
        self.interrupted_evaluation("launch", during_search=True)

    def test_sigterm_during_search_evaluation_settlement_stops_search(self):
        self.interrupted_evaluation("settle", during_search=True)

    def test_sigterm_during_engineer_turn_writes_receipt_and_stops_search(self):
        engineer = engineer_fixtures.EngineerTests(methodName="runTest")
        engineer.setUp()
        self.addCleanup(engineer.doCleanups)
        children, settlements = [], []
        next_decision = mock.Mock(return_value=self.owner.snapshot)

        class TerminateOnce(subprocess.Popen):
            interrupted = False

            def wait(self, *args, **kwargs):
                if not self.interrupted:
                    self.interrupted = True
                    os.kill(os.getpid(), signal.SIGTERM)
                return super().wait(*args, **kwargs)

        def launch(argv, **kwargs):
            child = TerminateOnce([sys.executable, "-B", "-c", "import time; time.sleep(30)"], **kwargs)
            children.append(child)
            return child

        def settle(process, attempt):
            settlements.append(attempt)
            return {"settled": process.returncode is not None}

        def search(**kwargs):
            engineer_fixtures.run_turn(engineer.request, engineer.attempts,
                backend_options={"model_profile": engineer.profile}, launch_worker=launch,
                settle_worker=settle, capture_tip=engineer.capture)
            return next_decision()

        self.owner.search.run.side_effect = search
        previous = signal.getsignal(signal.SIGTERM)
        code, value = self.invoke(cli=True)
        self.assertEqual((code, value["status"], value["best"]), (1, "error", None))
        next_decision.assert_not_called()
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)
        self.assertEqual(len(settlements), 1)
        receipts = list(engineer.attempts.glob("*/receipt.json"))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_bytes())
        self.assertEqual(receipt["process_status"], "interrupted")
        self.assertEqual(receipt["errors"], ["transport:CommandInterrupted"])
        self.assertTrue(receipt["process_reaped"] and receipt["settlement"]["settled"])
        self.assertGreater(receipt["elapsed_seconds"], 0)
        self.assertIsNotNone(receipt["captured_tip_source_id"])
        native = json.loads((self.directory / "native-result" / "result.json").read_bytes())
        self.assertEqual((native["validation_status"], native["error_kind"]),
                         ("command_incomplete", "CommandInterrupted"))
        self.assertEqual(native["command"]["stage"], "search")
        self.assertFalse((self.owner.root / "evo-command-search.json").exists())
        self.assertEqual(self.f.launches, [])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        dispatch._clean(self.repo, self.base)

    def test_sigterm_during_real_planner_search_retains_cost_and_stops(self):
        shared = host_fixtures.SharedHostTests(methodName="runTest")
        shared.setUp()
        self.addCleanup(shared.doCleanups)
        search = shared.prepare()
        self.owner.search.run.side_effect = search.run
        self.binding["max_steps"] = self.task["budget"]["max_rounds"] = 2

        def wait(**kwargs):
            shared.clock.now += 2
            os.kill(os.getpid(), signal.SIGTERM)

        shared.transport.wait = wait
        previous = signal.getsignal(signal.SIGTERM)
        code, value = self.invoke(cli=True)
        self.assertEqual((code, value["status"], value["best"]), (1, "error", None))
        self.assertEqual(len(shared.transport.launches), 1)
        calls = list((shared.root / "planner" / "calls").iterdir())
        self.assertEqual(len(calls), 1)
        receipt = json.loads(next(calls[0].glob("model/attempts/*/receipt.json")).read_bytes())
        self.assertEqual(receipt["status"], "failed")
        self.assertIn("CommandInterrupted", receipt["error"])
        self.assertTrue(receipt["settlement"]["settled"])
        self.assertEqual(receipt["elapsed_seconds"], 2.25)
        outcome = json.loads((calls[0] / "outcome.json").read_bytes())
        self.assertEqual(outcome, {"wall_seconds": 2.25, "execution_may_have_started": True})
        self.assertTrue((calls[0] / "failure.json").is_file())
        self.assertFalse((calls[0] / "completion.json").exists())
        native = json.loads((self.directory / "native-result" / "result.json").read_bytes())
        self.assertEqual((native["validation_status"], native["error_kind"]),
                         ("command_incomplete", "CommandInterrupted"))
        self.assertEqual(self.f.launches, [])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        dispatch._clean(self.repo, self.base)


if __name__ == "__main__":
    unittest.main()
