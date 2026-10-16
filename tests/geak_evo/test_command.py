"""CPU composition checks. Synthetic search/selection and owner callbacks.

The real ArenaEvaluator and result reader/exporter run; no provider or GPU does.
SharedHostTests separately exercises the real prepare/run path below.
"""

import copy
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest import mock

from geak_evo import command, result
from geak_evo.archive.refs import file_ref
from geak_evo.archive.store import _write
from geak_evo.host import HostServices, make_native_evaluator
from geak_evo.model_profile import ModelProfile
from geak_evo.planner.model import encoded
from test_result_boundary import ResultBoundaryTests
from test_shared_host import SharedHostTests
import test_evaluate_contracts as fixtures


class BudgetInterface:
    """Small service double; retained Budget's policy is tested by its owner."""
    def __init__(self, config):
        self.config, self.operations = config, {}
        self.refuse_new = False

    def read(self):
        return {"operations": copy.deepcopy(self.operations)}

    def reserve(self, operation, seconds, kind):
        if operation in self.operations:
            return False
        if self.refuse_new or any(row["status"] != "settled" for row in self.operations.values()):
            raise ValueError("admitted budget exhausted or another operation pending")
        self.operations[operation] = {"status": "reserved", "kind": kind, "executed": None,
                                     "reserved_ns": time.monotonic_ns(), "elapsed_seconds": None,
                                     "envelope_seconds": seconds}
        return True

    def settle(self, operation, seconds, executed, settled):
        self.operations[operation].update(status="settled" if settled else "unsettled",
                                         elapsed_seconds=seconds, executed=executed,
                                         envelope_exceeded=seconds > self.operations[operation]["envelope_seconds"])


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.boundary = b = ResultBoundaryTests(methodName="runTest")
        b.setUp()
        self.addCleanup(b.doCleanups)
        self.f = f = b.fixture
        self.root = f.root
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        clock = {"boot_id": boot, "session_id": "fixture", "started_monotonic_ns": time.monotonic_ns(), "wall_seconds": 3600}
        self.budget = BudgetInterface({"boot_id": boot, "session_id": "fixture",
                                      "deadline_ns": clock["started_monotonic_ns"] + 3600_000_000_000})
        self.services = HostServices(SimpleNamespace(), None, None, None, None, {}, {})
        self.config = {"schema": "geak-evo-admitted-command-v1", "baseline_candidate_id": f.candidate,
            "objective": {"kind": "geomean"},
            "final_measurements": {"baseline_seconds": 60, "selected_seconds": 60},
            "search": {"campaign_id": "campaign", "seed_candidate_id": f.candidate,
                "campaign_root": str(f.root), "shared_budget": self.budget.config,
                "clock_ref": self.save("clock.json", clock),
                "profile": ModelProfile("codex", "gpt-6-astra", "xhigh", str(f.root / "route.json"), "a" * 64).as_dict(),
                "templates": {"config": self.save("native.json", {
                    "contract_ref": self.save("contract.json", f.contract), "artifact_root": str(f.task),
                    "evaluation_root": str(f.engine.root), "backend_config": {"probe_config": {
                        "job_id": f.job_id, "measurement_epoch_id": "epoch", "native_envelope_seconds": 60}}})}}}
        self.snapshot = {"campaign_id": "campaign", "revision": 1, "state": {"phase": "idle"}}
        f.store.checkpoint("campaign", f.candidate, expected_revision=0,
            note={"claim": "fixture", "scope": "CPU only", "next_test": "fixture", "evolution": self.snapshot["state"]})
        self.search = SimpleNamespace(run=mock.Mock(return_value=self.snapshot))
        self.index = 0
        original_settle = f.engine.settle
        def settle(handle, request, operation):
            speed = 1.0 if request["binding"]["candidate_id"] == f.candidate else 0.5
            f.mutate = lambda host: host["observations"]["benchmark"].update(
                times_ms={"case-a": speed, "case-b": speed})
            return original_settle(handle, request, operation)
        f.engine.settle = settle

    def save(self, name, value):
        _write(self.root / name, encoded(value))
        return file_ref(self.root / name)

    def invoke(self):
        self.index += 1
        ref = self.save("owner.json", self.config)
        with mock.patch.object(command, "prepare_native_search", return_value=self.search) as prepare, \
             mock.patch.object(command, "validate_final_recommendation", return_value=self.boundary.selection), \
             mock.patch.object(result, "validate_final_recommendation", return_value=self.boundary.selection):
            value = command.run_admitted_task(ref, store=self.f.store, services=self.services,
                evaluator=self.f.engine, budget=self.budget, episode=lambda *args: None,
                output_dir=self.root / ("output-" + str(self.index)), max_steps=1)
        return value, prepare

    def test_search_then_two_real_evaluator_calls_and_checked_export(self):
        value, prepare = self.invoke()
        self.assertEqual(value["outcome"], "improved")
        self.assertEqual(value["final_speedup"], 2)
        self.assertTrue(Path(value["final_patch"]).is_file())
        self.assertEqual(len(self.f.launches), 2)
        self.assertEqual([row["kind"] for row in self.budget.operations.values()], ["baseline", "final"])
        self.assertTrue(all(row["status"] == "settled" for row in self.budget.operations.values()))
        prepare.assert_called_once()
        self.search.run.assert_called_once_with(max_steps=1)
        self.assertFalse(value["framework_final_pass"])
        costs = copy.deepcopy(self.budget.operations)
        replay, prepare = self.invoke()
        prepare.assert_not_called()
        self.assertEqual(len(self.f.launches), 2)
        self.assertEqual(self.budget.operations, costs)
        self.assertEqual(replay["command"]["selected_evaluation_id"], value["command"]["selected_evaluation_id"])

    def test_unsettled_final_recovers_same_id_without_search_or_launch(self):
        settle = self.f.engine.settle
        def pending(handle, request, operation):
            value = settle(handle, request, operation)
            if request["binding"]["candidate_id"] != self.f.candidate:
                value["receipt"]["cleanup"]["settled"] = False
            return value
        self.f.engine.settle = pending
        first, _ = self.invoke()
        self.assertEqual(first["validation_status"], "command_incomplete")
        self.assertEqual(first["command"]["stage"], "final")
        self.assertEqual(list(self.budget.operations.values())[-1]["status"], "unsettled")
        self.f.engine.settle = settle
        self.budget.refuse_new = True
        recovered, prepare = self.invoke()
        prepare.assert_not_called()
        self.assertEqual(recovered["outcome"], "improved")
        self.assertEqual(len(self.f.launches), 2)
        self.assertIsNone(self.f.handles[-1])

    def test_final_recovery_after_the_original_deadline_actually_expires(self):
        # A short real monotonic window, with each tiny CPU operation admitted
        # before it expires. The second invocation starts after that same bound.
        seconds = 0.05
        clock = {"boot_id": self.budget.config["boot_id"], "session_id": "fixture",
                 "started_monotonic_ns": time.monotonic_ns(), "wall_seconds": 0.3}
        self.budget.config["deadline_ns"] = clock["started_monotonic_ns"] + 300_000_000
        self.config["search"]["clock_ref"] = self.save("clock.json", clock)
        native = command._document(self.config["search"]["templates"]["config"])
        native["backend_config"]["probe_config"]["native_envelope_seconds"] = seconds
        self.config["search"]["templates"]["config"] = self.save("native.json", native)
        self.config["final_measurements"] = {"baseline_seconds": seconds, "selected_seconds": seconds}
        reserve = self.budget.reserve
        def bounded(operation, envelope, kind):
            if operation not in self.budget.operations:
                self.assertLess(time.monotonic_ns() + int(envelope * 1e9), self.budget.config["deadline_ns"])
            return reserve(operation, envelope, kind)
        self.budget.reserve = bounded
        settle = self.f.engine.settle
        def pending(handle, request, operation):
            value = settle(handle, request, operation)
            if request["binding"]["candidate_id"] != self.f.candidate:
                value["receipt"]["cleanup"]["settled"] = False
            return value
        self.f.engine.settle = pending
        first, _ = self.invoke()
        self.assertEqual(first["command"]["stage"], "final")
        operation = first["command"]["selected_evaluation_id"]
        baseline = first["command"]["baseline_evaluation_id"]
        baseline_cost = self.budget.operations[baseline]["elapsed_seconds"]
        time.sleep(max(0, (self.budget.config["deadline_ns"] - time.monotonic_ns()) / 1e9) + 0.01)
        self.assertGreater(time.monotonic_ns(), self.budget.config["deadline_ns"])
        self.f.engine.settle = settle
        recovered, prepare = self.invoke()
        prepare.assert_not_called()
        self.assertEqual(len(self.f.launches), 2)
        self.assertIsNone(self.f.handles[-1])
        self.assertEqual(self.budget.operations[baseline]["elapsed_seconds"], baseline_cost)
        self.assertEqual(self.budget.operations[operation]["status"], "settled")
        self.assertTrue(self.budget.operations[operation]["envelope_exceeded"])
        self.assertEqual(recovered["validation_status"], "command_incomplete")

    def test_failed_baseline_never_funds_final_even_on_replay(self):
        settle = self.f.engine.settle
        def failed(*args):
            value = settle(*args)
            value["receipt"]["observations"]["correctness"]["status"] = "failed"
            return value
        self.f.engine.settle = failed
        for _ in range(2):
            value, _ = self.invoke()
            self.assertEqual(value["command"]["stage"], "baseline")
            self.assertEqual(value["final_patch"], "")
            self.assertIsNone(value["final_speedup"])
        self.assertEqual(len(self.f.launches), 1)

    def test_recovery_cannot_erase_previously_observed_execution(self):
        settle = self.f.engine.settle
        execution_started = True
        def pending(handle, request, operation):
            value = settle(handle, request, operation)
            if request["binding"]["candidate_id"] != self.f.candidate:
                host = value["receipt"]
                host["cleanup"]["settled"] = False
                host["execution_started"] = execution_started
                host["costs"]["evaluations"] = int(execution_started)
            return value
        self.f.engine.settle = pending
        first, _ = self.invoke()
        operation = first["command"]["selected_evaluation_id"]
        before = self.budget.operations[operation].copy()
        self.assertIs(before["executed"], True)
        execution_started = False
        recovered, _ = self.invoke()
        after = self.budget.operations[operation]
        self.assertEqual(recovered["validation_status"], "command_incomplete")
        self.assertIs(after["executed"], True)
        self.assertEqual(after["status"], "unsettled")
        self.assertGreaterEqual(after["elapsed_seconds"], before["elapsed_seconds"])
        self.assertEqual(len(self.f.launches), 2)

    def test_search_crash_never_funds_a_second_batch(self):
        self.search.run.side_effect = ValueError("interrupted after a committed decision")
        first, _ = self.invoke()
        self.assertEqual(first["command"]["stage"], "search")
        second, prepare = self.invoke()
        prepare.assert_not_called()
        self.assertIn("started without a frozen return", second["error"])
        self.search.run.assert_called_once()
        self.assertEqual(self.f.launches, [])

    def test_exhausted_budget_and_changed_head_cannot_launch(self):
        self.budget.refuse_new = True
        value, _ = self.invoke()
        self.assertEqual(value["command"]["stage"], "baseline")
        self.budget.refuse_new = False
        self.f.store.checkpoint("campaign", self.f.candidate, expected_revision=1,
            note={"claim": "new", "scope": "fixture", "next_test": "fixture", "evolution": {"phase": "stopped"}})
        value, _ = self.invoke()
        self.assertIn("advanced beyond", value["error"])
        self.assertEqual(self.f.launches, [])

    def test_actual_shared_search_starts_once_and_retains_pending_without_native_launch(self):
        # Exercise actual prepare -> NativeSearch.run -> Loop -> NativePlanner.
        # Its synthetic provider evidence cannot authenticate a model result.
        contract = self.f.contract
        files = {name: (self.f.task / name).read_bytes() for name in contract["artifacts"]}
        shared = SharedHostTests(methodName="runTest")
        with mock.patch.object(fixtures, "fixture", return_value=(contract, files)):
            shared.setUp()
        self.addCleanup(shared.doCleanups)
        shared.clock.boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        original = {"boot_id": shared.clock.boot, "session_id": "fixture", "started_monotonic_ns": 100_000_000_000,
                    "started_unix": 1000, "wall_seconds": 10000}
        _write(shared.root / "clock.json", encoded(original))
        shared.arguments["clock_ref"] = file_ref(shared.root / "clock.json")
        shared.arguments["shared_budget"] = {"boot_id": shared.clock.boot, "session_id": "fixture", "deadline_ns": 10100_000_000_000}
        args = dict(shared.arguments)
        services = args.pop("services")
        args.update(campaign_id="campaign", seed_candidate_id=shared.fixture.candidate,
                    profile=shared.profile.as_dict())
        for key in ("campaign_root", "sealed_root", "root_state_root"):
            args[key] = str(args[key])
        config = dict(self.config, search=args, baseline_candidate_id=shared.fixture.candidate)
        ref = shared.save("command.json", config)
        evaluator = make_native_evaluator(shared.fixture.store, shared.config, callbacks=shared.fixture)
        value = command.run_admitted_task(ref, store=shared.fixture.store, services=services,
            evaluator=evaluator, budget=BudgetInterface(args["shared_budget"]), episode=lambda *args: None,
            output_dir=shared.root / "out", max_steps=1)
        self.assertEqual(value["validation_status"], "command_incomplete")
        self.assertEqual(len(shared.transport.launches), 1)
        self.assertEqual(shared.fixture.launches, [])


if __name__ == "__main__":
    unittest.main()
