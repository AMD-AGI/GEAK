"""CPU contracts for shared host composition; no model, GPU or owner launch."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import test_evaluate_contracts as native_fixture
from test_planner_model_port import FixedClock, SyntheticServices
from model_profile_fixture import codex_route

from geak_evo import host
from geak_evo.archive.refs import file_ref
from geak_evo.model_profile import ModelProfile
from geak_evo.planner import model
from geak_evo.planner.native import NativePlannerPending


class SharedHostTests(unittest.TestCase):
    def setUp(self):
        self.fixture = native_fixture.EvaluationContractTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root / "campaign"
        self.root.mkdir(mode=0o700)
        self.fixture.store = native_fixture.ResearchStore(self.root / "store",
            allowed_epochs=[(self.fixture.job_id, "epoch")],
            source_paths={self.fixture.job_id: self.fixture.contract["source_paths"]})
        source = self.fixture.store.capture_source(self.fixture.task, self.fixture.contract["source_paths"])
        self.fixture.candidate = self.fixture.store.candidate(self.fixture.job_id, "epoch", source)
        self.clock = FixedClock("cpu-fixture-boot")
        self.route = codex_route()

        def save(name, value):
            path = self.root / name
            model.put(path, value)
            return file_ref(path)

        self.save = save
        route_ref = save("route.json", self.route)
        self.profile = ModelProfile.from_route(self.route, route_ref, workflow_backend="codex")
        self.clock_ref = save("clock.json", {"boot_id": self.clock.boot, "session_id": "fixture",
            "started_monotonic_ns": 100_000_000_000,
            "started_unix": 1000, "wall_seconds": 10000})
        helper_ref = save("helper.json", {"synthetic": True})
        model_config = {"schema": "geak-standalone-model-config-v1", "launch_admitted": True,
            "workflow_backend": "codex", "clock_path": self.clock_ref["path"],
            "work_root": str(self.root / "work"), "read_roots": [], "state_root": str(self.root / "model"),
            "model_route_ref": route_ref, "tool_audit_ref": helper_ref,
            "budget_policy": "owner_wall_deadline", "codex_runtime_ref": helper_ref,
            "codex_native_root": str(self.root)}
        self.config = {"contract_ref": save("contract.json", self.fixture.contract),
            "artifact_root": str(self.fixture.task), "evaluation_root": str(self.root / "shared-evaluations"),
            "helper_ref": helper_ref, "tool_audit_ref": helper_ref,
            "imports": {"codex_native_root": str(self.root)},
            "backend_config": {"model_config": model_config, "probe_config": {
                "job_id": self.fixture.job_id, "measurement_epoch_id": "epoch", "native_envelope_seconds": 60}}}
        self.transport = SyntheticServices(model_config, self.route, self.clock)

        def prepared_runtime(config, route, route_ref, audit_ref):
            self.assertEqual(route, self.route)
            self.assertEqual(route_ref, self.profile.model_route_ref)
            self.assertEqual(audit_ref, helper_ref)
            self.transport.config = config
            self.transport.descriptor = {"workspace": config["work_root"],
                "read_roots": config["read_roots"], "operational_roots": [config["state_root"]]}
            return self.transport

        self.transport.create_runtime = prepared_runtime
        runtime = model.ModelRuntime(*self.transport.dependencies, clock=self.clock,
                                    routes=SimpleNamespace(validate=lambda route: route))
        services = host.HostServices(runtime, SimpleNamespace(fresh_workspace=self.forbidden),
            SimpleNamespace(), self.forbidden, self.forbidden, helper_ref, helper_ref)
        self.arguments = dict(campaign_root=self.root,
            templates={"config": save("config.json", self.config),
                "admission": save("admission.json", {"bdfs": ["0000:01:00.0"], "physical_ids": ["0000:01:00.0@1"]}),
                "entry": save("entry.json", {"source_paths": ["kernel.py"]}),
                "collection": save("collection.json", {})},
            clock_ref=self.clock_ref, search_policy={"arm": "EVO", "study_root": str(self.root.parent),
                "shared_read_roots": [str(Path(host.__file__).parent / "prompts")],
                "native_capabilities": ["continuation", "program_lease"]},
            shared_budget={"deadline_ns": 10100_000_000_000}, profile=self.profile, services=services,
            case_id="case-a", evaluation_limit=10, planner_seconds=60, planner_decision_limit=2,
            sealed_root=self.root / "sealed", root_state_root=self.root / "root-state", common_read_roots=[])

    def forbidden(self, *args, **kwargs):
        raise AssertionError("owner execution is forbidden in this CPU fixture")

    def prepare(self, **changes):
        return host.prepare_native_search(self.fixture.store, "campaign", self.fixture.candidate,
                                          **{**self.arguments, **changes})

    def test_composes_real_search_and_preserves_profile_policy_and_original_clock(self):
        search = self.prepare()
        self.assertEqual(search.profile, self.profile)
        self.assertEqual(search.loop.program_lease, 2)
        self.assertIs(search.planner.backend, model)
        self.assertIs(search.assignment.workspace_helper, self.arguments["services"].workspace)
        self.assertIs(search.assignment.seal_operation, self.arguments["services"].seal_operation)
        self.assertEqual(search.planner.model_config["model_route_ref"], self.profile.model_route_ref)
        self.assertEqual(search.loop.snapshot()["state"]["deadline"], 11000)
        self.assertEqual(self.transport.launches, [])
        self.assertEqual(self.prepare().loop.snapshot(), search.loop.snapshot())
        with self.assertRaisesRegex(ValueError, "saved host binding changed"):
            self.prepare(planner_decision_limit=3)
        with self.assertRaisesRegex(ValueError, "saved host binding changed"):
            self.prepare(case_id="case-b")
        selected = self.root / "admitted-prompts"
        selected.mkdir()
        for source in (Path(host.__file__).parent / "prompts").glob("*.md"):
            (selected / source.name).write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "saved host binding changed"):
            self.prepare(prompt_dir=selected)
        with self.assertRaisesRegex(ValueError, "original clock"):
            self.prepare(clock=FixedClock("another-boot"))
        changed_clock = model.read(self.clock_ref["path"])
        changed_clock["deadline_ns"] = 10100_000_000_000 - 1000
        short_ref = self.save("short-clock.json", changed_clock)
        with self.assertRaisesRegex(ValueError, "duration and absolute deadline"):
            self.prepare(clock_ref=short_ref)

    def test_pending_search_never_dispatches_a_second_operation(self):
        search = self.prepare()
        with self.assertRaisesRegex(ValueError, "bounded host driver"):
            search.run(max_steps=0)
        # Run the actual Loop -> NativePlanner -> model.run path. The CPU
        # transport lacks real provider identity evidence, so the planner must
        # retain the call as pending even though its process settled.
        with self.assertRaises(NativePlannerPending):
            search.run(max_steps=1)
        snapshot = search.loop.snapshot()
        self.assertEqual(len(self.transport.launches), 1)
        receipts = list(self.root.glob("planner/calls/*/model/attempts/*/receipt.json"))
        self.assertEqual(len(receipts), 1)
        receipt = model.read(receipts[0])
        self.assertEqual(receipt["model_profile"], self.profile.as_dict())
        self.assertTrue(receipt["settlement"]["settled"])
        with self.assertRaisesRegex(ValueError, "same-ID reconciliation"):
            self.prepare().run(max_steps=1)
        self.assertEqual(len(self.transport.launches), 1)
        self.assertEqual(search.loop.snapshot(), snapshot)

    def test_search_reserve_shortens_search_without_replacing_original_native_budget(self):
        original = model.read(self.clock_ref["path"])
        shared_budget = dict(self.arguments["shared_budget"])
        shared_budget["max_calls"] = 10
        search = self.prepare(search_reserve_seconds=4800, evaluation_limit=7, shared_budget=shared_budget)
        state = search.loop.snapshot()["state"]
        self.assertEqual(state["deadline"], 6200)
        self.assertEqual(state["limits"], {"wall_seconds": 5200, "evaluations": 7})
        self.assertEqual(search.assignment.shared_budget, shared_budget)
        self.assertEqual(search.assignment.clock_ref, self.clock_ref)
        self.assertEqual(search.planner.clock_ref, self.clock_ref)
        self.assertEqual(model.read(self.clock_ref["path"]), original)
        self.assertEqual(model.read(self.root / "host-binding.json")["search_reserve_seconds"], 4800)
        self.assertEqual(self.prepare(search_reserve_seconds=4800, evaluation_limit=7,
                                      shared_budget=shared_budget).loop.snapshot(), search.loop.snapshot())
        with self.assertRaisesRegex(ValueError, "saved host binding changed"):
            self.prepare(search_reserve_seconds=4700)
        # No complete planner window remains: return settled history without
        # calling a model or consuming the final measurement reserve.
        self.clock.now = 5290
        snapshot = search.run(max_steps=2)
        self.assertEqual(snapshot["state"]["phase"], "idle")
        self.assertEqual(snapshot["state"]["opportunities"], [])
        self.assertEqual(self.transport.launches, [])

    def test_invalid_search_reserves_fail_before_any_dispatch(self):
        for reserve in (-1, True, float("nan"), float("inf"), 10000, 10001):
            with self.subTest(reserve=reserve), self.assertRaisesRegex(ValueError, "search reserve"):
                self.prepare(search_reserve_seconds=reserve)
        self.assertEqual(self.transport.launches, [])

    def test_native_callbacks_keep_same_id_raw_metrics_and_physical_mapping(self):
        evaluator = host.make_native_evaluator(self.fixture.store, self.config, callbacks=self.fixture)
        first = evaluator.evaluate(self.fixture.candidate, evaluation_id="same-operation")
        recovered = host.make_native_evaluator(self.fixture.store, self.config, callbacks=self.fixture)
        second = recovered.evaluate(self.fixture.candidate, evaluation_id="same-operation")
        self.assertEqual(len(self.fixture.launches), 1)
        self.assertEqual(self.fixture.handles, ["synthetic-handle", None])
        self.assertEqual(first["receipt"]["per_case_times_ms"], second["receipt"]["per_case_times_ms"])
        self.assertEqual(second["receipt"]["per_case_times_ms"], {"case-a": 1.25, "case-b": 1.25})
        self.assertIn("raw/host/owner-report.json", second["artifacts"])
        self.assertFalse(second["receipt"]["measurement_valid"])


if __name__ == "__main__":
    unittest.main()
