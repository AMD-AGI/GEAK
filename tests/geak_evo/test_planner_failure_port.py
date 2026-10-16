"""Failed-call attribution with synthetic records and no process/model calls.

GEAK_EVO_OCT15_SOURCE enables comparisons with the frozen validator. Its native
identity helpers are loaded verbatim; transport adapters are synthetic.
"""

import ast
import copy
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest import mock

from geak_evo.planner import failure, model


def source_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError("frozen validation helpers are missing")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


class PlannerFailurePortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = cls.original_native = None
        source = os.environ.get("GEAK_EVO_OCT15_SOURCE")
        if source:
            root = Path(source)
            if root.is_file():
                root = root.parent.parent
            decoder = ModuleType("geak_evo.cli_planner")
            decoder.__dict__.update(json=json, Decimal=Decimal)
            source_functions(root / "evolution/src/geak_evo/cli_planner.py",
                             ("_need", "_json", "_decode"), decoder.__dict__)
            spec = importlib.util.spec_from_file_location("oct15_failed_native_planner", root / "bridge/failed_native_planner.py")
            cls.original = importlib.util.module_from_spec(spec)
            with mock.patch.dict("sys.modules", {"geak_evo.cli_planner": decoder}):
                spec.loader.exec_module(cls.original)
            cls.original_native = ModuleType("native_planner")
            cls.original_native.__dict__.update(
                copy=copy, Path=Path, MODEL="claude-opus-5", need=cls.original.need,
                _decode=decoder._decode, _read_ref=cls.original.checked_bytes, file_ref=cls.original.file_ref)
            source_functions(root / "bridge/native_planner.py",
                             ("planner_model_identity", "validate_planner_model_config", "validate_actual_planner_config"),
                             cls.original_native.__dict__)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.operation, self.campaign_id = "planner-op", "campaign"
        self.binding = {"job_id": "job", "measurement_epoch_id": "epoch"}
        self.call = self.root / "campaign/planner/calls" / failure.sha256_hex(self.operation.encode())
        self.attempt = self.call / "model/attempts" / self.operation
        self.attempt.mkdir(parents=True)
        (self.call / "scratch").mkdir()
        (self.call / "public").mkdir()
        self.source_root = self.root / "runtime"
        self.source_root.mkdir()
        self.modules = {}
        for name in ("responses_route", "codex_events", "codex_runtime", "codex_cpu_boundary"):
            module = ModuleType("codex_native." + name)
            path = self.source_root / (name + ".py")
            path.write_bytes(("# Synthetic dependency: " + name + "\n").encode())
            module.__file__ = str(path)
            self.modules[module.__name__] = module
        routes, events, transport, boundary = [self.modules["codex_native." + name] for name in
                                               ("responses_route", "codex_events", "codex_runtime", "codex_cpu_boundary")]
        routes.validate = transport.validate_model_route = lambda value: copy.deepcopy(value)
        routes.route_id = routes.verify_command = lambda *args: "route-identity"
        routes.canonical = failure.encoded
        events.TURN_EVENTS = {"thread.started", "turn.started", "turn.completed", "turn.failed"}
        events.inspect_codex_events = lambda *args, **kwargs: copy.deepcopy(self.observed)
        self.supervisor = b"# Synthetic worker supervisor, never executed.\n"
        boundary.BASE_SOURCE_SHA256 = failure.sha256_hex(self.supervisor)
        self.runtime = model.ModelRuntime(None, transport, None, routes=routes, events=events, boundary=boundary)
        package = ModuleType("codex_native")
        for name in ("responses_route", "codex_events", "codex_runtime", "codex_cpu_boundary"):
            setattr(package, name, self.modules["codex_native." + name])
        self.modules["codex_native"] = package
        claude = ModuleType("geak_evo.claude_transport")
        claude.validate_model_route = routes.validate
        self.modules[claude.__name__] = claude
        if self.original_native is not None:
            self.modules["native_planner"] = self.original_native
        self.patch = mock.patch.dict("sys.modules", self.modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write(self, name, value):
        raw = value if isinstance(value, bytes) else failure.encoded(value)
        path = self.paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        self.artifacts[name] = raw

    def mutate(self, name, change):
        value = json.loads(self.artifacts[name])
        change(value)
        self.write(name, value)

    def fixture(self, backend, explicit):
        route = {"requested_model": "gpt-6-astra", "accepted_reported_models": ["gpt-6-astra"],
                 "requested_reasoning_effort": "xhigh"}
        for name, value in (("route.json", route), ("clock.json", {"deadline": 1000}),
                            ("audit.json", {"synthetic": True}), ("cli", {"synthetic": True})):
            (self.root / name).write_bytes(failure.encoded(value))
        self.clock_ref = failure.file_ref(self.root / "clock.json")
        descriptor = {"workspace": str(self.call / "scratch"), "read_roots": [str(self.call / "public")],
                      "executable_ref": failure.file_ref(self.root / "cli"),
                      "private_home": "home", "private_temp": "temp", "private_codex_home": "codex-home"}
        (self.root / "descriptor.json").write_bytes(failure.encoded(descriptor))
        config = {"schema": "geak-standalone-model-config-v1", "launch_admitted": True,
                  "workflow_backend": backend, "clock_path": self.clock_ref["path"],
                  "work_root": descriptor["workspace"], "state_root": str(self.call / "model"),
                  "read_roots": descriptor["read_roots"], "model_route_ref": failure.file_ref(self.root / "route.json")}
        if backend == "codex":
            config.update(budget_policy="owner_wall_deadline", codex_native_root=str(self.source_root),
                          codex_runtime_ref=failure.file_ref(self.root / "descriptor.json"),
                          tool_audit_ref=failure.file_ref(self.root / "audit.json"))
        else:
            config["max_budget_usd"] = 10
        self.config = config
        self.paths = {name: self.call / name for name in failure.LOCAL_NAMES}
        self.paths.update({"native/" + name: self.attempt / name for name in failure.NATIVE_NAMES})
        if backend == "codex":
            self.paths["native/model-config.json"] = self.attempt / "model-config.json"
        self.artifacts = {}
        for name in self.paths:
            self.write(name, {})
        self.write("model-config.json", config)
        if backend == "codex":
            self.write("native/model-config.json", config)
        request = {"schema": "geak-standalone-model-request-v1", "invocation_id": self.operation,
                   "model": "gpt-6-astra", "label": "planner", "timeout_ms": 50_000,
                   "prompt": "Retained exact prompt", "cwd": descriptor["workspace"]}
        self.write("request.json", request)
        self.write("native/request.json", request)
        self.write("native/prompt.txt", request["prompt"].encode())
        self.write("native/stderr.txt", b"Retained error output\n")
        self.write("native/worker_supervisor.py", self.supervisor)
        empty = {"empty": True, "procs": [], "threads": [], "events": {"populated": "0"}}
        worker = {"operation_id": "worker-op", "worker_identity": "group-identity", "worker_path": "group-path",
                  "child_pid": 101, "child_start_ticks": 202, "unit": "worker-unit", "settled": True,
                  "child_reaped": True, "child_status": "exited", "child_exit_code": 1,
                  "elapsed_seconds": 3, "empty_observation": empty}
        self.write("native/worker_settlement.json", worker)
        self.write("native/worker_admission.json", {key: worker[key] for key in
                   ("operation_id", "worker_identity", "worker_path", "child_pid", "child_start_ticks", "unit")})
        plan = {"operation_id": "worker-op", "unit": "worker-unit", "attempt": str(self.attempt),
                "source_sha256": failure.sha256_hex(self.supervisor)}
        transport = {}
        if backend == "codex":
            transport = {"backend": "codex", "model_route_ref": config["model_route_ref"], "route_id": "route-identity",
                         "runtime_descriptor_ref": config["codex_runtime_ref"], "requested_model": "gpt-6-astra",
                         "requested_reasoning_effort": "xhigh", "cli": descriptor["executable_ref"],
                         "transport": failure.file_ref(self.runtime.transport.__file__)}
            argv = ["synthetic-cli", "exec"]
            plan.update(route_id="route-identity", argv=argv, expected_argv_sha256=failure.sha256_hex(failure.encoded(argv)),
                        executable_ref=descriptor["executable_ref"], cwd=descriptor["workspace"],
                        transport_boundary_source=failure.file_ref(self.runtime.boundary.__file__),
                        responses_route_source=failure.file_ref(self.runtime.routes.__file__))
            plan.update({key: descriptor[key] for key in ("private_home", "private_temp", "private_codex_home")})
        self.write("native/worker_plan.json", plan)
        self.observed = {"identity_valid": True, "reported_models": [] if backend == "codex" else ["gpt-6-astra"],
                         "reported_tools": None if backend == "codex" else ["Bash"], "reported_reasoning_effort": None,
                         "terminal": "turn.failed" if explicit else None, "report_text": None, "thread_id": "thread",
                         "protocol_errors": [], "usage_status": "unknown", "usage": None,
                         "native_total_cost_usd": None, "native_model_usage": None, "native_terminal": None}
        if backend == "codex":
            events = [{"type": "thread.started"}, {"type": "turn.started"}]
            if explicit:
                events.append({"type": "turn.failed"})
                self.observed["native_terminal"] = events[-1]
        else:
            events = [{"type": "system", "session_id": "thread"}]
            if explicit:
                terminal = {"type": "result", "subtype": "error", "is_error": True,
                            "stop_reason": "error", "session_id": "thread", "total_cost_usd": 0.025,
                            "usage": {"input_tokens": 20}, "modelUsage": {"model": {"costUSD": 0.025}}}
                events.append(terminal)
                self.observed.update(native_terminal={key: terminal[key] for key in
                    ("type", "subtype", "is_error", "stop_reason", "session_id")},
                    usage_status="reported_native_terminal_unqualified_incremental", usage=terminal["usage"],
                    native_total_cost_usd=terminal["total_cost_usd"], native_model_usage=terminal["modelUsage"])
        self.write("native/events.jsonl", b"\n".join(failure.encoded(event) for event in events) + b"\n")
        receipt = {"schema": "geak-standalone-model-receipt-v1", "invocation_id": self.operation,
                   "clock_ref": self.clock_ref, "model_route_ref": config["model_route_ref"],
                   "requested_model": "gpt-6-astra", "label": "planner", "model_backend_ref": failure.file_ref(model.__file__),
                   "original_native_timeout_ms": 50_000, "started_monotonic_ns": 110_000_000_000,
                   "effective_wall_seconds": 50, "tool_audit": failure.file_ref(self.root / "audit.json"),
                   "transport": transport, "status": "failed", "returncode": 1,
                   "error": "ValueError: retained failed turn", "observed": self.observed, "elapsed_seconds": 4,
                   "settlement": {"settled": True, "empty_observation": empty,
                       "receipt_sha256": failure.sha256_hex(self.artifacts["native/worker_settlement.json"]),
                       "receipt_path": str(self.attempt / "worker_settlement.json"), "operation_id": "worker-op",
                       "cgroup_identity": "group-identity", "cgroup_path": "group-path"}}
        if backend == "codex":
            receipt["model_config_ref"] = failure.file_ref(self.attempt / "model-config.json")
        self.write("native/receipt.json", receipt)
        self.write("native/intent.json", receipt)
        self.write("intent.json", {"operation_id": self.operation, "invocation_id": self.operation,
                   "campaign_id": self.campaign_id, "binding": self.binding,
                   "admission": {"model_config": config, "clock_ref": self.clock_ref,
                                 "campaign_root": str(self.call.parent.parent.parent), "backend_ref": receipt["model_backend_ref"]}})
        self.write("launch-intent.json", {"operation_id": self.operation,
                   "request_sha256": failure.sha256_hex(self.artifacts["request.json"])})
        self.write("failure.json", {"retry_permitted": False, "error_kind": "ValueError"})
        self.write("outcome.json", {"wall_seconds": 5, "execution_may_have_started": True})

    def validate(self, module):
        kwargs = {"runtime": self.runtime} if module is failure else {}
        return module.validate_failed_invocation(self.call, self.campaign_id, self.binding,
                                                 self.clock_ref, self.artifacts, **kwargs)

    def test_exact_failed_and_incomplete_turns_keep_costs_and_unknown_usage(self):
        for backend in ("claude", "codex"):
            for explicit in (False, True):
                with self.subTest(backend=backend, explicit=explicit):
                    self.fixture(backend, explicit)
                    actual = self.validate(failure)
                    if self.original is not None:
                        self.assertEqual(self.validate(self.original), actual)
                    self.assertEqual(actual["costs"], {"wall_seconds": 5, "evaluations": 0})
                    self.assertNotIn("proposal", actual)
                    self.assertEqual(actual["failure"]["native_status"], "failed")
                    self.assertEqual("terminal" in actual["failure"], explicit)
                    if backend == "codex" or not explicit:
                        self.assertIsNone(actual["reported_model_cost_usd"])
                        self.assertIsNone(actual["provider_usage"])
                    else:
                        self.assertEqual(actual["reported_model_cost_usd"], 0.025)
                        self.assertEqual(actual["provider_usage"], {"input_tokens": 20})

    def test_unresolved_or_mismatched_records_never_settle(self):
        changes = (
            ("native/receipt.json", lambda value: value.update(returncode=0)),
            ("native/receipt.json", lambda value: value["settlement"].update(settled=False)),
            ("native/worker_settlement.json", lambda value: value.update(child_reaped=False)),
            ("native/worker_settlement.json", lambda value: value["empty_observation"].update(procs=[102])),
            ("outcome.json", lambda value: value.update(wall_seconds=2)),
            ("outcome.json", lambda value: value.update(execution_may_have_started=False)),
            ("failure.json", lambda value: value.update(retry_permitted=True)),
            ("native/receipt.json", lambda value: value["observed"].update(native_total_cost_usd=1)),
        )
        modules = (failure,) if self.original is None else (self.original, failure)
        for backend in ("claude", "codex"):
            for name, change in changes:
                with self.subTest(backend=backend, artifact=name, change=change):
                    self.fixture(backend, False)
                    self.mutate(name, change)
                    errors = []
                    for module in modules:
                        with self.assertRaises(ValueError) as caught:
                            self.validate(module)
                        errors.append((type(caught.exception).__name__, str(caught.exception)))
                    self.assertTrue(all(error == errors[0] for error in errors))

    def test_existing_result_and_changed_artifact_are_rejected(self):
        self.fixture("claude", False)
        path = self.call / "native-result.json"
        path.write_text("null")
        for module in ((failure,) if self.original is None else (self.original, failure)):
            with self.assertRaisesRegex(ValueError, "failed-turn closure forbidden"):
                self.validate(module)
        path.unlink()
        (self.attempt / "stderr.txt").write_bytes(b"changed")
        for module in ((failure,) if self.original is None else (self.original, failure)):
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                self.validate(module)

    def test_checked_bytes_rejects_changed_content(self):
        path = self.root / "authority.json"
        path.write_bytes(b'{"authority":"synthetic"}')
        reference = failure.file_ref(path)
        self.assertEqual(failure.checked_bytes(reference), path.read_bytes())
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            failure.checked_bytes(reference)


if __name__ == "__main__":
    unittest.main()
