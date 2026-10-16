"""Synthetic planner model lifecycle checks; no model or subprocess is started.

Set GEAK_EVO_OCT15_SOURCE to the frozen source directory to also compare its
backend with the production port. The source file is loaded read-only.
"""

import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from geak_evo.planner import model
from geak_evo.clock import SystemClock
from model_profile_fixture import codex_route


class FixedClock:
    def __init__(self, boot):
        self.now = 110.0
        self.boot = boot

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return int(self.now * 1_000_000_000)

    def boot_id(self):
        return self.boot


class SyntheticServices:
    """Implement the existing dependency interfaces with an in-memory process."""

    def __init__(self, config, route, clock, case="completed"):
        self.config, self.route, self.clock, self.case = config, route, clock, case
        self.launches, self.waits, self.audits = [], [], []
        self.descriptor = {
            "workspace": config["work_root"], "read_roots": config["read_roots"],
            "operational_roots": [config["state_root"]],
        }
        self.qualified = SimpleNamespace(QualifiedClaudeTransport=self.claude_route)
        self.settings = SimpleNamespace(FileSettingsRuntime=lambda runtime: runtime)

    @property
    def dependencies(self):
        if self.config["workflow_backend"] == "codex":
            return None, self, None
        return self.qualified, self, self.settings

    def claude_route(self, path, **kwargs):
        return SimpleNamespace(tool_audit=self.config["tool_audit_ref"], runtime=lambda work: self)

    def validate_model_route(self, value):
        return copy.deepcopy(value)

    def budget_policy(self, config, continuation):
        assert continuation == {"mode": "fresh", "prior_receipt_ref": None}

    def create_runtime(self, config, route, route_ref, audit_ref):
        assert config == self.config and route == self.route
        assert route_ref == config["model_route_ref"] and audit_ref == config["tool_audit_ref"]
        return self

    def audit(self, workspace, audit):
        self.audits.append((str(workspace), audit))

    def command(self, continuation, report_path):
        assert continuation == {"mode": "fresh", "prior_receipt": None}
        return ["synthetic-process"]

    def identity(self):
        return {"backend": self.config["workflow_backend"], "requested_model": self.route["requested_model"]}

    def boundary(self, seconds, workspace, inputs):
        self.constructor_seconds = seconds
        self.created_monotonic = self.clock.monotonic()
        self.work_deadline_monotonic = self.created_monotonic + seconds
        self.wall_seconds, self.plan = seconds, None
        if self.case == "setup_exhausted":
            self.created_monotonic += 49
        return self

    def _verify(self):
        pass

    def launch(self, argv, **kwargs):
        self.launches.append({"argv": argv, "prompt": kwargs["stdin"].read(),
                              "cwd": kwargs["cwd"], "shell": kwargs["shell"],
                              "start_new_session": kwargs["start_new_session"]})
        if self.case == "launch_failed":
            raise OSError(5, "synthetic launch failure")
        kwargs["stdout"].write(b'{"synthetic":true}\n')
        kwargs["stderr"].write(b"synthetic stderr\n")
        return self

    def wait(self, *, timeout):
        self.waits.append(timeout)
        self.clock.now += 2
        if self.case == "timeout":
            raise subprocess.TimeoutExpired("synthetic-process", timeout)
        return 1 if self.case == "nonzero" else 0

    def settle(self, process, attempt):
        self.clock.now += 0.25
        if self.case == "settlement_error":
            raise RuntimeError("synthetic settlement observer failure")
        if self.case == "concurrent_unsettled":
            model.put(Path(self.config["state_root"]) / "STOP_UNSETTLED",
                      {"invocation_id": "another", "reason": "synthetic unresolved operation"})
        return {"settled": self.case != "unsettled"}

    def observation(self):
        if self.case == "inspection_error":
            raise ValueError("synthetic event parse failure")
        codex = self.config["workflow_backend"] == "codex"
        return {"identity_valid": True,
                "reported_models": [] if codex else [self.route["requested_model"]],
                "reported_tools": None if codex else ["Bash"],
                "terminal": "turn.completed", "report_text": "null",
                "protocol_errors": ["synthetic invalid event"] if self.case == "invalid_events" else []}

    def inspect_codex_events(self, path, *, report_path):
        return self.observation()

    def inspect_claude_events(self, path, *, model_route):
        return self.observation()

    def completed(self, observed):
        return observed["terminal"] == "turn.completed" and not observed["protocol_errors"]


class PlannerModelPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = None
        source = os.environ.get("GEAK_EVO_OCT15_SOURCE")
        if source:
            path = Path(source)
            if path.is_dir():
                path /= "bridge/planner_model_backend.py"
            spec = importlib.util.spec_from_file_location("oct15_planner_model_backend", path)
            cls.original = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.original)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.boot = SystemClock().boot_id()
        self.work, self.read_root, self.state = [self.root / name for name in ("work", "inputs", "state")]
        for path in (self.work, self.read_root):
            path.mkdir(mode=0o700)
        self.route = codex_route()
        model.put(self.root / "route.json", self.route)
        model.put(self.root / "audit.json", {"synthetic": True})
        self.clock_record = {"boot_id": self.boot, "started_monotonic_ns": 100_000_000_000,
                             "wall_seconds": 100, "started_unix": 1000}
        model.put(self.root / "clock.json", self.clock_record)
        self.config = {"schema": "geak-standalone-model-config-v1", "launch_admitted": True,
                       "workflow_backend": "codex", "clock_path": str(self.root / "clock.json"),
                       "work_root": str(self.work), "read_roots": [str(self.read_root)],
                       "state_root": str(self.state), "model_route_ref": model.ref(self.root / "route.json"),
                       "tool_audit_ref": model.ref(self.root / "audit.json")}
        self.request = {"schema": "geak-standalone-model-request-v1", "invocation_id": "episode.1",
                        "prompt": "Exact planner prompt: λ\nDo not trim.\n", "label": "planner",
                        "cwd": str(self.work), "model": "gpt-6-astra", "timeout_ms": 50_000}

    def modules(self):
        return (model,) if self.original is None else (self.original, model)

    def test_default_claude_route_validation_needs_no_runtime(self):
        from geak_evo.engineer import backend
        route = {"schema": "geak-native-model-route-v1", "workflow_backend": "claude",
                 "wire_protocol": "anthropic_messages", "requested_model": "test-model",
                 "accepted_reported_models": ["test-model"]}
        path = self.root / "claude-route.json"
        model.put(path, route)
        config = dict(self.config, workflow_backend="claude", model_route_ref=model.ref(path))
        actual = model.resolve_model_route(config)
        self.assertEqual(actual, (route, config["model_route_ref"]))
        if self.original is not None:
            with mock.patch.object(self.original.importlib, "import_module", return_value=backend) as load:
                self.assertEqual(actual, self.original.resolve_model_route(config))
            load.assert_called_once_with("geak_evo.claude_transport")

    def test_existing_model_state_must_remain_private(self):
        for module in self.modules():
            with self.subTest(module=module.__name__):
                self.state.mkdir(mode=0o755, exist_ok=True)
                self.state.chmod(0o755)
                services = SyntheticServices(self.config, self.route, FixedClock(self.boot))
                with self.assertRaisesRegex(ValueError, "model state must be private"):
                    module.run(self.config, self.request, dependencies=services.dependencies)
                self.assertEqual(services.launches, [])
                self.assertFalse((self.state / "attempts").exists())
                self.state.rmdir()

    def execute(self, module, *, backend="codex", case="completed"):
        if self.state.exists():
            shutil.rmtree(self.state)
        self.config["workflow_backend"] = backend
        current_backend = self.route.get("backend", self.route.get("workflow_backend"))
        if current_backend != backend:
            self.route = codex_route() if backend == "codex" else {
                "schema": "geak-native-model-route-v1", "workflow_backend": "claude",
                "wire_protocol": "anthropic_messages", "requested_model": "gpt-6-astra",
                "requested_reasoning_effort": "ultracode", "accepted_reported_models": ["gpt-6-astra"]}
            route_path = self.root / (backend + "-route.json")
            if not route_path.exists():
                model.put(route_path, self.route)
            self.config["model_route_ref"] = model.ref(route_path)
        clock = FixedClock(self.boot)
        services = SyntheticServices(self.config, self.route, clock, case)
        runtime = model.ModelRuntime(*services.dependencies, clock=clock)
        result, error = None, None
        try:
            if module is model:
                result = module.run(self.config, self.request, runtime=runtime)
            else:
                with mock.patch.object(module, "time", clock):
                    result = module.run(self.config, self.request, dependencies=services.dependencies)
        except Exception as exc:
            error = (type(exc).__name__, str(exc))
        files = {str(path.relative_to(self.state)): path.read_bytes()
                 for path in self.state.rglob("*") if path.is_file()}
        return {"result": result, "error": error, "files": files}, services, runtime

    def assert_same_records(self, original, port):
        expected = copy.deepcopy(original)
        for name in ("intent.json", "receipt.json"):
            key = "attempts/episode.1/" + name
            if key in expected["files"]:
                record = json.loads(expected["files"][key])
                self.assertNotIn("model_profile", record)
                # New calls add exactly the host's explicit profile; all other
                # saved fields and bytes still compare with the original.
                record["model_profile"] = self.expected_profile()
                record["model_backend_ref"] = model.ref(model.__file__)
                expected["files"][key] = model.encoded(record) + b"\n"
        if expected["result"] is not None:
            expected["result"]["receipt_ref"]["sha256"] = hashlib.sha256(
                expected["files"]["attempts/episode.1/receipt.json"]).hexdigest()
        self.assertEqual(expected, port)

    def expected_profile(self):
        reference = self.config["model_route_ref"]
        return {"workflow_backend": self.config["workflow_backend"],
                "requested_model": self.route["requested_model"],
                "requested_reasoning_effort": self.route["requested_reasoning_effort"],
                "route_path": reference["path"], "route_sha256": reference["sha256"]}

    def assert_profile_records(self, result):
        for name in ("intent.json", "receipt.json"):
            key = "attempts/episode.1/" + name
            if key in result["files"]:
                record = json.loads(result["files"][key])
                self.assertEqual(record["model_profile"], self.expected_profile())
                self.assertEqual(record["model_route_ref"], self.config["model_route_ref"])
                self.assertEqual(record["model_backend_ref"], model.ref(model.__file__))
        if result["result"] is not None:
            self.assertEqual(result["result"]["receipt_ref"],
                             model.ref(self.state / "attempts/episode.1/receipt.json"))

    def test_completed_and_failed_lifecycles_preserve_source_records(self):
        cases = ("completed", "nonzero", "timeout", "launch_failed", "invalid_events",
                 "inspection_error", "unsettled", "settlement_error", "concurrent_unsettled",
                 "setup_exhausted")
        for backend in ("claude", "codex"):
            for case in cases:
                with self.subTest(backend=backend, case=case):
                    source = self.execute(self.original, backend=backend, case=case)[0] if self.original else None
                    port, services, _ = self.execute(model, backend=backend, case=case)
                    self.assert_profile_records(port)
                    if source is not None:
                        self.assert_same_records(source, port)
                    if case == "completed":
                        self.assertIsNone(port["error"])
                        self.assertEqual(port["result"]["text"], "null")
                        self.assertTrue(port["result"]["settled"])
                        self.assertEqual(services.launches[0], {
                            "argv": ["synthetic-process"], "prompt": self.request["prompt"].encode(),
                            "cwd": str(self.work), "shell": False, "start_new_session": True})
                        self.assertEqual(services.waits, [80])
                    else:
                        self.assertIsNone(port["result"])
                        self.assertIsNotNone(port["error"])
                    if case in ("unsettled", "settlement_error", "concurrent_unsettled"):
                        self.assertIn("STOP_UNSETTLED", port["files"])

    def test_no_replay_and_unsettled_fence_do_not_launch_again(self):
        for case, invocation in (("completed", "episode.1"), ("unsettled", "episode.2"),
                                 ("setup_exhausted", "episode.1")):
            with self.subTest(case=case):
                _, services, runtime = self.execute(model, case=case)
                before = len(services.launches)
                request = dict(self.request, invocation_id=invocation)
                with self.assertRaises((ValueError, FileExistsError)):
                    model.run(self.config, request, runtime=runtime)
                self.assertEqual(len(services.launches), before)

    def test_request_validation_and_route_integrity_match_source(self):
        invalid = ({"invocation_id": "../escape"}, {"timeout_ms": True}, {"timeout_ms": 4999},
                   {"timeout_ms": 14_400_001}, {"timeout_ms": float("nan")}, {"prompt": ""},
                   {"model": "other-model"}, {"cwd": str(self.read_root)}, {"label": "x" * 513},
                   {"extra": "field"})
        for module in self.modules():
            services = SyntheticServices(self.config, self.route, FixedClock(self.boot))
            self.assertEqual(module.validate_request(self.config, self.request, transport=services),
                             (self.work, self.work))
            for update in invalid:
                with self.subTest(module=module.__name__, update=update), self.assertRaises(ValueError):
                    module.validate_request(self.config, dict(self.request, **update), transport=services)
            wrong = copy.deepcopy(self.config)
            wrong["model_route_ref"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "reference changed"):
                module.validate_request(wrong, self.request, transport=services)

    def test_budget_caps_and_boot_identity_match_source(self):
        cases = ((110_000_000_000, 50_000, self.boot, 50),
                 (160_000_000_000, 50_000, self.boot, 10),
                 (165_000_000_000, 50_000, self.boot, 5),
                 (165_000_000_001, 50_000, self.boot, None),
                 (110_000_000_000, 50_000, "wrong-boot", None))
        for module in self.modules():
            for now, timeout, boot, expected in cases:
                with self.subTest(module=module.__name__, now=now, boot=boot):
                    request = dict(self.request, timeout_ms=timeout)
                    if expected is None:
                        with self.assertRaises(ValueError):
                            module.remaining_budget(self.config, request, now_ns=now, boot_id=boot)
                    else:
                        self.assertEqual(module.remaining_budget(self.config, request, now_ns=now, boot_id=boot),
                                         (expected, model.ref(self.root / "clock.json")))
        with self.assertRaisesRegex(ValueError, "running experiment"):
            model.remaining_budget(self.config, self.request, clock=FixedClock("wrong-boot"))

    def test_json_bytes_and_invalid_json_match_source(self):
        value = {"unicode": "λ", "number": 1.25, "null": None, "list": [False, 2]}
        for module in self.modules():
            self.assertEqual(module.encoded(value), model.encoded(value))
            for raw in (b'{"duplicate":1,"duplicate":2}', b'{"x":NaN}', b'{"x":Infinity}'):
                with self.subTest(module=module.__name__, raw=raw), self.assertRaises(ValueError):
                    module.decode(raw)
            destination = self.root / (module.__name__ + ".json")
            module.put(destination, value)
            self.assertEqual(destination.read_bytes(), model.encoded(value) + b"\n")
            with self.assertRaises(FileExistsError):
                module.put(destination, value)

    def test_explicit_runtime_and_original_dependencies_tuple_are_supported(self):
        _, services, runtime = self.execute(model)
        self.assertEqual(model.imports(runtime=runtime), services.dependencies)
        self.assertEqual(model.codex_imports(self.config, runtime=runtime), services.dependencies)
        self.assertEqual(model.resolve_model_route(self.config, runtime=runtime),
                         (self.route, self.config["model_route_ref"]))
        with self.assertRaisesRegex(ValueError, "explicit model runtime"):
            model.imports()
        shutil.rmtree(self.state)
        clock = FixedClock(self.boot)
        services = SyntheticServices(self.config, self.route, clock)
        with mock.patch("geak_evo.clock.SystemClock", return_value=clock):
            result = model.run(self.config, self.request, dependencies=services.dependencies)
        self.assertTrue(result["settled"])
        if self.original is not None:
            for name in ("need", "encoded", "decode", "read", "put", "ref", "canonical_dir",
                         "workflow_backend", "codex_imports", "resolve_model_route", "validate_request",
                         "remaining_budget", "imports", "run"):
                old = inspect.signature(getattr(self.original, name)).parameters
                new = inspect.signature(getattr(model, name)).parameters
                self.assertEqual({key: new[key] for key in old}, dict(old), name)
        self.assertFalse(hasattr(model, "main"), "the port does not publish a model CLI")


if __name__ == "__main__":
    unittest.main()
