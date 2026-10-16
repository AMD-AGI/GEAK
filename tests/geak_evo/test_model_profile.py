"""CPU-only proof that new calls share an explicit profile and recovery preserves it."""
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import test_engineer_port as engineer_fixture
import test_planner_model_port as planner_fixture
from geak_evo.model_profile import ModelProfile
from geak_evo.planner import model, native
from geak_evo.selection import final


class ModelProfileTests(unittest.TestCase):
    def setUp(self):
        self.planner = planner_fixture.PlannerModelPortTests()
        self.planner.setUp()
        self.addCleanup(self.planner.doCleanups)
        p = self.planner
        self.profile = ModelProfile.from_route(p.route, p.config["model_route_ref"], workflow_backend="codex")
        self.routes = SimpleNamespace(validate=lambda value: value, __file__="/fixture/responses_route.py")
        self.runtime = SimpleNamespace(routes=self.routes, transport=object())

    def test_same_profile_reaches_planner_backend_final_verifier_and_engineer_command(self):
        p = self.planner
        self.assertEqual(native.planner_model_profile(p.config, runtime=self.runtime), self.profile)
        config = dict(p.config, codex_native_root="/fixture")
        self.assertEqual(final._model_identity(config, lambda ref: Path(ref["path"]).read_bytes(), self.runtime),
                         (self.profile.requested_model, self.profile.model_route_ref))
        result, services, _ = p.execute(model)
        self.assertIsNone(result["error"])
        receipt = json.loads(result["files"]["attempts/episode.1/receipt.json"])
        self.assertEqual(receipt["model_profile"], self.profile.as_dict())
        self.assertEqual(receipt["requested_reasoning_effort"], "xhigh")
        self.assertEqual(len(services.launches), 1)
        command = engineer_fixture.codex_command({"continuation": {"mode": "fresh"}}, p.root / "report",
                                                  model_profile=self.profile)
        self.assertEqual(command[command.index("--model") + 1], "gpt-6-astra")
        self.assertIn('model_reasoning_effort="xhigh"', command)

    def test_no_implicit_new_call_and_pinned_route_drift_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "explicit pinned profile"):
            native.planner_model_profile({"workflow_backend": "claude"})
        self.assertEqual(native.planner_model_identity({"workflow_backend": "claude"}),
                         ("claude-opus-5", None))  # Historical read only.
        route = dict(self.planner.route, requested_reasoning_effort=None)
        with self.assertRaisesRegex(ValueError, "explicit model and reasoning"):
            ModelProfile.from_route(route, self.profile.model_route_ref, workflow_backend="codex")
        for backend in (None, "claude"):
            route = dict(self.planner.route)
            if backend is None:
                del route["backend"]
            else:
                route["backend"] = backend
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "backend differs"):
                ModelProfile.from_route(route, self.profile.model_route_ref, workflow_backend="codex")
        for backend in ("codex", "claude"):
            with self.subTest(conflicting_field=backend), self.assertRaisesRegex(ValueError, "backend differs"):
                ModelProfile.from_route(dict(self.planner.route, workflow_backend=backend),
                                        self.profile.model_route_ref, workflow_backend="codex")
        with self.assertRaisesRegex(ValueError, "backend differs"):
            ModelProfile.from_route(self.planner.route, self.profile.model_route_ref, workflow_backend="claude")
        Path(self.profile.route_path).write_text(json.dumps(dict(self.planner.route, requested_model="changed")))
        with self.assertRaises(ValueError):
            engineer_fixture.codex_command({"continuation": {"mode": "fresh"}}, self.planner.root / "report",
                                           model_profile=self.profile)

    def test_historical_route_without_backend_remains_readable(self):
        p = self.planner
        route = dict(p.route)
        del route["backend"]
        model.put(p.root / "historical-route.json", route)
        reference = model.ref(p.root / "historical-route.json")
        config = dict(p.config, codex_native_root="/fixture", model_route_ref=reference)
        self.assertEqual(final._model_identity(config, lambda ref: Path(ref["path"]).read_bytes(), self.runtime),
                         (self.profile.requested_model, reference))

    @unittest.skipUnless(os.environ.get("GEAK_EVO_CODEX_ROUTE_SOURCE"), "retained public route validator not supplied")
    def test_retained_codex_validator_accepts_exact_route_and_preserves_identity(self):
        source = Path(os.environ["GEAK_EVO_CODEX_ROUTE_SOURCE"])
        spec = importlib.util.spec_from_file_location("retained_responses_route", source)
        routes = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(routes)
        original = Path(self.profile.route_path).read_bytes()
        route = routes.validate(json.loads(original))
        identity = routes.route_id(route)
        profile = ModelProfile.from_route(route, self.profile.model_route_ref, workflow_backend="codex")
        runtime = SimpleNamespace(routes=routes)
        self.assertEqual(native.planner_model_profile(self.planner.config, runtime=runtime), profile)
        self.assertEqual(routes.route_id(profile.verify()), identity)
        self.assertEqual(Path(profile.route_path).read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "unexpected public route fields"):
            routes.validate(dict(route, workflow_backend="codex"))

    def test_recovery_retains_original_profile_without_launching_and_resume_rejects_change(self):
        e = engineer_fixture.EngineerTests()
        e.setUp()
        self.addCleanup(e.doCleanups)
        result = e.execute(e.events(), settled=False)
        original = Path(result["receipt_ref"]["path"]).read_bytes()
        recovered = engineer_fixture.recover_turn(result["receipt_ref"], capture_tip=e.capture,
            verify_recovery=lambda *args: {"settled": True, "process_reaped": True, "scope": "synthetic CPU settlement"})
        self.assertEqual(recovered["receipt"]["model_profile"], e.profile.as_dict())
        self.assertEqual(Path(result["receipt_ref"]["path"]).read_bytes(), original)
        resume = e.continue_from(recovered, "resume")
        different = copy.deepcopy(e.profile.as_dict())
        different["route_path"] = self.profile.route_path
        different["route_sha256"] = self.profile.route_sha256
        with self.assertRaisesRegex(ValueError, "original model profile"):
            engineer_fixture.codex_command(resume, e.root / "report", model_profile=ModelProfile(**different))


if __name__ == "__main__":
    unittest.main()
