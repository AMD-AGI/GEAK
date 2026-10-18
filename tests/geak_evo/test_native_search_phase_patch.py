"""Exercise the prospective publisher projection at its real sealing boundary.

The retained Publisher class is compiled without importing its runtime. The real
constructor, state-authority checks and publish method run against synthetic
inputs. Filesystem/clock custody, dispatch reconstruction and the sealing writer
are doubles; no root install, model, device or helper enforcement is exercised.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


def need(condition, message):
    if not condition:
        raise ValueError(message)


class NativeSearchPhasePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repository = Path(__file__).resolve().parents[2]
        base = json.loads((repository / "docs/evo/native-search-phase-base.json").read_bytes())
        if not all(Path(row["source_path"]).is_file() for row in base["files"]):
            raise unittest.SkipTest("retained publisher source is not installed in this CPU checkout")
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for row in base["files"]:
            raw = Path(row["source_path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise AssertionError("retained publisher differs from the pinned patch base")
            target = root / row["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        subprocess.run(["/usr/bin/git", "apply", str(repository / "docs/evo/native-search-phase.patch")],
                       cwd=root, check=True, capture_output=True)
        for row in base["files"]:
            raw = (root / row["name"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["patched_sha256"]:
                raise AssertionError("patched publisher differs from the registered result")
        cls.source_path = root / "bridge/finite_publisher.py"
        tree = ast.parse(cls.source_path.read_bytes(), filename=str(cls.source_path))
        publisher = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == "Publisher")
        cls.code = compile(ast.Module(body=[publisher], type_ignores=[]), str(cls.source_path), "exec")

    def setUp(self):
        self.seal = mock.Mock(return_value={"fixture": "sealed"})
        self.trusted = mock.Mock(side_effect=lambda path, **kwargs: Path(path))
        self.uid = mock.Mock(return_value=0)
        namespace = {"copy": copy, "math": math, "Path": Path, "need": need, "re": re, "json": json,
                     "os": SimpleNamespace(geteuid=self.uid), "trusted_path": self.trusted,
                     "publish_sealed": self.seal}
        exec(self.code, namespace)
        self.namespace = namespace
        self.Publisher = namespace["Publisher"]
        self.publisher = object.__new__(namespace["Publisher"])
        self.publisher.used = set()
        self.publisher.sealed_root = Path("/fixture/sealed")
        self.publisher.root_state_root = Path("/fixture/owner-state")
        self.publisher.clock = {"started_unix": 1000, "wall_seconds": 9000,
                                "started_monotonic_ns": 100000, "session_id": "original-study"}
        self.publisher.shared_budget = {"session_id": "original-study", "deadline_ns": 9000000100000}
        self.publisher.binding = {"job_id": "task"}
        self.publisher.limits = {"wall_seconds": 3300, "evaluations": 24}
        self.path = self.publisher.sealed_root / "engineer-1"
        clock_ref = {"path": "/fixture/original-clock.json", "sha256": "a" * 64}
        config = {"store_root": "/fixture/store", "contract_id": "task",
                  "backend_config": {"clock_ref": clock_ref,
                                     "probe_config": {"measurement_epoch_id": "epoch"}}}
        config_ref = {"path": str(self.path / "host-config.json"),
                      "sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
        template = {"phase_deadline_seconds": 6505, "search_phase_deadline_seconds": 3270,
                    "search_deadline_seconds": 3300,
                    "study_id": "original-study", "arm_seconds": 9000, "contract_id": "task",
                    "host_config_refs": {"native": config_ref},
                    "arms": [{"name": "native", "measurement_epoch_id": "epoch",
                              "source_object_root": "/fixture/store/objects",
                              "study_clock_path": clock_ref["path"]}]}
        self.publisher.templates = {"admission": copy.deepcopy(template)}
        self.files = {"entry-input.json": {"turn_id": "engineer-1"}, "host-config.json": config,
                      "admission.json": copy.deepcopy(template),
                      "backend-binding.json": {"config_ref": config_ref},
                      "collection-config.json": {"measurement_epoch_id": "epoch"}}
        self.validated = copy.deepcopy(self.files)
        self.publisher.validate = mock.Mock(return_value=self.validated)

    def state(self, seconds=3300):
        return {"schema_version": 1, "phase": "dispatching", "binding": copy.deepcopy(self.publisher.binding),
                "limits": copy.deepcopy(self.publisher.limits),
                "deadline": self.publisher.clock["started_unix"] + seconds,
                "pending": {"operation_id": self.path.name}}

    def construct(self, state):
        """Real constructor, with filesystem/clock custody and store readers doubled."""
        docs = {"config": {}, "admission": self.publisher.templates["admission"], "entry": {}, "collection": {},
                "clock": self.publisher.clock}
        refs = {key: {"path": "/fixture/" + key, "sha256": "a" * 64} for key in docs}
        self.namespace.update(canonical=lambda path, **kwargs: Path(path),
            validate_policy=lambda policy, campaign: policy,
            _within=lambda path, root: path == root or root in path.parents,
            read_ref=lambda ref, **kwargs: json.dumps(docs[Path(ref["path"]).name]).encode())
        with mock.patch.object(self.Publisher, "_trust_templates"), \
             mock.patch.object(self.Publisher, "_clock") as clock_check, \
             mock.patch.object(self.Publisher, "_state", return_value=state):
            result = self.Publisher(SimpleNamespace(root=Path("/fixture/campaign/store")), "campaign",
                templates={key: refs[key] for key in ("config", "admission", "entry", "collection")},
                campaign_root="/fixture/campaign", sealed_root="/fixture/sealed", root_state_root="/fixture/owner-state",
                clock_ref=refs["clock"], shared_budget=self.publisher.shared_budget,
                common_read_roots=[], tools_manifest_ref=None)
        clock_check.assert_called_once_with()
        return result

    def test_projects_only_sealed_engineer_admission_after_validation(self):
        template_before = copy.deepcopy(self.publisher.templates)
        files_before = copy.deepcopy(self.files)
        clock_before = copy.deepcopy(self.publisher.clock)
        budget_before = copy.deepcopy(self.publisher.shared_budget)
        events = []

        def validate(sealed, files):
            self.assertEqual(sealed, self.path)
            self.assertIs(files, self.files)
            self.assertEqual(files, files_before)
            self.assertEqual(self.publisher.templates, template_before)
            events.append("validate")
            return self.validated

        def seal(sealed, files):
            self.assertEqual(events, ["validate"])
            self.assertIn(sealed.name, self.publisher.used)
            events.append("seal")
            return {"fixture": "sealed"}

        self.publisher.validate.side_effect = validate
        self.seal.side_effect = seal
        result = self.publisher.publish(self.path, self.files)
        expected = copy.deepcopy(files_before)
        expected["admission.json"]["phase_deadline_seconds"] = 3270
        self.seal.assert_called_once_with(self.path, expected)
        self.assertEqual(result, {"fixture": "sealed"})
        self.assertEqual(events, ["validate", "seal"])
        self.assertEqual(self.files, files_before)
        self.assertEqual(self.publisher.templates, template_before)
        self.assertEqual(self.publisher.clock, clock_before)
        self.assertEqual(self.publisher.shared_budget, budget_before)
        self.trusted.assert_called_once_with(self.publisher.sealed_root, directory=True)

    def test_absent_search_field_preserves_existing_files_even_without_phase(self):
        for inherited in (6505, None):
            with self.subTest(inherited=inherited):
                self.publisher.templates["admission"].pop("search_phase_deadline_seconds", None)
                self.publisher.templates["admission"].pop("search_deadline_seconds", None)
                self.validated["admission.json"].pop("search_phase_deadline_seconds", None)
                self.validated["admission.json"].pop("search_deadline_seconds", None)
                if inherited is None:
                    self.publisher.templates["admission"].pop("phase_deadline_seconds")
                    self.validated["admission.json"].pop("phase_deadline_seconds")
                before = copy.deepcopy(self.validated)
                self.publisher.publish(self.path, self.files)
                self.assertIs(self.seal.call_args.args[1], self.validated)
                self.assertEqual(self.validated, before)

    def test_accepts_equal_and_fractional_positive_search_cutoffs(self):
        self.publisher.templates["admission"].pop("search_deadline_seconds")
        for seconds in (6505, 6505.0, 0.5):
            with self.subTest(seconds=seconds):
                self.publisher.templates["admission"]["search_phase_deadline_seconds"] = seconds
                self.publisher.publish(self.path, self.files)
                self.assertEqual(self.seal.call_args.args[1]["admission.json"]["phase_deadline_seconds"], seconds)

    def test_rejects_invalid_or_widening_search_cutoffs_before_sealing(self):
        for seconds in (True, False, None, "3270", 0, -1, 6506, 10**400,
                        float("nan"), float("inf"), float("-inf")):
            with self.subTest(seconds=seconds):
                self.publisher.templates["admission"]["search_phase_deadline_seconds"] = seconds
                with self.assertRaisesRegex(ValueError, "search phase must"):
                    self.publisher.publish(self.path, self.files)
                self.assertEqual(self.publisher.used, set())
                self.seal.assert_not_called()
                self.trusted.assert_not_called()

    def test_requires_valid_inherited_phase_when_projection_is_requested(self):
        for inherited in (True, False, None, "6505", 0, -1,
                          float("nan"), float("inf"), float("-inf")):
            with self.subTest(inherited=inherited):
                self.publisher.templates["admission"]["phase_deadline_seconds"] = inherited
                with self.assertRaisesRegex(ValueError, "search phase requires"):
                    self.publisher.publish(self.path, self.files)
        self.publisher.templates["admission"].pop("phase_deadline_seconds")
        with self.assertRaisesRegex(ValueError, "search phase requires"):
            self.publisher.publish(self.path, self.files)
        self.assertEqual(self.publisher.used, set())
        self.seal.assert_not_called()

    def test_constructor_accepts_exact_search_deadline_and_preserves_original_clock(self):
        clock, budget = copy.deepcopy(self.publisher.clock), copy.deepcopy(self.publisher.shared_budget)
        result = self.construct(self.state())
        self.assertEqual(result._campaign_deadline(), 4300)
        self.assertEqual(result.clock, clock)
        self.assertEqual(result.shared_budget, budget)
        for seconds in (3270, 3301, 9000):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, "campaign deadline differs"):
                self.construct(self.state(seconds))

    def test_constructor_without_state_cutoff_keeps_original_deadline(self):
        self.publisher.templates["admission"].pop("search_deadline_seconds")
        self.assertEqual(self.construct(self.state(9000))._campaign_deadline(), 10000)
        self.publisher.templates["admission"].pop("search_phase_deadline_seconds")
        self.publisher.templates["admission"].pop("phase_deadline_seconds")
        self.assertEqual(self.construct(self.state(9000))._campaign_deadline(), 10000)
        with self.assertRaisesRegex(ValueError, "campaign deadline differs"):
            self.construct(self.state())

    def test_state_cutoff_requires_valid_ordered_bounds_within_original_wall(self):
        admission = self.publisher.templates["admission"]
        for seconds in (True, False, None, "3300", 0, -1, 3269, 6506,
                        float("nan"), float("inf"), float("-inf")):
            with self.subTest(seconds=seconds):
                admission["search_deadline_seconds"] = seconds
                with self.assertRaisesRegex(ValueError, "search (deadline|phase)"):
                    self.construct(self.state())
        admission["search_deadline_seconds"] = 3300
        self.publisher.clock["wall_seconds"] = 3299
        with self.assertRaisesRegex(ValueError, "search deadline"):
            self.construct(self.state())
        self.publisher.clock["wall_seconds"] = 9000
        admission.pop("search_phase_deadline_seconds")
        with self.assertRaisesRegex(ValueError, "native search phase"):
            self.construct(self.state())

    def test_real_validate_rechecks_state_deadline_binding_and_limits(self):
        self.namespace.update(NAMES=tuple(self.files), canonical=lambda path, **kwargs: Path(path),
            encoded=lambda value: json.dumps(value, sort_keys=True, allow_nan=False).encode())
        class AuthorityPassed(Exception):
            pass
        self.publisher._clock = mock.Mock(side_effect=AuthorityPassed)
        for seconds in (3300, 3270, 9000):
            self.publisher._state = mock.Mock(return_value=self.state(seconds))
            expected = AuthorityPassed if seconds == 3300 else ValueError
            with self.subTest(seconds=seconds), self.assertRaises(expected):
                self.Publisher.validate(self.publisher, self.path, self.files)
        for field in ("binding", "limits"):
            state = self.state()
            state[field] = {"changed": True}
            self.publisher._state.return_value = state
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "campaign authority changed"):
                self.Publisher.validate(self.publisher, self.path, self.files)
        self.assertEqual(self.publisher._clock.call_count, 1)
        self.publisher.templates["admission"].pop("search_deadline_seconds")
        self.publisher._state.return_value = self.state(9000)
        with self.assertRaises(AuthorityPassed):
            self.Publisher.validate(self.publisher, self.path, self.files)
        self.publisher._state.return_value = self.state()
        with self.assertRaisesRegex(ValueError, "campaign authority changed"):
            self.Publisher.validate(self.publisher, self.path, self.files)
        self.seal.assert_not_called()

    def test_validation_failure_prevents_projection_or_sealing(self):
        self.publisher.validate.side_effect = ValueError("native files differ from fixed templates")
        before = copy.deepcopy(self.validated)
        with self.assertRaisesRegex(ValueError, "native files differ"):
            self.publisher.publish(self.path, self.files)
        self.assertEqual(self.validated, before)
        self.assertEqual(self.publisher.used, set())
        self.seal.assert_not_called()

    def test_root_requirement_still_precedes_validation(self):
        self.uid.return_value = 1000
        with self.assertRaisesRegex(ValueError, "only finite root coordinator may seal"):
            self.publisher.publish(self.path, self.files)
        self.publisher.validate.assert_not_called()
        self.seal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
