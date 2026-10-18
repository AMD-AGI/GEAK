"""Compare real operation production with the retained publisher's seal recipe.

Templates, source objects, requests, pins and copied context are real temporary
files. The producer and retained _public/_expected methods run unchanged. Only
the clock and Codex descriptor factory are synthetic inputs. This does not run
the root coordinator, qualify worker custody, launch a model or evaluate code.
"""

import ast
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from geak_evo.archive.refs import file_ref, read_ref
from geak_evo.engineer import instructions, operations
from geak_evo.engineer.investigation import encoded, read
from geak_evo.engineer.request import validate_request

import test_engineer_port as base


def differences(left, right, path=""):
    """Report leaf paths so transitive reference changes remain visible."""
    if type(left) is dict and type(right) is dict:
        result = []
        for key in sorted(set(left) | set(right)):
            child = path + "/" + key
            result.extend(differences(left[key], right[key], child)
                          if key in left and key in right else [child])
        return result
    if type(left) is list and type(right) is list and len(left) == len(right):
        return [child for index, (a, b) in enumerate(zip(left, right))
                for child in differences(a, b, path + "/" + str(index))]
    return [] if left == right else [path]


def compile_members(path, names, namespace):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    members = [node for node in tree.body if getattr(node, "name", None) in names
               or (isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id in names
                           for target in node.targets))]
    exec(compile(ast.Module(body=members, type_ignores=[]), str(path), "exec"), namespace)


class PublisherOperationContractTests(unittest.TestCase):
    capture = base.EngineerTests.capture

    @classmethod
    def setUpClass(cls):
        repository = Path(__file__).resolve().parents[2]
        registered = json.loads((repository / "docs/evo/native-search-phase-base.json").read_bytes())
        publisher = next(row for row in registered["files"] if row["name"] == "bridge/finite_publisher.py")
        original = Path(publisher["source_path"])
        if not original.is_file():
            raise unittest.SkipTest("retained publisher patch base is not installed")
        if hashlib.sha256(original.read_bytes()).hexdigest() != publisher["sha256"]:
            raise AssertionError("retained publisher differs from registered patch base")
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        patched = Path(temporary.name) / publisher["name"]
        patched.parent.mkdir(parents=True)
        patched.write_bytes(original.read_bytes())
        subprocess.run(["/usr/bin/git", "apply", str(repository / "docs/evo/native-search-phase.patch")],
                       cwd=temporary.name, check=True, capture_output=True)
        if hashlib.sha256(patched.read_bytes()).hexdigest() != publisher["patched_sha256"]:
            raise AssertionError("publisher patch differs from registered result")
        contract = json.loads((repository / "docs/evo/publisher-contract-base.json").read_bytes())
        contract_row = next(row for row in contract["files"] if row["name"] == publisher["name"])
        if hashlib.sha256(patched.read_bytes()).hexdigest() != contract_row["sha256"]:
            raise AssertionError("publisher contract has a different composed patch base")
        subprocess.run(["/usr/bin/git", "apply", "--include=bridge/finite_publisher.py",
                        str(repository / "docs/evo/publisher-contract.patch")],
                       cwd=temporary.name, check=True, capture_output=True)
        if hashlib.sha256(patched.read_bytes()).hexdigest() != contract_row["patched_sha256"]:
            raise AssertionError("publisher contract differs from registered result")
        native = original.parent.parent
        dependencies = {
            "bridge/native_investigation.py": (
                "d1d156e2223baf0dad187b417b13a347e8fca9e2c4f9e3c008a31c37357dd0c0",
                {"need", "encoded"}),
            "bridge/operation_files.py": (
                "1f0b6817f3a6ac090b89f037a9452b1a5747c98cc9c21e2b94f0544669307bcd",
                {"NAMES", "native_only_mode", "future_ref"}),
            "evolution/src/geak_evo/search_policy.py": (
                "6f50511ad58ffb69e401ddb36813409156cf32359cb0bb3927bd9e36c7eab2df",
                {"native_capability_context"}),
        }
        namespace = dict(copy=copy, contextlib=contextlib, hashlib=hashlib, json=json,
                         os=os, Path=Path, re=re, shlex=shlex, stat=stat)
        for relative, (digest, names) in dependencies.items():
            path = native / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise AssertionError("retained dependency differs: " + relative)
            compile_members(path, names, namespace)
        compile_members(patched, {"canonical", "read_ref", "_within", "Publisher"}, namespace)
        cls.Publisher = namespace["Publisher"]

    def setUp(self):
        base.EngineerTests.setUp(self)
        self.study = self.root / "study"
        self.campaign = self.study / "campaign"
        self.campaign.mkdir(parents=True)
        self.common = self.root / "common"
        self.common.mkdir()
        self.note.rename(self.common / "note.md")
        self.note = self.common / "note.md"
        self.request["read_only"] = [file_ref(self.note), self.request["tip"]["manifest"]]
        self.request["budget"]["wall_seconds"] = 900
        self.request = validate_request(self.request)
        self.runtime = self.root / "runtime"
        (self.runtime / "host").mkdir(parents=True)
        for name in ("root_helper.py", "host_driver.py"):
            (self.runtime / name).write_text("# CPU fixture; never executed.\n")
        self.initial = self.root / "initial-workspace"
        self.initial.mkdir()
        self.clock = {"boot_id": "cpu-fixture-boot", "session_id": "original-study",
                      "started_monotonic_ns": 100_000_000_000, "started_unix": 1000,
                      "wall_seconds": 9000}
        self.clock_ref = self.save(self.root / "clock.json", self.clock)
        self.shared_budget = {"boot_id": self.clock["boot_id"], "session_id": self.clock["session_id"],
                              "deadline_ns": 9_100_000_000_000, "root": str(self.root / "budget")}
        self.fake_clock = SimpleNamespace(boot_id=lambda: self.clock["boot_id"],
                                         monotonic_ns=lambda: 200_000_000_000)
        self.deadline = 1_100_000_000_000
        self.pins = {"workspace": str(self.workspace), "initial_workspace": str(self.initial),
                     "idea_id": self.request["direction_id"], "job_id": self.job_id,
                     "measurement_epoch_id": self.epoch, "session_id": self.clock["session_id"],
                     "starting_source_id": self.request["tip"]["source_id"],
                     "patch_base_source_id": self.request["tip"]["source_id"],
                     "source_objects_root": str(self.store.root / "objects"), "source_paths": self.paths}
        self.workspace_ref = self.save(self.root / "workspace-pins.json", self.pins)
        self.prompt = self.root / "outer-assignment.txt"
        self.prompt.write_text("Use the exact source manifest and measured evidence.\n")
        self.common_read_roots = [str(self.common), str(self.store.root / "objects")]
        self.policy = {"arm": "EVO", "study_root": str(self.study),
                       "shared_read_roots": self.common_read_roots,
                       "native_capabilities": ["continuation", "program_lease"],
                       "program_lease_max_episodes": 2}
        self.parents = [{"candidate_id": self.request["tip"]["candidate_id"],
                         "source": self.request["tip"]["manifest"],
                         "evidence": [file_ref(self.note)], "note": "Retain exact source α."}]
        self.proposal = {"direction_id": self.request["direction_id"], "expected_revision": 0,
                         "parent_candidate_id": self.request["tip"]["candidate_id"],
                         "island_id": "unassigned", "engineer_id": "engineer-1",
                         "question": self.request["next_question"], "reason": "Check exact contract",
                         "evidence_uptake": "Read pinned source", "premises": [], "repeat_delta": None,
                         "exchange_ref": None, "protection_ref": None, "override_reason": None,
                         "membership_ref": None, "reservation": {"wall_seconds": 900, "evaluations": 2}}

    def save(self, path, value):
        path.write_bytes(encoded(value))
        return file_ref(path)

    def make_templates(self, *, native_only=True, codex=True):
        directory = self.root / "templates"
        directory.mkdir()
        audit = self.save(self.runtime / "audit.json", {"fixture": "seed audit"})
        model = {"workflow_backend": "codex", "tool_audit_ref": audit}
        if codex:
            model["codex_runtime_family_ref"] = self.save(self.runtime / "family.json", {"fixture": "family"})
        config = {"native_only": native_only, "imports": {"host_root": str(self.runtime / "host")},
                  "hardware_profile": None if native_only else "hardware",
                  "hardware_evaluation_root": None if native_only else "/template/hardware",
                  "helper_ref": file_ref(self.runtime / "root_helper.py"), "tool_audit_ref": audit,
                  "contract_ref": file_ref(self.job_path), "contract_id": self.job_id,
                  "artifact_root": str(self.artifacts),
                  "backend_config": {"model_config": model,
                                     "probe_config": {"native_only": native_only,
                                                      "max_calls_per_turn": 24,
                                                      "native_envelope_seconds": 1500}}}
        config_ref = self.save(directory / "config.json", config)
        admission = {"schema": "geak-dense-native-admission-v1", "native_only": native_only,
                     "phase_deadline_seconds": 6505, "search_phase_deadline_seconds": 3270,
                     "search_deadline_seconds": 3300, "prospective_unsealed": False,
                     "arms": [{"name": "native"}]}
        if native_only:
            admission["host_config_refs"] = {"native": config_ref}
        else:
            admission["arms"].append({"name": "hardware"})
            admission["diagnostic"] = {"profiles": ["hardware"],
                                       "host_config_refs": {name: config_ref for name in ("native", "hardware")}}
        return {"config": config_ref,
                "admission": self.save(directory / "admission.json", admission),
                "entry": self.save(directory / "entry.json", {"protocol": "geak-one-engineer-v1",
                    "direction": {"id": "template", "title": "template", "prompt": "template"},
                    "tool_addendum": "Old template instructions must not survive."}),
                "collection": self.save(directory / "collection.json", {"fixture": "collection template"})}

    def build_pair(self, *, native_only=True, codex=True, reservation=2):
        templates = self.make_templates(native_only=native_only, codex=codex)
        self.proposal["reservation"]["evaluations"] = reservation
        op = "engineer-operation-1"
        sealed = self.root / "sealed" / op
        descriptor_calls = []

        def descriptor_factory(model_config, **kwargs):
            descriptor_calls.append(copy.deepcopy(kwargs))
            return {"workspace": str(kwargs["workspace"]), "read_roots": kwargs["read_roots"],
                    "operational_roots": kwargs["operational_roots"], "fixture": "CPU descriptor"}

        produced = operations.build_files(templates=templates, request=self.request, operation_id=op,
            prompt_ref=file_ref(self.prompt), workspace_ref=self.workspace_ref, store_root=self.store.root,
            campaign_root=self.campaign, sealed_operation_root=sealed, clock_ref=self.clock_ref,
            turn_deadline_ns=self.deadline, shared_budget=self.shared_budget,
            common_read_roots=self.common_read_roots, parents=self.parents, search_policy=self.policy,
            root_state_root=self.root / "root-state", common_tool_addendum="Common pilot tools.\n",
            proposal=self.proposal, descriptor_factory=descriptor_factory, clock=self.fake_clock)
        publisher = object.__new__(self.Publisher)
        publisher.templates = {name: read(ref) for name, ref in templates.items()}
        publisher.template_refs = templates
        publisher.campaign = self.campaign
        publisher.store_root = self.store.root
        publisher.clock = self.clock
        publisher.clock_ref = self.clock_ref
        publisher.shared_budget = self.shared_budget
        publisher.common_read_roots = self.common_read_roots
        publisher.search_policy = self.policy
        publisher.parent_contract = SimpleNamespace(tool_addendum=instructions.tool_addendum)
        publisher.common_tool_addendum = "Common pilot tools.\n"
        publisher.root_state_root = self.root / "root-state"
        publisher.driver_ref = file_ref(self.runtime / "host_driver.py")
        public = self.campaign / "native-operations" / op / "readonly"
        allowed_objects = {path.name for path in (self.store.root / "objects").iterdir()}
        insights, parents = publisher._public(self.request, public, self.prompt.read_bytes(),
                                              allowed_objects, self.parents)
        expected = publisher._expected(sealed, op, self.request, self.pins, self.workspace_ref,
                                       self.deadline, insights, parents, self.proposal)
        self.assertEqual(len(descriptor_calls), int(codex))
        self.assertEqual(set(produced), set(operations.NAMES))
        self.assertEqual(set(expected), set(operations.NAMES))
        self.assertEqual(produced["entry-input.json"]["insights"], insights)
        self.assertEqual(produced["collection-config.json"]["parents"], parents)
        for ref in insights:
            read_ref(ref)
        self.publisher, self.public, self.allowed_objects = publisher, public, allowed_objects
        self.expected_args = (sealed, op, self.request, self.pins, self.workspace_ref,
                              self.deadline, insights, parents, self.proposal)
        return produced, expected

    def test_native_only_five_files_match_retained_seal_recipe(self):
        produced, expected = self.build_pair()
        self.assertEqual(differences(produced, expected), [])
        self.assertEqual(encoded(produced), encoded(expected))

    def test_diagnostic_five_files_match_retained_seal_recipe(self):
        produced, expected = self.build_pair(native_only=False)
        self.assertEqual(differences(produced, expected), [])
        self.assertEqual(encoded(produced), encoded(expected))

    def test_reservation_clamp_survives_both_recipes(self):
        produced, expected = self.build_pair(reservation=1)
        for files in (produced, expected):
            probe = files["host-config.json"]["backend_config"]["probe_config"]
            self.assertEqual(probe["max_calls_per_turn"], 1)
            self.assertIn("per-turn request ceiling is 1", files["entry-input.json"]["tool_addendum"])

    def test_retained_reader_rejects_changed_copied_prompt(self):
        self.build_pair()
        with (self.public / "outer-assignment.txt").open("ab") as stream:
            stream.write(b"unadmitted replacement text\n")
        with self.assertRaisesRegex(ValueError, "reference changed"):
            self.publisher._public(self.request, self.public, self.prompt.read_bytes(),
                                   self.allowed_objects, self.parents)

    def test_retained_recipe_rejects_changed_descriptor_scope(self):
        self.build_pair()
        descriptor = self.public.parent / "model" / "codex-runtime.json"
        value = json.loads(descriptor.read_bytes())
        value["read_roots"].append(str(self.root / "unadmitted"))
        descriptor.chmod(0o600)
        descriptor.write_bytes(encoded(value))
        descriptor.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "derived descriptor differs"):
            self.publisher._expected(*self.expected_args)


if __name__ == "__main__":
    unittest.main()
