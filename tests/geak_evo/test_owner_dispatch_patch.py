"""Apply the prospective patch to exact retained files and exercise its callsite.

Owner custody/import/provider services are doubles; this tests argument/service
wiring only. No installed owner, UID transition, model or device is exercised.
"""
import hashlib
import importlib
import importlib.util
import json
import marshal
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from geak_evo.archive import refs
from geak_evo.host import HostServices
from geak_evo.evaluate.jobs import job_digest
from test_external_ut import fixture as external_fixture


class OwnerDispatchPatchTests(unittest.TestCase):
    def setUp(self):
        self.repository = Path(__file__).resolve().parents[2]
        base = json.loads((self.repository / "docs/evo/owner-dispatch-base.json").read_bytes())
        if not all(Path(row["source_path"]).is_file() for row in base["files"]):
            self.skipTest("retained owner sources are not installed in this CPU checkout")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for row in base["files"]:
            raw = Path(row["source_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
            (self.root / row["name"]).write_bytes(raw)
        subprocess.run(["/usr/bin/git", "apply", str(self.repository / "docs/evo/owner-dispatch.patch")],
                       cwd=self.root, check=True, capture_output=True)
        self.addCleanup(self.clear_contract_namespace)

    @staticmethod
    def clear_contract_namespace():
        for name in list(sys.modules):
            if name == "_geak_evo_pilot_contracts" or name.startswith("_geak_evo_pilot_contracts."):
                del sys.modules[name]

    def load(self, name):
        spec = importlib.util.spec_from_file_location("test_" + name, self.root / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def contract_inputs(self):
        cleaned = self.root / "cleaned"
        names = ("_util", "evaluate.jobs_v1", "evaluate.jobs", "evaluate.external",
                 "evaluate.native_contract", "evaluate.generated_task")
        source_refs = {}
        for name in names:
            relative = "geak_evo/" + name.replace(".", "/") + ".py"
            path = cleaned / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.repository / relative).read_bytes())
            source_refs[relative] = refs.file_ref(path)
        manifest = {"cleaned_root": str(cleaned), "cleaned_source_refs": source_refs,
                    "source_refs": list(source_refs.values())}
        task = self.root / "task"
        contract = external_fixture(task)
        # Non-ASCII metadata remains part of the canonical registered identity.
        contract["source"]["repository"] = "external:任务"
        forbidden_source = b"raise RuntimeError('task source must never run at root preflight')\n"
        for name in ("kernel_src/module.py", "baseline_src/module.py"):
            (task / name).write_bytes(forbidden_source)
            contract["artifacts"][name] = hashlib.sha256(forbidden_source).hexdigest()
        path = self.root / "contract.json"
        path.write_text(json.dumps(contract, ensure_ascii=False))
        return manifest, path, task, contract

    def test_existing_owner_entry_reaches_dispatch_main_with_same_final_services(self):
        self._owner_entry_dispatch()

    def test_checkpoint_owner_uses_cleaned_registration_and_identical_host_options(self):
        self._owner_entry_dispatch(["H0.5", "H1", "H2", "H4", "H6", "H8", "END"])

    def _owner_entry_dispatch(self, checkpoint_horizons=()):
        root = self.root
        entry, adapter = self.load("host_driver"), self.load("native_host_adapter")
        stable = root / "native"
        (stable / "workspace_helpers").mkdir(parents=True)
        (stable / "host").mkdir()
        (stable / "workspace_helpers/collect_investigation.py").write_text("# CPU fixture\n")
        (stable / "host_driver.py").write_text("# CPU fixture\n")
        campaign = root / "campaign"; campaign.mkdir()
        binding = root / "owner.json"; binding.write_text('{"max_steps":1}')
        binding_ref = refs.file_ref(binding)
        template_refs = {name: {"path": str(root / (name + ".json")), "sha256": "a" * 64}
                         for name in ("config", "admission", "entry", "collection", "clock")}
        config = {"backend_config": {"probe_config": {"job_id": "job", "measurement_epoch_id": "epoch"},
                   "model_config": {"model_route_ref": {}, "workflow_backend": "codex"}},
                  "imports": {"host_root": str(stable / "host")}}
        store = NS(capture_source=mock.Mock(return_value="source"), candidate=mock.Mock(return_value="seed"))
        episode = lambda *args: None
        search = NS(assignment=NS(native_evidence=NS(episode=episode)))
        dispatch_main = mock.Mock(return_value=0)
        cleaned = NS(store=NS(ResearchStore=mock.Mock(return_value=store)), refs=refs,
            model=NS(resolve_model_route=mock.Mock(return_value=({}, {}))),
            profile=NS(ModelProfile=NS(from_route=mock.Mock(return_value="profile"))),
            host=NS(HostServices=HostServices, prepare_native_search=mock.Mock(return_value=search)),
            dispatch=NS(main=dispatch_main))
        args = NS(campaign_root=str(campaign), frontier_settings=["frontier", "sha"], campaign_id="campaign",
            seed_root="seed", seed_source_id="source", case_id="case", evaluation_limit=10,
            planner_seconds=1, planner_decision_limit=1, sealed_root="sealed", root_state_root="state",
            common_read_root=[], common_refs=[], common_tool_addendum="", review_objective="ordinary", max_steps=1)
        budget_config, budget, evaluator = {"existing": "budget", "max_calls": 10}, object(), object()
        owner_config = root / "owner-config.json"
        owner_config.write_text(json.dumps({"baseline_candidate_id": "seed", "search": {
            "campaign_id": "campaign", "seed_candidate_id": "seed", "campaign_root": str(campaign),
            "templates": {key: template_refs[key] for key in ("config", "admission", "entry", "collection")},
            "clock_ref": template_refs["clock"], "shared_budget": budget_config,
            "search_reserve_seconds": 4800, "evaluation_limit": 7,
            **({"checkpoint_horizons": checkpoint_horizons} if checkpoint_horizons else {})}}))
        binding.write_text(json.dumps({"max_steps": 1, "owner_config_ref": refs.file_ref(owner_config)}))
        binding_ref = refs.file_ref(binding)
        make_budget = mock.Mock(return_value=budget)
        native_modules = {"host": NS(Budget=make_budget), "adapter": object()}
        retained = NS(validate_config=mock.Mock(return_value={**config, "_admission_path": template_refs["admission"]["path"]}),
                      load_imports=mock.Mock(return_value=native_modules))
        coordinator = NS(__file__="fixture_coordinator.py", checked=mock.Mock(return_value=b"{}"),
                         reference=lambda pair: {"path": pair[0], "sha256": pair[1]})
        def owner_main(*, contract_loader):
            self.assertEqual(contract_loader("contract", artifact_root="artifacts", expected_digest="job"), {})
            return coordinator.host(args, template_refs, {"config": config, "entry": {"source_paths": ["kernel.py"]}},
                                    {}, budget_config, {}, object())
        coordinator.main = owner_main
        configure_frontier = mock.Mock()
        modules = {"pilot_retained_coordinator": coordinator, "pilot_cleaned_adapter": adapter,
                   "pilot_dispatch_retained_driver": retained}
        with mock.patch.object(entry, "checked"), mock.patch.object(entry, "authenticated"), \
             mock.patch.object(entry, "load_cleaned_contract", return_value={}) as contract_loader, \
             mock.patch.object(entry, "installation", return_value={"coordinator_ref": {}, "adapter_ref": {}, "retained_driver_ref": {}}), \
             mock.patch.object(entry, "load", side_effect=lambda name, ref: modules[name]), \
             mock.patch.object(entry, "owner_services", return_value=(cleaned, object(), object(), object(),
                NS(native_episode_material=lambda *args: None), NS(configure_store_frontier=configure_frontier), {})), \
             mock.patch.object(adapter, "evaluator_for_driver", return_value=evaluator), \
             mock.patch.object(adapter.os, "geteuid", return_value=1000):
            code = entry.owner_main(["--pilot-manifest", "manifest", "sha", "--dispatch-owner", binding_ref["path"],
                binding_ref["sha256"], "--task", "task.json", "--out", "result.json"])
        self.assertEqual(code, 0)
        dispatch_main.assert_called_once()
        argv, = dispatch_main.call_args.args
        self.assertEqual(argv, ["--task", "task.json", "--out", "result.json"])
        supplied = dispatch_main.call_args.kwargs
        self.assertIs(supplied["store"], store)
        self.assertIs(supplied["evaluator"], evaluator)
        self.assertIs(supplied["budget"], budget)
        self.assertIs(supplied["episode"], episode)
        self.assertEqual(supplied["owner_binding_ref"], binding_ref)
        make_budget.assert_called_once_with(budget_config)
        cleaned.host.prepare_native_search.assert_called_once()
        self.assertEqual(cleaned.host.prepare_native_search.call_args.kwargs["search_reserve_seconds"], 4800)
        self.assertEqual(cleaned.host.prepare_native_search.call_args.kwargs["evaluation_limit"], 7)
        self.assertEqual(cleaned.host.prepare_native_search.call_args.kwargs["shared_budget"]["max_calls"], 10)
        if checkpoint_horizons:
            configure_frontier.assert_not_called()
            self.assertEqual(cleaned.host.prepare_native_search.call_args.kwargs["checkpoint_horizons"],
                             checkpoint_horizons)
        else:
            configure_frontier.assert_called_once()
            self.assertNotIn("checkpoint_horizons", cleaned.host.prepare_native_search.call_args.kwargs)
        contract_loader.assert_called_once()

    def test_preflight_external_contract_preserves_pins_without_importing_host_or_task(self):
        entry = self.load("host_driver")
        manifest, path, task, contract = self.contract_inputs()
        existing = {name: module for name, module in sys.modules.items()
                    if name == "geak_evo" or name.startswith("geak_evo.")}
        # Only custody is a double: real hashes, import closure, task validation
        # and the UTF-8 registered job identity are exercised.
        with mock.patch.object(entry, "immutable", side_effect=Path):
            actual = entry.load_cleaned_contract(manifest, path, artifact_root=task,
                                                  expected_digest=job_digest(contract))
        self.assertEqual(actual, contract)
        self.assertTrue(all(sys.modules[name] is module for name, module in existing.items()))
        namespace = "_geak_evo_pilot_contracts"
        imported = {name.removeprefix(namespace) for name in sys.modules
                    if name == namespace or name.startswith(namespace + ".")}
        self.assertEqual(imported, {"", ".evaluate", "._util", ".evaluate.jobs_v1", ".evaluate.jobs",
                                    ".evaluate.external", ".evaluate.native_contract", ".evaluate.generated_task"})
        self.assertEqual(sys.modules[namespace].__path__, [])
        self.assertEqual(sys.modules[namespace + ".evaluate"].__path__, [])
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module(namespace + ".host")

    def test_preflight_executes_verified_source_even_with_matching_poisoned_bytecode(self):
        entry = self.load("host_driver")
        manifest, path, task, contract = self.contract_inputs()
        source = Path(manifest["cleaned_source_refs"]["geak_evo/_util.py"]["path"])
        cache = Path(importlib.util.cache_from_source(str(source)))
        cache.parent.mkdir()
        poison = compile("raise RuntimeError('unregistered cached code executed')", str(source), "exec")
        metadata = struct.pack("<II", int(source.stat().st_mtime), source.stat().st_size)
        cache.write_bytes(importlib.util.MAGIC_NUMBER + b"\0" * 4 + metadata + marshal.dumps(poison))
        with mock.patch.object(entry, "immutable", side_effect=Path):
            actual = entry.load_cleaned_contract(manifest, path, artifact_root=task,
                                                  expected_digest=job_digest(contract))
        self.assertEqual(actual, contract)

    def test_preflight_refuses_changed_reader_before_importing_any_module(self):
        entry = self.load("host_driver")
        manifest, path, task, contract = self.contract_inputs()
        source = Path(manifest["cleaned_source_refs"]["geak_evo/evaluate/external.py"]["path"])
        source.write_text(source.read_text() + "\nraise RuntimeError('unreviewed source')\n")
        with mock.patch.object(entry, "immutable", side_effect=Path), mock.patch.object(entry, "load") as load:
            with self.assertRaisesRegex(ValueError, "source reference differs"):
                entry.load_cleaned_contract(manifest, path, artifact_root=task, expected_digest=job_digest(contract))
        load.assert_not_called()
        self.assertNotIn("_geak_evo_pilot_contracts", sys.modules)

    def test_preflight_refuses_reader_omitted_from_owner_inventory(self):
        entry = self.load("host_driver")
        manifest, path, task, contract = self.contract_inputs()
        manifest["source_refs"].pop()
        with mock.patch.object(entry, "immutable", side_effect=Path), mock.patch.object(entry, "load") as load:
            with self.assertRaisesRegex(ValueError, "omitted from pinned cleaned source closure"):
                entry.load_cleaned_contract(manifest, path, artifact_root=task, expected_digest=job_digest(contract))
        load.assert_not_called()
        self.assertNotIn("_geak_evo_pilot_contracts", sys.modules)

    def test_preflight_refuses_contract_id_mismatch(self):
        entry = self.load("host_driver")
        manifest, path, task, _ = self.contract_inputs()
        with mock.patch.object(entry, "immutable", side_effect=Path):
            with self.assertRaisesRegex(ValueError, "registered Arena job digest mismatch"):
                entry.load_cleaned_contract(manifest, path, artifact_root=task, expected_digest="0" * 64)

    def test_preflight_refuses_changed_task_artifact(self):
        entry = self.load("host_driver")
        manifest, path, task, contract = self.contract_inputs()
        (task / "ut_main.py").write_text("# changed after task registration\n")
        with mock.patch.object(entry, "immutable", side_effect=Path):
            with self.assertRaisesRegex(ValueError, "external artifact changed"):
                entry.load_cleaned_contract(manifest, path, artifact_root=task, expected_digest=job_digest(contract))

    def test_coordinator_threads_loader_into_existing_readonly_inspection(self):
        coordinator = self.load("coordinator")
        args, loader = object(), object()
        class InspectionReached(Exception):
            pass
        with mock.patch.object(coordinator, "parser", return_value=NS(parse_args=lambda: args)), \
             mock.patch.object(coordinator.sys, "flags", NS(isolated=True, dont_write_bytecode=True)), \
             mock.patch.dict(coordinator.os.environ, {"PYTHONDONTWRITEBYTECODE": "1"}), \
             mock.patch.object(coordinator, "inspect", side_effect=InspectionReached) as inspect:
            with self.assertRaises(InspectionReached):
                coordinator.main(contract_loader=loader)
        inspect.assert_called_once_with(args, contract_loader=loader)


if __name__ == "__main__":
    unittest.main()
