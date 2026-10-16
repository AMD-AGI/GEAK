"""Apply the prospective patch to exact retained files and exercise its callsite.

Owner custody/import/provider services are doubles; this tests argument/service
wiring only. No installed owner, UID transition, model or device is exercised.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from geak_evo.archive import refs
from geak_evo.host import HostServices


class OwnerDispatchPatchTests(unittest.TestCase):
    def test_existing_owner_entry_reaches_dispatch_main_with_same_final_services(self):
        repository = Path(__file__).resolve().parents[2]
        base = json.loads((repository / "docs/evo/owner-dispatch-base.json").read_bytes())
        if not all(Path(row["source_path"]).is_file() for row in base["files"]):
            self.skipTest("retained owner sources are not installed in this CPU checkout")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in base["files"]:
                raw = Path(row["source_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
                (root / row["name"]).write_bytes(raw)
            subprocess.run(["/usr/bin/git", "apply", str(repository / "docs/evo/owner-dispatch.patch")],
                           cwd=root, check=True, capture_output=True)
            def load(name):
                spec = importlib.util.spec_from_file_location("test_" + name, root / (name + ".py"))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            entry, adapter = load("host_driver"), load("native_host_adapter")
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
            budget_config, budget, evaluator = {"existing": "budget"}, object(), object()
            owner_config = root / "owner-config.json"
            owner_config.write_text(json.dumps({"baseline_candidate_id": "seed", "search": {
                "campaign_id": "campaign", "seed_candidate_id": "seed", "campaign_root": str(campaign),
                "templates": {key: template_refs[key] for key in ("config", "admission", "entry", "collection")},
                "clock_ref": template_refs["clock"], "shared_budget": budget_config}}))
            binding.write_text(json.dumps({"max_steps": 1, "owner_config_ref": refs.file_ref(owner_config)}))
            binding_ref = refs.file_ref(binding)
            make_budget = mock.Mock(return_value=budget)
            native_modules = {"host": NS(Budget=make_budget), "adapter": object()}
            retained = NS(validate_config=mock.Mock(return_value={**config, "_admission_path": template_refs["admission"]["path"]}),
                          load_imports=mock.Mock(return_value=native_modules))
            coordinator = NS(__file__="fixture_coordinator.py", checked=mock.Mock(return_value=b"{}"),
                             reference=lambda pair: {"path": pair[0], "sha256": pair[1]})
            def owner_main():
                return coordinator.host(args, template_refs, {"config": config, "entry": {"source_paths": ["kernel.py"]}},
                                        {}, budget_config, {}, object())
            coordinator.main = owner_main
            modules = {"pilot_retained_coordinator": coordinator, "pilot_cleaned_adapter": adapter,
                       "pilot_dispatch_retained_driver": retained}
            with mock.patch.object(entry, "checked"), mock.patch.object(entry, "authenticated"), \
                 mock.patch.object(entry, "installation", return_value={"coordinator_ref": {}, "adapter_ref": {}, "retained_driver_ref": {}}), \
                 mock.patch.object(entry, "load", side_effect=lambda name, ref: modules[name]), \
                 mock.patch.object(entry, "owner_services", return_value=(cleaned, object(), object(), object(),
                    NS(native_episode_material=lambda *args: None), NS(configure_store_frontier=mock.Mock()), {})), \
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


if __name__ == "__main__":
    unittest.main()
