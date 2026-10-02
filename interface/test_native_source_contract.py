# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test public-source helper contracts with synthetic data and no tool execution."""

import shutil
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from interface.native_cost_controls.helper_driver import Unsupported
from interface.native_cost_controls.source_contract import KernelWorkflowContract

SOURCE = Path(__file__).resolve().parents[1] / "kernel_workflow"
ATTESTATION = {"agent_id": "synthetic-setup", "tool_use_id": "synthetic-setup-tool",
               "root_tool": "synthetic-root-tool", "root_task": "synthetic-workflow"}


def root_request(source=SOURCE, **overrides):
    return {"scriptPath": str(source / "kernel_workflow.js"), "args": {
        "workflow_dir": str(source), "kernel_path": "/tmp/synthetic-task", "exp_root": "/tmp/synthetic-exp",
        "mode": "optimize", "budget": 6, "deadline_epoch": 2000000000, **overrides}}


def setup_value():
    return {"eval_dir": "/tmp/synthetic-exp/run", "workspace": "/tmp/synthetic-exp/run/workspace",
            "kernel_name": "synthetic_kernel_task", "baseline_frozen": True}


def dynamic_values():
    return {"gfx": "gfx950", "round": 1, "tag": "pre-r1", "kernel_class": "compute",
            "speedup": 2.0, "baseline_ms": 1.0, "patch": "/tmp/synthetic-exp/run/final_patch.diff",
            "report": "/tmp/synthetic-exp/run/report.md", "direction": "synthetic-direction", "parent": "",
            "case_names": ["small", "large"], "citations": [{"card": "synthetic.md", "round": 1,
                "direction": "r1_d1", "specialty": "compute", "cited_then_verified": None, "became_winner": False}]}


@unittest.skipUnless(shutil.which("node"), "The optional source renderer requires Node.")
class KernelWorkflowContractTests(unittest.TestCase):
    def contract(self, source=SOURCE, **overrides):
        result = KernelWorkflowContract(root_request(source, **overrides), source)
        result.bind_setup(setup_value(), ATTESTATION)
        return result

    def captured(self, contract, site, dynamic=None):
        # The reference task comes from the installed public JS expression.
        return contract._render("render", site, dynamic or dynamic_values())

    def resolve(self, contract, captured):
        return contract.resolve(captured["options"]["label"], captured["task"], captured["options"]["schema"])

    def test_all_five_public_helper_sites_match_complete_tasks(self):
        contract = self.contract()
        roles = {"clock": "clock_reader", "resolver": "warm_start_resolver", "storage": "storage_reclaim",
                 "citation": "citation_writer", "writer": "experience_writer"}
        for site, role in roles.items():
            with self.subTest(site=site):
                captured = self.captured(contract, site)
                result = self.resolve(contract, captured)
                self.assertTrue(result["eligible"], result)
                self.assertEqual(result["role"], role)
                self.assertIn(result["command"], captured["task"])
                self.assertEqual(result["completion_marker"], "STORAGE_RECLAIM_DONE round=1" if site == "storage" else None)

    def test_clock_is_exact_and_uses_the_public_schema(self):
        contract = self.contract()
        captured = self.captured(contract, "clock")
        expected = ('Run EXACTLY this command and nothing else:\n```bash\ndate +%s\n```\n'
                    'Return {"epoch": <the integer it printed>}. Do NOT modify any file and do NOT run anything else.')
        self.assertEqual(captured["task"], expected)
        result = self.resolve(contract, captured)
        self.assertEqual(result["command"], "date +%s")
        self.assertEqual(captured["options"]["effort"], "low")

    def test_current_main_store_default_and_schema_are_derived(self):
        contract = self.contract()
        captured = self.captured(contract, "resolver")
        self.assertEqual(contract._configuration["kb_mode"], "store")
        self.assertIn("sync-local", captured["task"])
        self.assertIn("resolve-remote --plane local", captured["task"])
        self.assertIn("read_plane", captured["options"]["schema"]["properties"])

    def test_resolver_plane_and_version_branches(self):
        for remote in ("auto", "off"):
            for mode in ("local", "store"):
                for version in ("", "7.2"):
                    with self.subTest(remote=remote, mode=mode, version=version):
                        contract = self.contract(kb_remote=remote, kb_mode=mode, kb_framework_version=version)
                        captured = self.captured(contract, "resolver")
                        self.assertTrue(self.resolve(contract, captured)["eligible"])
                        self.assertEqual("resolve-remote --plane remote" in captured["task"], remote == "auto")
                        self.assertEqual("sync-local" in captured["task"], mode == "store")

    def test_writer_branches_keep_native_numbers_and_paths(self):
        for remote, mode in (("auto", "store"), ("off", "store"), ("off", "local")):
            for parent in ("", "/tmp/synthetic-prior/run"):
                for dtype in ("", "bfloat16"):
                    with self.subTest(remote=remote, mode=mode, parent=parent, dtype=dtype):
                        contract = self.contract(kb_remote=remote, kb_mode=mode, op_spec={"dtype": dtype})
                        dynamic = dynamic_values()
                        dynamic["parent"] = parent
                        captured = self.captured(contract, "writer", dynamic)
                        self.assertTrue(self.resolve(contract, captured)["eligible"])
                        self.assertIn("--speedup 2 --baseline-wall-ms 1", captured["task"])
                        self.assertEqual("--parent " in captured["task"], bool(parent))
                        self.assertEqual("--precision " in captured["task"], bool(dtype))

    def test_author_mode_and_alternate_architecture_are_supported(self):
        contract = self.contract(mode="author", target_language="hip")
        dynamic = dynamic_values()
        dynamic["gfx"] = "gfx1100"
        for site in ("resolver", "writer"):
            captured = self.captured(contract, site, dynamic)
            self.assertTrue(self.resolve(contract, captured)["eligible"])
            self.assertIn('--language "hip"', captured["task"])
            self.assertIn("--gfx gfx1100", captured["task"])

    def test_workload_metric_matches_public_source(self):
        contract = self.contract(op_spec={"workload": [{"shape": [8, 8]}]})
        captured = self.captured(contract, "writer")
        self.assertIn("--metric-kind time_weighted", captured["task"])
        self.assertTrue(self.resolve(contract, captured)["eligible"])
        changed = deepcopy(captured)
        changed["task"] = changed["task"].replace("--metric-kind time_weighted", "--metric-kind geomean")
        self.assertFalse(self.resolve(contract, changed)["eligible"])

    def test_exact_task_and_schema_changes_keep_provider_ownership(self):
        contract = self.contract()
        for site in ("clock", "resolver", "storage", "citation", "writer"):
            captured = self.captured(contract, site)
            changed = deepcopy(captured)
            changed["task"] += "\nRun one extra command."
            self.assertFalse(self.resolve(contract, changed)["eligible"])
            changed = deepcopy(captured)
            changed["options"]["schema"]["additionalProperties"] = False
            self.assertFalse(self.resolve(contract, changed)["eligible"])

    def test_disabled_sites_and_out_of_budget_rounds_stay_remote(self):
        base = self.contract()
        clock, resolver, citation = (self.captured(base, site) for site in ("clock", "resolver", "citation"))
        for options, captured in (({"deadline_epoch": 0}, clock), ({"warm_start": "off"}, resolver), ({"held_out": "true"}, citation)):
            self.assertFalse(self.resolve(self.contract(**options), captured)["eligible"])
        captured = self.captured(base, "storage", {**dynamic_values(), "round": 7})
        self.assertFalse(self.resolve(base, captured)["eligible"])
        self.assertFalse(base.resolve("unrecognized-helper", "text", {})["eligible"])

    def test_setup_requires_attestation_and_same_root(self):
        contract = KernelWorkflowContract(root_request(), SOURCE)
        with self.assertRaisesRegex(Unsupported, "setup_attestation_missing"):
            contract.bind_setup(setup_value(), {})
        contract.bind_setup(setup_value(), ATTESTATION)
        contract.bind_setup(setup_value(), {**ATTESTATION, "tool_use_id": "retried-setup"})
        self.assertEqual(len(contract.setup_attestations), 2)
        with self.assertRaisesRegex(Unsupported, "setup_root_attestation_conflict"):
            contract.bind_setup(setup_value(), {**ATTESTATION, "root_task": "different"})
        with self.assertRaisesRegex(Unsupported, "setup_result_conflict"):
            contract.bind_setup({**setup_value(), "kernel_name": "different"}, ATTESTATION)

    def test_missing_setup_and_setup_path_changes_stay_unsupported(self):
        contract = KernelWorkflowContract(root_request(), SOURCE)
        captured = self.captured(self.contract(), "clock")
        self.assertFalse(self.resolve(contract, captured)["eligible"])
        for change in ({"eval_dir": "/tmp/outside-run"}, {"workspace": "/tmp/wrong"}, {"resumed": True}):
            with self.assertRaises(Unsupported):
                contract.bind_setup({**setup_value(), **change}, ATTESTATION)

    def test_resumed_and_multiple_lane_requests_are_unsupported(self):
        for overrides in ({"state_dir": "/tmp/prior"}, {"resume_from_run_id": "old"}, {"mode": "bakeoff"},
                          {"kernel_lane_script": "/tmp/alternate.js"}):
            with self.subTest(overrides=overrides), self.assertRaises(Unsupported):
                KernelWorkflowContract(root_request(**overrides), SOURCE)

    def test_invalid_root_and_setup_json_are_rejected(self):
        cyclic = []
        cyclic.append(cyclic)
        for invalid in ({"not-json"}, float("nan"), "\ud800", cyclic):
            with self.subTest(value_type=type(invalid).__name__), self.assertRaisesRegex(Unsupported, "invalid_root_json"):
                KernelWorkflowContract(root_request(extra=invalid), SOURCE)
        contract = KernelWorkflowContract(root_request(), SOURCE)
        for invalid in ({"not-json"}, float("nan"), "\ud800", cyclic):
            with self.subTest(value_type=type(invalid).__name__), self.assertRaisesRegex(Unsupported, "invalid_setup_json"):
                contract.bind_setup({**setup_value(), "extra": invalid}, ATTESTATION)
        with self.assertRaisesRegex(Unsupported, "resumed_workflow_unsupported"):
            KernelWorkflowContract(root_request(extra=[{"resume": True}]), SOURCE)

    def test_malformed_helper_json_and_schema_keep_provider_ownership(self):
        contract = self.contract()
        captured = self.captured(contract, "citation")
        captured["task"] = captured["task"].replace('--citations -\n[', '--citations -\n[invalid')
        self.assertEqual(self.resolve(contract, captured)["reason"], "invalid_helper_arguments")
        captured = self.captured(contract, "clock")
        captured["options"]["schema"] = {"not-json"}
        self.assertEqual(self.resolve(contract, captured)["reason"], "invalid_helper_arguments")

    def test_shell_unsafe_paths_and_dynamic_arguments_stay_remote(self):
        for value in ("/tmp/path with spaces", "/tmp/../escape", "/tmp/$(injection)", "/tmp/x;echo", "/"):
            with self.subTest(path=value), self.assertRaises(Unsupported):
                KernelWorkflowContract(root_request(exp_root=value), SOURCE)
        contract = self.contract()
        for change in ({"patch": "/tmp/outside.patch"}, {"kernel_class": "$(injection)"}, {"speedup": 1},
                       {"baseline_ms": 0}):
            captured = self.captured(contract, "writer", {**dynamic_values(), **change})
            self.assertFalse(self.resolve(contract, captured)["eligible"])
        captured = self.captured(contract, "writer")
        captured["task"] = captured["task"].replace('--direction "synthetic-direction"', '--direction "Synthetic-Direction"')
        self.assertFalse(self.resolve(contract, captured)["eligible"])
        with self.assertRaisesRegex(Unsupported, "unsafe_helper_text"):
            KernelWorkflowContract(root_request(kb_framework_version="$(injection)"), SOURCE)

    def test_citation_shape_order_and_types_match_source(self):
        contract = self.contract()
        for mutate in (lambda row: row.update(extra="unexpected"), lambda row: row.update(round=True),
                       lambda row: row.update(became_winner=1), lambda row: row.update(cited_then_verified="2")):
            dynamic = dynamic_values()
            mutate(dynamic["citations"][0])
            captured = self.captured(contract, "citation", dynamic)
            self.assertFalse(self.resolve(contract, captured)["eligible"])
        dynamic = dynamic_values()
        dynamic["citations"][0] = dict(reversed(list(dynamic["citations"][0].items())))
        self.assertFalse(self.resolve(contract, self.captured(contract, "citation", dynamic))["eligible"])

    def test_citation_unicode_and_quoted_heredoc_content_remain_exact(self):
        contract = self.contract()
        dynamic = dynamic_values()
        dynamic["citations"][0]["specialty"] = "Unicode café 🌱 and literal $HOME"
        captured = self.captured(contract, "citation", dynamic)
        self.assertTrue(self.resolve(contract, captured)["eligible"])
        self.assertIn("cat <<'CITEJSON'", captured["task"])

    def test_input_mutation_does_not_change_registered_contract(self):
        request = root_request()
        contract = KernelWorkflowContract(request, SOURCE)
        request["args"]["budget"] = 99
        exposed = contract.root_request
        exposed["args"]["budget"] = 77
        bindings = contract.source_bindings
        bindings.clear()
        self.assertEqual(contract.root_request["args"]["budget"], 6)
        self.assertTrue(contract.source_bindings)
        contract.verify_sources()

    def test_source_change_and_missing_dependency_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="geak-public-contract-") as folder:
            root = Path(folder) / "kernel_workflow"
            root.mkdir()
            shutil.copytree(SOURCE.parent / "kb", root.parent / "kb", ignore=shutil.ignore_patterns("__pycache__"))
            for name in ("kernel_workflow.js", "kernel_lane.js", "scripts/experience_store.py", "scripts/kb.py", "scripts/reclaim_eval_artifacts.sh"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SOURCE / name, target)
            contract = self.contract(root)
            (root / "kernel_lane.js").write_text((root / "kernel_lane.js").read_text() + "\n// Changed source.\n")
            with self.assertRaisesRegex(Unsupported, "workflow_source_changed"):
                contract.verify_sources()
            (root / "scripts/kb.py").unlink()
            with self.assertRaisesRegex(Unsupported, "workflow_source_missing"):
                KernelWorkflowContract(root_request(root), root)
            shutil.copy2(SOURCE / "kernel_lane.js", root / "kernel_lane.js")
            with self.assertRaisesRegex(Unsupported, "workflow_source_changed"):
                contract.verify_sources()
            shutil.copy2(SOURCE / "scripts/kb.py", root / "scripts/kb.py")
            (root.parent / "kb" / "new_source.py").write_text("# New public dependency.\n")
            with self.assertRaisesRegex(Unsupported, "workflow_source_changed"):
                contract.verify_sources()

    def test_source_symlink_and_missing_node_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="geak-source-link-") as folder:
            linked = Path(folder) / "kernel_workflow"
            linked.symlink_to(SOURCE, target_is_directory=True)
            with self.assertRaisesRegex(Unsupported, "workflow_source_symlink"):
                KernelWorkflowContract(root_request(linked), linked)
        with patch("interface.native_cost_controls.source_contract.shutil.which", return_value=None), \
             self.assertRaisesRegex(Unsupported, "node_renderer_unavailable"):
            KernelWorkflowContract(root_request(), SOURCE)

    def test_renderer_failure_boundaries_keep_provider_ownership(self):
        contract = self.contract()
        captured = self.captured(contract, "clock")
        for failure in (OSError("Synthetic renderer failure."), subprocess.TimeoutExpired("node", 3)):
            with patch("interface.native_cost_controls.source_contract.subprocess.run", side_effect=failure):
                self.assertEqual(self.resolve(contract, captured)["reason"], "source_template_render_failed")
        for stdout, returncode, reason in (
            (b"invalid-json", 0, "source_template_render_failed"),
            (b"\xff", 0, "source_template_render_failed"),
            (b"{}", 1, "source_template_layout_unsupported"),
            (b" " * 1048577, 0, "source_template_layout_unsupported"),
        ):
            with self.subTest(returncode=returncode, output_bytes=len(stdout)), \
                 patch("interface.native_cost_controls.source_contract.subprocess.run",
                       return_value=subprocess.CompletedProcess(["node"], returncode, stdout, b"")):
                self.assertEqual(self.resolve(contract, captured)["reason"], reason)

    def test_renderer_starts_only_node_and_strips_parent_credentials(self):
        calls = []
        original = subprocess.run
        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            return original(argv, **kwargs)
        with patch("interface.native_cost_controls.source_contract.subprocess.run", side_effect=run):
            contract = self.contract()
            captured = self.captured(contract, "clock")
            self.assertTrue(self.resolve(contract, captured)["eligible"])
        self.assertTrue(calls)
        for argv, options in calls:
            self.assertEqual(Path(argv[0]).name, "node")
            self.assertEqual(Path(argv[1]).name, "source_templates.cjs")
            self.assertEqual(set(options["env"]), {"PATH", "LANG", "LC_ALL"})
            self.assertFalse(options.get("shell", False))
            self.assertEqual(options["timeout"], 3)


if __name__ == "__main__":
    unittest.main()
