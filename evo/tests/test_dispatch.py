"""CPU dispatch regression against actual pinned main, with synthetic agents.

Run: python3 -B -m unittest discover -s evo/tests -p test_dispatch.py -v
These tests execute workflow JavaScript, not either production Workflow host.
Agent replies, measurements, archive commands and GPU identity are synthetic.
"""

from __future__ import annotations

import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import types
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[2]
PINNED_MAIN = "a48ac8bef9aeab299ba5ac9cb44164d997e94524"
HARNESS = Path(__file__).with_name("dispatch_harness.cjs")
NODE = shutil.which("node") or str(Path.home() / ".local/bin/node")
SAFE_ENV = {"PATH": "/usr/bin:/bin", "GEAK_AGENT_AUTO": "0", "PYTHONDONTWRITEBYTECODE": "1"}
JAVASCRIPT = (
    "kernel_workflow/kernel_workflow.js",
    "kernel_workflow/kernel_lane.js",
    "e2e_workflow/e2e_workflow.js",
)
RUNNER = "interface/run_e2e.py"
LEGACY_FLAGS = ({}, {"optimizer": None}, {"optimizer": ""}, {"optimizer": "greedy"}, {"optimizer": "  GrEeDy  "})
BASE_ARGS = {
    "kernel_path": "/fixture/kernel",
    "workflow_dir": str(ROOT / "kernel_workflow"),
    "expected_gfx": "gfx950",
    "expected_target": "unknown",
    "budget": 2,
    "kernel_budget": 2,
    "gpu_ids": "2,5",
    "task": "Deterministic CPU dispatch fixture",
    "warm_start": "off",
    "update_experience": "off",
    "use_learned_kb": "false",
    "agent_timeout_ms": 0,
}


def pinned_source(filename):
    result = subprocess.run(
        ["git", "show", f"{PINNED_MAIN}:{filename}"], cwd=ROOT, env=SAFE_ENV,
        check=True, text=True, capture_output=True, timeout=15,
    )
    return result.stdout


class FixedDatetime(datetime.datetime):
    @classmethod
    def now(cls, timezone=None):
        return cls(2026, 11, 11, 12, 0, 0, tzinfo=timezone)


class DispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pinned = {filename: pinned_source(filename) for filename in JAVASCRIPT}
        cls.current = {filename: (ROOT / filename).read_text() for filename in JAVASCRIPT}
        cls.current["evo/kernel_lane.js"] = (ROOT / "evo/kernel_lane.js").read_text()
        cls.pinned_runner = pinned_source(RUNNER)
        cls.current_runner = (ROOT / RUNNER).read_text()

    def run_workflow(self, entry, args=None, *, pinned=False, **options):
        request = {
            "root": str(ROOT), "sources": self.pinned if pinned else self.current,
            "entry": entry, "args": {**BASE_ARGS, **(args or {})}, **options,
        }
        if entry.startswith("e2e_workflow/"):
            request["args"]["workflow_dir"] = str(ROOT / "e2e_workflow")
        result = subprocess.run(
            [NODE, str(HARNESS)], input=json.dumps(request), text=True, capture_output=True,
            env=SAFE_ENV, timeout=15, check=True,
        )
        outcome = json.loads(result.stdout)
        self.assertEqual(outcome["trace"]["mockErrors"], [], "Incomplete mock, not a product failure")
        return outcome

    def assert_success(self, outcome):
        self.assertIsNone(outcome["error"], outcome["error"])
        return outcome["result"]

    def assert_rejected_before_agents(self, outcome, message):
        self.assertIsNotNone(outcome["error"], {
            "mode": (outcome["result"] or {}).get("mode"),
            "validation_status": (outcome["result"] or {}).get("validation_status"),
            "agents_started": len(outcome["trace"]["agents"]),
        })
        self.assertRegex(outcome["error"]["message"], message)
        self.assertEqual(outcome["trace"]["agents"], [])

    def test_pinned_lane_is_the_reviewed_main(self):
        self.assertEqual(
            hashlib.sha256(self.pinned[JAVASCRIPT[1]].encode()).hexdigest(),
            "6ab6132593eb9675df7a37b62fab428f61df82247a17f97ed55652e331e78b83",
        )

    def test_generated_worker_is_current_without_regeneration(self):
        result = subprocess.run(
            [sys.executable, "-B", "evo/build_workflow.py", "--check"],
            cwd=ROOT, env=SAFE_ENV, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.current["evo/kernel_lane.js"], (ROOT / "evo/kernel_lane.js").read_text(),
                         "Generated worker changed during the suite; rerun against one snapshot")

    def test_legacy_direct_lane_complete_trace_matches_pinned_main(self):
        for mode in ("optimize", "author"):
            for flags in LEGACY_FLAGS:
                with self.subTest(mode=mode, flags=flags):
                    arguments = {"mode": mode, **flags}
                    before = self.run_workflow(JAVASCRIPT[1], arguments, pinned=True)
                    after = self.run_workflow(JAVASCRIPT[1], arguments)
                    result = self.assert_success(after)
                    self.assertEqual(after, before)
                    self.assertEqual(result["validation_status"], "accepted")
                    self.assertEqual(result["rounds"], 1)
                    self.assertEqual(result["budget_used"], 2)
                    self.assertEqual(result["stopped_by"], "budget")
                    self.assertEqual(after["trace"]["maxDepth"], 1)
                    labels = [call["options"]["label"] for call in after["trace"]["agents"]]
                    for required in ("integrate r1", "commit r1", "director:validate"):
                        self.assertIn(required, labels)

    def test_legacy_dispatcher_routes_and_forwards_exactly_as_pinned_main(self):
        for mode in ("optimize", "author", "", None):
            for flags in LEGACY_FLAGS:
                with self.subTest(mode=mode, flags=flags):
                    arguments = {"mode": mode, **flags, "sentinel": {"nested": [1, "unchanged"]}}
                    before = self.run_workflow(JAVASCRIPT[0], arguments, pinned=True, stubWorkflows=True)
                    after = self.run_workflow(JAVASCRIPT[0], arguments, stubWorkflows=True)
                    self.assert_success(after)
                    self.assertEqual(after, before)
                    self.assertEqual(after["trace"]["agents"], [])
                    self.assertEqual(len(after["trace"]["workflows"]), 1)
                    forwarded = after["trace"]["workflows"][0]
                    self.assertEqual(forwarded["scriptPath"], str(ROOT / "kernel_workflow/kernel_lane.js"))
                    self.assertEqual(forwarded["args"]["sentinel"], arguments["sentinel"])

    def test_legacy_custom_worker_route_remains_unchanged(self):
        for flags in LEGACY_FLAGS:
            with self.subTest(flags=flags):
                arguments = {**flags, "kernel_lane_script": "/fixture/custom.js"}
                before = self.run_workflow(JAVASCRIPT[0], arguments, pinned=True, stubWorkflows=True)
                after = self.run_workflow(JAVASCRIPT[0], arguments, stubWorkflows=True)
                self.assert_success(after)
                self.assertEqual(before, after)
                self.assertEqual(after["trace"]["workflows"][0]["scriptPath"], "/fixture/custom.js")

    def test_legacy_dispatcher_completes_with_identical_child_and_ledger_prompts(self):
        for flags in LEGACY_FLAGS:
            for mode in ("optimize", "author"):
                with self.subTest(flags=flags, mode=mode):
                    arguments = {**flags, "mode": mode}
                    before = self.run_workflow(JAVASCRIPT[0], arguments, pinned=True)
                    after = self.run_workflow(JAVASCRIPT[0], arguments)
                    self.assert_success(after)
                    self.assertEqual(after, before)
                    self.assertEqual(after["trace"]["maxDepth"], 2)
                    labels = [call["options"]["label"] for call in after["trace"]["agents"]]
                    self.assertIn("file_writer:persist_llm_stats", labels)

    def test_legacy_bakeoff_still_enters_freeze(self):
        for flags in LEGACY_FLAGS:
            with self.subTest(flags=flags):
                arguments = {**flags, "mode": "bakeoff"}
                replies = {"oracle_freezer": {"op_kind": "gemm", "task_dir": "/fixture/oracle", "smoke": "fail"}}
                before = self.run_workflow(JAVASCRIPT[0], arguments, pinned=True, replies=replies)
                after = self.run_workflow(JAVASCRIPT[0], arguments, replies=replies)
                self.assertEqual(before, after)
                self.assertEqual(len(after["trace"]["agents"]), 1)
                self.assertEqual(after["trace"]["agents"][0]["options"]["label"], "oracle_freezer")

    def test_legacy_failure_paths_preserve_prompts_and_terminal_results(self):
        cases = (
            {"director:setup": None},
            {"benchmark_engineer": None},
            {"eng fixture_0:memory": None},
            {"verify fixture_0": {"status": "rejected", "correctness": "fail", "verified_geomean": 9}},
            {"tech_lead:plan r1": {"stop": True, "directions": []}},
            {"director:validate": {"validation_status": "rejected", "director_verified_speedup_geomean": 0}},
        )
        for replies in cases:
            for flags in LEGACY_FLAGS[:3]:
                with self.subTest(replies=replies, flags=flags):
                    before = self.run_workflow(JAVASCRIPT[1], flags, pinned=True, replies=replies)
                    after = self.run_workflow(JAVASCRIPT[1], flags, replies=replies)
                    self.assertEqual(before, after)

    def test_legacy_e2e_passthrough_completes_with_identical_lane_prompts(self):
        for flags in LEGACY_FLAGS:
            with self.subTest(flags=flags):
                before = self.run_workflow(JAVASCRIPT[2], flags, pinned=True)
                after = self.run_workflow(JAVASCRIPT[2], flags)
                result = self.assert_success(after)
                self.assertEqual(after, before)
                self.assertTrue(result["ran"], result)
                self.assertEqual(result["validation_status"], "accepted")
                self.assertEqual(after["trace"]["maxDepth"], 2)
                self.assertEqual(len(after["trace"]["workflows"]), 1)

    def test_legacy_schema_inventory_is_identical(self):
        for flags in LEGACY_FLAGS:
            with self.subTest(flags=flags):
                before = self.run_workflow(JAVASCRIPT[1], flags, pinned=True, inspect="schemas")
                after = self.run_workflow(JAVASCRIPT[1], flags, inspect="schemas")
                self.assert_success(after)
                self.assertEqual(before, after)
                self.assertGreaterEqual(len(after["result"]), 15)
                self.assertEqual(after["trace"]["agents"], [])

    def test_evo_changes_only_the_planner_lineage_schema(self):
        before = self.assert_success(self.run_workflow(JAVASCRIPT[1], pinned=True, inspect="schemas"))
        after = self.assert_success(self.run_workflow("evo/kernel_lane.js", {
            "optimizer": "evo", "evolution_source_paths": ["kernel.py"],
        }, inspect="schemas"))
        properties = after["PLAN_SCHEMA"]["properties"]["directions"]["items"]["properties"]
        self.assertEqual(properties.pop("parent_id"), {"type": "string"})
        self.assertEqual(properties.pop("donor_ids"), {"type": "array", "items": {"type": "string"}})
        self.assertEqual(after, before)

    def test_unknown_optimizers_reject_before_any_agent_or_child_workflow(self):
        for entry in JAVASCRIPT:
            for optimizer in ("evp", "evolution", "greed", "null", "evo/../greedy", ["evo", "greedy"], {"name": "evo"}):
                with self.subTest(entry=entry, optimizer=optimizer):
                    outcome = self.run_workflow(entry, {"optimizer": optimizer})
                    self.assert_rejected_before_agents(outcome, "optimizer must be greedy or evo")
                    self.assertEqual(outcome["trace"]["workflows"], [])

    def test_evo_conflicting_custom_worker_is_rejected(self):
        outcome = self.run_workflow(JAVASCRIPT[0], {
            "optimizer": "evo", "kernel_lane_script": "/fixture/custom.js",
            "evolution_source_paths": ["kernel.py"],
        })
        self.assert_rejected_before_agents(outcome, "cannot override kernel_lane_script")
        self.assertEqual(outcome["trace"]["workflows"], [])

    def test_evo_dispatcher_bakeoff_is_explicitly_unsupported(self):
        for mode in ("bakeoff", "  BaKeOfF  "):
            with self.subTest(mode=mode):
                outcome = self.run_workflow(JAVASCRIPT[0], {
                    "optimizer": "evo", "mode": mode, "evolution_source_paths": ["kernel.py"],
                })
                self.assert_rejected_before_agents(outcome, "not bakeoff")
                self.assertEqual(outcome["trace"]["workflows"], [])

    def test_direct_evo_worker_cannot_bypass_unsupported_bakeoff(self):
        for entry in (JAVASCRIPT[1], "evo/kernel_lane.js"):
            with self.subTest(entry=entry):
                outcome = self.run_workflow(entry, {
                    "optimizer": "evo", "mode": "bakeoff", "evolution_source_paths": ["kernel.py"],
                })
                self.assert_rejected_before_agents(outcome, "bakeoff")

    def test_direct_evo_worker_cannot_silently_ignore_custom_script_conflict(self):
        for entry in (JAVASCRIPT[1], "evo/kernel_lane.js"):
            with self.subTest(entry=entry):
                outcome = self.run_workflow(entry, {
                    "optimizer": "evo", "kernel_lane_script": "/fixture/custom.js",
                    "evolution_source_paths": ["kernel.py"],
                })
                self.assert_rejected_before_agents(outcome, "kernel_lane_script")

    def test_evo_direct_dispatcher_and_e2e_use_only_one_child_level(self):
        for entry, expected_depth, child_count in (
            ("evo/kernel_lane.js", 1, 0),
            (JAVASCRIPT[0], 2, 1),
            (JAVASCRIPT[1], 2, 1),
            (JAVASCRIPT[2], 2, 1),
        ):
            for optimizer in ("evo", "  EvO  "):
                with self.subTest(entry=entry, optimizer=optimizer):
                    outcome = self.run_workflow(entry, {
                        "optimizer": optimizer, "evolution_source_paths": ["kernel.py"],
                    })
                    result = self.assert_success(outcome)
                    self.assertEqual(result["validation_status"], "accepted", result)
                    self.assertEqual(outcome["trace"]["maxDepth"], expected_depth)
                    self.assertEqual(len(outcome["trace"]["workflows"]), child_count)
                    for call in outcome["trace"]["workflows"]:
                        self.assertEqual(Path(call["scriptPath"]).resolve(), ROOT / "evo/kernel_lane.js")
                    labels = [call["options"]["label"] for call in outcome["trace"]["agents"]]
                    self.assertEqual(labels.count("evo:init"), 1)
                    self.assertEqual(labels.count("evo:prepare"), 2)
                    self.assertEqual(labels.count("evo:record-batch"), 1)
                    self.assertEqual(labels.count("director:validate"), 1)
                    self.assertEqual(sum(label.startswith("verify ") for label in labels), 2)
                    self.assertIn("integrate r1", labels)
                    self.assertIn("commit r1", labels)
                    if entry != JAVASCRIPT[2]:
                        self.assertEqual(result["evolution"]["errors"], [])

    def test_evo_worker_requires_explicit_opt_in_and_safe_source_paths(self):
        for flags in (*LEGACY_FLAGS, {"optimizer": "invalid"}):
            with self.subTest(flags=flags):
                outcome = self.run_workflow("evo/kernel_lane.js", {**flags, "evolution_source_paths": ["kernel.py"]})
                self.assert_rejected_before_agents(outcome, "requires optimizer=evo")
        for source_paths in (None, [], "kernel.py", [""], [3], ["/absolute.py"], ["../escape.py"], ["a/../b.py"], ["a//b.py"], ["a\n.py"]):
            with self.subTest(source_paths=source_paths):
                outcome = self.run_workflow("evo/kernel_lane.js", {
                    "optimizer": "evo", "evolution_source_paths": source_paths,
                })
                self.assert_rejected_before_agents(outcome, "evolution_source_paths")

    def test_evo_author_route_and_ordinary_verifier_prompts_are_preserved(self):
        arguments = {"mode": "author", "optimizer": "evo", "evolution_source_paths": ["kernel.py"]}
        legacy = self.run_workflow(JAVASCRIPT[1], {"mode": "author"}, pinned=True)
        for entry in (JAVASCRIPT[0], JAVASCRIPT[1], "evo/kernel_lane.js"):
            with self.subTest(entry=entry):
                evolved = self.run_workflow(entry, arguments)
                result = self.assert_success(evolved)
                self.assertTrue(result["authored"])
                self.assertEqual(result["mode"], "author")
                self.assertEqual(result["evolution"]["errors"], [])
                for label in ("director:setup", "author:triton", "tech_lead:analyze", "benchmark_engineer",
                              "profile_engineer:baseline", "verify fixture_0", "verify fixture_1", "director:validate"):
                    before = next(call for call in legacy["trace"]["agents"] if call["options"]["label"] == label)
                    after = next(call for call in evolved["trace"]["agents"] if call["options"]["label"] == label)
                    prompt = after["prompt"]
                    if label.startswith("verify "):
                        index = label.rsplit("_", 1)[1]
                        mutable_patch = f"/fixture/eval/round_1/engineer_{index}/best_patch.diff"
                        digest = hashlib.sha256(mutable_patch.encode()).hexdigest()
                        immutable_patch = f"/fixture/eval/evolution_archive/objects/{digest}"
                        self.assertIn(immutable_patch, prompt)
                        prompt = prompt.replace(immutable_patch, mutable_patch)
                    self.assertEqual(before["prompt"], prompt, label)
                    self.assertEqual(before["options"], after["options"], label)

    def test_failed_evo_prepare_cannot_reach_engineers_or_verifiers(self):
        outcome = self.run_workflow("evo/kernel_lane.js", {
            "optimizer": "evo", "evolution_source_paths": ["kernel.py"],
        }, replies={"evo:prepare": {"exit_code": 1, "stdout": "", "stderr": "synthetic prepare failure"}})
        result = self.assert_success(outcome)
        errors = result["evolution"]["errors"]
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(error["operation"] == "prepare" for error in errors))
        labels = [call["options"]["label"] for call in outcome["trace"]["agents"]]
        self.assertFalse(any(label.startswith(("eng ", "verify ", "commit ")) for label in labels))
        self.assertIn("director:validate", labels)

    def test_bad_evo_init_receipts_stop_before_planning(self):
        for receipt in ({"ok": False, "command": "init"}, {"ok": True, "command": "query"},
                        {"ok": True, "command": "init", "archive": "/wrong/archive"}, [], "invalid JSON"):
            with self.subTest(receipt=receipt):
                stdout = receipt if isinstance(receipt, str) else json.dumps(receipt)
                outcome = self.run_workflow("evo/kernel_lane.js", {
                    "optimizer": "evo", "evolution_source_paths": ["kernel.py"],
                }, replies={"evo:init": {"exit_code": 0, "stdout": stdout, "stderr": ""}})
                self.assertIsNotNone(outcome["error"])
                labels = [call["options"]["label"] for call in outcome["trace"]["agents"]]
                self.assertFalse(any(label.startswith(("tech_lead:plan", "eng ", "verify ")) for label in labels))

    def test_nonstring_flags_reject_consistently_before_work(self):
        module = self.load_runner(self.current_runner, {})
        for optimizer in (False, True, 0, 1, [], ["evo"], ["greedy"], {"name": "evo"}):
            with self.subTest(optimizer=optimizer):
                with self.assertRaisesRegex(ValueError, "optimizer must be greedy or evo"):
                    self.map_handoff(module, {"optimizer": optimizer})
                for entry in (*JAVASCRIPT, "evo/kernel_lane.js"):
                    self.assert_rejected_before_agents(self.run_workflow(entry, {"optimizer": optimizer}), "optimizer")

    def test_custom_nested_caller_must_select_evo_worker_directly(self):
        options = {"optimizer": "evo", "evolution_source_paths": ["kernel.py"]}
        nested_legacy = self.run_workflow("kernel_workflow/kernel_lane.js", options, initialDepth=2)
        self.assert_rejected_before_agents(nested_legacy, "nesting")
        nested_evo = self.run_workflow("evo/kernel_lane.js", options, initialDepth=2)
        self.assert_success(nested_evo)
        self.assertEqual(nested_evo["trace"]["maxDepth"], 2)

    def test_e2e_lane_args_and_all_bound_wrappers_preserve_legacy_defaults(self):
        for flags in LEGACY_FLAGS:
            for identity in ({}, {"expected_gfx": "gfx1201", "expected_target": "r9700"}):
                with self.subTest(flags=flags, identity=identity):
                    arguments = {**flags, **identity, "ablation_arm": "B6", "ablation_seed": "fixed"}
                    probe = {"kernel_path": "/fixture/head", "workflow_dir": str(ROOT / "kernel_workflow"),
                             "use_learned_kb": "true", "gpu_ids": "5", "budget": 2}
                    before = self.run_workflow(JAVASCRIPT[2], arguments, pinned=True,
                                               inspect="e2e", probeArgs=probe, stubWorkflows=True)
                    after = self.run_workflow(JAVASCRIPT[2], arguments,
                                              inspect="e2e", probeArgs=probe, stubWorkflows=True)
                    self.assert_success(after)
                    self.assertEqual(after, before)
                    self.assertEqual(len(after["trace"]["workflows"]), 4)
                    for call in after["trace"]["workflows"]:
                        self.assertNotIn("optimizer", call["args"])
                        self.assertEqual(call["args"]["gpu_ids"], "5")
                        self.assertEqual(call["args"]["ablation_arm"], "B6")
                        self.assertEqual(call["args"]["use_learned_kb"], "false" if identity else "true")

    def test_e2e_evo_lane_args_reach_every_bound_wrapper(self):
        probe = {"kernel_path": "/fixture/head", "workflow_dir": str(ROOT / "kernel_workflow")}
        outcome = self.run_workflow(JAVASCRIPT[2], {
            "optimizer": "evo", "evolution_source_paths": ["kernel.py"],
            "kernel_workflow_dir": "/fixture/custom-kernel///", "llm_stats": False,
        }, inspect="e2e", probeArgs=probe, stubWorkflows=True)
        self.assert_success(outcome)
        self.assertEqual(len(outcome["trace"]["workflows"]), 4)
        for call in outcome["trace"]["workflows"]:
            self.assertEqual(call["scriptPath"], "/fixture/custom-kernel/../evo/kernel_lane.js")
            self.assertEqual(call["args"]["optimizer"], "evo")
            self.assertEqual(call["args"]["evolution_source_paths"], ["kernel.py"])
            self.assertEqual(call["args"]["llm_stats"], "false")
            self.assertEqual(call["args"]["use_learned_kb"], "false")

    def test_every_e2e_lane_call_uses_the_shared_argument_builder(self):
        source = self.current[JAVASCRIPT[2]]
        calls = list(re.finditer(r"(\w+)\(\{ scriptPath: KERNEL_WF_SCRIPT \},\s*", source))
        self.assertGreaterEqual(len(calls), 6)
        for match in calls:
            with self.subTest(offset=match.start()):
                caller = match.group(1)
                if caller == "workflow":
                    self.assertTrue(source[match.end():].startswith("laneArgs("))
                else:
                    self.assertIn(caller, ("fastBoundedWorkflow", "deepBoundedWorkflow", "reserveBoundedWorkflow"))

    def load_runner(self, source, environment):
        module = types.ModuleType("dispatch_runner_fixture")
        module.__file__ = str(ROOT / RUNNER)
        with mock.patch.dict(os.environ, {**SAFE_ENV, **environment}, clear=True):
            with mock.patch.object(sys, "path", [str(ROOT), *sys.path]):
                with mock.patch.object(sys, "dont_write_bytecode", True):
                    exec(compile(source, module.__file__, "exec"), module.__dict__)
        module.datetime = FixedDatetime
        module._dt = types.SimpleNamespace(datetime=FixedDatetime, timezone=datetime.timezone)
        module.resolve_tracelens_report = mock.Mock(return_value={})
        return module

    def map_handoff(self, module, handoff):
        with mock.patch.dict(os.environ, SAFE_ENV, clear=True):
            with mock.patch.object(uuid, "uuid4", return_value=uuid.UUID(int=1)):
                with mock.patch.object(subprocess, "run", side_effect=AssertionError("Unexpected host process")):
                    return module.map_args(copy.deepcopy(handoff), dry_run=True)

    def test_python_mapping_and_host_selection_match_pinned_main(self):
        for environment, expected_runtime in (({}, False), ({"GEAK_AGENT_BACKEND": "codex"}, True),
                                              ({"GEAK_AGENT_BACKEND": "claude"}, True),
                                              ({"GEAK_AGENT_PROFILE": "fixture-profile"}, True)):
            with self.subTest(environment=environment):
                before = self.load_runner(self.pinned_runner, environment)
                after = self.load_runner(self.current_runner, environment)
                self.assertEqual(before.USE_RUNTIME, expected_runtime)
                self.assertEqual(after.USE_RUNTIME, expected_runtime)
                self.assertEqual(before.EFFECTIVE_BACKEND, after.EFFECTIVE_BACKEND)
                for identity in ({}, {"expected_gfx": "gfx1201", "expected_target": "r9700"}):
                    handoff = {"model_path": "/fixture/model", "exp_root": "/fixture/exp",
                               "eval_dir": "/fixture/eval", "tp": 2, "gpu_ids": "2,5", **identity}
                    for flags in LEGACY_FLAGS:
                        self.assertEqual(self.map_handoff(before, {**handoff, **flags}),
                                         self.map_handoff(after, {**handoff, **flags}))
                    legacy = self.map_handoff(before, handoff)
                    evolved = self.map_handoff(after, {**handoff, "optimizer": "  EvO  ",
                                                      "evolution_source_paths": ["kernel.py"]})
                    self.assertEqual(evolved.pop("optimizer"), "evo")
                    self.assertEqual(evolved.pop("evolution_source_paths"), ["kernel.py"])
                    self.assertEqual(evolved, legacy)
                    self.assertEqual(after.USE_RUNTIME, expected_runtime)

    def test_python_rejects_unknown_optimizer_and_missing_sources_before_host_probe(self):
        module = self.load_runner(self.current_runner, {})
        handoff = {"model_path": "/fixture/model", "exp_root": "/fixture/exp"}
        for optimizer in ("evolution", "evp", "null", {"name": "evo"}):
            with self.subTest(optimizer=optimizer):
                with self.assertRaisesRegex(ValueError, "optimizer must be greedy or evo"):
                    self.map_handoff(module, {**handoff, "optimizer": optimizer})
        for source_paths in (None, [], "kernel.py", [""], [None], [3]):
            with self.subTest(source_paths=source_paths):
                with self.assertRaisesRegex(ValueError, "evolution_source_paths"):
                    self.map_handoff(module, {**handoff, "optimizer": "evo",
                                               "evolution_source_paths": source_paths})

    def test_host_selection_source_and_shared_prompt_files_are_unchanged(self):
        for function in ("_side_env_names", "_derive_agent_from_env"):
            definitions = []
            for source in (self.pinned_runner, self.current_runner):
                node = next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == function)
                definitions.append(ast.dump(node, include_attributes=False))
            self.assertEqual(*definitions)
        for name in ("AGENT_BACKEND", "AGENT_PROFILE", "AUTO_BACKEND", "EFFECTIVE_BACKEND", "USE_RUNTIME"):
            assignments = []
            for source in (self.pinned_runner, self.current_runner):
                node = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
                assignments.append(ast.dump(node, include_attributes=False))
            self.assertEqual(*assignments)
        result = subprocess.run(
            ["git", "diff", "--exit-code", PINNED_MAIN, "--", "kernel_workflow/roles",
             "kernel_workflow/knowledge", "interface/effective_config.py", "interface/runtime"],
            cwd=ROOT, env=SAFE_ENV, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
