"""CPU dispatch regression with synthetic agents; EVO-on is compared with EVO-off in the same tree.

Run: python3 -B -m unittest discover -s evo/tests -p test_dispatch.py -v
These tests execute workflow JavaScript, not either production Workflow host.
Agent replies, measurements, archive commands and GPU identity are synthetic.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import types
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[2]
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


class FixedDatetime(datetime.datetime):
    @classmethod
    def now(cls, timezone=None):
        return cls(2026, 11, 11, 12, 0, 0, tzinfo=timezone)


class DispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = {filename: (ROOT / filename).read_text() for filename in JAVASCRIPT}
        cls.current_runner = (ROOT / RUNNER).read_text()

    def run_workflow(self, entry, args=None, **options):
        sources = self.current
        if options.get("inspect") == "schemas":
            options.pop("inspect")
            source = sources[entry]
            marker = "phase('Setup');"
            self.assertEqual(source.count(marker), 1, "Lane schema inspection seam changed")
            names = re.findall(r"const ([A-Z_]+_SCHEMA)\s*=", source[:source.index(marker)])
            self.assertGreaterEqual(len(names), 15, "Lane schema capture is incomplete")
            sources = {**sources, entry: source.replace(marker, f"return {{ {', '.join(names)} }};\n{marker}", 1)}
        request = {
            "root": str(ROOT), "sources": sources,
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

    def test_shared_lane_has_no_generated_worker_or_extension(self):
        for filename in ("evo/kernel_lane.js", "evo/build_workflow.py", "evo/extension.js"):
            with self.subTest(filename=filename):
                self.assertFalse((ROOT / filename).exists())
        for filename, source in self.current.items():
            with self.subTest(filename=filename):
                self.assertNotIn("/../evo/kernel_lane.js", source)

    def test_evo_changes_only_the_planner_lineage_schema(self):
        before = self.assert_success(self.run_workflow(JAVASCRIPT[1], inspect="schemas"))
        after = self.assert_success(self.run_workflow(JAVASCRIPT[1], {
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

    def test_shared_lane_evo_cannot_bypass_unsupported_bakeoff(self):
        outcome = self.run_workflow(JAVASCRIPT[1], {
            "optimizer": "evo", "mode": "bakeoff", "evolution_source_paths": ["kernel.py"],
        })
        self.assert_rejected_before_agents(outcome, "bakeoff")
        self.assertEqual(outcome["trace"]["workflows"], [])

    def test_shared_lane_evo_cannot_silently_ignore_custom_script_conflict(self):
        outcome = self.run_workflow(JAVASCRIPT[1], {
            "optimizer": "evo", "kernel_lane_script": "/fixture/custom.js",
            "evolution_source_paths": ["kernel.py"],
        })
        self.assert_rejected_before_agents(outcome, "kernel_lane_script")
        self.assertEqual(outcome["trace"]["workflows"], [])

    def test_evo_direct_dispatcher_and_e2e_use_only_one_child_level(self):
        for entry, expected_depth, child_count in (
            (JAVASCRIPT[0], 2, 1),
            (JAVASCRIPT[1], 1, 0),
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
                        self.assertEqual(Path(call["scriptPath"]).resolve(), ROOT / JAVASCRIPT[1])
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

    def test_legacy_ignores_evolution_sources_without_initializing_evo(self):
        for entry in JAVASCRIPT:
            for flags in LEGACY_FLAGS:
                for source_paths in ([], ["../escape.py"], ["kernel.py"]):
                    with self.subTest(entry=entry, flags=flags, source_paths=source_paths):
                        arguments = {**flags, "evolution_source_paths": source_paths}
                        before = self.run_workflow(entry, flags)          # same tree, no source list
                        after = self.run_workflow(entry, arguments)
                        self.assert_success(after)
                        # The dispatcher forwards caller arguments verbatim to its child lane; that
                        # passthrough is the only expected difference when EVO is off.
                        for call in after["trace"]["workflows"]:
                            if entry == JAVASCRIPT[0]:
                                self.assertEqual(call["args"].pop("evolution_source_paths"), source_paths)
                            self.assertNotIn("evolution_source_paths", call["args"])
                        self.assertEqual(after, before)
                        labels = [call["options"]["label"] for call in after["trace"]["agents"]]
                        self.assertFalse(any(label.startswith("evo:") for label in labels))

    def assert_ordinary_lane_fallback(self, outcome, message, entry=JAVASCRIPT[1]):
        """EVO could not start: the error is recorded and the ordinary lane runs with no EVO inputs."""
        result = self.assert_success(outcome)
        calls = outcome["trace"]["agents"]
        labels = [call["options"]["label"] for call in calls]
        self.assertFalse(any(label.startswith("evo:") and label != "evo:init" for label in labels), labels)
        self.assertFalse(any("\n- EVOLUTION: " in call["prompt"] or "evo/roles/" in call["prompt"] for call in calls))
        self.assertTrue(any(label.startswith(("eng ", "deep ")) for label in labels), labels)
        for call in calls:
            if call["options"]["label"].startswith(("tech_lead:plan", "tech_lead:replan")):
                lineage = call["options"]["schema"]["properties"]["directions"]["items"]["properties"]
                self.assertNotIn("parent_id", lineage)
                self.assertNotIn("donor_ids", lineage)
        if entry != JAVASCRIPT[2]:
            self.assertEqual(result["validation_status"], "accepted")
            errors = result["evolution"]["errors"]
            self.assertEqual([(error["operation"], error["fallback"]) for error in errors],
                             [("init", "ordinary_lane")])
            self.assertIn(message, errors[0]["message"])

    def test_lane_script_uses_no_native_forbidden_constructs(self):
        # The native Workflow host has no clock/randomness/process/require (resume parity); see
        # interface/runtime/engine/conformance.mjs A-forbidden. Same patterns, applied to lane code.
        code = re.sub(r"/\*.*?\*/", "", (ROOT / JAVASCRIPT[1]).read_text(), flags=re.DOTALL)
        code = "\n".join(line.split("//", 1)[0] if not re.search(r"['\"`][^'\"`]*//", line) else line
                         for line in code.splitlines())
        for name, pattern in (("Date.now", r"\bDate\.now\b"), ("Math.random", r"\bMath\.random\b"),
                              ("new Date", r"\bnew\s+Date\b"), ("process.", r"\bprocess\."),
                              ("require(", r"\brequire\s*\(")):
            self.assertIsNone(re.search(pattern, code), name)

    def test_record_batch_passes_the_report_digest(self):
        outcome = self.run_workflow(JAVASCRIPT[1], {"optimizer": "evo", "evolution_source_paths": ["kernel.py"]})
        result = self.assert_success(outcome)
        record = next(call for call in outcome["trace"]["agents"] if call["options"]["label"] == "evo:record-batch")
        script = record["prompt"].split("```bash\n", 1)[1].split("\n```", 1)[0]
        words = shlex.split(script.splitlines()[-1])
        report_json = words[words.index("--report-json") + 1]
        self.assertEqual(words[words.index("--report-sha256") + 1],
                         hashlib.sha256(report_json.encode("utf-8")).hexdigest())
        self.assertEqual(json.loads(report_json)[0]["id"], "r1_d0")
        self.assertNotIn("operation_seconds", result["evolution"])   # workflow scripts have no clock

    def test_unusable_evo_source_list_falls_back_to_the_ordinary_lane(self):
        for source_paths in (
            "kernel.py", [""], [None], [3], ["/absolute.py"], ["../escape.py"],
            ["a/../b.py"], ["a//b.py"], ["a\n.py"], ["./kernel.py"], ["a/./b.py"],
            ["a/"], ["a\\b.py"], [".git/config"], ["a/.GiT/config"], ["a\t.py"],
            ["a\x00.py"], ["a\x7f.py"], ["\u00e9.py"], ["kernel.py", "kernel.py"],
            ["a", "a/b.py"], ["a/b.py", "a"], ["a" * 1025],
            [f"kernel_{index}.py" for index in range(257)],
        ):
            for entry in JAVASCRIPT:
                with self.subTest(entry=entry, source_paths=source_paths):
                    outcome = self.run_workflow(entry, {
                        "optimizer": "evo", "evolution_source_paths": source_paths,
                    })
                    if entry == JAVASCRIPT[2] and not isinstance(source_paths, list):
                        # A non-list is a caller typo, rejected once before any lane starts.
                        self.assert_rejected_before_agents(outcome, "evolution_source_paths")
                    else:
                        self.assert_ordinary_lane_fallback(outcome, "evolution_source_paths", entry)

    def test_evo_uses_main_editable_sources_even_with_narrow_legacy_list(self):
        editable = ["kernel.py", "knn_wrapper.py", "src/knn.cpp"]
        for entry in JAVASCRIPT:
            for supplemental in ({}, {"evolution_source_paths": None}, {"evolution_source_paths": []},
                                 {"evolution_source_paths": ["kernel.py"]}):
                with self.subTest(entry=entry, supplemental=supplemental):
                    outcome = self.run_workflow(entry, {"optimizer": "evo", **supplemental},
                                                replies={"tech_lead:analyze": {"kernel_type": "fixture",
                                                         "roadmap_summary": "Ordinary main sources",
                                                         "modifiable_files": editable}})
                    self.assert_success(outcome)
                    planners = [call for call in outcome["trace"]["agents"]
                                if call["options"]["label"].startswith("tech_lead:plan ")]
                    self.assertTrue(planners)
                    for call in planners:
                        evolution = json.loads(call["prompt"].split("\n- EVOLUTION: ", 1)[1].splitlines()[0])
                        self.assertEqual(evolution["source_paths"], editable)
                    initialization = next(call for call in outcome["trace"]["agents"]
                                          if call["options"]["label"] == "evo:init")
                    for source in editable:
                        self.assertIn("'--source-path' '" + source + "'", initialization["prompt"])

    def test_evo_supplemental_new_file_does_not_replace_main_editable_sources(self):
        outcome = self.run_workflow(JAVASCRIPT[1], {
            "optimizer": "evo", "evolution_source_paths": ["new_source.py"],
        })
        self.assert_success(outcome)
        planner = next(call for call in outcome["trace"]["agents"]
                       if call["options"]["label"].startswith("tech_lead:plan "))
        evolution = json.loads(planner["prompt"].split("\n- EVOLUTION: ", 1)[1].splitlines()[0])
        self.assertEqual(evolution["source_paths"], ["kernel.py", "new_source.py"])

    def test_unusable_analyzed_scope_falls_back_before_archive_initialization(self):
        for editable in ([], "kernel.py", ["../outside.py"], [".git/config"], [None],
                         ["kernel.py", "kernel.py"], ["src", "src/kernel.py"]):
            with self.subTest(editable=editable):
                outcome = self.run_workflow(JAVASCRIPT[1], {"optimizer": "evo"},
                                            replies={"tech_lead:analyze": {"kernel_type": "fixture",
                                                     "roadmap_summary": "Invalid source fixture",
                                                     "modifiable_files": editable}})
                self.assert_ordinary_lane_fallback(outcome, "")
                labels = [call["options"]["label"] for call in outcome["trace"]["agents"]]
                self.assertNotIn("evo:init", labels)

    def test_evo_accepts_explicit_sources_when_optional_analyze_sources_are_absent(self):
        for entry in JAVASCRIPT:
            for fields in ({}, {"modifiable_files": []}):
                with self.subTest(entry=entry, fields=fields):
                    replies = {"tech_lead:analyze": {"kernel_type": "fixture",
                               "roadmap_summary": "Optional source list", **fields}}
                    legacy = self.run_workflow(entry, replies=replies)
                    outcome = self.run_workflow(entry, {
                        "optimizer": "evo", "evolution_source_paths": ["kernel.py"],
                    }, replies=replies)
                    self.assert_success(outcome)
                    labels = [call["options"]["label"] for call in outcome["trace"]["agents"]]
                    ordinary_labels = [call["options"]["label"] for call in legacy["trace"]["agents"]]
                    engineers = [label for label in labels if label.startswith(("eng ", "deep "))]
                    ordinary_engineers = [label for label in ordinary_labels if label.startswith(("eng ", "deep "))]
                    self.assertEqual(len(engineers), 2)
                    self.assertEqual(engineers, ordinary_engineers)
                    self.assertIn("evo:prepare", labels)
                    initialization = next(call for call in outcome["trace"]["agents"]
                                          if call["options"]["label"] == "evo:init")
                    self.assertIn("'--source-path' 'kernel.py'", initialization["prompt"])

    def test_evo_reports_missing_source_scope_across_entries(self):
        for entry in JAVASCRIPT:
            for fields in ({}, {"modifiable_files": []}):
                with self.subTest(entry=entry, fields=fields):
                    outcome = self.run_workflow(entry, {"optimizer": "evo"}, replies={
                        "tech_lead:analyze": {"kernel_type": "fixture",
                            "roadmap_summary": "Missing source list", **fields},
                    })
                    self.assert_ordinary_lane_fallback(
                        outcome, "source files from Analyze modifiable_files or evolution_source_paths", entry)
                    labels = [call["options"]["label"] for call in outcome["trace"]["agents"]]
                    self.assertNotIn("evo:init", labels)

    def test_evo_author_route_and_ordinary_verifier_prompts_are_preserved(self):
        arguments = {"mode": "author", "optimizer": "evo", "evolution_source_paths": ["kernel.py"]}
        legacy = self.run_workflow(JAVASCRIPT[1], {"mode": "author"})
        for entry in (JAVASCRIPT[0], JAVASCRIPT[1]):
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

    def test_evo_role_guidance_pointers_reach_only_planners_and_engineers(self):
        planner_guide = f"{ROOT}/kernel_workflow/../evo/roles/planner.md"
        engineer_guide = f"{ROOT}/kernel_workflow/../evo/roles/engineer.md"
        for guide in (planner_guide, engineer_guide):
            self.assertTrue(Path(guide).is_file(), guide)
        for mode in ("optimize", "author"):
            for specialty in ("memory", "deep_explore"):
                for optimizer in ("greedy", "evo"):
                    with self.subTest(mode=mode, specialty=specialty, optimizer=optimizer):
                        outcome = self.run_workflow(JAVASCRIPT[1], {
                            "mode": mode, "optimizer": optimizer, "budget": 2 if specialty == "deep_explore" else 1,
                            "evolution_source_paths": ["kernel.py"],
                        }, replies={"tech_lead:plan r1": {"stop": False, "directions": [{
                            "id": "fixture_0", "title": "Guidance placement", "specialty": specialty,
                            "focus_files": ["kernel.py"], "prompt": "Synthetic source candidate",
                        }]}})
                        self.assert_success(outcome)
                        planners = engineers = 0
                        for call in outcome["trace"]["agents"]:
                            label, prompt = call["options"]["label"], call["prompt"]
                            is_planner = label.startswith("tech_lead:plan ")
                            is_engineer = label.startswith(("eng ", "deep "))
                            planners += is_planner
                            engineers += is_engineer
                            self.assertEqual(planner_guide in prompt, optimizer == "evo" and is_planner, label)
                            self.assertEqual(engineer_guide in prompt, optimizer == "evo" and is_engineer, label)
                            self.assertEqual("\n- EVOLUTION: " in prompt,
                                             optimizer == "evo" and (is_planner or is_engineer), label)
                            if optimizer == "evo" and (is_planner or is_engineer):
                                evolution = json.loads(prompt.split("\n- EVOLUTION: ", 1)[1].splitlines()[0])
                                fields = {"archive", "query_command", "source_paths"}
                                if is_engineer:
                                    fields.update(("parent_id", "donor_ids", "prepared"))
                                self.assertEqual(set(evolution), fields)
                                self.assertEqual(evolution["source_paths"], ["kernel.py"])
                                self.assertEqual(evolution["archive"], "/fixture/eval/evolution_archive")
                        self.assertEqual(planners, 1)
                        self.assertEqual(engineers, 1)

    def test_forced_replan_preserves_legacy_trace_and_evo_guidance(self):
        replies = {
            "clock pre-r1": {"epoch": 100},
            "clock replan-r1": {"epoch": 100},
            "tech_lead:plan r1": {"stop": True, "directions": []},
            "tech_lead:replan r1#1": {"stop": False, "directions": [{
                "id": "fixture_0", "title": "Replan guidance", "specialty": "memory",
                "focus_files": ["kernel.py"], "prompt": "Synthetic source candidate",
            }]},
        }
        for flags in (*LEGACY_FLAGS, {"optimizer": "evo"}):
            with self.subTest(flags=flags):
                arguments = {**flags, "budget": 1, "deadline_epoch": 10000,
                             "evolution_source_paths": ["kernel.py"]}
                outcome = self.run_workflow(JAVASCRIPT[1], arguments, replies=replies)
                self.assertEqual(self.assert_success(outcome)["validation_status"], "accepted")
                planning = [call for call in outcome["trace"]["agents"]
                            if call["options"]["label"].startswith(("tech_lead:plan ", "tech_lead:replan "))]
                self.assertEqual(len(planning), 2)
                self.assertIn("STOP_NOT_PERMITTED", planning[1]["prompt"])
                if flags.get("optimizer") == "evo":
                    for call in planning:
                        self.assertIn(f"Read {ROOT}/kernel_workflow/../evo/roles/planner.md", call["prompt"])
                    inputs = [json.loads(call["prompt"].split("\n- EVOLUTION: ", 1)[1].splitlines()[0])
                              for call in planning]
                    self.assertEqual(*inputs)
                else:
                    before = self.run_workflow(JAVASCRIPT[1], {**arguments, "optimizer": None}, replies=replies)
                    self.assertEqual(outcome, before)

    def test_failed_evo_prepare_cannot_reach_engineers_or_verifiers(self):
        outcome = self.run_workflow(JAVASCRIPT[1], {
            "optimizer": "evo", "evolution_source_paths": ["kernel.py"],
        }, replies={"evo:prepare": {"exit_code": 1, "stdout": "", "stderr": "synthetic prepare failure"}})
        result = self.assert_success(outcome)
        errors = result["evolution"]["errors"]
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(error["operation"] == "prepare" for error in errors))
        labels = [call["options"]["label"] for call in outcome["trace"]["agents"]]
        self.assertFalse(any(label.startswith(("eng ", "verify ", "commit ")) for label in labels))
        self.assertIn("director:validate", labels)

    def test_bad_evo_init_receipts_fall_back_to_the_ordinary_lane(self):
        for receipt in ({"ok": False, "command": "init"}, {"ok": True, "command": "query"},
                        {"ok": True, "command": "init", "archive": "/wrong/archive"}, [], "invalid JSON"):
            with self.subTest(receipt=receipt):
                stdout = receipt if isinstance(receipt, str) else json.dumps(receipt)
                outcome = self.run_workflow(JAVASCRIPT[1], {
                    "optimizer": "evo", "evolution_source_paths": ["kernel.py"],
                }, replies={"evo:init": {"exit_code": 0, "stdout": stdout, "stderr": ""}})
                self.assert_ordinary_lane_fallback(outcome, "EVO init")

    def test_nonstring_flags_reject_consistently_before_work(self):
        module = self.load_runner(self.current_runner, {})
        for optimizer in (False, True, 0, 1, [], ["evo"], ["greedy"], {"name": "evo"}):
            with self.subTest(optimizer=optimizer):
                with self.assertRaisesRegex(ValueError, "optimizer must be greedy or evo"):
                    self.map_handoff(module, {"optimizer": optimizer})
                for entry in JAVASCRIPT:
                    self.assert_rejected_before_agents(self.run_workflow(entry, {"optimizer": optimizer}), "optimizer")

    def test_custom_nested_caller_uses_shared_lane_without_extra_depth(self):
        for flags in (*LEGACY_FLAGS, {"optimizer": "evo"}, {"optimizer": "  EvO  "}):
            for mode in ("optimize", "author"):
                with self.subTest(flags=flags, mode=mode):
                    options = {**flags, "mode": mode, "evolution_source_paths": ["kernel.py"]}
                    nested = self.run_workflow(JAVASCRIPT[1], options, initialDepth=2)
                    result = self.assert_success(nested)
                    self.assertEqual(result["validation_status"], "accepted")
                    self.assertEqual(nested["trace"]["maxDepth"], 2)
                    self.assertEqual(nested["trace"]["workflows"], [])
                    if str(flags.get("optimizer", "")).strip().lower() == "evo":
                        self.assertEqual(result["evolution"]["errors"], [])
                    else:
                        before = self.run_workflow(JAVASCRIPT[1], {"mode": mode}, initialDepth=2)
                        self.assertEqual(nested, before)

    def test_e2e_lane_args_and_all_bound_wrappers_preserve_legacy_defaults(self):
        for flags in LEGACY_FLAGS:
            for identity in ({}, {"expected_gfx": "gfx1201", "expected_target": "r9700"},
                             {"expected_gfx": "gfx1151", "expected_target": "unknown"}):
                with self.subTest(flags=flags, identity=identity):
                    arguments = {**flags, **identity, "ablation_arm": "B6", "ablation_seed": "fixed"}
                    probe = {"kernel_path": "/fixture/head", "workflow_dir": str(ROOT / "kernel_workflow"),
                             "use_learned_kb": "true", "gpu_ids": "5", "budget": 2}
                    before = self.run_workflow(JAVASCRIPT[2], {**identity, "ablation_arm": "B6", "ablation_seed": "fixed"},
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
            self.assertEqual(call["scriptPath"], "/fixture/custom-kernel/kernel_lane.js")
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

    def test_python_legacy_flags_and_evo_mapping_match_evo_off(self):
        for environment, expected_runtime in (({}, False), ({"GEAK_AGENT_BACKEND": "codex"}, True),
                                              ({"GEAK_AGENT_BACKEND": "claude"}, True),
                                              ({"GEAK_AGENT_PROFILE": "fixture-profile"}, True)):
            with self.subTest(environment=environment):
                after = self.load_runner(self.current_runner, environment)
                self.assertEqual(after.USE_RUNTIME, expected_runtime)
                for identity in ({}, {"expected_gfx": "gfx1201", "expected_target": "r9700"},
                                 {"expected_gfx": "gfx1151", "expected_target": "unknown"}):
                    handoff = {"model_path": "/fixture/model", "exp_root": "/fixture/exp",
                               "eval_dir": "/fixture/eval", "tp": 2, "gpu_ids": "2,5", **identity}
                    legacy = self.map_handoff(after, handoff)
                    for flags in LEGACY_FLAGS:
                        self.assertEqual(self.map_handoff(after, {**handoff, **flags}), legacy)
                    evolved = self.map_handoff(after, {**handoff, "optimizer": "  EvO  ",
                                                      "evolution_source_paths": ["kernel.py"]})
                    self.assertEqual(evolved.pop("optimizer"), "evo")
                    self.assertEqual(evolved.pop("evolution_source_paths"), ["kernel.py"])
                    self.assertEqual(evolved, legacy)
                    self.assertEqual(after.USE_RUNTIME, expected_runtime)

    def test_python_rejects_unknown_optimizer_and_invalid_supplemental_sources(self):
        module = self.load_runner(self.current_runner, {})
        handoff = {"model_path": "/fixture/model", "exp_root": "/fixture/exp"}
        for optimizer in ("evolution", "evp", "null", {"name": "evo"}):
            with self.subTest(optimizer=optimizer):
                with self.assertRaisesRegex(ValueError, "optimizer must be greedy or evo"):
                    self.map_handoff(module, {**handoff, "optimizer": optimizer})
        for source_paths in ("kernel.py", [""], [None], [3]):
            with self.subTest(source_paths=source_paths):
                with self.assertRaisesRegex(ValueError, "evolution_source_paths"):
                    self.map_handoff(module, {**handoff, "optimizer": "evo",
                                               "evolution_source_paths": source_paths})

    def test_python_evo_flag_alone_keeps_main_host_selection(self):
        module = self.load_runner(self.current_runner, {})
        handoff = {"model_path": "/fixture/model", "exp_root": "/fixture/exp"}
        legacy = self.map_handoff(module, handoff)
        evolved = self.map_handoff(module, {**handoff, "optimizer": "evo"})
        self.assertEqual(evolved.pop("optimizer"), "evo")
        self.assertNotIn("evolution_source_paths", evolved)
        self.assertEqual(evolved, legacy)


if __name__ == "__main__":
    unittest.main()
