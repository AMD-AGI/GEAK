# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the v4 attempt handoff and frozen UT on CPU."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[3]
SKILLS = ROOT / "kernel_skills"


class KernelSkillsHooks(unittest.TestCase):
    def test_prompt_delegates_to_the_kernel_supervisor(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "work with spaces"
            prompt = Path(tmp) / "prompt"
            steer = 'SEARCH=avo and literal $(false)'
            env = dict(os.environ, GEAK_ROOT=str(ROOT), K="task", ATTEMPT="2",
                       TASK_DIR=str(run / "task"), KERNEL_SRC_ALL="main.py nested/helper.py",
                       RUN_DIR=str(run), EVAL_DIR=str(run / "attempt_2"), WORK_ROOT=tmp,
                       GPU_IDS="0,1", ATTEMPT_CAP="15", BUDGET="3", MIN_IMPROVE="0.02",
                       BEST_SPEEDUP="1.1", KEEP_THRESHOLD="1.04", MEASURE_TIMEOUT="5",
                       EXTRA_STEER=steer, PROMPT_OUT=str(prompt))
            subprocess.run(["bash", METHOD / "prompt.sh"], env=env, check=True)
            text = prompt.read_text()
            for value in (METHOD / "SKILL.md", SKILLS / "v4/kernel-supervisor/SKILL.md",
                          run / "attempt_2/kernel-skills", run / "HANDOFF.md",
                          "GPUS=0,1", "KERNEL_SRC_ALL=main.py nested/helper.py", steer):
                self.assertIn(str(value), text)

    def test_round_team_can_change_and_combine_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision = Path(tmp) / "decision.json"
            cmd = [sys.executable, SKILLS / "bin/check_decision.py", "--run", tmp,
                   "--decision", decision, "--strategy", SKILLS / "v4/search/evo"]
            investigate = dict(op="investigate", mode="instrument", question="Which timer?",
                               why="Two native timing samples disagree")
            measure = dict(op="measure", nodes=["c000"], why="Compare against the frozen UT")

            def check(team, actions, *, require_team=True, max_heavy=None):
                doc = dict(round=1, strategy="evo", actions=actions)
                if team is not None:
                    doc["team"] = team
                decision.write_text(json.dumps(doc))
                options = ["--require-team"] if require_team else []
                if max_heavy is not None:
                    options.extend(["--max-heavy", str(max_heavy)])
                return subprocess.run(cmd + options, capture_output=True, text=True)

            solo = {"timing_analyst": "Use benchmark and investigator. Report timing to the supervisor."}
            split = {"analyst": "Send the diagnosis to verifier.", "verifier": "Rerun the frozen UT."}
            for team, actions in [
                (solo, [dict(investigate, owner="timing_analyst"), dict(measure, owner="timing_analyst")]),
                (split, [dict(investigate, owner="analyst"), dict(measure, owner="verifier")]),
                ({}, [dict(op="stop", why="Attempt budget exhausted")]),
            ]:
                result = check(team, actions)
                self.assertEqual(result.returncode, 0, result.stdout)
            bad = check(split, [dict(measure, owner="timing_analyst")])
            self.assertEqual(bad.returncode, 1)
            self.assertIn("has no assignment", bad.stdout)
            self.assertNotEqual(check(None, [measure]).returncode, 0)
            self.assertEqual(check(None, [measure], require_team=False).returncode, 0)
            sequential = [dict(measure, owner="verifier")] * 3
            self.assertEqual(check(split, sequential).returncode, 0)
            self.assertNotEqual(check(split, sequential, max_heavy=2).returncode, 0)
            self.assertNotEqual(check(None, sequential, require_team=False).returncode, 0)

    def test_role_workspace_measures_against_the_phase_local_frozen_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task with spaces"
            golden = Path("captures/decode/reference_io.pt")
            (task / golden).parent.mkdir(parents=True)
            (task / golden).write_text("42")
            (task / "kernel_src").mkdir()
            (task / "kernel_src/answer").write_text("42")
            (task / "unittest.py").write_text(
                'from pathlib import Path\n'
                'assert Path("kernel_src/answer").read_text() == '
                'Path("captures/decode/reference_io.pt").read_text()\n'
                'print("CORRECTNESS: PASS\\nGEAK_GEOMEAN_SPEEDUP 1.0")\n'
            )
            run = root / "search"
            ws = run / "cands/c000/ws"
            subprocess.run(["bash", ROOT / "kernel_workflow/scripts/materialize_workspace.sh",
                            "--src", task, "--dst", run / "orig"], check=True, capture_output=True)
            subprocess.run(["bash", SKILLS / "bin/mkws.sh", run, ws], check=True,
                           capture_output=True)
            self.assertEqual((ws / golden).resolve(strict=True), task / golden)
            for answer, status, native_exit in [("42", "1", "0"), ("41", "0", "1")]:
                with self.subTest(answer=answer):
                    (ws / "kernel_src/answer").write_text(answer)
                    receipt = root / f"receipt-{answer}"
                    receipt.mkdir()
                    env = dict(os.environ, TASK_DIR=str(ws), RUN_DIR=str(receipt),
                               MEASURE_TIMEOUT="10", MEASURE_NATIVE_RC_FILE=str(receipt / "rc"))
                    result = subprocess.run(["bash", METHOD / "measure.sh"], env=env,
                                            capture_output=True, text=True, check=True)
                    fields = dict(field.split("=", 1) for field in result.stdout.split())
                    self.assertEqual(fields["STATUS"], status)
                    self.assertEqual(fields["SPEEDUP"], "1.0" if status == "1" else "")
                    self.assertEqual((receipt / "rc").read_text().strip(), native_exit)
            self.assertEqual((task / "kernel_src/answer").read_text(), "42")
