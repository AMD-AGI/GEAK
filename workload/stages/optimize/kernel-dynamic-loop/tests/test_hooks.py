# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only method boundary checks; synthetic scores are not performance evidence."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

METHOD = Path(__file__).resolve().parents[1]
ROOT = METHOD.parents[3]


class Hooks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="kernel-loop-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.task = self.base / "task with spaces"
        (self.task / "kernel_src/nested").mkdir(parents=True)
        self.sources(3, 3)
        (self.task / "cases.py").write_text("EXPECTED = 6\n")
        (self.task / "meta.json").write_text(json.dumps({
            "candidate_bind": {"editable_leaves": ["main.py", "nested/helper.py"]}
        }))
        (self.task / "unittest.py").write_text('''import unittest.mock
from pathlib import Path
from cases import EXPECTED
a = int(Path("kernel_src/main.py").read_text())
b = int(Path("kernel_src/nested/helper.py").read_text())
assert a + b == EXPECTED, "incorrect candidate"
print("CORRECTNESS: PASS")
print("GEAK_GEOMEAN_SPEEDUP", 1.2 if (a, b) == (2, 4) else 1.0)
''')
        self.run = self.base / "work/runs/task"
        self.run.mkdir(parents=True)
        self.env = dict(os.environ, GEAK_ROOT=str(ROOT), TASK_DIR=str(self.task),
                        WORK_ROOT=str(self.base / "work"), RUN_DIR=str(self.run),
                        EVAL_DIR=str(self.run / "attempt_1"), K="task", ATTEMPT="1",
                        ATTEMPT_CAP="2", GPU_IDS="0,1,2", BUDGET="5", MIN_IMPROVE="0.02",
                        KERNEL_SRC_ALL="main.py nested/helper.py", BEST_SPEEDUP="1.0",
                        KEEP_THRESHOLD="1.01", MEASURE_TIMEOUT="5",
                        PROMPT_OUT=str(self.base / "prompt.txt"))

    def sources(self, a, b):
        (self.task / "kernel_src/main.py").write_text(str(a))
        (self.task / "kernel_src/nested/helper.py").write_text(str(b))

    def call(self, script, *args, env=None, timeout=30):
        return subprocess.run(["bash", str(script), *args], check=False, env=env or self.env,
                              capture_output=True, text=True, timeout=timeout)

    def test_context_is_literal_and_does_not_require_workflow_runtime(self):
        marker = self.base / "must-not-exist"
        env = dict(self.env, GEAK_ROOT=str(self.base / "no-workflow-runtime"),
                   EXTRA_STEER=f"Literal $(touch {marker}) and `touch {marker}`")
        before = (self.task / "meta.json").read_bytes()
        result = self.call(METHOD / "prompt.sh", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        prompt = Path(env["PROMPT_OUT"]).read_text()
        for value in (str(METHOD / "SKILL.md"), str(self.task), env["EVAL_DIR"],
                      env["KERNEL_SRC_ALL"], env["GPU_IDS"], env["EXTRA_STEER"]):
            self.assertIn(value, prompt)
        self.assertFalse(marker.exists())
        self.assertEqual((self.task / "meta.json").read_bytes(), before)

    def test_missing_context_fails_before_writing_a_prompt(self):
        env = dict(self.env)
        del env["TASK_DIR"]
        result = self.call(METHOD / "prompt.sh", env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(Path(env["PROMPT_OUT"]).exists())

    def test_measurement_keeps_native_failure_distinct_from_hook_exit(self):
        receipt = self.base / "native-exit"
        env = dict(self.env, MEASURE_NATIVE_RC_FILE=str(receipt))
        for sources, status, native_exit in [((2, 4), "1", "0"), ((1, 4), "0", "1")]:
            with self.subTest(sources=sources):
                self.sources(*sources)
                result = self.call(METHOD / "measure.sh", env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                fields = dict(part.split("=", 1) for part in result.stdout.split())
                self.assertEqual(fields["STATUS"], status)
                self.assertEqual(receipt.read_text().strip(), native_exit)
                self.assertEqual(fields["SPEEDUP"], "1.2" if status == "1" else "")

    def test_shared_worker_keeps_valid_edits_and_rejects_frozen_edits(self):
        runner = self.base / "runner.sh"
        runner.write_text('''#!/usr/bin/env bash
set -eu
printf 2 > "$2/kernel_src/main.py"
printf 4 > "$2/kernel_src/nested/helper.py"
if [ "${BREAK_HARNESS:-0}" = 1 ]; then printf 'EXPECTED = 99\n' > "$2/cases.py"; fi
printf 'CPU fixture completed\n' > "$3"
''')
        for broken in (False, True):
            with self.subTest(frozen_edit=broken):
                self.sources(3, 3)
                work = self.base / ("invalid-work" if broken else "valid-work")
                env = dict(self.env, SKILL_ROOT=str(METHOD), RUNNER=str(runner),
                           # Allow startup/receipt IO without exhausting the smoke budget before dispatch.
                           WORK_ROOT=str(work), TARGET="3", ATTEMPT_MIN="1",
                           BREAK_HARNESS=str(int(broken)))
                result = self.call(ROOT / "workload/engine/campaign/worker.sh",
                                   "task", "cpu-fixture", env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                state = json.loads((work / "state/task.status.json").read_text())
                self.assertEqual(state["best_speedup"], 1.0 if broken else 1.2,
                                 (work / "logs/task/worker.log").read_text())
                self.assertEqual((self.task / "cases.py").read_text(), "EXPECTED = 6\n")
                self.assertEqual((self.task / "kernel_src/main.py").read_text(),
                                 "3" if broken else "2")
                self.assertTrue(list((work / "runs/task").glob(".prompt_*.txt")))


if __name__ == "__main__":
    unittest.main()
