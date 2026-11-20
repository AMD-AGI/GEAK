"""CPU regressions for the config checks and the end-of-run report in engine/core/stage.py."""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CORE = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE))


def load(name):
    spec = importlib.util.spec_from_file_location("core_" + name, CORE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage = load("stage")
target = load("target")


class FinalReportTest(unittest.TestCase):
    def test_cmd_all_renders_the_report_once_more_after_the_last_stage(self):
        cfg = {"order": ["baseline"], "run_root": "/tmp/does-not-matter"}
        calls = []
        with patch.object(stage, "load_state", return_value={}), \
             patch.object(stage, "start_reporter", return_value=None), \
             patch.object(stage, "stop_reporter"), \
             patch.object(stage, "run_order", return_value=7), \
             patch.object(stage, "render_final_report", side_effect=calls.append):
            rc = stage.cmd_all(cfg, SimpleNamespace(stages=[]))
        self.assertEqual(rc, 7)
        self.assertEqual(calls, [cfg])

    def test_render_uses_the_run_root_and_never_changes_the_return_code(self):
        with tempfile.TemporaryDirectory() as root:
            stage.render_final_report({"run_root": root})
            self.assertTrue(Path(root, "REPORT.md").exists())
        with patch.dict(sys.modules, {"report": SimpleNamespace(
                main=lambda argv: (_ for _ in ()).throw(RuntimeError("boom")))}):
            stage.render_final_report({"run_root": "/nonexistent"})      # must not raise


class NamingTest(unittest.TestCase):
    cfg = {"target": {"model": "amd/Some-Model-FP4", "name_tokens": ["somemodel"]}}

    def test_leaf_or_parent_directory_may_carry_the_stack_name(self):
        self.assertEqual(target.naming_problems(self.cfg, "/data/somemodel_run1"), [])
        self.assertEqual(target.naming_problems(self.cfg, "/data/somemodel/run1"), [])

    def test_a_name_that_carries_no_token_anywhere_is_still_rejected(self):
        problems = target.naming_problems(self.cfg, "/data/other/run1")
        self.assertEqual(len(problems), 1)
        self.assertIn("run_root is named 'run1'", problems[0])


if __name__ == "__main__":
    unittest.main()
