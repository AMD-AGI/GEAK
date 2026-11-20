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


def cfg_with(selection, top_k=None, pct=None):
    params = {"selection": selection}
    if top_k is not None:
        params["selection_top_k"] = top_k
    if pct is not None:
        params["selection_overall_pct_gt"] = pct
    return {"stages": {"profile": {"params": params}}}


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
