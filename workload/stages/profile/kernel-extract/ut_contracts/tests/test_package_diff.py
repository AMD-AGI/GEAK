"""CPU tests for a whole-package bind's declared scope (run/baseline_package.json +
meta.ut_contract.frozen_differs): every file of kernel_src/<package>/ that differs from the
installed package must be an editable leaf or a frozen_differs entry with its why. The Muse Spark
R9 tree differed from the image's sgl_amd_mono in 102 places while its editable_leaves named 20
files (it had been diffed against v0, not against the installed baseline); lint passed.

Run: python3 workload/stages/profile/kernel-extract/ut_contracts/tests/test_package_diff.py
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = os.path.dirname(HERE)
sys.path.insert(0, UTC)
import scaffold_ut  # noqa: E402
import ut_contract as U  # noqa: E402

_spec = importlib.util.spec_from_file_location("package_diff_driver", os.path.join(UTC, "ut_driver.py"))
driver = importlib.util.module_from_spec(_spec)
_path = sys.path[:]
try:
    _spec.loader.exec_module(driver)
finally:
    sys.path[:] = _path

PKG = "kernel_src/pkg"
STAGE = PKG + "/model/gfx/stage.py"
GUARD = PKG + "/common/guard.py"
SYNC = PKG + "/common/sync.py"
OTHER = PKG + "/other_model/kernel.py"
TREE = {"__init__.py": "", "model/gfx/stage.py": "x = a * scale\n",
        "common/guard.py": "def deadline():\n    pass\n", "common/sync.py": "def spin():\n    pass\n",
        "other_model/kernel.py": "K = 1\n"}


def _write_tree(root, tree):
    for rel, text in tree.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


class PackageDiff(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="package_diff_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.installed = os.path.join(self.root, "site", "pkg")
        _write_tree(self.installed, TREE)
        self.task = os.path.join(self.root, "mono")
        with contextlib.redirect_stdout(io.StringIO()):
            scaffold_ut.main(["new", "--family", "mono_layer", "--task", self.task,
                              "--target", "pkg.model.runner:Runner.launch", "--bind-module", "pkg",
                              "--bind-file", PKG + "/__init__.py",
                              "--axis", "hidden=8192", "--axis", "world=8"])
        _write_tree(os.path.join(self.task, PKG), TREE)   # the scaffold's copy of the install
        self.control = {"leaf": STAGE, "find": "a * scale", "replace": "a * scale * 0.5",
                        "expect": "the stage output halves", "probe_sigs": []}
        self._set([STAGE])

    def record(self):
        with contextlib.redirect_stdout(io.StringIO()):
            scaffold_ut.main(["baseline-package", "--task", self.task, "--root", self.installed])

    def edit(self, rel, text):
        _write_tree(os.path.join(self.task, PKG), {rel: text})

    def _set(self, leaves, frozen=None, waivers=None):
        p = os.path.join(self.task, "meta.json")
        with open(p) as f:
            meta = json.load(f)
        mc = meta["ut_contract"]
        mc["editable_leaves"] = leaves
        mc["negative_controls"] = [self.control]
        mc["leaf_control_waivers"] = waivers or []
        if frozen is None:
            mc.pop("frozen_differs", None)
        else:
            mc["frozen_differs"] = frozen
        with open(p, "w") as f:
            json.dump(meta, f)

    def errors(self):
        return [e for e in U.lint_task(self.task)["errors"]
                if "installed package" in e or "frozen_differs" in e or "baseline_package" in e]

    def test_a_package_bind_without_the_record_is_an_error(self):
        errs = self.errors()
        self.assertEqual(len(errs), 1, errs)
        self.assertIn("scaffold_ut.py baseline-package", errs[0])

    def test_record_holds_the_installed_files(self):
        self.record()
        with open(os.path.join(self.task, U.BASELINE_PACKAGE)) as f:
            rec = json.load(f)
        self.assertEqual(rec["package"], "pkg")
        self.assertEqual(sorted(rec["files"]), sorted(TREE))
        self.assertEqual(self.errors(), [])

    def test_an_undeclared_runtime_change_is_an_error(self):
        # the R9 case: guard.py gained scalar_deadline that the declared sync.py leaf calls
        self.record()
        self.edit("model/gfx/stage.py", "x = a * scale  # faster\n")
        self.edit("common/guard.py", "def deadline():\n    pass\ndef scalar_deadline():\n    pass\n")
        self.edit("common/sync.py", "from .guard import scalar_deadline\n")
        self._set([STAGE, SYNC], waivers=[{"leaf": SYNC, "why": "device runtime"}])
        self.assertEqual(self.errors(), [
            "%s is changed relative to the installed package but is neither an editable leaf nor "
            "in frozen_differs: the candidate overlay carries it" % GUARD])
        self._set([STAGE, SYNC, GUARD], waivers=[{"leaf": SYNC, "why": "device runtime"},
                                                 {"leaf": GUARD, "why": "deadline runtime"}])
        self.assertEqual(self.errors(), [])

    def test_added_and_removed_files_count(self):
        self.record()
        self.edit("common/stamps.py", "S = 0\n")
        os.remove(os.path.join(self.task, OTHER))
        errs = self.errors()
        self.assertTrue(any(e.startswith(PKG + "/common/stamps.py is added") for e in errs), errs)
        self.assertTrue(any(e.startswith(OTHER + " is removed") for e in errs), errs)
        self.assertEqual(len(errs), 2, errs)
        self._set([STAGE, PKG + "/common/stamps.py"],
                  frozen=[{"path": OTHER, "why": "another model; nothing on this path imports it"}],
                  waivers=[{"leaf": PKG + "/common/stamps.py", "why": "trace stamps"}])
        self.assertEqual(self.errors(), [])

    def test_frozen_directory_covers_another_models_subtree(self):
        self.record()
        self.edit("other_model/kernel.py", "K = 2\n")
        self.edit("other_model/new.py", "N = 1\n")
        self.assertEqual(len(self.errors()), 2)
        self._set([STAGE], frozen=[{"path": PKG + "/other_model/",
                                    "why": "another model's subpackage; the task never imports it"}])
        self.assertEqual(self.errors(), [])

    def test_frozen_entries_are_checked(self):
        self.record()
        self.edit("other_model/kernel.py", "K = 2\n")
        self.edit("model/gfx/stage.py", "x = a * scale  # faster\n")
        self._set([STAGE], frozen=[
            {"path": OTHER, "why": " "},                              # no why: not frozen
            {"path": GUARD, "why": "stale: guard.py is unchanged"},
            {"path": STAGE, "why": "a leaf"},
            {"path": PKG + "/model/", "why": "holds a leaf"},
            {"path": "kernel_src/elsewhere.py", "why": "outside"}])
        errs = self.errors()
        self.assertTrue(any("need path and why" in e for e in errs), errs)
        self.assertTrue(any(GUARD + ", which does not differ" in e for e in errs), errs)
        self.assertTrue(any(STAGE + ", which is an editable leaf" in e for e in errs), errs)
        self.assertTrue(any("directory %s/model/ holds editable leaves" % PKG in e for e in errs), errs)
        self.assertTrue(any("not under the bound package" in e for e in errs), errs)
        self.assertTrue(any(e.startswith(OTHER + " is changed") for e in errs), errs)

    def test_frozen_differs_needs_a_package_bind(self):
        p = os.path.join(self.task, "meta.json")
        with open(p) as f:
            meta = json.load(f)
        meta["candidate_bind"] = {"kind": "module", "module": "pkg.model.gfx.stage", "file": STAGE}
        meta["ut_contract"]["frozen_differs"] = [{"path": OTHER, "why": "x"}]
        with open(p, "w") as f:
            json.dump(meta, f)
        self.assertEqual(self.errors(), ["meta.ut_contract.frozen_differs is only for a "
                                         "whole-package bind (candidate_bind of a package __init__.py)"])

    def test_bytecode_is_ignored(self):
        self.record()
        cache = os.path.join(self.task, PKG, "common", "__pycache__")
        os.makedirs(cache)
        with open(os.path.join(cache, "sync.cpython-310.pyc"), "wb") as f:
            f.write(b"\0")
        self.assertEqual(self.errors(), [])

    def test_binding_gate_rechecks_the_record_against_the_install(self):
        self.record()
        self.assertEqual(U.check_baseline_package(self.task, self.installed), [])
        _write_tree(self.installed, {"common/guard.py": "def deadline():\n    return 1\n"})
        errs = U.check_baseline_package(self.task, self.installed)
        self.assertEqual(len(errs), 1)
        self.assertIn("common/guard.py", errs[0])

    def test_binding_gate_finds_the_package_the_baseline_imports(self):
        with open(os.path.join(self.task, "meta.json")) as f:
            meta = json.load(f)
        T = SimpleNamespace(meta=meta)
        base = os.path.join(self.task, "baseline_overlay")
        with patch.dict(os.environ, {"PYTHONPATH": os.path.dirname(self.installed)}):
            root = driver.installed_package_root(T, base)
        self.assertEqual(os.path.realpath(root), os.path.realpath(self.installed))


if __name__ == "__main__":
    unittest.main()
