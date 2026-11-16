"""CPU tests for meta.ut_contract.leaf_control_waivers: an editable leaf of a whole-package
candidate (a monokernel bound at its package, DESIGN 6.0 -- the Muse Spark R9 tree edits
muse_spark/gfx950/** and common/sync.py) may go without a negative control of its own only when
a waiver names it with a why.

Run: python3 workload/stages/ut/kernel-ut/ut_contracts/tests/test_leaf_control_waivers.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = os.path.dirname(HERE)
sys.path.insert(0, UTC)
import scaffold_ut  # noqa: E402
import ut_contract as U  # noqa: E402

PKG = "kernel_src/pkg"
STAGE = PKG + "/gfx/stage.py"
LAYOUT = PKG + "/gfx/layout.py"
SYNC = PKG + "/common/sync.py"


class LeafControlWaivers(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="leaf_waivers_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.task = os.path.join(self.root, "mono")
        with contextlib.redirect_stdout(io.StringIO()):
            scaffold_ut.main(["new", "--family", "mono_layer", "--task", self.task,
                              "--target", "pkg.runner:Runner.launch", "--bind-module", "pkg",
                              "--bind-file", PKG + "/__init__.py",
                              "--axis", "hidden=8192", "--axis", "world=8"])
        for rel, text in ((PKG + "/__init__.py", ""), (STAGE, "x = a * scale\n"),
                          (LAYOUT, "ROWS = 8\n"), (SYNC, "def spin():\n    pass\n")):
            path = os.path.join(self.task, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        # the installed package = this tree: no file differs, so only the waiver rules speak
        with contextlib.redirect_stdout(io.StringIO()):
            scaffold_ut.main(["baseline-package", "--task", self.task,
                              "--root", os.path.join(self.task, PKG)])
        self.control = {"leaf": STAGE, "find": "a * scale", "replace": "a * scale * 0.5",
                        "expect": "the stage output halves", "probe_sigs": []}

    def _set(self, leaves, controls, waivers=None):
        p = os.path.join(self.task, "meta.json")
        with open(p) as f:
            meta = json.load(f)
        mc = meta["ut_contract"]
        mc["editable_leaves"] = leaves
        mc["negative_controls"] = controls
        if waivers is None:
            mc.pop("leaf_control_waivers", None)
        else:
            mc["leaf_control_waivers"] = waivers
        with open(p, "w") as f:
            json.dump(meta, f)

    def errors(self):
        return [e for e in U.lint_task(self.task)["errors"]
                if "negative control" in e or "leaf_control_waivers" in e or "waived" in e
                or "editable leaf" in e]

    def test_a_leaf_without_control_or_waiver_is_an_error(self):
        self._set([STAGE, LAYOUT, SYNC], [self.control])
        errs = self.errors()
        self.assertIn("editable leaf %s has no negative control" % LAYOUT, errs)
        self.assertIn("editable leaf %s has no negative control" % SYNC, errs)
        self.assertEqual(len(errs), 2, errs)

    def test_waived_leaves_with_a_why_pass(self):
        self._set([STAGE, LAYOUT, SYNC], [self.control],
                  [{"leaf": LAYOUT, "why": "offsets only; any edit shows in the stage outputs"},
                   {"leaf": SYNC, "why": "device runtime of every stage's hand-off"}])
        self.assertEqual(self.errors(), [])

    def test_a_waiver_without_why_does_not_waive(self):
        self._set([STAGE, LAYOUT], [self.control], [{"leaf": LAYOUT, "why": "  "}])
        errs = self.errors()
        self.assertTrue(any("need leaf and why" in e for e in errs), errs)
        self.assertIn("editable leaf %s has no negative control" % LAYOUT, errs)

    def test_a_waiver_must_name_an_editable_leaf(self):
        self._set([STAGE], [self.control], [{"leaf": LAYOUT, "why": "not declared editable"}])
        self.assertTrue(any("not an editable leaf" in e for e in self.errors()))

    def test_a_controlled_leaf_cannot_also_be_waived(self):
        self._set([STAGE, LAYOUT], [self.control],
                  [{"leaf": STAGE, "why": "stale"}, {"leaf": LAYOUT, "why": "offsets only"}])
        self.assertEqual([e for e in self.errors() if "drop the waiver" in e],
                         ["leaf_control_waivers names %s, which has a negative control: drop the waiver"
                          % STAGE])

    def test_waiving_every_leaf_is_an_error(self):
        every = "every editable leaf is waived: at least one needs a negative control"
        self._set([LAYOUT, SYNC], [],
                  [{"leaf": LAYOUT, "why": "offsets"}, {"leaf": SYNC, "why": "runtime"}])
        self.assertEqual(self.errors(), [every])
        # the same task without the waivers: each leaf lacks a control, and nothing claims a waiver
        self._set([LAYOUT, SYNC], [])
        errs = self.errors()
        self.assertNotIn(every, errs)
        self.assertEqual(sorted(errs), sorted("editable leaf %s has no negative control" % l
                                              for l in (LAYOUT, SYNC)))

    def test_waivers_need_a_whole_package_bind(self):
        p = os.path.join(self.task, "meta.json")
        with open(p) as f:
            meta = json.load(f)
        meta["candidate_bind"] = {"kind": "module", "module": "pkg.gfx.stage", "file": STAGE}
        with open(p, "w") as f:
            json.dump(meta, f)
        self._set([STAGE, LAYOUT], [self.control], [{"leaf": LAYOUT, "why": "offsets only"}])
        errs = self.errors()
        self.assertTrue(any("only for a whole-package bind" in e for e in errs), errs)
        self.assertIn("editable leaf %s has no negative control" % LAYOUT, errs)

    def test_a_waiver_outside_the_bound_package_is_an_error(self):
        other = "kernel_src/other.py"
        with open(os.path.join(self.task, other), "w") as f:
            f.write("Y = 1\n")
        self._set([STAGE, other], [self.control], [{"leaf": other, "why": "not in the package"}])
        self.assertTrue(any("outside the bound package" in e for e in self.errors()))

    def test_waivers_must_be_a_list(self):
        self._set([STAGE], [self.control], {"leaf": LAYOUT, "why": "x"})
        self.assertTrue(any("must be a list" in e for e in self.errors()))


if __name__ == "__main__":
    unittest.main()
