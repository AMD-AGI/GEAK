#!/usr/bin/env python3
"""Tests for tuning_skillset_sync.py — the integrity gate on the VENDORED tuning skillset.

Why this matters enough to test: the skillset is developed and validated STANDALONE (its own
`validate/claims.py` and per-skill SKILL.md set). That validation only says anything about a GEAK run if
the tree GEAK ships is the tree that was validated. The manifest is what makes drift loud instead of
silent, so the manifest checker itself needs to be trustworthy.

No GPU, no network, no serving stack.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
"""

import importlib.util
import os
import shutil
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))          # e2e_workflow/scripts/tests
MODULE_PATH = os.path.join(os.path.dirname(HERE), "tuning_skillset_sync.py")


def _load():
    spec = importlib.util.spec_from_file_location("tuning_skillset_sync", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sync_mod = _load()


class TestVendoredSkillsetShipsIntact(unittest.TestCase):
    """The real vendored tree, as committed."""

    def test_skillset_is_vendored(self):
        self.assertTrue(
            os.path.isdir(sync_mod.DEFAULT_SKILLSET_DIR),
            "the tuning skillset must be vendored into the repo",
        )

    def test_entry_points_survive_vendoring(self):
        # The skillset is invoked through its OWN router + per-skill files. If vendoring flattened or
        # pruned them, the role's instructions would dangle.
        for rel in (
            "README.md",
            "tuning-core/SKILL.md",
            "tuning-core/measurement.md",
            "tuning-core/engagement_verification.md",
            "tuning-core/correctness_gates.md",
            "tuning-aiter/SKILL.md",
            "tuning-in-vllm/SKILL.md",
            "tuning-in-sglang/SKILL.md",
            "env-setup/audit_tools.sh",
            "validate/claims.py",
            "tuning-kb/README.md",
        ):
            with self.subTest(rel=rel):
                self.assertTrue(
                    os.path.isfile(os.path.join(sync_mod.DEFAULT_SKILLSET_DIR, rel)),
                    f"vendored skillset is missing its own entry point: {rel}",
                )

    def test_committed_tree_matches_committed_manifest(self):
        self.assertEqual(
            0,
            sync_mod.verify(sync_mod.DEFAULT_SKILLSET_DIR, sync_mod.DEFAULT_MANIFEST),
            "vendored skillset drifted from its manifest — re-sync it, do not patch it in place",
        )

    def test_no_build_artifacts_vendored(self):
        files = sync_mod.iter_files(sync_mod.DEFAULT_SKILLSET_DIR)
        self.assertFalse([f for f in files if "__pycache__" in f or f.endswith(".pyc")])
        self.assertTrue(files, "vendored tree is empty")


class TestManifestRoundTrip(unittest.TestCase):
    """Manifest behavior against a synthetic tree, so we can actually mutate it."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tree = os.path.join(self.tmp, "skillset")
        os.makedirs(os.path.join(self.tree, "tuning-core"))
        self._write("README.md", "# router\n")
        self._write("tuning-core/SKILL.md", "---\nname: tuning-core\n---\nthe loop\n")
        self.manifest = os.path.join(self.tmp, "manifest.sha256")
        sync_mod.write_manifest(self.manifest, sync_mod.build_manifest(self.tree), self.tree)

    def _write(self, rel, text):
        path = os.path.join(self.tree, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def test_clean_tree_verifies(self):
        self.assertEqual(0, sync_mod.verify(self.tree, self.manifest))

    def test_modified_file_is_caught(self):
        # The failure mode that matters: someone "improves" a skill in place, and the standalone
        # validation silently stops describing what GEAK runs.
        self._write("tuning-core/SKILL.md", "---\nname: tuning-core\n---\nthe loop, edited\n")
        self.assertEqual(1, sync_mod.verify(self.tree, self.manifest))

    def test_deleted_file_is_caught(self):
        os.remove(os.path.join(self.tree, "tuning-core", "SKILL.md"))
        self.assertEqual(1, sync_mod.verify(self.tree, self.manifest))

    def test_added_file_is_caught(self):
        # GEAK-specific additions belong in EVAL_DIR or e2e_workflow/, never inside the vendored tree.
        self._write("geak_notes.md", "local tweak\n")
        self.assertEqual(1, sync_mod.verify(self.tree, self.manifest))

    def test_pycache_is_ignored_not_flagged(self):
        self._write("__pycache__/x.cpython-311.pyc", "junk")
        self.assertEqual(0, sync_mod.verify(self.tree, self.manifest))

    def test_missing_tree_or_manifest_fails_loudly(self):
        self.assertEqual(1, sync_mod.verify(os.path.join(self.tmp, "nope"), self.manifest))
        self.assertEqual(1, sync_mod.verify(self.tree, os.path.join(self.tmp, "nope.sha256")))

    def test_update_reblesses_the_tree(self):
        self._write("tuning-core/SKILL.md", "changed\n")
        self.assertEqual(1, sync_mod.verify(self.tree, self.manifest))
        sync_mod.write_manifest(self.manifest, sync_mod.build_manifest(self.tree), self.tree)
        self.assertEqual(0, sync_mod.verify(self.tree, self.manifest))

    def test_manifest_is_stable_and_sorted(self):
        first = sync_mod.build_manifest(self.tree)
        self.assertEqual(first, sync_mod.build_manifest(self.tree))
        sync_mod.write_manifest(self.manifest, first, self.tree)
        self.assertEqual(first, sync_mod.read_manifest(self.manifest))
        rels = [r for r in sync_mod.iter_files(self.tree)]
        self.assertEqual(rels, sorted(rels))


class TestSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.src = os.path.join(self.tmp, "upstream")
        os.makedirs(os.path.join(self.src, "tuning-core"))
        for rel, text in (("README.md", "# upstream\n"), ("tuning-core/SKILL.md", "loop\n")):
            path = os.path.join(self.src, rel)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        self.dest = os.path.join(self.tmp, "vendored")

    def test_sync_copies_the_whole_tree(self):
        sync_mod.sync(self.src, self.dest)
        self.assertEqual(["README.md", "tuning-core/SKILL.md"], sync_mod.iter_files(self.dest))

    def test_sync_replaces_rather_than_merges(self):
        # A stale file left behind from a previous vendor would be invisible drift.
        os.makedirs(self.dest)
        with open(os.path.join(self.dest, "stale.md"), "w", encoding="utf-8") as fh:
            fh.write("old\n")
        sync_mod.sync(self.src, self.dest)
        self.assertNotIn("stale.md", sync_mod.iter_files(self.dest))

    def test_sync_rejects_a_non_skillset_source(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaises(SystemExit):
            sync_mod.sync(empty, self.dest)

    def test_synced_tree_verifies_against_its_fresh_manifest(self):
        manifest = os.path.join(self.tmp, "m.sha256")
        sync_mod.main(["--skillset-dir", self.dest, "--manifest", manifest, "--sync", self.src])
        self.assertEqual(0, sync_mod.main(["--skillset-dir", self.dest, "--manifest", manifest, "--verify"]))


if __name__ == "__main__":
    unittest.main()
