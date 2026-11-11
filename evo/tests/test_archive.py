"""Offline integration tests using only temporary files and fixture Git commits."""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HELPER = Path(__file__).resolve().parents[1] / "archive.py"
SPEC = importlib.util.spec_from_file_location("evo_archive", HELPER)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="evo-archive-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.canonical = self.root / "canonical"
        self.canonical.mkdir()
        self.run_git(self.canonical, "init", "--quiet", "--template=")
        self.source_paths = ["src/kernel.py", "new.py"]
        (self.canonical / "src").mkdir()
        (self.canonical / "src/kernel.py").write_bytes(b"value = 1\n")
        (self.canonical / "outside.txt").write_bytes(b"unchanged\n")
        self.commit(self.canonical)
        self.storage = self.root / "archive"
        self.workspace = self.clone("workspace")
        self.patch = self.root / "best.diff"

    def run_git(self, workspace, *arguments, input_data=None):
        return subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
             "-C", str(workspace), *arguments], input=input_data, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout

    def commit(self, workspace):
        self.run_git(workspace, "add", "--all")
        self.run_git(workspace, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "commit", "--quiet", "-m", "fixture")

    def clone(self, name):
        target = self.root / name
        self.run_git(self.root, "clone", "--quiet", "--no-hardlinks", str(self.canonical), str(target))
        return target

    def initialize(self):
        return archive.initialize(str(self.storage), str(self.canonical), self.source_paths)

    def prepare(self, parent="INCUMBENT", donors=None, workspace=None):
        return archive.prepare(str(self.storage), str(workspace or self.workspace), parent, donors)

    def report(self, report_id="round1", **updates):
        result = {"id": report_id, "parent_id": "INCUMBENT", "donor_ids": [], "title": "Fixture",
                  "workspace": str(self.workspace), "patch": str(self.patch),
                  "engineer": {"status": "ok", "speedup": 1.25},
                  "verifier": {"status": "verified", "case_weights": [0.1, 0.9]}}
        result.update(updates)
        return result

    def record(self, reports=None, canonical=None):
        return archive.record_batch(str(self.storage), str(canonical or self.canonical),
                                    json.dumps(reports if reports is not None else [self.report()]))

    def entry(self, entry_id):
        return archive.query(str(self.storage), entry_id)["entry"]

    def source(self, entry_id):
        entry = self.entry(entry_id)
        store = archive.Store(self.storage)
        manifest, blobs = store.source(entry["source_id"], sorted(self.source_paths))
        return {name: (blobs[metadata["sha256"]], metadata["mode"]) if metadata["present"] else None
                for name, metadata in manifest["files"].items()}

    def save_patch(self, data=b"value = 2\n", workspace=None):
        workspace = workspace or self.workspace
        (workspace / "src/kernel.py").write_bytes(data)
        self.patch.write_bytes(self.run_git(workspace, "diff", "--binary", "--no-ext-diff", "--no-textconv"))

    def assert_error(self, code, function, *arguments):
        with self.assertRaises(archive.ArchiveError) as caught:
            function(*arguments)
        self.assertEqual(code, caught.exception.code)

    def cli(self, *arguments):
        completed = subprocess.run([sys.executable, "-B", str(HELPER), *arguments],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(b"", completed.stderr)
        return completed.returncode, json.loads(completed.stdout)

    def test_initialization_founder_and_original_are_distinct(self):
        (self.canonical / "src/kernel.py").write_bytes(b"warmstart = True\r\n")
        os.chmod(self.canonical / "src/kernel.py", 0o640)
        initialized = self.initialize()
        self.assertFalse(initialized["reused"])
        self.assertEqual((b"warmstart = True\r\n", 0o640), self.source("FOUNDER")["src/kernel.py"])
        self.assertIsNone(self.source("FOUNDER")["new.py"])
        index = archive.query(self.storage)["index"]
        self.assertEqual("external_not_captured", index["original_benchmark"]["status"])
        self.assertIsNone(index["original_benchmark"]["source_id"])
        self.assertNotIn("ORIGINAL", index["entries"])
        self.assertTrue(self.initialize()["reused"])
        (self.canonical / "src/kernel.py").write_bytes(b"another warmstart\n")
        self.assert_error("init_conflict", self.initialize)

    def test_prepare_founder_preserves_canonical_index_and_unselected_files(self):
        self.initialize()
        canonical_before = self.run_git(self.canonical, "status", "--porcelain")
        workspace_index = (self.workspace / ".git/index").read_bytes()
        (self.workspace / "src/kernel.py").write_bytes(b"unfinished\n")
        (self.workspace / "new.py").write_bytes(b"should disappear\n")
        (self.workspace / "outside.txt").write_bytes(b"local outside\n")
        prepared = self.prepare("FOUNDER")
        self.assertEqual(b"value = 1\n", (self.workspace / "src/kernel.py").read_bytes())
        self.assertFalse((self.workspace / "new.py").exists())
        self.assertEqual(b"local outside\n", (self.workspace / "outside.txt").read_bytes())
        self.assertEqual(workspace_index, (self.workspace / ".git/index").read_bytes())
        self.assertEqual(canonical_before, self.run_git(self.canonical, "status", "--porcelain"))
        self.assertEqual(prepared, self.prepare("FOUNDER"))
        self.assert_error("canonical_workspace", archive.prepare, self.storage, self.canonical, "FOUNDER")

    def test_incumbent_leaves_ordinary_workspace_bytes_modes_and_mtime_alone(self):
        self.initialize()
        source = self.workspace / "src/kernel.py"
        source.write_bytes(b"main prepared this\n")
        source.chmod(0o751)
        before = source.stat()
        index_before = (self.workspace / ".git/index").read_bytes()
        prepared = self.prepare()
        self.assertEqual(b"main prepared this\n", source.read_bytes())
        self.assertEqual((before.st_mode, before.st_mtime_ns, before.st_ino),
                         (source.stat().st_mode, source.stat().st_mtime_ns, source.stat().st_ino))
        self.assertEqual(index_before, (self.workspace / ".git/index").read_bytes())
        self.assertEqual("INCUMBENT", prepared["receipt"]["requested_parent_id"])
        self.assertNotEqual(self.entry("FOUNDER")["source_id"], prepared["receipt"]["prepared_source_id"])

    def test_best_patch_and_development_have_separate_evidence_bindings(self):
        self.initialize()
        prepared = self.prepare()
        self.save_patch()
        (self.workspace / "src/kernel.py").write_bytes(b"unfinished = 999\n")
        index_before = (self.canonical / ".git/index").read_bytes()
        result = self.record()
        self.assertFalse(result["has_failures"])
        best = self.entry("round1-best")
        development = self.entry("round1-development")
        self.assertEqual(b"value = 2\n", self.source("round1-best")["src/kernel.py"][0])
        self.assertEqual(b"unfinished = 999\n", self.source("round1-development")["src/kernel.py"][0])
        self.assertEqual("author_reported_only", best["verification"]["status"])
        self.assertEqual("unverified_development", development["verification"]["status"])
        self.assertIsNone(development["verification"]["bound_source_id"])
        self.assertEqual(best["source_id"], best["verification"]["bound_source_id"])
        self.assertEqual(prepared["receipt_id"], best["parent_binding"]["receipt_id"])
        self.assertEqual(archive.digest(self.patch.read_bytes()), best["patch_sha256"])
        self.assertEqual(b"value = 1\n", (self.canonical / "src/kernel.py").read_bytes())
        self.assertEqual(index_before, (self.canonical / ".git/index").read_bytes())

    def test_same_bytes_deduplicate_but_keep_both_references(self):
        self.initialize()
        self.save_patch()
        self.record()
        best = self.entry("round1-best")
        development = self.entry("round1-development")
        self.assertEqual(best["source_id"], development["source_id"])
        self.assertNotEqual(best["verification"], development["verification"])
        self.assertEqual("unknown_no_prepare_receipt", best["parent_binding"]["status"])

    def test_sequential_rounds_older_parent_patch_uses_current_canonical(self):
        self.initialize()
        self.prepare()
        self.save_patch(b"value = 20\n")
        self.record()
        (self.canonical / "src/kernel.py").write_bytes(b"value = 20\n")
        self.commit(self.canonical)
        second = self.clone("round2-workspace")
        second_index = (second / ".git/index").read_bytes()
        self.prepare("FOUNDER", ["round1-best"], second)
        self.assertEqual(b"value = 1\n", (second / "src/kernel.py").read_bytes())
        self.assertEqual(second_index, (second / ".git/index").read_bytes())
        self.save_patch(b"value = 3\n", second)
        result = self.record([self.report("round2", workspace=str(second), parent_id="FOUNDER", donor_ids=["round1-best"])])
        self.assertFalse(result["has_failures"])
        self.assertEqual(b"value = 3\n", self.source("round2-best")["src/kernel.py"][0])
        self.assertEqual(self.entry("round1-best")["source_id"], self.entry("round2-best")["round_canonical_source_id"])
        (self.canonical / "src/kernel.py").write_bytes(b"value = 3\n")
        self.commit(self.canonical)
        third = self.clone("round3-workspace")
        self.prepare("round1-best", workspace=third)
        self.save_patch(b"value = 4\n", third)
        result = self.record([self.report("round3", workspace=str(third), parent_id="round1-best")])
        self.assertFalse(result["has_failures"])
        self.assertEqual(b"value = 4\n", self.source("round3-best")["src/kernel.py"][0])
        self.assertEqual(b"value = 3\n", (self.canonical / "src/kernel.py").read_bytes())

    def test_patch_base_uses_canonical_bytes_even_when_head_differs(self):
        self.initialize()
        (self.canonical / "src/kernel.py").write_bytes(b"round = 2\n")
        staging = self.clone("patch-base")
        (staging / "src/kernel.py").write_bytes(b"round = 2\n")
        self.commit(staging)
        self.save_patch(b"round = 3\n", staging)
        result = self.record()
        self.assertFalse(result["has_failures"])
        self.assertEqual(b"round = 3\n", self.source("round1-best")["src/kernel.py"][0])
        self.assertEqual(b"round = 2\n", (self.canonical / "src/kernel.py").read_bytes())

    def test_missing_patch_failed_engineer_retains_unverified_development(self):
        self.initialize()
        (self.workspace / "src/kernel.py").write_bytes(b"useful unfinished\n")
        result = self.record([self.report(patch=None, engineer={"status": "failed", "speedup": 0.8}, verifier=None)])
        self.assertEqual("partial", result["results"][0]["status"])
        self.assertEqual("missing_patch", self.entry("round1-best")["diagnostic"]["code"])
        self.assertEqual("retained", self.entry("round1-development")["source_retention"])
        self.assertEqual("unverified_development", self.entry("round1-development")["verification"]["status"])
        self.assertEqual(0.8, self.entry("round1-development")["author_reported_evidence"]["engineer"]["speedup"])

    def test_optional_candidate_evidence_and_absent_patch_path(self):
        self.initialize()
        result = self.record([{"id": "minimal", "workspace": str(self.workspace)}])
        self.assertEqual("partial", result["results"][0]["status"])
        evidence = self.entry("minimal-development")["author_reported_evidence"]
        self.assertIsNone(evidence["engineer"])
        self.assertIsNone(evidence["verifier"])
        result = self.record([self.report("absent")])
        self.assertEqual("missing_file", self.entry("absent-best")["diagnostic"]["code"])
        self.assertEqual("partial", result["results"][0]["status"])

    def test_malformed_patch_does_not_abort_other_reports(self):
        self.initialize()
        self.patch.write_bytes(b"not a patch\n")
        reports = [self.report("bad"), self.report("missing", patch=None), {"id": "broken", "workspace": None}]
        result = self.record(reports)
        self.assertEqual(["partial", "partial", "failed"], [item["status"] for item in result["results"]])
        self.assertIn("diagnostic", self.entry("bad-best"))
        stored = archive.query(self.storage)["index"]["batches"][result["batch_id"]]
        self.assertEqual(result["results"], stored["results"])

    def test_exact_retry_and_changed_patch_or_metadata_conflict(self):
        self.initialize()
        self.save_patch()
        first = self.record()
        index_before = (self.storage / "index.json").read_bytes()
        self.assertEqual(first, self.record())
        self.assertEqual(index_before, (self.storage / "index.json").read_bytes())
        old_entry = self.entry("round1-best")
        self.save_patch(b"value = 7\n")
        conflict = self.record()
        self.assertEqual("conflict", conflict["results"][0]["status"])
        self.assertEqual("id_conflict", conflict["results"][0]["diagnostics"][0]["code"])
        self.assertEqual(old_entry, self.entry("round1-best"))
        self.save_patch()
        conflict = self.record([self.report(title="changed title")])
        self.assertEqual("conflict", conflict["results"][0]["status"])

    def test_duplicate_and_unsafe_ids_are_retained_as_failures(self):
        self.initialize()
        result = self.record([self.report(), self.report(), self.report("../escape"), self.report("FOUNDER")])
        self.assertEqual(["conflict", "conflict", "failed", "failed"], [item["status"] for item in result["results"]])
        self.assertEqual({"FOUNDER"}, set(archive.query(self.storage)["index"]["entries"]))
        self.assertEqual(result["results"], archive.query(self.storage)["index"]["batches"][result["batch_id"]]["results"])

    def test_metadata_case_weights_provenance_and_raw_json_unchanged(self):
        self.initialize()
        self.save_patch()
        forbidden = self.root / "do-not-read"
        os.mkfifo(forbidden)
        engineer = {"status": "slower", "geomean": None, "case_weights": {"case-z": 2, "case-a": 0.125},
                    "patch_file": str(forbidden), "measurements": [{"name": "baseline", "duration": None}]}
        verifier = {"status": "ok", "case_weights": [0.01, 0.99], "provenance": {"ORIGINAL": "original-only",
                    "trace": str(forbidden)}, "raw": [False, None, "N/A"]}
        report = self.report(engineer=engineer, verifier=verifier)
        raw = json.dumps([report], indent=2)
        result = archive.record_batch(self.storage, str(self.canonical), raw)
        for entry_id in ("round1-best", "round1-development"):
            evidence = self.entry(entry_id)["author_reported_evidence"]
            self.assertEqual(engineer, evidence["engineer"])
            self.assertEqual(verifier, evidence["verifier"])
        stored = archive.query(self.storage)["index"]["batches"][result["batch_id"]]
        self.assertEqual(raw.encode(), archive.Store(self.storage).get(stored["raw_report_json_sha256"]))

    def test_traversal_and_overlapping_source_scope_rejected(self):
        for relative in ("../outside", "/absolute", "src/../outside", ".git/config", "src//kernel.py", "src/./kernel.py", "src\\kernel.py"):
            with self.subTest(relative=relative):
                self.assert_error("unsafe_path", archive.initialize, self.storage, str(self.canonical), [relative])
        self.assert_error("unsafe_path", archive.initialize, self.storage, str(self.canonical), ["src", "src/kernel.py"])
        self.assert_error("unsafe_path", archive.initialize, self.canonical / "archive", str(self.canonical), self.source_paths)

    def test_symlink_source_components_and_patch_are_rejected(self):
        target = self.root / "target.py"
        target.write_bytes(b"must not change")
        (self.canonical / "new.py").symlink_to(target)
        self.assert_error("unsafe_path", self.initialize)
        (self.canonical / "new.py").unlink()
        self.initialize()
        (self.workspace / "src/kernel.py").unlink()
        (self.workspace / "src/kernel.py").symlink_to(target)
        self.assert_error("unsafe_path", self.prepare, "FOUNDER")
        (self.workspace / "src/kernel.py").unlink()
        (self.workspace / "src").rmdir()
        (self.workspace / "src").symlink_to(self.canonical / "src", target_is_directory=True)
        self.assert_error("unsafe_path", self.prepare, "FOUNDER")
        self.patch.symlink_to(target)
        result = self.record()
        self.assertTrue(result["has_failures"])
        self.assertEqual("unsafe_path", self.entry("round1-best")["diagnostic"]["code"])
        self.assertEqual(b"must not change", target.read_bytes())

    def test_symlink_archive_or_workspace_root_is_rejected(self):
        self.initialize()
        alias = self.root / "alias"
        alias.symlink_to(self.storage, target_is_directory=True)
        self.assert_error("unsafe_path", archive.query, alias)
        workspace_alias = self.root / "workspace-alias"
        workspace_alias.symlink_to(self.workspace, target_is_directory=True)
        self.assert_error("unsafe_path", archive.prepare, self.storage, str(workspace_alias))

    def test_corrupted_source_bytes_fail_before_prepare_mutation(self):
        self.initialize()
        manifest = json.loads(archive.Store(self.storage).get(self.entry("FOUNDER")["source_id"]))
        blob = self.storage / "objects" / manifest["files"]["src/kernel.py"]["sha256"]
        blob.chmod(0o644)
        blob.write_bytes(b"corrupted")
        before = (self.workspace / "src/kernel.py").read_bytes()
        self.assert_error("corrupt_archive", self.prepare, "FOUNDER")
        self.assertEqual(before, (self.workspace / "src/kernel.py").read_bytes())
        self.assert_error("corrupt_archive", archive.query, self.storage)

    def test_mode_change_and_binary_exact_bytes(self):
        self.initialize()
        (self.workspace / "src/kernel.py").chmod(0o755)
        self.save_patch(b"\x00\xff\r\n\x80binary\x00")
        result = self.record()
        self.assertFalse(result["has_failures"])
        self.assertEqual((b"\x00\xff\r\n\x80binary\x00", 0o755), self.source("round1-best")["src/kernel.py"])
        self.prepare("round1-best")
        self.assertEqual(0o755, stat.S_IMODE((self.workspace / "src/kernel.py").stat().st_mode))

    def test_missing_deletion_and_declared_new_file(self):
        self.initialize()
        (self.workspace / "src/kernel.py").unlink()
        (self.workspace / "new.py").write_bytes(b"new source\n")
        self.run_git(self.workspace, "add", "--intent-to-add", "new.py")
        self.patch.write_bytes(self.run_git(self.workspace, "diff", "--binary"))
        result = self.record()
        self.assertFalse(result["has_failures"])
        retained = self.source("round1-best")
        self.assertIsNone(retained["src/kernel.py"])
        self.assertEqual(b"new source\n", retained["new.py"][0])
        self.prepare("FOUNDER")
        self.assertTrue((self.workspace / "src/kernel.py").exists())
        self.assertFalse((self.workspace / "new.py").exists())
        self.prepare("round1-best")
        self.assertFalse((self.workspace / "src/kernel.py").exists())
        self.assertTrue((self.workspace / "new.py").exists())

    def test_out_of_scope_patch_regular_modes_and_traversal(self):
        self.initialize()
        (self.workspace / "outside.txt").write_bytes(b"changed\n")
        self.patch.write_bytes(self.run_git(self.workspace, "diff", "--binary"))
        self.record([self.report("outside")])
        self.assertEqual("out_of_scope_patch", self.entry("outside-best")["diagnostic"]["code"])
        self.patch.write_bytes(b"diff --git a/new.py b/new.py\nnew file mode 120000\nindex 0000000..0000001\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+/tmp/victim\n")
        self.record([self.report("link")])
        self.assertEqual("nonregular_patch", self.entry("link-best")["diagnostic"]["code"])
        self.patch.write_bytes(b"diff --git a/../outside b/../outside\nnew file mode 100644\n--- /dev/null\n+++ b/../outside\n@@ -0,0 +1 @@\n+attack\n")
        self.record([self.report("traversal")])
        self.assertEqual("failed", self.entry("traversal-best")["source_retention"])
        self.assertFalse((self.root / "outside").exists())

    def test_unknown_donor_parent_and_stale_receipt(self):
        self.initialize()
        before = (self.workspace / "src/kernel.py").read_bytes()
        self.assert_error("unknown_source", self.prepare, "FOUNDER", ["missing"])
        self.assert_error("unknown_source", self.prepare, "missing")
        self.assertEqual(before, (self.workspace / "src/kernel.py").read_bytes())
        self.save_patch()
        result = self.record([self.report(parent_id="FOUNDER")])
        self.assertEqual("missing_prepare_receipt", result["results"][0]["diagnostics"][0]["code"])
        self.prepare("FOUNDER")
        self.save_patch()
        result = self.record([self.report("wrong", donor_ids=["missing"])])
        self.assertEqual("unknown_source", result["results"][0]["diagnostics"][0]["code"])
        self.run_git(self.workspace, "add", "src/kernel.py")
        result = self.record([self.report("stale", parent_id="FOUNDER")])
        self.assertEqual("stale_prepare_receipt", result["results"][0]["diagnostics"][0]["code"])

    def test_submodule_sources_rejected_without_following_repository(self):
        head = self.run_git(self.canonical, "rev-parse", "HEAD").decode().strip()
        self.run_git(self.canonical, "update-index", "--add", "--cacheinfo", "160000", head, "nested")
        self.assert_error("submodule_source", archive.initialize, self.storage, str(self.canonical), ["nested/source.py"])
        self.initialize()

    def test_caps_fifo_and_lock_are_bounded(self):
        self.initialize()
        with open(self.storage / "lock", "rb") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assert_error("archive_busy", archive.query, self.storage)
        os.mkfifo(self.workspace / "new.py")
        self.assert_error("nonregular_source", self.prepare)
        (self.workspace / "new.py").unlink()
        with mock.patch.object(archive, "MAX_SOURCE_BYTES", 4):
            self.assert_error("size_limit", self.prepare)
        self.assert_error("size_limit", archive.record_batch, self.storage, str(self.canonical), " " * (archive.MAX_REPORT_BYTES + 1))

    def test_no_task_hooks_filters_or_metadata_paths_execute(self):
        self.initialize()
        self.save_patch()
        marker = self.root / "executed"
        script = self.root / "danger.sh"
        script.write_text("#!/bin/sh\ntouch " + str(marker) + "\n")
        script.chmod(0o755)
        for workspace in (self.workspace, self.canonical):
            self.run_git(workspace, "config", "core.fsmonitor", str(script))
            self.run_git(workspace, "config", "core.hooksPath", str(self.root))
            self.run_git(workspace, "config", "filter.task.clean", str(script))
            (workspace / ".gitattributes").write_text("*.py filter=task\n")
        self.prepare()
        report = self.report(engineer={"script": str(script), "patch_file": str(script)})
        result = self.record([report])
        self.assertFalse(result["has_failures"])
        self.assertFalse(marker.exists())

    def test_cli_json_contract_and_full_query(self):
        code, initialized = self.cli("init", "--archive", str(self.storage), "--canonical", str(self.canonical),
                                     "--source-path", "src/kernel.py", "--source-path", "new.py")
        self.assertEqual(0, code)
        self.assertTrue(initialized["ok"])
        code, prepared = self.cli("prepare", "--archive", str(self.storage), "--workspace", str(self.workspace))
        self.assertEqual(0, code)
        self.assertEqual("INCUMBENT", prepared["receipt"]["requested_parent_id"])
        code, result = self.cli("record-batch", "--archive", str(self.storage), "--canonical", str(self.canonical),
                               "--report-json", json.dumps([self.report(patch=None)]))
        self.assertEqual(0, code)
        self.assertTrue(result["has_failures"])
        code, queried = self.cli("query", "--archive", str(self.storage))
        self.assertEqual(0, code)
        self.assertEqual({"FOUNDER", "round1-best", "round1-development"}, set(queried["index"]["entries"]))
        code, queried = self.cli("query", "--archive", str(self.storage), "--entry", "round1-development")
        self.assertEqual("unverified_development", queried["entry"]["verification"]["status"])
        for arguments in (("query",), ("bogus",), ("query", "--archive", str(self.storage), "--entry", "../bad")):
            code, failure = self.cli(*arguments)
            self.assertEqual(2, code)
            self.assertFalse(failure["ok"])

    def test_invalid_json_is_explicit_without_mutating_index(self):
        self.initialize()
        before = (self.storage / "index.json").read_bytes()
        for payload in ("{", '[{"id":"first","id":"second"}]', '[{"number":NaN}]', '[{"number":1e999}]'):
            self.assert_error("invalid_json", archive.record_batch, self.storage, str(self.canonical), payload)
        self.assertEqual(before, (self.storage / "index.json").read_bytes())

    def test_check_patch_receipt_matches_record_and_does_not_mutate(self):
        self.initialize()
        self.save_patch()
        archive_before = (self.storage / "index.json").read_bytes()
        canonical_index = (self.canonical / ".git/index").read_bytes()
        canonical_source = (self.canonical / "src/kernel.py").read_bytes()
        code, checked = self.cli("check-patch", "--archive", str(self.storage), "--canonical", str(self.canonical),
                                 "--patch", str(self.patch))
        self.assertEqual(0, code)
        self.assertEqual("check-patch", checked["command"])
        self.assertEqual(str(self.storage), checked["archive"])
        self.assertTrue(checked["source_scope_valid"])
        self.assertEqual(checked["receipt"]["patch_sha256"], checked["patch_sha256"])
        self.assertEqual(checked["receipt"]["round_canonical_source_id"], checked["canonical_source_id"])
        self.assertEqual(checked["receipt"]["source_id"], checked["source_id"])
        self.assertEqual("retained", checked["source_retention"])
        self.assertFalse(checked["entry_recorded"])
        self.assertEqual(archive_before, (self.storage / "index.json").read_bytes())
        self.assertEqual(canonical_index, (self.canonical / ".git/index").read_bytes())
        self.assertEqual(canonical_source, (self.canonical / "src/kernel.py").read_bytes())
        frozen = Path(checked["patch_object_path"])
        self.assertEqual(self.patch.read_bytes(), frozen.read_bytes())
        self.assertEqual(0o444, stat.S_IMODE(frozen.stat().st_mode))
        self.patch.write_bytes(b"changed after check\n")
        self.record([self.report(patch=str(frozen))])
        best = self.entry("round1-best")
        for field in ("patch_sha256", "round_canonical_source_id", "source_id"):
            self.assertEqual(best[field], checked["receipt"][field])
        self.assertEqual(b"value = 2\n", self.source("round1-best")["src/kernel.py"][0])

    def test_query_donor_survives_workspace_and_patch_cleanup(self):
        self.initialize()
        self.save_patch()
        self.record()
        self.patch.unlink()
        (self.workspace / "src/kernel.py").unlink()
        result = archive.query(self.storage, "round1-best")
        metadata = result["source_manifest"]["files"]["src/kernel.py"]
        blob = Path(result["blob_directory"]) / metadata["sha256"]
        self.assertEqual(b"value = 2\n", blob.read_bytes())
        self.assertEqual(metadata["sha256"], archive.digest(blob.read_bytes()))
        self.prepare("round1-best")
        self.assertEqual(b"value = 2\n", (self.workspace / "src/kernel.py").read_bytes())

    def test_check_patch_rejects_out_of_scope_and_missing_patch(self):
        self.initialize()
        (self.workspace / "outside.txt").write_bytes(b"change outside scope\n")
        self.patch.write_bytes(self.run_git(self.workspace, "diff"))
        code, rejected = self.cli("check-patch", "--archive", str(self.storage), "--canonical", str(self.canonical),
                                  "--patch", str(self.patch))
        self.assertEqual(2, code)
        self.assertEqual("out_of_scope_patch", rejected["error"]["code"])
        self.patch.unlink()
        self.assert_error("missing_file", archive.check_patch, self.storage, str(self.canonical), str(self.patch))

    def test_prepare_allows_declared_new_intent_to_add_but_not_staging(self):
        self.initialize()
        self.prepare()
        (self.workspace / "new.py").write_bytes(b"new source\n")
        self.run_git(self.workspace, "add", "--intent-to-add", "new.py")
        self.save_patch()
        self.run_git(self.workspace, "status", "--porcelain")
        result = self.record()
        self.assertFalse(result["has_failures"], result)
        self.assertEqual(b"new source\n", self.source("round1-best")["new.py"][0])
        self.run_git(self.workspace, "add", "new.py")
        result = self.record([self.report("staged")])
        self.assertEqual("stale_prepare_receipt", result["results"][0]["diagnostics"][0]["code"])

    def test_undeclared_intent_to_add_and_empty_staged_file_rejected(self):
        self.initialize()
        self.prepare()
        (self.workspace / "new.py").write_bytes(b"")
        self.run_git(self.workspace, "add", "new.py")
        result = self.record()
        self.assertEqual("stale_prepare_receipt", result["results"][0]["diagnostics"][0]["code"])
        separate = self.clone("separate")
        self.prepare(workspace=separate)
        (separate / "unknown.py").write_bytes(b"unknown\n")
        self.run_git(separate, "add", "--intent-to-add", "unknown.py")
        result = self.record([self.report("unknown", workspace=str(separate))])
        self.assertEqual("stale_prepare_receipt", result["results"][0]["diagnostics"][0]["code"])

    def test_stat_refresh_allowed_but_head_change_rejected(self):
        self.initialize()
        self.prepare()
        source = self.workspace / "src/kernel.py"
        source.write_bytes(source.read_bytes())
        self.run_git(self.workspace, "status", "--porcelain")
        self.run_git(self.workspace, "diff")
        self.save_patch()
        result = self.record()
        self.assertFalse(result["has_failures"], result)
        self.run_git(self.workspace, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "commit", "--allow-empty", "--quiet", "-m", "different fixture HEAD")
        result = self.record([self.report("changedhead")])
        self.assertEqual("stale_prepare_receipt", result["results"][0]["diagnostics"][0]["code"])

    def test_actual_main_materializer_and_initialize_index_then_older_parent(self):
        self.initialize()
        materialized = self.root / "materialized"
        script = HELPER.parents[1] / "kernel_workflow/scripts/materialize_workspace.sh"
        result = subprocess.run(["bash", str(script), "--src", str(self.canonical), "--dst", str(materialized)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertFalse((materialized / ".git").exists())
        (materialized / "src/kernel.py").write_bytes(b"value = 10\n")
        prepared = archive.prepare(self.storage, str(materialized), "FOUNDER", initialize_index=True)
        self.assertEqual("created_declared_source_index", prepared["receipt"]["index_initialization"]["status"])
        self.assertIsNone(prepared["receipt"]["workspace_index_baseline"]["head"])
        tracked = self.run_git(materialized, "ls-files").decode().splitlines()
        self.assertEqual(["src/kernel.py"], tracked)
        self.assertEqual(b"value = 1\n", (materialized / "src/kernel.py").read_bytes())
        self.assertEqual(b"value = 10\n", self.run_git(materialized, "show", ":src/kernel.py"))
        self.assertEqual(prepared, archive.prepare(self.storage, str(materialized), "FOUNDER", initialize_index=True))
        (self.canonical / "src/kernel.py").write_bytes(b"value = 10\n")
        (materialized / "new.py").write_bytes(b"new\n")
        self.run_git(materialized, "add", "--intent-to-add", "new.py")
        self.save_patch(b"value = 5\n", materialized)
        self.run_git(materialized, "status", "--porcelain")
        result = self.record([self.report(workspace=str(materialized), parent_id="FOUNDER")])
        self.assertFalse(result["has_failures"], result)
        self.assertEqual(b"value = 5\n", self.source("round1-best")["src/kernel.py"][0])
        self.assertEqual(b"new\n", self.source("round1-best")["new.py"][0])

    def test_cli_initialize_index_flag_preserves_existing_index(self):
        self.initialize()
        before = (self.workspace / ".git/index").read_bytes()
        code, result = self.cli("prepare", "--archive", str(self.storage), "--workspace", str(self.workspace),
                                "--parent", "FOUNDER", "--initialize-index")
        self.assertEqual(0, code)
        self.assertEqual("existing_index_preserved", result["receipt"]["index_initialization"]["status"])
        self.assertEqual(before, (self.workspace / ".git/index").read_bytes())

    def test_corrupted_index_metadata_still_returns_json_error(self):
        self.initialize()
        index = json.loads((self.storage / "index.json").read_bytes())
        index["entries"]["FOUNDER"] = []
        (self.storage / "index.json").write_text(json.dumps(index))
        code, result = self.cli("query", "--archive", str(self.storage))
        self.assertEqual(2, code)
        self.assertEqual("corrupt_archive", result["error"]["code"])


    def test_declared_canonical_guard_rejects_materializer_exclusions_before_index(self):
        (self.canonical / "build").mkdir()
        (self.canonical / "build/kernel.py").write_text("value = 7\n")
        self.source_paths.append("build/kernel.py")
        self.commit(self.canonical)
        self.initialize()
        materializer = HELPER.parents[1] / "kernel_workflow/scripts/materialize_workspace.sh"
        for parent in ("INCUMBENT", "FOUNDER"):
            with self.subTest(parent=parent):
                workspace = self.root / ("excluded-" + parent)
                subprocess.run(["bash", str(materializer), "--src", str(self.canonical), "--dst", str(workspace)],
                               check=True, capture_output=True)
                self.assertFalse((workspace / "build/kernel.py").exists())
                with self.assertRaises(archive.ArchiveError) as raised:
                    archive.prepare(self.storage, str(workspace), parent, initialize_index=True,
                                    canonical=str(self.canonical))
                self.assertEqual(raised.exception.code, "materialization_mismatch")
                self.assertFalse((workspace / ".git").exists())
        self.assertEqual((self.canonical / "build/kernel.py").read_text(), "value = 7\n")

    def test_declared_canonical_guard_records_exact_source_identity(self):
        self.initialize()
        code, receipt = self.cli("prepare", "--archive", str(self.storage), "--workspace", str(self.workspace),
                                 "--canonical", str(self.canonical), "--parent", "FOUNDER")
        self.assertEqual(code, 0, receipt)
        self.assertEqual(receipt["receipt"]["materialized_canonical_source_id"], self.entry("FOUNDER")["source_id"])


if __name__ == "__main__":
    unittest.main()
