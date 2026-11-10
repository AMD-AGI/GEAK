#!/usr/bin/env python3
"""CPU-only tests for metadata validation; no model or workflow execution."""

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("read_native_planner_seed.py")
SPEC = importlib.util.spec_from_file_location("native_seed_reader", SCRIPT)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def ref(path, **fields):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), **fields}


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return ref(path)


class SeedFixture:
    def __init__(self, root):
        self.root = root
        self.task = root / "task"
        (self.task / "kernel_src").mkdir(parents=True)
        (self.task / "kernel_src" / "kernel.py").write_text("# frozen source; never imported\n")
        write_json(self.task / "meta.json", {"schema": "task/1"})
        write_json(self.task / "case_manifest.json", {"cases": [{"name": "case0"}]})
        self.patch = root / "retained.diff"
        self.patch.write_text("--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-old\n+new\n")
        self.other_patch = root / "development.diff"
        self.other_patch.write_text("# distinct frozen, unverified attempt\n")
        sources = [ref(self.patch, entry_id="r3_d0"), ref(self.other_patch, entry_id="r4_d0")]
        self.original = {
            "schema": "geak.evolution/1", "base": "/historical/base", "base_commit": "abc123",
            "baseline": {"id": "ORIGINAL", "source": "/historical/base", "speedup": 1.0,
                         "origin": "historical_initialization"},
            "best_id": "r3_d0",
            "entries": [
                {"id": "r3_d0", "base_commit": "abc123", "status": "verified", "verified": True,
                 "patch_ref": {"path": "/historical/retained.diff", "sha256": sources[0]["sha256"],
                               "bytes": sources[0]["bytes"], "base_commit": "abc123"},
                 "engineer": {"measurement_valid": True, "per_case": [{"name": "case0", "speedup": 1}],
                              "notes": None, "evidence_context": {"base_commit": "keep_nested_fact"}}},
                {"id": "r3_d1", "status": "failed", "patch_ref": None, "failure": "unavailable"},
                {"id": "r3_d2", "status": "partial"},
                {"id": "r4_d0", "status": "partial", "verified": False,
                 "patch_ref": {"path": "/historical/development.diff", "sha256": sources[1]["sha256"]}},
            ],
        }
        self.view = self.expected_view(sources)
        self.seed = {
            "schema": "geak.native_planner_seed/1", "id": "J0_test",
            "original_archive": {}, "archive": {}, "source_refs": sources,
            "task_root": str(self.task),
            "task_files": [ref(self.task / "meta.json"), ref(self.task / "case_manifest.json"),
                           ref(self.task / "kernel_src" / "kernel.py")],
            "context": {
                "eval_dir": str(root / "not_created_eval"), "round": 1, "budget_remaining": None,
                "cumulative_speedup": None, "baseline_geomean_ms": None, "baseline_per_case": None,
                "current_best_per_case": None, "current_best_ref": None, "final_winner": None,
                "profile_summary": {"unavailable": "historical profile not frozen"},
                "history": {"unavailable": "historical history not frozen", "scope": "J0"},
                "gfx": None, "device_target": None, "physical_cu_count": None, "roofline_status": None,
                "kk_operator": "", "kk_language": "", "kk_refs": [], "research_brief_path": None,
                "facts": {},
            },
        }
        self.save_archives()

    def expected_view(self, sources=None):
        sources = sources if sources is not None else self.seed["source_refs"]
        by_id = {item["entry_id"]: item for item in sources}
        view = copy.deepcopy(self.original)
        view["base"] = str(self.task)
        view["base_commit"] = None
        view["baseline"]["source"] = str(self.task)
        for row in view["entries"]:
            if "base_commit" in row:
                row["base_commit"] = None
            if row.get("patch_ref") is not None:
                row["patch_ref"]["path"] = by_id[row["id"]]["path"]
                if "base_commit" in row["patch_ref"]:
                    row["patch_ref"]["base_commit"] = None
        return view

    def save_archives(self):
        self.seed["original_archive"] = write_json(self.root / "original.json", self.original)
        self.seed["archive"] = write_json(self.root / "view.json", self.view)

    def save(self):
        return write_json(self.root / "seed.json", self.seed)


class NativePlannerSeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = SeedFixture(Path(self.temp.name))

    def rejected(self, code=None):
        with self.assertRaises(reader.SeedError) as caught:
            reader.read_seed(self.fixture.save())
        if code is not None:
            self.assertEqual(str(caught.exception), code)

    def test_cli_accepts_frozen_unknown_context_without_writing_inputs(self):
        seed_ref = self.fixture.save()
        paths = [path for path in self.fixture.root.rglob("*") if path.is_file()]
        before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        result = subprocess.run([sys.executable, "-I", "-B", str(SCRIPT), seed_ref["path"],
                                 seed_ref["sha256"], str(seed_ref["bytes"])],
                                capture_output=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(set(value), {"seed", "archive", "verified_refs"})
        self.assertEqual(value["seed"], self.fixture.seed)
        self.assertEqual(value["archive"], self.fixture.view)
        self.assertEqual(value["verified_refs"], [seed_ref, self.fixture.seed["original_archive"],
                         self.fixture.seed["archive"], *self.fixture.seed["source_refs"],
                         *self.fixture.seed["task_files"]])
        self.assertEqual(before, {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in paths})
        self.assertEqual(set(before), {str(path) for path in self.fixture.root.rglob("*") if path.is_file()})
        self.assertFalse((self.fixture.root / "not_created_eval").exists())

    def test_preserves_failed_entries_nulls_missingness_and_nested_history(self):
        value = reader.read_seed(self.fixture.save())["archive"]
        self.assertEqual([row["id"] for row in value["entries"]], ["r3_d0", "r3_d1", "r3_d2", "r4_d0"])
        self.assertIsNone(value["entries"][1]["patch_ref"])
        self.assertNotIn("patch_ref", value["entries"][2])
        self.assertEqual(value["entries"][0]["engineer"]["evidence_context"]["base_commit"], "keep_nested_fact")
        self.assertEqual(value["baseline"]["origin"], "historical_initialization")

    def test_plain_context_preserves_real_observations_and_exact_reference_cap(self):
        seed = self.fixture.seed
        seed.update(original_archive=None, archive=None, source_refs=[])
        profile_ref = write_json(self.fixture.root / "profile.json", {"counters": {"bytes": 128}})
        seed["context_refs"] = [profile_ref]
        rows = [{"name": "case0", "baseline_ms": 2, "optimized_ms": 2, "speedup": 1}]
        seed["context"].update(initial_original=True, cumulative_speedup=1,
                               baseline_per_case=rows, current_best_per_case=copy.deepcopy(rows),
                               profile_summary={"bottleneck": "memory", "counters": {"bytes": 128}},
                               history={"bottleneck_now": "memory", "rounds": []},
                               profile_evidence_inputs={"PROFILE_SUMMARY": None,
                                                        "EVIDENCE_STATUS": {"items": []}})
        for profile in [seed["context"]["profile_summary"], None]:
            seed["context"]["profile_summary"] = profile
            seed_ref = self.fixture.save()
            expected_refs = [seed_ref, *seed["task_files"], profile_ref]
            with mock.patch.object(reader, "MAX_REFS", len(expected_refs)):
                result = reader.read_seed(seed_ref)
            self.assertEqual(result["seed"], seed)
            self.assertIsNone(result["archive"])
            self.assertEqual(result["verified_refs"], expected_refs)
            with mock.patch.object(reader, "MAX_REFS", len(expected_refs) - 1):
                self.rejected("reference_count_limit")

    def test_rejects_inconsistent_plain_context_and_optional_fields(self):
        seed = self.fixture.seed
        seed.update(original_archive=None, archive=None, source_refs=[])
        rows = [{"name": "case0", "time_ms": 2}]
        seed["context"].update(initial_original=True, cumulative_speedup=1,
                               baseline_per_case=rows, current_best_per_case=copy.deepcopy(rows))
        pristine = copy.deepcopy(seed)
        changes = [
            (lambda s: s.update(archive={}), "inconsistent_absent_archive"),
            (lambda s: s.update(original_archive={}), "inconsistent_absent_archive"),
            (lambda s: s.update(source_refs=[ref(self.fixture.patch, entry_id="r3_d0")]),
             "inconsistent_absent_archive"),
            (lambda s: s.update(context_refs={}), "invalid_context_refs"),
            (lambda s: s["context"].update(initial_original=1), "invalid_initial_original"),
            (lambda s: s["context"].update(current_best_per_case=[{"name": "case0", "time_ms": 3}]),
             "initial_original_context_mismatch"),
            (lambda s: s["context"].update(final_winner={"id": "ORIGINAL"}),
             "initial_original_context_mismatch"),
            (lambda s: s["context"].update(cumulative_speedup=2), "initial_original_context_mismatch"),
            (lambda s: s["context"].update(baseline_per_case=None), "initial_original_context_mismatch"),
            (lambda s: s["context"].update(profile_evidence_inputs={"EVOLUTION_ENTRIES": []}),
             "invalid_profile_evidence_inputs"),
            (lambda s: s["context"].update(profile_evidence_inputs={"EVIDENCE_STATUS": None}),
             "invalid_profile_evidence_value"),
            (lambda s: s["context"].update(profile_evidence_inputs={"PROFILE_SUMMARY": []}),
             "invalid_profile_evidence_value"),
            (lambda s: s["context"].update(profile_summary=[]), "invalid_observation_context"),
            (lambda s: s["context"].update(history=None), "invalid_observation_context"),
        ]
        for mutate, code in changes:
            with self.subTest(code=code):
                self.fixture.seed = copy.deepcopy(pristine)
                mutate(self.fixture.seed)
                self.rejected(code)

    def test_context_ref_hash_and_path_binding_are_enforced(self):
        self.fixture.seed.update(original_archive=None, archive=None, source_refs=[])
        artifact = self.fixture.root / "profile.txt"
        artifact.write_text("original profile\n")
        bound_ref = ref(artifact)
        self.fixture.seed["context_refs"] = [bound_ref]
        artifact.write_text("modified profile\n")
        self.rejected("file_hash_mismatch")
        artifact.write_text("original profile\n")
        self.fixture.seed["context_refs"].append(dict(bound_ref, sha256="0" * 64))
        self.rejected("conflicting_path_references")

    def test_rejects_view_changes_even_with_updated_view_hash(self):
        mutations = [
            lambda view: view["entries"][0]["engineer"]["per_case"][0].update(speedup=2),
            lambda view: view["entries"][0]["engineer"]["per_case"][0].update(speedup=True),
            lambda view: view["entries"][0].update(status="failed"),
            lambda view: view.update(best_id="ORIGINAL"),
            lambda view: view["entries"].reverse(),
            lambda view: view["entries"][2].update(patch_ref=None),
            lambda view: view["entries"][0]["engineer"]["evidence_context"].update(base_commit=None),
        ]
        pristine = copy.deepcopy(self.fixture.view)
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.fixture.view = copy.deepcopy(pristine)
                mutate(self.fixture.view)
                self.fixture.save_archives()
                self.rejected("archive_view_mismatch")

    def test_rejects_duplicate_or_unknown_archive_id(self):
        self.fixture.original["entries"][1]["id"] = "r3_d0"
        self.fixture.save_archives()
        self.rejected("duplicate_or_reserved_archive_id")

    def test_rejects_empty_archive(self):
        self.fixture.original["entries"] = []
        self.fixture.save_archives()
        self.rejected("empty_archive")

    def test_source_reference_bijection(self):
        original_refs = copy.deepcopy(self.fixture.seed["source_refs"])
        variants = [original_refs[:-1], original_refs + [dict(original_refs[0], entry_id="extra")]]
        for sources in variants:
            with self.subTest(sources=sources):
                self.fixture.seed["source_refs"] = sources
                self.rejected("source_reference_closure_mismatch")
        self.fixture.seed["source_refs"] = original_refs + [original_refs[0]]
        self.rejected("duplicate_source_entry_id")

    def test_rejects_digest_replacement_not_bound_to_original(self):
        self.fixture.patch.write_text("different source with an independently valid new hash\n")
        self.fixture.seed["source_refs"][0] = ref(self.fixture.patch, entry_id="r3_d0")
        self.fixture.view["entries"][0]["patch_ref"].update(
            sha256=self.fixture.seed["source_refs"][0]["sha256"],
            bytes=self.fixture.seed["source_refs"][0]["bytes"])
        self.fixture.save_archives()
        self.rejected("original_patch_hash_mismatch")

    def test_rejects_task_closure_missing_duplicate_or_escape(self):
        task_files = copy.deepcopy(self.fixture.seed["task_files"])
        self.fixture.seed["task_files"] = []
        self.rejected("invalid_seed_ref_lists")
        self.fixture.seed["task_files"] = task_files + [task_files[0]]
        self.rejected("duplicate_task_reference")
        sibling = self.fixture.root / "task-sibling"
        sibling.mkdir()
        external = sibling / "extra.py"
        external.write_text("not beneath task root\n")
        self.fixture.seed["task_files"] = task_files + [ref(external)]
        self.rejected("task_reference_outside_root")

    def test_hash_size_and_conflicting_reference_failures(self):
        seed_ref = self.fixture.save()
        with self.assertRaisesRegex(reader.SeedError, "file_hash_mismatch"):
            reader.read_seed(dict(seed_ref, sha256="0" * 64))
        with self.assertRaisesRegex(reader.SeedError, "file_size_mismatch"):
            reader.read_seed(dict(seed_ref, bytes=seed_ref["bytes"] + 1))
        state = reader.Reader()
        state.read(seed_ref)
        with self.assertRaisesRegex(reader.SeedError, "conflicting_path_references"):
            state.read(dict(seed_ref, sha256="0" * 64))

    def test_json_rejects_duplicate_keys_constants_and_overflow(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":{"b":1,"b":2}}', b'{"a":NaN}',
                    b'{"a":Infinity}', b'{"a":1e999}'):
            with self.subTest(raw=raw), self.assertRaises(reader.SeedError):
                reader.decode_json(raw)

    def test_exact_schema_and_explicit_unknowns(self):
        original = copy.deepcopy(self.fixture.seed)
        changes = [
            (lambda seed: seed.update(extra="not allowed"), "invalid_seed_schema"),
            (lambda seed: seed["context"].update(extra="not allowed"), "invalid_context_fields"),
            (lambda seed: seed["context"].update(round=True), "invalid_context_positive_integer"),
            (lambda seed: seed["context"].update(budget_remaining=0), "invalid_context_positive_integer"),
            (lambda seed: seed["context"].update(baseline_geomean_ms=False), "invalid_context_scalar"),
            (lambda seed: seed["context"].update(current_best_per_case=[]), "invalid_context_case_rows"),
            (lambda seed: seed["context"].update(history={"unavailable": "unknown", "rounds": []}),
             "invalid_unavailable_context"),
            (lambda seed: seed["context"].update(profile_summary={"unavailable": " "}),
             "invalid_unavailable_context"),
            (lambda seed: seed["context"].update(final_winner={"entry_id": "unlisted"}),
             "unknown_context_entry_id"),
        ]
        for mutate, code in changes:
            with self.subTest(code=code):
                self.fixture.seed = copy.deepcopy(original)
                mutate(self.fixture.seed)
                self.rejected(code)

    def test_known_positive_context_is_valid(self):
        self.fixture.seed["context"].update(
            budget_remaining=8, cumulative_speedup=1.2, baseline_geomean_ms=0.3,
            physical_cu_count=304, gfx="gfx950", device_target="mi355x",
            baseline_per_case=[{"name": "case0", "ms": 0.3}],
            current_best_per_case=[{"name": "case0", "baseline_ms": 0.3, "optimized_ms": 0.25}],
            current_best_ref={"entry_id": "r3_d0"}, final_winner={"id": "r3_d0"})
        reader.read_seed(self.fixture.save())

    def test_rejects_final_and_ancestor_symlinks_and_noncanonical_paths(self):
        seed_ref = self.fixture.save()
        link = self.fixture.root / "seed-link.json"
        link.symlink_to(seed_ref["path"])
        with self.assertRaises((reader.SeedError, OSError)):
            reader.read_seed(dict(seed_ref, path=str(link)))
        directory_link = self.fixture.root / "directory-link"
        directory_link.symlink_to(self.fixture.task, target_is_directory=True)
        target_ref = ref(self.fixture.task / "meta.json")
        with self.assertRaises((reader.SeedError, OSError)):
            reader.Reader().read(dict(target_ref, path=str(directory_link / "meta.json")))
        for path in (str(self.fixture.root) + "/./seed.json", str(self.fixture.root) + "//seed.json",
                     str(self.fixture.root) + "/task/../seed.json"):
            with self.subTest(path=path), self.assertRaises(reader.SeedError):
                reader.read_seed(dict(seed_ref, path=path))

    def test_fifo_is_rejected_without_reading(self):
        fifo = self.fixture.root / "fifo"
        os.mkfifo(fifo)
        with mock.patch.object(reader.os, "read") as read_call:
            with self.assertRaisesRegex(reader.SeedError, "not_regular_file"):
                reader.Reader().read({"path": str(fifo), "sha256": "0" * 64, "bytes": 0})
            read_call.assert_not_called()

    def test_file_change_during_read_is_rejected(self):
        patch_ref = ref(self.fixture.patch)
        before = self.fixture.patch.stat()
        real_read = os.read
        changed = False

        def change_after_read(fd, count):
            nonlocal changed
            data = real_read(fd, count)
            if not changed:
                changed = True
                self.fixture.patch.write_bytes(b"x" * patch_ref["bytes"])
                # A same-size write may share a filesystem timestamp tick with
                # the initial snapshot. Make the tested metadata change explicit.
                os.utime(self.fixture.patch, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
            return data

        with mock.patch.object(reader.os, "read", side_effect=change_after_read):
            with self.assertRaisesRegex(reader.SeedError, "file_changed"):
                reader.Reader().read(patch_ref)

    def test_path_replacement_during_read_is_rejected(self):
        patch_ref = ref(self.fixture.patch)
        replacement = self.fixture.root / "replacement"
        replacement.write_bytes(self.fixture.patch.read_bytes())
        real_read = os.read
        replaced = False

        def replace_after_read(fd, count):
            nonlocal replaced
            data = real_read(fd, count)
            if not replaced:
                replaced = True
                os.replace(replacement, self.fixture.patch)
            return data

        with mock.patch.object(reader.os, "read", side_effect=replace_after_read):
            with self.assertRaises(reader.SeedError):
                reader.Reader().read(patch_ref)

    def test_limits_fail_without_partial_stdout(self):
        seed_ref = self.fixture.save()
        with self.assertRaisesRegex(reader.SeedError, "invalid_reference_size"):
            reader.Reader().read(dict(seed_ref, bytes=reader.MAX_FILE_BYTES + 1))
        with mock.patch.object(reader, "MAX_TOTAL_READ_BYTES", 1):
            with self.assertRaisesRegex(reader.SeedError, "read_size_limit"):
                reader.read_seed(seed_ref)
        with mock.patch.object(reader, "MAX_REFS", 3):
            with self.assertRaisesRegex(reader.SeedError, "reference_count_limit"):
                reader.read_seed(seed_ref)
        output = type("Output", (), {"buffer": io.BytesIO()})()
        error = io.StringIO()
        with mock.patch.object(reader, "MAX_OUTPUT_BYTES", 10), \
                mock.patch.object(reader.sys, "stdout", output), \
                mock.patch.object(reader.sys, "stderr", error):
            code = reader.main([seed_ref["path"], seed_ref["sha256"], str(seed_ref["bytes"])])
        self.assertEqual(code, 1)
        self.assertEqual(output.buffer.getvalue(), b"")
        self.assertIn("output_size_limit", error.getvalue())


if __name__ == "__main__":
    unittest.main()
