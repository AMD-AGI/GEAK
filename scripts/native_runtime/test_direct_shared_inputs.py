# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for manifest-bound shared inputs in the direct measurement adapter.

The existing synthetic runner writes raw reports without launching task code.
Only the read-only mount flag is simulated; input and evidence checks are real.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.native_runtime import test_direct_external_command as base

adapter = base.adapter


class DirectSharedInputTests(unittest.TestCase):
    argv = base.DirectExternalCommandTests.argv
    invoke = base.DirectExternalCommandTests.invoke
    native_root = base.DirectExternalCommandTests.native_root

    def setUp(self):
        base.DirectExternalCommandTests.setUp(self)
        self.shared_name = "captures/blob.bin"
        self.shared_bytes = b"\x00synthetic shared input\xff\n"
        for root in (self.task, self.workspace):
            path = root / self.shared_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.shared_bytes)
        self.contract["artifacts"][self.shared_name] = adapter.digest(self.shared_bytes)
        for root in (self.task, self.workspace):
            (root / "task.json").write_text(json.dumps(self.contract))
        self.manifest = {
            "schema": "geak-native-shared-inputs-v1",
            "task_root": "/producer/task",
            "artifacts": dict(self.contract["artifacts"]),
            "sizes": {name: (self.task / name).stat().st_size for name in self.contract["artifacts"]},
            "shared_directories": ["captures"],
            "shared_files": {self.shared_name: self.contract["artifacts"][self.shared_name]},
        }
        self.manifest_path = self.root / "shared-inputs.json"
        self.manifest_sha = self.write_manifest(self.manifest)
        mount = mock.patch.object(adapter.os, "statvfs",
                                 return_value=SimpleNamespace(f_flag=adapter.os.ST_RDONLY))
        self.mount = mount.start()
        self.addCleanup(mount.stop)

    def fake_native(self, request, workspace, output, **options):
        return base.DirectExternalCommandTests.fake_native(self, request, workspace, output, **options)

    def write_manifest(self, value):
        raw = adapter.canonical_json(value) + b"\n"
        self.manifest_path.write_bytes(raw)
        return adapter.digest(raw)

    def shared_options(self):
        return ["--shared-input-manifest", str(self.manifest_path),
                "--shared-input-sha256", self.manifest_sha]

    def capture(self, *, shared=False):
        code, rows = self.invoke("measure", extra=["--capture-only",
            *(self.shared_options() if shared else [])])
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["capture_only"])
        return rows[0]

    @staticmethod
    def tree_bytes(root):
        return {path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()}

    def test_manifest_accepts_exact_pins_at_relocated_task_root(self):
        self.assertNotEqual(self.manifest["task_root"], str(self.task))
        self.assertEqual(adapter.load_shared_inputs(
            self.manifest_path, self.manifest_sha, self.task), self.manifest)
        self.assertEqual(adapter.load_shared_inputs(
            self.manifest_path, self.manifest_sha, self.task, self.contract), self.manifest)
        self.runner.assert_not_called()

    def test_manifest_rejects_changed_bytes_source_sharing_and_declaration_mismatches(self):
        self.manifest_path.write_bytes(self.manifest_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            adapter.load_shared_inputs(self.manifest_path, self.manifest_sha, self.task)
        source = self.contract["source_paths"][0]
        cases = [
            ("fields", {**self.manifest, "extra": True}, "fields differ"),
            ("schema", {**self.manifest, "schema": "other"}, "unsupported"),
            ("artifacts", {**self.manifest, "artifacts": {
                **self.manifest["artifacts"], self.shared_name: "0" * 64}}, "task artifacts differ"),
            ("shared hash", {**self.manifest, "shared_files": {
                self.shared_name: "0" * 64}}, "protected artifacts"),
            ("source", {**self.manifest, "shared_directories": ["kernel_src"],
                "shared_files": {source: self.contract["artifacts"][source]}}, "never candidate source"),
            ("size", {**self.manifest, "sizes": {
                **self.manifest["sizes"], self.shared_name: len(self.shared_bytes) + 1}}, "file size differs"),
            ("size closure", {**self.manifest, "sizes": {
                self.shared_name: len(self.shared_bytes)}}, "exact task artifact sizes"),
        ]
        for name, value, message in cases:
            with self.subTest(case=name):
                expected = self.write_manifest(value)
                with self.assertRaisesRegex(ValueError, message):
                    adapter.load_shared_inputs(self.manifest_path, expected, self.task)
        self.runner.assert_not_called()

    def test_shared_opt_in_requires_both_pins_and_readonly_task_root(self):
        for extra in (["--shared-input-manifest", str(self.manifest_path)],
                      ["--shared-input-sha256", self.manifest_sha]):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "supplied together"):
                self.invoke("measure", extra=["--capture-only", *extra])
        self.mount.return_value = SimpleNamespace(f_flag=0)
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.capture(shared=True)
        self.mount.side_effect = lambda path: SimpleNamespace(
            f_flag=0 if path == self.task / self.shared_name else adapter.os.ST_RDONLY)
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.capture(shared=True)
        self.runner.assert_not_called()
        self.assertFalse(self.state.exists())

    def test_exact_shared_link_requires_opt_in_and_passes_exact_shared_files(self):
        (self.workspace / self.shared_name).unlink()
        (self.workspace / self.shared_name).symlink_to(self.task / self.shared_name)
        with self.assertRaises((OSError, ValueError)):
            self.capture()
        self.runner.assert_not_called()
        identity = self.capture(shared=True)
        code, rows = self.invoke("measure", identity, extra=self.shared_options())
        self.assertEqual(code, 0)
        self.assertTrue(rows[-1]["external_measurement_valid"])
        self.runner.assert_called_once()
        self.assertEqual(self.runner.call_args.kwargs,
                         {"protected_root": self.task, "shared_files": self.manifest["shared_files"]})
        self.assertTrue((self.workspace / self.shared_name).is_symlink())
        self.assertEqual((self.task / self.shared_name).read_bytes(), self.shared_bytes)

    def test_pinned_environment_supplies_shared_manifest_defaults(self):
        local = self.workspace / self.shared_name
        local.unlink()
        local.symlink_to(self.task / self.shared_name)
        with mock.patch.dict(os.environ, {
                "GEAK_SHARED_INPUT_MANIFEST": str(self.manifest_path),
                "GEAK_SHARED_INPUT_SHA256": self.manifest_sha,
                "GEAK_SHARED_INPUT_ROOT": str(self.task)}):
            identity = self.capture()
            code, rows = self.invoke("measure", identity)
        self.assertEqual(code, 0)
        self.assertTrue(rows[-1]["external_measurement_valid"])
        self.assertEqual(self.runner.call_args.kwargs,
                         {"protected_root": self.task, "shared_files": self.manifest["shared_files"]})

    def test_captured_measurement_keeps_shared_inputs_outside_source_snapshot(self):
        identity = self.capture(shared=True)
        self.candidate.write_bytes(b"# independent successor\n")
        code, rows = self.invoke("measure", identity, extra=["--captured", *self.shared_options()])
        self.assertEqual(code, 0)
        self.assertEqual(rows[-1]["source_id"], identity["source_id"])
        self.assertFalse((Path(identity["source_workspace"]) / self.shared_name).exists())
        self.assertEqual(self.runner.call_args.kwargs,
                         {"protected_root": self.task, "shared_files": self.manifest["shared_files"]})

    def test_local_protected_changes_and_unadmitted_links_are_rejected(self):
        local = self.workspace / self.shared_name
        alternate = self.root / "same-bytes.bin"
        alternate.write_bytes(self.shared_bytes)
        for kind in ("changed shared", "wrong target", "relative target", "dangling target", "changed unshared"):
            with self.subTest(kind=kind):
                if kind == "changed unshared":
                    changed = self.workspace / "ut_main.py"
                    original = changed.read_bytes()
                    changed.write_bytes(b"# changed nonshared harness\n")
                else:
                    local.unlink()
                    if kind == "changed shared":
                        local.write_bytes(b"replacement shared input\n")
                    else:
                        target = (alternate if kind == "wrong target" else
                                  Path(os.path.relpath(self.task / self.shared_name, local.parent))
                                  if kind == "relative target" else self.root / "absent")
                        local.symlink_to(target)
                with self.assertRaises((OSError, ValueError)):
                    self.capture(shared=True)
                self.runner.assert_not_called()
                if kind == "changed unshared":
                    changed.write_bytes(original)
                else:
                    local.unlink()
                    local.write_bytes(self.shared_bytes)

    def test_missing_shared_input_source_links_and_parent_links_are_rejected(self):
        local = self.workspace / self.shared_name
        local.unlink()
        with self.assertRaises((OSError, ValueError)):
            self.capture(shared=True)
        local.symlink_to(self.task / self.shared_name)
        moved = self.workspace / "moved-captures"
        local.parent.rename(moved)
        local.parent.symlink_to(moved, target_is_directory=True)
        with self.assertRaises((OSError, ValueError)):
            self.capture(shared=True)
        local.parent.unlink()
        moved.rename(local.parent)
        for name in ("ut_main.py", self.contract["source_paths"][0]):
            path = self.workspace / name
            raw = path.read_bytes()
            path.unlink()
            path.symlink_to(self.task / name)
            with self.assertRaises((OSError, ValueError)):
                self.capture(shared=True)
            path.unlink()
            path.write_bytes(raw)
        self.runner.assert_not_called()

    def test_same_size_task_tampering_fails_even_with_valid_workspace_copy(self):
        path = self.task / self.shared_name
        path.write_bytes(b"x" * len(self.shared_bytes))
        with self.assertRaises(ValueError):
            self.capture(shared=True)
        self.runner.assert_not_called()
        self.assertEqual((self.workspace / self.shared_name).read_bytes(), self.shared_bytes)

    def test_shared_links_preserve_capture_ids_and_raw_replay(self):
        full = self.capture()
        code, measured = self.invoke("measure", full)
        self.assertEqual(code, 0)
        self.assertEqual(self.runner.call_args.kwargs, {})
        raw_before = self.tree_bytes(self.native_root(full))
        request_before = (self.native_root(full).parent / "request.json").read_bytes()
        (self.workspace / self.shared_name).unlink()
        (self.workspace / self.shared_name).symlink_to(self.task / self.shared_name)
        linked = self.capture(shared=True)
        for key in ("candidate_id", "source_id"):
            self.assertEqual(linked[key], full[key])
        self.assertNotEqual(linked["evaluation_id"], full["evaluation_id"])
        for action in ("read", "measure"):
            code, replayed = self.invoke(action, full, extra=self.shared_options())
            self.assertEqual(code, 0)
            self.assertEqual(replayed[-1], measured[-1])
        self.assertEqual(self.tree_bytes(self.native_root(full)), raw_before)
        self.assertEqual((self.native_root(full).parent / "request.json").read_bytes(), request_before)
        self.runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
