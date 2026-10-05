# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the real materializer profile using small immutable CPU inputs."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interface.native_cost_controls.quality_stop_controller import (
    StopRejected,
    canonical,
    sha,
)
from interface.native_cost_controls.quality_stop_startup import (
    GITIGNORE,
    derive_seed_contract,
)

WORKFLOW = Path(__file__).resolve().parents[1] / "kernel_workflow"


class StartupContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = self.root / "task"
        (self.task / "kernel_src").mkdir(parents=True)
        (self.task / "kernel_src/kernel.py").write_text("raise AssertionError('Candidate code must not execute.')\n")
        self.reference = b"A pinned synthetic immutable reference.\x00\x01"
        (self.task / "reference_io.pt").write_bytes(self.reference)
        (self.task / "meta.json").write_text("{}\n")
        self.digest = hashlib.sha256(self.reference).hexdigest()

    def derive(self, **changes):
        arguments = {"task_root": self.task, "actor_task_root": "/actor/frozen-task",
                     "reference_io_sha256": self.digest, "workflow_root": WORKFLOW}
        arguments.update(changes)
        return derive_seed_contract(**arguments)

    def test_pins_seed_and_provenance_before_actor_output_exists(self):
        result = self.derive()
        expected_metadata = (json.dumps({"ancestors": ["/actor/frozen-task"]}) + "\n").encode()
        self.assertEqual(result["metadata_bytes_utf8"][".geak/workspace.json"], expected_metadata.decode())
        self.assertEqual(result["metadata_files"], {".geak/workspace.json": {
            "mode": "100644", "sha256": sha(expected_metadata), "bytes": len(expected_metadata)}})
        files = {row["path"]: row for row in result["expected_seed_files"]}
        self.assertEqual(set(files), {".gitignore", "meta.json", "kernel_src/kernel.py", "reference_io.pt"})
        self.assertEqual(files["reference_io.pt"]["sha256"], self.digest)
        self.assertEqual(files["reference_io.pt"]["mode"], "100644")
        self.assertEqual(files[".gitignore"]["sha256"], sha(GITIGNORE))
        self.assertFalse(result["actor_output_read"])
        self.assertFalse(result["candidate_code_executed"])
        self.assertEqual(result["seed_manifest_sha256"], sha(canonical({
            "files": result["expected_seed_files"], "metadata": result["metadata_files"]}).encode()))
        self.assertEqual((self.task / "reference_io.pt").read_bytes(), self.reference)
        self.assertFalse((self.task / ".geak").exists())

    def test_materializer_exclusions_and_source_modes_remain_exact(self):
        (self.task / "__pycache__").mkdir()
        (self.task / "__pycache__/ignored.pyc").write_bytes(b"not source")
        (self.task / "ignored.so").write_bytes(b"not source")
        (self.task / "kernel_src/kernel.py").chmod(0o755)
        result = self.derive()
        files = {row["path"]: row for row in result["expected_seed_files"]}
        self.assertNotIn("ignored.so", files)
        self.assertNotIn("__pycache__/ignored.pyc", files)
        self.assertEqual(files["kernel_src/kernel.py"]["mode"], "100755")

    def test_ambient_shell_options_do_not_change_preparation(self):
        original_run = subprocess.run
        seen = []
        def record(*args, **kwargs):
            seen.append(kwargs["env"])
            return original_run(*args, **kwargs)
        with patch.dict(os.environ, {"TAR_OPTIONS": "--definitely-invalid-option", "BASH_ENV": "/does/not/exist"}), \
                patch("interface.native_cost_controls.quality_stop_startup.subprocess.run", side_effect=record):
            self.derive()
        self.assertNotIn("TAR_OPTIONS", seen[0])
        self.assertNotIn("BASH_ENV", seen[0])
        self.assertEqual(seen[0].get("HOME"), os.environ.get("HOME"))

    def test_unknown_reference_metadata_paths_and_links_are_rejected(self):
        for changes in ({"reference_io_sha256": "0" * 64}, {"actor_task_root": "relative"},
                        {"task_root": Path("relative")}, {"workflow_root": Path("relative")}):
            with self.subTest(changes=changes), self.assertRaises(StopRejected):
                self.derive(**changes)
        (self.task / ".geak").mkdir()
        with self.assertRaisesRegex(StopRejected, "startup_inherited_metadata_unsupported"):
            self.derive()
        (self.task / ".geak").rmdir()
        (self.task / "kernel_src/alias.py").symlink_to("kernel.py")
        with self.assertRaisesRegex(StopRejected, "startup_source_not_regular"):
            self.derive()

    def test_gitignore_contract_and_materializer_errors_are_rejected(self):
        workflow = self.root / "workflow"
        (workflow / "scripts").mkdir(parents=True)
        (workflow / "roles").mkdir()
        for relative in ("scripts/materialize_workspace.sh", "scripts/workspace_sources.py", "roles/director.md"):
            shutil.copy2(WORKFLOW / relative, workflow / relative)
        (workflow / "roles/director.md").write_text("Changed setup instructions.\n")
        with self.assertRaisesRegex(StopRejected, "startup_gitignore_profile_changed"):
            self.derive(workflow_root=workflow)
        with patch("interface.native_cost_controls.quality_stop_startup.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 86, b"", b"synthetic failure")), \
                self.assertRaisesRegex(StopRejected, "startup_materializer_failed"):
            self.derive()

    def test_default_materializer_still_uses_the_original_reference_link(self):
        destination = self.root / "default"
        result = subprocess.run(["/bin/bash", str(WORKFLOW / "scripts/materialize_workspace.sh"),
                                 "--src", str(self.task), "--dst", str(destination)],
                                capture_output=True, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertTrue((destination / "reference_io.pt").is_symlink())
        self.assertEqual((destination / "reference_io.pt").read_bytes(), self.reference)
