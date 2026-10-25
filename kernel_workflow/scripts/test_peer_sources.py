# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks using the real native task, identity, patch and evidence readers."""

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geak.native._util import canonical_json
from kernel_workflow.scripts.peer_sources import import_source, snapshot_peers
from scripts.native_runtime.direct_external_command import checked_result, create_request, digest, snapshot
from tests.native.fixtures import fixture, raw_reports


def fixture_pair(root):
    task, peer, local = root / "task", root / "peer", root / "local"
    task.mkdir()
    contract = fixture(task)
    (task / "task.json").write_bytes(canonical_json(contract))
    shutil.copytree(task, peer / "base")
    shutil.copytree(task, local)
    def git(path, *args):
        return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.PIPE)
    commits = []
    for index, directory in enumerate((peer / "base", local)):
        git(directory, "init", "-q")
        git(directory, "config", "user.name", "Saptarshi Majumder")
        git(directory, "config", "user.email", "32464366+iraj465@users.noreply.github.com")
        git(directory, "add", "-A")
        git(directory, "commit", "-qm", f"CPU fixture base {index}")
        commits.append(git(directory, "rev-parse", "HEAD").decode().strip())
    candidate = root / "candidate"
    shutil.copytree(peer / "base", candidate)
    (candidate / "kernel_src/module.py").write_bytes(b"# peer implementation; never imported\n")
    (candidate / "kernel_src/module.py").chmod(0o755)
    (peer / "patches").mkdir()
    patch = peer / "patches/r1_d0.diff"
    patch.write_bytes(git(candidate, "diff", "--binary"))
    def verified(epoch, evaluation_id):
        request = create_request(contract, task, epoch, evaluation_id,
                                 snapshot(candidate, contract["source_paths"]), candidate)
        native = root / evaluation_id
        for name, raw in raw_reports(request).items():
            path = native / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return checked_result(request, native, peer / "base", patch)
    verifier = verified("publisher-epoch", "publisher-evaluation")
    row = {"id": "r1_d0", "title": "peer candidate", "approach": "CPU fixture",
           "parent_id": "ORIGINAL", "donor_ids": [], "verified": True, "status": "correct",
           "speedup": verifier["primary_speedup"], "verifier": verifier,
           "patch_ref": {"path": "/publisher/evolution/patches/r1_d0.diff", "sha256": digest(patch.read_bytes())}}
    other = copy.deepcopy(row)
    other["id"] = "r1_d1"
    other["patch_ref"]["path"] = "/publisher/evolution/patches/r1_d1.diff"
    (peer / "patches/r1_d1.diff").write_bytes(patch.read_bytes())
    archive = {"schema": "geak.evolution/1", "base": "/publisher/evolution/base",
               "base_commit": commits[0], "best_id": "r1_d0", "entries": [row, other]}
    (peer / "archive.json").write_bytes(canonical_json(archive))
    return {"peer": {"id": "publisher", "root": str(peer), "native_epoch": "publisher-epoch"},
            "task_root": str(task), "base": str(local), "base_commit": commits[1],
            "archive_root": str(root / "recipient"), "local_verifier": verified("recipient-epoch", "local-evaluation")}


class PeerSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = fixture_pair(self.root)
        (self.root / "recipient/patches").mkdir(parents=True)

    def freeze(self, name="round_1"):
        value = snapshot_peers([self.request["peer"]], self.root / name)
        return {**self.request, "id": value["offers"][0]["id"],
                "snapshot": {key: value[key] for key in ("path", "sha256")}}

    def test_selected_copy_identity_and_different_git_labels(self):
        request = self.freeze()
        self.assertEqual(list((self.root / "recipient/patches").iterdir()), [])
        original = snapshot(Path(request["base"]), ["kernel_src/module.py"])
        row = import_source(request)
        self.assertEqual(len(list((self.root / "recipient/patches").iterdir())), 1, "unselected patch is not copied")
        self.assertNotEqual(row["base_commit"], row["origin"]["base_commit"])
        self.assertEqual(row["origin"]["native_epoch"], "publisher-epoch")
        self.assertNotEqual(row["origin"]["external_evaluation"]["candidate_id"],
                            self.request["local_verifier"]["external_evaluation"]["candidate_id"])
        self.assertFalse(row["verified"])
        self.assertIsNone(row["speedup"])
        self.assertGreater(row["origin"]["speedup"], 0)
        self.assertEqual(Path(row["patch_ref"]["path"]).read_bytes(), (self.root / "peer/patches/r1_d0.diff").read_bytes())
        self.assertEqual(snapshot(Path(request["base"]), ["kernel_src/module.py"]), original)
        self.assertEqual(import_source(request), row)
        # The frozen archive remains usable after an atomic publisher update.
        (self.root / "peer/archive.json").write_text("{uncompleted next archive")
        self.assertEqual(import_source(request), row)

    def test_selected_source_rejections(self):
        request = self.freeze()
        paths = {
            "patch": self.root / "peer/patches/r1_d0.diff",
            "recipient_base": self.root / "local/kernel_src/module.py",
            "publisher_base": self.root / "peer/base/kernel_src/module.py",
            "recipient_task": self.root / "local/task.json",
            "publisher_task": self.root / "peer/base/task.json",
            "archive_snapshot": self.root / "round_1/peer_publisher.archive.json",
            "snapshot_index": Path(request["snapshot"]["path"]),
        }
        for name, path in paths.items():
            with self.subTest(name=name):
                old = path.read_bytes()
                try:
                    path.write_bytes(old + b"\nchanged\n")
                    with self.assertRaises((ValueError, KeyError)):
                        import_source(request)
                    self.assertEqual(list((self.root / "recipient/patches").iterdir()), [])
                finally:
                    path.write_bytes(old)
        path = paths["publisher_base"]
        path.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "actual run bases"):
            import_source(request)

    def test_declared_native_identity_and_path_mismatches(self):
        path = self.root / "peer/archive.json"
        original = json.loads(path.read_bytes())
        changes = {
            "candidate": lambda row: row["verifier"]["external_evaluation"].update(candidate_id="0" * 64),
            "evaluation": lambda row: row["verifier"]["external_evaluation"].update(evaluation_id=""),
            "hash": lambda row: row["patch_ref"].update(sha256="0" * 64),
            "path": lambda row: row["patch_ref"].update(path="/unrelated/patches/r1_d0.diff"),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                archive = copy.deepcopy(original)
                change(archive["entries"][0])
                path.write_bytes(canonical_json(archive))
                with self.assertRaises(ValueError):
                    import_source(self.freeze(name))
        path.write_bytes(canonical_json(original))
        self.request["peer"]["native_epoch"] = "wrong-epoch"
        with self.assertRaisesRegex(ValueError, "native verifier identity"):
            import_source(self.freeze("epoch"))

    def test_missing_partial_and_unverified_archives(self):
        path = self.root / "peer/archive.json"
        archive = json.loads(path.read_bytes())
        for row in archive["entries"]:
            row["verified"] = False
        path.write_bytes(canonical_json(archive))
        self.assertEqual(snapshot_peers([self.request["peer"]], self.root / "unverified")["offers"], [])
        path.write_text("{partial")
        result = snapshot_peers([self.request["peer"]], self.root / "partial")
        self.assertEqual(result["offers"], [])
        self.assertEqual(len(result["errors"]), 1)
        path.unlink()
        self.assertEqual(len(snapshot_peers([self.request["peer"]], self.root / "missing")["errors"]), 1)

    def test_late_invalid_row_discards_only_that_peers_offers(self):
        other = self.root / "other-peer"
        shutil.copytree(self.root / "peer", other)
        path = self.root / "peer/archive.json"
        archive = json.loads(path.read_bytes())
        archive["entries"][1]["id"] = "invalid-id"
        path.write_bytes(canonical_json(archive))
        valid_peer = {**self.request["peer"], "id": "other", "root": str(other)}
        result = snapshot_peers([valid_peer, self.request["peer"]], self.root / "atomic")
        self.assertEqual([peer["id"] for peer in result["peers"]], ["other"])
        self.assertEqual([offer["peer_id"] for offer in result["offers"]], ["other", "other"])
        self.assertEqual([error["peer_id"] for error in result["errors"]], ["publisher"])
        frozen = json.loads(Path(result["path"]).read_bytes())
        self.assertEqual(frozen["offers"], result["offers"])

    def test_archive_verdict_is_authoritative_but_import_needs_local_verification(self):
        path = self.root / "peer/archive.json"
        original = json.loads(path.read_bytes())
        for kind in ("normalized", "omitted", "contradictory"):
            with self.subTest(kind=kind):
                archive = copy.deepcopy(original)
                row = archive["entries"][0]
                if kind == "normalized":
                    row["verifier"]["correctness"] = " PASS: checked "
                elif kind == "omitted":
                    row.pop("status")
                    for key in ("status", "correctness", "external_measurement_valid"):
                        row["verifier"].pop(key, None)
                else:
                    row["status"] = "unverified"
                    row["verifier"].update(correctness="fail", external_measurement_valid=False)
                path.write_bytes(canonical_json(archive))
                imported = import_source(self.freeze(kind))
                self.assertEqual(imported["origin"]["id"], row["id"])
                self.assertEqual(imported["origin"]["external_evaluation"], row["verifier"]["external_evaluation"])
                self.assertFalse(imported["verified"])
                self.assertIsNone(imported["speedup"])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--fixture":
        print(json.dumps(fixture_pair(Path(sys.argv[2]))))
    else:
        unittest.main()
