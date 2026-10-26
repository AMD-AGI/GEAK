# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of population discovery and real native source/evidence replay.

Only synthetic retained evidence is produced. The query invokes the stock
native *reader* as a subprocess; no measurement command or GPU is used.
"""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.native_runtime import direct_external_command as native
from tests.native import fixtures

spec = importlib.util.spec_from_file_location("query_population", Path(__file__).with_name("query_population.py"))
population = importlib.util.module_from_spec(spec)
spec.loader.exec_module(population)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.PIPE).decode().strip()


class PopulationQueryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.population = self.root / "population"
        self.task = self.root / "task"
        self.contract = fixtures.fixture(self.task)
        save(self.task / "task.json", self.contract)
        self.manifest = {"schema": "geak.population/1", "kernel_path": str(self.task),
                         "objective": {"kind": "external_primary", "primary_regime": "extend"},
                         "task_sha256": sha(self.task / "task.json"), "lineages": []}

    def lane(self, id_, optimizer="evo"):
        eval_dir = self.root / "lineages" / id_ / "eval"
        eval_dir.mkdir(parents=True)
        lane = {"id": id_, "optimizer": optimizer, "imports": [], "eval_dir": str(eval_dir),
                "external_result_command": [sys.executable, str(REPO / "scripts/native_runtime/direct_external_command.py"),
                    "--task-root", str(self.task), "--state-root", str(eval_dir / "native"),
                    "--gpu-pool", "999999", "--epoch", "population-cpu-test", "read"]}
        self.manifest["lineages"].append(lane)
        self.flush()
        return lane

    def flush(self):
        save(self.population / "manifest.json", self.manifest)

    def checkpoint(self, lane, round_=1, *, original=False, changed_base=False):
        eval_dir = Path(lane["eval_dir"])
        base = eval_dir / "evolution" / "base"
        if not base.exists():
            shutil.copytree(self.task, base)
            if changed_base:
                (base / "notes.txt").write_text("incompatible baseline source tree\n")
                (base / "kernel_src/module.py").write_text("# accepted warm start source\n")
            git(base, "init", "-q")
            git(base, "add", "-A")
        base_tree = git(base, "write-tree")
        workspace = eval_dir / f"candidate-{round_}"
        shutil.copytree(base, workspace, ignore=shutil.ignore_patterns(".git"))
        git(workspace, "init", "-q")
        git(workspace, "add", "-A")
        if not original:
            (workspace / "kernel_src/module.py").write_text(f"# synthetic candidate {lane['id']} round {round_}\n")
        patch_raw = subprocess.check_output(["git", "-C", str(workspace), "diff", "--binary"])
        git(workspace, "add", "-A")
        source_tree = git(workspace, "write-tree")
        publication = self.population / lane["id"]
        publication.mkdir(exist_ok=True)
        patch = publication / f"r{round_}.diff"
        patch.write_bytes(patch_raw)
        original_workspace = eval_dir / f"delivery-{round_}"
        shutil.copytree(self.task, original_workspace)
        git(original_workspace, "init", "-q")
        git(original_workspace, "add", "-A")
        for name in self.contract["source_paths"]:
            (original_workspace / name).write_bytes((workspace / name).read_bytes())
        delivery = publication / f"r{round_}.delivery.diff"
        delivery.write_bytes(subprocess.check_output(["git", "-C", str(original_workspace), "diff", "--binary"]))
        captured = native.snapshot(workspace, self.contract["source_paths"])
        evaluation_id = f"cpu-{lane['id']}-r{round_}"
        request = native.create_request(self.contract, self.task, "population-cpu-test", evaluation_id, captured, workspace)
        evaluation = eval_dir / "native" / "evaluations" / evaluation_id
        save(evaluation / "request.json", request)
        for name, raw in fixtures.raw_reports(request, original=original).items():
            path = evaluation / "native" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        reference = {"candidate_id": request["binding"]["candidate_id"], "evaluation_id": evaluation_id}
        checkpoint = {"schema": "geak.population.checkpoint/1", "lineage_id": lane["id"], "optimizer": lane["optimizer"],
            "round": round_, "base": str(base), "base_commit": "1" * 40, "base_tree": base_tree, "source_tree": source_tree,
            "patch_ref": {"path": str(patch), "sha256": sha(patch), "base_commit": "1" * 40, "base_tree": base_tree},
            "metric": self.manifest["objective"], "speedup": 1.0 if original else 200 / 70,
            "verified": True, "qualification": "native_source_bound", "external_evaluation": reference,
            "base_source_id": native.identities(self.contract, "population-cpu-test", native.snapshot(base, self.contract["source_paths"]))[1],
            "source_id": request["binding"]["candidate_source_id"],
            "delivery_patch_ref": {"path": str(delivery), "sha256": sha(delivery), "base": "ORIGINAL"},
            "evidence": {"untrusted": True}}
        path = publication / f"r{round_}.json"
        save(path, checkpoint)
        save(publication / "latest.json", {"schema": "geak.population.pointer/1", "lineage_id": lane["id"],
                                         "checkpoint": str(path), "sha256": sha(path)})
        return checkpoint, path

    def query(self, **options):
        return population.query(self.population, **options)

    def test_prior_committed_gain_survives_corrupt_pointer_malformed_peer_and_absent_final(self):
        evo = self.lane("evo")
        normal = self.lane("geak", "greedy")
        crashed = self.lane("crashed")
        self.checkpoint(evo, 2)
        self.checkpoint(normal, 3)
        (self.population / "evo" / "latest.json").write_text("{broken")
        (self.population / "crashed").mkdir()
        (self.population / "crashed" / "r4.json").write_text("{broken")
        result = self.query()
        self.assertEqual(result["selected"]["lineage_id"], "evo")
        self.assertEqual(result["selected"]["kind"], "checkpoint")
        self.assertEqual([r["status"] for r in result["records"]], ["invalid", "qualified", "qualified"])
        self.assertTrue(next(r for r in result["lineages"] if r["id"] == "evo")["errors"])
        self.assertFalse((Path(crashed["eval_dir"]) / "population_final_result.json").exists())
        self.assertEqual(result["selected"]["speedup"], 200 / 70)

    def test_empty_initial_patch_is_native_no_gain_and_query_does_not_change_population(self):
        lane = self.lane("normal", "greedy")
        self.checkpoint(lane, 0, original=True)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        result = self.query()
        self.assertEqual(result["records"][0]["status"], "no_improvement")
        self.assertEqual(result["selected"]["status"], "no_gain")
        self.assertEqual(result["selected"]["round"], 0)
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_finalists_keep_each_members_best_deliverable_source(self):
        leader = self.lane("a_evo")
        ordinary = self.lane("b_geak", "greedy")
        no_gain = self.lane("no_gain")
        self.lane("pending")
        self.checkpoint(leader, 0, original=True)
        self.checkpoint(leader, 2)
        self.checkpoint(leader, 3)
        self.checkpoint(ordinary, 0, original=True)
        self.checkpoint(ordinary, 1)
        self.checkpoint(no_gain, 0, original=True)
        result = self.query()
        self.assertEqual(result["selected"]["lineage_id"], "a_evo")
        self.assertEqual([(row["lineage_id"], row["round"], row["status"]) for row in result["finalists"]],
                         [("a_evo", 2, "eligible"), ("b_geak", 1, "eligible"), ("no_gain", 0, "no_gain")])
        self.assertEqual(next(row for row in result["lineages"] if row["id"] == "pending")["status"], "pending")
        self.assertEqual(result["best_available_source"]["lineage_id"], "a_evo")

    def test_patch_and_base_tampering_disqualify_only_affected_candidates(self):
        stable = self.lane("stable", "greedy")
        patch_lane = self.lane("patch")
        base_lane = self.lane("base")
        self.checkpoint(stable, 0, original=True)
        patch, _ = self.checkpoint(patch_lane)
        base, _ = self.checkpoint(base_lane)
        Path(patch["patch_ref"]["path"]).write_text("tampered\n")
        (Path(base["base"]) / "kernel_src/module.py").write_text("changed base\n")
        result = self.query()
        self.assertEqual(result["selected"]["lineage_id"], "stable")
        reasons = {r["lineage_id"]: r.get("reason", "") for r in result["candidates"]}
        self.assertIn("patch hash", reasons["patch"])
        self.assertIn("patch does not apply", reasons["base"])

    def test_claimed_scores_flags_and_checkpoint_commands_cannot_replace_native_replay(self):
        lane = self.lane("normal", "greedy")
        checkpoint, path = self.checkpoint(lane)
        checkpoint["speedup"] = 100000
        checkpoint["evidence"] = {"native_reproducible_gain": True, "primary_speedup": 100000}
        checkpoint["external_result_command"] = ["false"]
        save(path, checkpoint)
        result = self.query()
        self.assertIsNone(result["selected"])
        self.assertIn("claimed checkpoint score", result["records"][0]["reason"])
        checkpoint["speedup"] = 200 / 70
        save(path, checkpoint)
        result = self.query()
        self.assertEqual(result["selected"]["speedup"], 200 / 70)
        self.assertTrue(result["records"][0]["evidence"]["native_reproducible_gain"])
        # Tampering retained raw timing evidence breaks native receipt replay.
        report = Path(lane["eval_dir"]) / "native/evaluations/cpu-normal-r1/native/process_03/ut_result.json"
        report.write_bytes(report.read_bytes() + b" ")
        result = self.query()
        self.assertIsNone(result["selected"])
        self.assertEqual(result["records"][0]["status"], "invalid")

    def test_native_original_delivery_compares_warmstart_cohorts_and_import_filter_is_exact(self):
        first = self.lane("a")
        second = self.lane("b")
        a, _ = self.checkpoint(first)
        self.checkpoint(second, changed_base=True)
        result = self.query()
        self.assertTrue(result["compatible"])
        self.assertEqual(result["selected"]["lineage_id"], "a")
        self.assertEqual(len(result["compatible_groups"]), 2)
        filtered = self.query(base_source_id=a["base_source_id"], lineages=["a", "b"], checkpoints_only=True)
        self.assertEqual(filtered["selected"]["lineage_id"], "a")
        self.assertEqual([row["status"] for row in filtered["records"]], ["qualified", "invalid"])
        b_path = self.population / "b/r1.json"
        b = json.loads(b_path.read_text())
        b["base_source_id"] = a["base_source_id"]
        save(b_path, b)
        filtered = self.query(base_source_id=a["base_source_id"], lineages=["b"], checkpoints_only=True)
        self.assertEqual(filtered["records"][0]["status"], "invalid")
        self.assertIn("base source identity differs", filtered["records"][0]["reason"])

    def test_final_source_join_and_task_pin_are_rechecked(self):
        lane = self.lane("normal", "greedy")
        checkpoint, _ = self.checkpoint(lane)
        final_patch = Path(lane["eval_dir"]) / "host_final_patch.diff"
        final_patch.write_bytes(Path(checkpoint["patch_ref"]["path"]).read_bytes())
        final = {"validation_status": "accepted", "final_speedup": checkpoint["speedup"],
                 "external_evaluation": checkpoint["external_evaluation"], "final_patch": str(final_patch), "rounds": 2,
                 "final_selection": {**checkpoint["external_evaluation"], "source_id": checkpoint["source_id"]}}
        final_path = Path(lane["eval_dir"]) / "population_final_result.json"
        save(final_path, final)
        result = self.query()
        self.assertEqual([r["status"] for r in result["candidates"]], ["eligible", "eligible"])
        final["final_selection"]["source_id"] = "0" * 64
        save(final_path, final)
        result = self.query()
        self.assertIn("final selection differs", result["candidates"][-1]["reason"])
        final["final_selection"]["source_id"] = checkpoint["source_id"]
        save(final_path, final)
        final_patch.write_text("wrong patch\n")
        result = self.query()
        self.assertEqual(result["selected"]["kind"], "checkpoint")
        self.assertEqual(result["candidates"][-1]["status"], "disqualified")
        (self.task / "task.json").write_bytes((self.task / "task.json").read_bytes() + b" ")
        result = self.query()
        self.assertIsNone(result["selected"])
        self.assertIn("task bytes changed", result["records"][0]["reason"])

    def test_invalid_delivery_preserves_importable_checkpoint_but_cannot_be_selected(self):
        lane = self.lane("warm")
        checkpoint, path = self.checkpoint(lane, changed_base=True)
        self.assertEqual(self.query()["selected"]["delivery_patch_ref"], checkpoint["delivery_patch_ref"])
        # A valid peer-base patch cannot masquerade as an ORIGINAL delivery.
        delivery = Path(checkpoint["delivery_patch_ref"]["path"])
        delivery.write_bytes(Path(checkpoint["patch_ref"]["path"]).read_bytes())
        checkpoint["delivery_patch_ref"]["sha256"] = sha(delivery)
        save(path, checkpoint)
        result = self.query()
        self.assertIsNone(result["selected"])
        self.assertEqual(result["records"][0]["status"], "qualified")
        self.assertEqual(result["candidates"][0]["source_status"], "qualified")
        self.assertEqual(result["best_available_source"]["patch_ref"], checkpoint["patch_ref"])
        self.assertIn("delivery:", result["candidates"][0]["reason"])

    def test_legacy_verifier_provenance_and_nonfinite_malformed_rows(self):
        lane = self.lane("ordinary", "greedy")
        checkpoint, path = self.checkpoint(lane)
        self.manifest["objective"] = {"kind": "geomean"}
        self.flush()
        checkpoint.update(metric=self.manifest["objective"], qualification="legacy_verifier", speedup=1.4,
                          evidence={"status": "verified", "correctness": "pass", "verified_geomean": 1.4})
        save(path, checkpoint)
        (path.parent / "r2.json").write_text('{"speedup":NaN}')
        result = self.query()
        self.assertIsNone(result["selected"])
        self.assertEqual([r["status"] for r in result["records"]], ["legacy_verified", "invalid"])
        self.assertEqual(result["candidates"][0]["speedup"], 1.4)
        self.assertEqual(result["candidates"][0]["status"], "checkpoint_only")
        self.assertEqual(result["best_available_source"]["speedup"], 1.4)
        self.assertEqual(result["best_available_source"]["qualification"], "legacy_verifier")
        self.assertEqual(result["best_available_source"]["source_base"], checkpoint["base"])
        # No legacy delivery, even an applicable empty diff, substitutes for a
        # final verifier's ORIGINAL-relative result.
        delivery = Path(checkpoint["delivery_patch_ref"]["path"])
        delivery.write_bytes(b"")
        checkpoint["delivery_patch_ref"]["sha256"] = sha(delivery)
        save(path, checkpoint)
        self.assertIsNone(self.query()["selected"])
        final_patch = Path(lane["eval_dir"]) / "final.diff"
        final_patch.write_bytes(Path(checkpoint["patch_ref"]["path"]).read_bytes())
        save(Path(lane["eval_dir"]) / "population_final_result.json",
             {"validation_status": "accepted", "final_speedup": 1.3, "final_patch": str(final_patch)})
        final = self.query()["selected"]
        self.assertEqual(final["kind"], "final")
        self.assertEqual(final["qualification"], "legacy_verifier")
        self.assertEqual(final["speedup"], 1.3)
        # An unverifiable native checkpoint cannot fall back to a legacy scalar.
        self.manifest["objective"] = {"kind": "external_primary", "primary_regime": "extend"}
        self.flush()
        self.assertIsNone(self.query()["selected"])


if __name__ == "__main__":
    unittest.main()
