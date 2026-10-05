# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the stopping state machine with explicit CPU-only test adapters."""

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from interface.native_cost_controls.quality_stop_controller import (
    PREFIX,
    PROTOCOL,
    ExecutionBoundary,
    ProcessEvaluator,
    QualityStopController,
    SelectedArtifacts,
    StopRejected,
    canonical,
    parse_task,
    source_manifest,
)


def fixture_census():
    return {"schema": "geak-quality-native-census-v1", "synthetic_unit_fixture": True}


def capture_census(observed, result):
    observed.append(deepcopy(result))
    return fixture_census()


class FixtureBoundary(ExecutionBoundary):
    """A controlled fixture. This adapter makes no production isolation claim."""

    def __init__(self):
        self.active = False
        self.events = []
        self.reject_stage = None
        self.reject_exit = False

    def check(self, *, stage, candidate_root, protected_paths):
        self.events.append(stage)
        if stage == self.reject_stage:
            raise StopRejected("fixture_authority_lost")
        if stage != "native_return" and not self.active:
            raise AssertionError("A measurement check escaped its lease.")
        if not protected_paths or not candidate_root.is_dir():
            raise AssertionError("The fixture lost its source binding.")

    @contextmanager
    def lease(self, *, candidate_root, protected_paths):
        if self.active:
            raise AssertionError("A second fixture lease started.")
        self.active = True
        self.events.append("lease_enter")
        try:
            yield
        finally:
            self.active = False
            self.events.append("lease_exit")
            if self.reject_exit:
                raise StopRejected("fixture_lease_exit_failed")


class FixtureSigner:
    """A recording test signer. Real RSA verification has separate tests."""

    def __init__(self, boundary):
        self.public_jwk = {"fixture_only": True}
        self.boundary = boundary
        self.values = []

    def sign(self, value):
        if not self.boundary.active:
            raise AssertionError("The decision escaped its protected lease.")
        self.values.append(deepcopy(value))
        payload = canonical(value)
        return {"payload": payload, "signature": hashlib.sha256(payload.encode()).hexdigest()}


class FixtureEvaluator(ProcessEvaluator):
    def __init__(self, boundary):
        self.boundary = boundary
        self.calls = []
        self.scores = [1.08, 1.081, 1.082]
        self.before = None
        self.change_record = None

    def run_process(self, *, snapshot, snapshot_sha256, process_id, order):
        if not self.boundary.active:
            raise AssertionError("A process escaped its protected lease.")
        state = json.loads((snapshot.parent / "state.json").read_text())
        look = state["looks"][process_id.split(":look:", 1)[1].split(":", 1)[0]]
        if len(look["orders"]) != 3 or look["processes"][-1]["process_id"] != process_id:
            raise AssertionError("The process started before its admission persisted.")
        if (snapshot / ".git").exists():
            raise AssertionError("The snapshot includes Git execution state.")
        if any(p.stat().st_mode & 0o222 for p in snapshot.rglob("*") if p.is_file()):
            raise AssertionError("A snapshot file is writable.")
        index = len(self.calls)
        self.calls.append({"process_id": process_id, "order": order, "snapshot": snapshot})
        if self.before:
            self.before(index, snapshot)
        score = self.scores[index % len(self.scores)]
        result = {"process_id": process_id, "snapshot_sha256": snapshot_sha256, "order": order,
                  "buckets": [{"bucket_id": "a", "reference_ms": 10.0, "candidate_ms": 10.0 / score},
                              {"bucket_id": "b", "reference_ms": 20.0, "candidate_ms": 20.0 / score}]}
        if self.change_record:
            self.change_record(result)
        return result


class QualityStopControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "candidate"
        self.repo.mkdir()
        self.git("init", "-q", "--template=")
        (self.repo / "kernel.py").write_text("def kernel(value):\n    return value\n")
        (self.repo / "meta.json").write_text('{"fixture":true}\n')
        self.git("add", "kernel.py", "meta.json")
        self.git("commit", "-q", "-m", "fixture source")
        self.seed_commit = self.git("rev-parse", "HEAD")
        (self.repo / "kernel.py").write_text("def kernel(value):\n    return value + 0\n")
        self.git("add", "kernel.py")
        self.git("commit", "-q", "-m", "fixture candidate")
        self.canonical_patch = self.git("diff", "--no-ext-diff", "--no-textconv", "--no-color",
                                        "--src-prefix=a/", "--dst-prefix=b/", self.seed_commit, "HEAD", "--", raw=True)
        self.patch_path = self.root / "final.patch"
        self.export_root = self.root / "exports"
        self.actor_patch_path = "/actor/reports/final.patch"
        self.actor_export_root = "/actor/exports"
        self.restore_artifacts()
        self.boundary = FixtureBoundary()
        self.evaluator = FixtureEvaluator(self.boundary)
        self.signer = FixtureSigner(self.boundary)
        self.now = 1000.0
        self.serial = 0
        self.census_calls = 0

    def git(self, *args, raw=False):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
        result = subprocess.check_output(["git", "-c", "user.name=Fixture", "-c",
                                        "user.email=fixture@example.invalid", "-c",
                                        "core.hooksPath=/dev/null", "-C", str(self.repo), *args],
                                       stderr=subprocess.PIPE, env=env, text=True)
        return result if raw else result.strip()

    def restore_artifacts(self):
        for path in (self.patch_path, self.export_root):
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        self.patch_path.write_text(self.canonical_patch)
        (self.export_root / "optimized").mkdir(parents=True)
        (self.export_root / "optimized" / "kernel.py").write_bytes((self.repo / "kernel.py").read_bytes())

    def selected_artifacts(self, **changes):
        options = {"baseline_commit": self.seed_commit, "patch_path": self.patch_path,
                   "actor_patch_path": self.actor_patch_path, "export_root": self.export_root,
                   "actor_export_root": self.actor_export_root, "files": {"kernel.py": "optimized/kernel.py"}}
        options.update(changes)
        return SelectedArtifacts(**options)

    def controller(self, **changes):
        self.serial += 1
        options = {"state_dir": self.root / ("state_" + str(self.serial)), "candidate_root": self.repo,
                       "trial_id": "fixture_trial_1234567890", "budget": 6, "max_no_improve": 2,
                       "deadline_epoch": 2000.0, "buckets": {"a": 2.0, "b": 1.0},
                       "boundary": self.boundary, "evaluator": self.evaluator, "signer": self.signer,
                       "selected_artifacts": self.selected_artifacts(),
                       "clock": lambda: self.now}
        options.update(changes)
        return QualityStopController(**options)

    def request(self, **changes):
        value = {"protocol": PROTOCOL, "trial_id": "fixture_trial_1234567890", "stage": "boundary",
                     "look_index": 1, "round": 1, "dispatched": 2, "budget": 6, "no_improve": 0, "max_no_improve": 2,
                     "forced_replans": 0, "deadline_epoch": 2000.0, "candidate_root": str(self.repo)}
        if changes.get("stage") == "finalize":
            value.update(final_patch=self.actor_patch_path, director_final_patch=self.actor_patch_path,
                         export_root=self.actor_export_root)
        value.update(changes)
        return PREFIX + canonical(value)

    def binding(self, agent_id="checkpoint_agent", **changes):
        result = {"session_id": "session", "run_id": "run", "root_tool": "tool", "root_task": "task", "agent_id": agent_id}
        result.update(changes)
        return result

    def census(self):
        self.census_calls += 1
        if not self.boundary.active:
            raise AssertionError("The census escaped the protected lease.")

    def checkpoint(self, controller, task=None, binding=None, census=None):
        return controller.checkpoint(task or self.request(), binding or self.binding(),
                                     census=census or self.census)

    def consume(self, controller, census=None):
        return self.checkpoint(controller, self.request(stage="consume"), self.binding("consume_agent"), census=census)

    def prepare_seed_handoff(self, *, metadata=False, stopping_enabled=True, expected_changes=None):
        self.git("reset", "--hard", self.seed_commit)
        metadata_files = {}
        if metadata:
            path = self.repo / ".geak/workspace.json"
            path.parent.mkdir()
            data = b'{"fixture":"trusted launch metadata"}\n'
            path.write_bytes(data)
            metadata_files[path.relative_to(self.repo).as_posix()] = {
                "mode": "100644", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        expected = source_manifest(self.repo, metadata_files=metadata_files)["files"]
        if expected_changes:
            expected_changes(expected)
        selected = self.selected_artifacts(baseline_commit=None, expected_seed_files=expected, metadata_files=metadata_files)
        return self.controller(selected_artifacts=selected, stopping_enabled=stopping_enabled)

    def seed_request(self, controller, **changes):
        value = {"protocol": PROTOCOL, "trial_id": controller.trial_id, "stage": "seed",
                 "candidate_root": controller.actor_candidate_root,
                 "seed_manifest_sha256": controller.selected_artifacts.expected_seed_sha256}
        value.update(changes)
        return PREFIX + canonical(value)

    def seed(self, controller, **changes):
        return self.checkpoint(controller, self.seed_request(controller, **changes), self.binding("seed_agent"))

    def successful_result(self, controller):
        first = self.checkpoint(controller)
        consumed = self.consume(controller)
        final = self.checkpoint(controller, self.request(stage="finalize"), self.binding("final_agent"))
        result = {"quality_stop": {"enabled": True, "qualifying": True, "stopping_enabled": controller.stopping_enabled,
                                 "final_check": json.loads(final["payload"]),
                                 "certificate": {"request": parse_task(self.request()),
                                                 "decision": json.loads(first["payload"]),
                                                 "consumption": json.loads(consumed["payload"])}},
                "stopped_by": "quality_certificate", "budget_used": 2, "budget_total": 6,
                "final_patch": self.actor_patch_path}
        if controller.seed_attempt:
            result["quality_stop"]["seed"] = json.loads(controller.seed_attempt["envelope"]["payload"])
        return result

    def ordinary_result(self, controller):
        result = {"quality_stop": {"enabled": True, "qualifying": False, "stopping_enabled": controller.stopping_enabled,
                                   "final_artifacts": {"final_patch": self.actor_patch_path,
                                                       "director_final_patch": self.actor_patch_path,
                                                       "export_root": self.actor_export_root}},
                  "stopped_by": "budget", "budget_used": 6, "budget_total": 6, "final_patch": self.actor_patch_path}
        if controller.seed_attempt:
            result["quality_stop"]["seed"] = json.loads(controller.seed_attempt["envelope"]["payload"])
        return result

    def test_valid_boundary_is_not_a_final_qualification(self):
        controller = self.controller()
        envelope = self.checkpoint(controller)
        decision = json.loads(envelope["payload"])
        self.assertTrue(decision["certified"])
        self.assertFalse(decision["qualifying"])
        self.assertFalse(controller.finalized)
        self.assertEqual(len(self.evaluator.calls), 3)
        self.assertEqual(len({x["process_id"] for x in self.evaluator.calls}), 3)
        self.assertEqual(self.census_calls, 2)
        self.assertFalse(self.boundary.active)
        self.assertEqual(controller.looks["1"]["orders"], [x["order"] for x in self.evaluator.calls])

    def test_actor_candidate_mapping_uses_only_host_source_for_measurement(self):
        actor_root = "/actor/candidate"
        controller = self.controller(actor_candidate_root=actor_root)
        self.assertEqual(controller.public_config["candidate_root"], actor_root)
        self.assertEqual(controller.candidate_root, self.repo)
        with self.assertRaises(StopRejected):
            self.checkpoint(controller)
        boundary_task = self.request(candidate_root=actor_root)
        first = self.checkpoint(controller, boundary_task)
        consumed = self.checkpoint(controller, self.request(stage="consume", candidate_root=actor_root),
                                  self.binding("consume_agent"))
        final = self.checkpoint(controller, self.request(stage="finalize", candidate_root=actor_root),
                               self.binding("final_agent"))
        result = {"quality_stop": {"enabled": True, "qualifying": True, "stopping_enabled": controller.stopping_enabled,
                                   "final_check": json.loads(final["payload"]),
                                   "certificate": {"request": parse_task(boundary_task),
                                                   "decision": json.loads(first["payload"]),
                                                   "consumption": json.loads(consumed["payload"])}},
                  "stopped_by": "quality_certificate", "budget_used": 2, "budget_total": 6,
                  "final_patch": self.actor_patch_path}
        controller.bind_native_closure(lambda _: fixture_census())
        controller.confirm_native_return(result)
        self.assertFalse(controller.failed)
        for process in self.evaluator.calls:
            self.assertTrue(process["snapshot"].is_relative_to(controller.state_dir))
            self.assertEqual((process["snapshot"] / "kernel.py").read_bytes(), (self.repo / "kernel.py").read_bytes())

    def test_unbound_seed_blocks_measurement_and_binds_from_host_git_once(self):
        controller = self.prepare_seed_handoff()
        self.assertFalse(controller.seed_bound)
        with self.assertRaises(StopRejected):
            self.checkpoint(controller)
        self.assertFalse(self.evaluator.calls)
        payload = json.loads(self.seed(controller)["payload"])
        self.assertTrue(payload["seed_bound"])
        self.assertFalse(payload["certified"])
        self.assertFalse(payload["qualifying"])
        self.assertEqual(payload["seed_commit"], self.git("rev-parse", "HEAD"))
        self.assertEqual(controller.selected_artifacts.baseline_commit, self.seed_commit)
        self.assertIsNone(controller.public_config["selected_artifacts"]["baseline_commit"])
        self.assertTrue(json.loads((controller.state_dir / "state.json").read_text())["seed_bound"])
        with self.assertRaises(StopRejected):
            self.seed(controller)
        self.assertTrue(controller.failed)
        self.assertFalse(self.evaluator.calls)

    def test_seed_handoff_and_allowed_edit_preserve_metadata_outside_snapshot(self):
        controller = self.prepare_seed_handoff(metadata=True)
        self.seed(controller)
        (self.repo / "kernel.py").write_text("def kernel(value):\n    return value + 0\n")
        self.git("add", "kernel.py")
        self.git("commit", "-q", "-m", "candidate after host seed handoff")
        self.canonical_patch = self.git("diff", "--no-ext-diff", "--no-textconv", "--no-color",
                                        "--src-prefix=a/", "--dst-prefix=b/", self.seed_commit, "HEAD", "--", raw=True)
        self.restore_artifacts()
        result = self.successful_result(controller)
        controller.bind_native_closure(lambda _: fixture_census())
        controller.confirm_native_return(result)
        self.assertFalse(controller.failed)
        self.assertEqual(controller.certificate["manifest"]["metadata"], controller.metadata_files)
        self.assertNotIn(".geak/workspace.json", [x["path"] for x in controller.certificate["manifest"]["files"]])
        self.assertFalse((self.evaluator.calls[0]["snapshot"] / ".geak").exists())

    def test_seed_mismatch_or_nonfresh_history_cannot_bind(self):
        for failure in ("content", "history"):
            with self.subTest(failure=failure):
                changes = (lambda records: records[0].update(sha256="0"*64)) if failure == "content" else None
                controller = self.prepare_seed_handoff(expected_changes=changes)
                if failure == "history":
                    self.git("commit", "-q", "--allow-empty", "-m", "extra seed history")
                payload = json.loads(self.seed(controller)["payload"])
                self.assertFalse(payload["seed_bound"])
                self.assertIsNone(payload["seed_commit"])
                self.assertTrue(controller.failed)
                self.assertFalse(controller.seed_bound)
                self.assertIsNone(controller.selected_artifacts.baseline_commit)
                with self.assertRaises(StopRejected):
                    self.checkpoint(controller)
        self.assertFalse(self.evaluator.calls)

    def test_changed_seed_manifest_request_has_no_authority(self):
        controller = self.prepare_seed_handoff()
        with self.assertRaises(StopRejected):
            self.seed(controller, seed_manifest_sha256="0"*64)
        self.assertFalse(controller.seed_bound)
        self.assertIsNone(controller.seed_attempt)
        self.assertFalse(self.evaluator.calls)

    def test_control_requires_seed_and_never_admits_stopping_measurements(self):
        controller = self.prepare_seed_handoff(stopping_enabled=False)
        with self.assertRaises(StopRejected):
            self.checkpoint(controller)
        self.seed(controller)
        self.assertTrue(controller.seed_bound)
        self.assertFalse(controller.public_config["stopping_enabled"])
        with self.assertRaises(StopRejected):
            self.checkpoint(controller)
        self.assertFalse(controller.looks)
        self.assertFalse(self.evaluator.calls)

    def test_unseeded_control_still_checks_native_closure_and_rejects_success(self):
        controller = self.prepare_seed_handoff(stopping_enabled=False)
        observed = []
        controller.bind_native_closure(lambda result: capture_census(observed, result))
        result = self.ordinary_result(controller)
        with self.assertRaises(StopRejected):
            controller.confirm_native_return(result)
        self.assertEqual(observed, [result])
        self.assertTrue(controller.failed)
        self.assertFalse(self.evaluator.calls)

    def test_ordinary_control_closure_binds_artifacts_without_stopping_samples(self):
        controller = self.controller(stopping_enabled=False)
        controller.bind_native_closure(lambda _: fixture_census())
        result = self.ordinary_result(controller)
        controller.confirm_native_return(result)
        self.assertFalse(controller.failed)
        self.assertIsNotNone(controller.closure_receipt)
        self.assertTrue((controller.state_dir / "snapshot_final/kernel.py").is_file())
        self.assertFalse(self.evaluator.calls)
        self.assertFalse(controller.looks)
        payload = json.loads(controller.closure_receipt["payload"])
        census = controller.state_dir / payload["native_census"]["path"]
        self.assertEqual(json.loads(census.read_bytes()), fixture_census())
        self.assertEqual(hashlib.sha256(census.read_bytes()).hexdigest(), payload["native_census"]["sha256"])
        self.assertEqual(len(census.read_bytes()), payload["native_census"]["bytes"])

    def test_missing_closure_census_cannot_produce_a_signed_receipt(self):
        controller = self.controller(stopping_enabled=False)
        controller.bind_native_closure(lambda _: None)
        with self.assertRaisesRegex(StopRejected, "native_closure_census_missing"):
            controller.confirm_native_return(self.ordinary_result(controller))
        self.assertIsNone(controller.closure_receipt)

    def test_closure_census_write_failure_cannot_produce_a_signed_receipt(self):
        controller = self.controller(stopping_enabled=False)
        controller.bind_native_closure(lambda _: fixture_census())
        (controller.state_dir / "native_census.json").write_text("prior evidence")
        with self.assertRaises(FileExistsError):
            controller.confirm_native_return(self.ordinary_result(controller))
        self.assertIsNone(controller.closure_receipt)

    def test_ordinary_closure_rejects_impossible_budget_or_malformed_artifacts(self):
        changes = [lambda x: x.update(budget_used=7),
                   lambda x: x["quality_stop"].update(final_artifacts=[]),
                   lambda x: x["quality_stop"].pop("stopping_enabled"),
                   lambda x: x["quality_stop"].update(stopping_enabled=0)]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                controller = self.controller(stopping_enabled=False)
                controller.bind_native_closure(lambda _: fixture_census())
                result = self.ordinary_result(controller)
                change(result)
                with self.assertRaises(StopRejected):
                    controller.confirm_native_return(result)
                self.assertTrue(controller.failed)

    def test_changed_metadata_and_immutable_seed_files_reject_before_measurement(self):
        controller = self.prepare_seed_handoff(metadata=True)
        self.seed(controller)
        metadata = self.repo / ".geak/workspace.json"
        original = metadata.read_bytes()
        metadata.write_text("changed metadata\n")
        self.assertFalse(json.loads(self.checkpoint(controller)["payload"])["certified"])
        self.assertFalse(self.evaluator.calls)
        metadata.write_bytes(original)
        controller = self.controller(selected_artifacts=self.selected_artifacts(metadata_files=controller.metadata_files))
        (self.repo / "meta.json").write_text('{"fixture":false}\n')
        self.git("add", "meta.json")
        self.git("commit", "-q", "-m", "forbidden immutable source change")
        self.assertFalse(json.loads(self.checkpoint(controller)["payload"])["certified"])
        self.assertFalse(self.evaluator.calls)

    def test_final_qualification_needs_native_closure_and_exact_result(self):
        controller = self.controller()
        result = self.successful_result(controller)
        observed = []
        controller.bind_native_closure(lambda value: capture_census(observed, value))
        controller.confirm_native_return(result)
        self.assertEqual(observed, [result])
        self.assertTrue(controller.finalized)
        self.assertFalse(controller.failed)
        self.assertEqual(len(self.evaluator.calls), 3)

    def test_selected_artifact_receipt_binds_seed_patch_and_export(self):
        controller = self.controller()
        result = self.successful_result(controller)
        receipt = result["quality_stop"]["final_check"]["selected_artifacts"]
        self.assertNotEqual(self.seed_commit, self.git("rev-parse", "HEAD"))
        self.assertTrue(self.canonical_patch)
        self.assertEqual(receipt["baseline_commit"], self.seed_commit)
        self.assertEqual(receipt["final_patch"], {"path": self.actor_patch_path,
                                                "bytes": len(self.canonical_patch.encode()),
                                                "sha256": hashlib.sha256(self.canonical_patch.encode()).hexdigest()})
        self.assertEqual(receipt["files"][0]["candidate_path"], "kernel.py")
        self.assertEqual(receipt["files"][0]["path"], self.actor_export_root + "/optimized/kernel.py")
        self.assertEqual(receipt["files"][0]["sha256"], hashlib.sha256((self.repo / "kernel.py").read_bytes()).hexdigest())
        self.assertEqual(controller.public_config["selected_artifacts"], self.selected_artifacts().public_config)

    def test_finalization_requires_fresh_signed_consumption(self):
        controller = self.controller()
        with self.assertRaises(StopRejected):
            self.consume(controller)
        self.checkpoint(controller)
        with self.assertRaises(StopRejected):
            self.checkpoint(controller, self.request(stage="finalize"), self.binding("final_agent"))
        self.assertFalse(controller.consumed)
        consumed = self.consume(controller)
        payload = json.loads(consumed["payload"])
        self.assertTrue(payload["consumed"])
        self.assertTrue(payload["certified"])
        self.assertFalse(payload["qualifying"])
        self.assertTrue(controller.consumed)
        final = json.loads(self.checkpoint(controller, self.request(stage="finalize"), self.binding("final_agent"))["payload"])
        self.assertEqual(final["consumption_sha256"], hashlib.sha256(consumed["payload"].encode("ascii")).hexdigest())
        self.assertTrue(final["qualifying"])
        self.assertEqual(len(self.evaluator.calls), 3)

    def test_consumption_checks_current_source_clock_census_and_authority(self):
        for failure in ("source", "clock", "census", "authority"):
            with self.subTest(failure=failure):
                controller = self.controller()
                self.checkpoint(controller)
                original = (self.repo / "kernel.py").read_bytes()
                census = self.census
                if failure == "source":
                    (self.repo / "kernel.py").write_text("changed before consumption\n")
                elif failure == "clock":
                    self.now = 2000.0
                elif failure == "authority":
                    self.boundary.reject_stage = "consume"
                else:
                    def census():
                        raise StopRejected("producer_returned_before_exit")
                payload = json.loads(self.consume(controller, census=census)["payload"])
                self.assertFalse(payload["consumed"])
                self.assertFalse(payload["certified"])
                self.assertTrue(controller.failed)
                self.assertFalse(controller.consumed)
                self.assertEqual(len(controller.looks["1"]["processes"]), 3)
                with self.assertRaises(StopRejected):
                    self.checkpoint(controller, self.request(stage="finalize"), self.binding("final_agent"))
                (self.repo / "kernel.py").write_bytes(original)
                self.now = 1000.0
                self.boundary.reject_stage = None
                with self.assertRaises(StopRejected):
                    self.consume(controller)

    def test_consumption_duplicate_is_checked_without_more_measurements(self):
        controller = self.controller()
        self.checkpoint(controller)
        first = self.consume(controller)
        self.assertEqual(self.consume(controller), first)
        self.assertEqual(len(self.evaluator.calls), 3)
        with self.assertRaises(StopRejected):
            self.checkpoint(controller, self.request(stage="consume"), self.binding("different_consume_agent"))
        with self.assertRaises(StopRejected):
            self.checkpoint(controller, self.request(stage="consume", round=2), self.binding("consume_agent"))
        self.now = 2000.0
        with self.assertRaises(StopRejected):
            self.consume(controller)
        self.assertTrue(controller.failed)
        self.assertTrue(json.loads((controller.state_dir / "state.json").read_text())["failed"])
        self.assertEqual(len(self.evaluator.calls), 3)

    def test_consumption_persistence_and_lease_failures_cannot_replay_success(self):
        for failure in ("persistence", "lease_exit"):
            with self.subTest(failure=failure):
                controller = self.controller()
                self.checkpoint(controller)
                if failure == "persistence":
                    persist = controller._persist
                    def persist_consumption(controller=controller, persist=persist):
                        if controller.certificate.get("consume_envelope"):
                            controller.failed = True
                            raise OSError("consume persistence failed")
                        persist()
                    with patch.object(controller, "_persist", side_effect=persist_consumption), self.assertRaises(OSError):
                        self.consume(controller)
                else:
                    self.boundary.reject_exit = True
                    with self.assertRaises(StopRejected):
                        self.consume(controller)
                    self.boundary.reject_exit = False
                self.assertTrue(controller.failed)
                with self.assertRaises(StopRejected):
                    self.consume(controller)
                with self.assertRaises(StopRejected):
                    self.checkpoint(controller, self.request(stage="finalize"), self.binding("final_agent"))

    def test_final_artifacts_reject_missing_changed_redirected_or_extra_files(self):
        for failure in ("patch_missing", "patch_changed", "patch_link", "patch_hardlink",
                        "export_missing", "export_changed", "export_link", "export_hardlink",
                        "export_extra", "export_mode", "export_root_link", "export_directory_link"):
            with self.subTest(failure=failure):
                self.restore_artifacts()
                controller = self.controller()
                self.checkpoint(controller)
                self.consume(controller)
                exported = self.export_root / "optimized/kernel.py"
                if failure == "patch_missing":
                    self.patch_path.unlink()
                elif failure == "patch_changed":
                    self.patch_path.write_text(self.canonical_patch + "\n")
                elif failure == "patch_link":
                    self.patch_path.unlink()
                    self.patch_path.symlink_to(self.repo / "kernel.py")
                elif failure == "patch_hardlink":
                    os.link(self.patch_path, self.root / ("patch_alias_" + str(self.serial)))
                elif failure == "export_missing":
                    exported.unlink()
                elif failure == "export_changed":
                    # The patch still names certified A. The export contains B.
                    exported.write_text("def kernel(value):\n    return value * 2\n")
                elif failure == "export_link":
                    exported.unlink()
                    exported.symlink_to(self.repo / "kernel.py")
                elif failure == "export_hardlink":
                    os.link(exported, self.root / ("export_alias_" + str(self.serial)))
                elif failure == "export_extra":
                    (self.export_root / "extra.py").write_text("extra\n")
                elif failure == "export_mode":
                    exported.chmod(0o700)
                elif failure == "export_root_link":
                    shutil.rmtree(self.export_root)
                    self.export_root.symlink_to(self.repo, target_is_directory=True)
                else:
                    shutil.rmtree(exported.parent)
                    exported.parent.symlink_to(self.repo, target_is_directory=True)
                decision = json.loads(self.checkpoint(controller, self.request(stage="finalize"),
                                                      self.binding("final_agent"))["payload"])
                self.assertFalse(decision["qualifying"])
                self.assertIsNone(decision["selected_artifacts"])
                self.assertTrue(controller.failed)
                with self.assertRaises(StopRejected):
                    self.checkpoint(controller, self.request(stage="finalize"), self.binding("final_agent"))

    def test_final_request_requires_exact_actor_paths_and_host_mapping(self):
        for field in ("final_patch", "director_final_patch", "export_root"):
            with self.subTest(field=field):
                controller = self.controller()
                self.checkpoint(controller)
                self.consume(controller)
                decision = json.loads(self.checkpoint(controller, self.request(stage="finalize", **{field: "/actor/wrong"}),
                                                      self.binding("final_agent"))["payload"])
                self.assertFalse(decision["qualifying"])
                self.assertTrue(controller.failed)
        for mapping in ({"kernel.py": "missing.py"}, {"meta.json": "optimized/kernel.py"}):
            with self.subTest(mapping=mapping):
                controller = self.controller(selected_artifacts=self.selected_artifacts(files=mapping))
                before = len(self.evaluator.calls)
                boundary = json.loads(self.checkpoint(controller)["payload"])
                if "kernel.py" not in mapping:
                    self.assertFalse(boundary["certified"])
                    self.assertEqual(len(self.evaluator.calls), before)
                    continue
                self.consume(controller)
                decision = json.loads(self.checkpoint(controller, self.request(stage="finalize"),
                                                      self.binding("final_agent"))["payload"])
                self.assertFalse(decision["qualifying"])

    def test_late_patch_or_export_changes_reject_duplicate_and_native_return(self):
        for stage in ("duplicate", "native_return"):
            for target in ("patch", "export"):
                with self.subTest(stage=stage, target=target):
                    self.restore_artifacts()
                    controller = self.controller()
                    result = self.successful_result(controller)
                    controller.bind_native_closure(lambda _: fixture_census())
                    path = self.patch_path if target == "patch" else self.export_root / "optimized/kernel.py"
                    path.write_text("late mutation\n")
                    with self.assertRaises(StopRejected):
                        if stage == "duplicate":
                            self.checkpoint(controller, self.request(stage="finalize"), self.binding("final_agent"))
                        else:
                            controller.confirm_native_return(result)
                    self.assertTrue(json.loads((controller.state_dir / "state.json").read_text())["failed"])

    def test_artifact_configuration_rejects_unsafe_or_ambiguous_paths(self):
        cases = ({"baseline_commit": "HEAD"}, {"patch_path": self.export_root / "patch.diff"},
                 {"actor_patch_path": "relative.diff"}, {"actor_export_root": "/actor/../exports"},
                 {"files": {}}, {"files": {"../kernel.py": "kernel.py"}},
                 {"files": {"kernel.py": "/absolute.py"}},
                 {"files": {"kernel.py": "same.py", "meta.json": "same.py"}})
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(StopRejected):
                self.selected_artifacts(**changes)

    def test_duplicate_delivery_does_not_repeat_measurements(self):
        controller = self.controller()
        first = self.checkpoint(controller)
        second = self.checkpoint(controller)
        self.assertEqual(first, second)
        self.assertEqual(len(self.evaluator.calls), 3)
        first["payload"] = "caller mutation"
        self.assertNotEqual(self.checkpoint(controller)["payload"], first["payload"])
        self.consume(controller)
        final = self.checkpoint(controller, self.request(stage="finalize"), self.binding("final_agent"))
        self.assertEqual(final, self.checkpoint(controller, self.request(stage="finalize"),
                                                self.binding("final_agent")))
        self.assertEqual(len(self.evaluator.calls), 3)

    def test_inadmissible_native_search_starts_no_measurement(self):
        for fields in ({"dispatched": 6}, {"no_improve": 2}):
            with self.subTest(fields=fields):
                controller = self.controller()
                with self.assertRaises(StopRejected):
                    self.checkpoint(controller, self.request(**fields))
                self.assertFalse(controller.looks)
        controller = self.controller()
        self.now = 2000.0
        with self.assertRaises(StopRejected):
            self.checkpoint(controller)
        self.assertFalse(self.evaluator.calls)

    def test_untrusted_host_adapters_and_invalid_weights_are_rejected(self):
        for fields in ({"boundary": object()}, {"evaluator": object()}, {"budget": True},
                       {"selected_artifacts": object()},
                       {"actor_candidate_root": "relative"}, {"actor_candidate_root": "/actor/../candidate"},
                       {"deadline_epoch": math.inf}, {"max_no_improve": 0}, {"buckets": {}},
                       {"buckets": {"a": True}}, {"buckets": {"a": math.nan}},
                       {"trial_id": "short"}, {"state_dir": self.repo / "inside"}):
            with self.subTest(keys=list(fields)), self.assertRaises(StopRejected):
                self.controller(**fields)

    def test_invalid_request_does_not_consume_a_look(self):
        controller = self.controller()
        requests = ["not a checkpoint", PREFIX + "[]", PREFIX + "{}", self.request(extra=1),
                    self.request(protocol="changed"), self.request(stage="changed"),
                    self.request(look_index=True), self.request(round=-1), self.request(budget=5),
                    PREFIX + '{"bad":NaN}',
                    self.request(trial_id="different_trial_1234"), self.request(candidate_root=str(self.root)),
                    self.request()[:-1] + ',"round":1}']
        for request in requests:
            with self.subTest(request=request[:40]), self.assertRaises(StopRejected):
                self.checkpoint(controller, request)
        self.assertFalse(controller.looks)
        self.assertFalse(self.evaluator.calls)

    def test_unknown_native_or_os_census_never_certifies(self):
        for failure in ("native", "freeze", "certify"):
            with self.subTest(failure=failure):
                controller = self.controller()
                self.boundary.reject_stage = failure
                def census(failure=failure):
                    self.census()
                    if failure == "native":
                        raise StopRejected("fixture_pending_producer")
                decision = json.loads(self.checkpoint(controller, census=census)["payload"])
                self.assertFalse(decision["certified"])
                self.assertEqual(controller.looks["1"]["status"], "rejected")
                self.assertFalse(self.boundary.active)

    def test_invalid_process_consumes_look_without_replacement(self):
        controller = self.controller()
        def fail(index, snapshot):
            raise StopRejected("fixture_invalid_process")
        self.evaluator.before = fail
        first = self.checkpoint(controller)
        self.assertFalse(json.loads(first["payload"])["certified"])
        self.assertEqual(len(self.evaluator.calls), 1)
        self.assertEqual(self.checkpoint(controller), first)
        self.assertEqual(len(self.evaluator.calls), 1)
        self.evaluator.before = None
        self.assertTrue(json.loads(self.checkpoint(controller, self.request(look_index=2, round=2),
                                                   self.binding("checkpoint_agent_2"))["payload"])["certified"])
        self.assertEqual(len(self.evaluator.calls), 4)

    def test_three_rejected_looks_exhaust_the_budget(self):
        controller = self.controller()
        self.evaluator.scores = [1.01, 1.02, 1.03]
        for look in (1, 2, 3):
            decision = json.loads(self.checkpoint(controller, self.request(look_index=look, round=look),
                                                 self.binding("agent_" + str(look)))["payload"])
            self.assertFalse(decision["certified"])
        self.assertEqual(len(self.evaluator.calls), 9)
        with self.assertRaises(StopRejected):
            self.checkpoint(controller, self.request(look_index=4, round=4))
        self.assertEqual(len(self.evaluator.calls), 9)

    def test_new_look_requires_new_round_and_stable_native_scope(self):
        controller = self.controller()
        self.evaluator.scores = [1.01, 1.02, 1.03]
        self.checkpoint(controller)
        with self.assertRaises(StopRejected):
            self.checkpoint(controller, self.request(look_index=2, round=1))
        for field in ("session_id", "run_id", "root_tool", "root_task"):
            with self.subTest(field=field), self.assertRaises(StopRejected):
                self.checkpoint(controller, self.request(look_index=2, round=2),
                                self.binding("second_agent", **{field: "different"}))
        self.assertEqual(len(self.evaluator.calls), 3)

    def test_process_identity_bucket_schema_and_timing_are_checked(self):
        mutations = [lambda r: r.update(process_id="wrong"), lambda r: r.update(snapshot_sha256="0" * 64),
                     lambda r: r.update(order="wrong"), lambda r: r.update(extra=True),
                     lambda r: r["buckets"].pop(),
                     lambda r: r["buckets"][1].update(bucket_id="a"),
                     lambda r: r["buckets"][0].update(candidate_ms=True),
                     lambda r: r["buckets"][0].update(reference_ms=0),
                     lambda r: r["buckets"][0].update(candidate_ms=math.inf)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                controller = self.controller()
                self.evaluator.change_record = mutate
                decision = json.loads(self.checkpoint(controller)["payload"])
                self.assertFalse(decision["certified"])
                self.assertEqual(len(controller.looks["1"]["processes"]), 1)

    def test_trusted_call_weights_control_the_score(self):
        controller = self.controller(buckets={"a": 1, "b": 100})
        def records(result):
            index = len(self.evaluator.calls)
            result["buckets"] = [{"bucket_id": "a", "reference_ms": 100, "candidate_ms": 50},
                                 {"bucket_id": "b", "reference_ms": 1, "candidate_ms": 10 + index / 100}]
        self.evaluator.change_record = records
        decision = json.loads(self.checkpoint(controller)["payload"])
        self.assertFalse(decision["certified"])
        self.assertAlmostEqual(controller.looks["1"]["processes"][0]["score"], 200 / 1051)

    def test_changed_candidate_or_snapshot_rejects_the_batch(self):
        for target in ("candidate", "snapshot", "extra_snapshot"):
            with self.subTest(target=target):
                controller = self.controller()
                original = (self.repo / "kernel.py").read_bytes()
                def change(index, snapshot, target=target):
                    path = self.repo / "kernel.py" if target == "candidate" else snapshot / "kernel.py"
                    if target == "extra_snapshot":
                        path = snapshot / "untracked.py"
                    if path.exists():
                        path.chmod(0o600)
                    path.write_text("changed fixture\n")
                self.evaluator.before = change
                decision = json.loads(self.checkpoint(controller)["payload"])
                self.assertFalse(decision["certified"])
                (self.repo / "kernel.py").write_bytes(original)

    def test_deadline_expiry_during_measurement_never_certifies(self):
        controller = self.controller()
        self.evaluator.before = lambda index, snapshot: setattr(self, "now", 2001.0)
        decision = json.loads(self.checkpoint(controller)["payload"])
        self.assertFalse(decision["certified"])

    def test_late_duplicate_cannot_revive_a_stale_certificate(self):
        for changed in ("source", "clock", "census"):
            with self.subTest(changed=changed):
                controller = self.controller()
                self.checkpoint(controller)
                original = (self.repo / "kernel.py").read_bytes()
                census = self.census
                if changed == "source":
                    (self.repo / "kernel.py").write_text("changed after certificate\n")
                elif changed == "clock":
                    self.now = 2001.0
                else:
                    def census():
                        raise StopRejected("new_producer")
                with self.assertRaises(StopRejected):
                    self.checkpoint(controller, census=census)
                self.assertTrue(controller.failed)
                (self.repo / "kernel.py").write_bytes(original)
                self.now = 1000.0

    def test_persistence_failure_cannot_return_success_on_retry(self):
        controller = self.controller()
        persist = controller._persist
        def fail_once_envelope_exists():
            if controller.looks.get("1", {}).get("envelope"):
                controller.failed = True
                raise OSError("fixture persistence failure")
            persist()
        with patch.object(controller, "_persist", side_effect=fail_once_envelope_exists), self.assertRaises(OSError):
            self.checkpoint(controller)
        self.assertTrue(controller.failed)
        self.assertEqual(len(self.evaluator.calls), 3)
        with self.assertRaises(StopRejected):
            self.checkpoint(controller)
        self.assertEqual(len(self.evaluator.calls), 3)

    def test_failed_final_persistence_cannot_replay_qualification(self):
        controller = self.controller()
        self.checkpoint(controller)
        self.consume(controller)
        def fail():
            controller.failed = True
            raise OSError("fixture final persistence failure")
        with patch.object(controller, "_persist", side_effect=fail), self.assertRaises(OSError):
            self.checkpoint(controller, self.request(stage="finalize"), self.binding("final_agent"))
        with self.assertRaises(StopRejected):
            self.checkpoint(controller, self.request(stage="finalize"), self.binding("final_agent"))

    def test_lease_exit_failure_never_delivers_its_prepared_success(self):
        controller = self.controller()
        self.boundary.reject_exit = True
        with self.assertRaises(StopRejected):
            self.checkpoint(controller)
        self.assertTrue(controller.failed)
        self.boundary.reject_exit = False
        with self.assertRaises(StopRejected):
            self.checkpoint(controller)

    def test_source_change_before_finalization_is_nonqualifying(self):
        controller = self.controller()
        self.checkpoint(controller)
        self.consume(controller)
        (self.repo / "kernel.py").write_text("changed during finalization\n")
        value = json.loads(self.checkpoint(controller, self.request(stage="finalize"),
                                          self.binding("final_agent"))["payload"])
        self.assertFalse(value["qualifying"])
        self.assertTrue(controller.failed)

    def test_native_return_rejects_changed_evidence_and_bool_coercion(self):
        mutations = [lambda r: r.update(budget_used=True), lambda r: r.update(budget_total=True),
                     lambda r: r.update(final_patch="/actor/another.patch"),
                     lambda r: r.pop("final_patch"),
                     lambda r: r.update(stopped_by="budget"),
                     lambda r: r["quality_stop"]["final_check"].update(qualifying=1),
                     lambda r: r["quality_stop"]["certificate"]["request"].update(no_improve=False),
                     lambda r: r["quality_stop"]["certificate"]["consumption"].update(consumed=1),
                     lambda r: r["quality_stop"]["certificate"]["decision"].update(look_index=True)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                controller = self.controller()
                result = self.successful_result(controller)
                controller.bind_native_closure(lambda _: fixture_census())
                mutate(result)
                with self.assertRaises(StopRejected):
                    controller.confirm_native_return(result)
                self.assertTrue(controller.failed)

    def test_native_closure_failure_and_late_source_change_are_latched(self):
        for failure in ("closure", "source"):
            with self.subTest(failure=failure):
                controller = self.controller()
                result = self.successful_result(controller)
                original = (self.repo / "kernel.py").read_bytes()
                def closure(_, failure=failure):
                    if failure == "closure":
                        raise StopRejected("native_evidence_changed")
                    return fixture_census()
                controller.bind_native_closure(closure)
                if failure == "source":
                    (self.repo / "kernel.py").write_text("late source change\n")
                with self.assertRaises(StopRejected):
                    controller.confirm_native_return(result)
                self.assertTrue(json.loads((controller.state_dir / "state.json").read_text())["failed"])
                (self.repo / "kernel.py").write_bytes(original)

    def test_committed_tree_rejects_dirty_index_untracked_and_links(self):
        original = (self.repo / "kernel.py").read_bytes()
        baseline = source_manifest(self.repo)
        (self.repo / "kernel.py").write_text("changed\n")
        with self.assertRaises(StopRejected):
            source_manifest(self.repo)
        self.git("add", "kernel.py")
        with self.assertRaises(StopRejected):
            source_manifest(self.repo)
        self.git("reset", "-q", "HEAD", "--", "kernel.py")
        (self.repo / "kernel.py").write_bytes(original)
        (self.repo / "untracked.py").write_text("extra\n")
        with self.assertRaises(StopRejected):
            source_manifest(self.repo)
        (self.repo / "untracked.py").unlink()
        (self.repo / "kernel.py").unlink()
        (self.repo / "kernel.py").symlink_to(self.repo / "meta.json")
        with self.assertRaises(StopRejected):
            source_manifest(self.repo)
        (self.repo / "kernel.py").unlink()
        (self.repo / "kernel.py").write_bytes(original)
        self.assertEqual(source_manifest(self.repo), baseline)

    def test_source_inspection_does_not_execute_a_repository_filter(self):
        (self.repo / ".gitattributes").write_text("kernel.py filter=probe\n")
        self.git("add", ".gitattributes")
        self.git("commit", "-q", "-m", "fixture attribute")
        marker = self.root / "filter_was_executed"
        program = "from pathlib import Path; Path(" + repr(str(marker)) + ").write_text('unexpected')"
        command = shlex.join([sys.executable, "-c", program])
        self.git("config", "filter.probe.clean", command)
        original = (self.repo / "kernel.py").read_bytes()
        (self.repo / "kernel.py").write_bytes(original)
        try:
            source_manifest(self.repo)
        except StopRejected:
            pass
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
