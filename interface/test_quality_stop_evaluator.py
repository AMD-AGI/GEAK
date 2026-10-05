# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise process binding on CPUs and decode one immutable archived record.

The runner below supplies synthetic isolation attestations deliberately. Its
real subprocesses test lifecycle and record checks, not production isolation.
No test imports the archived scorer or executes candidate code on a GPU.
"""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from copy import deepcopy
from unittest.mock import patch

from interface.native_cost_controls.quality_stop_controller import StopRejected, canonical
from interface.native_cost_controls.quality_stop_evaluator import (
    CALLS, IMAGE_ID, IMAGE_TAG, LaunchPlan, ProtectedPairedRunner, FrozenPairedEvaluator,
    REQUIRED_TASK_FILES, decode_paired,
)

ARCHIVE = Path(__file__).parent / "testdata" / "quality_stop_paired_archive.json"
# R/evaluation/control/methods/control/repetition_1/paired.json, unchanged.
ARCHIVE_SHA256 = "ba05ac33421e9a4ae92286d279523e816d0dd35b5bb1be91354b66485e3d2dac"


def archive_request(value):
    return {"scorer_order": value["order"], "actor_task_dir": value["task_dir"],
            "actor_candidate_source": value["candidate_source"], "scorer_sha256": value["source_hashes"]["scorer"],
            "scorer_task_hashes": value["source_hashes"]["frozen_task"],
            "candidate_hashes": value["source_hashes"]["candidate"],
            "reference_sha256": value["reference_binding"]["sha256"],
            "actor_reference_source": value["reference_binding"]["source"], "call_weights": dict(CALLS)}


class ArchivedDecoderTests(unittest.TestCase):
    def setUp(self):
        self.value = json.loads(ARCHIVE.read_bytes())
        self.request = archive_request(self.value)

    def decode(self):
        return decode_paired(json.dumps(self.value).encode(), request=self.request)

    def test_archived_eight_bucket_record(self):
        self.assertEqual(hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), ARCHIVE_SHA256)
        buckets = decode_paired(ARCHIVE.read_bytes(), request=self.request)
        self.assertEqual(len(buckets), 8)
        self.assertEqual([row["bucket_id"] for row in buckets], list(CALLS))
        for row, expected in zip(buckets, self.value["score"]["per_case"]):
            self.assertEqual(row, {"bucket_id": expected["sig"], "reference_ms": expected["baseline_ms"],
                                   "candidate_ms": expected["optimized_ms"]})

    def test_foreign_weights_fail_even_with_consistent_recomputed_totals(self):
        rows = self.value["score"]["per_case"]
        rows[0]["calls"] = 1024
        rows[0]["weight"] = 1024 * rows[0]["baseline_ms"]
        reference = sum(row["calls"] * row["baseline_ms"] for row in rows)
        candidate = sum(row["calls"] * row["optimized_ms"] for row in rows)
        self.value["score"].update(reference_lifecycle_ms=reference, candidate_lifecycle_ms=candidate,
                                   weighted_speedup=reference / candidate)
        with self.assertRaisesRegex(StopRejected, "foreign_call_weight"):
            self.decode()

    def test_independent_record_corruptions_reject(self):
        original = deepcopy(self.value)
        changes = [
            lambda x: x.update(status="error"),
            lambda x: x.update(order="candidate-first"),
            lambda x: x.update(input_seed=True),
            lambda x: x["source_hashes"]["frozen_task"].pop("baseline_ref/gemm_kernels.py.orig"),
            lambda x: x["reference_binding"].update(sha256="0" * 64),
            lambda x: x["candidate_binding"].update(source="/foreign/source.py"),
            lambda x: x["correctness"].update(passed=False),
            lambda x: x["correctness"]["report"]["eager"][0].update(correct=False),
            lambda x: x["timing_policy"].update(repeats=51),
            lambda x: x["score"]["per_case"][0].update(calls=True),
            lambda x: x["score"]["per_case"][0].update(baseline_ms=0),
            lambda x: x["score"].update(weighted_speedup=9.0),
            lambda x: x["observations"].reverse(),
            lambda x: x["observations"][0]["receipt"].update(ms=999.0),
            lambda x: x["observations"][0]["receipt"].update(timer="host_clock"),
            lambda x: x.update(unprimed_observations=[{"sig": "foreign", "leg": "reference"}]),
            lambda x: x["timed_input_hashes"].pop("qkv_proj_decode_M1"),
            lambda x: x.update(finished_unix=x["started_unix"] - 1),
            lambda x: x.update(score=[]),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.value = deepcopy(original)
                change(self.value)
                with self.assertRaises(StopRejected):
                    self.decode()

    def test_duplicate_nonfinite_and_truncated_json_reject(self):
        for raw in (b'{"status":"pass","status":"pass"}', b'{"x":NaN}', b'{"x":1e999}', b'{'):
            with self.subTest(raw=raw), self.assertRaises(StopRejected):
                decode_paired(raw, request=self.request)

    def test_archive_cannot_be_reused_as_a_fresh_measurement(self):
        with self.assertRaisesRegex(StopRejected, "not_fresh"):
            decode_paired(ARCHIVE.read_bytes(), request=self.request, child_started_unix=time.time())

    def test_unprimed_measurements_retain_original_scoring_limit(self):
        self.value["observations"][0]["receipt"]["primed"] = False
        row = self.value["observations"][0]
        self.value["unprimed_observations"] = [{"sig": row["sig"], "leg": row["leg"]}]
        self.value["device_only_interpretation_supported"] = False
        self.assertEqual(len(self.decode()), 8)

    def test_frozen_correctness_contract_rejects_missing_or_changed_coverage(self):
        original = deepcopy(self.value)
        changes = [
            lambda x: x["correctness"]["report"].pop("random"),
            lambda x: x["correctness"].update(report={"eager": [{"case": "unspecified", "correct": True}]}),
            lambda x: x["correctness"].update(tolerance=1.0),
            lambda x: x["correctness"].update(random_draws=0),
            lambda x: x["correctness"].update(eager_cases=0),
            lambda x: x["correctness"].update(preserved_oracle_exclusions=[]),
            lambda x: x["correctness"]["report"]["random"][0].update(case=x["correctness"]["report"]["random"][1]["case"]),
            lambda x: x["correctness"].update(extra_gate=True),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.value = deepcopy(original)
                change(self.value)
                with self.assertRaisesRegex(StopRejected, "correctness_(contract|schema)_changed"):
                    self.decode()


CHILD = r'''
import json, sys, time
request=json.load(open(sys.argv[1]))
value=json.load(open(sys.argv[2]))
mode=sys.argv[3]
if mode=='timeout':
    time.sleep(30)
    raise SystemExit(0)
started=time.time()
time.sleep(.08)
ended=time.time()
value['source_hashes']={'scorer':request['scorer_sha256'],'frozen_task':request['scorer_task_hashes'],'candidate':request['candidate_hashes']}
value['candidate_source']=request['actor_candidate_source']
value['candidate_binding']['source']=request['actor_candidate_source']
value['task_dir']=request['actor_task_dir']
value['reference_binding'].update(source=request['actor_reference_source'],sha256=request['reference_sha256'],snapshot_source_sha256=request['reference_sha256'])
value['order']=request['scorer_order']
if mode!='stale':
    value['started_unix']=started
    value['finished_unix']=ended
legs=['reference','candidate'] if value['order']=='ref-first' else ['candidate','reference']
names=list(request['call_weights'])
by_name={(row['sig'],row['leg']):row for row in value['observations']}
value['observations']=[by_name[(name,leg)] for name in names for leg in legs]
for i,row in enumerate(value['observations']):
    row['started_unix']=started+(ended-started)*(i+1)/17
if mode=='foreign_weight':
    value['score']['per_case'][0]['calls']=999
if mode!='empty':
    print(json.dumps(value),flush=True)
raise SystemExit(3 if mode=='exit_failure' else 0)
'''


class SyntheticCPURunner(ProtectedPairedRunner):
    """Test real child lifecycle with explicitly synthetic isolation evidence."""

    def __init__(self, root, *, mode="valid"):
        self.root, self.mode = root, mode
        self.starts = 0
        self.process = None
        self.runtime_exit = None
        self.fail_cleanup = False
        self.corrupt_after_start = None
        self.root.joinpath("child.py").write_text(CHILD)
        self.wait_arguments = []

    def prepare(self, request, *, record_dir):
        data = self.root / (str(self.starts) + "-request.json")
        data.write_text(json.dumps(request))
        self.last_request = deepcopy(request)
        return LaunchPlan(argv=(sys.executable, "-I", "-B", str(self.root / "child.py"), str(data), str(ARCHIVE), self.mode),
                          environment=tuple(request["environment"].items()), cwd=str(self.root),
                          runtime_id="synthetic-cpu-" + request["process_id"],
                          details={"kind": "synthetic_cpu_fixture", "production_isolation": False})

    def start(self, request, plan, *, stdout, stderr):
        self.starts += 1
        self.process = subprocess.Popen(list(plan.argv), cwd=plan.cwd, env=dict(plan.environment),
                                        stdout=stdout, stderr=stderr, start_new_session=True)
        self.original_wait = self.process.wait
        if self.mode == "timeout":
            def short_wait(timeout=None):
                self.wait_arguments.append(timeout)
                return self.original_wait(timeout=min(timeout, .02) if timeout is not None else None)
            self.process.wait = short_wait
        if self.corrupt_after_start:
            self.corrupt_after_start(request)
        return self.process

    def attest(self, request, plan, process, *, stage):
        if stage == "stopped" and self.fail_cleanup:
            raise RuntimeError("Synthetic unknown container cleanup")
        value = {"stage": stage, "runtime_id": plan.runtime_id, "host_pid": process.pid if process else None,
            "request_sha256": hashlib.sha256(canonical(request).encode()).hexdigest(),
            "source_protection": {"snapshot": "read_only", "task": "read_only", "scorer": "read_only",
                                  "recorder": "unmounted", "host_controller": "unmounted"},
            "os_evidence": {"fixture": "synthetic_cpu_only", "production_isolation": False,
                            "actual_pid": process.pid if process else None}}
        if stage == "stopped":
            if process is not None and process.poll() is None:
                raise RuntimeError("The actual CPU child still runs")
            value.update(runtime_stopped=True, active_descendants=[],
                         runtime_exit_code=process.returncode if self.runtime_exit is None else self.runtime_exit)
        return value

    def stop(self, request, plan, process):
        if self.fail_cleanup:
            raise RuntimeError("Synthetic cleanup failure")
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            self.original_wait(timeout=5)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.original_wait(timeout=5)


class EvaluatorProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task, self.snapshot = self.root / "task", self.root / "snapshot"
        self.task.mkdir()
        self.snapshot.mkdir()
        for name in REQUIRED_TASK_FILES:
            path = self.task / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic fixed task " + name)
        self.hashes = {str(path.relative_to(self.task)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in self.task.rglob("*") if path.is_file()}
        self.snapshot.joinpath("kernel_src").mkdir()
        self.snapshot.joinpath("kernel_src/flydsl_hgemm_impl.py").write_text("# Synthetic candidate bytes.\n")
        self.scorer = self.root / "synthetic-scorer.py"
        self.scorer.write_text("# The CPU runner does not execute the GPU scorer.\n")
        self.patcher = patch("interface.native_cost_controls.quality_stop_evaluator.SCORER_SHA256",
                             hashlib.sha256(self.scorer.read_bytes()).hexdigest())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.environment = {"HOME": os.environ["HOME"], "GEAK_FREEZE_BASELINE": "0", "HARNESS_CACHE_FLUSH_MB": "512",
                            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}

    def evaluator(self, *, mode="valid", calls=None):
        runner = SyntheticCPURunner(self.root, mode=mode)
        self.addCleanup(runner.close)
        evaluator = FrozenPairedEvaluator(state_dir=self.root / "receipts", scorer_path=self.scorer,
            task_dir=self.task, task_hashes=self.hashes, call_weights=CALLS if calls is None else calls,
            runner=runner, environment=self.environment)
        return evaluator, runner

    def run_one(self, evaluator, *, identity="trial:look:1:process:1", order="reference_first"):
        return evaluator.run_process(snapshot=self.snapshot, snapshot_sha256="1" * 64,
                                     process_id=identity, order=order)

    def receipt(self, identity="trial:look:1:process:1"):
        path = self.root / "receipts" / hashlib.sha256(identity.encode()).hexdigest() / "receipt.json"
        return json.loads(path.read_text())

    def test_real_fresh_child_binds_all_receipts_and_exact_controller_abi(self):
        evaluator, runner = self.evaluator()
        result = self.run_one(evaluator)
        self.assertEqual(set(result), {"process_id", "snapshot_sha256", "order", "buckets"})
        self.assertEqual(len(result["buckets"]), 8)
        record = self.receipt()
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["actual_host_exit_code"], 0)
        self.assertEqual(record["stopped_evidence"]["runtime_exit_code"], 0)
        self.assertEqual(record["cleanup"], "verified_stopped")
        self.assertEqual(record["timeout_seconds"], 600)
        self.assertEqual(record["host_pid"], runner.process.pid)
        self.assertIn("environment", record["launch"])
        self.assertIn("raw_json_sha256", record)
        self.assertEqual(runner.last_request["scorer_order"], "ref-first")

    def test_order_comes_from_each_controller_call(self):
        evaluator, runner = self.evaluator()
        for index, order in enumerate(("candidate_first", "candidate_first", "reference_first")):
            result = self.run_one(evaluator, identity=f"process:{index}", order=order)
            self.assertEqual(result["order"], order)
        self.assertEqual(runner.starts, 3)

    def test_duplicate_process_never_starts_another_child(self):
        evaluator, runner = self.evaluator()
        self.run_one(evaluator)
        with self.assertRaisesRegex(StopRejected, "already_attempted"):
            self.run_one(evaluator)
        self.assertEqual(runner.starts, 1)

    def test_foreign_call_map_rejects_before_a_child_starts(self):
        calls = dict(CALLS, qkv_proj_decode_M1=1024)
        with self.assertRaisesRegex(StopRejected, "foreign_call_weights"):
            self.evaluator(calls=calls)

    def test_real_nonzero_exit_overrides_passing_json(self):
        evaluator, _ = self.evaluator(mode="exit_failure")
        with self.assertRaisesRegex(StopRejected, "process_exit_failed"):
            self.run_one(evaluator)
        self.assertEqual(self.receipt()["actual_host_exit_code"], 3)
        self.assertEqual(self.receipt()["status"], "invalid")

    def test_runtime_exit_must_also_pass(self):
        evaluator, runner = self.evaluator()
        runner.runtime_exit = 137
        with self.assertRaisesRegex(StopRejected, "process_exit_failed"):
            self.run_one(evaluator)
        self.assertEqual(self.receipt()["actual_host_exit_code"], 0)

    def test_timeout_stops_actual_child_and_never_retries(self):
        evaluator, runner = self.evaluator(mode="timeout")
        with self.assertRaisesRegex(StopRejected, "process_timeout"):
            self.run_one(evaluator)
        self.assertIsNotNone(runner.process.poll())
        self.assertEqual(runner.starts, 1)
        self.assertGreater(runner.wait_arguments[0], 590)
        self.assertEqual(self.receipt()["status"], "unknown")
        self.assertEqual(self.receipt()["cleanup"], "verified_stopped")

    def test_unknown_cleanup_blocks_every_later_process(self):
        evaluator, runner = self.evaluator(mode="timeout")
        runner.fail_cleanup = True
        with self.assertRaises(StopRejected):
            self.run_one(evaluator)
        self.assertEqual(self.receipt()["cleanup"], "unknown")
        with self.assertRaisesRegex(StopRejected, "cleanup_unknown_blocks_admission"):
            self.run_one(evaluator, identity="different-process")
        self.assertEqual(runner.starts, 1)

    def test_changed_snapshot_rejects_even_when_child_json_matches_old_request(self):
        evaluator, runner = self.evaluator()
        runner.corrupt_after_start = lambda request: (Path(request["snapshot"]) / "kernel_src/flydsl_hgemm_impl.py").write_text("changed")
        with self.assertRaisesRegex(StopRejected, "sources_changed"):
            self.run_one(evaluator)
        self.assertEqual(self.receipt()["status"], "invalid")

    def test_stale_json_is_not_a_fresh_process_measurement(self):
        evaluator, _ = self.evaluator(mode="stale")
        with self.assertRaisesRegex(StopRejected, "not_fresh"):
            self.run_one(evaluator)

    def test_empty_successful_child_output_is_invalid(self):
        evaluator, runner = self.evaluator(mode="empty")
        with self.assertRaisesRegex(StopRejected, "result_size_invalid"):
            self.run_one(evaluator)
        self.assertEqual(self.receipt()["actual_host_exit_code"], 0)
        self.assertEqual(self.receipt()["status"], "invalid")
        with self.assertRaisesRegex(StopRejected, "already_attempted"):
            self.run_one(evaluator)
        self.assertEqual(runner.starts, 1)

    def test_snapshot_link_rejects_before_process_start(self):
        evaluator, runner = self.evaluator()
        source = self.snapshot / "kernel_src/flydsl_hgemm_impl.py"
        source.unlink()
        source.symlink_to(self.scorer)
        with self.assertRaisesRegex(StopRejected, "source_not_regular"):
            self.run_one(evaluator)
        self.assertEqual(runner.starts, 0)

    def test_scorer_or_task_change_rejects_before_process_start(self):
        evaluator, runner = self.evaluator()
        self.scorer.write_text("changed scorer")
        with self.assertRaisesRegex(StopRejected, "sources_changed"):
            self.run_one(evaluator)
        self.assertEqual(runner.starts, 0)

    def test_missing_runtime_evidence_cannot_authorize_a_result(self):
        evaluator, runner = self.evaluator()
        original = runner.attest
        def invalid_evidence(request, plan, process, *, stage):
            value = original(request, plan, process, stage=stage)
            if stage == "running":
                value["os_evidence"] = {}
            return value
        runner.attest = invalid_evidence
        with self.assertRaisesRegex(StopRejected, "runtime_evidence_missing"):
            self.run_one(evaluator)
        self.assertIsNotNone(runner.process.poll())
        self.assertEqual(self.receipt()["cleanup"], "verified_stopped")

    def test_unqualified_runner_is_not_a_default(self):
        with self.assertRaisesRegex(StopRejected, "trusted_paired_runner_required"):
            FrozenPairedEvaluator(state_dir=self.root / "receipts", scorer_path=self.scorer,
                task_dir=self.task, task_hashes=self.hashes, call_weights=CALLS, runner=object(),
                environment=self.environment)

    def test_prepare_exception_with_an_idle_child_blocks_later_admission(self):
        evaluator, runner = self.evaluator()
        def partial_prepare(request, *, record_dir):
            runner.process = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            runner.original_wait = runner.process.wait
            raise RuntimeError("Synthetic prepare interrupted after creating an idle runtime")
        runner.prepare = partial_prepare
        with self.assertRaises(RuntimeError):
            self.run_one(evaluator)
        self.assertTrue(evaluator.blocked)
        self.assertEqual(self.receipt()["cleanup"], "unknown")
        self.assertEqual(self.receipt()["status"], "unknown")
        self.assertEqual(self.receipt()["reason"], "paired_prepare_outcome_unknown")
        with self.assertRaisesRegex(StopRejected, "cleanup_unknown_blocks_admission"):
            self.run_one(evaluator, identity="second-process")


if __name__ == "__main__":
    unittest.main()
