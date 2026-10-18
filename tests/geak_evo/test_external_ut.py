"""External UT evidence and existing Arena/result composition; CPU only."""

import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import unittest
from unittest import mock

from geak_evo import result
from geak_evo.archive.store import ResearchStore
from geak_evo.evaluate.evaluator import ArenaEvaluator, canonical, sha256
from geak_evo.evaluate.external import metadata_view, project_external_task, read_external_report, validate_objective
from geak_evo.evaluate import external_runner
from geak_evo.evaluate.jobs import job_digest, validate_arena_job


def fixture(root):
    cases = [{"name": name, "regime": regime, "calls": calls, "stat": "min", "legs": 5,
              "warmup": 5, "repeats": 20, "inner": 1, "graph": False, "timed": True}
             for name, regime, calls in (("a", "extend", 100), ("b", "extend", 100), ("c", "draft", 1))]
    meta = {"cases": cases, "candidate_module_file": "kernel_src/module.py", "candidate_attr": "run",
            "baseline_module_file": "baseline_src/module.py", "baseline_attr": "run",
            "tol": None, "rms_tol": None, "seed": 0, "random_draws": 3}
    bodies = {name: b"# synthetic source; never imported\n" for name in
              ("kernel_src/module.py", "baseline_src/module.py", "ut_main.py", "weight_lib.py", "harness_lib.py")}
    bodies["meta.json"] = canonical(meta)
    for name, body in bodies.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    contract = project_external_task(meta, artifacts={name: sha256(body) for name, body in bodies.items()},
        original_source_map={"kernel_src/module.py": "baseline_src/module.py"}, primary_regime="extend",
        source={"repository": "external:fixture", "revision": "a"*64, "task_selector": "fixture"},
        runtime={"image": "example/image@sha256:"+"b"*64, "config_digest": "sha256:"+"c"*64})
    return validate_arena_job(contract, root)


def raw_reports(request, *, original=False, draft_regression=False):
    source = {row["path"]: row["blob_id"] for row in request["source"]["files"]}
    artifacts = {"START.json": canonical({"parent_evaluation": {"operation_id": request["operation_id"],
        "request_sha256": sha256(canonical(request)), "binding": request["binding"]},
        "source_files": {**request["contract"]["artifacts"], **source}, "declared_processes": 3,
        "gain_required_exit": False})}
    history = []
    # The median complete fold differs from a fold of separately reduced cases.
    series = [((100., 50.), (1., 1.)), ((1., 1.), (100., 20.)), ((100., 20.), (100., 50.))]
    for i, pairs in enumerate(series, 1):
        rows = []
        for c, (base, candidate) in zip(request["contract"]["metadata"]["value"]["cases"],
                                       [*pairs, (1., 2. if draft_regression else .5)]):
            if original:
                candidate = base
            rows.append({"name": c["name"], "regime": c["regime"], "calls": c["calls"],
                "graph_requested": False, "timer": "cuda_event", "measurement_valid": True,
                "measurement_errors": [], "legs_kept": 4,
                "arm_timers": {role: ["cuda_event"]*5 for role in ("baseline", "candidate", "control")},
                "baseline_legs": [base]*4, "optimized_legs": [candidate]*4, "control_legs": [base]*4,
                "baseline_ms": base, "optimized_ms": candidate, "control_ms": base,
                "speedup": base/candidate, "control_ratio": 1., "mde_frac": 0.})
        folds = {}
        for regime in ("draft", "extend"):
            selected = [row for row in rows if row["regime"] == regime]
            folds[regime] = sum(row["calls"]*row["baseline_ms"] for row in selected)/sum(row["calls"]*row["optimized_ms"] for row in selected)
        fp = sha256(canonical(source))
        runtime = {"packages": {"torch": "synthetic"}, "env": {}}
        runtime_id = sha256(canonical(runtime))
        history.append({"schema_version": 2, "process_id": f"synthetic:{i}:100", "fp": fp,
            "runtime_id": runtime_id, "measurement_valid": True, "correctness_pass": True,
            "weighted": folds["extend"], "weighted_mde": 0., "per_case": rows,
            "complete_regimes": ["draft", "extend"], "folds": folds,
            "buckets": {row["name"]: {"sp": row["speedup"], "mde": 0.} for row in rows}})
        gain = i >= 2 and not original and not draft_regression
        regression = i >= 2 and draft_regression and not original
        report = {"schema_version": 2, "candidate_fingerprint": fp, "runtime_fingerprint": runtime_id,
            "runtime": runtime, "correctness_pass": True, "measurement_valid": True,
            "performance_scope_complete": True, "measurement_errors": [], "missing_cases": [],
            "not_deployment_timed": [], "e2e_verified": False, "expected_case_count": 3,
            "measured_case_count": 3, "per_case": rows, "complete_regimes": ["draft", "extend"],
            "folds": {key: {"weighted": value} for key, value in folds.items()}, "primary_regime": "extend",
            "call_weighted": folds["extend"], "admissible": gain, "reproducible_gain": gain,
            "reproducible_regression": regression,
            "repro": {"runs": i, "enough": i >= 2, "reproducible_gain": gain, "reproducible_regression": regression}}
        prefix = f"process_{i:02d}/"
        artifacts[prefix+"ut_result.json"] = canonical(report)
        artifacts[prefix+"_runs_v2.jsonl"] = b"\n".join(canonical(row) for row in history)+b"\n"
        artifacts[prefix+"RECEIPT.json"] = canonical({"index": i, "returncode": 0, "sources_unchanged": True,
            **{name+"_sha256": sha256(artifacts[prefix+name]) for name in ("ut_result.json", "_runs_v2.jsonl")}})
    return artifacts


class ExternalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.contract = fixture(self.root/"task")
        job = job_digest(self.contract)
        self.store = ResearchStore(self.root/"store", allowed_epochs=[(job, "epoch")], source_paths={job: self.contract["source_paths"]})
        self.original_source = self.store.capture_source(self.root/"task", self.contract["source_paths"])
        self.original = self.store.candidate(job, "epoch", self.original_source)
        (self.root/"task/kernel_src/module.py").write_bytes(b"# synthetic changed candidate\n")
        self.selected_source = self.store.capture_source(self.root/"task", self.contract["source_paths"])
        self.selected = self.store.candidate(job, "epoch", self.selected_source)
        (self.root/"task/kernel_src/module.py").write_bytes(b"# synthetic source; never imported\n")
        self.regression = False
        self.evaluator = ArenaEvaluator(self.store, contract=self.contract, contract_id=job,
            artifact_root=self.root/"task", output_root=self.root/"evaluations", measurement_epoch_id="epoch",
            launch=lambda request, directory: "synthetic", settle=self.settle)

    def settle(self, handle, request, directory):
        raw = raw_reports(request, original=request["binding"]["candidate_id"] == self.original,
                          draft_regression=self.regression)
        report = read_external_report(request, raw)
        artifacts = {"observer/"+name: body for name, body in raw.items()}
        artifacts.update({"host/cleanup.json": b"{}", "host/runtime.json": b"{}"})
        groups = copy.deepcopy(report["observations"])
        for group in groups.values():
            group["evidence"] = ["observer/"+name for name in group["evidence"]]
        return {"artifacts": artifacts, "receipt": {"schema": "geak-arena-settlement-v1",
            "operation_id": request["operation_id"], "request_sha256": sha256(canonical(request)),
            "binding": request["binding"], "execution_started": True,
            "cleanup": {"settled": True, "evidence": ["host/cleanup.json"]},
            "runtime": {**self.contract["runtime"], "physical_devices": ["synthetic-device"], "evidence": ["host/runtime.json"]},
            "phase_exit_codes": {"evaluate": 0}, "observations": groups,
            "artifacts_sha256": {name: sha256(body) for name, body in artifacts.items()},
            "costs": {"wall_seconds": 1., "evaluations": 1}, "errors": []}}

    def export(self):
        self.evaluator.evaluate(self.original, evaluation_id="baseline")
        self.evaluator.evaluate(self.selected, evaluation_id="selected")
        choice = {"candidate_id": self.selected, "source_id": self.selected_source,
                  "binding": {"campaign_id": "campaign", "job_id": self.evaluator.contract_id, "measurement_epoch_id": "epoch"}}
        with mock.patch.object(result, "validate_final_recommendation", return_value={"status": "selected", "choice": choice, "record": {"disposition": "set"}}):
            return result.export_final_result(self.store, {"campaign_id": "campaign", "revision": 1, "state": {"phase": "blocked"}},
                evaluator=self.evaluator, selection_options={}, baseline_candidate_id=self.original,
                baseline_evaluation_id="baseline", selected_evaluation_id="selected", output_dir=self.root/"out",
                objective={"kind": "external_primary", "primary_regime": "extend"})

    def test_existing_arena_and_export_keep_native_phase_reduction(self):
        value = self.export()
        self.assertEqual(value["outcome"], "improved", value)
        self.assertEqual(value["final_speedup"], statistics.median([101/51, 101/21, 200/70]))
        self.assertIsNone(value["final_geomean"])
        self.assertIsNone(value["final_arithmetic"])
        self.assertEqual(set(value["phase_speedups"]), {"extend", "draft"})
        self.assertTrue(Path(value["final_patch"]).is_file())
        self.assertTrue(all(value[key] is False for key in ("measurement_valid", "custody_proven", "reward_eligible", "framework_final_pass")))

    def test_secondary_regression_keeps_native_gain_veto(self):
        self.regression = True
        value = self.export()
        self.assertEqual(value["outcome"], "no_improvement", value)
        self.assertFalse(value["native_reproducible_gain"])
        self.assertEqual(value["final_patch"], "")

    def test_incomplete_or_wrong_operation_cannot_be_a_result(self):
        request = self.evaluator._request(self.selected, "test")
        for change in (lambda a: a.pop("process_03/ut_result.json"),
                       lambda a: a.update({"START.json": canonical({"parent_evaluation": {}})})):
            artifacts = raw_reports(request)
            change(artifacts)
            with self.assertRaises((ValueError, KeyError)):
                read_external_report(request, artifacts)
        artifacts = raw_reports(request)
        receipt = json.loads(artifacts["process_03/RECEIPT.json"])
        receipt["artifact_errors"] = ["stdout.log: unstable or unsafe file"]
        artifacts["process_03/RECEIPT.json"] = canonical(receipt)
        with self.assertRaises(ValueError):
            read_external_report(request, artifacts)

    def test_rejects_a_blended_objective(self):
        with self.assertRaises(ValueError):
            validate_objective(self.contract, {"kind": "geomean"})

    def test_runner_forwards_only_recorded_runtime_values(self):
        request = self.evaluator._request(self.original, "runner")
        raw = raw_reports(request)
        forwarded = []
        def process(argv, **options):
            forwarded.append(dict(options["env"]))
            prefix = f"process_{len(forwarded):02d}/"
            for name in ("ut_result.json", "_runs_v2.jsonl"):
                (options["cwd"]/name).write_bytes(raw[prefix+name])
            options["stdout"].write(b"synthetic stdout\n")
            options["stderr"].write(b"synthetic stderr\n")
            return subprocess.CompletedProcess(argv, 0)
        allowed = {"PATH": "/opt/venv/bin:/usr/bin", "LD_LIBRARY_PATH": "/opt/rocm/lib",
                   "HIP_VISIBLE_DEVICES": "0", "GEAK_PHYSICAL_DEVICE_MAP": '[{"local_ordinal":0}]',
                   "GEAK_GPU_SYSFS_BDFS": "0000:01:00.0", "AITER_ROOT_DIR": "/results/cache/aiter_root",
                   "CXX": ""}
        inherited = {**allowed, "OPENAI_API_KEY": "private-test-value", "HF_TOKEN": "private-test-value",
                     "HTTPS_PROXY": "private-test-value", "HOME": "private-test-value",
                     "UNRELATED_WORKFLOW_FIELD": "private-test-value", "GLM_UT_REQUIRE_REPRODUCIBLE_GAIN": "1"}
        output = self.root/"runner"
        with mock.patch.dict(os.environ, inherited, clear=True), mock.patch.object(external_runner.subprocess, "run", side_effect=process):
            self.assertEqual(external_runner.run(request, self.root/"task", output), 0)
        expected = {**allowed, "GLM_UT_REQUIRE_REPRODUCIBLE_GAIN": "0"}
        self.assertEqual(forwarded, [expected]*3)
        start = json.loads((output/"START.json").read_bytes())
        self.assertEqual(start["environment"], expected)
        self.assertNotIn("CC", start["environment"])
        self.assertNotIn(b"private-test-value", (output/"START.json").read_bytes())
        for index in range(1, 4):
            leg = output/f"process_{index:02d}"
            receipt = json.loads((leg/"RECEIPT.json").read_bytes())
            for name in ("ut_result.json", "_runs_v2.jsonl", "stdout.log", "stderr.log"):
                self.assertEqual(receipt[name+"_sha256"], sha256((leg/name).read_bytes()))

    def test_runner_rejects_symlinked_ut_reports_without_copying_target(self):
        target = self.root/"unrelated-private-file"
        target.write_bytes(b"private target must not be retained")
        for index, unsafe in enumerate(("ut_result.json", "_runs_v2.jsonl")):
            with self.subTest(unsafe=unsafe):
                request = self.evaluator._request(self.original, f"unsafe-{index}")
                raw = raw_reports(request)
                def process(argv, **options):
                    for name in ("ut_result.json", "_runs_v2.jsonl"):
                        path = options["cwd"]/name
                        if name == unsafe:
                            path.symlink_to(target)
                        else:
                            path.write_bytes(raw["process_01/"+name])
                    return subprocess.CompletedProcess(argv, 0)
                output = self.root/f"unsafe-{index}"
                with mock.patch.object(external_runner.subprocess, "run", side_effect=process) as run:
                    self.assertEqual(external_runner.run(request, self.root/"task", output), 1)
                self.assertEqual(run.call_count, 1)
                self.assertFalse((output/"process_01"/unsafe).exists())
                receipt = json.loads((output/"process_01/RECEIPT.json").read_bytes())
                self.assertTrue(any(error.startswith(unsafe+": ") for error in receipt["artifact_errors"]))
                self.assertNotIn(b"private target must not be retained", (output/"process_01/RECEIPT.json").read_bytes())

    def test_absent_tolerance_and_untimed_case_are_preserved(self):
        meta = json.loads((self.root/"task/meta.json").read_text())
        del meta["rms_tol"]
        meta["cases"].append({"name": "edge", "timed": False})
        view = metadata_view(meta)
        self.assertNotIn("rms_tol", view)
        self.assertEqual(view["cases"][-1], {"name": "edge", "timed": False})
        contract = project_external_task(meta, artifacts=self.contract["artifacts"],
            original_source_map=self.contract["original_source_map"], source=self.contract["source"],
            runtime=self.contract["runtime"], primary_regime="extend")
        self.assertIn("edge", contract["correctness_case_ids"])
        self.assertNotIn("edge", contract["benchmark_case_ids"])


if __name__ == "__main__":
    unittest.main()
