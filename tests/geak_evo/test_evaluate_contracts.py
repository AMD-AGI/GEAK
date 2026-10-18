"""CPU callback fixtures shared by the original and ported evaluator layouts.

Set GEAK_EVO_EVALUATION_LAYOUT=original with the pinned source on PYTHONPATH
for parity. These callbacks manufacture test evidence and never run a workload.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

if os.environ.get("GEAK_EVO_EVALUATION_LAYOUT") == "original":
    from geak_evo import arena_job as jobs, arena_evaluator as arena
    from geak_evo import native_evaluator as native
    from geak_evo import jobs as jobs_v1, native_arena_contract as native_contract
    from geak_evo import artifact_transport as artifacts, measurements as timing
    from geak_evo.research_state import ResearchStore
    import native_quantities as quantities
else:
    from geak_evo.evaluate import jobs, evaluator as arena, native_evaluator as native
    from geak_evo.evaluate import quantities, jobs_v1, native_contract, artifacts, timing
    from geak_evo.archive.store import ResearchStore


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture():
    cases = ["case-a", "case-b"]
    meta = {"cases": cases, "tol": 0.001, "candidate_bind": "kernel.py"}
    files = {"kernel.py": b"# synthetic source; never imported\n", "metadata.json": encoded(meta),
             "timer.py": b"# protected synthetic timing declaration\n"}
    contract = {
        "schema": "geak-arena-native-operator-v2",
        "source": {"repository": "fixture://cpu", "revision": "1" * 40,
                   "task_selector": "fixture", "snapshot_manifest_sha256": "2" * 64},
        "runtime": {"image": "fixture@sha256:" + "3" * 64, "config_digest": "sha256:" + "4" * 64},
        "runtime_requirements": {"candidate_bind": "kernel.py"},
        "provenance": {"kind": "arena_operator_replay", "values": "generated_from_pinned_recipe",
                       "capture_revision": None, "model_revision": None, "framework_revision": None,
                       "observed_frequencies": None, "evidence_ref": "metadata.json"},
        "artifacts": {name: sha(raw) for name, raw in files.items()}, "aliases": {},
        "source_paths": ["kernel.py"], "fixtures": {}, "correctness_case_ids": cases,
        "benchmark_case_ids": cases, "benchmark_report_ids": {case: case for case in cases},
        "case_bindings": {case: {"artifact_ref": "metadata.json", "json_pointer": "/cases/" + str(i),
                                  "sha256": sha(encoded(case))} for i, case in enumerate(cases)},
        "phases": {phase: {"argv": ["synthetic-" + phase], "timeout_seconds": 1}
                   for phase in ("compile", "correctness", "benchmark")},
        "numerical_policy": {"tol": 0.001},
        "timing_policy": {
            "primary": {"kind": "cuda_event_direct_calls", "warmup_calls": 10, "measured_calls": 100,
                        "statistic": "mean", "unit": "ms", "graph": False},
            "fallback": {"kind": "native_ut_fallback", "comparable": False, "actual_timer": None},
            "source_refs": ["timer.py"]},
        "report_contract": {"format": "synthetic-fixture"},
        "measurement_valid": False, "custody_proven": False, "reward_eligible": False,
    }
    return contract, files


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.contract, files = fixture()
        self.task = self.root / "task"
        self.task.mkdir()
        for name, raw in files.items():
            (self.task / name).write_bytes(raw)
        self.job_id = jobs.job_digest(self.contract)
        self.store = ResearchStore(self.root / "store", allowed_epochs=[(self.job_id, "epoch")],
                                   source_paths={self.job_id: self.contract["source_paths"]})
        self.source = self.store.capture_source(self.task, self.contract["source_paths"])
        self.candidate = self.store.candidate(self.job_id, "epoch", self.source)
        self.launches = []
        self.handles = []
        self.mutate = lambda receipt: None
        self.engine = arena.ArenaEvaluator(self.store, contract=self.contract, contract_id=self.job_id,
            artifact_root=self.task, output_root=self.root / "evaluations", measurement_epoch_id="epoch",
            launch=self.launch, settle=self.settle)

    def launch(self, request, operation):
        self.assertTrue((operation / "request.json").is_file())
        self.assertTrue((operation / "dispatch_intent.json").is_file())
        self.launches.append(copy.deepcopy(request))
        return "synthetic-handle"

    def settle(self, handle, request, operation):
        self.handles.append(handle)
        bdf, physical = "0000:01:00.0", "0000:01:00.0@1"
        mapping = [{"local_ordinal": 0, "bdf": bdf, "physical_id": physical}]
        raw = {"host/cleanup.json": b"{}", "native/correctness.json": b"{}", "native/benchmark.json": b"{}",
               "host/plan.json": encoded({"allocation_mode": "disjoint-physical-v1", "physical_ids": [physical], "bdfs": [bdf]}),
               "host/owner-report.json": encoded({"smoke": {"physical_devices": [bdf], "device_mapping": mapping}})}
        receipt = {"schema": "geak-arena-settlement-v1", "operation_id": request["operation_id"],
            "request_sha256": sha(arena.canonical(request)), "binding": copy.deepcopy(request["binding"]),
            "execution_started": True, "cleanup": {"settled": True, "evidence": ["host/cleanup.json"]},
            "runtime": {**self.contract["runtime"], "physical_devices": [bdf], "evidence": ["host/plan.json"]},
            "phase_exit_codes": {phase: 0 for phase in request["phases"]},
            "observations": {
                "correctness": {"status": "passed", "case_ids": list(request["correctness_case_ids"]),
                                "evidence": ["native/correctness.json"]},
                "benchmark": {"status": "passed", "times_ms": {case: 1.25 for case in request["benchmark_report_ids"].values()},
                              "evidence": ["native/benchmark.json"]}},
            "artifacts_sha256": {name: sha(data) for name, data in raw.items()},
            "costs": {"wall_seconds": None, "evaluations": 1}, "errors": []}
        self.mutate(receipt)
        return {"receipt": receipt, "artifacts": raw}

    def test_loader_exact_identity_and_unknown_schema_never_falls_back(self):
        path = self.root / "job.json"
        path.write_bytes(encoded(self.contract))
        self.assertEqual(jobs.load_engineer_job(path, self.task, self.job_id), self.contract)
        path.write_bytes(encoded({**self.contract, "schema": "unknown", "schema_version": 1}))
        with self.assertRaisesRegex(ValueError, "unsupported explicit task schema"):
            jobs.load_engineer_job(path)

    def test_complete_observation_keeps_false_flags_and_unknown_cost(self):
        result = self.engine.evaluate(self.candidate, evaluation_id="complete")
        receipt = result["receipt"]
        self.assertTrue(receipt["settled"] and receipt["native_suite_pass_observed"])
        self.assertEqual(receipt["per_case_times_ms"], {"case-a": 1.25, "case-b": 1.25})
        self.assertTrue(all(receipt[key] is False for key in ("measurement_valid", "custody_proven", "reward_eligible")))
        self.assertIsNone(result["costs"]["wall_seconds"])
        self.assertEqual(self.launches[0]["binding"]["candidate_source_id"], self.source)

    def test_partial_case_pass_is_retained_failure(self):
        self.mutate = lambda receipt: receipt["observations"]["benchmark"]["times_ms"].pop("case-b")
        result = self.engine.evaluate(self.candidate, evaluation_id="partial")
        self.assertTrue(result["receipt"]["settled"])
        self.assertFalse(result["receipt"]["native_suite_pass_observed"])
        self.assertTrue(result["receipt"]["collection_errors"])
        self.assertEqual(result["receipt"]["per_case_times_ms"], {})

    def test_wrong_source_settlement_is_unsettled_with_raw_evidence(self):
        self.mutate = lambda receipt: receipt["binding"].update(candidate_source_id="f" * 64)
        with self.assertRaises(arena.ArenaEvaluationUnsettled) as caught:
            self.engine.evaluate(self.candidate, evaluation_id="wrong-source")
        result = caught.exception.result
        self.assertFalse(result["receipt"]["settled"])
        self.assertIn("raw/native/benchmark.json", result["artifacts"])
        self.assertIsNone(result["costs"]["evaluations"])

    def test_recovery_reconciles_same_operation_without_launch(self):
        self.mutate = lambda receipt: receipt["cleanup"].update(settled=False)
        with self.assertRaises(arena.ArenaEvaluationUnsettled):
            self.engine.evaluate(self.candidate, evaluation_id="recover")
        self.mutate = lambda receipt: None
        result = self.engine.evaluate(self.candidate, evaluation_id="recover")
        self.assertTrue(result["receipt"]["reconciled"] and result["receipt"]["settled"])
        self.assertEqual(len(self.launches), 1)
        self.assertEqual(self.handles, ["synthetic-handle", None])

    def test_prepared_id_is_not_launch_permission_or_another_source(self):
        self.engine.prepare(self.candidate, evaluation_id="prepared")
        self.engine.evaluate(self.candidate, evaluation_id="prepared")
        self.assertEqual(self.launches, [])
        (self.task / "kernel.py").write_bytes(b"# changed candidate\n")
        source = self.store.capture_source(self.task, self.contract["source_paths"])
        other = self.store.candidate(self.job_id, "epoch", source)
        # Restore the protected founding artifact before admitting a candidate.
        (self.task / "kernel.py").write_bytes(fixture()[1]["kernel.py"])
        with self.assertRaisesRegex(ValueError, "different contract/source/epoch"):
            self.engine.evaluate(other, evaluation_id="prepared")

    def test_quantities_require_source_bound_raw_trials_not_claims(self):
        result = self.engine.evaluate(self.candidate, evaluation_id="quantity")
        evidence = self.store.archive_evidence(self.candidate, result["artifacts"])
        names = {"evaluation": "arena_evaluation.json", "host": "host_receipt.json", "request": "request.json",
                 "plan": "raw/host/plan.json", "owner_report": "raw/host/owner-report.json"}
        refs = {key: {"evidence_id": evidence, "artifact": name} for key, name in names.items()}
        producer = quantities.NativeQuantities(self.store, "no-campaign", contract=self.contract,
            admission={"physical_ids": ["0000:01:00.0@1"], "bdfs": ["0000:01:00.0"]}, case_id="case-a",
            producer_ref={"path": "synthetic-producer", "sha256": "5" * 64})
        pair, flags = producer._derive(self.candidate, refs)
        self.assertEqual(pair["quantity"]["value"], "1.25")
        self.assertFalse(any(flags.values()))
        with self.assertRaisesRegex(ValueError, "provenance is incomplete"):
            producer._derive(self.candidate, {"claim": refs["evaluation"]})
        self.assertFalse(producer({"evidence_id": evidence, "artifact": "quantity.json"}, pair, pair["binding"]))

    def test_quantities_bind_registered_ascii_and_unicode_tasks(self):
        for label in ("ascii", "Unicode: μ × β"):
            with self.subTest(label=label):
                contract = copy.deepcopy(self.contract)
                contract["report_contract"]["description"] = label
                job_id = jobs.job_digest(contract)
                if label == "ascii":
                    self.assertEqual(job_id, sha(encoded(contract)))
                else:
                    self.assertNotEqual(job_id, sha(encoded(contract)))
                store = ResearchStore(self.root / label, allowed_epochs=[(job_id, "epoch")],
                                      source_paths={job_id: contract["source_paths"]})
                source = store.capture_source(self.task, contract["source_paths"])
                self.assertEqual(source, self.source)
                candidate = store.candidate(job_id, "epoch", source)
                producer = quantities.NativeQuantities(store, "no-campaign", contract=contract,
                    admission={}, case_id="case-a", producer_ref={"synthetic": True})
                for case, binding in producer.case_bindings(candidate).items():
                    self.assertEqual(producer.descriptors[case]["job_id"], job_id)
                    self.assertEqual(binding["job_id"], job_id)
                    self.assertEqual(binding["source_id"], source)
                engine = arena.ArenaEvaluator(store, contract=contract, contract_id=job_id,
                    artifact_root=self.task, output_root=self.root / (label + "-evaluation"),
                    measurement_epoch_id="epoch", launch=self.launch, settle=self.settle)
                request = engine.prepare(candidate, evaluation_id="quantity-identity")
                # Task IDs use UTF-8; quantity and request JSON keep their existing bytes.
                self.assertEqual(quantities.encoded(contract), encoded(contract))
                self.assertEqual(arena.canonical(request), encoded(request))
                self.assertEqual(quantities.digest(request), sha(encoded(request)))

    def test_native_contract_rejects_comparable_fallback(self):
        changed = copy.deepcopy(self.contract)
        changed["timing_policy"]["fallback"]["comparable"] = True
        with self.assertRaisesRegex(ValueError, "fallback"):
            jobs.validate_arena_job(changed)

    def test_exact_original_exception_types_are_preserved(self):
        # Exception names are saved in error records; a subclass is not parity.
        changed = {**self.contract, "measurement_valid": True}
        empty_host = {"cleanup": {"evidence": []}, "runtime": {"evidence": []},
                      "observations": {}, "artifacts_sha256": {}}
        checks = [
            ("legacy schema", lambda: jobs_v1.validate_job({}), ValueError),
            ("Arena schema", lambda: jobs.validate_arena_job({}), ValueError),
            ("native flags", lambda: native_contract.validate_native_job(changed), ValueError),
            ("missing job", lambda: jobs.load_arena_job(self.root / "missing.json"), ValueError),
            ("quantity mapping", lambda: quantities.allocation({}), ValueError),
            ("quantity JSON", lambda: quantities.decode(b'{"value":NaN}'), ValueError),
            ("transport limits", lambda: artifacts.pack({}, {}, {}, {}, max_count=-1), ValueError),
            ("transport closure", lambda: artifacts.pack({"orphan": b"x"}, empty_host, {}, {}), ValueError),
            ("Arena receipt", lambda: arena.ArenaEvaluator._envelope({}, {}, {}), arena.ArenaEvaluationError),
            ("native JSON", lambda: native._parse(b'{"x":1,"x":2}'), native.NativeEvaluationError),
            ("timing input", lambda: timing.analyze_measurements({}, timing_aggregation="median"), timing.MeasurementError),
        ]
        for name, operation, expected in checks:
            with self.subTest(name=name):
                try:
                    operation()
                except Exception as error:
                    self.assertIs(type(error), expected)
                    self.assertEqual(type(error).__name__, expected.__name__)
                else:
                    self.fail("invalid input did not raise")

    def test_legacy_native_assessment_rejects_partial_or_wrong_scope(self):
        evaluator = object.__new__(native.NativeEvaluator)
        evaluator.case_ids = ["case-a", "case-b"]
        request = {"operation_id": "native-op", "mode": "performance"}
        rows = [{"test_case_id": case, "execution_time_ms": 1.0,
                 "params": {"timing_scope": "complete_callable_synchronized_wall"}}
                for case in evaluator.case_ids]
        for mode in ("complete", "partial", "wrong-scope"):
            selected = copy.deepcopy(rows)
            if mode == "partial": selected.pop()
            if mode == "wrong-scope": selected[0]["params"]["timing_scope"] = "kernel_only"
            raw = {"stdout.bin": b"", "stderr.bin": b"", "performance_report.json": encoded({"test_cases": selected})}
            receipt = {"schema": "geak-evo-native-settlement-v1", "operation_id": "native-op",
                "request_sha256": sha(native.canonical(request)), "execution_started": True, "settled": True,
                "evaluation_integrity_valid": True, "measurement_valid": False, "peer_exception_active": False,
                "artifacts_sha256": {name: sha(data) for name, data in raw.items()}, "errors": [],
                "exit_code": 0, "qualification_ref": {"synthetic": True}, "physical_devices": ["fixture-device"]}
            assessed = evaluator._assess(request, receipt, raw, [])
            self.assertEqual(assessed["success"], mode == "complete")
            self.assertTrue(assessed["settled"])


if __name__ == "__main__":
    unittest.main()
