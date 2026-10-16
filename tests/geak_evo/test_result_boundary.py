"""Synthetic owner collections only; no task, model or GPU execution."""
import json
import math
from pathlib import Path
import subprocess
import unittest
from unittest import mock

from geak_evo import result as boundary
import test_evaluate_contracts as fixtures


class ResultBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = f = fixtures.EvaluationContractTests()
        contract, files = fixtures.fixture()
        self.manifest = {"cases": [{"case_id": case} for case in contract["benchmark_case_ids"]],
                         "measurement": {"method": "cuda_graph", "benchmark_iterations": 100,
                                         "warmup_iterations": 10, "refresh_inputs": "each_replay",
                                         "initialize_outputs": "each_replay", "validate_outputs": "each_replay"}}
        files["cases.json"] = fixtures.encoded(self.manifest)
        contract["artifacts"]["cases.json"] = fixtures.sha(files["cases.json"])
        contract["timing_policy"]["primary"].update(kind="cuda_event_graph_replay", graph=True)
        contract["report_contract"] = {"published_protocol": {
            "schema": "geak-published-task-protocol-v1", "report_format": "task-report-v1",
            "case_manifest": {"path": "cases.json", "sha256": fixtures.sha(files["cases.json"])},
            "private_cache_scope": "phase", "preserve_cache_vars": ["AITER_JIT_DIR"],
            "private_image_caches": {"AITER_JIT_DIR": {"manifest_sha256": "a" * 64}}}}
        with mock.patch.object(fixtures, "fixture", return_value=(contract, files)):
            f.setUp()
        f.engine.settle = self.settle
        self.change_report = lambda report: None
        self.addCleanup(f.doCleanups)
        self.selected_root = f.root / "candidate-source"; self.selected_root.mkdir()
        (self.selected_root / "kernel.py").write_bytes(b"# selected synthetic source\n")
        source = f.store.capture_source(self.selected_root, f.contract["source_paths"])
        self.selected = f.store.candidate(f.job_id, "epoch", source)
        self.choice = {"candidate_id": self.selected, "source_id": source,
                       "binding": {"campaign_id": "campaign", "job_id": f.job_id, "measurement_epoch_id": "epoch"}}
        self.selection = {"status": "selected", "choice": self.choice, "record": {"disposition": "set"}}
        self.snapshot = {"campaign_id": "campaign", "revision": 3, "state": {"phase": "blocked"}}
        self.options = {"expected_binding": self.choice["binding"]}
        self.index = 0

    def settle(self, handle, request, operation):
        f = self.fixture
        collected = f.settle(handle, request, operation)
        host, artifacts = collected["receipt"], collected["artifacts"]
        protocol = f.contract["report_contract"]["published_protocol"]
        source = {row["path"]: row["blob_id"] for row in request["source"]["files"]}
        native_request = {"request_id": fixtures.sha(request["operation_id"].encode())[:48], "phase": "performance", "source_sha256": source,
                          "manifest_sha256": fixtures.sha(fixtures.encoded(self.manifest))}
        report = {"schema_version": 1, "status": "ok", "request": native_request, "cases": [
            {"case": row, "correct": True, "benchmark_method": "cuda_graph", "warmup_iterations": 10,
             "fresh_input_resets": 100, "output_initializations": 100, "oracle_checks": 100,
             "samples_ms": [host["observations"]["benchmark"]["times_ms"].get(row["case_id"], 1)] * 100}
            for row in self.manifest["cases"]]}
        self.change_report(report)
        prefix = "observer/performance/"
        name = prefix + "raw-reports/build/performance_report.json"
        artifacts[name] = fixtures.encoded(report)
        binding = {"schema": "geak-published-task-protocol-v1", "phase": "performance", "request": native_request, "source_sha256": source,
                   "manifest_sha256": native_request["manifest_sha256"], "protocol_sha256": fixtures.sha(fixtures.encoded(protocol))}
        artifacts[prefix + "protocol-binding.json"] = fixtures.encoded(binding)
        artifacts[prefix + "protocol-request.json"] = fixtures.encoded(native_request)
        artifacts[prefix + "phase.json"] = fixtures.encoded({"phase": "performance", "returncode": 0,
            "timed_out": False, "settlement": {"empty": True}, "tracked_file_changes": [],
            "protocol_binding_sha256": fixtures.sha(artifacts[prefix + "protocol-binding.json"]),
            "reports": {"build/performance_report.json": {"changed_since_phase_start": True,
                        "sha256": fixtures.sha(artifacts[name]), "size_bytes": len(artifacts[name])}}})
        host["observations"]["benchmark"]["evidence"] += [name, prefix + "protocol-binding.json",
                                                            prefix + "protocol-request.json", prefix + "phase.json"]
        artifacts["native/performance/private-cache-receipt.json"] = fixtures.encoded({
            "AITER_JIT_DIR": {"private_path": "protocol-cache/AITER_JIT_DIR", "seed_manifest_sha256": "a" * 64}})
        artifacts["native/performance/empty-cache-receipt.json"] = fixtures.encoded({"XDG_CACHE_HOME": "empty-cache/xdg"})
        host["artifacts_sha256"] = {name: fixtures.sha(raw) for name, raw in artifacts.items()}
        return collected

    def measure(self, before=None, after=None, mutate=None):
        f = self.fixture
        for candidate, name, times in [(f.candidate, "baseline", before or {"case-a": 1.0, "case-b": 1.0}),
                                       (self.selected, "selected", after or {"case-a": 0.5, "case-b": 0.25})]:
            def change(receipt):
                receipt["observations"]["benchmark"]["times_ms"] = times
                if mutate is not None:
                    mutate(name, receipt)
            f.mutate = change
            f.engine.evaluate(candidate, evaluation_id=name)

    def export(self, **overrides):
        f = self.fixture; self.index += 1
        arguments = dict(evaluator=f.engine, selection_options=self.options,
                         baseline_candidate_id=f.candidate, baseline_evaluation_id="baseline",
                         selected_evaluation_id="selected", output_dir=f.root / ("result-" + str(self.index)))
        arguments.update(overrides)
        with mock.patch.object(boundary, "validate_final_recommendation", return_value=self.selection) as replay:
            result = boundary.export_final_result(f.store, self.snapshot, **arguments)
        replay.assert_called_once_with(f.store, self.snapshot, **self.options)
        return result

    def test_exact_selected_patch_complete_metrics_and_provisional_scope(self):
        self.measure()
        launches = len(self.fixture.launches)
        result = self.export()
        self.assertEqual(len(self.fixture.launches), launches)
        self.assertEqual(result["outcome"], "improved")
        self.assertAlmostEqual(result["final_geomean"], 8 ** 0.5)
        self.assertEqual(result["final_arithmetic"], 3)
        self.assertIsNone(result["final_weighted"])
        self.assertEqual(result["campaign_phase"], "blocked")
        self.assertEqual(result["validation_status"], "provisional_native_observation")
        self.assertEqual(result["timing_basis"], "cuda_graph")
        self.assertTrue(all(result[key] is False for key in
                            ("measurement_valid", "custody_proven", "reward_eligible", "framework_final_pass")))
        materialized = Path(result["eval_dir"]) / "selected-source/kernel.py"
        self.assertEqual(materialized.read_bytes(), (self.selected_root / "kernel.py").read_bytes())
        self.assertIn(b"+\x23 selected synthetic source", Path(result["final_patch"]).read_bytes())

    def test_weighted_primary_can_reject_a_positive_geomean(self):
        self.measure(after={"case-a": 2.0, "case-b": 0.25})
        objective = {"kind": "time_weighted", "weights": {"case-a": 0.75, "case-b": 0.25}}
        result = self.export(objective=objective)
        self.assertGreater(result["final_geomean"], 1)
        self.assertEqual(result["final_speedup"], 0.64)
        self.assertEqual(result["outcome"], "no_improvement")
        self.assertEqual(result["final_patch"], "")
        self.assertTrue(Path(result["selected_patch"]).is_file())
        self.assertEqual(result["objective"], objective)

    def test_lane_time_shares_and_provenance_are_preserved(self):
        self.measure(before={"case-a": 2.0, "case-b": 4.0}, after={"case-a": 1.0, "case-b": 1.0})
        for weights in ({"case-a": 3, "case-b": 1}, {"case-a": 0.75, "case-b": 0.25}):
            objective = {"kind": "time_weighted", "weights": weights,
                         "workload_spec_path": "/caller/workload.json",
                         "workload": [{"name": "case-a", "weight_source": "regime"}]}
            with self.subTest(weights=weights):
                result = self.export(objective=objective)
                self.assertEqual(result["final_speedup"], 16 / 7)
                self.assertEqual(result["final_weighted"], 16 / 7)
                self.assertAlmostEqual(result["final_geomean"], 8 ** 0.5)
                self.assertEqual(result["objective"], objective)
                self.assertEqual(json.loads((Path(result["eval_dir"]) / "result.json").read_bytes())["objective"], objective)

    def test_partial_missing_or_invalid_weights_are_not_renormalized(self):
        for weights in (None, {"case-a": 1}, {"case-a": 1, "case-b": 1, "extra": 1},
                        {"case-a": 0, "case-b": 1}, {"case-a": -1, "case-b": 1},
                        {"case-a": True, "case-b": 1}, {"case-a": float("inf"), "case-b": 1},
                        {"case-a": float("nan"), "case-b": 1}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.export(objective={"kind": "time_weighted", "weights": weights})

    def test_varying_samples_match_the_published_fsum_producer_exactly(self):
        samples = [0.1, 0.2] * 50
        published_mean = math.fsum(samples) / len(samples)
        naive_mean = sum(samples) / len(samples)
        self.assertNotEqual(published_mean, naive_mean)
        def varying(report):
            for row in report["cases"]:
                row["samples_ms"] = samples
        self.change_report = varying
        for scalar, expected in ((published_mean, "no_improvement"), (naive_mean, "error")):
            with self.subTest(scalar=scalar):
                self.measure(before={"case-a": scalar, "case-b": scalar},
                             after={"case-a": scalar, "case-b": scalar})
                result = self.export()
                self.assertEqual(result["outcome"], expected)
                if expected == "error":
                    self.assertIn("native timing scalar differs", result["error"])
                    self.assertIsNone(result["final_speedup"])
                    self.assertEqual(result["final_patch"], "")
                else:
                    self.assertEqual(result["final_speedup"], 1)
                    self.assertEqual(result["validation_status"], "provisional_native_observation")

    def test_clear_and_missing_never_read_measurements_or_choose_a_fallback(self):
        for record, expected in (({"disposition": "clear"}, "no_improvement"), (None, "error")):
            self.selection = {"status": "missing", "choice": None, "record": record}
            with mock.patch.object(boundary, "read_evaluation", side_effect=AssertionError("must not read")):
                result = self.export()
            self.assertEqual(result["outcome"], expected)
            self.assertEqual(result["final_patch"], "")
            self.assertIsNone(result["final_geomean"])

    def test_source_substitution_or_wrong_original_cannot_supply_a_result(self):
        self.measure()
        result = self.export(baseline_candidate_id=self.selected)
        self.assertEqual(result["outcome"], "error")
        self.assertIn("admitted task source", result["error"])
        self.choice["source_id"] = self.fixture.source
        result = self.export()
        self.assertIn("domain differs", result["error"])
        self.assertEqual(result["final_patch"], "")

    def test_changed_retained_assessment_is_rejected(self):
        self.measure()
        path = self.fixture.engine.root / "selected/collection-0001/arena_evaluation.json"
        value = json.loads(path.read_bytes()); value["per_case_times_ms"]["case-a"] = 0.001
        path.chmod(0o644); path.write_bytes(fixtures.encoded(value))
        result = self.export()
        self.assertEqual(result["outcome"], "error")
        self.assertIn("assessment differs", result["error"])
        self.assertIsNone(result["final_geomean"])

    def test_incomplete_latest_collection_never_falls_back(self):
        self.measure()
        (self.fixture.engine.root / "selected/collection-0002").mkdir()
        result = self.export()
        self.assertEqual(result["outcome"], "error")
        self.assertIsNone(result["final_speedup"])

    def test_busy_owner_cannot_export_an_older_pass(self):
        self.measure()
        with self.fixture.engine._lock():
            result = self.export()
        self.assertEqual(result["outcome"], "error")
        self.assertEqual(result["error_kind"], "BlockingIOError")
        self.assertIsNone(result["final_speedup"])

    def test_missing_case_or_failed_correctness_cannot_supply_a_number(self):
        for failure in ("case", "correctness"):
            def mutate(role, receipt):
                if role == "selected":
                    if failure == "case":
                        receipt["observations"]["benchmark"]["times_ms"].pop("case-b")
                    else:
                        receipt["observations"]["correctness"]["status"] = "failed"
            self.measure(mutate=mutate)
            result = self.export()
            self.assertEqual(result["outcome"], "error")
            self.assertIsNone(result["final_speedup"])
            self.assertEqual(result["validation_status"], "measurement_rejected")

    def test_observed_device_mismatch_cannot_supply_a_number(self):
        self.measure(mutate=lambda role, receipt: receipt["runtime"].update(
            physical_devices=["0000:02:00.0"] if role == "selected" else ["0000:01:00.0"]))
        result = self.export()
        self.assertIn("runtime/device differ", result["error"])
        self.assertIsNone(result["final_geomean"])

    def test_actual_timer_fallback_cannot_supply_a_ratio(self):
        def fallback(report):
            if report["request"]["request_id"] == fixtures.sha(b"selected")[:48]:
                report["cases"][0]["benchmark_method"] = "wall"
        self.change_report = fallback
        self.measure()
        result = self.export()
        self.assertEqual(result["outcome"], "error")
        self.assertIn("fallback is not comparable", result["error"])
        self.assertIsNone(result["final_geomean"])
        self.assertEqual(result["final_patch"], "")

    def test_patch_applies_exact_bytes_modes_deletions_and_empty_files(self):
        original = {"edit.py": (b"old\r\nlast", 0), "mode.py": (b"same\n", 0),
                    "delete.py": (b"gone\n", 0), "empty.py": (None, 0), "space name.py": (b"old\n", 0),
                    'quote"name.py': (b"old\n", 0), "empty-delete.py": (b"", 0)}
        selected = {"edit.py": (b"new\r\nlast\n", 0), "mode.py": (b"same\n", 0o111),
                    "delete.py": (None, 0), "empty.py": (b"", 0), "space name.py": (b"new\n", 0),
                    'quote"name.py': (b"new\n", 0), "empty-delete.py": (None, 0)}
        root = self.fixture.root / "apply"; root.mkdir()
        for name, (body, mode) in original.items():
            if body is not None:
                path = root / name; path.write_bytes(body); path.chmod(0o644 | mode)
        patch = self.fixture.root / "patch.diff"; patch.write_bytes(boundary._patch(original, selected))
        checked = subprocess.run(["git", "apply", str(patch)], cwd=root, capture_output=True, check=False)
        self.assertEqual(checked.returncode, 0, checked.stderr.decode())
        for name, (body, mode) in selected.items():
            path = root / name
            if body is None:
                self.assertFalse(path.exists())
            else:
                self.assertEqual(path.read_bytes(), body)
                self.assertEqual(path.stat().st_mode & 0o111, mode)
        with self.assertRaises(ValueError):
            boundary._patch({"binary": (b"\x00", 0)}, {"binary": (b"\x01", 0)})


if __name__ == "__main__":
    unittest.main()
