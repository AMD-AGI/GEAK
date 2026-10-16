"""Synthetic declaration tests. No generated task, tensor or native execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from geak_evo.evaluate.jobs import load_engineer_job, validate_arena_job, job_digest
from geak_evo.evaluate.generated_task import validate_generated_task, fixture_admission, digest, project_generated_task


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


class GeneratedFixture:
    """Shared CPU-only source fixture for contract, runner and observer tests."""
    def __init__(self, root):
        self.root = Path(root)
        self.artifact_root = self.root / "task"; self.artifact_root.mkdir()
        self.fixture_root = self.root / "fixture_inputs"; self.fixture_root.mkdir()
        oracle = b"synthetic fixture bytes; deliberately not a torch pickle"
        (self.fixture_root / "oracle.bin").write_bytes(oracle)
        oracle_sha = hashlib.sha256(oracle).hexdigest()
        self.runtime = {"image": "example.invalid/runtime@sha256:" + "a" * 64,
                        "config_digest": "sha256:" + "b" * 64}
        self.meta = {"target_callable": "serving.attention:run", "candidate_bind": {
            "kind": "module", "module": "serving.attention", "file": "kernel_src/attention.py"},
            "reference_io": "reference_io.pt", "reference_io_sha256": oracle_sha,
            "tol": 0.02, "random_draws": 3, "regime": {"cuda_graph": False}}
        self.case_rows = [
            {"case_id": "case-a", "signature": "batch=1,heads=16", "descriptor": {"shape": [1, 16], "dtype": "bf16"},
             "correctness": True, "benchmark": True},
            {"case_id": "case-b", "signature": "batch=4,heads=16", "descriptor": {"shape": [4, 16], "dtype": "bf16"},
             "correctness": True, "benchmark": True}]
        files = {name: b"# Synthetic protected source fixture; never executed.\n" for name in
                 ("cases.py", "harness_lib.py", "leg_runner.py", "overlay_setup.py", "unittest.py", "evidence_helper.py")}
        files["kernel_src/attention.py"] = b"def run(*args, **kwargs): pass\n"
        files["baseline_overlay/manifest.json"] = encoded({"synthetic": "frozen baseline overlay"})
        files["extraction.json"] = encoded({"synthetic": "new task provenance; no qualification"})
        self.baseline_closure = {"schema": "geak-generated-task-baseline-closure-v1", "runtime": self.runtime,
            "target_callable": self.meta["target_callable"], "overlay_root": "baseline_overlay",
            "task_files": {"baseline_overlay/manifest.json": {
                "sha256": hashlib.sha256(files["baseline_overlay/manifest.json"]).hexdigest(), "executable": 0}},
            "external_files": [{"path": "/opt/example/serving/attention.py", "sha256": "c" * 64, "executable": 0}]}
        files["meta.json"] = encoded(self.meta)
        files["generated_cases.json"] = encoded({"schema": "geak-generated-task-cases-v1", "cases": self.case_rows})
        files["baseline_closure.json"] = encoded(self.baseline_closure)
        for name, raw in files.items():
            path = self.artifact_root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw); path.chmod(0o644)
        self.contract = {
            "schema": "geak-arena-generated-task-v1",
            "source": {"repository": "hyperloom-generated-task", "revision": "d" * 64,
                       "task_selector": "tasks/generated/example", "snapshot_manifest_sha256": "e" * 64},
            "runtime": self.runtime,
            "runtime_requirements": {"target_callable": self.meta["target_callable"],
                                     "candidate_bind": copy.deepcopy(self.meta["candidate_bind"]), "task_run_root": "."},
            "provenance": {"kind": "hyperloom_kernel_extraction", "evidence_ref": "extraction.json"},
            "task_metadata": {"task_root": str(self.artifact_root), "meta_path": "meta.json", "oracle_path": "reference_io.pt",
                "case_manifest_path": "generated_cases.json", "baseline_closure_path": "baseline_closure.json",
                "shared_harness_paths": ["cases.py", "harness_lib.py", "leg_runner.py", "overlay_setup.py", "unittest.py", "evidence_helper.py"]},
            "artifacts": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
            "artifact_modes": {name: 0 for name in files}, "aliases": {}, "source_paths": ["kernel_src/attention.py"],
            "fixtures": {"reference_io.pt": {"manifest_path": "oracle.bin", "sha256": oracle_sha, "size_bytes": len(oracle)}},
            "correctness_case_ids": [row["case_id"] for row in self.case_rows],
            "benchmark_case_ids": [row["case_id"] for row in self.case_rows],
            "benchmark_report_ids": {row["case_id"]: row["signature"] for row in self.case_rows},
            "case_bindings": {row["case_id"]: {"artifact_ref": "generated_cases.json", "json_pointer": "/cases/" + str(index),
                                               "sha256": digest(row)} for index, row in enumerate(self.case_rows)},
            "phases": {"evaluate": {"argv": ["python3", "unittest.py"], "timeout_seconds": 3600}},
            "numerical_policy": {"kind": "geak-generated-task-correctness-v1", "meta_ref": "meta.json",
                "source_refs": ["cases.py", "harness_lib.py", "unittest.py"], "result": "run_correctness_ok_and_report",
                "compile_parity_exception": "retain_soft_degraded"},
            "timing_policy": {"kind": "geak-generated-task-paired-timing-v1",
                "in_process": {"warmup_calls": 10, "measured_calls": 50, "inner_calls": 1, "statistic": "upper_middle", "unit": "ms"},
                "replication": {"order": ["baseline", "candidate"], "fresh_process": True, "max_reps": 3,
                                "undecided": [0.95, 1.10], "stop_rule": "median_ratio_outside_inclusive_interval"},
                "aggregation": {"per_leg": "median", "even_count": "mean_of_middle_pair",
                                "per_case": "baseline_median_over_candidate_median", "lane": "geometric_mean_of_case_ratios"},
                "fallback": "retain_actual_timer_and_detail", "source_refs": ["harness_lib.py", "leg_runner.py", "unittest.py"]},
            "report_contract": {"protocol": "geak-generated-task-paired-report-v1",
                "evidence_schema": "geak-extracted-measurement-evidence-v1", "stdout_prefix": "GEAK_MEASUREMENT_EVIDENCE ",
                "writable_paths": ["_cand_overlay", "_baseline_random.pt", "__pycache__", "kernel_src/__pycache__"]},
            "measurement_valid": False, "custody_proven": False, "reward_eligible": False}

    def write_artifact(self, name, value):
        raw = value if type(value) is bytes else encoded(value)
        path = self.artifact_root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        path.chmod(0o644)
        self.contract["artifacts"][name] = hashlib.sha256(raw).hexdigest()
        self.contract["artifact_modes"][name] = 0

    def validate(self):
        return validate_generated_task(self.contract, self.artifact_root)

    def contract_path(self):
        path = self.root / "contract.json"; path.write_bytes(encoded(self.contract)); return path


class GeneratedContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.root = Path(self.temporary.name)
        self.fixture = GeneratedFixture(self.root)

    def tearDown(self): self.temporary.cleanup()

    def test_complete_task_and_loader_digest(self):
        f = self.fixture; parsed = f.validate()
        self.assertEqual(parsed, f.contract)
        self.assertEqual(validate_arena_job(f.contract, f.artifact_root), parsed)
        self.assertEqual(load_engineer_job(f.contract_path(), f.artifact_root, job_digest(parsed)), parsed)
        parsed["runtime"]["image"] = "changed"
        self.assertNotEqual(parsed, f.contract)

    def projection_inputs(self):
        f = self.fixture
        rows = {name: {"sha256": sha, "executable": f.contract["artifact_modes"][name],
                       "size_bytes": (f.artifact_root / name).stat().st_size}
                for name, sha in f.contract["artifacts"].items()}
        rows["reference_io.pt"] = {**f.contract["fixtures"]["reference_io.pt"], "executable": 0}
        rows["reference_io.pt"].pop("manifest_path")
        observation = {"schema": "geak-generated-task-admission-observation-v1",
            "task_dir": str(f.artifact_root), "runtime": f.runtime,
            "files": {name: row for name, row in rows.items()
                      if name not in {"generated_cases.json", "baseline_closure.json", "extraction.json"}}, "aliases": {},
            "case_manifest": {"schema": "geak-generated-task-cases-v1", "cases": f.case_rows},
            "effective_measurement": {"max_reps": 3, "undecided": [0.95, 1.10]}}
        policy = {"runtime": f.runtime, "evaluation_seconds": 3600,
            "shared_harness_sha256": {name: rows[name]["sha256"] for name in ("harness_lib.py", "leg_runner.py")},
            "writable_paths": f.contract["report_contract"]["writable_paths"]}
        return dict(task=f.artifact_root, observation=observation, policy=policy,
                    provenance_ref="extraction.json", artifact_rows=rows, meta=f.meta)

    def test_projection_preserves_cases_runtime_timing_and_no_authority(self):
        inputs = self.projection_inputs()
        result = project_generated_task(**inputs)
        self.assertEqual(validate_arena_job(result, self.fixture.artifact_root), result)
        for key in ("runtime", "runtime_requirements", "case_bindings", "correctness_case_ids",
                    "benchmark_case_ids", "benchmark_report_ids", "timing_policy", "numerical_policy"):
            self.assertEqual(result[key], self.fixture.contract[key])
        self.assertTrue(all(result[key] is False for key in ("measurement_valid", "custody_proven", "reward_eligible")))
        result["runtime"]["config_digest"] = "changed"
        self.assertEqual(inputs["observation"]["runtime"], self.fixture.runtime)

    def test_projection_refuses_missing_observation_or_conflicting_host_policy(self):
        for change in (lambda v: v["observation"].pop("case_manifest"),
                       lambda v: v["observation"].pop("effective_measurement"),
                       lambda v: v["observation"].update(task_dir="/another/task"),
                       lambda v: v["policy"].update(runtime={"image": "other"}),
                       lambda v: v["policy"]["shared_harness_sha256"].update({"harness_lib.py": "0" * 64}),
                       lambda v: v["policy"].pop("evaluation_seconds")):
            inputs = copy.deepcopy(self.projection_inputs())
            change(inputs)
            with self.assertRaises(ValueError):
                project_generated_task(**inputs)

    def test_projection_cannot_substitute_or_add_source_under_observed_identity(self):
        inputs = self.projection_inputs()
        inputs["artifact_rows"] = copy.deepcopy(inputs["artifact_rows"])
        inputs["artifact_rows"]["kernel_src/attention.py"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "copied artifact differs"):
            project_generated_task(**inputs)
        inputs = self.projection_inputs()
        inputs["artifact_rows"]["kernel_src/extra.py"] = dict(inputs["artifact_rows"]["kernel_src/attention.py"])
        with self.assertRaisesRegex(ValueError, "artifact inventory differs"):
            project_generated_task(**inputs)

    def test_unknown_schema_never_falls_back(self):
        f = self.fixture; f.contract["schema"] = "unknown"
        with self.assertRaises(ValueError): load_engineer_job(f.contract_path(), f.artifact_root)

    def test_no_declaration_can_assert_qualification(self):
        for flag in ("measurement_valid", "custody_proven", "reward_eligible"):
            f = self.fixture; f.contract[flag] = True
            with self.subTest(flag=flag), self.assertRaises(ValueError): f.validate()
            f.contract[flag] = False

    def test_changed_source_bytes_modes_and_symlink_rejected(self):
        f = self.fixture; path = f.artifact_root / "kernel_src/attention.py"; original = path.read_bytes()
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError): f.validate()
        path.write_bytes(original); path.chmod(0o755)
        with self.assertRaises(ValueError): f.validate()
        path.chmod(0o644); path.unlink(); target = self.root / "same"; target.write_bytes(original); path.symlink_to(target)
        with self.assertRaises(ValueError): f.validate()

    def test_execute_mode_inventory_can_admit_exact_executable_source(self):
        f = self.fixture; name = "kernel_src/attention.py"
        f.contract["artifact_modes"][name] = 0o111; (f.artifact_root / name).chmod(0o755)
        self.assertEqual(f.validate(), f.contract)

    def test_large_baseline_build_identity_and_same_size_corruption(self):
        f = self.fixture; name = "baseline_overlay/native.bin"
        path = f.artifact_root / name
        block = b"\x00" * (1024 * 1024); hasher = hashlib.sha256()
        with path.open("xb") as stream:
            for _ in range(33):
                stream.write(block); hasher.update(block)
        path.chmod(0o644)
        f.contract["artifacts"][name] = hasher.hexdigest()
        f.contract["artifact_modes"][name] = 0
        f.baseline_closure["task_files"][name] = {"sha256": hasher.hexdigest(), "executable": 0}
        f.write_artifact("baseline_closure.json", f.baseline_closure)
        self.assertEqual(f.validate(), f.contract)
        with path.open("r+b") as stream:
            stream.seek(-1, 2); stream.write(b"\x01")
        with self.assertRaises(ValueError): f.validate()

    def test_semantically_decoded_metadata_keeps_small_file_limit(self):
        f = self.fixture
        for name in ("meta.json", "generated_cases.json", "baseline_closure.json"):
            original = (f.artifact_root / name).read_bytes()
            f.write_artifact(name, original + b" " * (8 * 1024 * 1024))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "oversize small artifact"):
                f.validate()
            f.write_artifact(name, original)

    def test_fixture_body_is_separate_and_streamed_with_actual_root_path(self):
        f = self.fixture; f.validate()
        receipt = fixture_admission(f.contract, f.fixture_root, "/work/generated")
        self.assertEqual(receipt["mount_plan"][0]["destination"], "/work/generated/reference_io.pt")
        self.assertFalse(receipt["mounts_applied"])
        (f.fixture_root / "oracle.bin").write_bytes(b"changed")
        f.validate()  # Small-artifact parsing never loads fixture bodies.
        with self.assertRaises(ValueError): fixture_admission(f.contract, f.fixture_root)

    def test_oracle_path_is_not_prefixed_with_ut(self):
        f = self.fixture; f.contract["task_metadata"]["oracle_path"] = "ut/reference_io.pt"
        f.contract["fixtures"]["ut/reference_io.pt"] = f.contract["fixtures"].pop("reference_io.pt")
        with self.assertRaises(ValueError): f.validate()

    def test_rehashed_meta_cannot_change_actual_callable_or_oracle_binding(self):
        f = self.fixture
        for key, value in (("target_callable", "other:run"), ("reference_io_sha256", "0" * 64),
                           ("reference_io", "different.pt"), ("candidate_bind", {"kind": "module", "module": "other", "file": "kernel_src/attention.py"})):
            changed = dict(f.meta, **{key: value}); f.write_artifact("meta.json", changed)
            with self.subTest(key=key), self.assertRaises(ValueError): f.validate()
        f.write_artifact("meta.json", f.meta)

    def test_full_case_descriptor_hash_and_signature_join(self):
        f = self.fixture; rows = copy.deepcopy(f.case_rows); rows[0]["descriptor"]["shape"] = [99, 16]
        f.write_artifact("generated_cases.json", {"schema": "geak-generated-task-cases-v1", "cases": rows})
        with self.assertRaises(ValueError): f.validate()

    def test_missing_duplicate_extra_case_and_report_mappings_fail(self):
        f = self.fixture; original = copy.deepcopy(f.contract)
        mutations = [("correctness_case_ids", ["case-a"]), ("correctness_case_ids", ["case-a", "case-a"]),
                     ("benchmark_case_ids", ["case-a"]), ("benchmark_report_ids", {"case-a": "wrong", "case-b": "batch=4,heads=16"}),
                     ("case_bindings", {"case-a": original["case_bindings"]["case-a"]})]
        for key, value in mutations:
            f.contract = copy.deepcopy(original); f.contract[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): f.validate()

    def test_protected_metadata_and_output_scopes_cannot_overlap_editable_files(self):
        f = self.fixture
        for roots in (["kernel_src"], ["baseline_overlay"], ["reference_io.pt"], ["meta.json"], ["."]):
            f.contract["report_contract"]["writable_paths"] = roots
            with self.subTest(roots=roots), self.assertRaises(ValueError): f.validate()

    def test_phase_must_be_single_bound_evaluate(self):
        f = self.fixture
        for phases in ({"compile": {"argv": ["python3", "unittest.py"], "timeout_seconds": 3}},
                       {"evaluate": {"argv": ["python3", "other.py"], "timeout_seconds": 3}},
                       {"evaluate": {"argv": ["python3", "unittest.py"], "timeout_seconds": True}}):
            f.contract["phases"] = phases
            with self.subTest(phases=phases), self.assertRaises(ValueError): f.validate()

    def test_real_positive_max_reps_is_bound_without_invented_cap(self):
        f = self.fixture
        for reps in (1, 2, 3, 5):
            f.contract["timing_policy"]["replication"]["max_reps"] = reps
            self.assertEqual(f.validate(), f.contract)
        for reps in (0, -1, True, 1.5):
            f.contract["timing_policy"]["replication"]["max_reps"] = reps
            with self.subTest(reps=reps), self.assertRaises(ValueError): f.validate()

    def test_effective_undecided_interval_is_bound_and_well_formed(self):
        f = self.fixture
        f.contract["timing_policy"]["replication"]["undecided"] = [0.97, 1.04]
        self.assertEqual(f.validate(), f.contract)
        for interval in ([0, 1], [1.1, 0.95], [True, 2], [0.9], [0.9, float("inf")]):
            f.contract["timing_policy"]["replication"]["undecided"] = interval
            with self.subTest(interval=interval), self.assertRaises(ValueError): f.validate()

    def test_correctness_domain_can_include_an_untimed_case(self):
        f = self.fixture; f.case_rows[1]["benchmark"] = False
        f.contract["benchmark_case_ids"] = ["case-a"]
        del f.contract["benchmark_report_ids"]["case-b"]
        f.contract["case_bindings"]["case-b"]["sha256"] = digest(f.case_rows[1])
        f.write_artifact("generated_cases.json", {"schema": "geak-generated-task-cases-v1", "cases": f.case_rows})
        self.assertEqual(f.validate(), f.contract)

    def test_exact_loader_digest_and_duplicate_json_keys_fail(self):
        f = self.fixture; path = f.contract_path()
        with self.assertRaises(ValueError): load_engineer_job(path, f.artifact_root, "0" * 64)
        path.write_bytes(b'{"schema":"geak-arena-generated-task-v1",' + encoded(f.contract)[1:])
        with self.assertRaises(ValueError): load_engineer_job(path, f.artifact_root)

    def test_timer_statistic_and_paired_aggregation_cannot_be_relabelled(self):
        f = self.fixture; original = copy.deepcopy(f.contract["timing_policy"])
        for section, key, value in (("in_process", "measured_calls", 100), ("in_process", "statistic", "mean"),
                                     ("in_process", "inner_calls", True), ("replication", "order", ["candidate", "baseline"]),
                                     ("aggregation", "even_count", "upper_middle"), ("aggregation", "per_case", "median_of_ratios")):
            f.contract["timing_policy"] = copy.deepcopy(original); f.contract["timing_policy"][section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): f.validate()

    def test_declared_fallback_and_compile_parity_semantics_are_retained(self):
        result = self.fixture.validate()
        self.assertEqual(result["timing_policy"]["fallback"], "retain_actual_timer_and_detail")
        self.assertEqual(result["numerical_policy"]["compile_parity_exception"], "retain_soft_degraded")

    def test_baseline_runtime_callable_and_complete_overlay_join(self):
        f = self.fixture; original = copy.deepcopy(f.baseline_closure)
        for key, value in (("runtime", dict(f.runtime, image="other")), ("target_callable", "other:run"),
                           ("task_files", {}), ("external_files", [])):
            f.write_artifact("baseline_closure.json", dict(original, **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError): f.validate()

    def test_case_signatures_may_contain_native_id_invalid_characters(self):
        self.assertIn(",", self.fixture.validate()["benchmark_report_ids"]["case-a"])

    def test_non_ascii_signature_uses_shared_canonical_utf8_identity(self):
        f = self.fixture; signature = "batch=1,op=注意力,état=référence"
        f.case_rows[0]["signature"] = signature
        f.contract["benchmark_report_ids"]["case-a"] = signature
        f.contract["case_bindings"]["case-a"]["sha256"] = digest(f.case_rows[0])
        f.write_artifact("generated_cases.json", {"schema": "geak-generated-task-cases-v1", "cases": f.case_rows})
        self.assertEqual(f.validate()["benchmark_report_ids"]["case-a"], signature)
        self.assertEqual(job_digest(f.contract), hashlib.sha256(encoded(f.contract)).hexdigest())



if __name__ == "__main__": unittest.main()
