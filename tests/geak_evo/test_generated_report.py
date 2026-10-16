"""CPU contract tests; no native task, model, framework, or device imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest


from geak_evo.evaluate import generated_report as READER
from test_generated_task_contract import GeneratedFixture


class TimingReductionTests(unittest.TestCase):
    def test_upper_middle_does_not_average_even_samples(self):
        self.assertEqual(READER.upper_middle(list(range(1, 51))), 26)
        self.assertEqual(READER.conventional_median([25, 26]), 25.5)

    def test_timer_rejects_missing_nonfinite_boolean_and_negative(self):
        for samples in ([1.0] * 49, [1.0] * 49 + [float("nan")],
                        [1.0] * 49 + [True], [1.0] * 49 + [-1]):
            with self.subTest(samples=samples[-1]):
                with self.assertRaises(ValueError):
                    READER.upper_middle(samples)

    def test_separate_medians_are_not_median_pair_ratio(self):
        result = READER.summarize_replicates([1.0, 2.4], [1.0, 1.1])
        self.assertEqual(result["baseline_ms"], 1.7)
        self.assertEqual(result["optimized_ms"], 1.05)
        self.assertEqual(result["speedup"], 1.7 / 1.05)
        self.assertNotEqual(result["speedup"], READER.conventional_median([1.0, 2.4 / 1.1]))

    def test_boundary_ratios_remain_undecided(self):
        for value in (0.95, 1.10):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "stopped"):
                    READER.summarize_replicates([value], [1.0])
                self.assertEqual(READER.summarize_replicates([value] * 3, [1.0] * 3)["reps"], 3)

    def test_decisive_first_pair_forbids_more_pairs(self):
        self.assertEqual(READER.summarize_replicates([2.0], [1.0])["reps"], 1)
        with self.assertRaisesRegex(ValueError, "continued"):
            READER.summarize_replicates([2.0, 1.0], [1.0, 1.0])

    def test_unpaired_or_incomplete_sequence_rejected(self):
        for baseline, candidate in (([], []), ([1], []), ([1, 1], [1, 1]),
                                    ([1] * 4, [1] * 4), ([1, None, 1], [1, 1, 1])):
            with self.subTest(baseline=baseline, candidate=candidate):
                with self.assertRaises(ValueError):
                    READER.summarize_replicates(baseline, candidate)

    def test_job_and_source_canonical_match_their_existing_consumers(self):
        value = {"path": "kernel_src/注意.py"}
        self.assertNotEqual(READER.canonical(value), READER.source_canonical(value))
        self.assertEqual(READER.source_canonical(value), json.dumps(value, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode())

    def test_declared_replication_parameters_are_not_replaced_with_defaults(self):
        out = READER.summarize_replicates([1.05] * 4, [1.0] * 4, max_reps=4, undecided=[0.9, 1.2])
        self.assertEqual(out["reps"], 4)


class ObservationFixture:
    """Synthetic native artifacts, with the actual producer's schema and nesting."""
    def __init__(self, root):
        self.root = Path(root)
        self.generated = GeneratedFixture(root)
        f = self.generated
        self.contract = f.contract
        self.sources = {name: self.contract["artifacts"][name] for name in self.contract["source_paths"]}
        self.task_dir = "/native/attempt/task"
        self.parent_pid = 100
        self.records = []
        self.counter = 0
        self.phase_dir = self.root / "evaluate"
        self.phase_dir.mkdir()
        self.parent = {"operation_id": "actual-evaluate-operation", "request_sha256": "f" * 64,
            "binding": {"job_id": READER.digest(READER.canonical(self.contract)), "measurement_epoch_id": "epoch-v1",
                "candidate_id": "1" * 64, "candidate_source_id": READER.digest(READER.canonical({"kind": "source", "files": [
                    {"path": name, "blob_id": self.sources[name], "executable": self.contract["artifact_modes"][name]}
                    for name in sorted(self.sources)]}))}}
        self.identity = {"task": self.contract["source"]["task_selector"][6:],
            "source_commit": self.contract["source"]["revision"],
            "generated_protocol": {"schema": "geak-generated-task-protocol-v1", "contract": self.contract},
            "public_image": self.contract["runtime"]["image"], "public_config_digest": self.contract["runtime"]["config_digest"],
            "generated_evaluation_binding": self.parent,
            "task_file_manifest": {name: {"sha256": sha, "executable": self.contract["artifact_modes"][name]}
                                    for name, sha in self.contract["artifacts"].items()}}
        for name, ref in self.contract["fixtures"].items():
            self.identity["task_file_manifest"][name] = {"sha256": ref["sha256"], "size_bytes": ref["size_bytes"]}
        self.binding = {"schema": "geak-generated-task-protocol-v1", "phase": "evaluate",
            "parent_evaluation": self.parent, "contract_sha256": READER.digest(READER.canonical(self.contract)),
            "source_sha256": self.sources, "runtime": self.contract["runtime"],
            "protected_artifacts_sha256": {name: value for name, value in self.contract["artifacts"].items() if name not in self.sources}}
        argv = ["/usr/bin/python3", "-B", "-u", "/native/native_capture.py", "unittest.py"]
        self.launch = {"cwd": self.task_dir, "executed_argv": argv}
        self.phase = {"phase": "evaluate", "process_pid": self.parent_pid, "timed_out": False, "returncode": 0,
            "settlement": {"empty": True}, "tracked_file_changes": [], "executed_argv": argv,
            "declared_command": self.contract["phases"]["evaluate"]["argv"], "inner_timeout_seconds": 3600,
            "outer_timeout_seconds": 3600}
        self.emit("correctness", {"status": "ok", "ok": True, "error": None, "report": {
            "eager": [{"case": row["signature"], "correct": True, "max_rel_err": 0.0} for row in f.case_rows]
                     + [{"case": "output_independence", "correct": True, "max_rel_err": None}],
            "random": [{"case": "random[0]:shape", "correct": True, "max_rel_err": 0.0}],
            "compile_parity": [{"case": "compile_parity", "correct": True, "max_rel_err": None,
                                "note": "compile_soft_degrade (NON-FATAL): synthetic"}]}})
        policy = {"max_reps_requested": 3, "max_reps_effective": 3, "undecided": [0.95, 1.1],
                  "child_timer_reducer": "upper_middle", "replicate_reducer": "median",
                  "pair_order": ["baseline", "candidate"]}
        measurement = {"measurement_id": "100:1", "task_dir": self.task_dir, "policy": policy,
                       "buckets": [], "per_case": [], "status": "started", "error": None}
        self.counter = 1
        self.emit("measurement_start", measurement)
        for role in ("baseline", "candidate"):
            self.control_leg("resolve", role, {"module": "serving.attention", "qualname": "run",
                "file": "/opt/example/serving/attention.py" if role == "baseline"
                        else self.task_dir + "/_cand_overlay/serving/attention.py"})
        self.control_leg("list", "baseline", {"sigs": [row["signature"] for row in f.case_rows]})
        for index, case in enumerate(f.case_rows):
            bms, cms = ([1.0, 2.4], [1.0, 1.1]) if index == 0 else ([4.0], [2.0])
            bucket = {"sig": case["signature"], "pairs": [], "baseline_ms": bms,
                      "candidate_ms": cms, "stop_reason": "outside_undecided"}
            for n, (bm, cm) in enumerate(zip(bms, cms), 1):
                bid = self.leg(case["signature"], "baseline", bm)
                cid = self.leg(case["signature"], "candidate", cm)
                bucket["pairs"].append({"replicate": n, "baseline_leg_id": bid, "candidate_leg_id": cid})
            measurement["buckets"].append(bucket)
            measurement["per_case"].append({"sig": case["signature"], "regime": "prefill", "m": 1,
                **READER.summarize_replicates(bms, cms)})
        measurement["status"] = "ok"
        self.emit("measurement", measurement)

    def emit(self, kind, payload):
        self.records.append({"schema": "geak-extracted-measurement-evidence-v1", "kind": kind,
            "parent_pid": self.parent_pid, "sequence": len(self.records) + 1,
            "monotonic_ns": (len(self.records) + 1) * 100, "payload": copy.deepcopy(payload)})

    def leg(self, sig, role, ms):
        self.counter += 1
        leg_id = "100:" + str(self.counter)
        prefix = self.task_dir + ("/baseline_overlay" if role == "baseline" else "/_cand_overlay")
        payload = {"leg_id": leg_id, "task_dir": self.task_dir, "overlay": prefix, "mode": "time", "bucket": sig,
            "seed": 0, "draws": 0, "argv": ["/usr/bin/python3", self.task_dir + "/leg_runner.py", "--task",
                self.task_dir, "--mode", "time", "--seed", "0", "--bucket", sig], "child_pid": None,
            "timeout_s": 3600, "status": "started", "returncode": None, "stdout": None, "stderr": None,
            "result": None, "error": None}
        self.emit("leg_start", payload)
        cache = {"mode": "read_evict", "bytes": 536870912, "deployment_calibrated": False}
        detail = {"ms": ms, "wall_ms": ms + 1, "timer": "cuda_event", "cache_condition": cache,
                  "host_ms": 0.01, "primed": True}
        timing = {"requested": {"warmup": 10, "repeats": 50, "inner": 1, "graph": False},
            "effective": {"warmup": 10, "repeats": 50, "inner": 1}, "timer": "cuda_event",
            "device_ms_samples": [ms] * 50, "wall_ms_samples": [ms + 1] * 50, "status": "ok", "error": None,
            "graph_capture": {"status": "not_requested", "error": None}}
        row = {"sig": sig, "regime": "prefill", "m": 1, **{key: detail[key] for key in ("ms", "wall_ms", "timer", "cache_condition")},
               "timing_detail": detail, "timing_evidence": timing,
               "compile_evidence": {"requested": False, "compile_error": None}}
        file = "/opt/example/serving/attention.py" if role == "baseline" else self.task_dir + "/_cand_overlay/serving/attention.py"
        sha = "c" * 64 if role == "baseline" else next(iter(self.sources.values()))
        result = {"cases": [row], "identity": {"module": "serving.attention", "file": file, "qualname": "run"},
            "identity_file_observation": {"path": file, "sha256": sha, "status": "ok", "error": None,
                "claim": "resolved_file_bytes_observed_after_timing_not_loaded_code_proof"}}
        payload.update(child_pid=1000 + self.counter, status="ok", returncode=0, stdout=json.dumps(result) + "\n",
                       stderr="", result=result)
        self.emit("leg", payload)
        return leg_id

    def control_leg(self, mode, role, result):
        self.counter += 1
        prefix = self.task_dir + ("/baseline_overlay" if role == "baseline" else "/_cand_overlay")
        payload = {"leg_id": "100:" + str(self.counter), "task_dir": self.task_dir, "overlay": prefix,
            "mode": mode, "bucket": "", "seed": 0, "draws": 0,
            "argv": ["/usr/bin/python3", self.task_dir + "/leg_runner.py", "--task", self.task_dir,
                     "--mode", mode, "--seed", "0"], "child_pid": None, "timeout_s": 600,
            "status": "started", "returncode": None, "stdout": None, "stderr": None, "result": None, "error": None}
        self.emit("leg_start", payload)
        payload.update(child_pid=1000 + self.counter, status="ok", returncode=0,
                       stdout=json.dumps(result) + "\n", stderr="", result=result)
        self.emit("leg", payload)

    def rewrite_child_stdout(self):
        for record in self.records:
            if record["kind"] == "leg" and record["payload"]["result"] is not None:
                record["payload"]["stdout"] = json.dumps(record["payload"]["result"]) + "\n"

    def write(self):
        for name, value in (("identity.json", self.identity), ("evaluate/protocol-binding.json", self.binding),
                            ("evaluate/launch-settings.json", self.launch)):
            (self.root / name).write_bytes(READER.canonical(value) + b"\n")
        stdout = b"ordinary retained log\n" + b"".join(self.contract["report_contract"]["stdout_prefix"].encode()
                      + READER.canonical(row) + b"\n" for row in self.records)
        (self.phase_dir / "stdout.bin").write_bytes(stdout)
        (self.phase_dir / "stderr.bin").write_bytes(b"diagnostic stderr\n")
        self.phase.update(protocol_binding_sha256=READER.digest((self.phase_dir / "protocol-binding.json").read_bytes()),
            launch_settings_ref={"path": "launch-settings.json", "sha256": READER.digest((self.phase_dir / "launch-settings.json").read_bytes())},
            stdout_sha256=READER.digest(stdout), stderr_sha256=READER.digest((self.phase_dir / "stderr.bin").read_bytes()))
        (self.phase_dir / "phase.json").write_bytes(READER.canonical(self.phase) + b"\n")

    def retained(self):
        return {path.relative_to(self.root).as_posix(): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def observe(self):
        self.write()
        return READER.read_generated_report(self.contract, self.retained(), self.sources)


class PairedObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.f = ObservationFixture(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def leg(self):
        return next(row["payload"] for row in self.f.records if row["kind"] == "leg" and row["payload"]["mode"] == "time")

    def test_full_pair_observation_matches_existing_arena_shape(self):
        out = self.f.observe()
        self.assertEqual(out["issues"], [])
        self.assertEqual(set(out["observations"]), {"correctness", "benchmark"})
        self.assertEqual(out["observations"]["correctness"]["status"], "passed")
        self.assertEqual(out["observations"]["benchmark"]["times_ms"], {"batch=1,heads=16": 1.05, "batch=4,heads=16": 2.0})
        self.assertEqual(out["paired_score"]["parent_evaluation"], self.f.parent)
        self.assertAlmostEqual(out["paired_score"]["geomean_speedup"], ((1.7 / 1.05) * 2) ** 0.5)
        self.assertIn("evaluate/stdout.bin", out["artifacts"])
        self.assertTrue(out["phase_validated"]["evaluate"])
        self.assertFalse(out["measurement_valid"] or out["custody_proven"] or out["reward_eligible"])

    def test_paired_score_artifact_has_explicit_roles_without_self_reference(self):
        out = self.f.observe()
        score = out["paired_score"]
        self.assertEqual(out["artifacts"]["paired-score.json"], READER.canonical(score) + b"\n")
        self.assertEqual(json.loads(out["artifacts"]["paired-score.json"]), score)
        self.assertNotIn("evidence", score)
        self.assertIn("paired-score.json", out["observations"]["benchmark"]["evidence"])
        self.assertIn("paired-score.json", out["observations"]["correctness"]["evidence"])
        self.assertEqual(score["baseline_source_id"], self.f.parent["binding"]["candidate_source_id"])
        self.assertEqual(score["candidate_source_id"], self.f.parent["binding"]["candidate_source_id"])
        self.assertEqual(score["baseline_closure_sha256"], self.f.contract["artifacts"]["baseline_closure.json"])
        self.assertEqual(score["job_id"], READER.digest(READER.canonical(self.f.contract)))
        self.assertEqual(score["comparison_contract_sha256"], READER.digest(READER.source_canonical(self.f.contract)))
        self.assertEqual(score["runtime"], self.f.contract["runtime"])
        self.assertEqual(score["case_bindings"], self.f.contract["case_bindings"])

    def test_failure_still_retains_derived_companion_with_no_score(self):
        self.f.phase["returncode"] = 1
        out = self.f.observe()
        self.assert_no_score(out)
        score = json.loads(out["artifacts"]["paired-score.json"])
        self.assertIsNone(score["geomean_speedup"])
        self.assertEqual(score["baseline_source_id"], out["paired_score"]["baseline_source_id"])

    def assert_no_score(self, out):
        self.assertNotEqual(out["observations"]["benchmark"]["status"], "passed")
        self.assertEqual(out["observations"]["benchmark"]["times_ms"], {})
        self.assertIsNone(out["paired_score"]["geomean_speedup"])
        self.assertFalse(out["phase_validated"]["evaluate"])
        self.assertTrue(out["issues"])

    def test_forged_parent_pid_and_sequence_are_rejected(self):
        for field, value in (("parent_pid", 999), ("sequence", 99), ("monotonic_ns", 1)):
            original = self.f.records[1][field]
            self.f.records[1][field] = value
            self.assert_no_score(self.f.observe())
            self.f.records[1][field] = original

    def test_missing_or_duplicate_case_cannot_pass(self):
        measurement = self.f.records[-1]["payload"]
        measurement["buckets"][1]["sig"] = measurement["buckets"][0]["sig"]
        self.assert_no_score(self.f.observe())

    def test_reused_or_reversed_children_cannot_pass(self):
        pair = self.f.records[-1]["payload"]["buckets"][0]["pairs"][0]
        pair["baseline_leg_id"], pair["candidate_leg_id"] = pair["candidate_leg_id"], pair["baseline_leg_id"]
        self.assert_no_score(self.f.observe())

    def test_missing_sample_and_forged_scalar_cannot_pass(self):
        leg = self.leg()
        leg["result"]["cases"][0]["timing_evidence"]["device_ms_samples"].pop()
        self.f.rewrite_child_stdout()
        out = self.f.observe()
        self.assert_no_score(out)
        self.assertEqual(out["observations"]["correctness"]["status"], "passed")
        self.assertEqual(len(out["paired_score"]["raw_records"]), len(self.f.records))

    def test_aggregate_tampering_is_recomputed(self):
        self.f.records[-1]["payload"]["per_case"][0]["speedup"] = 999.0
        self.assert_no_score(self.f.observe())

    def test_child_decoded_result_must_match_raw_stdout(self):
        self.leg()["result"]["cases"][0]["ms"] = 2.0
        self.assert_no_score(self.f.observe())

    def test_file_identity_must_match_independent_closure(self):
        self.leg()["result"]["identity_file_observation"]["sha256"] = "0" * 64
        self.f.rewrite_child_stdout()
        self.assert_no_score(self.f.observe())

    def test_candidate_source_parent_binding_cannot_be_reused(self):
        self.f.parent["binding"]["candidate_source_id"] = "0" * 64
        self.assert_no_score(self.f.observe())

    def test_current_candidate_executable_mode_is_bound_to_actual_source_id(self):
        name = self.f.contract["source_paths"][0]
        self.f.identity["task_file_manifest"][name]["executable"] = 0o111
        (self.f.generated.artifact_root / name).chmod(0o755)
        manifest = {"kind": "source", "files": [{"path": name, "blob_id": self.f.sources[name], "executable": 0o111}]}
        self.f.parent["binding"]["candidate_source_id"] = READER.digest(READER.source_canonical(manifest))
        out = self.f.observe()
        self.assertEqual(out["issues"], [])
        self.assertEqual(out["paired_score"]["candidate_source_id"], self.f.parent["binding"]["candidate_source_id"])
        self.assertNotEqual(out["paired_score"]["baseline_source_id"], out["paired_score"]["candidate_source_id"])

    def test_missing_resolve_or_wrong_list_cannot_pass(self):
        record = next(row for row in self.f.records if row["kind"] == "leg" and row["payload"]["mode"] == "list")
        record["payload"]["result"]["sigs"].reverse()
        self.f.rewrite_child_stdout()
        self.assert_no_score(self.f.observe())

    def test_resolve_identity_must_match_timed_identity(self):
        record = next(row for row in self.f.records if row["kind"] == "leg" and row["payload"]["mode"] == "resolve")
        record["payload"]["result"]["qualname"] = "different"
        self.f.rewrite_child_stdout()
        self.assert_no_score(self.f.observe())

    def test_overlay_diagnostics_are_recognized_and_retained(self):
        extra = []
        for kind, status in (("overlay_build_start", "started"), ("overlay_build", "ok")):
            extra.append({"schema": "geak-extracted-measurement-evidence-v1", "kind": kind, "parent_pid": 100,
                "sequence": 0, "monotonic_ns": 0, "payload": {"build_id": "100:999", "task_dir": self.f.task_dir,
                    "argv": ["overlay_setup.py"], "status": status, "returncode": 0 if status == "ok" else None,
                    "stdout": "full build output" if status == "ok" else None, "stderr": "", "error": None}})
        self.f.records[2:2] = extra
        for index, row in enumerate(self.f.records, 1):
            row["sequence"] = index
            row["monotonic_ns"] = index * 100
        out = self.f.observe()
        self.assertEqual(out["issues"], [])
        self.assertEqual(out["paired_score"]["raw_records"][3]["payload"]["stdout"], "full build output")

    def test_failed_correctness_retains_full_raw_report_without_score(self):
        self.f.records[0]["payload"]["ok"] = False
        out = self.f.observe()
        self.assert_no_score(out)
        self.assertEqual(out["observations"]["correctness"]["status"], "failed")
        self.assertFalse(out["paired_score"]["raw_records"][0]["payload"]["ok"])

    def test_partial_leg_is_retained_without_score(self):
        self.leg().update(status="timeout", returncode=-15, error={"type": "TimeoutExpired", "message": "fixture"})
        out = self.f.observe()
        self.assert_no_score(out)
        self.assertIn(b'TimeoutExpired', out["artifacts"]["evaluate/stdout.bin"])

    def test_native_failure_retains_parseable_failure_records(self):
        self.f.phase["returncode"] = 1
        out = self.f.observe()
        self.assert_no_score(out)
        self.assertEqual(len(out["paired_score"]["raw_records"]), len(self.f.records))

    def test_native_stream_tampering_is_rejected(self):
        self.f.write()
        (self.f.phase_dir / "stdout.bin").write_bytes(b"changed")
        self.assert_no_score(READER.read_generated_report(self.f.contract, self.f.retained(), self.f.sources))

    def test_producer_incomplete_evidence_diagnostic_precludes_score(self):
        self.f.write()
        raw = b"GEAK_MEASUREMENT_EVIDENCE_INCOMPLETE OSError\n"
        (self.f.phase_dir / "stderr.bin").write_bytes(raw)
        self.f.phase["stderr_sha256"] = READER.digest(raw)
        (self.f.phase_dir / "phase.json").write_bytes(READER.canonical(self.f.phase))
        out = READER.read_generated_report(self.f.contract, self.f.retained(), self.f.sources)
        self.assert_no_score(out)
        self.assertIn(raw, out["artifacts"]["evaluate/stderr.bin"])

    def test_changed_protected_task_source_rejected(self):
        (self.f.generated.artifact_root / "cases.py").write_bytes(b"changed")
        self.assert_no_score(self.f.observe())

    def test_soft_compile_degrade_remains_visible_and_not_new_failure(self):
        out = self.f.observe()
        self.assertEqual(out["observations"]["correctness"]["status"], "passed")
        self.assertIn("compile_soft_degrade", out["paired_score"]["raw_records"][0]["payload"]["report"]["compile_parity"][0]["note"])

    def test_wall_fallback_preserves_actual_timer_and_absent_primed(self):
        for record in self.f.records:
            if record["kind"] != "leg":
                continue
            if record["payload"]["mode"] != "time":
                continue
            row = record["payload"]["result"]["cases"][0]
            row["timer"] = row["timing_detail"]["timer"] = row["timing_evidence"]["timer"] = "wall"
            row["wall_ms"] = row["timing_detail"]["wall_ms"] = row["ms"]
            row["timing_detail"].pop("host_ms"); row["timing_detail"].pop("primed")
            row["timing_evidence"]["device_ms_samples"] = []
            row["timing_evidence"]["wall_ms_samples"] = [row["ms"]] * 50
        self.f.rewrite_child_stdout()
        out = self.f.observe()
        self.assertEqual(out["issues"], [])
        detail = out["paired_score"]["per_case"][0]["pairs"][0]["baseline"]["timing_detail"]
        self.assertEqual(detail["timer"], "wall")
        self.assertNotIn("primed", detail)

    def test_graph_capture_fallback_does_not_create_a_new_failure(self):
        row = self.leg()["result"]["cases"][0]
        row["timing_evidence"]["requested"]["graph"] = True
        row["timing_evidence"]["graph_capture"] = {"status": "error", "error": {"type": "RuntimeError", "message": "capture"}}
        self.assertEqual(READER._timing(row, True), row["ms"])

    def test_zero_raw_sample_does_not_discard_a_positive_timer_reduction(self):
        row = self.leg()["result"]["cases"][0]
        row["timing_evidence"]["device_ms_samples"][0] = 0.0
        self.f.rewrite_child_stdout()
        self.assertEqual(self.f.observe()["issues"], [])


if __name__ == "__main__":
    unittest.main()
