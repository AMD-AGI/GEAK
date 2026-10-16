"""Current Arena/export composition with producer-shaped synthetic raw evidence.

No task Python, model, GPU or live operational store is read or executed.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from geak_evo import result
from geak_evo.archive.store import ResearchStore
from geak_evo.evaluate.evaluator import ArenaEvaluator, canonical, sha256
from geak_evo.evaluate.generated_report import read_generated_report
from geak_evo.evaluate.jobs import job_digest
from test_generated_task_contract import GeneratedFixture
from test_generated_report import ObservationFixture


class GeneratedResultTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fixture = f = GeneratedFixture(self.root)
        self.contract, self.task = f.contract, f.artifact_root
        self.job_id = job_digest(self.contract)
        self.store = ResearchStore(self.root / "store", allowed_epochs=[(self.job_id, "epoch")],
                                   source_paths={self.job_id: self.contract["source_paths"]})
        self.original_source = self.store.capture_source(self.task, self.contract["source_paths"])
        self.original = self.store.candidate(self.job_id, "epoch", self.original_source)
        work = self.root / "candidate"; (work / "kernel_src").mkdir(parents=True)
        (work / "kernel_src/attention.py").write_bytes(b"# synthetic changed source, never executed\n")
        self.selected_source = self.store.capture_source(work, self.contract["source_paths"])
        self.selected = self.store.candidate(self.job_id, "epoch", self.selected_source)
        self.engine = ArenaEvaluator(self.store, contract=self.contract, contract_id=self.job_id,
            artifact_root=self.task, output_root=self.root / "evaluations", measurement_epoch_id="epoch",
            launch=lambda request, operation: "synthetic-handle", settle=self.settle)
        self.choice = {"candidate_id": self.selected, "source_id": self.selected_source,
                       "binding": {"campaign_id": "campaign", "job_id": self.job_id, "measurement_epoch_id": "epoch"}}
        self.selection = {"status": "selected", "choice": self.choice, "record": {"disposition": "set"}}
        self.snapshot = {"campaign_id": "campaign", "revision": 3, "state": {"phase": "blocked"}}
        self.change_raw = lambda role, fixture: None
        self.change_host = lambda role, host, artifacts: None
        self.index = 0

    def settle(self, handle, request, operation):
        directory = operation / "synthetic-observer"; directory.mkdir()
        f = ObservationFixture(directory)
        f.contract = request["contract"]
        f.sources = {row["path"]: row["blob_id"] for row in request["source"]["files"]}
        for row in request["source"]["files"]:
            (f.generated.artifact_root / row["path"]).write_bytes(
                (self.store.root / "objects" / row["blob_id"]).read_bytes())
            f.identity["task_file_manifest"][row["path"]] = {"sha256": row["blob_id"], "executable": row["executable"]}
        f.parent = {"operation_id": request["operation_id"], "request_sha256": sha256(canonical(request)),
                    "binding": copy.deepcopy(request["binding"])}
        f.identity["generated_protocol"]["contract"] = request["contract"]
        f.identity["generated_evaluation_binding"] = f.parent
        f.binding.update(parent_evaluation=f.parent, contract_sha256=self.job_id, source_sha256=f.sources)
        for record in f.records:
            if record["kind"] == "leg" and record["payload"]["mode"] == "time":
                leg = record["payload"]
                if leg["overlay"].endswith("/_cand_overlay"):
                    leg["result"]["identity_file_observation"]["sha256"] = next(iter(f.sources.values()))
        role = "baseline" if request["binding"]["candidate_id"] == self.original else "candidate"
        self.change_raw(role, f)
        f.rewrite_child_stdout(); f.write()
        observed = read_generated_report(request["contract"], f.retained(), f.sources)
        self.assertEqual(observed["issues"], [])
        artifacts = {"observer/" + name: raw for name, raw in observed["artifacts"].items()}
        artifacts.update({"host/cleanup.json": b"{}", "host/runtime.json": b"{}"})
        groups = copy.deepcopy(observed["observations"])
        for group in groups.values():
            group["evidence"] = ["observer/" + name for name in group["evidence"]]
        host = {"schema": "geak-arena-settlement-v1", "operation_id": request["operation_id"],
            "request_sha256": sha256(canonical(request)), "binding": request["binding"],
            "execution_started": True, "cleanup": {"settled": True, "evidence": ["host/cleanup.json"]},
            "runtime": {**self.contract["runtime"], "physical_devices": ["0000:01:00.0"], "evidence": ["host/runtime.json"]},
            "phase_exit_codes": {"evaluate": 0}, "observations": groups,
            "costs": {"wall_seconds": 1.0, "evaluations": 1}, "errors": []}
        self.change_host(role, host, artifacts)
        host["artifacts_sha256"] = {name: sha256(raw) for name, raw in artifacts.items()}
        return {"receipt": host, "artifacts": artifacts}

    def measure(self):
        self.engine.evaluate(self.original, evaluation_id="baseline")
        self.engine.evaluate(self.selected, evaluation_id="selected")

    def export(self, **options):
        self.index += 1
        with mock.patch.object(result, "validate_final_recommendation", return_value=self.selection):
            return result.export_final_result(self.store, self.snapshot, evaluator=self.engine,
                selection_options={}, baseline_candidate_id=self.original, baseline_evaluation_id="baseline",
                selected_evaluation_id="selected", output_dir=self.root / ("export-" + str(self.index)), **options)

    def test_internal_pair_is_used_instead_of_two_candidate_side_measurements(self):
        # Both operations intentionally have identical candidate-side times.
        # Dividing those times would falsely yield 1.0 rather than the real pair.
        self.measure()
        output = self.export()
        self.assertEqual(output["outcome"], "improved", output)
        self.assertAlmostEqual(output["final_geomean"], ((1.7 / 1.05) * 2) ** 0.5)
        self.assertEqual(output["per_case"][0]["baseline_ms"], 1.7)
        self.assertEqual(output["per_case"][0]["candidate_ms"], 1.05)
        self.assertEqual(output["timing_basis"], "generated_task_paired")
        self.assertEqual(output["comparison_basis"], "selected_evaluation_internal_baseline_candidate_pair")
        self.assertTrue(Path(output["final_patch"]).is_file())
        self.assertTrue(all(output[key] is False for key in
            ("measurement_valid", "custody_proven", "reward_eligible", "framework_final_pass")))
        context = output["timing_context"]["candidate"]
        self.assertEqual(context["paired_cases"]["case-a"]["reps"], 2)
        self.assertIn("observer/paired-score.json", context["evidence"])

    def test_weighted_metric_uses_same_internal_pair(self):
        self.measure()
        output = self.export(objective={"kind": "time_weighted", "weights": {"case-a": 3, "case-b": 1}})
        self.assertAlmostEqual(output["final_speedup"], 4 / (3 / (1.7 / 1.05) + 1 / 2))

    def test_unindexed_evidence_cannot_supply_a_result(self):
        def change(role, host, artifacts):
            if role == "candidate":
                artifacts.pop("observer/evaluate/stdout.bin")
                for group in host["observations"].values():
                    group["evidence"].remove("observer/evaluate/stdout.bin")
        self.change_host = change
        self.measure()
        output = self.export()
        self.assertEqual(output["outcome"], "error")
        self.assertIsNone(output["final_speedup"])
        self.assertEqual(output["final_patch"], "")

    def test_consistent_report_with_another_parent_request_is_rejected(self):
        def change(role, fixture):
            if role == "candidate":
                fixture.parent["request_sha256"] = "0" * 64
        self.change_raw = change
        self.measure()
        output = self.export()
        self.assertEqual(output["outcome"], "error")
        self.assertIn("another owner request", output["error"])
        self.assertIsNone(output["final_speedup"])

    def test_admitted_command_runs_existing_evaluator_and_exports_generated_pairs(self):
        from types import SimpleNamespace
        import time
        from geak_evo import command
        from geak_evo.archive.refs import file_ref
        from geak_evo.host import HostServices
        from geak_evo.model_profile import ModelProfile
        from test_command import BudgetInterface

        def save(name, value):
            path = self.root / name; path.write_bytes(canonical(value))
            return file_ref(path)
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        clock = {"boot_id": boot, "session_id": "fixture", "started_monotonic_ns": time.monotonic_ns(), "wall_seconds": 3600}
        budget = BudgetInterface({"boot_id": boot, "session_id": "fixture",
                                  "deadline_ns": clock["started_monotonic_ns"] + 3600_000_000_000})
        config = {"schema": "geak-evo-admitted-command-v1", "baseline_candidate_id": self.original,
            "objective": {"kind": "geomean"}, "final_measurements": {"baseline_seconds": 60, "selected_seconds": 60},
            "search": {"campaign_id": "campaign", "seed_candidate_id": self.original,
                "campaign_root": str(self.root), "shared_budget": budget.config, "clock_ref": save("clock.json", clock),
                "profile": ModelProfile("codex", "gpt-6-astra", "xhigh", str(self.root / "route.json"), "a" * 64).as_dict(),
                "templates": {"config": save("native.json", {"contract_ref": save("contract.json", self.contract),
                    "artifact_root": str(self.task), "evaluation_root": str(self.engine.root),
                    "backend_config": {"probe_config": {"job_id": self.job_id, "measurement_epoch_id": "epoch",
                                                        "native_envelope_seconds": 60}}})}}}
        snapshot = {"campaign_id": "campaign", "revision": 1, "state": {"phase": "idle"}}
        self.store.checkpoint("campaign", self.original, expected_revision=0,
            note={"claim": "fixture", "scope": "CPU only", "next_test": "fixture", "evolution": snapshot["state"]})
        search = SimpleNamespace(run=mock.Mock(return_value=snapshot))
        with mock.patch.object(command, "prepare_native_search", return_value=search), \
             mock.patch.object(command, "validate_final_recommendation", return_value=self.selection), \
             mock.patch.object(result, "validate_final_recommendation", return_value=self.selection):
            output = command.run_admitted_task(save("owner.json", config), store=self.store,
                services=HostServices(SimpleNamespace(), None, None, None, None, {}, {}),
                evaluator=self.engine, budget=budget, episode=lambda *args: None,
                output_dir=self.root / "command-result", max_steps=1)
        self.assertEqual(output["outcome"], "improved", output)
        self.assertEqual(output["timing_basis"], "generated_task_paired")
        self.assertEqual(len(budget.operations), 2)
        self.assertTrue(all(row["status"] == "settled" for row in budget.operations.values()))

    def test_forged_derived_pair_rejected_even_with_updated_envelope_hash(self):
        def change(role, host, artifacts):
            if role == "candidate":
                pair = json.loads(artifacts["observer/paired-score.json"])
                pair["per_case"][0]["baseline_ms"] = 999
                artifacts["observer/paired-score.json"] = canonical(pair)
        self.change_host = change
        self.measure()
        output = self.export()
        self.assertEqual(output["outcome"], "error")
        self.assertIn("generated retained artifact differs", output["error"])

    def test_mixed_cache_is_retained_but_not_exported_as_comparable(self):
        def change(role, fixture):
            if role == "candidate":
                leg = next(row["payload"] for row in fixture.records if row["kind"] == "leg"
                           and row["payload"]["mode"] == "time")
                row = leg["result"]["cases"][0]
                row["cache_condition"]["mode"] = "write_evict"
        self.change_raw = change
        self.measure()
        output = self.export()
        self.assertEqual(output["outcome"], "error")
        self.assertIn("actual timer or cache conditions differ", output["error"])
        self.assertIsNone(output["final_speedup"])


if __name__ == "__main__":
    unittest.main()
