"""CPU integration tests: union -> shared receipts -> per-config costing -> merge/select."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SLOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SLOT))
import multi_config_sampling as S

IDENTITY = {"image": "test@sha256:abc", "model_revision": "model-sha",
            "source_revision": "source-sha", "gpu_arch": "gfx950"}
SERVING = {"model": "test", "tp": 4, "gpus": "0,1,2,3", "port": 9000,
           "args": ["--chunked-prefill-size", "8192", "--nccl-port", "1234"],
           "decode_capture": "profile_cuda_graph"}


def obs(batches, contexts=None):
    return {"decode_log_interval": 1, "raw_prefill_rows": [],
            "raw_decode_rows": [{"running": b, "token": b * (contexts or {}).get(b, 1024),
                                 "iterations": 1, "cuda_graph": True} for b in batches]}


def config(cid, batches, **kwargs):
    return dict({"config_id": cid, "serving": copy.deepcopy(SERVING),
                 "runtime_identity": IDENTITY, "observed": obs(batches)}, **kwargs)


class SharedSampling(unittest.TestCase):
    def test_union_keeps_middle_only_work_and_does_not_duplicate_overlap(self):
        configs = [config("low", [1, 2]), config("middle", [2, 3]), config("high", [2, 4])]
        result = S.plan(configs, 20, target_pct=100)
        self.assertEqual(result["candidate_points_before_union"], 6)
        self.assertEqual(result["capture_count"], 4)
        self.assertEqual({c["point"]["batch_size"] for c in result["captures"]}, {1, 2, 3, 4})
        self.assertEqual(len(result["groups"]), 1)
        for cfg in result["configs"].values():
            self.assertEqual(cfg["phases"]["DECODE"]["planned_proxy_coverage_pct"], 100)

    def test_normalizing_configs_avoids_long_log_domination(self):
        a = S.plan([config("a", [1, 1, 2]), config("b", [3])], 1)
        b = S.plan([config("a", [1, 1, 2] * 100), config("b", [3])], 1)
        self.assertEqual(a["captures"][0]["point_id"], b["captures"][0]["point_id"])
        self.assertEqual(a["captures"][0]["point"]["batch_size"], 3)

    def test_transport_changes_share_but_scheduler_and_runtime_do_not(self):
        a, b = config("a", [1]), config("b", [1])
        b["serving"].update(port=9001, gpus="4,5,6,7", conc=4, _note="different host")
        b["serving"]["args"][-1] = "2345"
        self.assertEqual(S.plan([a, b], 10)["capture_count"], 1)
        b["serving"]["args"] += ["--max-running-requests", "20"]
        self.assertEqual(S.plan([a, b], 10)["capture_count"], 2)
        b = config("b", [1], runtime_identity=dict(IDENTITY, source_revision="changed"))
        self.assertEqual(len(S.plan([a, b], 10)["groups"]), 2)

    def test_context_cells_are_bounded_and_discrete_modes_are_not_merged(self):
        a = config("a", [2], observed=obs([2], {2: 1000}))
        b = config("b", [2], observed=obs([2], {2: 1001}))
        self.assertEqual(S.plan([a, b], 10)["capture_count"], 1)
        self.assertEqual(S.plan([a, b], 10, context_tolerance=1)["capture_count"], 2)
        b["observed"]["raw_decode_rows"][0]["cuda_graph"] = False
        self.assertEqual(S.plan([a, b], 10)["capture_count"], 2)
        b["observed"]["raw_decode_rows"][0].update(cuda_graph=True, token=100000)
        self.assertEqual(S.plan([a, b], 10)["capture_count"], 2)

    def test_budget_extension_keeps_first_captures_and_exposes_missing_mass(self):
        configs = [config("a", [1, 2, 3, 4])]
        small, large = S.plan(configs, 1), S.plan(configs, 4)
        self.assertEqual(small["status"], "budget_limited")
        self.assertEqual(len(small["unselected"]), 3)
        self.assertEqual(small["captures"][0]["point_id"], large["captures"][0]["point_id"])
        self.assertEqual(large["status"], "planned")

    def test_single_config_and_invalid_inputs(self):
        self.assertEqual(S.plan([config("only", [1])], 1)["capture_count"], 1)
        for budget in (0, -1, True):
            with self.assertRaises(ValueError):
                S.plan([config("only", [1])], budget)
        with self.assertRaises(ValueError):
            S.plan([config("a", [1], runtime_identity={})], 2)
        with self.assertRaises(ValueError):
            S.plan([config("a", [1]), config("a", [2])], 2)


class Projection(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        items = []
        for cid, batches in (("a", [1] * 9 + [2]), ("b", [1] + [2] * 9)):
            d = self.root / "inputs" / cid
            S.write(d / "serving.json", SERVING)
            S.write(d / "evidence/observed_workload.json", obs(batches))
            items.append((cid, str(d)))
        self.configs = S.load_configs(items, IDENTITY)
        self.plan = S.plan(self.configs, 4, target_pct=100)
        S.write_plan(self.root, self.plan, self.configs)
        self.pool = Path(next(iter(self.plan["groups"].values()))["eval_dir"])
        self.make_measurements()

    def make_measurements(self):
        timeline = {"points": {}}
        for c in self.plan["captures"]:
            p = c["point"]
            pid, b = p["point_id"], p["batch_size"]
            S.write(self.pool / "evidence/census" / (pid + ".json"), {
                "point_id": pid, "phase": "DECODE", "achieved_shape": b, "instances": 1,
                "per_kernel": {"kernel_a" if b == 1 else "kernel_b": 10 if b == 1 else 20}})
            S.write(self.pool / "evidence/probe" / (pid + ".json"), {"status": "ok"})
            timeline["points"][pid] = {"measurement_signature": p["measurement_signature"],
                                       "capture_mode": "profile_cuda_graph", "phase": "DECODE"}
        S.write(self.pool / "profile_run.json", timeline)
        S.write(self.pool / "evidence/source_map.json", {"kernels": {}})

    def cli(self, *args):
        p = subprocess.run([sys.executable, "-B", str(SLOT / "multi_config.py"), *args],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p

    def test_real_projection_merge_and_selection_keep_config_frequencies(self):
        self.cli("project", "--group-dir", str(self.root))
        a = S.read(self.root / "projected/a/kernel_ranking.json")
        b = S.read(self.root / "projected/b/kernel_ranking.json")
        self.assertEqual(a["denominators"]["total_device_us"], 110)
        self.assertEqual(b["denominators"]["total_device_us"], 190)
        self.assertTrue((self.root / "projected/a/evidence/census").is_symlink())
        final = S.read(self.root / "projected/a/sampling_points_final.json")
        self.assertEqual(sum(p["represented"]["instances"] for p in final["points"]), 10)
        self.assertEqual(final["phase_summary"]["DECODE"]["instances_outside_tolerance"], 0)
        merged = self.root / "merged"
        self.cli("merge", "--group-dir", str(merged), "--config", "a=" + str(self.root / "projected/a"),
                 "--config", "b=" + str(self.root / "projected/b"))
        params = self.root / "params.json"
        S.write(params, {"selection_overall_pct_gt": 3, "selection_top_k": 10})
        self.cli("select", "--group-dir", str(merged), "--params", str(params))
        self.assertEqual(S.read(merged / "selection_targets.json")["target_count"], 2)
        # Idempotent projection uses the same raw measurements, never copies them.
        S.project(self.root)

    def test_changed_input_or_unsigned_measurement_is_rejected(self):
        run = S.read(self.pool / "profile_run.json")
        first = next(iter(run["points"].values()))
        first.pop("measurement_signature")
        S.write(self.pool / "profile_run.json", run)
        with self.assertRaisesRegex(ValueError, "measurement signature"):
            S.project(self.root)
        self.make_measurements()
        S.write(self.root / "inputs/a/serving.json", dict(SERVING, tp=8))
        with self.assertRaisesRegex(ValueError, "input changed"):
            S.project(self.root)

    def test_missing_phase_does_not_create_a_zero_time_ranking(self):
        S.project(self.root)
        for path in (self.pool / "evidence/census").glob("*.json"):
            path.unlink()
        result = S.project(self.root)
        self.assertEqual(result["status"], "missing_measurements")
        self.assertFalse((self.root / "projected/a/kernel_ranking.json").exists())
        self.assertTrue((self.root / "projected/a/kernel_ranking.previous.json").exists())

    def test_unsigned_pool_cannot_resume_as_compatible(self):
        S.write(self.pool / "profile_run.json", {"points": {}})
        p = subprocess.run([sys.executable, "-B", str(S.SCRIPTS / "run_points.py"),
                            "--eval-dir", str(self.pool), "--points", str(self.pool / "sampling_points.json"),
                            "--base-url", "http://127.0.0.1:1", "--trace-root", str(self.root / "traces"),
                            "--expect-files", "1", "--no-evidence"], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("unsigned shared census", p.stderr)

    def test_capture_dry_run_groups_deployments_without_starting_servers(self):
        p = self.cli("capture", "--group-dir", str(self.root), "--dry-run")
        commands = json.loads(p.stdout)
        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0]["argv"][1].endswith("/gpu_lock_set.py"))
        self.assertIn("--all-evidence", commands[0]["argv"])
        self.assertFalse((self.pool / "probe").exists())

    def test_graph_mode_gap_is_not_credited_as_same_shape_coverage(self):
        cfg = S.read(self.root / "inputs/a/evidence/observed_workload.json")
        cfg["raw_decode_rows"][0]["cuda_graph"] = False
        final = S.relabel(cfg, S.read(self.pool / "sampling_points.json"),
                          str(self.pool / "evidence/census"), str(self.pool / "evidence/probe"), 1.25, 1.5)
        summary = final["phase_summary"]["DECODE"]
        self.assertEqual(summary["instances_outside_sampled_partition"], 1)
        self.assertEqual(summary["instances_outside_tolerance"], 1)
        self.assertEqual(sum(p["represented"]["instances"] for p in final["points"]), 10)


if __name__ == "__main__":
    unittest.main()
