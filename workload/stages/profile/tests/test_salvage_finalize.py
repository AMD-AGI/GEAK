#!/usr/bin/env python3
"""The profile finalize on a salvage (partial) delivery, and unchanged without one.

Run:  python3 -m pytest -q workload/stages/profile/tests/test_salvage_finalize.py
(stdlib, CPU only; the receipts below are synthetic, not GPU evidence)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SLOT = Path(__file__).resolve().parents[1]
FINALIZE = SLOT / "finalize.py"
PARAMS = {"selection_top_k": 10, "selection_overall_pct_gt": 0.5,
          "selection_phases": ["PREFILL", "DECODE"]}
SALVAGE_WHY = "profile budget exhausted (salvage)"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def task(out, name, green=True):
    td = out / "tasks" / name
    (td / "kernel_src").mkdir(parents=True)
    for fn in ("unittest.py", "meta.json", "cases.py"):
        (td / fn).write_text("{}" if fn.endswith(".json") else "")
    write(td / "run" / "ut_result.json",
          {"ok": green, "failed_gates": [] if green else ["parity"]})
    if green:
        write(td / "run" / "RETURN.json", {"short_name": name})


class SalvageFinalizeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="profile_salvage_")
        self.addCleanup(shutil.rmtree, tmp)
        self.root = Path(tmp)
        self.out = self.root / "profile" / "out"
        self.base = self.root / "baseline" / "out"
        self.out.mkdir(parents=True)
        write(self.base / "baseline.json",
              {"multi_config": True, "points": [{"point_id": "c4"}, {"point_id": "c16"},
                                                {"point_id": "c64"}]})
        write(self.out / "params.json", PARAMS)
        write(self.out / "kernel_ranking.json", {
            "denominators": {"total_device_ms": 100.0,
                             "per_config": {"c4": {"total_device_ms": 50.0},
                                            "c16": {"total_device_ms": 50.0}},
                             "config_weights": {"c4": 0.5, "c16": 0.5}},
            "ranking": [{"id": "kA"}, {"id": "kB"}, {"id": "kC"}],
            "selection_scope": {"basis": "fixture"}, "candidate_inventory": {"all": ["kA"]},
            "coverage_summary": {"basis": "fixture"}})
        write(self.out / "selection_targets.json", {
            "policy": {"top_k_per_phase": 10, "overall_pct_gt": 0.5,
                       "phases": ["PREFILL", "DECODE"]},
            "targets": [{"id": "kA", "overall_pct": 20.0}, {"id": "kB", "overall_pct": 10.0},
                        {"id": "kC", "overall_pct": 5.0}]})
        task(self.out, "tA")
        self.results({"kA": "tA", "kB": None, "kC": None})

    def results(self, owners, tasks=("tA",)):
        rows = []
        for ident, owner in owners.items():
            if owner:
                rows.append({"id": ident, "task": owner, "status": "delivered",
                             "declared_pct_gpu_time": 20.0, "achieved_pct_gpu_time": 18.0})
            elif owner is None:
                rows.append({"id": ident, "status": "blocked", "why": SALVAGE_WHY})
        write(self.out / "results.json", {
            "identities": rows,
            "tasks": [{"task": t, "selection_pct_gpu_time": 20.0} for t in tasks],
            "coverage": {"delivered_declared_pct_gpu_time": 20.0,
                         "delivered_achieved_pct_gpu_time": 18.0}})

    def salvage(self):
        write(self.out / "salvage.json", {"schema": "workload/agent_salvage/v1",
                                          "sessions": []})

    def finalize(self):
        env = dict(os.environ, OUT=str(self.out), IN=str(self.base), STAGE="profile",
                   STAGE_SKILL="kernel-extract", PARAMS=str(self.out / "params.json"),
                   CONFIG=str(self.root / "config.resolved.json"))
        proc = subprocess.run([sys.executable, str(FINALIZE)], env=env,
                              capture_output=True, text=True)
        self.assertIn(proc.returncode, (0, 1), proc.stderr)
        return json.loads((self.out / "result.json").read_text())

    def test_without_marker_full_delivery_is_unchanged(self):
        res = self.finalize()
        self.assertEqual(res["status"], "ok", res.get("notes"))
        self.assertNotIn("delivery", res)
        self.assertNotIn("salvage", res["artifacts"])
        self.assertNotIn("delivery", res["metrics"])
        self.assertNotIn("PARTIAL", res.get("notes", ""))

    def test_partial_ok_with_qualified_and_blocked(self):
        self.salvage()
        res = self.finalize()
        self.assertEqual(res["status"], "ok", res.get("notes"))
        self.assertEqual(res["delivery"], "partial")
        self.assertTrue(res["notes"].startswith("PARTIAL DELIVERY (profile budget salvage)"))
        self.assertIn("'kB'", res["notes"])
        self.assertIn("'c64'", res["notes"])
        self.assertEqual(res["metrics"]["uncaptured_configs"], ["c64"])
        self.assertEqual(res["metrics"]["blocked_identity_ids"], ["kB", "kC"])
        self.assertEqual(res["artifacts"]["salvage"], "salvage.json")

    def test_partial_blocked_when_an_identity_is_unaccounted(self):
        self.salvage()
        self.results({"kA": "tA", "kB": None, "kC": False})   # kC: silently dropped
        res = self.finalize()
        self.assertEqual(res["status"], "blocked")
        self.assertEqual(res["delivery"], "partial")
        self.assertIn("kC", res["notes"])

    def test_partial_refuses_red_task_left_in_tasks(self):
        self.salvage()
        task(self.out, "tB", green=False)
        self.results({"kA": "tA", "kB": None, "kC": None})
        res = self.finalize()
        self.assertEqual(res["status"], "blocked")
        self.assertIn("tasks_unqualified/", res["notes"])
        self.assertIn("'tB'", res["notes"])

        # Moved out per the slot's salvage instructions: accepted, and counted.
        (self.out / "tasks_unqualified").mkdir()
        (self.out / "tasks" / "tB").rename(self.out / "tasks_unqualified" / "tB")
        res = self.finalize()
        self.assertEqual(res["status"], "ok", res.get("notes"))
        self.assertEqual(res["metrics"]["tasks_moved_unqualified"], 1)
        self.assertEqual(res["metrics"]["tasks_built"], 1)

    def test_partial_needs_at_least_one_qualified_task(self):
        self.salvage()
        shutil.rmtree(self.out / "tasks")
        self.results({"kA": None, "kB": None, "kC": None}, tasks=())
        res = self.finalize()
        self.assertEqual(res["status"], "blocked")
        self.assertIn("at least one", res["notes"])

    def test_partial_honours_configured_minimum(self):
        self.salvage()
        params = dict(PARAMS, minimum_qualified_tasks=2)
        write(self.out / "params.json", params)
        res = self.finalize()
        self.assertEqual(res["status"], "blocked")
        self.assertIn("minimum_qualified_tasks=2", res["notes"])

    def test_partial_blocked_when_results_lists_a_moved_task_as_delivered(self):
        # tB was moved out of tasks/ but results.json still names it as kB's owner.
        self.salvage()
        (self.out / "tasks_unqualified" / "tB" / "kernel_src").mkdir(parents=True)
        self.results({"kA": "tA", "kB": "tB", "kC": None}, tasks=("tA", "tB"))
        res = self.finalize()
        self.assertEqual(res["status"], "blocked")
        self.assertIn("kB", res["notes"])
        self.assertIn("tB", res["notes"])


def _load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


class SalvageDownstreamTests(unittest.TestCase):
    """No path hands an unqualified (moved) task to optimize or applyback.

    The readers below take the directory listing of tasks/ as the delivery; the
    salvage's `tasks_unqualified/`, a stale `results.json` row and a roofline-style order
    file naming a moved task must not bring it back."""

    def setUp(self):
        tmp = tempfile.mkdtemp(prefix="profile_salvage_down_")
        self.addCleanup(shutil.rmtree, tmp)
        self.out = Path(tmp) / "profile" / "out"
        for root, name, cfg in ((self.out / "tasks", "tA", "c4"),
                                (self.out / "tasks_unqualified", "tB", "c64")):
            td = root / name
            (td / "kernel_src").mkdir(parents=True)
            write(td / "meta.json", {"workload": {"cases": [{"config_id": cfg}]}})
        # A stale record that still lists the moved task, with the larger share.
        write(self.out / "results.json", {"tasks": [
            {"task": "tA", "selection_pct_gpu_time": 5.0},
            {"task": "tB", "selection_pct_gpu_time": 50.0}]})
        write(self.out / "task_plan.json", {"tasks": [{"name": "tB"}, {"name": "tA"}]})
        write(self.out / "order.json", {"kernels": [
            {"task": "tB", "expected_e2e_gain_pct": 9.0},
            {"task": "tA", "expected_e2e_gain_pct": 1.0}]})

    def test_scheduler_schedules_only_tasks_dirs(self):
        sched = _load("scheduler_salvage_test",
                      SLOT.parents[1] / "engine" / "campaign" / "scheduler.py")
        tasks = str(self.out / "tasks")
        self.assertEqual(sched.discover_tasks(tasks, order_by="share"), ["tA"])
        self.assertEqual(sched.discover_tasks(tasks, order_file=str(self.out / "order.json"),
                                              order_by="roofline"), ["tA"])

    def test_applyback_covers_only_delivered_configs(self):
        pts = _load("points_salvage_test", SLOT.parent / "applyback" / "points.py")
        self.assertEqual(pts.ut_configs(str(self.out / "tasks")), {"c4": ["tA"]})
        top = {"multi_config": True, "points": [{"point_id": "c4"}, {"point_id": "c64"}],
               "measure_order": ["c4", "c64"]}
        sel = pts.select(top, str(self.out / "tasks"), "covered")
        self.assertEqual((sel["points"], sel["required"]), (["c4"], ["c4"]))


if __name__ == "__main__":
    unittest.main()
