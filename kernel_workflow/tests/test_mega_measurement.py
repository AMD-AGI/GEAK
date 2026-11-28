"""Generic-mega measurement line: stage-meter reducer + route-report measured flags."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MRA = os.path.join(ROOT, "scripts", "multi_rank_analysis")
sys.path.insert(0, MRA)

import reduce_stage_meter as rsm  # noqa: E402

ROLES = [
    {"name": "gemm1", "stage": "stage1", "kind": "busy"},
    {"name": "dispatch_wait", "stage": "stage1", "kind": "wait"},
    {"name": "gemm2", "stage": "stage2", "kind": "busy"},
    {"name": "combine", "stage": "combine", "kind": "busy"},
    {"name": "combine_wait", "stage": "combine", "kind": "wait"},
]


def _dump(rank, scale=1):
    # 2 workgroups, tick_hz 1e6 -> 1 tick = 1 us.
    return {
        "rank": rank, "tick_hz": 1_000_000, "n_workgroups": 2, "overflow": 0,
        "roles": ROLES,
        "events": [
            [0, 1, 0, 100 * scale], [0, 0, 100 * scale, 1100 * scale],
            [0, 2, 1100 * scale, 2100 * scale],
            [1, 0, 0, 1000 * scale], [1, 4, 1000 * scale, 1500 * scale],
            [1, 3, 1500 * scale, 2000 * scale],
        ],
    }


class TestReduceStageMeter(unittest.TestCase):
    def test_busy_wait_are_per_workgroup_average_rank_max(self):
        out = rsm.reduce_dumps([_dump(0), _dump(1, scale=2)], "8192_uniform", "log")
        self.assertTrue(out["valid"], out["problems"])
        rows = {r["component"]: r for r in out["component_breakdown"]}
        # rank 1 is 2x slower -> rank-max picks it. stage1 busy = (1000+1000)*2/2 us.
        self.assertAlmostEqual(rows["stage1"]["busy_ms"], 2.0)
        self.assertAlmostEqual(rows["stage1"]["wait_ms"], 0.1)
        self.assertAlmostEqual(rows["combine"]["wait_ms"], 0.5)
        self.assertAlmostEqual(rows["stage1"]["span_ms"], 2.2)  # union [0,2200] on rank 1
        self.assertEqual(rows["stage2"]["method"], "timestamp")
        self.assertEqual(rows["stage2"]["guard"], "8192_uniform")
        self.assertAlmostEqual(out["kernel_ms_rank_max"], 4.2)

    def test_dead_or_broken_meter_is_invalid_not_zero(self):
        empty = dict(_dump(0), events=[])
        self.assertFalse(rsm.reduce_dumps([empty], "g", "")["valid"])
        self.assertFalse(rsm.reduce_dumps([dict(_dump(0), overflow=3)], "g", "")["valid"])
        self.assertFalse(rsm.reduce_dumps([], "g", "")["valid"])
        nested = _dump(0)
        nested["events"].append([0, 2, 0, 2100])  # overlaps its own roles
        self.assertFalse(rsm.reduce_dumps([nested], "g", "")["valid"])


PAIRED = [{"guard": "8192_uniform", "base": 10.0, "cand": 9.0}] * 3
BASE = [{"name": "8192_uniform", "latency_ms": 10.0}]


class TestRouteReportMeasured(unittest.TestCase):
    def _cli(self, extra, files):
        with tempfile.TemporaryDirectory() as d:
            paths = {}
            for key, obj in files.items():
                paths[key] = os.path.join(d, key + ".json")
                with open(paths[key], "w") as f:
                    json.dump(obj, f)
            out = os.path.join(d, "out.json")
            args = [sys.executable, os.path.join(MRA, "build_mega_route_report.py"),
                    "--paired", paths["paired"], "--baseline-per-case", paths["base"],
                    "--output", out]
            for flag, key in extra:
                args += [flag, paths.get(key, key)]
            subprocess.run(args, check=True, capture_output=True)
            with open(out) as f:
                return f.read()

    def test_without_new_flags_output_matches_head_builder(self):
        head = subprocess.run(
            ["git", "show", "HEAD:kernel_workflow/scripts/multi_rank_analysis/build_mega_route_report.py"],
            cwd=ROOT, capture_output=True, text=True)
        if head.returncode != 0:
            self.skipTest("HEAD builder unavailable")
        files = {"paired": PAIRED, "base": [{"name": "8192_uniform", "latency_ms": 10.0,
                                             "stage1": 4.0, "stage2_combine": 5.0}],
                 "ct": [{"name": "8192_uniform", "stage1": 3.0, "stage2_combine": 5.5}]}
        new = self._cli([("--component-timings", "ct")], files)
        with tempfile.TemporaryDirectory() as d:
            old_py = os.path.join(d, "old.py")
            with open(old_py, "w") as f:
                f.write(head.stdout)
            paths = {}
            for key, obj in files.items():
                paths[key] = os.path.join(d, key + ".json")
                with open(paths[key], "w") as f:
                    json.dump(obj, f)
            out = os.path.join(d, "out.json")
            subprocess.run([sys.executable, old_py, "--paired", paths["paired"],
                            "--baseline-per-case", paths["base"],
                            "--component-timings", paths["ct"], "--output", out],
                           check=True, capture_output=True)
            with open(out) as f:
                self.assertEqual(f.read(), new)

    def test_measured_split_and_wait(self):
        stages = [{"name": "8192_uniform", "stage1": 4.0, "stage2": 3.0, "combine": 2.0}]
        meter = rsm.reduce_dumps([_dump(0)], "8192_uniform", "log")
        rep = json.loads(self._cli(
            [("--baseline-stages", "stages"), ("--component-breakdown", "bd")],
            {"paired": PAIRED, "base": BASE, "stages": stages, "bd": meter}))
        rc = rep["route_comparisons"][0]
        pcd = rc["profile_category_delta"]
        self.assertAlmostEqual(pcd["stage1"]["candidate_rank_max"], 1.0)
        self.assertAlmostEqual(pcd["stage1"]["baseline_rank_max"], 4.0)
        self.assertAlmostEqual(pcd["combine"]["rank_max_delta_ms"], 0.25 - 2.0)
        self.assertAlmostEqual(pcd["fused_wait"]["candidate_rank_max"], 0.3)
        self.assertEqual(rc["profile_confidence"], "medium")
        self.assertNotIn("incomplete_reasons", rc)

    def test_lumped_baseline_pairs_with_split_candidate(self):
        bench_only = [{"name": "8192_uniform", "stage1": 4.0, "stage2_combine": 5.0}]
        rows = [{"component": "stage1", "busy_ms": 3.0, "wait_ms": 0.0, "method": "timestamp"},
                {"component": "stage2", "busy_ms": 2.0, "wait_ms": 0.0, "method": "timestamp"},
                {"component": "combine", "busy_ms": 1.0, "wait_ms": 0.0, "method": "timestamp"}]
        rep = json.loads(self._cli(
            [("--baseline-stages", "stages"), ("--component-breakdown", "bd"),
             ("--breakdown-guard", "8192_uniform")],
            {"paired": PAIRED, "base": BASE, "stages": bench_only, "bd": rows}))
        rc = rep["route_comparisons"][0]
        self.assertAlmostEqual(rc["profile_category_delta"]["stage2"]["rank_max_delta_ms"], -2.0)
        self.assertEqual(rc["profile_confidence"], "low")
        self.assertIn("stage2_combine lumped by bench harness", rc["incomplete_reasons"])

    def test_rows_without_guard_need_breakdown_guard(self):
        rows = [{"component": "stage1", "busy_ms": 3.0, "method": "timestamp"}]
        rep = json.loads(self._cli(
            [("--component-breakdown", "bd")], {"paired": PAIRED, "base": BASE, "bd": rows}))
        rc = rep["route_comparisons"][0]
        self.assertIn("no per-stage measurement for this guard", rc["incomplete_reasons"])

    def test_measured_report_passes_analyzer_and_validator(self):
        skill = os.path.join(ROOT, "knowledge", "analysis_skills", "moe_bottleneck")
        stages = [{"name": "8192_uniform", "stage1": 4.0, "stage2": 3.0, "combine": 2.0}]
        meter = rsm.reduce_dumps([_dump(0)], "8192_uniform", "log")
        report = self._cli([("--baseline-stages", "stages"), ("--component-breakdown", "bd")],
                           {"paired": PAIRED, "base": BASE, "stages": stages, "bd": meter})
        with tempfile.TemporaryDirectory() as d:
            rp, ap = os.path.join(d, "r.json"), os.path.join(d, "a.json")
            with open(rp, "w") as f:
                f.write(report)
            subprocess.run([sys.executable, os.path.join(skill, "analyze.py"), "--report", rp,
                            "--output", ap], check=True, capture_output=True)
            subprocess.run([sys.executable, os.path.join(skill, "validate_output.py"),
                            "--analysis", ap],
                           check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
