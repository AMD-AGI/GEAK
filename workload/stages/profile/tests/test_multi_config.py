#!/usr/bin/env python3
"""Two configs -> multi_config merge -> plan_ut_cases -> UT manifest -> meta cases -> harness.

Run:  python3 -I workload/stages/profile/tests/test_multi_config.py   (stdlib, CPU only)
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SLOT = Path(__file__).resolve().parents[1]
SKILL = SLOT / "kernel-extract"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


mc = _load("multi_config", SLOT / "multi_config.py")
puc = _load("plan_ut_cases", SKILL / "scripts" / "plan_ut_cases.py")
sca = _load("scaffold_ut", SKILL / "ut_contracts" / "scaffold_ut.py")
hl = _load("harness_lib_mc2", SKILL / "shared" / "harness_lib.py")

K = "fused_moe_kernel"


def _w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _config(root, cid, decode_shape, decode_inst, prefill_inst, moe_pct, band=None):
    d = os.path.join(root, "configs", cid)
    pts = [
        {"point_id": "de_%d" % decode_shape, "phase": "DECODE", "counted": True,
         "achieved_shape": decode_shape, "context_tokens": 40000,
         "represented": {"instances": decode_inst}, "census": "evidence/census/de.json"},
        {"point_id": "pf_2048", "phase": "PREFILL", "counted": True, "achieved_shape": 2048,
         "running": 1, "cached_tokens": 60000, "new_tokens": 2048,
         "represented": {"instances": prefill_inst}, "census": "evidence/census/pf.json"},
    ]
    summary = {"DECODE": {"unrepresented_bands": [], "proxy_mass_outside_tolerance_pct": 0.0},
               "PREFILL": {"unrepresented_bands": [], "proxy_mass_outside_tolerance_pct": 0.0}}
    if band:
        summary["DECODE"]["unrepresented_bands"] = [dict(band, nearest_point="de_%d" % decode_shape,
                                                         bracketing_points=["de_%d" % decode_shape])]
        summary["DECODE"]["proxy_mass_outside_tolerance_pct"] = band["proxy_mass_share_of_phase_pct"]
    _w(os.path.join(d, "sampling_points_final.json"),
       {"shape_tolerance": 1.25, "phase_summary": summary, "points": pts})
    for p in pts:
        _w(os.path.join(d, "evidence", "shapes", p["point_id"] + ".json"),
           {"point_id": p["point_id"], "kernels": {K: {"launches_per_instance": 48}}})
        _w(os.path.join(d, "evidence", "census", p["point_id"] + ".json"), {"point_id": p["point_id"]})
    _w(os.path.join(d, "kernel_ranking.json"), {"ranking": [
        {"id": K, "name": K, "kind": "compute", "mass_us": 1, "selection_pct_gpu_time": moe_pct,
         "share_of_phase_pct": {"DECODE": moe_pct, "PREFILL": moe_pct}},
        {"id": "small", "name": "small", "kind": "compute", "mass_us": 1,
         "selection_pct_gpu_time": 100 - moe_pct, "share_of_phase_pct": {"DECODE": 1, "PREFILL": 1}}]})
    _w(os.path.join(d, "evidence", "source_map.json"),
       {"packages": ["sglang"], "graph_replayed_only": [K] if cid == "c64" else [],
        "kernels": {K: {"frames": [] if cid == "c64" else [{"frame": "x"}]}}, "files": {}})
    return d


class MultiConfigProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.group = os.path.join(root, "ep8")
        c1 = _config(self.group, "c1", 1, 1000, 50, 2.0)
        c64 = _config(self.group, "c64", 64, 100000, 5000, 9.0,
                      band={"shape": 48, "lo": 40, "hi": 56, "instances": 20000,
                            "proxy_mass_share_of_phase_pct": 3.0})
        out = subprocess.run([sys.executable, "-I", str(SLOT / "multi_config.py"), "merge",
                              "--group-dir", self.group, "--config", "c1=" + c1,
                              "--config", "c64=" + c64], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)

    def tearDown(self):
        self.tmp.cleanup()

    def _j(self, *p):
        with open(os.path.join(self.group, *p)) as fh:
            return json.load(fh)

    def test_merged_ranking_is_the_equal_weight_mean(self):
        r = {row["id"]: row for row in self._j("kernel_ranking.json")["ranking"]}
        self.assertAlmostEqual(r[K]["selection_pct_gpu_time"], 5.5)
        self.assertEqual(r[K]["per_config_pct"], {"c1": 2.0, "c64": 9.0})
        sm = self._j("evidence", "source_map.json")
        self.assertEqual(sm["graph_replayed_only"], [])      # c1 saw it eagerly
        self.assertTrue(sm["kernels"][K]["frames"])
        self.assertTrue(os.path.exists(os.path.join(self.group, "evidence", "shapes", "c64__de_64.json")))

    def test_one_ut_with_per_config_cases_and_equal_shares(self):
        pts = self._j("sampling_points_final.json")
        task = {"label": "moe", "kernels": [K], "phases": ["PREFILL", "DECODE"],
                "case_axes": {"DECODE": {"M": "shape"}, "PREFILL": {"M": "shape"}}}
        plan = puc.plan(pts, os.path.join(self.group, "evidence", "shapes"), [task])
        cases = plan["tasks"]["moe"]["cases"]
        self.assertEqual(sorted({c["config_id"] for c in cases}), ["c1", "c64"])
        self.assertEqual(len([c for c in cases if c["kind"] == "bucket"]), 1)
        self.assertEqual(len({c["case_id"] for c in cases}), len(cases))
        bt = _load("build_task_plan", SKILL / "scripts" / "build_task_plan.py")
        man = bt.manifest(task, cases)
        sigs = [c["sig"] for c in man["cases"]]
        self.assertEqual(len(set(sigs)), len(sigs))
        self.assertIn("c64/DECODE_64", sigs)
        wl = sca.workload_cases(man)
        self.assertTrue(all(c.get("config_id") for c in wl))
        meta = {"workload": {"kind": "agentx_trace_replay", "cases": wl,
                             "serving_weight_model": {"source": "observed_workload",
                                                      "evidence_path": "x"}}}
        # c1 2x faster; c64 unchanged and 4x the per-launch time (larger M)
        rows = [{"sig": c["sig"], "regime": c["regime"], "m": c.get("m"),
                 "baseline_ms": 1.0 if c["config_id"] == "c1" else 4.0,
                 "optimized_ms": 0.5 if c["config_id"] == "c1" else 4.0} for c in wl]
        res = hl.serving_weighted_speedup(rows, meta)
        self.assertEqual(res["config_weighting"], "calls")
        self.assertAlmostEqual(res["per_config"]["c1"]["weighted"], 2.0)
        self.assertAlmostEqual(res["per_config"]["c64"]["weighted"], 1.0)
        # w(M) = mean over configs of freq_c(M); metric sum w t: c1 adds 0.5*1, c64 0.5*4
        self.assertAlmostEqual(res["weighted"], (0.5 * 1 + 0.5 * 4) / (0.5 * 0.5 + 0.5 * 4))
        meta["workload"]["serving_weight_model"]["config_weighting"] = "time"
        res = hl.serving_weighted_speedup(rows, meta)
        self.assertAlmostEqual(res["weighted"], 1 / (0.5 / 2.0 + 0.5 / 1.0))

    def test_select_is_per_config_then_union(self):
        # c1: moe 2% (under 5); c64: moe 9%. Mean 5.5% would also pass, so add an op that only
        # c64 carries: 6% there, 0 at c1 -> mean 3%, must still be selected.
        d64 = os.path.join(self.group, "configs", "c64")
        with open(os.path.join(d64, "kernel_ranking.json")) as fh:
            rk = json.load(fh)
        rk["ranking"].append({"id": "big_m_gemm", "name": "big_m_gemm", "kind": "compute",
                              "mass_us": 1, "selection_pct_gpu_time": 6.0,
                              "share_of_phase_pct": {"DECODE": 6.0, "PREFILL": 0.0}})
        rk["ranking"][1]["selection_pct_gpu_time"] = 85.0
        with open(os.path.join(d64, "kernel_ranking.json"), "w") as fh:
            json.dump(rk, fh)
        params = os.path.join(self.tmp.name, "params.json")
        with open(params, "w") as fh:
            json.dump({"selection_top_k": 1000, "selection_overall_pct_gt": 5,
                       "selection_phases": ["PREFILL", "DECODE"]}, fh)
        out = os.path.join(self.tmp.name, "sel.json")
        r = subprocess.run([sys.executable, "-I", str(SLOT / "multi_config.py"), "select",
                            "--group-dir", self.group, "--output", out, "--params", params,
                            "--skill-root", str(SKILL)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out) as fh:
            sel = json.load(fh)
        got = {t["id"]: t for t in sel["targets"]}
        self.assertIn("big_m_gemm", got)                       # 6% of c64's own workload
        self.assertEqual(got["big_m_gemm"]["selected_in_configs"], ["c64"])
        self.assertEqual(got[K]["selected_in_configs"], ["c64"])  # 2% at c1 is under 5
        self.assertEqual(got["small"]["selected_in_configs"], ["c1", "c64"])  # the toy's bulk
        self.assertEqual(sel["policy"]["combination"], "per_config_then_union")
        self.assertEqual(sel["policy"]["overall_pct_gt"], 5)

    def test_single_config_plan_is_unchanged(self):
        d = os.path.join(self.group, "configs", "c1")
        with open(os.path.join(d, "sampling_points_final.json")) as fh:
            pts = json.load(fh)
        task = {"label": "moe", "kernels": [K], "phases": ["PREFILL", "DECODE"]}
        plan = puc.plan(pts, os.path.join(d, "evidence", "shapes"), [task])
        self.assertNotIn("configs", plan)
        self.assertTrue(all("config_id" not in c for c in plan["tasks"]["moe"]["cases"]))


if __name__ == "__main__":
    unittest.main()
