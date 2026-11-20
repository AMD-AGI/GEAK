#!/usr/bin/env python3
"""kernels.json: a profile's selected kernels with path, shapes and weights, re-derivable.

Run:  python3 -I workload/stages/profile/tests/test_kernels.py   (stdlib only, CPU only)
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SLOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kernels", SLOT / "kernels.py")
K = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K)

PARAMS = {"selection_top_k": 5, "selection_overall_pct_gt": 3, "selection_phases": ["PREFILL", "DECODE"]}
ENTRY = {"target_callable": "pkg.moe:fused_moe", "path": "/src/pkg/moe.py", "sha256": "a" * 64,
         "resolved": True}


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


class KernelsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        r = self.root
        write(r / "kernel_ranking.json", {
            "denominators": {"total_device_ms": 1000.0},
            "phase_share": {"DECODE": 100.0},
            "selection_scope": {"selector": {"args": ["--top-k", "5", "--overall-pct-gt", "3",
                                                      "--phases", "PREFILL", "DECODE"]}},
            "candidate_inventory": [{"id": "moe_gemm"}], "coverage_summary": {"DECODE": 90},
            "ranking": [{"id": "moe_gemm", "share_of_phase_pct": {"DECODE": 40.0},
                         "selection_pct_gpu_time": 40.0}]})
        write(r / "selection_targets.json", {
            "policy": {"top_k_per_phase": 5, "overall_pct_gt": 3, "phases": ["PREFILL", "DECODE"]},
            "targets": [{"id": "moe_gemm", "overall_pct": 40.0, "op": "pkg.moe:fused_moe@Moe",
                         "selected_phases": ["DECODE"]}]})
        write(r / "sampling_points_final.json", {
            "phase_summary": {"DECODE": {"unrepresented_bands": []}},
            "points": [{"point_id": "de_4", "phase": "DECODE", "achieved_shape": 4,
                        "achieved_context": 1000, "counted": True, "represented": {"instances": 50}}]})
        write(r / "evidence/source_map.json", {"kernels": {"moe_gemm": {
            "op": {"key": "pkg.moe:fused_moe@Moe", "entry": ENTRY, "module_class": "Moe"},
            "launching_ops": ["aiter::fused_moe_"],
            "frames": [ENTRY, {"target_callable": "pkg.model:Layer.forward", "path": "/src/pkg/model.py",
                               "line": 9, "sha256": "b" * 64, "resolved": True}]}}})
        write(r / "evidence/adjacency.json", {"pairs": []})
        write(r / "evidence/census/de_4.json", {"phase": "DECODE", "achieved_shape": 4})
        write(r / "evidence/shapes/de_4.json", {
            "phase": "DECODE", "achieved_shape": 4, "point_id": "de_4",
            "kernels": {"moe_gemm": {"variants": [{
                "grid": [64, 1, 1], "block": [256, 1, 1], "launches_per_instance": 2.0,
                "us_per_instance": 30.0,
                "samples": [{"host_op": {"op": "aiter::fused_moe_", "input_dims": [[4, 6144]],
                                         "input_types": ["c10::BFloat16"],
                                         "input_strides": [[6144, 1]]}}]}]}}})

    def test_path_shapes_and_weight_of_each_selected_kernel(self):
        doc = K.build(self.root, PARAMS)
        k = doc["kernels"][0]
        self.assertEqual(k["path"]["entry"], "pkg.moe:fused_moe")
        self.assertEqual(k["path"]["entry_file"], "/src/pkg/moe.py")
        self.assertEqual([f["target_callable"] for f in k["path"]["frames"]],
                         ["pkg.moe:fused_moe", "pkg.model:Layer.forward"])
        p = k["points"][0]
        self.assertEqual((p["phase"], p["shape"], p["instances"], p["launches_per_instance"]),
                         ("DECODE", 4, 50.0, 2.0))
        self.assertEqual(p["variants"][0]["operands"][0]["input_dims"], [[4, 6144]])
        self.assertEqual(k["weight"]["launches"], 100.0)
        self.assertEqual(k["weight"]["overall_pct"], 40.0)
        self.assertEqual(k["gaps"], [])
        self.assertEqual(doc["operators"][0]["kernels"], ["moe_gemm"])
        self.assertIn("evidence/shapes/de_4.json", doc["files"])

    def test_verify_rederives_and_refuses_changed_evidence(self):
        K.write(self.root / "kernels.json", K.build(self.root, PARAMS))
        self.assertEqual(K.verify(self.root / "kernels.json")["contract"], K.CONTRACT)
        shapes = self.root / "evidence/shapes/de_4.json"
        doc = json.loads(shapes.read_text())
        doc["kernels"]["moe_gemm"]["variants"][0]["launches_per_instance"] = 3.0
        shapes.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "changed"):
            K.verify(self.root / "kernels.json")

    def test_a_selection_at_another_scope_is_refused(self):
        with self.assertRaisesRegex(ValueError, "configured"):
            K.build(self.root, dict(PARAMS, selection_top_k=10))


if __name__ == "__main__":
    unittest.main()
