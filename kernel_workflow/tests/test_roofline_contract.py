#!/usr/bin/env python3
"""Contract tests for the CPU-only roofline workflow core."""

import copy
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSER_PATH = os.path.join(ROOT, "scripts", "roofline_kernel.py")
FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "rocprof_compute",
    "multi_kernel_analyze.txt",
)
SPEC = importlib.util.spec_from_file_location("roofline_kernel_contract", PARSER_PATH)
roofline_kernel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roofline_kernel)
roofline_policy = roofline_kernel.roofline_policy


def _write_json(directory, name, value):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle)
    return path


def _fixture_text():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


class EndToEndContractTests(unittest.TestCase):
    def test_fixture_to_metrics_policy_and_summary_contract(self):
        kernels = roofline_kernel.parse_rocprof_compute(
            _fixture_text(), dtypes=["fp8"], saturation_pct=55.0
        )
        selected = roofline_kernel._select_kernel(kernels, ["^fp8_gemm"])
        self.assertIsNotNone(selected)
        metrics = selected["metrics"]
        classification = selected["classification"]
        case = {
            "case_id": "fp8_case",
            "status": "matched",
            "weight": 3.0,
            "matched_kernel_name": selected["kernel_name"],
            "kernel": selected["kernel_name"],
            "shape": [4096, 4096, 4096],
            "dtypes": ["fp8"],
            "peak_basis": metrics["peak_basis"],
            "compute_metric": metrics["compute_metric"],
            "metrics": metrics,
            "classification": classification,
        }
        summary = roofline_kernel.build_summary([case], saturation_pct=55.0)

        self.assertIn(case["status"], roofline_kernel.CASE_STATUSES)
        self.assertIsInstance(roofline_policy.POLICY_VERSION, int)
        self.assertEqual(metrics["hbm_empirical_peak_gbps"], 4000.0)
        self.assertEqual(metrics["hbm_spec_peak_gbps"], 5300.0)
        self.assertEqual(metrics["roofline_efficiency_pct"], 80.0)
        self.assertEqual(classification["observed_limit"], "hbm")
        self.assertEqual(summary["case_routes"][0]["status"], "matched")
        self.assertIsInstance(summary["priority_order"][0]["priority"], float)
        self.assertEqual(summary["priority_order"][0]["reason"], "weight_times_empirical_headroom")
        self.assertEqual(summary["dominant_case_id"], "fp8_case")
        dominant = summary["dominant_classification"]
        for key in (
            "theoretical_bound",
            "observed_limit",
            "recommended_levers",
            "confidence",
            "evidence",
        ):
            self.assertIn(key, dominant)

    def test_empty_manifest_is_structured_skip(self):
        manifest = {
            "target": {"logical_name": "empty", "kernel_patterns": ["kernel"]},
            "cases": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_json(directory, "manifest.json", manifest)
            with mock.patch.object(
                roofline_kernel,
                "locate_rocprof_compute",
                side_effect=AssertionError("tool lookup must not run"),
            ):
                report = roofline_kernel.collect_manifest(
                    manifest_path,
                    "baseline",
                    os.path.join(directory, "out"),
                    timeout_sec=12,
                    saturation_pct=71,
                )
            self.assertEqual(report["status"], "skipped")
            self.assertEqual(report["reason"], "no_profile_cases")
            self.assertEqual(report["cases"], [])
            self.assertEqual(report["policy"], {"version": 1, "saturation_pct": 71.0})
            self.assertIsInstance(report["tool_version"], str)
            dominant = report["summary"]["dominant_classification"]
            self.assertEqual(dominant["theoretical_bound"], "unknown")
            self.assertEqual(dominant["observed_limit"], "unknown")
            self.assertEqual(dominant["confidence"], "low")
            self.assertTrue(os.path.isfile(report["json_path"]))

    def test_unavailable_tool_is_structured_skip(self):
        manifest = {
            "target": {"logical_name": "gemm"},
            "cases": [{"case_id": "c0", "command": ["python3", "-V"]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_json(directory, "manifest.json", manifest)
            with mock.patch.object(
                roofline_kernel, "locate_rocprof_compute", return_value=(None, None)
            ):
                report = roofline_kernel.collect_manifest(
                    manifest_path, "baseline", os.path.join(directory, "out")
                )
            self.assertEqual(report["status"], "skipped")
            self.assertEqual(report["reason"], "rocprof_compute_unavailable")
            self.assertEqual(report["cases"], [])


class SelectionAndExecutionContractTests(unittest.TestCase):
    def _manifest_case(self, pattern):
        manifest = {
            "target": {
                "logical_name": "gemm",
                "kernel_patterns": [pattern],
                "gpu_id": "2",
            }
        }
        case = {
            "case_id": "c0",
            "command": ["python3", "bench.py"],
            "shape": [4096, 4096],
            "dtypes": ["fp8"],
            "weight": 2.0,
        }
        return manifest, case

    def test_pattern_mismatch_falls_back_to_dominant_target(self):
        # When no supplied pattern matches (e.g. the config swapped the GEMM
        # backend so the executed kernel name differs from the pattern),
        # selection falls back to the dominant target kernel with valid metrics
        # instead of reporting failure. Helper/runtime kernels stay excluded.
        kernels = roofline_kernel.parse_rocprof_compute(_fixture_text())
        fallback = roofline_kernel._select_kernel(kernels, ["definitely_not_present"])
        self.assertIsNotNone(fallback)
        self.assertTrue(roofline_kernel._is_target_kernel(fallback["kernel_name"]))
        self.assertFalse(
            roofline_kernel._kernel_matches_patterns(
                fallback, ["definitely_not_present"]
            )
        )

        manifest, case = self._manifest_case("^definitely_not_present$")
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                roofline_kernel,
                "_run",
                side_effect=[
                    (0, "--output-directory PATH", None),
                    (0, "profile output", None),
                    (0, _fixture_text(), None),
                ],
            ) as run:
                result = roofline_kernel._case_result(
                    manifest, case, "/fake/tool", "baseline", directory
                )
        self.assertEqual(result["status"], "matched")
        self.assertIsNotNone(result["matched_kernel_name"])
        self.assertEqual(result["selection_mode"], "fallback")
        self.assertTrue(
            roofline_kernel._is_target_kernel(result["matched_kernel_name"])
        )
        self.assertTrue(any("fallback" in item for item in result["warnings"]))
        profile_arguments = run.call_args_list[1].args[0]
        self.assertIn("--output-directory", profile_arguments)
        self.assertNotIn("-k", profile_arguments)
        profile_environment = run.call_args_list[1].kwargs["env"]
        self.assertEqual(profile_environment["ROCR_VISIBLE_DEVICES"], "2")
        self.assertEqual(profile_environment["HIP_VISIBLE_DEVICES"], "2")

    def test_helper_only_output_still_fails(self):
        # If every kernel in the analyze output is a helper/runtime kernel, there
        # is no legitimate target to fall back to and the case fails cleanly.
        helper_only = (
            "Top Kernels\n"
            "0 | void at::native::vectorized_elementwise_kernel<4> | 100.0\n"
            "1 | __amd_rocclr_copyBuffer | 50.0\n"
        )
        kernels = roofline_kernel.parse_rocprof_compute(helper_only)
        self.assertIsNone(
            roofline_kernel._select_kernel(kernels, ["definitely_not_present"])
        )

    def test_nonzero_commands_with_valid_target_still_match(self):
        manifest, case = self._manifest_case("^fp8_gemm")
        del case["dtypes"]
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                roofline_kernel,
                "_run",
                side_effect=[
                    (0, "--output-directory PATH", None),
                    (9, "profile failed after artifacts", None),
                    (7, _fixture_text(), None),
                ],
            ):
                result = roofline_kernel._case_result(
                    manifest,
                    case,
                    "/fake/tool",
                    "baseline",
                    directory,
                    timeout_sec=22,
                    saturation_pct=61,
                )
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["dtypes"], [])
        self.assertEqual(result["matched_kernel_name"], "fp8_gemm_kernel")
        self.assertEqual(result["classification"]["saturation_pct"], 61.0)
        self.assertEqual(
            result["classification"]["observed_limit"], "latency_occupancy"
        )
        self.assertGreaterEqual(len(result["warnings"]), 2)

    def test_collect_aggregates_nonzero_matched_case_as_partial(self):
        manifest, case = self._manifest_case("^fp8_gemm")
        manifest["cases"] = [case]
        matched = {
            "case_id": "c0",
            "status": "matched",
            "weight": 2.0,
            "matched_kernel_name": "fp8_gemm_kernel",
            "metrics": {"headroom_ratio": 1.5},
            "classification": roofline_policy.build_classification(
                {}, saturation_pct=73
            ),
            "profile_exit_code": 9,
            "analyze_exit_code": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _write_json(directory, "manifest.json", manifest)
            with mock.patch.object(
                roofline_kernel,
                "locate_rocprof_compute",
                return_value=("/fake/tool", "test"),
            ), mock.patch.object(
                roofline_kernel,
                "_tool_version",
                return_value={
                    "exit_code": 0,
                    "text": "test 1.0",
                    "timed_out": False,
                    "warning": None,
                },
            ), mock.patch.object(
                roofline_kernel, "_case_result", return_value=matched
            ) as case_result:
                report = roofline_kernel.collect_manifest(
                    manifest_path,
                    "baseline",
                    os.path.join(directory, "out"),
                    timeout_sec=44,
                    saturation_pct=73,
                )
        self.assertEqual(report["status"], "partial")
        self.assertIn(report["status"], roofline_kernel.TOP_LEVEL_STATUSES)
        self.assertEqual(report["tool_version"], "test 1.0")
        self.assertEqual(report["policy"]["saturation_pct"], 73.0)
        self.assertEqual(
            case_result.call_args.kwargs["saturation_pct"], 73.0
        )
        self.assertEqual(case_result.call_args.kwargs["timeout_sec"], 44.0)

    def test_timeout_is_structured(self):
        expired = subprocess.TimeoutExpired(
            cmd=["rocprof-compute", "profile"], timeout=3, output=b"partial"
        )
        with mock.patch.object(
            roofline_kernel.subprocess, "run", side_effect=expired
        ) as run:
            code, output, warning = roofline_kernel._run(
                ["rocprof-compute", "profile"], timeout_sec=3
            )
        self.assertEqual(code, 124)
        self.assertIn("partial", output)
        self.assertIn("timed out", warning)
        self.assertEqual(run.call_args.kwargs["timeout"], 3)

    def test_cli_collect_timeout_and_threshold_options(self):
        defaults = roofline_kernel._parser().parse_args(
            [
                "collect",
                "--manifest",
                "manifest.json",
                "--phase",
                "baseline",
                "--out-dir",
                "out",
            ]
        )
        self.assertEqual(defaults.timeout_sec, 3600.0)
        self.assertEqual(defaults.saturation_pct, 60.0)
        custom = roofline_kernel._parser().parse_args(
            [
                "collect",
                "--manifest",
                "manifest.json",
                "--phase",
                "after",
                "--out-dir",
                "out",
                "--timeout-sec",
                "7",
                "--saturation-pct",
                "72",
            ]
        )
        self.assertEqual(custom.timeout_sec, 7.0)
        self.assertEqual(custom.saturation_pct, 72.0)

    def test_geak_roofline_path_override(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = os.path.join(directory, "rocprof-compute")
            with open(executable, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\nexit 0\n")
            os.chmod(executable, 0o700)
            path, source = roofline_kernel.locate_rocprof_compute(
                {
                    "GEAK_ROOFLINE_COMPUTE_PATH": executable,
                    "PATH": "",
                }
            )
        self.assertEqual(path, executable)
        self.assertEqual(source, "GEAK_ROOFLINE_COMPUTE_PATH")


class ComparisonContractTests(unittest.TestCase):
    def _case(self, performance):
        metrics = {
            "ai_hbm": 200.0,
            "performance_gflops": performance,
            "compute_actual_gflops": performance,
            "compute_empirical_peak_gflops": 1000.0,
            "hbm_empirical_peak_gbps": 10.0,
            "compute_utilization_pct": performance / 10.0,
            "hbm_utilization_pct": 20.0,
            "peak_basis": "empirical",
            "compute_metric": "MFMA FLOPs (F16)",
        }
        return {
            "case_id": "c0",
            "status": "matched",
            "shape": [1024, 1024],
            "dtypes": ["fp16"],
            "matched_kernel_name": "gemm_kernel",
            "peak_basis": "empirical",
            "compute_metric": "MFMA FLOPs (F16)",
            "metrics": metrics,
            "classification": roofline_policy.build_classification(
                metrics, saturation_pct=60
            ),
        }

    def _report(self, performance):
        return {
            "status": "ok",
            "policy": {"version": 1, "saturation_pct": 60.0},
            "target": {
                "logical_name": "gemm",
                "device": "MI300X",
                "gpu_id": 0,
            },
            "cases": [self._case(performance)],
        }

    def _compare(self, before, after):
        with tempfile.TemporaryDirectory() as directory:
            before_path = _write_json(directory, "before.json", before)
            after_path = _write_json(directory, "after.json", after)
            return roofline_kernel.compare_reports(before_path, after_path)

    def test_matching_contract_compares(self):
        result = self._compare(self._report(500.0), self._report(750.0))
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["cases"][0]["performance_ratio"], 1.5)

    def test_policy_and_target_identity_must_match(self):
        before = self._report(500.0)
        after = self._report(750.0)
        after["policy"]["version"] = 2
        with self.assertRaises(ValueError):
            self._compare(before, after)

        after = self._report(750.0)
        after["policy"]["saturation_pct"] = 65.0
        with self.assertRaises(ValueError):
            self._compare(before, after)

        after = self._report(750.0)
        del after["target"]["gpu_id"]
        with self.assertRaises(ValueError):
            self._compare(before, after)

        after = self._report(750.0)
        after["target"]["device"] = "MI355X"
        with self.assertRaises(ValueError):
            self._compare(before, after)

    def test_only_matched_cases_are_comparable(self):
        before = self._report(500.0)
        after = self._report(750.0)
        after["cases"][0]["status"] = "failed"
        with self.assertRaises(ValueError):
            self._compare(before, after)

        after = self._report(750.0)
        after["cases"].append(copy.deepcopy(after["cases"][0]))
        after["cases"][1]["case_id"] = "other"
        with self.assertRaises(ValueError):
            self._compare(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
