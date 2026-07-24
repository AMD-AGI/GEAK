#!/usr/bin/env python3
"""Pure-CPU unit tests for rocprof-compute text parsing."""

import importlib.util
import os
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSER_PATH = os.path.join(ROOT, "scripts", "roofline_kernel.py")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "rocprof_compute")
SPEC = importlib.util.spec_from_file_location("roofline_kernel", PARSER_PATH)
roofline_kernel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(roofline_kernel)


class ParserTests(unittest.TestCase):
    def test_single_kernel_ansi_and_old_title(self):
        text = """
\x1b[1;34m4. Roofline Analysis\x1b[0m
Kernel Name: vector_kernel
│ 4.1.0 │ VALU FLOPs (F32) │ 30000 │ Gflop/s │ 60000 │
│ 4.1.7 │ HBM Bandwidth │ 1000 │ Gb/s │ 4000 │
│ 4.1.8 │ L2 Cache Bandwidth │ 10000 │ Gb/s │ 20000 │
│ 4.1.9 │ L1 Cache Bandwidth │ 9000 │ Gb/s │ 30000 │
│ 4.1.10 │ LDS Bandwidth │ 25000 │ Gb/s │ 50000 │
│ 4.2.0 │ AI HBM │ 100 │ Flops/byte │
│ 4.2.1 │ AI L2 │ 10 │ Flops/byte │
│ 4.2.2 │ AI L1 │ 5 │ Flops/byte │
│ 4.2.3 │ Performance (GFLOPs) │ 30000 │ Gflop/s │
│ 17.1.5 │ Peak HBM Bandwidth │ 5300 │ GB/s │
"""
        kernels = roofline_kernel.parse_rocprof_compute(text, dtypes=["fp32"])
        self.assertEqual(len(kernels), 1)
        kernel = kernels[0]
        self.assertEqual(kernel["kernel_name"], "vector_kernel")
        metrics = kernel["metrics"]
        self.assertEqual(metrics["compute_metric"], "VALU FLOPs (F32)")
        self.assertAlmostEqual(metrics["ai_hbm"], 100.0)
        self.assertAlmostEqual(metrics["ai_ridge_empirical"], 15.0)
        self.assertAlmostEqual(metrics["compute_utilization_pct"], 50.0)
        self.assertAlmostEqual(metrics["hbm_utilization_pct"], 25.0)
        self.assertAlmostEqual(metrics["l2_utilization_pct"], 50.0)
        self.assertAlmostEqual(metrics["lds_utilization_pct"], 50.0)
        self.assertAlmostEqual(metrics["hbm_spec_peak_gbps"], 5300.0)

    def test_multi_kernel_fixture_and_dtype_selection(self):
        with open(
            os.path.join(FIXTURES, "multi_kernel_analyze.txt"),
            "r",
            encoding="utf-8",
        ) as handle:
            text = handle.read()
        kernels = roofline_kernel.parse_rocprof_compute(text, dtypes=["mxfp4"])
        self.assertEqual([item["kernel_name"] for item in kernels], [
            "fp8_gemm_kernel",
            "mxfp4_and_int8_kernel",
        ])
        self.assertGreaterEqual(len(kernels[1]["compute_rates"]), 4)
        self.assertEqual(
            kernels[1]["metrics"]["compute_metric"], "MFMA FLOPs (MXFP4)"
        )
        self.assertAlmostEqual(kernels[1]["metrics"]["hbm_spec_peak_gbps"], 5300.0)
        self.assertIsNone(kernels[1]["metrics"]["ai_l2"])

    def test_global_section_17_is_propagated_to_every_kernel(self):
        text = """
Kernel 0: first_kernel
| 4.1.0 | MFMA FLOPs (F16) | 100 | Gflop/s | 1000 |
| 4.1.7 | HBM Bandwidth | 10 | GB/s | 100 |
| 4.2.0 | AI HBM | 10 | Flops/byte |
Kernel 1: second_kernel
| 4.1.0 | MFMA FLOPs (F16) | 200 | Gflop/s | 1000 |
| 4.1.7 | HBM Bandwidth | 20 | GB/s | 100 |
| 4.2.0 | AI HBM | 20 | Flops/byte |
| 17.1.5 | Peak HBM Bandwidth | 5300 | GB/s |
"""
        kernels = roofline_kernel.parse_rocprof_compute(text, dtypes=["fp16"])
        self.assertEqual(len(kernels), 2)
        self.assertEqual(
            [kernel["metrics"]["hbm_spec_peak_gbps"] for kernel in kernels],
            [5300.0, 5300.0],
        )

    def test_fp8_name_is_selected_from_manifest_dtype(self):
        with open(
            os.path.join(FIXTURES, "multi_kernel_analyze.txt"),
            "r",
            encoding="utf-8",
        ) as handle:
            kernels = roofline_kernel.parse_analyze_text(
                handle.read(), dtypes=["fp8_e4m3"]
            )
        first = kernels[0]
        self.assertEqual(first["metrics"]["compute_metric"], "MFMA FLOPs (FP8)")
        self.assertAlmostEqual(first["metrics"]["performance_gflops"], 800000.0)
        self.assertAlmostEqual(first["metrics"]["roofline_efficiency_pct"], 80.0)
        self.assertAlmostEqual(first["metrics"]["headroom_ratio"], 1.25)

    def test_fp8_does_not_select_higher_mxfp8_rate(self):
        rates = [
            {"metric": "MFMA FLOPs (FP8)", "value": 100.0},
            {"metric": "MFMA FLOPs (MXFP8)", "value": 900.0},
        ]
        selected = roofline_kernel.select_dominant_compute_rate(rates, ["fp8"])
        self.assertEqual(selected["metric"], "MFMA FLOPs (FP8)")

    def test_new_metric_layout_finds_performance_by_name(self):
        text = """
Kernel 0: current_kernel
| 4.1.0 | MFMA FLOPs (F16) | 500 | Gflop/s | 1000 |
| 4.1.7 | HBM Bandwidth | 10 | GB/s | 100 |
| 4.2.0 | AI HBM | 20 | Flops/byte |
| 4.2.3 | AI LDS | 7 | Flops/byte |
| 4.2.4 | Performance (GFLOPs) | 500 | Gflop/s |
"""
        kernel = roofline_kernel.parse_rocprof_compute(text, dtypes=["fp16"])[0]
        self.assertEqual(kernel["metrics"]["performance_gflops"], 500.0)

    def test_int8_iops_and_mx_names_are_preserved(self):
        text = """
Kernel 0: quant_kernel
| 4.1.0 | VALU IOPs (Int8) | 12 | Giop/s | 100 |
| 4.1.1 | MFMA IOPs (Int8) | 80 | Giop/s | 200 |
| 4.1.2 | MFMA FLOPs (MXFP8) | 0 | Gflop/s | 400 |
| 4.1.3 | MFMA FLOPs (F6) | 0 | Gflop/s | 400 |
| 4.1.4 | MFMA FLOPs (F4) | 0 | Gflop/s | 400 |
| 4.1.7 | HBM Bandwidth | 5 | GB/s | 10 |
| 4.2.0 | AI HBM | 20 | Ops/byte |
| 4.2.3 | Performance (GIOPs) | 80 | Giop/s |
"""
        kernels = roofline_kernel.parse_rocprof_compute(text, dtypes=["int8"])
        metrics = kernels[0]["metrics"]
        self.assertEqual(metrics["compute_metric"], "MFMA IOPs (Int8)")
        self.assertEqual(
            [rate["metric"] for rate in kernels[0]["compute_rates"]],
            [
                "VALU IOPs (Int8)",
                "MFMA IOPs (Int8)",
                "MFMA FLOPs (MXFP8)",
                "MFMA FLOPs (F6)",
                "MFMA FLOPs (F4)",
            ],
        )

    def test_largest_nonzero_rate_is_fallback(self):
        text = """
Kernel [0]: mixed_kernel
4.1.0  VALU FLOPs (F32)  50  Gflop/s  100
4.1.1  MFMA FLOPs (BF16)  500  Gflop/s  1000
4.1.7  HBM Bandwidth  10  GB/s  20
4.2.0  AI HBM  50  Flops/byte
4.2.3  Performance (GFLOPs)  N/A  Gflop/s
"""
        kernel = roofline_kernel.parse_rocprof_compute(text)[0]
        self.assertEqual(kernel["metrics"]["compute_metric"], "MFMA FLOPs (BF16)")
        self.assertAlmostEqual(kernel["metrics"]["performance_gflops"], 500.0)

    def test_f16_dtype_does_not_accidentally_match_bf16(self):
        text = """
Kernel 0: half_kernel
| 4.1.1 | MFMA FLOPs (BF16) | 900 | Gflop/s | 1000 |
| 4.1.2 | MFMA FLOPs (F16) | 400 | Gflop/s | 1000 |
| 4.1.7 | HBM Bandwidth | 5 | GB/s | 10 |
| 4.2.0 | AI HBM | 100 | Flops/byte |
"""
        kernel = roofline_kernel.parse_rocprof_compute(text, dtypes=["fp16"])[0]
        self.assertEqual(kernel["metrics"]["compute_metric"], "MFMA FLOPs (F16)")

    def test_all_na_values_remain_unknown(self):
        text = """
Kernel 0: empty_kernel
| 4.1.0 | VALU FLOPs | N/A | Gflop/s | N/A |
| 4.1.7 | HBM Bandwidth | N/A | GB/s | N/A |
| 4.2.0 | AI HBM | N/A | Flops/byte |
| 4.2.3 | Performance (GFLOPs) | N/A | Gflop/s |
"""
        kernel = roofline_kernel.parse_rocprof_compute(text)[0]
        self.assertIsNone(kernel["metrics"]["performance_gflops"])
        self.assertIsNone(kernel["metrics"]["roofline_efficiency_pct"])
        self.assertEqual(kernel["metrics"]["peak_basis"], "unavailable")
        self.assertEqual(kernel["classification"]["theoretical_bound"], "unknown")
        self.assertEqual(kernel["classification"]["observed_limit"], "unknown")
        self.assertFalse(roofline_kernel._has_valid_kernel_metrics(kernel))

    def test_peak_only_rows_are_not_valid_observations(self):
        text = """
Kernel 0: peak_only
| 4.1.0 | MFMA FLOPs (F16) | N/A | Gflop/s | 1000 |
| 4.1.7 | HBM Bandwidth | N/A | GB/s | 100 |
| 4.2.0 | AI HBM | N/A | Flops/byte |
"""
        kernel = roofline_kernel.parse_rocprof_compute(text, dtypes=["fp16"])[0]
        self.assertFalse(roofline_kernel._has_valid_kernel_metrics(kernel))


class CollectionHelperTests(unittest.TestCase):
    def test_string_command_is_explicitly_wrapped(self):
        arguments, wrapped = roofline_kernel._command_arguments("python bench.py")
        self.assertEqual(arguments, ["bash", "-lc", "python bench.py"])
        self.assertTrue(wrapped)

    def test_list_command_stays_an_argument_array(self):
        arguments, wrapped = roofline_kernel._command_arguments(
            ["python", "bench.py", "--shape", "1024"]
        )
        self.assertEqual(arguments[2], "--shape")
        self.assertFalse(wrapped)

    def test_summary_ranks_weight_times_headroom_without_ai_average(self):
        cases = [
            {
                "case_id": "small",
                "status": "matched",
                "weight": 2.0,
                "metrics": {"headroom_ratio": 4.0, "ai_hbm": 10.0},
                "classification": {"recommended_specialties": ["memory"]},
            },
            {
                "case_id": "large",
                "status": "matched",
                "weight": 10.0,
                "metrics": {"headroom_ratio": 1.5, "ai_hbm": 1000.0},
                "classification": {"recommended_specialties": ["compute"]},
            },
        ]
        summary = roofline_kernel.build_summary(cases)
        self.assertEqual(summary["priority_order"][0]["case_id"], "large")
        self.assertEqual(summary["recommended_specialties"], ["compute", "memory"])
        self.assertNotIn("ai_hbm", summary)

    def test_analysis_data_path_finds_nested_soc_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            soc_dir = os.path.join(directory, "case", "MI300X")
            os.makedirs(soc_dir)
            for name in ("roofline.csv", "sysinfo.csv"):
                with open(os.path.join(soc_dir, name), "w", encoding="utf-8"):
                    pass
            self.assertEqual(
                roofline_kernel._analysis_data_path(directory), soc_dir
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
