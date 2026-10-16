import copy
import json
import unittest

from geak_evo.evaluate.timing import (
    MeasurementError,
    analyze_measurements,
    format_json,
    format_text,
)


JOB = "1" * 64
BASE = "2" * 64
CANDIDATE = "3" * 64
BASE_WORK = "4" * 64
NEW_WORK = "5" * 64


def record():
    result = {
        "schema_version": 1,
        "job_id": JOB,
        "baseline_source_id": BASE,
        "candidate_source_id": CANDIDATE,
        "measurement_epoch_id": "epoch-1",
        "search_start_id": "search-A",
        "cases": [],
    }
    for case_id, count, pairs in (
        ("prefill", 3, [(10, 8), (12, 9), (11, 10)]),
        ("decode", 1, [(2, 3), (3, 4), (4, 5)]),
    ):
        case = {
            "case_id": case_id,
            "observed_frequency": {"count": count, "unit": "calls", "provenance_ref": "trace.json"},
            "baseline_work_id": BASE_WORK,
            "candidate_work_id": NEW_WORK,
            "blocks": [],
        }
        for index, (baseline, candidate) in enumerate(pairs):
            case["blocks"].append({
                "block_id": "block-" + str(index),
                "job_id": JOB,
                "case_id": case_id,
                "measurement_epoch_id": "epoch-1",
                "baseline": {"source_id": BASE, "work_id": BASE_WORK, "kind": "unprofiled", "samples": [baseline], "unit": "us"},
                "candidate": {"source_id": CANDIDATE, "work_id": NEW_WORK, "kind": "unprofiled", "samples": [candidate], "unit": "us"},
            })
        result["cases"].append(case)
    return result


def bound(case_id="prefill"):
    return {
        "job_id": JOB,
        "case_id": case_id,
        "source_id": CANDIDATE,
        "work_id": NEW_WORK,
        "measurement_epoch_id": "epoch-1",
        "kind": "modeled_bound",
        "bound_id": "6" * 64,
        "latency_bound": {"value": 6, "unit": "us"},
        "justification_ref": "bound-derivation.json",
        "assumptions": ["Illustrative synthetic bound; no physical calibration was performed."],
        "operation_work": {"work_id": NEW_WORK, "dtype": "bf16", "operation_class": "tensor_matmul", "unit": "FLOP", "amount": 1200, "scope": "full registered callable", "provenance_ref": "work.json"},
        "physical_traffic": {"kind": "modeled", "amount": 600, "unit": "bytes", "level": "HBM", "scope": "physical read and write transfers for full callable", "provenance_ref": "traffic.json"},
        "calibrated_rates": [
            {"calibration_id": "7" * 64, "quantity": "compute", "value": 2e8, "unit": "FLOP/s", "device_id": "synthetic-device", "runtime_id": "synthetic-runtime", "calibration_epoch_id": "calibration-1", "dtype": "bf16", "operation_class": "tensor_matmul", "scope": "synthetic compatible instruction class", "provenance_ref": "calibration.json", "compatibility_ref": "compatibility.json"},
            {"calibration_id": "8" * 64, "quantity": "bandwidth", "value": 1e8, "unit": "bytes/s", "device_id": "synthetic-device", "runtime_id": "synthetic-runtime", "calibration_epoch_id": "calibration-1", "dtype": "not_applicable_byte_stream", "operation_class": "HBM transfer", "scope": "synthetic matching physical memory boundary", "provenance_ref": "bandwidth.json", "compatibility_ref": "compatibility.json"},
        ],
    }


def analyze(value, **kwargs):
    return analyze_measurements(value, timing_aggregation="mean", resamples=200, **kwargs)


class MeasurementsTests(unittest.TestCase):
    def test_registered_aggregation_is_required_and_rank_reversal_is_rejected(self):
        value = record()
        for case in value["cases"]:
            for block in case["blocks"]:
                block["baseline"]["samples"] = [1, 1, 100]
                block["candidate"]["samples"] = [2, 2, 2]
        # This endpoint improves under means (34/2), but regresses under
        # medians (1/2). Never silently answer the latter job with the former.
        self.assertAlmostEqual(analyze(value)["aggregate"]["estimate"]["speedup"], 17)
        for aggregation in ("median", "minimum", None):
            with self.subTest(aggregation=aggregation), self.assertRaisesRegex(MeasurementError, "supports mean timing only"):
                analyze_measurements(value, timing_aggregation=aggregation)
        with self.assertRaisesRegex(TypeError, "timing_aggregation"):
            analyze_measurements(value)

    def test_weighted_latency_preserves_regressions(self):
        result = analyze(record())
        cases = {case["case_id"]: case for case in result["cases"]}
        self.assertEqual(cases["decode"]["status"], "regression")
        self.assertEqual(cases["prefill"]["status"], "improvement")
        estimate = result["aggregate"]["estimate"]
        self.assertAlmostEqual(estimate["baseline_s"], 9e-6)
        self.assertAlmostEqual(estimate["candidate_s"], 7.75e-6)
        self.assertAlmostEqual(estimate["speedup"], 9 / 7.75)
        self.assertAlmostEqual(result["aggregate"]["observed_total_candidate_s"], 31e-6)
        self.assertNotAlmostEqual(estimate["speedup"], ((11 / 9) * (3 / 4)) ** 0.5)

    def test_bootstrap_is_deterministic_and_order_independent(self):
        value = record()
        original = copy.deepcopy(value)
        first = analyze(value, seed=123)
        self.assertEqual(value, original)
        self.assertEqual(first, analyze(value, seed=123))
        value["cases"].reverse()
        self.assertEqual(first, analyze(value, seed=123))
        other = analyze(value, seed=1000)
        self.assertNotEqual(first["cases"][1]["interval_95"], other["cases"][1]["interval_95"])

    def test_block_resampling_keeps_pairing(self):
        value = record()
        for case in value["cases"]:
            for block in case["blocks"]:
                block["candidate"]["samples"] = [number * 0.5 for number in block["baseline"]["samples"]]
        result = analyze(value)
        for case in result["cases"]:
            self.assertAlmostEqual(case["interval_95"]["speedup"][0], 2)
            self.assertAlmostEqual(case["interval_95"]["speedup"][1], 2)
        self.assertAlmostEqual(result["aggregate"]["interval_95"]["speedup"][0], 2)

    def test_blocks_have_equal_weight_not_raw_sample_pooling(self):
        value = record()
        value["cases"][0]["blocks"][0]["baseline"]["samples"] *= 100
        repeated = analyze(value)
        original = analyze(record())
        for before, after in zip(original["cases"], repeated["cases"]):
            for key in before["estimate"]:
                self.assertAlmostEqual(before["estimate"][key], after["estimate"][key])
            for key in before["interval_95"]:
                for lower_or_upper in (0, 1):
                    self.assertAlmostEqual(before["interval_95"][key][lower_or_upper], after["interval_95"][key][lower_or_upper])

    def test_interval_means_match_raw_block_estimates_without_call_pooling(self):
        raw = record()
        intervals = copy.deepcopy(raw)
        for raw_case, interval_case in zip(raw["cases"], intervals["cases"]):
            for index, (raw_block, interval_block) in enumerate(zip(raw_case["blocks"], interval_case["blocks"])):
                for role_index, role in enumerate(("baseline", "candidate")):
                    mean = raw_block[role]["samples"][0]
                    samples = [mean - 1, mean + 1] * (index + role_index + 1)
                    raw_block[role].update({"samples": samples, "unit": "s"})
                    del interval_block[role]["samples"]
                    interval_block[role].update({
                        "kind": "unprofiled_interval_mean", "value": mean,
                        "unit": "s", "timed_calls": len(samples),
                        "raw_report_ref": {"path": role + "-" + str(index) + ".json", "sha256": "a" * 64},
                    })
        result = analyze(intervals)
        self.assertEqual(result, analyze(raw))
        self.assertEqual(result["aggregate"]["estimate"]["baseline_s"], 9)
        self.assertEqual(result["aggregate"]["estimate"]["candidate_s"], 7.75)
        self.assertEqual(json.loads(format_json(result, raw_record=intervals))["raw_evidence"], intervals)

    def test_interval_means_reject_ambiguous_or_invalid_evidence(self):
        timing = {
            "source_id": CANDIDATE, "work_id": NEW_WORK,
            "kind": "unprofiled_interval_mean", "value": 3, "unit": "us",
            "timed_calls": 4, "raw_report_ref": {"path": "native-report.json", "sha256": "a" * 64},
        }
        invalid = [("samples", []), ("samples", None), ("unit", "cycles"),
                   ("source_id", BASE), ("work_id", BASE_WORK)]
        invalid += [("value", item) for item in (None, 0, -1, float("nan"), float("inf"), True, "3")]
        invalid += [("timed_calls", item) for item in (None, 0, -1, True, 1.5)]
        invalid += [("raw_report_ref", item) for item in (
            None, "native-report.json", {"path": "native-report.json"}, {"sha256": "a" * 64},
            {"path": "native-report.json", "sha256": "a" * 64, "extra": True},
            {"path": " ", "sha256": "a" * 64}, {"path": "native-report.json", "sha256": "A" * 64},
        )]
        for key, item in invalid:
            value = record()
            value["cases"][1]["blocks"][0]["candidate"] = {**timing, key: item}
            with self.subTest(key=key, item=item), self.assertRaises(MeasurementError):
                analyze(value)

    def test_shared_schedule_preserves_cross_case_block_covariance(self):
        value = record()
        for case_index, case in enumerate(value["cases"]):
            case["observed_frequency"]["count"] = 1
            case["blocks"] = case["blocks"][:2]
            for block_index, block in enumerate(case["blocks"]):
                block["baseline"]["samples"] = [10]
                block["candidate"]["samples"] = [2 if case_index == block_index else 8]
        independent = analyze(value)
        value["block_schedule"] = "shared"
        shared = analyze(value)
        # Each actual block has mean candidate latency 5 us across its cases.
        # Independent case draws create variability absent from those blocks.
        interval = shared["aggregate"]["interval_95"]["candidate_s"]
        self.assertEqual(interval[0], interval[1])
        self.assertAlmostEqual(interval[0], 5e-6)
        independent_interval = independent["aggregate"]["interval_95"]["candidate_s"]
        self.assertLess(independent_interval[0], independent_interval[1])
        self.assertIn("shared block resampling", shared["aggregate"]["interval_assumption"])
        reversed_cases = copy.deepcopy(value)
        reversed_cases["cases"].reverse()
        self.assertEqual(shared, analyze(reversed_cases))
        for invalid_schedule in (None, "independent", "other"):
            invalid = copy.deepcopy(value)
            invalid["block_schedule"] = invalid_schedule
            with self.subTest(schedule=invalid_schedule), self.assertRaisesRegex(MeasurementError, "block_schedule"):
                analyze(invalid)
        for changed_blocks in (list(reversed(value["cases"][0]["blocks"])), value["cases"][0]["blocks"][:1]):
            invalid = copy.deepcopy(value)
            invalid["cases"][0]["blocks"] = changed_blocks
            with self.subTest(blocks=changed_blocks), self.assertRaisesRegex(MeasurementError, "same ordered block_id"):
                analyze(invalid)

    def test_units_are_explicit_and_normalized(self):
        value = record()
        value["cases"][0]["blocks"][0]["baseline"].update({"samples": [10000], "unit": "ns"})
        value["cases"][0]["blocks"][0]["candidate"].update({"samples": [0.008], "unit": "ms"})
        self.assertAlmostEqual(analyze(value)["aggregate"]["estimate"]["speedup"], 9 / 7.75)
        value["cases"][0]["blocks"][0]["candidate"]["unit"] = "cycles"
        with self.assertRaisesRegex(MeasurementError, "unit"):
            analyze(value)

    def test_raw_source_epoch_job_and_case_mismatches_fail(self):
        for key, bad_value in (("source_id", "9" * 64), ("work_id", BASE_WORK)):
            value = record()
            value["cases"][0]["blocks"][0]["candidate"][key] = bad_value
            with self.subTest(key=key), self.assertRaisesRegex(MeasurementError, "mismatch"):
                analyze(value)
        for key, bad_value in (("measurement_epoch_id", "epoch-2"), ("job_id", "9" * 64), ("case_id", "decode")):
            value = record()
            value["cases"][0]["blocks"][0][key] = bad_value
            with self.subTest(key=key), self.assertRaisesRegex(MeasurementError, "mismatch"):
                analyze(value)

    def test_trusted_host_binding_and_frozen_job_weights(self):
        value = record()
        expected = {key: value[key] for key in ("job_id", "baseline_source_id", "candidate_source_id", "measurement_epoch_id")}
        analyze(value, expected_binding=expected, observed_cases=copy.deepcopy(value["cases"]))
        expected["candidate_source_id"] = "9" * 64
        with self.assertRaisesRegex(MeasurementError, "mismatch"):
            analyze(value, expected_binding=expected)
        cases = copy.deepcopy(value["cases"])
        cases[0]["observed_frequency"]["count"] += 1
        with self.assertRaisesRegex(MeasurementError, "frozen observed frequencies"):
            analyze(value, observed_cases=cases)
        with self.assertRaisesRegex(MeasurementError, "frozen observed frequencies"):
            analyze(value, observed_cases=cases[1:])

    def test_unpaired_or_duplicate_blocks_fail(self):
        value = record()
        del value["cases"][0]["blocks"][0]["candidate"]
        with self.assertRaisesRegex(MeasurementError, "unpaired"):
            analyze(value)
        value = record()
        value["cases"][0]["blocks"].append(copy.deepcopy(value["cases"][0]["blocks"][0]))
        with self.assertRaisesRegex(MeasurementError, "duplicate block_id"):
            analyze(value)

    def test_profiler_timings_cannot_be_scored(self):
        value = record()
        value["cases"][0]["blocks"][0]["candidate"]["kind"] = "profiled"
        with self.assertRaisesRegex(MeasurementError, "unprofiled"):
            analyze(value)

    def test_invalid_latencies_fail(self):
        for invalid in (-1, 0, float("nan"), float("inf"), True, "10"):
            value = record()
            value["cases"][0]["blocks"][0]["candidate"]["samples"] = [invalid]
            with self.subTest(invalid=invalid), self.assertRaises(MeasurementError):
                analyze(value)

    def test_unobserved_weights_cannot_enter_aggregate(self):
        for invalid in (0, -1, 0.5, True, None):
            value = record()
            value["cases"][0]["observed_frequency"]["count"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(MeasurementError, "unobserved"):
                analyze(value)
        value = record()
        value["cases"][0]["inferred_frequency"] = {"count": 1e15, "unit": "calls", "basis_ref": "guess.json"}
        self.assertEqual(analyze(value), analyze(record()))

    def test_unknown_case_prevents_partial_aggregate(self):
        value = record()
        value["cases"][1].update({"blocks": [], "unknown_reason": "candidate failed correctness; timing withheld"})
        result = analyze(value)
        self.assertEqual(result["aggregate"]["status"], "unknown")
        self.assertIsNone(result["aggregate"]["estimate"])
        self.assertEqual(result["aggregate"]["missing_case_ids"], ["decode"])
        self.assertEqual(result["cases"][1]["status"], "improvement")

    def test_single_block_has_unknown_uncertainty(self):
        value = record()
        value["cases"][0]["blocks"] = value["cases"][0]["blocks"][:1]
        result = analyze(value)
        self.assertIsNone(result["cases"][1]["interval_95"])
        self.assertIsNone(result["aggregate"]["interval_95"])
        self.assertEqual(result["aggregate"]["status"], "complete")

    def test_source_bound_roofline_ratio_and_changed_work(self):
        value = record()
        value["cases"][0]["roofline"] = bound()
        roof = analyze(value)["cases"][1]["roofline"]
        self.assertEqual(roof["status"], "assessed")
        self.assertAlmostEqual(roof["bound_ratio_percent"], 100 * 6 / 9)
        # An algorithm may change work; reusing the baseline work identity is forbidden.
        value["cases"][0]["roofline"]["work_id"] = BASE_WORK
        with self.assertRaisesRegex(MeasurementError, "work_id mismatch"):
            analyze(value)
        value["cases"][0]["roofline"] = bound()
        value["cases"][0]["roofline"]["operation_work"]["work_id"] = BASE_WORK
        with self.assertRaisesRegex(MeasurementError, "work_id mismatch"):
            analyze(value)

    def test_roofline_epoch_source_binding(self):
        for key, bad_value in (("measurement_epoch_id", "epoch-2"), ("source_id", BASE)):
            value = record()
            value["cases"][0]["roofline"] = bound()
            value["cases"][0]["roofline"][key] = bad_value
            with self.subTest(key=key), self.assertRaisesRegex(MeasurementError, "mismatch"):
                analyze(value)

    def test_above_100_bound_is_unclipped_inconsistency(self):
        value = record()
        value["cases"][0]["roofline"] = bound()
        value["cases"][0]["roofline"]["latency_bound"]["value"] = 18
        result = analyze(value)
        roof = result["cases"][1]["roofline"]
        self.assertAlmostEqual(roof["bound_ratio_percent"], 200)
        self.assertEqual(roof["status"], "inconsistent_above_100")
        self.assertIn("200.000%", format_text(result))

    def test_missing_dtype_or_calibration_stays_unknown(self):
        value = record()
        self.assertEqual(analyze(value)["cases"][1]["roofline"]["status"], "unknown")
        value["cases"][0]["roofline"] = bound()
        value["cases"][0]["roofline"]["operation_work"]["dtype"] = None
        result = analyze(value)["cases"][1]["roofline"]
        self.assertEqual(result["status"], "unknown")
        self.assertIn("dtype", result["reason"])
        value["cases"][0]["roofline"]["operation_work"]["dtype"] = "unknown"
        self.assertEqual(analyze(value)["cases"][1]["roofline"]["status"], "unknown")
        value["cases"][0]["roofline"] = bound()
        value["cases"][0]["roofline"]["calibrated_rates"] = []
        self.assertEqual(analyze(value)["cases"][1]["roofline"]["status"], "unknown")

    def test_dtype_and_unit_mismatches_do_not_fallback(self):
        for key, invalid in (("dtype", "fp8"), ("operation_class", "scalar_add"), ("unit", "TFLOP/s")):
            value = record()
            value["cases"][0]["roofline"] = bound()
            value["cases"][0]["roofline"]["calibrated_rates"][0][key] = invalid
            with self.subTest(key=key), self.assertRaisesRegex(MeasurementError, "mismatch|unit"):
                analyze(value)

    def test_negative_and_nan_bound_quantities_fail(self):
        for container, key in (("latency_bound", "value"), ("operation_work", "amount"), ("physical_traffic", "amount")):
            for invalid in (-1, float("nan")):
                value = record()
                value["cases"][0]["roofline"] = bound()
                value["cases"][0]["roofline"][container][key] = invalid
                with self.subTest(container=container, invalid=invalid), self.assertRaises(MeasurementError):
                    analyze(value)

    def test_counter_evidence_remains_distinct(self):
        value = record()
        value["cases"][0]["roofline"] = bound()
        roof = value["cases"][0]["roofline"]
        roof["kind"] = "counter_informed_bound"
        with self.assertRaisesRegex(MeasurementError, "evidence kind disagree"):
            analyze(value)
        roof["physical_traffic"].update({"kind": "empirical_counter", "counter_ids": ["official-id"], "profiler_id": "synthetic-profiler", "raw_artifact_ref": "counters.json", "counter_definition_ref": "definitions.json"})
        result = analyze(value)
        self.assertEqual(result["cases"][1]["roofline"]["kind"], "counter_informed_bound")
        self.assertEqual(result["cases"][1]["block_count"], 3)

    def test_measured_reference_is_not_a_bound(self):
        value = record()
        value["cases"][0]["measured_reference"] = {
            "kind": "measured_reference", "job_id": JOB, "case_id": "prefill", "measurement_epoch_id": "epoch-1",
            "source_id": "9" * 64, "work_id": "a" * 64, "provenance_ref": "reference.json",
            "timing": {"source_id": "9" * 64, "work_id": "a" * 64, "kind": "unprofiled", "samples": [10, 12], "unit": "us"},
        }
        result = analyze(value)["cases"][1]
        self.assertEqual(result["roofline"]["status"], "unknown")
        self.assertAlmostEqual(result["measured_reference"]["reference_over_candidate"], 11 / 9)
        self.assertIn("not a lower latency bound", result["measured_reference"]["interpretation"])

    def test_full_json_preserves_raw_evidence_and_text_is_case_records(self):
        value = record()
        result = analyze(value)
        serialized = json.loads(format_json(result, raw_record=value))
        self.assertEqual(serialized["raw_evidence"], value)
        self.assertEqual(serialized["analysis"], result)
        rendered = format_text(result)
        self.assertIn("case decode: regression", rendered)
        self.assertIn("case prefill: improvement", rendered)
        self.assertIn("not independent search starts", rendered)
        self.assertNotIn("|", rendered)
        self.assertLessEqual(max(len(line) for line in rendered.splitlines()), 88)


if __name__ == "__main__":
    unittest.main()
