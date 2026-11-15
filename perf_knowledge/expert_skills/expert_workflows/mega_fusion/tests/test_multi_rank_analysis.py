"""Unit tests for `multi_rank_analysis` (stdlib only; no pytest, no torch, no GPU needed).

These lock two things:

1. **Shape fidelity.** Fed synthetic per-rank records shaped like the one existing multi-rank
   analysis artifact this framework generalizes (`artifacts/analysis/mega_moe_v2_analysis.json`,
   schema `geak-megamoe-analysis-v1`), `merge_rank_records`/`bucket_trace_events`/`build_report`
   must reproduce that artifact's field names and nesting — proving the generic library can stand in
   for the one-off generator without containing any operator-specific code.
2. **Never raises on partial/malformed input.** A missing metric on some ranks, an unreadable trace
   file, or a bad category-map pattern must degrade (excluded ranks / `error` field / `__errors__`
   key), never throw, so one bad rank cannot block the rest of an analysis.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from multi_rank_analysis import (  # noqa: E402
    ANALYSIS_BUNDLE_SCHEMA_VERSION,
    COLLECTION_PROVENANCE_SCHEMA_VERSION,
    EVIDENCE_CATALOG_SCHEMA_VERSION,
    EXPERIMENT_SCHEMA_VERSION,
    HARDWARE_CONTEXT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    analyze_category_overlap,
    bucket_trace_events,
    bundle_from_rank_report,
    build_report,
    classify_case_speedup,
    compare_controlled_variants,
    critical_path,
    load_category_map,
    load_instruction_category_map,
    merge_intervals,
    merge_rank_records,
    parse_att_stats_csv,
    read_att_occupancy,
    resolve_measurement_tracks,
    validate_experiment_manifest,
    validate_analysis_bundle,
    validate_collection_provenance,
    validate_hardware_context,
)

WORLD_SIZE = 8


def _synthetic_rank_records(base_ms=0.49, skew_rank=None, skew_factor=3.0):
    """8 per-rank timing_ms.stage1 records; optionally skew one rank (models route imbalance)."""
    records = []
    for r in range(WORLD_SIZE):
        ms = base_ms * skew_factor if r == skew_rank else base_ms
        records.append({"rank": r, "timing_ms": {"stage1": ms, "e2e": ms * 1.5}})
    return records


def _collection_provenance(raw_artifact="raw.json"):
    return {
        "schema_version": COLLECTION_PROVENANCE_SCHEMA_VERSION,
        "collector_id": "test",
        "tool_version": "1.0",
        "command": "test --collect",
        "timestamp": "2026-01-01T00:00:00Z",
        "scope": "synthetic test",
        "repetitions": 1,
        "raw_artifacts": [raw_artifact],
        "confidence": "high",
        "profiler_perturbation_pct": 0.0,
        "cross_checks": [],
        "units": {"synthetic_metric": "synthetic_unit"},
    }


class TestMergeRankRecords(unittest.TestCase):
    def test_uniform_records_shape_matches_artifact(self):
        records = _synthetic_rank_records()
        merged = merge_rank_records(records, ["timing_ms.stage1", "timing_ms.e2e"])
        for path in ("timing_ms.stage1", "timing_ms.e2e"):
            entry = merged[path]
            for key in ("rank_mean", "rank_max", "rank_min", "rank_tail_spread_pct", "missing_ranks"):
                self.assertIn(key, entry, f"{path} missing key {key}")
        self.assertAlmostEqual(merged["timing_ms.stage1"]["rank_mean"], 0.49, places=6)
        self.assertAlmostEqual(merged["timing_ms.stage1"]["rank_max"], 0.49, places=6)
        self.assertEqual(merged["timing_ms.stage1"]["rank_tail_spread_pct"], 0.0)
        self.assertEqual(merged["timing_ms.stage1"]["missing_ranks"], [])

    def test_skewed_rank_raises_tail_spread(self):
        records = _synthetic_rank_records(skew_rank=3, skew_factor=48.0)  # ~expert_max_to_mean seen in evidence
        merged = merge_rank_records(records, ["timing_ms.stage1"])
        entry = merged["timing_ms.stage1"]
        self.assertGreater(entry["rank_max"], entry["rank_min"])
        self.assertGreater(entry["rank_tail_spread_pct"], 600.0)

    def test_missing_metric_on_some_ranks_excludes_not_raises(self):
        records = _synthetic_rank_records()
        del records[5]["timing_ms"]["stage1"]  # rank 5 never produced this metric
        merged = merge_rank_records(records, ["timing_ms.stage1"])
        entry = merged["timing_ms.stage1"]
        self.assertEqual(entry["missing_ranks"], [5])
        self.assertIsNotNone(entry["rank_mean"])  # still computed from the remaining 7 ranks

    def test_all_missing_degrades_to_none_not_raise(self):
        records = [{"rank": rank, "timing_ms": {}} for rank in range(WORLD_SIZE)]
        merged = merge_rank_records(records, ["timing_ms.stage1"])
        entry = merged["timing_ms.stage1"]
        self.assertEqual(len(entry["missing_ranks"]), WORLD_SIZE)
        self.assertIsNone(entry["rank_mean"])
        self.assertIsNone(entry["rank_tail_spread_pct"])

    def test_repetitions_produce_runs_and_span_pct(self):
        reps = [_synthetic_rank_records(base_ms=m) for m in (0.4958, 0.4977, 0.4947)]
        merged = merge_rank_records(reps[0], ["timing_ms.stage1"], repetitions=reps)
        entry = merged["timing_ms.stage1"]
        self.assertEqual(len(entry["rank_mean_runs"]), 3)
        self.assertEqual(len(entry["rank_max_runs"]), 3)
        self.assertGreaterEqual(entry["rank_mean_span_pct"], 0.0)
        self.assertLess(entry["rank_mean_span_pct"], 5.0)  # matches the noise band seen in evidence
        self.assertAlmostEqual(
            entry["rank_max"],
            sum(entry["rank_max_runs"]) / len(entry["rank_max_runs"]),
        )

    def test_spread_uses_mean_denominator(self):
        records = [
            {"rank": 0, "metric": 1.0},
            {"rank": 1, "metric": 3.0},
        ]
        entry = merge_rank_records(records, ["metric"])["metric"]
        self.assertAlmostEqual(entry["rank_tail_spread_pct"], 100.0)

    def test_boolean_metric_is_missing_not_numeric(self):
        records = [{"rank": 0, "metric": True}, {"rank": 1, "metric": 2.0}]
        entry = merge_rank_records(records, ["metric"])["metric"]
        self.assertEqual(entry["missing_ranks"], [0])

    def test_duplicate_rank_rejected(self):
        records = _synthetic_rank_records()
        records[-1]["rank"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate rank"):
            merge_rank_records(records, ["timing_ms.stage1"])

    def test_expected_rank_absence_uses_real_rank_ids(self):
        records = [
            record
            for record in reversed(_synthetic_rank_records())
            if record["rank"] != 0
        ]
        merged = merge_rank_records(
            records,
            ["timing_ms.stage1"],
            expected_ranks=range(WORLD_SIZE),
        )
        self.assertEqual(merged["timing_ms.stage1"]["missing_ranks"], [0])


class TestClassifyCaseSpeedup(unittest.TestCase):
    def test_pass_above_floor(self):
        r = classify_case_speedup(candidate_rank_max=0.30, baseline_rank_max=0.84, min_pct=110.0)
        self.assertEqual(r["status"], "pass")
        self.assertGreater(r["speedup_pct"], 110.0)

    def test_fail_below_floor(self):
        r = classify_case_speedup(candidate_rank_max=0.80, baseline_rank_max=0.84, min_pct=110.0)
        self.assertEqual(r["status"], "fail")

    def test_unguarded_when_no_floor_given(self):
        r = classify_case_speedup(candidate_rank_max=0.30, baseline_rank_max=0.84)
        self.assertEqual(r["status"], "unguarded")

    def test_raises_on_nonpositive_baseline(self):
        with self.assertRaises(ValueError):
            classify_case_speedup(candidate_rank_max=0.3, baseline_rank_max=0.0)

    def test_rejects_invalid_candidate_latency(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                classify_case_speedup(candidate_rank_max=value, baseline_rank_max=1.0)


class TestTraceCategories(unittest.TestCase):
    def _write_trace(self, kernel_specs):
        """kernel_specs: list of (name, dur_us)."""
        events = [
            {"ph": "X", "cat": "kernel", "name": name, "pid": 2, "tid": 3, "ts": 0.0, "dur": dur}
            for name, dur in kernel_specs
        ]
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"traceEvents": events}, f)
        f.close()
        return f.name

    def test_bucketing_matches_evidence_categories(self):
        cat_map = load_category_map({
            "stage1": "megamoe_stage1",
            "stage2": "megamoe_stage2",
            "combine": "ep_combine",
            "quantize": "per_1x32_mx_quant",
        })
        path = self._write_trace([
            ("megamoe_stage1_compact_t128x512x256_w8", 2784.7),
            ("megamoe_stage2_compact_t64x256x256_sbm128", 2124.1),
            ("ep_combine_intranode_0", 312.5),
            ("per_1x32_mx_quant_fp8_n7168", 32.5),
            ("hipDeviceSynchronize", 15208.2),  # cuda_runtime, not "kernel" cat below -> excluded anyway
        ])
        try:
            result = bucket_trace_events(path, cat_map)
        finally:
            os.unlink(path)
        self.assertIn("stage1", result["per_category_ms"])
        self.assertIn("stage2", result["per_category_ms"])
        self.assertIn("combine", result["per_category_ms"])
        self.assertIn("quantize", result["per_category_ms"])
        self.assertAlmostEqual(result["per_category_ms"]["combine"], 0.3125, places=4)
        self.assertEqual(result["per_category_event_count"]["combine"], 1)
        self.assertEqual(len(result["per_category_intervals_us"]["combine"]), 1)

    def test_unmatched_kernel_goes_to_unclassified_not_dropped(self):
        cat_map = load_category_map({"stage1": "megamoe_stage1"})
        path = self._write_trace([("some_other_kernel_xyz", 100.0)])
        try:
            result = bucket_trace_events(path, cat_map)
        finally:
            os.unlink(path)
        self.assertIn("unclassified", result["per_category_ms"])
        self.assertNotIn("stage1", result["per_category_ms"])

    def test_unreadable_trace_degrades_not_raises(self):
        result = bucket_trace_events("/nonexistent/path/does_not_exist.json", {})
        self.assertIn("error", result)
        self.assertEqual(result["per_category_ms"], {})

    def test_bad_pattern_recorded_in_errors_not_raised(self):
        cat_map = load_category_map({"good": "megamoe_stage1", "bad": "("})
        self.assertIn("good", cat_map)
        self.assertIn("__errors__", cat_map)
        self.assertEqual(cat_map["__errors__"][0]["category"], "bad")

    def test_malformed_kernel_event_is_reported_not_raised(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "megamoe_stage1",
                        "ts": 0.0,
                        "dur": "bad",
                    }
                ]
            },
            f,
        )
        f.close()
        try:
            result = bucket_trace_events(
                f.name,
                load_category_map({"stage1": "megamoe_stage1"}),
            )
        finally:
            os.unlink(f.name)
        self.assertEqual(result["event_count"], 0)
        self.assertEqual(len(result["malformed_events"]), 1)


class TestATTInstructionAnalysis(unittest.TestCase):
    def _write_csv(self, rows, fields=None):
        fields = fields or [
            "CodeObj",
            "Vaddr",
            "Instruction",
            "Hitcount",
            "Latency",
            "Stall",
            "Idle",
            "Source",
        ]
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.close()
        return f.name

    def test_att_categories_and_cycle_semantics(self):
        path = self._write_csv(
            [
                {
                    "Instruction": "v_mfma_f32_16x16x32_fp8",
                    "Hitcount": "10",
                    "Latency": "100",
                    "Stall": "20",
                    "Idle": "5",
                },
                {
                    "Instruction": "s_barrier",
                    "Hitcount": "2",
                    "Latency": "20",
                    "Stall": "10",
                    "Idle": "80",
                },
                {
                    "Instruction": "buffer_load_dwordx4",
                    "Hitcount": "8",
                    "Latency": "40",
                    "Stall": "30",
                    "Idle": "0",
                },
            ]
        )
        category_map = load_instruction_category_map(
            {
                "mfma": "^v_mfma",
                "barrier": "^s_barrier",
                "vmem_read": "^buffer_load",
            }
        )
        try:
            report = parse_att_stats_csv(path, category_map)
        finally:
            os.unlink(path)
        self.assertEqual(report["categories"]["mfma"]["hitcount"], 10)
        self.assertEqual(report["categories"]["barrier"]["idle_cycles"], 80)
        self.assertEqual(report["totals"]["accounted_cycles"], 245)
        self.assertEqual(report["totals"]["issue_execute_cycles"], 100)
        self.assertAlmostEqual(
            report["categories"]["mfma"]["stall_within_latency_pct"],
            20.0,
        )
        self.assertEqual(
            report["top_stall_idle_instructions"][0]["category"],
            "barrier",
        )
        self.assertIn("sampled thread/wave", report["scope_warning"])

    def test_att_rejects_stall_greater_than_latency(self):
        path = self._write_csv(
            [
                {
                    "Instruction": "s_barrier",
                    "Hitcount": "1",
                    "Latency": "5",
                    "Stall": "6",
                    "Idle": "0",
                }
            ]
        )
        try:
            with self.assertRaisesRegex(ValueError, "Stall greater than Latency"):
                parse_att_stats_csv(path, {})
        finally:
            os.unlink(path)

    def test_att_missing_columns_rejected(self):
        path = self._write_csv([], fields=["Instruction", "Hitcount"])
        try:
            with self.assertRaisesRegex(ValueError, "missing columns"):
                parse_att_stats_csv(path, {})
        finally:
            os.unlink(path)

    def _write_occupancy(self, payload):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(payload, f)
        f.close()
        return f.name

    def test_att_occupancy_summarizes_balanced_wave_events(self):
        path = self._write_occupancy(
            {
                "version": "3.0.0",
                "dispatches": {"1": "kernel"},
                "0": [
                    [0, 0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 1, 1],
                    [10, 0, 0, 0, 0, 1],
                    [20, 1, 0, 0, 0, 1],
                ],
            }
        )
        try:
            report = read_att_occupancy(path)
        finally:
            os.unlink(path)
        self.assertEqual(report["schema_version"], "geak-att-occupancy-summary-v1")
        self.assertTrue(report["balanced_events"])
        self.assertEqual(report["sampled_cu_count"], 2)
        self.assertEqual(report["peak_active_waves_per_cu"], 1)
        self.assertAlmostEqual(report["avg_active_waves_per_cu"], 0.75)
        self.assertAlmostEqual(report["wave_slot_occupancy_pct"], 2.34375)
        self.assertEqual(
            report["shader_engines"]["0"]["per_cu"]["0"]["wave_cycles"],
            10,
        )

    def test_att_occupancy_reports_unbalanced_events(self):
        path = self._write_occupancy(
            {
                "version": "3.0.0",
                "dispatches": {"1": "kernel"},
                "0": [[0, 0, 0, 0, 0, 1]],
            }
        )
        try:
            report = read_att_occupancy(path)
        finally:
            os.unlink(path)
        self.assertFalse(report["balanced_events"])
        self.assertEqual(report["shader_engines"]["0"]["end_count"], 1)
        self.assertIn(
            "without matching start",
            report["shader_engines"]["0"]["malformed_events"][0]["reason"],
        )

    def test_att_occupancy_rejects_unknown_layout(self):
        path = self._write_occupancy(
            {"version": "4.0.0", "dispatches": {}, "0": []}
        )
        try:
            with self.assertRaisesRegex(ValueError, "unknown event layout"):
                read_att_occupancy(path)
        finally:
            os.unlink(path)


class TestIntervals(unittest.TestCase):
    def test_merge_and_overlap(self):
        self.assertEqual(merge_intervals([[0, 10], [5, 15], [20, 25]]), [[0.0, 15.0], [20.0, 25.0]])
        result = analyze_category_overlap(
            {
                "compute": [[0, 10], [20, 30]],
                "communication": [[5, 25]],
            }
        )
        pair = result["pairwise"]["communication|compute"]
        self.assertAlmostEqual(pair["overlap_ms"], 0.010)
        self.assertAlmostEqual(result["categorized_kernel_union_ms"], 0.030)

    def test_critical_path(self):
        result = critical_path(
            {"dispatch": 1.0, "gemm": 3.0, "combine": 2.0, "shared": 2.5},
            [["dispatch", "gemm"], ["gemm", "combine"]],
        )
        self.assertEqual(result["path"], ["dispatch", "gemm", "combine"])
        self.assertEqual(result["critical_path_ms"], 6.0)

    def test_critical_path_rejects_cycle(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            critical_path({"a": 1, "b": 1}, [["a", "b"], ["b", "a"]])


class TestControlledExperiments(unittest.TestCase):
    def test_manifest_and_variant_comparison(self):
        manifest = validate_experiment_manifest(
            {
                "schema_version": EXPERIMENT_SCHEMA_VERSION,
                "experiment_id": "dpep-vs-tpdp",
                "workload": {"tokens": 512, "seed": 1},
                "invariants": {"tokens": 512, "seed": 1},
                "variants": [
                    {
                        "name": "full",
                        "command": "run-full",
                        "changed_components": [],
                        "case_id": "full",
                        "correctness_status": "pass",
                        "repetitions": 3,
                        "provenance_ref": "prov:full",
                    },
                    {
                        "name": "no_payload",
                        "command": "run-no-payload",
                        "changed_components": ["payload"],
                        "case_id": "no_payload",
                        "correctness_status": "intentional_semantic_change",
                        "repetitions": 3,
                        "provenance_ref": "prov:no-payload",
                    },
                ],
            }
        )
        self.assertEqual(manifest["experiment_id"], "dpep-vs-tpdp")
        self.assertFalse(manifest["delta_additivity_allowed"])
        result = compare_controlled_variants(
            {"latency_ms": 10.0, "xgmi_bytes": 100.0},
            {"no_payload": {"latency_ms": 7.0, "xgmi_bytes": 10.0}},
            {"latency_ms": "lower", "xgmi_bytes": "lower"},
        )
        self.assertAlmostEqual(
            result["variants"]["no_payload"]["latency_ms"]["improvement_pct"],
            30.0,
        )
        self.assertIn("not additive", result["note"])

    def test_manifest_reports_changed_component_overlap(self):
        def variant(name, components):
            return {
                "name": name,
                "command": f"run-{name}",
                "changed_components": components,
                "case_id": name,
                "correctness_status": "pass",
                "repetitions": 3,
                "provenance_ref": f"prov:{name}",
            }

        manifest = validate_experiment_manifest(
            {
                "schema_version": EXPERIMENT_SCHEMA_VERSION,
                "experiment_id": "overlap",
                "workload": {"tokens": 512},
                "invariants": {"tokens": 512},
                "variants": [
                    variant("full", []),
                    variant("no_payload", ["payload", "signals"]),
                    variant("loopback", ["transport", "signals"]),
                ],
            }
        )
        self.assertEqual(
            manifest["overlap_pairs"][0]["changed_components"],
            ["signals"],
        )

    def test_manifest_requires_full_variant(self):
        with self.assertRaisesRegex(ValueError, "full"):
            validate_experiment_manifest(
                {
                    "schema_version": EXPERIMENT_SCHEMA_VERSION,
                    "experiment_id": "bad",
                    "workload": {"tokens": 1},
                    "invariants": {"tokens": 1},
                    "variants": [
                        {
                            "name": "control",
                            "command": "run",
                            "changed_components": ["payload"],
                            "case_id": "control",
                            "correctness_status": "pass",
                            "repetitions": 2,
                            "provenance_ref": "prov:control",
                        }
                    ],
                }
            )


class TestAnalysisBundle(unittest.TestCase):
    def test_normalizes_aiter_cases_rank_shape(self):
        report = {
            "schema_version": "aiter-test-v1",
            "status": "pass",
            "metadata": {"network": "v4_pro", "world_size": WORLD_SIZE},
            "cases": [
                {
                    "case_id": "v4_pro_bs128",
                    "ranks": _synthetic_rank_records(),
                    "repetitions": [
                        _synthetic_rank_records(base_ms=0.49),
                        _synthetic_rank_records(base_ms=0.50),
                    ],
                    "path": "fixed",
                }
            ],
        }
        bundle = bundle_from_rank_report(report, ["timing_ms.e2e"])
        self.assertEqual(bundle["schema_version"], ANALYSIS_BUNDLE_SCHEMA_VERSION)
        self.assertEqual(bundle["cases"][0]["case_id"], "v4_pro_bs128")
        self.assertEqual(len(bundle["cases"][0]["rank_records"]), WORLD_SIZE)
        self.assertEqual(bundle["cases"][0]["source_case"]["path"], "fixed")
        self.assertEqual(len(bundle["cases"][0]["repetitions"]), 2)
        self.assertEqual(bundle["source"]["status"], "pass")
        self.assertEqual(bundle["expected_world_size"], WORLD_SIZE)

    def test_normalizes_bare_rank_record_list(self):
        bundle = bundle_from_rank_report(
            _synthetic_rank_records(),
            ["timing_ms.e2e"],
        )
        self.assertEqual(bundle["cases"][0]["case_id"], "default")
        self.assertEqual(len(bundle["cases"][0]["rank_records"]), WORLD_SIZE)

    def test_rejects_comparison_to_unknown_case(self):
        bundle = {
            "schema_version": ANALYSIS_BUNDLE_SCHEMA_VERSION,
            "source": {"status": "pass"},
            "workload": {"tokens": 1},
            "expected_world_size": WORLD_SIZE,
            "world_size_source": "test",
            "metric_definitions": {
                "timing_ms.e2e": {
                    "unit": "ms",
                    "direction": "lower",
                    "reduction": "rank_max",
                    "semantic": "e2e_latency",
                }
            },
            "cases": [
                {
                    "case_id": "known",
                    "rank_records": _synthetic_rank_records(),
                    "metric_paths": ["timing_ms.e2e"],
                    "workload": {"tokens": 1},
                    "comparison_group": "tokens-1",
                }
            ],
            "route_comparisons": [
                {
                    "baseline_case_id": "known",
                    "candidate_case_id": "missing",
                    "metric_path": "timing_ms.e2e",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "unknown case"):
            validate_analysis_bundle(bundle)

    def test_complete_track_requires_evidence(self):
        bundle = bundle_from_rank_report(
            {"records": _synthetic_rank_records()},
            ["timing_ms.e2e"],
        )
        bundle["measurement_tracks"] = {
            "communication_bytes": {"status": "complete"}
        }
        with self.assertRaisesRegex(ValueError, "requires evidence"):
            validate_analysis_bundle(bundle)

    def test_att_file_must_be_named_in_provenance(self):
        bundle = bundle_from_rank_report(
            {"records": _synthetic_rank_records()},
            ["timing_ms.e2e"],
        )
        bundle["cases"][0]["att_stats_files"] = ["att.csv"]
        bundle["cases"][0]["att_provenance"] = _collection_provenance("other.csv")
        with self.assertRaisesRegex(ValueError, "missing from provenance"):
            validate_analysis_bundle(bundle)


class TestCollectionProvenance(unittest.TestCase):
    def test_valid_provenance(self):
        validated = validate_collection_provenance(_collection_provenance())
        self.assertEqual(validated["collector_id"], "test")

    def test_rejects_missing_raw_artifact(self):
        provenance = _collection_provenance()
        provenance["raw_artifacts"] = []
        with self.assertRaisesRegex(ValueError, "raw_artifacts"):
            validate_collection_provenance(provenance)

    def test_rejects_non_iso_timestamp(self):
        provenance = _collection_provenance()
        provenance["timestamp"] = "yesterday"
        with self.assertRaisesRegex(ValueError, "ISO-8601"):
            validate_collection_provenance(provenance)


class TestEvidenceResolution(unittest.TestCase):
    def test_unresolved_complete_track_becomes_invalid(self):
        tracks = {
            "communication_bytes": {
                "status": "complete",
                "evidence": {
                    "artifact_refs": ["missing"],
                    "metrics": ["missing:bytes"],
                    "provenance_refs": ["missing:provenance"],
                },
            }
        }
        catalog = {
            "schema_version": EVIDENCE_CATALOG_SCHEMA_VERSION,
            "entries": {},
            "provenance": {},
        }
        resolved = resolve_measurement_tracks(tracks, catalog)
        self.assertEqual(resolved["communication_bytes"]["status"], "invalid")
        self.assertTrue(
            resolved["communication_bytes"]["resolution_errors"]
        )

    def test_resolved_complete_track_records_owners(self):
        evidence_id = "case:c:software_counters"
        metric_id = f"{evidence_id}:payload_bytes"
        provenance_ref = "collector:counter:1"
        catalog = {
            "schema_version": EVIDENCE_CATALOG_SCHEMA_VERSION,
            "entries": {
                evidence_id: {
                    "kind": "software_counters",
                    "status": "complete",
                    "metric_ids": [metric_id],
                    "provenance_refs": [provenance_ref],
                }
            },
            "provenance": {
                provenance_ref: {
                    "kind": "collection",
                    "status": "complete",
                }
            },
        }
        tracks = {
            "communication_bytes": {
                "status": "complete",
                "evidence": {
                    "artifact_refs": [evidence_id],
                    "metrics": [metric_id],
                    "provenance_refs": [provenance_ref],
                },
            }
        }
        resolved = resolve_measurement_tracks(tracks, catalog)
        self.assertEqual(resolved["communication_bytes"]["status"], "complete")
        self.assertEqual(
            resolved["communication_bytes"]["resolved_evidence"][
                "metric_owners"
            ][metric_id],
            evidence_id,
        )


class TestCollectorRegistry(unittest.TestCase):
    def test_collector_specs_have_required_contract(self):
        collector_dir = (
            Path(__file__).resolve().parents[1] / "knowledge" / "collectors"
        )
        specs = sorted(collector_dir.glob("*.json"))
        self.assertTrue(specs)
        for path in specs:
            with self.subTest(path=path.name):
                spec = json.loads(path.read_text())
                self.assertEqual(spec["schema_version"], "geak-collector-v1")
                for field in (
                    "id",
                    "maturity",
                    "tool",
                    "purpose",
                    "capability_probe",
                    "preconditions",
                    "outputs",
                    "helps_diagnose",
                    "not_measured",
                    "provenance",
                ):
                    self.assertIn(field, spec)
                self.assertTrue(
                    spec.get("command_template") or spec.get("command_templates")
                )


class TestHardwareContext(unittest.TestCase):
    def _context(self):
        context = {
            "schema_version": HARDWARE_CONTEXT_SCHEMA_VERSION,
            "vendor": "AMD",
            "model": "MI355X",
            "arch": "gfx950",
            "device_count": 8,
            "execution_units_per_device": 256,
            "thread_group_width": 64,
            "local_memory_bytes_per_execution_unit": 163840,
            "device_memory_bytes": 309220868096,
            "interconnect": {"type": "XGMI", "topology": "fully_connected"},
            "runtime": {"rocm": "7.2"},
            "measured": {"all_to_all_interconnect_gbps": None},
        }
        context["provenance"] = {
            field: {
                "collector": "test",
                "timestamp": "2026-01-01T00:00:00Z",
                "confidence": "high",
                "raw_artifact": "synthetic:test",
            }
            for field in (
                "vendor",
                "model",
                "arch",
                "device_count",
                "execution_units_per_device",
                "thread_group_width",
                "local_memory_bytes_per_execution_unit",
                "device_memory_bytes",
                "interconnect",
                "runtime",
            )
        }
        return context

    def test_valid_context(self):
        context = validate_hardware_context(self._context())
        self.assertEqual(context["arch"], "gfx950")
        self.assertEqual(context["device_count"], 8)

    def test_invalid_execution_units_rejected(self):
        context = self._context()
        context["execution_units_per_device"] = 0
        with self.assertRaises(ValueError):
            validate_hardware_context(context)

    def test_missing_field_provenance_rejected(self):
        context = self._context()
        del context["provenance"]["arch"]
        with self.assertRaisesRegex(ValueError, "provenance.arch"):
            validate_hardware_context(context)

    def test_negative_measured_ceiling_rejected(self):
        context = self._context()
        context["measured"]["all_to_all_interconnect_gbps"] = -1.0
        context["provenance"]["measured.all_to_all_interconnect_gbps"] = {
            "collector": "test",
            "timestamp": "2026-01-01T00:00:00Z",
            "confidence": "high",
            "raw_artifact": "synthetic:test",
        }
        with self.assertRaisesRegex(ValueError, "positive finite"):
            validate_hardware_context(context)


class TestBuildReport(unittest.TestCase):
    def test_envelope_shape(self):
        report = build_report(
            primary_metric={
                "description": "candidate E2E rank-max latency",
                "path": "timing_ms.e2e",
                "unit": "ms",
                "direction": "lower",
                "reduction": "rank_max",
                "semantic": "e2e_latency",
            },
            cases=[{"case_id": "t512_uniform"}, {"case_id": "t512_skew"}],
            route_comparisons=[{"tokens_per_rank": 512}],
            secondary_comparator_role="secondary comparison only; never the speedup denominator",
            hardware_context={"schema_version": HARDWARE_CONTEXT_SCHEMA_VERSION},
            measurement_tracks={"communication_bytes": {"status": "complete"}},
            experiment_manifest={"schema_version": EXPERIMENT_SCHEMA_VERSION},
        )
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["cases"]), 2)
        self.assertIn("route_comparisons", report)
        self.assertIn("secondary_comparator_role", report)
        self.assertIn("hardware_context", report)
        self.assertIn("measurement_tracks", report)
        self.assertIn("experiment_manifest", report)

    def test_optional_fields_omitted_when_not_given(self):
        report = build_report(
            primary_metric={
                "description": "x",
                "path": "x",
                "unit": "count",
                "direction": "neutral",
                "reduction": "rank_max",
                "semantic": "opaque",
            },
            cases=[],
        )
        self.assertNotIn("route_comparisons", report)
        self.assertNotIn("secondary_comparator_role", report)


class TestRunnerCLI(unittest.TestCase):
    def test_bundle_builder_rejects_core_case_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            source = os.path.join(directory, "records.json")
            artifacts = os.path.join(directory, "case_artifacts.json")
            output = os.path.join(directory, "bundle.json")
            with open(source, "w") as f:
                json.dump(
                    {
                        "status": "pass",
                        "metadata": {"world_size": WORLD_SIZE},
                        "records": _synthetic_rank_records(),
                    },
                    f,
                )
            with open(artifacts, "w") as f:
                json.dump(
                    {"default": {"rank_records": _synthetic_rank_records()}},
                    f,
                )
            process = subprocess.run(
                [
                    sys.executable,
                    os.path.join(
                        root,
                        "scripts",
                        "multi_rank_analysis",
                        "build_bundle.py",
                    ),
                    "--rank-report",
                    source,
                    "--metric",
                    "timing_ms.e2e",
                    "--case-artifacts",
                    artifacts,
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("immutable/core fields", process.stderr)

    def test_bundle_builder_accepts_aiter_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            source = os.path.join(directory, "aiter.json")
            output = os.path.join(directory, "bundle.json")
            with open(source, "w") as f:
                json.dump(
                    {
                        "metadata": {"network": "v4_pro"},
                        "cases": [
                            {
                                "case_id": "v4_pro_bs128",
                                "ranks": _synthetic_rank_records(),
                            }
                        ],
                    },
                    f,
                )
            process = subprocess.run(
                [
                    sys.executable,
                    os.path.join(
                        root,
                        "scripts",
                        "multi_rank_analysis",
                        "build_bundle.py",
                    ),
                    "--rank-report",
                    source,
                    "--metric",
                    "timing_ms.e2e",
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with open(output) as f:
                bundle = json.load(f)
            self.assertEqual(
                bundle["schema_version"],
                ANALYSIS_BUNDLE_SCHEMA_VERSION,
            )
            self.assertEqual(bundle["cases"][0]["case_id"], "v4_pro_bs128")
            self.assertEqual(
                len(
                    bundle["artifacts"]["assembly_inputs"]["rank_report"][
                        "sha256"
                    ]
                ),
                64,
            )

    def test_rank_record_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            records = os.path.join(directory, "records.json")
            output = os.path.join(directory, "report.json")
            with open(records, "w") as f:
                json.dump({"records": _synthetic_rank_records()}, f)
            process = subprocess.run(
                [
                    sys.executable,
                    os.path.join(root, "scripts", "multi_rank_analysis", "runner.py"),
                    "--rank-records",
                    records,
                    "--metric",
                    "timing_ms.e2e",
                    "--case-id",
                    "case",
                    "--primary-metric",
                    "e2e rank max",
                    "--expected-world-size",
                    str(WORLD_SIZE),
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with open(output) as f:
                report = json.load(f)
            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(
                report["cases"][0]["rank_metrics"]["timing_ms.e2e"]["missing_ranks"],
                [],
            )
            self.assertEqual(
                len(report["input_artifacts"]["rank_records"]["sha256"]),
                64,
            )

    def test_runner_preserves_failed_source_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            source = os.path.join(directory, "failed.json")
            output = os.path.join(directory, "report.json")
            with open(source, "w") as f:
                json.dump(
                    {
                        "status": "fail",
                        "metadata": {"world_size": WORLD_SIZE},
                        "records": _synthetic_rank_records(),
                    },
                    f,
                )
            process = subprocess.run(
                [
                    sys.executable,
                    os.path.join(
                        root,
                        "scripts",
                        "multi_rank_analysis",
                        "runner.py",
                    ),
                    "--rank-records",
                    source,
                    "--metric",
                    "timing_ms.e2e",
                    "--primary-metric",
                    "e2e rank max",
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with open(output) as f:
                report = json.load(f)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["source"]["status"], "fail")

    def test_runner_marks_missing_rank_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            source = os.path.join(directory, "partial.json")
            output = os.path.join(directory, "report.json")
            with open(source, "w") as f:
                json.dump(
                    {
                        "status": "pass",
                        "metadata": {"world_size": WORLD_SIZE},
                        "records": _synthetic_rank_records()[:-1],
                    },
                    f,
                )
            process = subprocess.run(
                [
                    sys.executable,
                    os.path.join(
                        root,
                        "scripts",
                        "multi_rank_analysis",
                        "runner.py",
                    ),
                    "--rank-records",
                    source,
                    "--metric",
                    "timing_ms.e2e",
                    "--primary-metric",
                    "e2e rank max",
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with open(output) as f:
                report = json.load(f)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(
                report["cases"][0]["rank_metrics"]["timing_ms.e2e"][
                    "missing_ranks"
                ],
                [7],
            )

    def test_incomplete_route_comparison_emits_no_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            bundle_path = os.path.join(directory, "bundle.json")
            output = os.path.join(directory, "report.json")
            source = {
                "status": "pass",
                "metadata": {"world_size": WORLD_SIZE},
                "cases": [
                    {
                        "case_id": "uniform",
                        "tokens_per_rank": 512,
                        "world_size": WORLD_SIZE,
                        "ranks": _synthetic_rank_records(base_ms=1.0),
                    },
                    {
                        "case_id": "skew",
                        "tokens_per_rank": 512,
                        "world_size": WORLD_SIZE,
                        "ranks": _synthetic_rank_records(base_ms=1.2)[:-1],
                    },
                ],
            }
            bundle = bundle_from_rank_report(source, ["timing_ms.e2e"])
            bundle["route_comparisons"] = [
                {
                    "baseline_case_id": "uniform",
                    "candidate_case_id": "skew",
                    "metric_path": "timing_ms.e2e",
                    "tokens_per_rank": 512,
                }
            ]
            with open(bundle_path, "w") as f:
                json.dump(validate_analysis_bundle(bundle), f)
            process = subprocess.run(
                [
                    sys.executable,
                    os.path.join(
                        root,
                        "scripts",
                        "multi_rank_analysis",
                        "runner.py",
                    ),
                    "--analysis-bundle",
                    bundle_path,
                    "--primary-metric",
                    "e2e rank max",
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with open(output) as f:
                report = json.load(f)
            comparison = report["route_comparisons"][0]
            self.assertEqual(comparison["status"], "incomplete")
            self.assertNotIn("delta", comparison)
            self.assertEqual(report["status"], "partial")

    def test_runner_accepts_aiter_cases_ranks_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            source = os.path.join(directory, "aiter.json")
            output = os.path.join(directory, "report.json")
            with open(source, "w") as f:
                json.dump(
                    {
                        "metadata": {"network": "v4_pro"},
                        "cases": [
                            {
                                "case_id": "v4_pro_bs128",
                                "ranks": _synthetic_rank_records(),
                            }
                        ],
                    },
                    f,
                )
            process = subprocess.run(
                [
                    sys.executable,
                    os.path.join(root, "scripts", "multi_rank_analysis", "runner.py"),
                    "--rank-records",
                    source,
                    "--metric",
                    "timing_ms.e2e",
                    "--case-id",
                    "v4_pro_bs128",
                    "--primary-metric",
                    "e2e rank max",
                    "--expected-world-size",
                    str(WORLD_SIZE),
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with open(output) as f:
                report = json.load(f)
            self.assertEqual(report["cases"][0]["case_id"], "v4_pro_bs128")

    def test_runner_normalizes_trace_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            trace_path = os.path.join(directory, "trace_rank0.json")
            category_path = os.path.join(directory, "categories.json")
            bundle_path = os.path.join(directory, "bundle.json")
            output = os.path.join(directory, "report.json")
            with open(trace_path, "w") as f:
                json.dump(
                    {
                        "traceEvents": [
                            {
                                "ph": "X",
                                "cat": "kernel",
                                "name": "stage1_kernel",
                                "ts": replay * 3000,
                                "dur": 2000,
                            }
                            for replay in range(3)
                        ]
                    },
                    f,
                )
            with open(category_path, "w") as f:
                json.dump({"stage1": "stage1"}, f)
            bundle = bundle_from_rank_report(
                {
                    "status": "pass",
                    "metadata": {"world_size": 1},
                    "records": [
                        {"rank": 0, "timing_ms": {"e2e": 3.0}}
                    ],
                },
                ["timing_ms.e2e"],
            )
            bundle["cases"][0].update(
                {
                    "trace_files": [trace_path],
                    "trace_replays": 3,
                    "trace_provenance": _collection_provenance(trace_path),
                }
            )
            with open(bundle_path, "w") as f:
                json.dump(validate_analysis_bundle(bundle), f)
            process = subprocess.run(
                [
                    sys.executable,
                    os.path.join(
                        root,
                        "scripts",
                        "multi_rank_analysis",
                        "runner.py",
                    ),
                    "--analysis-bundle",
                    bundle_path,
                    "--primary-metric",
                    "e2e rank max",
                    "--category-map",
                    category_path,
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with open(output) as f:
                report = json.load(f)
            trace = report["cases"][0]["trace"]
            self.assertAlmostEqual(
                trace["categories"]["per_category_ms.stage1"]["rank_max"],
                2.0,
            )
            self.assertEqual(trace["replay_count"], 3)
            self.assertAlmostEqual(
                trace["per_rank_event_counts"]["0"]["per_replay"]["stage1"],
                1.0,
            )

    def test_runner_bundle_comparison_and_att(self):
        with tempfile.TemporaryDirectory() as directory:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            att_path = os.path.join(directory, "uniform_rank0.csv")
            with open(att_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "Instruction",
                        "Hitcount",
                        "Latency",
                        "Stall",
                        "Idle",
                        "Source",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Instruction": "v_mfma_f32_16x16x32_fp8",
                        "Hitcount": 1,
                        "Latency": 10,
                        "Stall": 2,
                        "Idle": 1,
                    }
                )
            baseline = _synthetic_rank_records(base_ms=1.0)
            candidate = _synthetic_rank_records(base_ms=1.2)
            bundle_path = os.path.join(directory, "bundle.json")
            output = os.path.join(directory, "report.json")
            with open(bundle_path, "w") as f:
                json.dump(
                    {
                        "schema_version": ANALYSIS_BUNDLE_SCHEMA_VERSION,
                        "source": {"status": "pass"},
                        "workload": {"tokens": 512},
                        "expected_world_size": WORLD_SIZE,
                        "world_size_source": "test",
                        "metric_definitions": {
                            "timing_ms.e2e": {
                                "unit": "ms",
                                "direction": "lower",
                                "reduction": "rank_max",
                                "semantic": "e2e_latency",
                            }
                        },
                        "cases": [
                            {
                                "case_id": "uniform",
                                "rank_records": baseline,
                                "metric_paths": ["timing_ms.e2e"],
                                "workload": {"tokens": 512},
                                "comparison_group": "tokens-512",
                                "att_stats_files": [att_path],
                                "att_provenance": _collection_provenance(att_path),
                            },
                            {
                                "case_id": "skew",
                                "rank_records": candidate,
                                "metric_paths": ["timing_ms.e2e"],
                                "workload": {"tokens": 512},
                                "comparison_group": "tokens-512",
                            },
                        ],
                        "route_comparisons": [
                            {
                                "baseline_case_id": "uniform",
                                "candidate_case_id": "skew",
                                "metric_path": "timing_ms.e2e",
                                "tokens_per_rank": 512,
                            }
                        ],
                    },
                    f,
                )
            process = subprocess.run(
                [
                    sys.executable,
                    os.path.join(root, "scripts", "multi_rank_analysis", "runner.py"),
                    "--analysis-bundle",
                    bundle_path,
                    "--primary-metric",
                    "e2e rank max",
                    "--expected-world-size",
                    str(WORLD_SIZE),
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with open(output) as f:
                report = json.load(f)
            self.assertGreater(
                report["route_comparisons"][0]["delta_pct"],
                0,
            )
            self.assertEqual(report["route_comparisons"][0]["status"], "complete")
            self.assertEqual(
                report["route_comparisons"][0]["metric"]["unit"],
                "ms",
            )
            self.assertEqual(
                report["cases"][0]["att"]["reports"][0]["categories"]["mfma"][
                    "hitcount"
                ],
                1,
            )
            self.assertEqual(
                report["raw_artifact_identities"]["att_stats_files"][0][
                    "status"
                ],
                "available",
            )

    def test_no_operator_names_in_schema_version(self):
        # Regression guard: the generic schema must never hardcode an operator name.
        for banned in ("mega", "moe", "aiter", "mori", "flydsl"):
            self.assertNotIn(banned, SCHEMA_VERSION.lower())


if __name__ == "__main__":
    unittest.main()
