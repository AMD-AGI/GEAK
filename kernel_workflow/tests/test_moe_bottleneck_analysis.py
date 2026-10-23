"""Tests for the deterministic measurement-first MoE analysis runner."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYZER = os.path.join(
    ROOT,
    "knowledge",
    "analysis_skills",
    "moe_bottleneck",
    "analyze.py",
)
VALIDATOR = os.path.join(
    ROOT,
    "knowledge",
    "analysis_skills",
    "moe_bottleneck",
    "validate_output.py",
)
SPEC = importlib.util.spec_from_file_location("moe_bottleneck_analyze", ANALYZER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _report(with_tracks=False):
    report = {
        "schema_version": "geak-megamoe-analysis-v1",
        "cases": [
            {"case_id": "t8192_uniform"},
            {"case_id": "t8192_rank_mixed_skew"},
        ],
        "route_comparisons": [
            {
                "tokens_per_rank": 8192,
                "e2e_rank_max_delta_ms": 0.8208,
                "e2e_rank_max_delta_pct": 17.30,
                "profile_category_delta": {
                    "combine": {
                        "rank_max_delta_ms": 0.343084,
                        "rank_max_delta_pct": 97.6772,
                    },
                    "stage1": {
                        "rank_max_delta_ms": 0.2785767,
                        "rank_max_delta_pct": 9.6523,
                    },
                    "stage2": {
                        "rank_max_delta_ms": 0.3557637,
                        "rank_max_delta_pct": 17.1293,
                    },
                },
            }
        ],
    }
    if with_tracks:
        report["schema_version"] = "geak-multirank-analysis-v2"
        report["status"] = "pass"
        report["input_artifacts"] = {
            "analysis_bundle": {"sha256": "a" * 64}
        }
        kinds = {
            "rank": ("rank_metric", "rank:max"),
            "trace": ("trace", "trace:stage1"),
            "software": ("software_counters", "counter:payload_bytes"),
            "xgmi": ("xgmi", "xgmi:bytes"),
            "experiment": (
                "controlled_experiment",
                "experiment:publication_latency",
            ),
            "dag": ("dependency_dag", "dag:critical_path_ms"),
            "residency": (
                "resource_residency",
                "residency:resident_ctas",
            ),
        }
        entries = {}
        for evidence_id, (kind, metric_id) in kinds.items():
            data = {}
            if kind == "xgmi":
                data = {
                    "logical_bytes": 100000000,
                    "physical_bytes": 125000000,
                    "duration_ms": 1000.0,
                    "effective_gbps": 1.0,
                    "ceiling_gbps": 2.0,
                    "utilization_pct": 50.0,
                }
            elif kind == "dependency_dag":
                data = {
                    "nodes": {"critical": 1.0},
                    "edges": [],
                    "critical_path": {
                        "critical_path_ms": 1.0,
                        "path": ["critical"],
                    },
                }
            elif kind == "resource_residency":
                data = {
                    "workgroup_size": 256,
                    "lds_bytes_per_workgroup": 32768,
                    "vgpr_per_thread": 64,
                    "resident_workgroups_per_cu": 2,
                    "liveness": {
                        "producer_progress": True,
                        "consumer_progress": True,
                        "termination_proven": True,
                    },
                }
            elif kind == "controlled_experiment":
                data = {
                    "variants": [
                        {"name": "full", "changed_components": []},
                        {
                            "name": "token",
                            "changed_components": [
                                "publication_granularity"
                            ],
                        },
                        {
                            "name": "tile",
                            "changed_components": [
                                "publication_granularity"
                            ],
                        },
                    ],
                    "overlap_pairs": [
                        {
                            "left": "token",
                            "right": "tile",
                            "changed_components": [
                                "publication_granularity"
                            ],
                        }
                    ],
                    "delta_additivity_allowed": False,
                }
            entries[evidence_id] = {
                "kind": kind,
                "status": "complete",
                "metric_ids": [metric_id],
                "provenance_refs": [f"prov:{evidence_id}"],
                "data": data,
            }
        report["evidence_catalog"] = {
            "schema_version": "geak-evidence-catalog-v1",
            "entries": entries,
            "provenance": {
                f"prov:{evidence_id}": {
                    "kind": "synthetic",
                    "status": "complete",
                }
                for evidence_id in kinds
            },
        }

        def track(artifacts, metrics):
            return {
                "status": "complete",
                "evidence": {
                    "artifact_refs": artifacts,
                    "metrics": metrics,
                    "provenance_refs": [
                        f"prov:{artifact}" for artifact in artifacts
                    ],
                },
            }

        report["measurement_tracks"] = {
            "ep_baseline_decomposition": track(
                ["rank", "trace"],
                ["rank:max", "trace:stage1"],
            ),
            "communication_bytes": track(
                ["software", "xgmi"],
                ["counter:payload_bytes", "xgmi:bytes"],
            ),
            "wait_padding": track(
                ["software"],
                ["counter:payload_bytes"],
            ),
            "publication_granularity": track(
                ["experiment"],
                ["experiment:publication_latency"],
            ),
            "fusion_dag": track(["dag"], ["dag:critical_path_ms"]),
            "resource_residency": track(
                ["residency"],
                ["residency:resident_ctas"],
            ),
        }
        report["measurement_tracks"]["fusion_dag"]["evidence"]["bounds"] = [
            {
                "name": "synthetic",
                "baseline_ms": 1.0,
                "lower_bound_ms": 0.5,
                "ceiling_speedup": 2.0,
                "assumptions": ["synthetic"],
                "provenance_refs": ["prov:dag"],
            }
        ]
        hardware_context = {
            "schema_version": "geak-hardware-context-v2",
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
            "measured": {
                "pairwise_interconnect_gbps": 1.0,
                "all_to_all_interconnect_gbps": 1.0,
                "device_memory_gbps": 1.0,
                "launch_overhead_us": 1.0,
            },
        }
        hardware_context["provenance"] = {
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
                "measured.pairwise_interconnect_gbps",
                "measured.all_to_all_interconnect_gbps",
                "measured.device_memory_gbps",
                "measured.launch_overhead_us",
            )
        }
        report["hardware_context"] = hardware_context
    return report


class TestMoEBottleneckAnalysis(unittest.TestCase):
    def test_workflow_uses_separate_validated_analysis_call(self):
        workflow_path = os.path.join(ROOT, "kernel_mega.js")
        with open(workflow_path) as f:
            source = f.read()
        self.assertIn("async function runProfileAnalysis", source)
        self.assertEqual(source.count("await runProfileAnalysis("), 3)
        self.assertIn("'analysis_engineer'", source)
        self.assertIn("ANALYSIS_RESULT_SCHEMA", source)
        self.assertIn("analysis_status=awaiting_measurement", source)
        self.assertIn("status=ready", source)
        self.assertNotIn("moe_analysis_json", source)
        self.assertTrue(
            os.path.exists(os.path.join(ROOT, "roles", "mega", "analysis_engineer.md"))
        )

    def test_full_builder_runner_analyzer_validator_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "rank_report.json")
            bundle_path = os.path.join(directory, "bundle.json")
            report_path = os.path.join(directory, "multi_rank.json")
            analysis_path = os.path.join(directory, "analysis.json")
            with open(source_path, "w") as f:
                json.dump(
                    {
                        "schema_version": "aiter-fixture-v1",
                        "status": "pass",
                        "record_type": "run",
                        "metadata": {"world_size": 2, "network": "test"},
                        "cases": [
                            {
                                "case_id": "case",
                                "network": "test",
                                "tokens_per_rank": 128,
                                "world_size": 2,
                                "ranks": [
                                    {
                                        "rank": 0,
                                        "timing_ms": {"e2e": 1.0},
                                    },
                                    {
                                        "rank": 1,
                                        "timing_ms": {"e2e": 1.1},
                                    },
                                ],
                            }
                        ],
                    },
                    f,
                )
            commands = [
                [
                    sys.executable,
                    os.path.join(
                        ROOT,
                        "scripts",
                        "multi_rank_analysis",
                        "build_bundle.py",
                    ),
                    "--rank-report",
                    source_path,
                    "--metric",
                    "timing_ms.e2e",
                    "--output",
                    bundle_path,
                ],
                [
                    sys.executable,
                    os.path.join(
                        ROOT,
                        "scripts",
                        "multi_rank_analysis",
                        "runner.py",
                    ),
                    "--analysis-bundle",
                    bundle_path,
                    "--primary-metric",
                    "E2E rank-max latency",
                    "--output",
                    report_path,
                ],
                [
                    sys.executable,
                    ANALYZER,
                    "--report",
                    report_path,
                    "--output",
                    analysis_path,
                ],
                [
                    sys.executable,
                    VALIDATOR,
                    "--analysis",
                    analysis_path,
                ],
            ]
            for command in commands:
                process = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
            with open(analysis_path) as f:
                analysis = json.load(f)
            self.assertEqual(
                analysis["schema_version"],
                "geak-moe-bottleneck-analysis-v4",
            )
            self.assertEqual(
                analysis["analysis_status"],
                "awaiting_measurement",
            )

    def test_corrects_relative_growth_vs_absolute_share(self):
        analysis = MODULE.build_analysis(_report())
        comparison = analysis["route_comparisons"][0]
        categories = comparison["categories"]
        self.assertAlmostEqual(
            categories["combine"]["relative_growth_pct"],
            97.6772,
        )
        self.assertAlmostEqual(
            categories["combine"]["positive_absolute_delta_share_pct"],
            35.1008,
            places=3,
        )
        self.assertGreater(
            categories["stage2"]["positive_absolute_delta_share_pct"],
            categories["combine"]["positive_absolute_delta_share_pct"],
        )
        hypotheses = analysis["hypotheses"]
        self.assertIn("stage2", hypotheses[0]["statement"])
        self.assertEqual(
            hypotheses[0]["supports"]["relative_growth_leader"],
            "combine",
        )
        self.assertNotIn("directions", analysis)
        serialized = json.dumps(analysis)
        self.assertNotIn('"directions"', serialized)
        self.assertNotIn("specialty", serialized)
        self.assertFalse(
            analysis["analysis_boundary"]["decision_fields_emitted"]
        )
        self.assertEqual(
            analysis["analysis_boundary"]["decision_owner"],
            "Step-3 TechLead",
        )

    def test_missing_tracks_force_draft_measurement_first(self):
        analysis = MODULE.build_analysis(_report())
        self.assertEqual(analysis["framework_status"], "ready")
        self.assertEqual(analysis["analysis_status"], "awaiting_measurement")
        self.assertFalse(analysis["claims"]["root_cause_proven"])
        self.assertEqual(len(analysis["unknowns"]), len(MODULE.REQUIRED_TRACKS) + 1)
        self.assertEqual(analysis["hardware_guidance"]["status"], "missing")

    def test_complete_tracks_remove_degradation(self):
        analysis = MODULE.build_analysis(_report(with_tracks=True))
        self.assertEqual(analysis["analysis_status"], "evidence_complete")
        self.assertEqual(analysis["degraded"], [])
        self.assertEqual(analysis["hardware_guidance"]["status"], "ready")

    def test_complete_label_without_evidence_is_invalid(self):
        report = _report(with_tracks=True)
        report["measurement_tracks"]["communication_bytes"] = {"status": "complete"}
        analysis = MODULE.build_analysis(report)
        self.assertEqual(analysis["analysis_status"], "awaiting_measurement")
        self.assertEqual(
            analysis["measurement_coverage"]["communication_bytes"]["status"],
            "invalid",
        )

    def test_unresolved_nonempty_evidence_is_invalid(self):
        report = _report(with_tracks=True)
        report["measurement_tracks"]["communication_bytes"]["evidence"] = {
            "artifact_refs": ["missing"],
            "metrics": ["missing:metric"],
            "provenance_refs": ["missing:provenance"],
        }
        analysis = MODULE.build_analysis(report)
        self.assertEqual(analysis["analysis_status"], "awaiting_measurement")
        self.assertEqual(
            analysis["measurement_coverage"]["communication_bytes"]["status"],
            "invalid",
        )

    def test_caller_claim_booleans_do_not_prove_claims(self):
        report = _report(with_tracks=True)
        report["root_cause_proven"] = True
        report["dispatch_independent_measurement"] = True
        analysis = MODULE.build_analysis(report)
        self.assertFalse(analysis["claims"]["root_cause_proven"])
        self.assertFalse(
            analysis["claims"]["dispatch_independently_measured"]
        )

    def test_inconsistent_xgmi_evidence_invalidates_track(self):
        report = _report(with_tracks=True)
        report["evidence_catalog"]["entries"]["xgmi"]["data"][
            "effective_gbps"
        ] = 99.0
        analysis = MODULE.build_analysis(report)
        self.assertEqual(
            analysis["measurement_coverage"]["communication_bytes"][
                "status"
            ],
            "invalid",
        )
        self.assertEqual(analysis["analysis_status"], "awaiting_measurement")

    def test_all_negative_categories_emit_no_positive_growth_hypothesis(self):
        report = _report()
        for values in report["route_comparisons"][0][
            "profile_category_delta"
        ].values():
            values["rank_max_delta_ms"] = -abs(values["rank_max_delta_ms"])
            values["rank_max_delta_pct"] = -abs(values["rank_max_delta_pct"])
        analysis = MODULE.build_analysis(report)
        self.assertEqual(analysis["hypotheses"], [])

    def test_generic_comparison_preserves_case_identity_and_noise(self):
        report = _report(with_tracks=True)
        report["route_comparisons"] = [
            {
                "status": "complete",
                "baseline_case_id": "t8192_uniform",
                "candidate_case_id": "t8192_rank_mixed_skew",
                "comparison_group": "t8192",
                "tokens_per_rank": 8192,
                "metric": {
                    "path": "timing_ms.e2e",
                    "unit": "ms",
                    "direction": "lower",
                    "reduction": "rank_max",
                    "semantic": "e2e_latency",
                },
                "baseline": {
                    "case_id": "t8192_uniform",
                    "rank_max": 4.745,
                    "rank_max_runs": [4.7, 4.8],
                    "rank_max_span_pct": 2.1,
                    "missing_ranks": [],
                },
                "candidate": {
                    "case_id": "t8192_rank_mixed_skew",
                    "rank_max": 5.5658,
                    "rank_max_runs": [5.5, 5.63],
                    "rank_max_span_pct": 2.3,
                    "missing_ranks": [],
                },
                "delta": 0.8208,
                "delta_pct": (5.5658 / 4.745 - 1.0) * 100.0,
            }
        ]
        analysis = MODULE.build_analysis(report)
        evidence = analysis["findings"][-1]["evidence"]
        self.assertEqual(evidence["baseline_case_id"], "t8192_uniform")
        self.assertEqual(
            evidence["candidate"]["rank_max_runs"],
            [5.5, 5.63],
        )

    def test_cli_writes_versioned_json(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = os.path.join(directory, "report.json")
            output_path = os.path.join(directory, "analysis.json")
            with open(report_path, "w") as f:
                json.dump(_report(), f)
            process = subprocess.run(
                [
                    sys.executable,
                    ANALYZER,
                    "--report",
                    report_path,
                    "--output",
                    output_path,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with open(output_path) as f:
                analysis = json.load(f)
            self.assertEqual(analysis["schema_version"], MODULE.SCHEMA_VERSION)
            self.assertEqual(
                len(analysis["analysis_inputs"]["report"]["sha256"]),
                64,
            )
            validated = subprocess.run(
                [
                    sys.executable,
                    VALIDATOR,
                    "--analysis",
                    output_path,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)

    def test_output_validator_rejects_decision_fields(self):
        analysis = MODULE.build_analysis(_report())
        analysis["directions"] = [{"specialty": "algorithm"}]
        with self.assertRaisesRegex(ValueError, "decision field"):
            MODULE.validate_analysis_output(analysis)


if __name__ == "__main__":
    unittest.main()
