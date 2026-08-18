#!/usr/bin/env python3
"""Unit tests for profiling-entity and executable-task identity separation."""

import importlib.util
import json
import os
import tempfile
import unittest


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(SCRIPTS_DIR, "op_identity.py")
SPEC = importlib.util.spec_from_file_location("op_identity", MODULE_PATH)
oi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oi)


class TestIdentityMaterialization(unittest.TestCase):
    def test_aggregate_entity_materializes_all_device_leaf_tasks(self):
        result = oi.normalize_candidates(
            {
                "framework": "serving_framework",
                "hot_kernels": [
                    {
                        "name": "framework::attention",
                        "kernel_category": "attention",
                        "device_kernel_names": [
                            "decode_attention_leaf",
                            "prefill_attention_leaf",
                        ],
                        "gpu_pct": 12.5,
                        "op_to_source_kind": "dispatch",
                    }
                ],
            }
        )

        self.assertEqual(len(result["profiling_entities"]), 1)
        entity = result["profiling_entities"][0]
        self.assertEqual(entity["execution_scope"], "expand_leaves")
        self.assertEqual(
            {task["short_name"] for task in result["executable_task_candidates"]},
            {"decode_attention_leaf", "prefill_attention_leaf"},
        )
        by_name = {
            task["short_name"]: task for task in result["executable_task_candidates"]
        }
        self.assertEqual(by_name["decode_attention_leaf"]["served_regimes"], ["decode"])
        self.assertEqual(by_name["prefill_attention_leaf"]["served_regimes"], ["prefill"])
        self.assertTrue(
            by_name["decode_attention_leaf"]["stable_task_key"].startswith(
                "task_decode_attention_leaf_"
            )
        )
        self.assertAlmostEqual(
            sum(task["pct_gpu_time"] for task in result["executable_task_candidates"]),
            entity["pct_gpu_time"],
        )

    def test_single_device_leaf_materializes_one_task(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::attention",
                        "classification": "library_attn",
                        "device_kernel_names": ["attention_leaf"],
                    }
                ]
            }
        )

        tasks = result["executable_task_candidates"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["short_name"], "attention_leaf")

    def test_aggregate_display_name_is_never_used_as_leaf_task_identity(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::attention",
                        "kernel_category": "attention",
                        "device_kernel_names": ["attention_leaf"],
                    }
                ]
            }
        )

        entity = result["profiling_entities"][0]
        task = result["executable_task_candidates"][0]
        self.assertNotEqual(task["stable_task_key"], entity["display_name"])
        self.assertEqual(task["short_name"], "attention_leaf")

    def test_fused_operation_remains_the_executable_unit(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::fused_experts",
                        "kernel_category": "fused_moe",
                        "device_kernel_names": ["internal_stage_one", "internal_stage_two"],
                    }
                ]
            }
        )

        entity = result["profiling_entities"][0]
        tasks = result["executable_task_candidates"]
        self.assertEqual(entity["execution_scope"], "executable_op")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["short_name"], "framework::fused_experts")
        self.assertEqual(tasks[0]["op_kind"], "moe")

    def test_generic_fusion_and_unfused_expert_matmul_are_not_fused_moe(self):
        pointwise = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "triton_poi_fused_0",
                        "device_kernel_names": ["triton_poi_fused_0"],
                    }
                ]
            }
        )["profiling_entities"][0]
        expert_matmul = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "expert_matmul_unfused",
                        "device_kernel_names": ["expert_matmul_unfused"],
                    }
                ]
            }
        )["profiling_entities"][0]
        fused_expert_matmul = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "fused_expert_matmul",
                        "is_fused_kernel": True,
                        "device_kernel_names": ["fused_expert_matmul"],
                    }
                ]
            }
        )["profiling_entities"][0]
        grouped_gemm = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "grouped_gemm",
                        "device_kernel_names": ["grouped_gemm"],
                    }
                ]
            }
        )["profiling_entities"][0]

        self.assertEqual(pointwise["op_kind"], "kernel")
        self.assertEqual(expert_matmul["op_kind"], "gemm")
        self.assertEqual(fused_expert_matmul["op_kind"], "gemm")
        self.assertEqual(grouped_gemm["op_kind"], "gemm")

    def test_attention_without_device_leaf_is_configuration_only(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::attention",
                        "classification": "library_attn",
                        "device_kernel_names": [],
                    }
                ]
            }
        )

        self.assertEqual(
            result["profiling_entities"][0]["execution_scope"], "config_only"
        )
        self.assertEqual(result["executable_task_candidates"], [])
        self.assertEqual(len(result["blocked_entities"]), 1)

    def test_nontransferable_collective_is_configuration_only(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::collective_all_reduce",
                        "device_kernel_names": ["stage_one", "stage_two"],
                        "e2e_transferable": False,
                    }
                ]
            }
        )

        self.assertEqual(
            result["profiling_entities"][0]["execution_scope"], "config_only"
        )
        self.assertEqual(result["executable_task_candidates"], [])

    def test_unresolved_synthetic_profile_entity_is_blocked(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::composite_kernel (Synthetic Op)",
                        "device_kernel_names": [
                            "framework::composite_kernel (Synthetic Op)"
                        ],
                        "op_to_source_patchable": None,
                    }
                ]
            }
        )

        self.assertEqual(result["profiling_entities"][0]["execution_scope"], "blocked")
        self.assertEqual(result["executable_task_candidates"], [])

    def test_repeated_signatures_merge_for_amdahl_but_deduplicate_leaves(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::attention",
                        "kernel_category": "attention",
                        "device_kernel_names": ["decode_attention_leaf"],
                        "gpu_pct": 3.0,
                        "duration_us": 1000,
                        "call_count": 4,
                    },
                    {
                        "name": "framework::attention",
                        "kernel_category": "attention",
                        "device_kernel_names": [
                            "decode_attention_leaf",
                            "prefill_attention_leaf",
                        ],
                        "gpu_pct": 7.0,
                        "duration_us": 2500,
                        "call_count": 6,
                        "op_to_source_kind": "dispatch",
                        "op_to_source_patchable": True,
                    },
                ]
            }
        )

        entity = result["profiling_entities"][0]
        self.assertEqual(entity["pct_gpu_time"], 10.0)
        self.assertEqual(entity["total_ms"], 3.5)
        self.assertEqual(entity["calls"], 10)
        self.assertEqual(
            entity["device_kernel_names"],
            ["decode_attention_leaf", "prefill_attention_leaf"],
        )
        self.assertEqual(entity["op_to_source_kind"], "dispatch")
        self.assertTrue(entity["op_to_source_patchable"])
        self.assertEqual(len(result["executable_task_candidates"]), 2)

    def test_unattributed_leaf_does_not_duplicate_parent_amdahl_weight(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::dispatch",
                        "device_kernel_name": "measured_leaf",
                        "device_kernel_names": ["measured_leaf", "unmeasured_leaf"],
                        "op_to_source_kind": "dispatch",
                        "gpu_pct": 10.0,
                    }
                ]
            }
        )

        by_name = {
            task["short_name"]: task for task in result["executable_task_candidates"]
        }
        self.assertEqual(by_name["measured_leaf"]["pct_gpu_time"], 10.0)
        self.assertEqual(by_name["unmeasured_leaf"]["pct_gpu_time"], 0.0)
        self.assertEqual(
            sum(task["pct_gpu_time"] for task in by_name.values()), 10.0
        )

    def test_equal_split_assigns_residual_and_strictly_conserves_parent(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::dispatch",
                        "device_kernel_names": ["leaf_a", "leaf_b", "leaf_c"],
                        "op_to_source_kind": "dispatch",
                        "gpu_pct": 1.0,
                    }
                ]
            }
        )

        self.assertEqual(
            sum(
                task["pct_gpu_time"]
                for task in result["executable_task_candidates"]
            ),
            1.0,
        )
        tiny = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::tiny_dispatch",
                        "device_kernel_names": [f"leaf_{i}" for i in range(6)],
                        "op_to_source_kind": "dispatch",
                        "gpu_pct": 0.000004,
                    }
                ]
            }
        )
        tiny_values = [
            task["pct_gpu_time"]
            for task in tiny["executable_task_candidates"]
        ]
        self.assertTrue(all(value >= 0 for value in tiny_values))
        self.assertEqual(sum(tiny_values), 0.000004)

    def test_parent_summary_owns_amdahl_weight_without_double_counting_regime_rows(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::attention (prefill)",
                        "kernel_category": "attention",
                        "device_kernel_names": ["prefill_attention_leaf"],
                        "gpu_pct": 2.0,
                        "duration_us": 2000,
                        "call_count": 2,
                    },
                    {
                        "name": "framework::attention (decode)",
                        "kernel_category": "attention",
                        "device_kernel_names": ["decode_attention_leaf"],
                        "gpu_pct": 8.0,
                        "duration_us": 8000,
                        "call_count": 20,
                    },
                    {
                        "name": "framework::attention",
                        "kernel_category": "attention",
                        "device_kernel_names": [
                            "prefill_attention_leaf",
                            "decode_attention_leaf",
                        ],
                        "gpu_pct": 12.0,
                        "duration_us": 12000,
                        "call_count": 22,
                    },
                ]
            }
        )

        self.assertEqual(len(result["profiling_entities"]), 1)
        entity = result["profiling_entities"][0]
        self.assertEqual(entity["display_name"], "framework::attention")
        self.assertEqual(entity["pct_gpu_time"], 12.0)
        self.assertEqual(entity["total_ms"], 12.0)
        self.assertEqual(entity["calls"], 22)
        self.assertEqual(len(result["executable_task_candidates"]), 2)
        self.assertAlmostEqual(
            sum(task["pct_gpu_time"] for task in result["executable_task_candidates"]),
            entity["pct_gpu_time"],
        )

    def test_explicit_aggregate_without_regime_suffix_is_not_double_counted(self):
        result = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        "name": "framework::dispatch",
                        "device_kernel_name": "leaf_one",
                        "device_kernel_names": ["leaf_one"],
                        "gpu_pct": 2.0,
                        "duration_us": 2000,
                        "calls": 2,
                    },
                    {
                        "name": "framework::dispatch",
                        "device_kernel_name": "leaf_two",
                        "device_kernel_names": ["leaf_two"],
                        "gpu_pct": 8.0,
                        "duration_us": 8000,
                        "calls": 8,
                    },
                    {
                        "name": "framework::dispatch",
                        "profiling_kind": "aggregate",
                        "device_kernel_names": ["leaf_one", "leaf_two"],
                        "gpu_pct": 10.0,
                        "duration_us": 10000,
                        "calls": 10,
                    },
                ]
            }
        )

        entity = result["profiling_entities"][0]
        self.assertEqual(entity["pct_gpu_time"], 10.0)
        self.assertEqual(entity["total_ms"], 10.0)
        self.assertEqual(entity["calls"], 10)

    def test_merge_classification_is_input_order_independent(self):
        rows = [
            {
                "name": "framework::dispatch",
                "device_kernel_names": ["leaf_one", "leaf_two"],
                "op_to_source_kind": "dispatch",
                "e2e_transferable": True,
            },
            {
                "name": "framework::dispatch",
                "device_kernel_names": ["nccl_collective_leaf"],
                "kernel_contract": "communication collective",
                "e2e_transferable": False,
            },
        ]

        forward = oi.normalize_candidates({"hot_kernels": rows})
        reverse = oi.normalize_candidates({"hot_kernels": list(reversed(rows))})

        self.assertEqual(
            forward["profiling_entities"][0]["execution_scope"], "config_only"
        )
        self.assertEqual(
            reverse["profiling_entities"][0]["execution_scope"], "config_only"
        )
        self.assertEqual(forward["executable_task_candidates"], [])
        self.assertEqual(reverse["executable_task_candidates"], [])
        self.assertEqual(forward, reverse)

    def test_case_only_classification_variants_are_order_independent(self):
        rows = [
            {
                "name": "framework::attention",
                "classification": "inferenceattention",
                "device_kernel_names": ["attention_leaf"],
            },
            {
                "name": "framework::attention",
                "classification": "InferenceAttention",
                "device_kernel_names": ["attention_leaf"],
            },
        ]

        forward = oi.normalize_candidates({"hot_kernels": rows})
        reverse = oi.normalize_candidates({"hot_kernels": list(reversed(rows))})

        self.assertEqual(forward, reverse)

    def test_conflicting_explicit_op_kinds_merge_deterministically(self):
        rows = [
            {
                "name": "framework::fused_moe_dispatch",
                "op_kind": "moe",
                "is_fused_kernel": True,
                "device_kernel_names": ["stage_one"],
            },
            {
                "name": "framework::fused_moe_dispatch",
                "op_kind": "kernel",
                "is_fused_kernel": True,
                "device_kernel_names": ["stage_two"],
            },
        ]

        forward = oi.normalize_candidates({"hot_kernels": rows})
        reverse = oi.normalize_candidates({"hot_kernels": list(reversed(rows))})

        self.assertEqual(forward, reverse)
        self.assertEqual(forward["profiling_entities"][0]["op_kind"], "moe")
        self.assertEqual(
            forward["profiling_entities"][0]["execution_scope"], "executable_op"
        )

    def test_regime_evidence_does_not_change_stable_leaf_identity(self):
        base = {
            "name": "framework::attention",
            "kernel_category": "attention",
            "device_kernel_name": "attention_leaf",
            "device_kernel_names": ["attention_leaf"],
        }
        first = oi.normalize_candidates({"hot_kernels": [base]})
        second = oi.normalize_candidates(
            {
                "hot_kernels": [
                    {
                        **base,
                        "phase": "both",
                        "source_file": "/package/prefill/launcher.py",
                    }
                ]
            }
        )

        first_task = first["executable_task_candidates"][0]
        second_task = second["executable_task_candidates"][0]
        self.assertEqual(first_task["stable_task_key"], second_task["stable_task_key"])
        self.assertEqual(second_task["served_regimes"], ["prefill", "decode"])

    def test_legacy_parser_topn_preserves_time_and_explicit_regime(self):
        result = oi.normalize_candidates(
            {
                "source": "torch-trace",
                "top_kernels": [
                    {
                        "name": "decode_attention_leaf",
                        "short_name": "decode_attention_leaf",
                        "classification": "library_attn",
                        "pct_gpu_time": 9.0,
                        "total_ms": 4.5,
                        "calls": 30,
                        "phase": "decode",
                    }
                ],
            }
        )

        entity = result["profiling_entities"][0]
        task = result["executable_task_candidates"][0]
        self.assertEqual(entity["total_ms"], 4.5)
        self.assertEqual(entity["calls"], 30)
        self.assertEqual(task["served_regimes"], ["decode"])


class TestCli(unittest.TestCase):
    def test_cli_writes_identity_document(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "candidates.json")
            dst = os.path.join(td, "identity.json")
            with open(src, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "hot_kernels": [
                            {
                                "name": "standalone_kernel",
                                "device_kernel_names": ["standalone_kernel"],
                                "op_to_source_patchable": True,
                            }
                        ]
                    },
                    fh,
                )

            self.assertEqual(oi.main(["--input", src, "--output", dst]), 0)
            with open(dst, encoding="utf-8") as fh:
                result = json.load(fh)

        self.assertEqual(result["schema"], oi.SCHEMA)
        self.assertEqual(len(result["profiling_entities"]), 1)
        self.assertEqual(len(result["executable_task_candidates"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
