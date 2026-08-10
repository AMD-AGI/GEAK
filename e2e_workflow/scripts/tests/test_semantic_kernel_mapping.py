import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_kernel_mapping as mapping


class SemanticKernelMappingTest(unittest.TestCase):
    def test_stage_precedence_keeps_norm_and_gemm_semantics(self):
        self.assertEqual(
            mapping._stage("add_rmsnorm_quant_kernel", "kernel"), "norm")
        self.assertEqual(
            mapping._stage("_gemm_a8w8_blockscale_kernel_cache_hint", "kernel"),
            "gemm")
        self.assertEqual(
            mapping._stage("_causal_conv1d_fwd_kernel", "kernel"), "linear_attn")
        self.assertEqual(
            mapping._stage(
                "fused_recurrent_gated_delta_rule_packed_decode_kernel", "kernel"),
            "linear_attn")
        self.assertEqual(mapping._stage("l2norm_fwd_kernel", "kernel"), "norm")

    def _patterns(self, root):
        path = os.path.join(root, "patterns.json")
        with open(path, "w") as fh:
            json.dump({
                "schema_version": 1,
                "patterns": [{
                    "pattern_id": "P_DENSE",
                    "pattern_display_name": "Dense",
                    "layer_ids": [0, 1],
                }],
                "coverage_check": {
                    "total_main_layers": 2, "covered": 2,
                    "mutually_exclusive": True, "full_coverage": True,
                },
                "quality": {"status": "pass"},
            }, fh)
        return path

    def _trace(self, root, annotated=True):
        events = []
        if annotated:
            events.extend([
                {"cat": "gpu_user_annotation",
                 "name": "execute_context_1(8)_generation_0(0)",
                 "ts": 0, "dur": 100},
                {"cat": "gpu_user_annotation",
                 "name": "execute_context_0(0)_generation_2(16)",
                 "ts": 200, "dur": 100},
            ])
        phases = [("prefill", 10 if annotated else 10, [10, 20]),
                  ("decode", 210 if annotated else 110, [4, 6])]
        ext = 0
        device = []
        for _, base, durations in phases:
            cursor = base
            for layer_id, duration in enumerate(durations):
                ext += 1
                events.append({
                    "cat": "cpu_op", "name": "model.layers.%d.mlp" % layer_id,
                    "ts": cursor - 1, "dur": duration + 2,
                    "args": {"External id": ext, "Input Dims": [[2, 4]],
                             "Input type": ["Half"]},
                })
                device.append({
                    "cat": "kernel", "name": "fused_mlp_kernel",
                    "ts": cursor, "dur": duration,
                    "args": {"External id": ext, "stream": 1},
                })
                cursor += duration + 2
        events.extend(device)
        path = os.path.join(root, "trace.json")
        with open(path, "w") as fh:
            json.dump({"traceEvents": events}, fh)
        return path

    def test_conservation_representative_and_exact_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = mapping.build(
                self._trace(tmp), self._patterns(tmp), os.path.join(tmp, "out"))
            self.assertEqual(result["status"], "pass")
            with open(result["quality_json"]) as fh:
                quality = json.load(fh)
            gate = quality["gates"]["analysis_window_conservation"]
            self.assertEqual(gate["input_event_count"], 4)
            self.assertEqual(gate["assigned_event_count"], 4)
            self.assertEqual(gate["status"], "pass")
            integrity = quality["gates"]["representative_layer_integrity"]
            self.assertEqual(integrity["status"], "pass")
            self.assertEqual(integrity["table_count"], 2)
            self.assertTrue(all(
                item["interval_complete"] and item["duration_matches"]
                for item in integrity["tables"]))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            self.assertIn(audit["representatives"]["P_DENSE"]["layer_id"], [0, 1])
            with open(result["semantic_table_json"]) as fh:
                tables = json.load(fh)["tables"]
            self.assertEqual({table["phase"] for table in tables},
                             {"prefill", "decode"})
            self.assertTrue(all(row["shape"]["source"] == "kernel_exact"
                                for table in tables for row in table["rows"]))
            self.assertTrue(all(
                "batch_size" in table["selected_bucket"]
                for table in tables))

    def test_representative_integrity_rejects_a_truncated_table(self):
        rows = [
            {"row_id": "event-1", "device_seq_index": 1, "duration_us": 1.0},
            {"row_id": "event-2", "device_seq_index": 2, "duration_us": 2.0},
        ]
        representatives = {"P": {"selected_instances": {
            "decode": {
                "first_device_seq_index": 1,
                "last_device_seq_index": 2,
            }}}}
        tables = [{
            "phase": "decode", "pattern_id": "P",
            "representative_layer_id": 0, "event_count": 1,
            "layer_total_us": 1.0,
            "rows": [dict(rows[0], pos=0)],
        }]
        gate = mapping._representative_integrity(
            rows, tables, representatives)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["tables"][0]["dropped_row_ids"], ["event-2"])

    def test_non_dominant_metadata_prefix_is_demoted_losslessly(self):
        rows = []
        sequence = 0
        for layer_id, stages in (
                (0, ["elementwise", "norm", "gemm"]),
                (1, ["norm", "gemm"]),
                (2, ["norm", "gemm"])):
            for index, stage in enumerate(stages):
                rows.append({
                    "row_id": "event-%d" % sequence,
                    "device_seq_index": sequence,
                    "phase": "prefill",
                    "step_id": "step-1",
                    "assignment": "layer_body",
                    "layer_id": layer_id,
                    "layer_instance_id": "instance-%d" % layer_id,
                    "pattern_id": "P_DENSE",
                    "stage": stage,
                    "layer_evidence": "module_span_sequence_medoid",
                    "layer_region": "layer_body",
                    "boundary_role": (
                        "body_start_kernel" if index == 0 else None),
                })
                sequence += 1
        diagnostics = [{
            "step_id": "step-1",
            "mapped_event_count": len(rows),
            "layer_boundaries": [
                {"layer_id": 0, "body_start_event": "event-0"},
                {"layer_id": 1, "body_start_event": "event-3"},
                {"layer_id": 2, "body_start_event": "event-5"},
            ],
        }]
        demotions = mapping._demote_non_dominant_prefixes(
            rows, diagnostics)
        self.assertEqual(len(demotions), 1)
        self.assertEqual(rows[0]["assignment"], "transition_global")
        self.assertEqual(
            rows[0]["layer_evidence"],
            "pattern_variant_prefix_demoted")
        self.assertEqual(rows[1]["boundary_role"], "body_start_kernel")
        self.assertEqual(
            diagnostics[0]["layer_boundaries"][0]["body_start_event"],
            "event-1")
        self.assertEqual(diagnostics[0]["mapped_event_count"], 6)

    def test_shared_external_id_is_parent_context_not_kernel_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = self._patterns(tmp)
            events = [
                {"cat": "gpu_user_annotation",
                 "name": "step[EXTEND bs=1 toks=8]", "ts": 0, "dur": 100},
                {"cat": "cpu_op", "name": "aiter::wrapper",
                 "ts": 5, "dur": 10,
                 "args": {"External id": 7, "Input Dims": [[8, 4]],
                          "Input type": ["BFloat16"],
                          "Module Hierarchy": "model.layers.0.mlp"}},
                {"cat": "kernel", "name": "child_kernel_a", "ts": 20,
                 "dur": 1, "args": {"External id": 7}},
                {"cat": "kernel", "name": "child_kernel_b", "ts": 22,
                 "dur": 1, "args": {"External id": 7}},
            ]
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"))
            with open(result["semantic_table_json"]) as fh:
                rows = json.load(fh)["tables"][0]["rows"]
            self.assertEqual(
                [row["shape"]["source"] for row in rows],
                ["parent_context", "parent_context"])
            self.assertTrue(all(
                row["parent_operator"]["mapping_cardinality"] == "1:N"
                for row in rows))
            with open(result["shape_capture_plan_json"]) as fh:
                plan = json.load(fh)
            self.assertEqual(plan["target_count"], 2)

    def test_missing_annotations_degrades_phase_without_losing_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = mapping.build(
                self._trace(tmp, annotated=False), self._patterns(tmp),
                os.path.join(tmp, "out"))
            self.assertEqual(result["status"], "partial")
            with open(result["quality_json"]) as fh:
                quality = json.load(fh)
            self.assertEqual(quality["gates"]["phase"]["status"], "partial")
            gate = quality["gates"]["analysis_window_conservation"]
            self.assertEqual(gate["input_event_count"], gate["assigned_event_count"])

    def test_sglang_module_spans_are_global_ordered_and_stream_aware(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "schema_version": 1,
                    "num_hidden_layers_main": 2,
                    "patterns": [
                        {"pattern_id": "P_LINEAR_ATTENTION",
                         "pattern_display_name": "Linear",
                         "attention_type": "linear_attention", "layer_ids": [0]},
                        {"pattern_id": "P_FULL_ATTENTION",
                         "pattern_display_name": "Full",
                         "attention_type": "full_attention", "layer_ids": [1]},
                    ],
                    "coverage_check": {"total_main_layers": 2, "covered": 2,
                                       "mutually_exclusive": True, "full_coverage": True},
                    "quality": {"status": "pass"},
                }, fh)
            events = [
                {"cat": "gpu_user_annotation", "name": "step[EXTEND bs=1 toks=8]",
                 "ts": 0, "dur": 100},
                {"cat": "python_function",
                 "name": "nn.Module: Qwen3_5LinearDecoderLayer_0",
                 "ts": 10, "dur": 20},
                {"cat": "python_function",
                 "name": "nn.Module: Qwen3_5AttentionDecoderLayer_0",
                 "ts": 40, "dur": 20},
                {"cat": "cpu_op", "name": "aten::mm", "ts": 12, "dur": 2,
                 "args": {"External id": 1, "Input Dims": [[2, 2]]}},
                {"cat": "cpu_op", "name": "aten::mm", "ts": 42, "dur": 2,
                 "args": {"External id": 2, "Input Dims": [[2, 2]]}},
                {"cat": "kernel", "name": "linear_attention_kernel", "ts": 70, "dur": 2,
                 "args": {"External id": 1, "stream": 7}},
                {"cat": "kernel", "name": "comm_interleaved", "ts": 72, "dur": 1,
                 "args": {"stream": 8}},
                {"cat": "kernel", "name": "paged_attention_kernel", "ts": 74, "dur": 2,
                 "args": {"External id": 2, "stream": 7}},
            ]
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            self.assertEqual(audit["module_scope_count"], 2)
            self.assertEqual(len(audit["instances"]), 2)
            self.assertTrue(all(item["boundary_complete"] for item in audit["instances"]))
            self.assertEqual({item["layer_id"] for item in audit["instances"]}, {0, 1})

    def test_module_medoid_partitions_moduleless_decode_without_named_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "schema_version": 1,
                    "num_hidden_layers_main": 2,
                    "patterns": [
                        {"pattern_id": "P_LINEAR_ATTENTION",
                         "pattern_display_name": "Linear",
                         "attention_type": "linear_attention", "layer_ids": [0]},
                        {"pattern_id": "P_FULL_ATTENTION",
                         "pattern_display_name": "Full",
                         "attention_type": "full_attention", "layer_ids": [1]},
                    ],
                    "coverage_check": {"total_main_layers": 2, "covered": 2,
                                       "mutually_exclusive": True, "full_coverage": True},
                    "quality": {"status": "pass"},
                }, fh)
            events = [
                {"cat": "gpu_user_annotation", "name": "step[EXTEND bs=1 toks=8]",
                 "ts": 0, "dur": 100},
                {"cat": "python_function",
                 "name": "nn.Module: HybridLinearDecoderLayer_0",
                 "ts": 10, "dur": 20},
                {"cat": "python_function",
                 "name": "nn.Module: HybridAttentionDecoderLayer_0",
                 "ts": 40, "dur": 20},
                {"cat": "cpu_op", "name": "linear_layer", "ts": 12, "dur": 2,
                 "args": {"External id": 1}},
                {"cat": "cpu_op", "name": "full_layer", "ts": 42, "dur": 2,
                 "args": {"External id": 2}},
                {"cat": "kernel", "name": "rmsnorm_kernel", "ts": 70, "dur": 1,
                 "args": {"External id": 1}},
                {"cat": "kernel", "name": "gated_delta_kernel", "ts": 72, "dur": 1,
                 "args": {"External id": 1}},
                {"cat": "kernel", "name": "quant_kernel", "ts": 74, "dur": 1,
                 "args": {"External id": 1}},
                {"cat": "kernel", "name": "rmsnorm_kernel", "ts": 76, "dur": 1,
                 "args": {"External id": 2}},
                {"cat": "kernel", "name": "paged_attention_kernel", "ts": 78, "dur": 1,
                 "args": {"External id": 2}},
                {"cat": "kernel", "name": "quant_kernel", "ts": 80, "dur": 1,
                 "args": {"External id": 2}},
                {"cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
                 "ts": 200, "dur": 100},
                {"cat": "kernel", "name": "quant_kernel", "ts": 210, "dur": 1, "args": {}},
                {"cat": "kernel", "name": "gated_delta_kernel", "ts": 212, "dur": 1,
                 "args": {}},
                {"cat": "kernel", "name": "quant_kernel", "ts": 214, "dur": 1, "args": {}},
                {"cat": "kernel", "name": "paged_attention_kernel", "ts": 216, "dur": 1,
                 "args": {}},
                {"cat": "kernel", "name": "quant_kernel", "ts": 218, "dur": 1, "args": {}},
            ]
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            diag = next(
                item for item in audit["boundary_partition_diagnostics"]
                if item["partition_method"] != "module_span_sequence_medoid")
            self.assertEqual(diag["status"], "mapped")
            self.assertIn(
                diag["partition_method"],
                {"repeated_sequence_medoid", "forced_best_alignment"})
            self.assertEqual({item["layer_id"] for item in audit["instances"]}, {0, 1})

    def test_module_span_is_not_overridden_by_sequence_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "schema_version": 1,
                    "num_hidden_layers_main": 2,
                    "patterns": [
                        {"pattern_id": "P_LINEAR_ATTENTION",
                         "pattern_display_name": "Linear",
                         "attention_type": "linear_attention", "ffn_type": "moe",
                         "structural_signature": {"is_moe": True},
                         "layer_ids": [0]},
                        {"pattern_id": "P_FULL_ATTENTION",
                         "pattern_display_name": "Full",
                         "attention_type": "full_attention", "ffn_type": "moe",
                         "structural_signature": {"is_moe": True},
                         "layer_ids": [1]},
                    ],
                    "coverage_check": {"total_main_layers": 2, "covered": 2,
                                       "mutually_exclusive": True, "full_coverage": True},
                    "quality": {"status": "pass"},
                }, fh)
            events = [
                {"cat": "gpu_user_annotation", "name": "step[EXTEND bs=1 toks=8]",
                 "ts": 0, "dur": 100},
                {"cat": "python_function",
                 "name": "nn.Module: HybridLinearDecoderLayer_0",
                 "ts": 10, "dur": 20},
                {"cat": "python_function",
                 "name": "nn.Module: HybridAttentionDecoderLayer_0",
                 "ts": 40, "dur": 20},
                {"cat": "cpu_op", "name": "layer_zero", "ts": 12, "dur": 2,
                 "args": {"External id": 1}},
                {"cat": "cpu_op", "name": "layer_one", "ts": 42, "dur": 2,
                 "args": {"External id": 2}},
                {"cat": "kernel", "name": "topk_kernel", "ts": 70, "dur": 1,
                 "args": {"External id": 1}},
                {"cat": "kernel", "name": "kernel_moe_gemm", "ts": 72, "dur": 1,
                 "args": {"External id": 1}},
                {"cat": "kernel", "name": "topk_kernel", "ts": 74, "dur": 1,
                 "args": {"External id": 2}},
                {"cat": "kernel", "name": "kernel_moe_gemm", "ts": 76, "dur": 1,
                 "args": {"External id": 2}},
            ]
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            self.assertEqual(len(audit["instances"]), 2)
            self.assertTrue(all(item["boundary_complete"]
                                for item in audit["instances"]))
            self.assertTrue(all(
                item["boundary_evidence"]["end_anchor_valid"]
                for item in audit["instances"]))
            self.assertEqual(
                audit["boundary_partition_diagnostics"][0]["partition_method"],
                "module_span_sequence_medoid")
            self.assertTrue(all(
                any(source == "module_span_sequence_medoid"
                    for source in item["boundary_evidence"]["sources"])
                for item in audit["instances"]))

    def test_sequence_partition_counts_fused_boundary_kernel_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "schema_version": 1,
                    "model_type": "unregistered_model",
                    "num_hidden_layers_main": 2,
                    "patterns": [
                        {
                            "pattern_id": "P_ATTN_DENSE",
                            "pattern_display_name": "Dense",
                            "attention_type": "attention",
                            "ffn_type": "dense",
                            "layer_ids": [0],
                        },
                        {
                            "pattern_id": "P_ATTN_MOE",
                            "pattern_display_name": "MoE",
                            "attention_type": "attention",
                            "ffn_type": "moe",
                            "layer_ids": [1],
                        },
                    ],
                    "coverage_check": {"total_main_layers": 2, "covered": 2,
                                       "mutually_exclusive": True, "full_coverage": True},
                    "quality": {"status": "pass"},
                }, fh)
            fusion = "opaque_fused_boundary_kernel"
            names = [
                "quant_kernel", "gemm_kernel", fusion,
                "quant_kernel", "gemm_kernel", fusion,
            ]
            events = [{"cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
                       "ts": 0, "dur": 100}]
            events.extend(
                {"cat": "kernel", "name": name, "ts": 10 + index * 5,
                 "dur": 1, "args": {}}
                for index, name in enumerate(names))
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            self.assertEqual(len(audit["instances"]), 2)
            self.assertTrue(all(
                item["boundary_complete"] for item in audit["instances"]))
            with open(result["semantic_event_audit_jsonl"]) as fh:
                rows = [json.loads(line) for line in fh]
            fused_rows = [row for row in rows if row["raw_name"] == fusion]
            self.assertEqual(len(fused_rows), 2)
            self.assertEqual(
                len({row["layer_instance_id"] for row in fused_rows}), 2)
            self.assertEqual(
                sum(item["event_count"] for item in audit["instances"]), 6)


if __name__ == "__main__":
    unittest.main()
