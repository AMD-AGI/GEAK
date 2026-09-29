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
                "num_hidden_layers_main": 2,
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
                    "cat": "python_function",
                    "name": "nn.Module: GenericDecoderLayer_%d" % layer_id,
                    "ts": cursor - 1, "dur": duration + 1,
                })
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
            self.assertTrue(all(
                table["pattern_layer_ids"] == [0, 1]
                and table["pattern_layer_count"] == 2
                and table["representative_layer_id"]
                in table["pattern_layer_ids"]
                for table in tables))
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

    def test_authoritative_boundary_is_not_rewritten_by_stage_similarity(self):
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
                    "layer_evidence": "python_module_span_external_id",
                    "layer_region": "layer_body",
                    "boundary_role": (
                        "body_start_kernel" if index == 0 else None),
                })
                sequence += 1
        diagnostics, _ = mapping._authoritative_layer_partition(rows, {
            "num_hidden_layers_main": 3,
            "patterns": [{"pattern_id": "P_DENSE", "layer_ids": [0, 1, 2]}],
        })
        self.assertEqual(diagnostics[0]["status"], "mapped")
        self.assertEqual(rows[0]["assignment"], "layer_body")
        self.assertEqual(rows[0]["layer_id"], 0)
        self.assertEqual(rows[0]["boundary_role"], "body_start_kernel")
        self.assertEqual(
            diagnostics[0]["layer_boundaries"][0]["body_start_event"],
            "event-0")

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
            with open(trace) as fh:
                events = json.load(fh)["traceEvents"]
            with open(patterns) as fh:
                pattern_doc = json.load(fh)
            rows, _, _, _, _ = mapping._event_rows(events, pattern_doc)
            self.assertEqual(
                [row["shape"]["source"] for row in rows],
                ["parent_context", "parent_context"])
            self.assertTrue(all(
                row["parent_operator"]["mapping_cardinality"] == "1:N"
                for row in rows))

    def test_missing_annotations_degrades_phase_without_losing_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = mapping.build(
                self._trace(tmp, annotated=False), self._patterns(tmp),
                os.path.join(tmp, "out"))
            self.assertEqual(result["status"], "fail")
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

    def test_cpu_module_spans_use_the_cpu_step_window_not_the_gpu_one(self):
        """Eager decode: host wall-clock >> device time, so the GPU-side
        `step[...]` window covers only a fraction of the CPU-side
        `nn.Module: DecoderLayer_N` spans.  Locating CPU spans against the GPU
        window dropped most layers and silently degraded decode boundaries to
        anchor segmentation; they must be located against the CPU window."""
        events = [
            # CPU-side step annotation spans the whole host-side step.
            {"ph": "X", "cat": "user_annotation", "name": "step[DECODE bs=4]",
             "ts": 1000, "dur": 10000},
            # GPU-side annotation covers only the tail: the device work.
            {"ph": "X", "cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
             "ts": 10500, "dur": 400},
        ]
        # Three DecoderLayer spans, all inside the CPU window, only the last
        # inside the GPU window.
        for layer_id, ts in enumerate((1500, 5000, 10600)):
            events.append({
                "ph": "X", "cat": "python_function", "ts": ts, "dur": 100,
                "name": "nn.Module: DeepseekV2AttentionDecoderLayer_%d" % layer_id,
            })
        spans = mapping._collect_step_spans(events)
        self.assertTrue(spans, "step span not recognised")
        self.assertEqual(len(spans[0]), 9, "CPU window not carried on the span")
        self.assertEqual((spans[0][7], spans[0][8]), (1000, 11000))
        pattern_doc = {
            "num_hidden_layers_main": 3,
            "patterns": [{"pattern_id": "P_FULL", "attention_type": "full",
                          "layer_ids": [0, 1, 2]}],
        }
        scopes, diagnostics = mapping._module_layer_scopes(
            events, spans, pattern_doc)
        self.assertEqual(len(scopes), 3)
        self.assertEqual(diagnostics[0]["full_passes"], 1)
        self.assertEqual([s["layer_id"] for s in scopes], [0, 1, 2])

    def test_moduleless_decode_is_not_partitioned_from_recurring_stages(self):
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
            diag = next(item for item in
                        audit["boundary_partition_diagnostics"]
                        if item["phase"] == "decode")
            self.assertEqual(diag["status"], "boundary_unresolved")
            self.assertEqual(diag["partition_method"], "none")
            self.assertIn("diagnostic_only_recurring_stages", diag)
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
                "authoritative_scope_ownership")
            self.assertTrue(all(
                any(source.startswith("python_module_span")
                    for source in item["boundary_evidence"]["sources"])
                for item in audit["instances"]))

    def test_recurring_fused_kernel_never_creates_layer_boundaries(self):
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
            self.assertEqual(audit["instances"], [])
            self.assertEqual(
                audit["boundary_partition_diagnostics"][0]["status"],
                "boundary_unresolved")
            with open(result["semantic_event_audit_jsonl"]) as fh:
                rows = [json.loads(line) for line in fh]
            fused_rows = [row for row in rows if row["raw_name"] == fusion]
            self.assertEqual(len(fused_rows), 2)
            self.assertTrue(all(
                row["layer_instance_id"] is None for row in fused_rows))
            self.assertTrue(all(
                row["assignment"] == "transition_global"
                for row in fused_rows))


class PhaseCoverageTest(unittest.TestCase):
    """Regression tests for the decode-coverage defects."""

    def test_phase_tag_and_sibling_discovery(self):
        self.assertEqual(
            mapping._phase_tag("x/1787.0-TP-0-EXTEND.trace.json.gz"), "EXTEND")
        self.assertEqual(
            mapping._phase_tag("x/1787.0-TP-3-DECODE.trace.json.gz"), "DECODE")
        self.assertIsNone(mapping._phase_tag("x/plain.trace.json.gz"))

    def test_sibling_discovery_finds_the_other_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = ["1787.0-TP-0-EXTEND.trace.json.gz",
                     "1787.0-TP-0-DECODE.trace.json.gz",
                     "1787.0-TP-1-DECODE.trace.json.gz"]
            for name in names:
                open(os.path.join(tmp, name), "w").close()
            found = mapping._sibling_phase_traces(
                os.path.join(tmp, names[0]))
            self.assertEqual(sorted(found), ["DECODE", "EXTEND"])
            # rank 1 must not be adopted into a rank 0 analysis
            self.assertNotIn("TP-1", found["DECODE"])

    def test_b1_table_phases_never_reports_all(self):
        """`table_phases` must name observed phases, not the word 'all'."""
        coverage = mapping._phase_coverage(
            instances=[{"phase": "extend"}],
            tables=[{"phase": "prefill", "rows": [
                {"shape": {"source": "kernel_exact"}}]}],
            trace_paths=["a-TP-0-EXTEND.trace.json.gz"],
            adopted_siblings=[], table_phases=None, require_phases=None)
        self.assertEqual(coverage["phases_in_tables"], ["prefill"])
        self.assertNotIn("all", coverage["phases_in_tables"])
        self.assertTrue(coverage["single_phase"])
        self.assertFalse(coverage["decode_sequence_covered"])
        self.assertEqual(coverage["decode_evidence"], "no_decode_trace_analysed")

    def test_sequence_and_shape_coverage_fail_independently(self):
        """A replay DECODE trace gives the sequence but no shapes."""
        coverage = mapping._phase_coverage(
            instances=[{"phase": "extend"}, {"phase": "decode"}],
            tables=[
                {"phase": "prefill", "rows": [
                    {"shape": {"source": "kernel_exact"}}]},
                {"phase": "decode", "rows": [
                    {"shape": {"source": "unresolved"}},
                    {"shape": {"source": "unresolved"}}]}],
            trace_paths=["a-TP-0-EXTEND.trace.json.gz",
                         "a-TP-0-DECODE.trace.json.gz"],
            adopted_siblings=[{"phase": "DECODE", "path": "a-TP-0-DECODE.trace.json.gz"}],
            table_phases=None, require_phases=None)
        self.assertTrue(coverage["decode_sequence_covered"])
        self.assertFalse(coverage["decode_shapes_covered"])
        self.assertFalse(coverage["decode_covered"])
        self.assertTrue(coverage["decode_requires_graph_capture"])
        self.assertEqual(coverage["decode_evidence"],
                         "sequence_only_shapes_unresolved")
        self.assertEqual(
            coverage["shape_resolution_by_phase"]["decode"]["resolved_fraction"],
            0.0)

    def test_shape_capture_plan_declares_graph_construction_only(self):
        coverage = {
            "decode_sequence_covered": True,
            "decode_shapes_covered": False,
            "decode_covered": False,
        }
        plan = mapping._shape_capture_plan(
            tables=[], pattern_doc={}, trace_path=__file__, coverage=coverage)
        policy = plan["capture_policy"]
        self.assertEqual(
            policy["decode_capture_windows_implemented"],
            ["graph_construction"])
        self.assertNotIn("eager", json.dumps(policy).lower())

    def test_require_phases_reports_the_missing_one(self):
        coverage = mapping._phase_coverage(
            instances=[], tables=[{"phase": "prefill", "rows": []}],
            trace_paths=["a-TP-0-EXTEND.trace.json.gz"],
            adopted_siblings=[], table_phases=None,
            require_phases=["prefill", "decode"])
        self.assertEqual(coverage["missing_required_phases"], ["decode"])

    def test_multi_trace_load_orders_by_first_timestamp(self):
        """DECODE follows EXTEND in wall clock; concatenation must preserve it."""
        with tempfile.TemporaryDirectory() as tmp:
            early = os.path.join(tmp, "early.trace.json")
            late = os.path.join(tmp, "late.trace.json")
            with open(early, "w") as fh:
                json.dump({"traceEvents": [{"name": "e", "ts": 10.0}]}, fh)
            with open(late, "w") as fh:
                json.dump({"traceEvents": [{"name": "l", "ts": 900.0}]}, fh)
            merged = mapping._load_events_multi([late, early])
            self.assertEqual([e["name"] for e in merged], ["e", "l"])


class DiagnosticStageRecurrenceTest(unittest.TestCase):
    def test_periodic_stage_is_diagnostic_only(self):
        rows = []
        for index in range(60):
            rows.append({
                "row_id": "event-%d" % index,
                "device_seq_index": index,
                "phase": "decode",
                "step_id": "step-1",
                "stage": "opaque_periodic" if index % 4 == 3 else "other",
                "assignment": "concurrent_unresolved",
                "layer_id": None,
                "layer_instance_id": None,
                "pattern_id": None,
                "layer_evidence": "unresolved",
                "layer_region": None,
                "boundary_role": None,
            })
        diagnostics, _ = mapping._authoritative_layer_partition(rows, {
            "num_hidden_layers_main": 60,
            "patterns": [{"pattern_id": "P0", "layer_ids": list(range(60))}],
        })
        self.assertEqual(diagnostics[0]["status"], "boundary_unresolved")
        self.assertTrue(diagnostics[0]["diagnostic_only_recurring_stages"])
        self.assertTrue(all(row["layer_id"] is None for row in rows))

    def test_stage_names_do_not_change_authoritative_boundaries(self):
        def make_rows(prefix):
            rows = []
            for layer_id in range(2):
                for offset in range(2):
                    rows.append({
                        "row_id": "%s-%d-%d" % (prefix, layer_id, offset),
                        "device_seq_index": layer_id * 2 + offset,
                        "phase": "decode", "step_id": "step-1",
                        "stage": "%s-stage-%d" % (prefix, offset),
                        "assignment": "layer_body", "layer_id": layer_id,
                        "layer_instance_id": "instance-%d" % layer_id,
                        "pattern_id": "P0",
                        "layer_evidence": "explicit_layer_marker_test",
                        "layer_region": "layer_body", "boundary_role": None,
                    })
            return rows

        patterns = {"num_hidden_layers_main": 2,
                    "patterns": [{"pattern_id": "P0", "layer_ids": [0, 1]}]}
        first, second = make_rows("a"), make_rows("randomized")
        first_diag, _ = mapping._authoritative_layer_partition(first, patterns)
        second_diag, _ = mapping._authoritative_layer_partition(second, patterns)
        self.assertEqual(first_diag[0]["status"], "mapped")
        self.assertEqual(second_diag[0]["status"], "mapped")
        self.assertEqual(
            [row["layer_id"] for row in first],
            [row["layer_id"] for row in second])


class GatingStepAuditsTest(unittest.TestCase):
    def _audit(self, step, phase, status, boundary="mapped", instances=2):
        return {"step_id": step, "phase": phase, "status": status,
                "boundary_source_status": boundary, "actual_instance_count": instances}

    def test_extra_unresolved_step_of_a_mapped_phase_does_not_gate(self):
        audits = [self._audit("s1", "prefill", "pass"),
                  self._audit("s2", "decode", "pass"),
                  self._audit("s3", "prefill", "fail", "boundary_unresolved", 0)]
        gating = mapping._gating_step_audits(audits)
        self.assertEqual([a["step_id"] for a in gating], ["s1", "s2"])
        self.assertEqual(audits[2]["status"], "not_gating_unresolved_extra_step")

    def test_phase_without_any_mapped_step_still_gates(self):
        audits = [self._audit("s1", "decode", "pass"),
                  self._audit("s2", "prefill", "fail", "boundary_unresolved", 0)]
        gating = mapping._gating_step_audits(audits)
        self.assertEqual([a["status"] for a in gating], ["pass", "fail"])

    def test_contradicting_or_partial_mapping_still_gates(self):
        audits = [self._audit("s1", "decode", "pass"),
                  self._audit("s2", "decode", "fail", "mapped", 2),
                  self._audit("s3", "decode", "fail", "boundary_unresolved", 1)]
        gating = mapping._gating_step_audits(audits)
        self.assertEqual([a["step_id"] for a in gating], ["s1", "s2", "s3"])


if __name__ == "__main__":
    unittest.main()
