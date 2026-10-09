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

    # ------------------------------------------------------------------ #
    # declared dispatch-op layer anchors (the torch.compile / hybrid-model path)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hybrid_pattern_doc(n=8, linear_branch="vllm::qwen_gdn_attention_core",
                            full_branch="vllm::unified_attention_with_output"):
        """n layers, every 4th one full-attention -- the Qwen3.5 hybrid shape."""
        full = [i for i in range(n) if i % 4 == 3]
        lin = [i for i in range(n) if i % 4 != 3]
        def pat(pid, attn, layers, branch):
            sig = {"attention_type": attn, "runtime_dispatch_branch": branch}
            return {"pattern_id": pid, "attention_type": attn,
                    "layer_ids": layers, "structural_signature": sig}
        return {"num_hidden_layers_main": n,
                "patterns": [pat("P_lin", "linear", lin, linear_branch),
                             pat("P_full", "full", full, full_branch)]}

    @staticmethod
    def _step_and_anchors(doc, n=8, base=1000, step=10, swap=None, drop=0):
        """One annotated step plus one cpu_op anchor per layer, in layer order."""
        events = [{"cat": "gpu_user_annotation", "name": "step[EXTEND bs=1 toks=64]",
                   "ts": base, "dur": step * (n + 2)},
                  {"cat": "user_annotation", "name": "step[EXTEND bs=1 toks=64]",
                   "ts": base, "dur": step * (n + 2)}]
        by_layer = {}
        for pattern in doc["patterns"]:
            for layer in pattern["layer_ids"]:
                by_layer[layer] = pattern["structural_signature"][
                    "runtime_dispatch_branch"]
        for layer in range(n - drop):
            name = by_layer[layer]
            if swap and layer in swap:
                name = swap[layer]
            events.append({"cat": "cpu_op", "name": name,
                           "ts": base + step * (layer + 1), "dur": 1})
        return events

    def test_dispatch_anchors_resolve_every_layer_when_modules_are_compiled_away(self):
        """The vllm case: no nn.Module frames, one splitting op per layer."""
        doc = self._hybrid_pattern_doc()
        events = self._step_and_anchors(doc)
        spans = mapping._collect_step_spans(events)
        scopes, diag = mapping._dispatch_anchor_scopes(events, spans, doc)

        self.assertEqual(diag["status"], "mapped")
        self.assertEqual(len(scopes), 8)
        self.assertEqual([s["layer_id"] for s in scopes], list(range(8)))
        # Pattern assignment must follow the declared layout, not the anchor order alone.
        self.assertEqual([s["pattern_id"] for s in scopes],
                         ["P_lin", "P_lin", "P_lin", "P_full"] * 2)
        # Contiguous, non-overlapping partition.
        for earlier, later in zip(scopes, scopes[1:]):
            self.assertEqual(earlier["end"], later["ts"])
        self.assertTrue(all(s["scope_source"] == "declared_dispatch_op" for s in scopes))

    def test_dispatch_cuts_are_keyed_by_successor_pattern(self):
        """A dispatch cut holds the NEXT layer's head, so its table key includes it.

        On the hybrid layout (every 4th layer full attention) the full-attention head
        (qkv/qk-norm/RoPE/KV write) only occurs in segments followed by a full layer.
        Keyed by core Pattern alone, a P_lin representative followed by another P_lin
        never contained it.
        """
        doc = self._hybrid_pattern_doc()
        events = self._step_and_anchors(doc)
        for index, event in enumerate(list(events)):
            if event.get("cat") == "cpu_op":
                event["args"] = {"External id": 700 + index}
                events.append({"cat": "kernel", "name": "k_%d" % index,
                               "ts": event["ts"] + 2, "dur": 1,
                               "args": {"External id": 700 + index}})
        rows, _, _, _, _ = mapping._event_rows(events, doc)
        mapping._authoritative_layer_partition(rows, doc)
        segment_doc = mapping._segment_pattern_doc(doc, rows)
        by_id = {p["pattern_id"]: p for p in segment_doc["patterns"]}
        self.assertEqual(sorted(by_id), ["P_full>END", "P_full>P_lin",
                                         "P_lin>P_full", "P_lin>P_lin"])
        self.assertEqual(by_id["P_lin>P_full"]["layer_ids"], [2, 6])
        self.assertEqual(by_id["P_lin>P_lin"]["layer_ids"], [0, 1, 4, 5])
        self.assertEqual(by_id["P_full>END"]["layer_ids"], [7])
        self.assertEqual(sum(len(p["layer_ids"]) for p in by_id.values()), 8)
        self.assertEqual(by_id["P_lin>P_full"]["core_pattern_id"], "P_lin")
        row = next(r for r in rows if r.get("layer_id") == 2
                   and r.get("assignment") == "layer_body")
        self.assertEqual((row["pattern_id"], row["core_pattern_id"]),
                         ("P_lin>P_full", "P_lin"))

    def test_module_span_cuts_keep_their_pattern(self):
        doc = self._hybrid_pattern_doc()
        rows = [{"assignment": "layer_body", "layer_id": 0, "pattern_id": "P_lin",
                 "layer_evidence": "python_module_span_external_id"}]
        self.assertIs(mapping._segment_pattern_doc(doc, rows), doc)
        self.assertEqual(rows[0]["pattern_id"], "P_lin")

    @staticmethod
    def _dispatch_rows(layers=2, per_layer=3, phase="decode"):
        rows = []
        for layer in range(layers):
            for pos in range(per_layer):
                rows.append({
                    "row_id": "r%d-%d" % (layer, pos),
                    "step_id": "step-0",
                    "device_seq_index": layer * per_layer + pos,
                    "layer_id": layer,
                    "layer_instance_id": "step-0:pass-0:layer-%d" % layer,
                    "layer_evidence": "declared_dispatch_op_span",
                    "stage": ("attn", "gemm", "norm")[pos % 3],
                    "assignment": "layer_body", "phase": phase,
                })
        return rows

    def test_dispatch_scoped_rows_keep_their_ownership_verbatim(self):
        """An exact op boundary must not be re-cut.

        On fix/vllm a medoid refinement meant for module spans applied a PREFILL template
        to a decode step and pushed the attention run into the previous segment -- the
        decode P_full_attn representative came back with no `attn` kernel. Mainline's
        partition only accepts scope ownership, so each layer must still open on its own
        attention kernel.
        """
        rows = self._dispatch_rows()
        doc = {"num_hidden_layers_main": 2, "patterns": [
            {"pattern_id": "P0", "layer_ids": [0]},
            {"pattern_id": "P1", "layer_ids": [1]}]}
        diagnostics, _ = mapping._authoritative_layer_partition(rows, doc)
        self.assertEqual(diagnostics[0]["status"], "mapped")
        self.assertEqual(diagnostics[0]["mapped_event_count"], 6)
        self.assertEqual(rows[0]["stage"], "attn")
        self.assertEqual(rows[0]["boundary_role"], "body_start_kernel")
        self.assertEqual(rows[2]["boundary_role"], "end_kernel")
        self.assertEqual(rows[3]["stage"], "attn")
        self.assertEqual(rows[3]["boundary_role"], "body_start_kernel")

    def test_unclaimed_rows_before_the_first_layer_stay_global(self):
        """An event no scope claimed is not folded into a neighbour.

        Folding it in would inflate that layer's measured cost, which is the number the
        fusion ranking is built on.
        """
        rows = self._dispatch_rows()
        for row in rows:
            row["device_seq_index"] += 1
        rows.insert(0, {"row_id": "pre", "step_id": "step-0", "device_seq_index": 0,
                        "layer_id": None, "layer_instance_id": None,
                        "layer_evidence": "unresolved", "stage": "memory",
                        "assignment": "transition_global", "phase": "decode"})
        doc = {"num_hidden_layers_main": 2, "patterns": []}
        diagnostics, _ = mapping._authoritative_layer_partition(rows, doc)
        self.assertEqual(diagnostics[0]["status"], "mapped")
        self.assertEqual(rows[0]["assignment"], "transition_global")
        self.assertIsNone(rows[0]["layer_id"])

    def test_unclaimed_row_inside_a_dispatch_scope_leaves_the_step_unresolved(self):
        """Fail closed: a hole inside a layer breaks ownership contiguity.

        fix/vllm tolerated this (the hole became transition_global and the step still
        mapped). Mainline requires every authoritative instance to be contiguous, so the
        step is reported boundary_unresolved instead. Pinned so that a real vLLM trace
        which trips it is seen as this case, not as a regression elsewhere.
        """
        rows = self._dispatch_rows()
        rows[1].update(layer_id=None, layer_instance_id=None,
                       layer_evidence="unresolved", assignment="transition_global")
        doc = {"num_hidden_layers_main": 2, "patterns": []}
        diagnostics, _ = mapping._authoritative_layer_partition(rows, doc)
        self.assertEqual(diagnostics[0]["status"], "boundary_unresolved")

    def test_dispatch_anchors_decline_when_a_pattern_declares_no_branch(self):
        """A layer kind with no anchor is the 6-of-24 defect; refuse, do not part-map.

        Half-anchoring is worse than not anchoring: it segments the layers it can see
        and silently swallows the rest into whatever segment happens to be open.
        """
        doc = self._hybrid_pattern_doc()
        doc["patterns"][1]["structural_signature"]["runtime_dispatch_branch"] = ""
        events = self._step_and_anchors(doc)
        spans = mapping._collect_step_spans(events)
        scopes, diag = mapping._dispatch_anchor_scopes(events, spans, doc)
        self.assertEqual(scopes, [])
        self.assertEqual(diag["status"], "patterns_without_dispatch_branch")
        self.assertEqual(diag["patterns_missing_branch"], ["P_full"])

    def test_dispatch_anchors_decline_a_step_that_is_missing_anchors(self):
        """CUDA-graph decode emits no per-layer cpu_op; that step must not be mapped."""
        doc = self._hybrid_pattern_doc()
        events = self._step_and_anchors(doc, drop=3)
        spans = mapping._collect_step_spans(events)
        scopes, diag = mapping._dispatch_anchor_scopes(events, spans, doc)
        self.assertEqual(scopes, [])
        self.assertEqual(diag["steps"][0]["status"], "anchor_count_mismatch")
        self.assertEqual(diag["steps"][0]["anchor_count"], 5)

    def test_dispatch_anchors_decline_when_order_disagrees_with_the_patterns(self):
        """The check that makes this evidence: anchor kind must match the layer's Pattern."""
        doc = self._hybrid_pattern_doc()
        events = self._step_and_anchors(
            doc, swap={0: "vllm::unified_attention_with_output"})
        spans = mapping._collect_step_spans(events)
        scopes, diag = mapping._dispatch_anchor_scopes(events, spans, doc)
        self.assertEqual(scopes, [])
        self.assertEqual(diag["steps"][0]["status"],
                         "anchor_order_disagrees_with_patterns")
        self.assertEqual(diag["steps"][0]["first_mismatch_layer"], 0)

    def test_module_spans_still_win_over_dispatch_anchors(self):
        """The fallback must not displace real module evidence when it exists."""
        doc = self._hybrid_pattern_doc()
        events = self._step_and_anchors(doc)
        for layer in range(8):
            events.append({"cat": "python_function",
                           "name": "nn.Module: SomeDecoderLayer_%d" % layer,
                           "ts": 1000 + 10 * (layer + 1) - 2, "dur": 8})
        spans = mapping._collect_step_spans(events)
        scopes, diag = mapping._module_layer_scopes(events, spans, doc)
        self.assertEqual(len(scopes), 8)
        self.assertNotIn("scope_source", scopes[0])
        self.assertFalse(any(d.get("source", "").startswith("declared_dispatch")
                             for d in diag))

    # ------------------------------------------------------------------ #
    # attention stages resolved from the parent operator
    # ------------------------------------------------------------------ #
    def test_generically_named_attention_kernel_resolves_from_its_parent_op(self):
        """vLLM's TRITON_ATTN launches `_fwd_kernel`, which matches no name rule.

        It landed as `unknown`, so the LARGEST prefill row on Qwen3.5-2B (298.5 us, the
        attention operator itself) was not a donor -- and Phase 2.1's escalation gate then
        demanded a fusion candidate for the model's own attention. The parent op is
        authoritative: it is the registered op the kernel ran under.

        `_fwd_kernel` itself now also matches a name rule (SGLang's triton attention
        launches it too), so the parent-op path is exercised with a name no rule knows.
        """
        stage, _, _ = mapping._stage_detail(
            "_fwd_kernel", "kernel", "vllm::unified_attention_with_output")
        self.assertEqual(stage, "attn")
        stage, rule, source = mapping._stage_detail(
            "_triton_kernel", "kernel", "vllm::unified_attention_with_output")
        self.assertEqual(stage, "attn")
        self.assertEqual(source, "parent_operator")
        self.assertEqual(rule, "attention.full.parent")

    def test_snake_case_moe_and_fp8_gemm_kernels_are_classified(self):
        # `\b` treats `_` as a word char; these were `unknown` on sglang and vllm alike.
        for name, stage in (("fused_moe_kernel", "moe"),
                            ("moe_sum_vec_kernel", "moe"),
                            ("_w8a8_triton_block_scaled_mm", "gemm"),
                            ("triton_mm_0", "gemm")):
            self.assertEqual(mapping._stage_detail(name, "kernel")[0], stage, name)
        for name in ("commit_kernel", "ammo_kernel", "immediate_kernel"):
            self.assertEqual(mapping._stage_detail(name, "kernel")[0], "unknown", name)

    def test_gdn_core_parent_resolves_linear_attention(self):
        stage, rule, source = mapping._stage_detail(
            "_fused_post_conv_kernel", "kernel", "vllm::qwen_gdn_attention_core")
        self.assertEqual((stage, source), ("linear_attn", "parent_operator"))

    def test_parent_op_fallback_still_prefers_linear_attention(self):
        stage, _rule, _src = mapping._stage_detail(
            "chunk_fwd_kernel", "kernel", "ChunkGatedDeltaRuleFunction")
        self.assertEqual(stage, "linear_attn")

    def test_kernel_name_rules_still_win_over_the_parent(self):
        # A kernel whose own name is conclusive must not be re-decided by its parent.
        stage, _rule, source = mapping._stage_detail(
            "kernel_paged_attention_2d", "kernel", "ChunkGatedDeltaRuleFunction")
        self.assertEqual(stage, "attn")
        self.assertEqual(source, "kernel_name")

    def test_dispatch_anchor_rows_are_an_authoritative_partition(self):
        """End to end: the anchor scopes must survive the authoritative-only partition.

        Stage recurrence may never assign layer ids, so without an accepted scope a
        compiled vllm step is boundary_unresolved. Each kernel here is launched by its
        layer's dispatch op (External id), which is what puts it in that layer's scope.
        """
        doc = self._hybrid_pattern_doc()
        events = self._step_and_anchors(doc)
        for index, event in enumerate(list(events)):
            if event.get("cat") != "cpu_op":
                continue
            event["args"] = {"External id": 500 + index}
            events.append({"cat": "kernel", "name": "k_%d" % index,
                           "ts": event["ts"] + 2, "dur": 1,
                           "args": {"External id": 500 + index}})
        rows, _, _, _, _ = mapping._event_rows(events, doc)
        self.assertTrue(rows)
        self.assertTrue(all(row["layer_evidence"] == "declared_dispatch_op_span"
                            for row in rows))
        diagnostics, _ = mapping._authoritative_layer_partition(rows, doc)
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["status"], "mapped")
        self.assertEqual(diagnostics[0]["boundary_evidence"],
                         ["declared_dispatch_op_span"])
        self.assertEqual([item["layer_id"] for item in diagnostics[0]["layer_boundaries"]],
                         list(range(8)))
        self.assertEqual(mapping._boundary_rank(
            {"boundary_evidence": {"sources": ["declared_dispatch_op_span"]}}), 1)

    def test_unannotated_single_file_trace_names_the_missing_annotation(self):
        """vllm without the phase-annotation hook: NOTHING is phase-tagged.

        Pinned because the old wording for this case was `no_decode_trace_analysed`, which
        reads as "we did not look at a decode trace" and sends you to re-capture -- when the
        actual fault is that the trace carries no step spans at all, so prefill is equally
        untagged. The remedy is arming the hook, not another capture.
        """
        coverage = mapping._phase_coverage(
            instances=[{"phase": None}],
            tables=[{"phase": None, "rows": [{"shape": {"source": "unresolved"}}]}],
            trace_paths=["vllm-instance-rank-0.1234.pt.trace.json.gz"],
            adopted_siblings=[], table_phases=None, require_phases=None)
        self.assertEqual(coverage["trace_phase_tags"], [])
        self.assertFalse(coverage["phase_annotation_present"])
        self.assertEqual(coverage["decode_evidence"], "no_phase_annotation_in_trace")

    def test_annotated_single_file_trace_blames_the_window_not_the_capture(self):
        """Annotation worked, but this window held only prefill steps."""
        coverage = mapping._phase_coverage(
            instances=[{"phase": "extend"}],
            tables=[{"phase": "prefill", "rows": [
                {"shape": {"source": "kernel_exact"}}]}],
            trace_paths=["vllm-instance-rank-0.1234.pt.trace.json.gz"],
            adopted_siblings=[], table_phases=None, require_phases=None)
        self.assertTrue(coverage["phase_annotation_present"])
        self.assertEqual(coverage["decode_evidence"],
                         "mixed_trace_no_decode_steps_in_window")

    def test_unresolved_decode_steps_are_not_reported_as_an_empty_window(self):
        """vLLM graph replay: 13 decode steps in the window, no decode boundary."""
        coverage = mapping._phase_coverage(
            instances=[{"phase": "extend"}],
            tables=[{"phase": "prefill", "rows": [
                {"shape": {"source": "kernel_exact"}}]}],
            trace_paths=["dp0_pp0_tp0_dcp0_ep0_rank0.1.pt.trace.json.gz"],
            adopted_siblings=[], table_phases=None, require_phases=None,
            step_phases={"prefill", "decode"})
        self.assertEqual(coverage["decode_evidence"],
                         "decode_steps_present_boundaries_unresolved")
        self.assertTrue(coverage["decode_requires_boundary_donor"])
        self.assertFalse(coverage["decode_requires_graph_capture"])
        plan = mapping._shape_capture_plan(
            [], {"patterns": []}, __file__, coverage=coverage)
        self.assertIn("boundary donor",
                      plan["capture_policy"]["decode_capture_requires"][0])

    def test_split_file_capture_keeps_its_original_verdict(self):
        """sglang's per-phase filenames are conclusive; that arm must not shift."""
        coverage = mapping._phase_coverage(
            instances=[{"phase": "extend"}],
            tables=[{"phase": "prefill", "rows": [
                {"shape": {"source": "kernel_exact"}}]}],
            trace_paths=["a-TP-0-EXTEND.trace.json.gz"],
            adopted_siblings=[], table_phases=None, require_phases=None)
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


class IncompleteEvidenceStepTest(unittest.TestCase):
    """Unresolved steps whose trace cannot support ownership (Kimi-K2.5, vLLM 0.21)."""

    @staticmethod
    def _diag(step_id, status, instances=4, phase="prefill"):
        return {"step_id": step_id, "phase": phase, "status": status,
                "module_instance_count": instances, "configured_layer_count": 4}

    @staticmethod
    def _link(dropped=0, graph_rows=0):
        return {"host_launches": 100, "launches_without_device_record": dropped,
                "graph_replay_rows": graph_rows}

    def test_dropped_device_records_and_graph_replayed_mixed_steps_are_excused(self):
        diagnostics = [self._diag("ok", "mapped"),
                       self._diag("dropped", "boundary_unresolved", instances=3),
                       self._diag("mixed", "boundary_unresolved")]
        links = {"ok": self._link(), "dropped": self._link(dropped=40),
                 "mixed": self._link(graph_rows=70)}
        excused = mapping._excused_incomplete_steps(diagnostics, links)
        self.assertEqual({k: v["reason"] for k, v in excused.items()},
                         {"dropped": "device_records_dropped",
                          "mixed": "graph_replay_rows_unowned"})

    def test_an_unresolved_step_with_intact_evidence_still_gates(self):
        diagnostics = [self._diag("ok", "mapped"),
                       self._diag("bad", "boundary_unresolved", instances=3)]
        links = {"ok": self._link(), "bad": self._link()}
        self.assertEqual(mapping._excused_incomplete_steps(diagnostics, links), {})

    def test_plain_graph_decode_is_left_to_the_boundary_donor(self):
        diagnostics = [self._diag("ok", "mapped", phase="decode"),
                       self._diag("graph", "boundary_unresolved", instances=0,
                                  phase="decode")]
        links = {"ok": self._link(), "graph": self._link(graph_rows=900)}
        self.assertEqual(mapping._excused_incomplete_steps(diagnostics, links), {})

    def test_nothing_is_excused_in_a_phase_with_no_mapped_step(self):
        diagnostics = [self._diag("a", "boundary_unresolved", instances=2),
                       self._diag("b", "boundary_unresolved", instances=2)]
        links = {"a": self._link(dropped=5), "b": self._link(dropped=5)}
        self.assertEqual(mapping._excused_incomplete_steps(diagnostics, links), {})

    def test_link_audit_counts_lost_records_and_graph_replayed_rows(self):
        spans = [(100, 200, "P", 8, 1, "s0", "legacy_execute", 0, 50)]
        events = [
            {"cat": "hip_runtime", "name": "hipLaunchKernel", "ts": 10,
             "args": {"correlation": 1}},
            {"cat": "hip_runtime", "name": "hipLaunchKernel", "ts": 11,
             "args": {"correlation": 2}},
            {"cat": "hip_runtime", "name": "hipGraphLaunch", "ts": 12,
             "args": {"correlation": 3}},
            {"cat": "kernel", "name": "k1", "ts": 110, "args": {"correlation": 1}},
            # correlation 2 was launched but its device record is missing
            {"cat": "kernel", "name": "g1", "ts": 120, "args": {"correlation": 3}},
            {"cat": "kernel", "name": "g2", "ts": 130, "args": {"correlation": 3}},
        ]
        self.assertEqual(mapping._step_link_audit(events, spans)["s0"], {
            "host_launches": 2, "launches_without_device_record": 1,
            "graph_replay_rows": 2})


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


class VllmStepSpanTest(unittest.TestCase):
    """vLLM execute_* step windows, as measured on MiniMax-M3 TP8 (vLLM 0.30 ROCm)."""

    PREFILL = "execute_32_context_1(sq32sk8096sqsq1024sqsk259072)_generation_0(sq0sk0sqsq0sqsk0)"
    DECODE = ("execute_16_context_0(sq0sk0sqsq0sqsk0)"
              "_generation_16(sq16sk129712sqsq16sqsk129712)")

    @staticmethod
    def _ann(cat, name, ts, dur, tid=4):
        return {"cat": cat, "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": tid}

    def test_one_step_annotated_on_two_streams_is_one_span(self):
        """The same step on stream 4 and stream 2 split a prefill into 4 + 56 layers."""
        events = [
            self._ann("gpu_user_annotation", self.PREFILL, 1000, 500, tid=4),
            self._ann("gpu_user_annotation", self.PREFILL, 1020, 450, tid=2),
            self._ann("gpu_user_annotation", self.DECODE, 1600, 20, tid=4),
        ]
        spans = mapping._collect_step_spans(events)
        self.assertEqual([(s[0], s[1], s[2]) for s in spans],
                         [(1000, 1500, "P"), (1600, 1620, "D")])

    def test_back_to_back_steps_of_the_same_shape_stay_separate(self):
        events = [
            self._ann("gpu_user_annotation", self.DECODE, 1000, 20),
            self._ann("gpu_user_annotation", self.DECODE, 1020, 20),
        ]
        self.assertEqual(len(mapping._collect_step_spans(events)), 2)

    def test_host_window_is_paired_by_name_without_overlap(self):
        """--async-scheduling: the host runs a whole step ahead of the device."""
        events = [
            self._ann("user_annotation", self.PREFILL, 100, 200),
            self._ann("gpu_user_annotation", self.PREFILL, 900, 500),
        ]
        span = mapping._collect_step_spans(events)[0]
        self.assertEqual((span[7], span[8]), (100, 300))

    def test_host_window_is_not_widened_to_the_device_end(self):
        """Eager decode: the device trails the host, and the next step's host work
        starts before this step's device work ends. The host window must stop at its
        own end, or the last layer swallows the next step's input prep."""
        events = [
            self._ann("user_annotation", self.DECODE, 1000, 200),
            self._ann("gpu_user_annotation", self.DECODE, 1010, 240),
        ]
        span = mapping._collect_step_spans(events)[0]
        self.assertEqual((span[7], span[8]), (1000, 1200))

    def test_same_name_steps_pair_with_the_host_window_that_launched_them(self):
        """vLLM 0.21 repeats a chunked-prefill name step after step; with the host more
        than a step ahead, 'latest same-name window before the device start' is the
        next step's. The launches decide instead."""
        name = "execute_context_2(8181)_generation_11(11)"
        events = []
        for step, (host_lo, device_lo) in enumerate(((0, 500), (200, 800), (400, 1100))):
            events.append(self._ann("user_annotation", name, host_lo, 150))
            events.append(self._ann("gpu_user_annotation", name, device_lo, 250))
            for index in range(3):
                correlation = 10 * step + index
                events.append({"cat": "hip_runtime", "name": "hipLaunchKernel",
                               "ts": host_lo + 10 + index, "dur": 1,
                               "args": {"correlation": correlation}})
                events.append({"cat": "kernel", "name": "k", "ts": device_lo + 10 + index,
                               "dur": 1, "args": {"correlation": correlation}})
        spans = mapping._collect_step_spans(events)
        self.assertEqual([(s[7], s[8]) for s in spans],
                         [(0, 150), (200, 350), (400, 550)])

    def test_flow_owner_comes_from_the_host_launch_not_the_device_endpoint(self):
        """The device endpoint shares the clock with host spans it is unrelated to."""
        scopes = [{"ts": 100, "end": 200, "layer_id": 3,
                   "layer_instance_id": "s:pass-0:layer-3"}]
        starts = [100]
        events = [
            # launched before any layer scope; its device endpoint lands at 150,
            # inside layer 3's host span
            {"ph": "s", "id": 1, "ts": 50, "cat": "ac2g"},
            {"ph": "f", "id": 1, "ts": 150, "cat": "ac2g"},
            # launched inside layer 3
            {"ph": "s", "id": 2, "ts": 120, "cat": "ac2g"},
            {"ph": "f", "id": 2, "ts": 400, "cat": "ac2g"},
        ]
        index = mapping._flow_layer_index(events, scopes, starts)
        self.assertNotIn(150, index)
        self.assertEqual(index[400]["layer_id"], 3)


if __name__ == "__main__":
    unittest.main()
