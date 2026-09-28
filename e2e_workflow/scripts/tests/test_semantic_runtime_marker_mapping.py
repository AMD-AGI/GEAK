import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_runtime_marker_mapping as mapping


class RuntimeMarkerMappingTest(unittest.TestCase):
    def test_operator_schema_is_resolved_from_observed_argument_structure(self):
        manifest = {
            "schemas": [
                {
                    "name": "aten::mul", "overload_name": "Tensor",
                    "qualified_name": "aten::mul.Tensor",
                    "schema": "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
                    "arguments": [
                        {"index": 0, "name": "self", "type": "Tensor",
                         "has_default": False, "alias_info": None},
                        {"index": 1, "name": "other", "type": "Tensor",
                         "has_default": False, "alias_info": None},
                    ],
                },
                {
                    "name": "aten::mul", "overload_name": "Scalar",
                    "qualified_name": "aten::mul.Scalar",
                    "schema": "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor",
                    "arguments": [
                        {"index": 0, "name": "self", "type": "Tensor",
                         "has_default": False, "alias_info": None},
                        {"index": 1, "name": "other", "type": "Scalar",
                         "has_default": False, "alias_info": None},
                    ],
                },
            ]}
        resolution = mapping._resolve_operator_schema(
            "aten::mul", [[4, 8], [4, 8]],
            ["bfloat16", "bfloat16"], mapping._schema_index(manifest))
        self.assertEqual(resolution["status"], "matched_unique")
        self.assertEqual(resolution["schema"]["overload_name"], "Tensor")

    def test_ambiguous_operator_schema_keeps_argument_roles_unresolved(self):
        schema = {
            "name": "custom::opaque", "overload_name": "",
            "qualified_name": "custom::opaque",
            "schema": "custom::opaque(Tensor x) -> Tensor",
            "arguments": [{
                "index": 0, "name": "x", "type": "Tensor",
                "has_default": False, "alias_info": None}],
        }
        manifest = {"schemas": [schema, dict(schema)]}
        resolution = mapping._resolve_operator_schema(
            "custom::opaque", [[4, 8]], ["bfloat16"],
            mapping._schema_index(manifest))
        self.assertEqual(resolution["status"], "ambiguous")
        self.assertIsNone(resolution["schema"])

    def test_graph_trace_external_id_shape_beats_broad_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            out_path = os.path.join(tmp, "mapped.json")
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [{
                    "row_id": "event-1",
                    "phase": "decode",
                    "pattern_id": "P0",
                    "representative_layer_id": 26,
                    "pos": 0,
                    "raw_name": "quant_kernel",
                    "selected_bucket": {
                        "phase": "decode",
                        "batch_size": 4,
                        "input_tokens": 0,
                    },
                }]}, fh)
            marker = (
                "GEAK_SEMANTICS|op=op-1|phase=DECODE|bs=4|toks=4|"
                "layer=26|path=model.layers.26.proj")
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": [
                    {"cat": "user_annotation", "name": marker,
                     "pid": 1, "tid": 2, "ts": 10, "dur": 50,
                     "args": {"External id": 90}},
                    {"cat": "cpu_op",
                     "name": "aiter::dynamic_per_token_scaled_quant",
                     "pid": 1, "tid": 2, "ts": 20, "dur": 5,
                     "args": {
                         "External id": 7,
                         "Input Dims": [
                             [4, 4096], [128, 128], [4, 32], []],
                         "Input type": [
                             "float8_e4m3fnuz", "bfloat16", "float32", ""],
                     }},
                    {"cat": "hip_runtime", "name": "hipLaunchKernel",
                     "pid": 1, "tid": 2, "ts": 22, "dur": 1,
                     "args": {"kernel": "quant_kernel",
                              "correlation": 1, "External id": 7}},
                    # A second launch in the same broad module marker must not
                    # prevent the External-ID-exact quant shape from winning.
                    {"cat": "hip_runtime", "name": "hipModuleLaunchKernel",
                     "pid": 1, "tid": 2, "ts": 30, "dur": 1,
                     "args": {"kernel": "gemm_kernel",
                              "correlation": 2}},
                    {"cat": "kernel", "name": "quant_kernel", "ts": 100,
                     "dur": 2, "args": {"correlation": 1}},
                    {"cat": "kernel", "name": "gemm_kernel", "ts": 103,
                     "dur": 8, "args": {"correlation": 2}},
                ]}, fh)
            result = mapping.map_plan(plan_path, trace_path, out_path)
            self.assertEqual(
                result["kernel_trace_shape_matched_target_count"], 1)
            with open(out_path) as fh:
                target = json.load(fh)["capture_targets"][0]
            self.assertEqual(target["mapping_cardinality"], "1:1")
            self.assertEqual(
                target["source_mapping_status"],
                "runtime_kernel_trace_shape")
            self.assertEqual(
                target["candidate_terminal_launcher"],
                "aiter::dynamic_per_token_scaled_quant")
            self.assertEqual(
                target["kernel_trace_shape"]["input_dims"],
                [[4, 4096], [128, 128], [4, 32], []])
            self.assertEqual(
                target["kernel_trace_shape"]["bucket_match"], "exact")

    def test_graph_trace_shape_requires_one_launch_per_external_id(self):
        marker = {
            "index": 0,
            "name": "GEAK_SEMANTICS|op=x|phase=DECODE|bs=4|toks=4|"
                    "layer=1|path=model.layers.1.proj",
            "cat": "user_annotation", "pid": 1, "tid": 2,
            "ts": 0, "dur": 100,
        }
        events = [
            marker,
            {"cat": "cpu_op", "name": "custom::compound",
             "args": {"External id": 7, "Input Dims": [[4, 8]],
                      "Input type": ["bfloat16"]}},
            {"cat": "hip_runtime", "name": "hipLaunchKernel",
             "pid": 1, "tid": 2, "ts": 10,
             "args": {"External id": 7, "correlation": 1,
                      "kernel": "kernel_a"}},
            {"cat": "hip_runtime", "name": "hipLaunchKernel",
             "pid": 1, "tid": 2, "ts": 20,
             "args": {"External id": 7, "correlation": 2,
                      "kernel": "kernel_b"}},
        ]
        _, entries = mapping._runtime_entries(events)
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(
            entry["kernel_trace_shape"] is None for entry in entries))

    def test_missing_required_phase_marker_fails_coverage_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            out_path = os.path.join(tmp, "mapped.json")
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [{
                    "phase": "decode",
                    "representative_layer_id": 1,
                    "pos": 0,
                    "raw_name": "decode_kernel",
                    "selected_bucket": {
                        "phase": "decode",
                        "batch_size": 4,
                        "input_tokens": 0,
                    },
                }]}, fh)
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": []}, fh)
            result = mapping.map_plan(
                plan_path, trace_path, out_path,
                required_phases=["decode"])
            self.assertFalse(result["phase_coverage_complete"])
            self.assertEqual(
                result["missing_marker_buckets"],
                ["decode|1|4|0"])

    def test_source_backed_targeted_probe_covers_missing_marker_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            shape_log = os.path.join(tmp, "shape.jsonl")
            out_path = os.path.join(tmp, "mapped.json")
            target = {
                "row_id": "event-1",
                "phase": "prefill",
                "representative_layer_id": 2,
                "pos": 0,
                "raw_name": "target_kernel",
                "selected_bucket": {
                    "phase": "prefill",
                    "batch_size": 1,
                    "input_tokens": 128,
                },
            }
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [target]}, fh)
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": []}, fh)
            with open(shape_log, "w") as fh:
                fh.write(json.dumps({
                    "op_type": "targeted_python_launcher",
                    "op_instance_id": "geak-targeted-1",
                    "op_path": (
                        "model.layers.2.proj::launcher:pkg.mod:launch"),
                    "phase": "prefill",
                    "layer_id": 2,
                }) + "\n")
            result = mapping.map_plan(
                plan_path, trace_path, out_path,
                shape_log_path=shape_log,
                callable_kernel_map=[{
                    "kernel_pattern": "^target_kernel$",
                    "target": "pkg.mod:launch",
                    "scope": "kernel",
                    "source": [{"path": "/runtime/pkg/mod.py"}],
                }],
                required_phases=["prefill"])
            self.assertTrue(result["phase_coverage_complete"])
            self.assertEqual(result["missing_marker_buckets"], [])
            self.assertEqual(
                result["source_only_covered_buckets"],
                ["prefill|2|1|128"])
            with open(out_path) as fh:
                mapped = json.load(fh)["capture_targets"][0]
            self.assertEqual(
                mapped["source_mapping_status"],
                "source_targeted_launcher_probe")

    def test_mapping_summary_excludes_non_shape_runtime_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            out_path = os.path.join(tmp, "mapped.json")
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [
                    {"phase": "decode", "representative_layer_id": 1,
                     "stage": "communication", "raw_name": "allreduce"},
                    {"phase": "decode", "representative_layer_id": 1,
                     "stage": "elementwise",
                     "raw_name": "__amd_rocclr_fillBufferAligned"},
                ]}, fh)
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": []}, fh)
            result = mapping.map_plan(plan_path, trace_path, out_path)
            self.assertEqual(result["shape_eligible_target_count"], 0)
            self.assertEqual(result["shape_ineligible_target_count"], 2)
            self.assertEqual(result["shape_eligible_match_fraction"], 1.0)
            self.assertEqual(
                result["shape_mapping_by_phase"]["decode"]
                ["shape_eligible_target_count"], 0)

    def test_clean_sequence_mismatch_rejects_cross_trace_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            clean_path = os.path.join(tmp, "clean.json")
            out_path = os.path.join(tmp, "mapped.json")
            target = {
                "phase": "decode", "pattern_id": "P",
                "representative_layer_id": 1, "pos": 0,
                "raw_name": "capture_kernel",
                "selected_bucket": {"phase": "decode", "batch_size": 4,
                                    "input_tokens": 0},
            }
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [target]}, fh)
            marker = (
                "GEAK_SEMANTICS|op=op-1|phase=DECODE|bs=4|toks=4|"
                "layer=1|path=model.layers.1.proj")
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": [
                    {"cat": "user_annotation", "name": marker,
                     "pid": 1, "tid": 2, "ts": 10, "dur": 20},
                    {"cat": "hip_runtime", "name": "hipModuleLaunchKernel",
                     "pid": 1, "tid": 2, "ts": 15, "dur": 1,
                     "args": {"kernel": "capture_kernel", "correlation": 1}},
                ]}, fh)
            with open(clean_path, "w") as fh:
                json.dump({"tables": [{
                    "phase": "decode", "pattern_id": "P",
                    "representative_layer_id": 9,
                    "rows": [{"pos": 0, "raw_name": "different_kernel"}],
                }]}, fh)
            result = mapping.map_plan(
                plan_path, trace_path, out_path,
                clean_table_path=clean_path)
            self.assertEqual(result["clean_table_sequence_audit"]["status"],
                             "fail")
            self.assertEqual(result["sequence_rejected_target_count"], 1)
            with open(out_path) as fh:
                mapped = json.load(fh)["capture_targets"][0]
            self.assertEqual(mapped["runtime_marker_mapping_status"],
                             "clean_sequence_mismatch")

    def test_decode_uses_honest_layer_wrapper_when_marker_is_thread_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            shape_log = os.path.join(tmp, "shape.jsonl")
            out_path = os.path.join(tmp, "mapped.json")
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [{
                    "phase": "decode",
                    "representative_layer_id": 1,
                    "pos": 0,
                    "raw_name": "decode_kernel",
                    "selected_bucket": {
                        "phase": "decode",
                        "batch_size": 4,
                        "input_tokens": 0,
                    },
                }]}, fh)
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": []}, fh)
            with open(shape_log, "w") as fh:
                fh.write(json.dumps({
                    "phase": "decode",
                    "layer_id": 1,
                    "op_path": "model.layers.1",
                    "op_type": "DecoderLayer",
                    "op_instance_id": "decode-layer-1",
                }) + "\n")
            result = mapping.map_plan(
                plan_path, trace_path, out_path, shape_log)
            self.assertTrue(result["phase_coverage_complete"])
            self.assertEqual(
                result["shape_log_layer_fallback_matched_target_count"], 1)
            with open(out_path) as fh:
                target = json.load(fh)["capture_targets"][0]
            self.assertEqual(target["mapping_cardinality"], "1:N")
            self.assertEqual(
                target["source_mapping_status"],
                "runtime_shape_log_layer_wrapper")

    def test_unique_semantic_wrapper_refines_without_order_guessing(self):
        records = [
            {
                "phase": "decode",
                "layer_id": 3,
                "op_path": "model.layers.3",
                "op_type": "DecoderLayer",
                "op_instance_id": "layer",
            },
            {
                "phase": "decode",
                "layer_id": 3,
                "op_path": "model.layers.3.rotary_emb",
                "op_type": "RotaryEmbedding",
                "op_instance_id": "rope",
            },
            {
                "phase": "decode",
                "layer_id": 3,
                "op_path": "model.layers.3.q_proj",
                "op_type": "Linear",
                "op_instance_id": "q",
            },
            {
                "phase": "decode",
                "layer_id": 3,
                "op_path": "model.layers.3.o_proj",
                "op_type": "Linear",
                "op_instance_id": "o",
            },
        ]
        targets = [
            {
                "phase": "decode",
                "representative_layer_id": 3,
                "stage": "rope",
                "raw_name": "rope_kernel",
                "runtime_marker_mapping_status": "not_found",
            },
            {
                "phase": "decode",
                "representative_layer_id": 3,
                "stage": "gemm",
                "raw_name": "gemm_kernel",
                "runtime_marker_mapping_status": "not_found",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            shape_log = os.path.join(tmp, "shape.jsonl")
            with open(shape_log, "w") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")
            count = mapping._apply_shape_log_semantic_wrapper_mapping(
                targets, shape_log, {("decode", 3)})
        self.assertEqual(count, 1)
        self.assertEqual(
            targets[0]["candidate_op_path"],
            "model.layers.3.rotary_emb")
        self.assertEqual(
            targets[0]["mapping_cardinality"], "1:N")
        self.assertEqual(
            targets[1]["runtime_marker_mapping_status"], "not_found")

    def test_partial_marker_bucket_uses_unique_semantic_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            shape_log = os.path.join(tmp, "shape.jsonl")
            out_path = os.path.join(tmp, "mapped.json")
            bucket = {"phase": "decode", "batch_size": 4,
                      "input_tokens": 0}
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [
                    {
                        "phase": "decode", "pattern_id": "P0",
                        "representative_layer_id": 3,
                        "selected_bucket": bucket, "pos": 0,
                        "stage": "gemm", "raw_name": "known_kernel",
                    },
                    {
                        "phase": "decode", "pattern_id": "P0",
                        "representative_layer_id": 3,
                        "selected_bucket": bucket, "pos": 1,
                        "stage": "topk", "raw_name": "missing_kernel",
                    },
                ]}, fh)
            marker = (
                "GEAK_SEMANTICS|op=known-op|phase=DECODE|bs=4|toks=4|"
                "layer=3|path=model.layers.3.q_proj")
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": [
                    {
                        "cat": "user_annotation", "pid": 1, "tid": 2,
                        "name": (
                            "GEAK_LAYER_SCOPE|phase=DECODE|bs=4|toks=4|"
                            "layer=3|path=model.layers.3"),
                        "ts": 0, "dur": 100,
                    },
                    {"cat": "user_annotation", "name": marker,
                     "pid": 1, "tid": 2, "ts": 10, "dur": 20},
                    {"cat": "hip_runtime", "name": "hipLaunchKernel",
                     "pid": 1, "tid": 2, "ts": 15, "dur": 1,
                     "args": {"kernel": "known_kernel", "correlation": 1}},
                ]}, fh)
            with open(shape_log, "w") as fh:
                fh.write(json.dumps({
                    "phase": "decode", "layer_id": 3,
                    "op_path": "model.layers.3.mlp.topk",
                    "op_name": "topk", "op_type": "TopK",
                    "op_instance_id": "topk-op",
                }) + "\n")

            result = mapping.map_plan(
                plan_path, trace_path, out_path, shape_log)
            self.assertEqual(result["matched_target_count"], 2)
            self.assertEqual(
                result["shape_log_semantic_wrapper_matched_target_count"],
                1)
            with open(out_path) as fh:
                mapped = json.load(fh)["capture_targets"]
            self.assertEqual(
                mapped[1]["candidate_op_instance_id"], "topk-op")

    def test_semantic_fallback_rejects_duplicate_wrapper_invocations(self):
        targets = [{
            "phase": "decode", "representative_layer_id": 3,
            "stage": "norm", "raw_name": "norm_kernel",
            "runtime_marker_mapping_status": "not_found",
        }]
        records = [
            {
                "phase": "decode", "layer_id": 3,
                "op_path": "model.layers.3.norm",
                "op_type": "RMSNorm", "op_instance_id": op_id,
            }
            for op_id in ("input-norm", "post-attention-norm")]
        with tempfile.TemporaryDirectory() as tmp:
            shape_log = os.path.join(tmp, "shape.jsonl")
            with open(shape_log, "w") as fh:
                for record in records:
                    fh.write(json.dumps(record) + "\n")
            count = mapping._apply_shape_log_semantic_wrapper_mapping(
                targets, shape_log, {("decode", 3)})
        self.assertEqual(count, 0)
        self.assertEqual(
            targets[0]["runtime_marker_mapping_status"], "not_found")

    def test_semantic_wrapper_prefers_unique_typed_and_deepest_wrapper(self):
        records = [
            {
                "op_path": "model.layers.25.self_attn",
                "op_name": "self_attn",
                "op_type": "DeepseekAttentionMLA",
            },
            {
                "op_path": "model.layers.25.self_attn.attn_mqa",
                "op_name": "attn_mqa",
                "op_type": "RadixAttention",
            },
            {
                "op_path": "model.layers.25.post_attention_layernorm",
                "op_name": "post_attention_layernorm",
                "op_type": "RMSNorm",
            },
            {
                "op_path": "model.layers.25.mlp.gate",
                "op_name": "gate",
                "op_type": "MoEGate",
            },
            {
                "op_path": "model.layers.25.mlp.topk",
                "op_name": "topk",
                "op_type": "TopK",
            },
        ]
        attn = mapping._semantic_wrapper_candidates(
            records, "attn", 25)
        topk = mapping._semantic_wrapper_candidates(
            records, "topk", 25)
        self.assertEqual(
            [item["op_path"] for item in attn],
            ["model.layers.25.self_attn.attn_mqa"])
        self.assertEqual(
            [item["op_path"] for item in topk],
            ["model.layers.25.mlp.topk"])

    def test_source_verified_callable_maps_without_profiler_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            shape_log = os.path.join(tmp, "shape.jsonl")
            record = {
                "op_type": "targeted_python_launcher",
                "op_instance_id": "geak-call-1",
                "op_path": "model.layers.1.proj::launcher:pkg.mod:gemm",
                "phase": "decode",
                "layer_id": 1,
            }
            with open(shape_log, "w") as fh:
                fh.write(json.dumps(record) + "\n")
            target = {
                "phase": "decode",
                "representative_layer_id": 1,
                "raw_name": "_gemm_kernel",
                "runtime_marker_mapping_status": "not_found",
            }
            count = mapping._apply_source_callable_mapping(
                [target], shape_log, [{
                    "kernel_pattern": "^_gemm",
                    "target": "pkg.mod:gemm",
                    "scope": "kernel",
                    "source": "pkg/mod.py:10",
                }])
            self.assertEqual(count, 1)
            self.assertEqual(target["mapping_cardinality"], "1:1")
            self.assertEqual(
                target["source_mapping_status"],
                "source_targeted_launcher_probe")

    def test_source_verified_wrapper_maps_without_order_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            shape_log = os.path.join(tmp, "shape.jsonl")
            record = {
                "op_type": "Attention",
                "op_instance_id": "geak-op-1",
                "op_path": "model.layers.3.attn",
                "phase": "decode",
                "layer_id": 3,
            }
            with open(shape_log, "w") as fh:
                fh.write(json.dumps(record) + "\n")
            target = {
                "phase": "decode",
                "pattern_id": "P_ATTN",
                "representative_layer_id": 3,
                "pos": 7,
                "raw_name": "paged_kernel",
                "runtime_marker_mapping_status": "not_found",
            }
            count = mapping._apply_source_wrapper_mapping(
                [target], shape_log, [{
                    "pattern_id": "P_ATTN",
                    "pos_start": 6,
                    "pos_end": 8,
                    "op_path": "model.layers.{layer}.attn",
                    "source": "model.py:100-120",
                }])
            self.assertEqual(count, 1)
            self.assertEqual(target["mapping_cardinality"], "1:N")
            self.assertEqual(
                target["source_mapping_status"],
                "source_verified_wrapper_probe")

    def test_single_kernel_targeted_launcher_is_p_kernel_candidate(self):
        target = {}
        candidate = {
            "marker": {
                "op_path": (
                    "model.layers.1.q_proj::launcher:"
                    "sglang.fp8_utils:gemm_op"),
                "op_instance_id": "geak-call-1",
                "name": "marker",
                "external_id": None,
            },
            "runtime_name": "hipModuleLaunchKernel",
            "runtime_event_index": 2,
            "correlation": 3,
            "raw_name": "kernel",
            "device_event": None,
        }
        mapping._apply_mapping(
            target, candidate, "/tmp/trace.json", "targeted", 1)
        self.assertEqual(target["mapping_cardinality"], "1:1")
        self.assertTrue(
            target["runtime_marker_evidence"]["targeted_launcher_probe"])

    def test_unique_runtime_containment_maps_clean_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            out_path = os.path.join(tmp, "mapped.json")
            target = {
                "phase": "prefill",
                "representative_layer_id": 1,
                "pos": 2,
                "row_id": "event-2",
                "raw_name": (
                    "_gemm_BLOCK_SIZE_K_128_GRID_MN_10_cache_modifier_NONE"),
            }
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [target]}, fh)
            marker = (
                "GEAK_SEMANTICS|op=geak-op-1|phase=EXTEND|bs=1|"
                "toks=8|layer=1|path=model.layers.1.q_proj")
            events = [
                {"cat": "user_annotation", "name": marker,
                 "pid": 1, "tid": 2, "ts": 10, "dur": 100,
                 "args": {"External id": 70}},
                {"cat": "cuda_runtime", "name": "hipModuleLaunchKernel",
                 "pid": 1, "tid": 2, "ts": 50, "dur": 2,
                 "args": {
                     "kernel": (
                         "_gemm_BLOCK_SIZE_K_128_GRID_MN_99_"
                         "cache_modifier_NONE"),
                     "correlation": 7,
                 }},
                {"cat": "kernel", "name": "_gemm", "ts": 200, "dur": 3,
                 "args": {"correlation": 7}},
            ]
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.map_plan(plan_path, trace_path, out_path)
            self.assertEqual(result["matched_target_count"], 1)
            with open(out_path) as fh:
                mapped = json.load(fh)["capture_targets"][0]
            self.assertEqual(
                mapped["candidate_op_instance_id"], "geak-op-1")
            self.assertEqual(
                mapped["candidate_op_path"], "model.layers.1.q_proj")
            self.assertEqual(
                mapped["source_mapping_status"],
                "runtime_marker_contained")

    def test_decode_selects_one_eager_forward_and_accepts_token_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            out_path = os.path.join(tmp, "mapped.json")
            target = {
                "phase": "decode",
                "representative_layer_id": 1,
                "selected_bucket": {
                    "phase": "decode",
                    "batch_size": 4,
                    "input_tokens": 0,
                },
                "pos": 0,
                "row_id": "event-1",
                "raw_name": "decode_kernel",
            }
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [target]}, fh)

            events = []
            for forward, base in enumerate((10, 100)):
                marker = (
                    "GEAK_SEMANTICS|op=geak-op-%s|phase=DECODE|bs=4|"
                    "toks=4|layer=1|path=model.layers.1.q_proj" %
                    (forward + 1))
                events.extend([
                    {"cat": "user_annotation", "name": marker,
                     "pid": 1, "tid": 2, "ts": base, "dur": 50},
                    {"cat": "hip_runtime", "name": "hipModuleLaunchKernel",
                     "pid": 1, "tid": 2, "ts": base + 10, "dur": 2,
                     "args": {
                         "kernel": "decode_kernel",
                         "correlation": forward + 1,
                     }},
                ])
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": events}, fh)

            result = mapping.map_plan(plan_path, trace_path, out_path)
            self.assertEqual(result["matched_target_count"], 1)
            self.assertEqual(result["ambiguous_target_count"], 0)
            with open(out_path) as fh:
                mapped = json.load(fh)["capture_targets"][0]
            self.assertEqual(
                mapped["candidate_op_instance_id"], "geak-op-1")

    def test_decode_layer_scope_keeps_repeated_wrapper_in_one_forward(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            out_path = os.path.join(tmp, "mapped.json")
            bucket = {
                "phase": "decode",
                "batch_size": 4,
                "input_tokens": 0,
            }
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [
                    {
                        "phase": "decode", "pattern_id": "P0",
                        "representative_layer_id": 1,
                        "selected_bucket": bucket, "pos": 0,
                        "row_id": "event-1", "raw_name": "attn_kernel",
                    },
                    {
                        "phase": "decode", "pattern_id": "P0",
                        "representative_layer_id": 1,
                        "selected_bucket": bucket, "pos": 1,
                        "row_id": "event-2", "raw_name": "mlp_kernel",
                    },
                ]}, fh)

            def marker(op_id, path, ts, dur):
                return {
                    "cat": "user_annotation", "pid": 1, "tid": 2,
                    "name": (
                        "GEAK_SEMANTICS|op=%s|phase=DECODE|bs=4|"
                        "toks=4|layer=1|path=%s" % (op_id, path)),
                    "ts": ts, "dur": dur,
                }

            repeated = (
                "model.layers.1::launcher:"
                "sglang.srt.layers.layernorm:_forward_with_allreduce_fusion")
            events = [
                {
                    "cat": "user_annotation", "pid": 1, "tid": 2,
                    "name": (
                        "GEAK_LAYER_SCOPE|phase=DECODE|bs=4|toks=4|"
                        "layer=1|path=model.layers.1"),
                    "ts": 10, "dur": 80,
                },
                marker("geak-call-1", repeated, 12, 20),
                {"cat": "hip_runtime", "name": "hipLaunchKernel",
                 "pid": 1, "tid": 2, "ts": 15, "dur": 1,
                 "args": {"kernel": "attn_kernel", "correlation": 1}},
                marker("geak-call-2", repeated, 40, 20),
                {"cat": "hip_runtime", "name": "hipLaunchKernel",
                 "pid": 1, "tid": 2, "ts": 45, "dur": 1,
                 "args": {"kernel": "mlp_kernel", "correlation": 2}},
                # A later replay of the same bucket must not be mixed in.
                {
                    "cat": "user_annotation", "pid": 1, "tid": 2,
                    "name": (
                        "GEAK_LAYER_SCOPE|phase=DECODE|bs=4|toks=4|"
                        "layer=1|path=model.layers.1"),
                    "ts": 100, "dur": 80,
                },
                marker("geak-call-3", repeated, 102, 20),
                {"cat": "hip_runtime", "name": "hipLaunchKernel",
                 "pid": 1, "tid": 2, "ts": 105, "dur": 1,
                 "args": {"kernel": "attn_kernel", "correlation": 3}},
                marker("geak-call-4", repeated, 140, 20),
                {"cat": "hip_runtime", "name": "hipLaunchKernel",
                 "pid": 1, "tid": 2, "ts": 145, "dur": 1,
                 "args": {"kernel": "mlp_kernel", "correlation": 4}},
            ]
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": events}, fh)

            result = mapping.map_plan(plan_path, trace_path, out_path)
            self.assertEqual(result["matched_target_count"], 2)
            self.assertEqual(result["ambiguous_target_count"], 0)
            self.assertEqual(
                result["selected_forward_marker_counts"]
                ["decode|1|4|0"], 2)
            with open(out_path) as fh:
                mapped = json.load(fh)["capture_targets"]
            self.assertEqual(
                [item["candidate_op_instance_id"] for item in mapped],
                ["geak-call-1", "geak-call-2"])

    def test_repeated_kernel_is_resolved_by_mapped_neighbor_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            out_path = os.path.join(tmp, "mapped.json")
            bucket = {
                "phase": "decode",
                "batch_size": 4,
                "input_tokens": 0,
            }
            targets = [
                {"phase": "decode", "representative_layer_id": 1,
                 "selected_bucket": bucket, "pos": pos,
                 "row_id": "event-%s" % pos, "raw_name": name}
                for pos, name in enumerate(("before", "generic", "after"))
            ]
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": targets}, fh)

            events = [{
                "cat": "user_annotation", "pid": 1, "tid": 2,
                "name": (
                    "GEAK_LAYER_SCOPE|phase=DECODE|bs=4|toks=4|"
                    "layer=1|path=model.layers.1"),
                "ts": 0, "dur": 100,
            }]
            launches = (
                ("generic", "noise", 10),
                ("before", "before", 30),
                ("generic", "target", 50),
                ("after", "after", 70),
            )
            for correlation, (kernel, path, timestamp) in enumerate(
                    launches, 1):
                marker = (
                    "GEAK_SEMANTICS|op=geak-op-%s|phase=DECODE|bs=4|"
                    "toks=4|layer=1|path=model.layers.1.%s" %
                    (correlation, path))
                events.extend([
                    {"cat": "user_annotation", "name": marker,
                     "pid": 1, "tid": 2, "ts": timestamp, "dur": 10},
                    {"cat": "hip_runtime", "name": "hipModuleLaunchKernel",
                     "pid": 1, "tid": 2, "ts": timestamp + 1, "dur": 1,
                     "args": {
                         "kernel": kernel,
                         "correlation": correlation,
                     }},
                ])
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": events}, fh)

            result = mapping.map_plan(plan_path, trace_path, out_path)
            self.assertEqual(result["matched_target_count"], 3)
            self.assertEqual(result["ambiguous_target_count"], 0)
            with open(out_path) as fh:
                mapped = json.load(fh)["capture_targets"]
            self.assertEqual(
                mapped[1]["candidate_op_instance_id"], "geak-op-3")
            self.assertIn(
                "neighboring clean kernel positions",
                mapped[1]["runtime_marker_evidence"]["rule"])

    def test_prefill_uses_nearest_compatible_probe_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "plan.json")
            trace_path = os.path.join(tmp, "trace.json")
            out_path = os.path.join(tmp, "mapped.json")
            target = {
                "phase": "prefill",
                "representative_layer_id": 1,
                "selected_bucket": {
                    "phase": "prefill",
                    "batch_size": 1,
                    "input_tokens": 7238,
                },
                "pos": 0,
                "row_id": "event-1",
                "raw_name": "kernel",
            }
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [target]}, fh)
            marker = (
                "GEAK_SEMANTICS|op=geak-op-1|phase=EXTEND|bs=3|"
                "toks=22272|layer=1|path=model.layers.1.proj")
            events = [
                {"cat": "user_annotation", "name": marker,
                 "pid": 1, "tid": 2, "ts": 10, "dur": 20},
                {"cat": "hip_runtime", "name": "hipModuleLaunchKernel",
                 "pid": 1, "tid": 2, "ts": 15, "dur": 1,
                 "args": {"kernel": "kernel", "correlation": 1}},
            ]
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": events}, fh)

            result = mapping.map_plan(plan_path, trace_path, out_path)
            self.assertEqual(result["matched_target_count"], 1)
            with open(out_path) as fh:
                mapped = json.load(fh)["capture_targets"][0]
            self.assertEqual(
                mapped["candidate_op_instance_id"], "geak-op-1")

    def test_decode_mla_anchors_separate_k_and_v_absorb(self):
        targets = [
            {"phase": "decode", "pattern_id": "P", "representative_layer_id": 2,
             "pos": 0, "stage": "norm", "raw_name": "fused_qk_rmsnorm"},
            {"phase": "decode", "pattern_id": "P", "representative_layer_id": 2,
             "pos": 1, "stage": "gemm", "raw_name": "batched_gemm_prequant"},
            {"phase": "decode", "pattern_id": "P", "representative_layer_id": 2,
             "pos": 2, "stage": "kv_cache", "raw_name": "rope_cache"},
            {"phase": "decode", "pattern_id": "P", "representative_layer_id": 2,
             "pos": 3, "stage": "attn", "raw_name": "mla_attention"},
            {"phase": "decode", "pattern_id": "P", "representative_layer_id": 2,
             "pos": 4, "stage": "attn", "raw_name": "mla_reduce"},
            {"phase": "decode", "pattern_id": "P", "representative_layer_id": 2,
             "pos": 5, "stage": "elementwise", "raw_name": "copy"},
            {"phase": "decode", "pattern_id": "P", "representative_layer_id": 2,
             "pos": 6, "stage": "gemm", "raw_name": "Cijk_bf16"},
            {"phase": "decode", "pattern_id": "P", "representative_layer_id": 2,
             "pos": 7, "stage": "quant", "raw_name": "quant"},
            {"phase": "decode", "pattern_id": "P", "representative_layer_id": 2,
             "pos": 8, "stage": "communication", "raw_name": "allreduce"},
        ]
        mapping._annotate_decode_semantic_regions(targets)
        self.assertEqual(targets[1]["semantic_region"], "k_absorb")
        self.assertEqual(targets[2]["semantic_region"], "rope_kv")
        self.assertEqual(targets[5]["semantic_region"], "v_absorb")
        self.assertEqual(targets[6]["semantic_region"], "v_absorb")
        self.assertEqual(targets[7]["semantic_region"], "o_proj")

    def test_unique_torch_bmm_probe_maps_only_vabsorb_gemm(self):
        with tempfile.TemporaryDirectory() as tmp:
            shape_log = os.path.join(tmp, "shape.jsonl")
            record = {
                "phase": "decode", "layer_id": 2,
                "op_instance_id": "bmm-1",
                "op_type": "targeted_python_launcher",
                "op_path": "model.layers.2.self_attn::launcher:torch:bmm",
            }
            with open(shape_log, "w") as fh:
                fh.write(json.dumps(record) + "\n")
            targets = [{
                "phase": "decode", "representative_layer_id": 2,
                "stage": "gemm", "semantic_region": "v_absorb",
                "semantic_region_path": "model.layers.2.self_attn.v_absorb",
                "runtime_marker_mapping_status": "not_found",
            }]
            self.assertEqual(
                mapping._apply_vabsorb_bmm_probe(targets, shape_log), 1)
            self.assertEqual(targets[0]["mapping_cardinality"], "1:1")
            self.assertEqual(
                targets[0]["candidate_op_path"],
                "model.layers.2.self_attn.v_absorb")


if __name__ == "__main__":
    unittest.main()
