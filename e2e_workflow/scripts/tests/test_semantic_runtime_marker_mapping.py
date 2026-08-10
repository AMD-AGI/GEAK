import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_runtime_marker_mapping as mapping


class RuntimeMarkerMappingTest(unittest.TestCase):
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
                plan_path, trace_path, out_path)
            self.assertFalse(result["phase_coverage_complete"])
            self.assertEqual(
                result["missing_marker_buckets"],
                ["decode|1|4|0"])

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

            events = []
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


if __name__ == "__main__":
    unittest.main()
