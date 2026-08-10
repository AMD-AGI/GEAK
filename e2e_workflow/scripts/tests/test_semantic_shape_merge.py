import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_shape_merge as merge


class SemanticShapeMergeTest(unittest.TestCase):
    def _write(self, root, name, value, jsonl=False):
        path = os.path.join(root, name)
        with open(path, "w") as fh:
            if jsonl:
                for item in value:
                    fh.write(json.dumps(item) + "\n")
            else:
                json.dump(value, fh)
        return path

    def test_layer_wrapper_shape_does_not_claim_kernel_operand_roles(self):
        row = {
            "stage": "gemm",
            "semantic_evidence": {
                "level": "P",
                "probe_scope": "wrapper",
                "wrapper_scope": "phase_layer_wrapper",
            },
            "shape": {"logger_schema": {"tensors": [
                {
                    "io": "input",
                    "tensor_path": "args[0]",
                    "shape": [4],
                    "dtype": "int64",
                },
                {
                    "io": "input",
                    "tensor_path": "args[1]",
                    "shape": [4, 8],
                    "dtype": "bfloat16",
                },
                {
                    "io": "output",
                    "tensor_path": "output",
                    "shape": [4, 8],
                    "dtype": "bfloat16",
                },
            ]}},
        }
        text = merge._shape_text(row)
        self.assertIn("wrapper_input_0=INT64[4]", text)
        self.assertIn("wrapper_output_0=BF16[4×8]", text)
        self.assertNotIn("weight=", text)

    def test_k_first_parent_context_and_trace_m_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = {
                "schema_version": 2,
                "tables": [{
                    "phase": "decode",
                    "pattern_id": "P",
                    "representative_layer_id": 2,
                    "selected_bucket": {
                        "phase": "decode", "batch_size": 4,
                        "input_tokens": 4},
                    "event_count": 2,
                    "layer_total_us": 3.0,
                    "rows": [
                        {
                            "pos": 0, "row_id": "event-1",
                            "raw_event_index": 1, "device_seq_index": 1,
                            "raw_name": "exact", "short_name": "exact",
                            "duration_us": 1.0,
                            "shape": {"source": "kernel_exact",
                                      "input_dims": [[4, 8]]},
                            "parent_operator": {"canonical_op": "aten::mm"},
                        },
                        {
                            "pos": 1, "row_id": "event-2",
                            "raw_event_index": 2, "device_seq_index": 2,
                            "raw_name": "native_gemm",
                            "short_name": "native_gemm", "duration_us": 2.0,
                            "shape": {"source": "unresolved",
                                      "input_dims": [[4, 8]]},
                            "parent_operator": {"canonical_op": "unresolved"},
                        },
                    ],
                }],
            }
            plan = {
                "capture_targets": [{
                    "row_id": "event-2",
                    "candidate_wrapper": "model.layers.2.proj",
                    "candidate_terminal_launcher": None,
                    "mapping_cardinality": "1:N",
                    "parent_operator": "unresolved",
                }],
            }
            records = [
                {
                    "phase": "decode", "rank": 0, "layer_id": 2,
                    "batch_size": 8, "input_tokens": 8,
                    "op_instance_id": "op1", "op_name": "proj",
                    "op_type": "Linear",
                    "op_path": "model.layers.2.proj",
                    "io": "input", "tensor_path": "args[0]",
                    "arg_name": "input", "tensor_role": "input",
                    "shape": [8, 8], "dtype": "bf16",
                    "device": "cuda:0", "stride": [8, 1],
                },
                {
                    "phase": "decode", "rank": 0, "layer_id": 2,
                    "batch_size": 8, "input_tokens": 8,
                    "op_instance_id": "op1", "op_name": "proj",
                    "op_type": "Linear",
                    "op_path": "model.layers.2.proj",
                    "io": "weight", "tensor_path": "param.weight",
                    "arg_name": "weight", "tensor_role": "weight",
                    "shape": [16, 8], "dtype": "fp8",
                    "device": "cuda:0", "stride": [8, 1],
                },
                {
                    "phase": "decode", "rank": 0, "layer_id": 2,
                    "batch_size": 8, "input_tokens": 8,
                    "op_instance_id": "op1", "op_name": "proj",
                    "op_type": "Linear",
                    "op_path": "model.layers.2.proj",
                    "io": "output", "tensor_path": "output",
                    "arg_name": "output", "tensor_role": "output",
                    "shape": [8, 16], "dtype": "bf16",
                    "device": "cuda:0", "stride": [16, 1],
                },
            ]
            result = merge.merge(
                self._write(tmp, "table.json", table),
                self._write(tmp, "plan.json", plan),
                self._write(tmp, "shape.jsonl", records, jsonl=True),
                os.path.join(tmp, "out"))
            self.assertEqual(result["status"], "pass")
            with open(result["semantic_table_json"]) as fh:
                rows = json.load(fh)["tables"][0]["rows"]
            self.assertEqual(rows[0]["semantic_evidence"]["level"], "K")
            self.assertEqual(rows[0]["shape"]["input_dims"], [[4, 8]])
            self.assertEqual(rows[1]["semantic_evidence"]["level"], "P")
            self.assertEqual(
                rows[1]["semantic_evidence"]["probe_scope"], "wrapper")
            linear = rows[1]["semantic_evidence"]["schema"]["linear_interface"]
            self.assertEqual(linear["M"]["value"], 4)
            self.assertEqual(linear["M"]["source"], "clean_trace")
            self.assertEqual(linear["K"]["value"], 8)
            self.assertEqual(linear["N"]["value"], 16)
            with open(result["semantic_table_json"]) as fh:
                layer_io = json.load(fh)["tables"][0]["layer_io"]
            self.assertEqual(layer_io["bucket_match"], "compatible")
            self.assertEqual(layer_io["input"]["shape"], [8, 8])
            self.assertEqual(layer_io["input"]["effective_shape"], [4, 8])
            self.assertEqual(
                layer_io["input"]["axis_0_source"], "clean_trace_step")
            with open(result["semantic_table_md"]) as fh:
                markdown = fh.read()
            self.assertIn("shape type", markdown)
            self.assertNotIn("| evidence |", markdown)
            self.assertIn(
                "P(wrapper): x=BF16[4×8]<br>weight=FP8[16×8]"
                "<br><br>y=BF16[4×16]",
                markdown)
            self.assertIn(
                "representative layer I/O: `source=shape_logger, "
                "input=BF16[4×8], output=BF16[4×16], bucket=compatible`",
                markdown)

    def test_kernel_gemm_shape_uses_semantic_roles_and_canonical_dtypes(self):
        row = {
            "stage": "gemm",
            "semantic_evidence": {"level": "P", "probe_scope": "kernel"},
            "shape": {"logger_schema": {"tensors": [
                {"io": "input", "tensor_path": "args[0]",
                 "dtype": "float8_e4m3fnuz", "shape": [7238, 7168]},
                {"io": "input", "tensor_path": "args[1]",
                 "dtype": "float8_e4m3fnuz", "shape": [2112, 7168]},
                {"io": "output", "tensor_path": "output",
                 "dtype": "bfloat16", "shape": [7238, 2112]},
            ]}},
        }
        self.assertEqual(
            merge._shape_text(row),
            "P(kernel): x=FP8[7238×7168]<br>"
            "weight=FP8[2112×7168]<br><br>"
            "y=BF16[7238×2112]")

    def test_trace_quant_shape_is_reordered_by_operator_semantics(self):
        row = {
            "stage": "quant",
            "parent_operator": {
                "canonical_op": "aiter::dynamic_per_token_scaled_quant"},
            "semantic_evidence": {"level": "K"},
            "shape": {
                "input_types": [
                    "c10::Float8_e4m3fnuz", "c10::BFloat16", "float"],
                "input_dims": [[7238, 7168], [405328, 128], [7238, 56]],
            },
        }
        self.assertEqual(
            merge._shape_text(row),
            "K: x=BF16[405328×128]<br><br>"
            "y=FP8[7238×7168]<br>scale=FP32[7238×56]")

    def test_trace_kv_cache_shape_uses_cache_semantics(self):
        row = {
            "stage": "kv_cache",
            "parent_operator": {"canonical_op": "sglang::store_cache"},
            "semantic_evidence": {"level": "K"},
            "shape": {
                "input_types": [
                    "c10::BFloat16", "c10::BFloat16",
                    "c10::BFloat16", "c10::BFloat16", "long int"],
                "input_dims": [
                    [4, 256], [4, 256], [100, 256], [100, 256], [4]],
            },
        }
        self.assertEqual(
            merge._shape_text(row),
            "K: k=BF16[4×256]<br>v=BF16[4×256]<br>"
            "slot_mapping=INT64[4]<br><br>"
            "k_cache_out=BF16[100×256]<br>"
            "v_cache_out=BF16[100×256]")

    def test_unknown_axis_and_missing_wrapper_stay_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = {
                "pos": 0, "row_id": "event-1", "raw_event_index": 1,
                "device_seq_index": 1, "raw_name": "opaque",
                "short_name": "opaque", "stage": "memory",
                "duration_us": 1.0,
                "shape": {"source": "unresolved", "input_dims": []},
                "parent_operator": {"canonical_op": "unresolved"},
            }
            table = {"tables": [{
                "phase": "prefill", "pattern_id": "P",
                "representative_layer_id": 1,
                "selected_bucket": {"batch_size": 1, "input_tokens": 8},
                "rows": [row]}]}
            plan = {"capture_targets": [{
                "row_id": "event-1", "parent_operator": "unresolved",
                "candidate_op_path": None, "candidate_wrapper": None}]}
            result = merge.merge(
                self._write(tmp, "table.json", table),
                self._write(tmp, "plan.json", plan),
                self._write(tmp, "shape.jsonl", [{
                    "phase": "prefill", "rank": 0, "layer_id": 1,
                    "batch_size": 1, "input_tokens": 8,
                    "op_instance_id": "op1", "op_name": "proj",
                    "op_type": "Linear",
                    "op_path": "model.layers.1.proj",
                    "io": "input", "tensor_path": "args[0]",
                    "arg_name": "input", "tensor_role": "input",
                    "shape": [8, 4], "dtype": "bf16",
                    "device": "cuda:0", "stride": [4, 1],
                }], jsonl=True),
                os.path.join(tmp, "out"))
            with open(result["semantic_table_json"]) as fh:
                output = json.load(fh)["tables"][0]["rows"][0]
            self.assertEqual(output["semantic_evidence"]["level"], "U")
            self.assertEqual(output["shape"]["source"], "unresolved")
            self.assertEqual(
                output["semantic_evidence"]["source"],
                "no_unique_parent_wrapper")
            self.assertEqual(
                output["semantic_evidence"]["reason_code"],
                "no_source_confirmed_wrapper")

    def test_geak_metadata_schema_is_flattened_for_merge(self):
        records = [{
            "schema": "geak.semantics_metadata.v1",
            "op_instance_id": "op-1", "rank": 0, "layer_id": 22,
            "phase": "decode", "bucket": "decode:4:4",
            "target_op": "proj", "op_path": "model.layers.22.proj",
            "mapping_cardinality": "1:N",
            "inputs": {"kind": "tuple", "items": [{
                "kind": "tensor", "shape": [4, 8],
                "dtype": "torch.bfloat16", "device": "cuda:0",
                "stride": [8, 1], "contiguous": True, "alias_id": "a0"}]},
            "kwargs": {"kind": "dict", "items": {}},
            "output": {"kind": "tensor", "shape": [4, 16],
                       "dtype": "torch.bfloat16", "device": "cuda:0",
                       "stride": [16, 1], "contiguous": True,
                       "alias_id": "a1"},
        }]
        groups = merge._groups(records)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["batch_size"], 4)
        self.assertEqual(groups[0]["input_tokens"], 4)
        self.assertEqual(
            [tensor["io"] for tensor in groups[0]["tensors"]],
            ["input", "output"])

    def test_runtime_buffer_operation_is_not_promoted_to_probe_shape(self):
        self.assertTrue(merge._is_runtime_internal({
            "short_name": "__amd_rocclr_fillBufferAligned"}))
        self.assertFalse(merge._is_runtime_internal({
            "short_name": "_gemm_a8w8"}))

    def test_unmatched_memcpy_has_specific_reason_code(self):
        reason_code, _ = merge._unavailable_reason(
            {"event_type": "gpu_memcpy", "short_name": "Memcpy"},
            {"runtime_marker_mapping_status": "not_found"}, 0)
        self.assertEqual(
            reason_code, "runtime_copy_without_unique_tensor")


if __name__ == "__main__":
    unittest.main()
