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

    def test_kernel_trace_shape_has_priority_over_wrapper_shape_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = {
                "schema_version": 2,
                "tables": [{
                    "phase": "decode",
                    "pattern_id": "P0",
                    "pattern_layer_ids": [26],
                    "pattern_layer_count": 1,
                    "representative_layer_id": 26,
                    "selected_bucket": {
                        "phase": "decode", "batch_size": 4,
                        "input_tokens": 0},
                    "event_count": 1,
                    "layer_total_us": 6.0,
                    "rows": [{
                        "pos": 0, "row_id": "event-1",
                        "raw_event_index": 1, "device_seq_index": 1,
                        "raw_name": "quant_kernel",
                        "short_name": "quant_kernel",
                        "duration_us": 6.0, "stage": "quant",
                        "event_type": "kernel",
                        "shape": {"source": "unresolved",
                                  "input_dims": []},
                        "parent_operator": {
                            "canonical_op": "unresolved"},
                    }],
                }],
            }
            plan = {
                "runtime_marker_mapping": {
                    "clean_table_sequence_audit": {"status": "pass"}},
                "capture_targets": [{
                    "row_id": "event-1", "phase": "decode",
                    "pattern_id": "P0", "pos": 0,
                    "representative_layer_id": 26,
                    "candidate_op_instance_id": "wrapper-op",
                    "candidate_wrapper": "model.layers.26.proj",
                    "candidate_terminal_launcher": (
                        "aiter::dynamic_per_token_scaled_quant"),
                    "mapping_cardinality": "1:1",
                    "kernel_trace_shape": {
                        "source": "graph_capture_trace_external_id",
                        "mapping_cardinality": "1:1",
                        "op_name": (
                            "aiter::dynamic_per_token_scaled_quant"),
                        "external_id": 7,
                        "cpu_event_index": 10,
                        "runtime_launch_count": 1,
                        "bucket_match": "exact",
                        "input_dims": [
                            [4, 4096], [128, 128], [4, 32], []],
                        "input_types": [
                            "float8_e4m3fnuz", "bfloat16", "float32", ""],
                        "raw_operands": [
                            {"arg_index": 0, "shape": [4, 4096],
                             "dtype": "float8_e4m3fnuz", "stride": None},
                            {"arg_index": 1, "shape": [128, 128],
                             "dtype": "bfloat16", "stride": None},
                            {"arg_index": 2, "shape": [4, 32],
                             "dtype": "float32", "stride": None},
                            {"arg_index": 3, "shape": [],
                             "dtype": "", "stride": None},
                        ],
                        "operator_schema_resolution": {
                            "status": "matched_unique"},
                        "operator_schema": {
                            "name": "aiter::dynamic_per_token_scaled_quant",
                            "schema": "aiter::dynamic_per_token_scaled_quant(Tensor(a0!) out, Tensor(a1!) input, Tensor(a2!) scales, Tensor? scale_ub=None) -> ()",
                            "arguments": [
                                {"index": 0, "name": "out",
                                 "type": "Tensor",
                                 "alias_info": {"is_write": True}},
                                {"index": 1, "name": "input",
                                 "type": "Tensor",
                                 "alias_info": {"is_write": True}},
                                {"index": 2, "name": "scales",
                                 "type": "Tensor",
                                 "alias_info": {"is_write": True}},
                                {"index": 3, "name": "scale_ub",
                                 "type": "Optional[Tensor]",
                                 "alias_info": None},
                            ],
                        },
                    },
                }],
            }
            # Deliberately wrong/broad wrapper tensors.  They must remain
            # contextual and must not overwrite the one-to-one trace shape.
            records = [{
                "phase": "decode", "rank": 0, "layer_id": 26,
                "batch_size": 4, "input_tokens": 4,
                "op_instance_id": "wrapper-op", "op_name": "proj",
                "op_type": "MergedColumnParallelLinear",
                "op_path": "model.layers.26.proj",
                "io": "input", "tensor_path": "args[0]",
                "shape": [4, 4096], "dtype": "bfloat16",
            }, {
                "phase": "decode", "rank": 0, "layer_id": 26,
                "batch_size": 4, "input_tokens": 4,
                "op_instance_id": "wrapper-op", "op_name": "proj",
                "op_type": "MergedColumnParallelLinear",
                "op_path": "model.layers.26.proj",
                "io": "weight", "tensor_path": "parameters.weight",
                "shape": [2560, 4096], "dtype": "float8_e4m3fnuz",
            }]
            result = merge.merge(
                self._write(tmp, "table.json", table),
                self._write(tmp, "plan.json", plan),
                self._write(tmp, "shape.jsonl", records, jsonl=True),
                os.path.join(tmp, "out"))
            with open(result["semantic_table_json"]) as fh:
                row = json.load(fh)["tables"][0]["rows"][0]
            self.assertEqual(row["semantic_evidence"]["level"], "P")
            self.assertEqual(
                row["semantic_evidence"]["probe_scope"], "kernel")
            self.assertEqual(
                row["semantic_evidence"]["source"],
                "graph_capture_trace_external_id")
            self.assertEqual(row["shape"]["source"], "runtime_trace_kernel")
            self.assertEqual(
                row["shape"]["input_dims"],
                [[4, 4096], [128, 128], [4, 32], []])
            self.assertNotIn([2560, 4096], row["shape"]["input_dims"])
            self.assertEqual(
                row["parent_operator"]["canonical_op"],
                "aiter::dynamic_per_token_scaled_quant")
            kernel_shape = row["shape"]["kernel_shape"]
            self.assertEqual(
                [operand["schema_name"] for operand in
                 kernel_shape["operands"]],
                ["out", "input", "scales", "scale_ub"])
            self.assertEqual(
                kernel_shape["operands"][1]["semantic_role"], "input")
            self.assertEqual(
                kernel_shape["operands"][0]["direction"], "output")
            self.assertEqual(
                kernel_shape["operands"][2]["direction"],
                "mutable_unresolved")

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
                    "pattern_layer_ids": [1, 2, 3],
                    "pattern_layer_count": 3,
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
                                      "input_dims": [[4, 8], [8, 16]],
                                      "input_types": ["bf16", "bf16"]},
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
                "runtime_marker_mapping": {
                    "clean_table_sequence_audit": {"status": "pass"}},
                "capture_targets": [{
                    "row_id": "old-event-2",
                    "phase": "decode", "pattern_id": "P", "pos": 1,
                    "candidate_op_instance_id": "op1",
                    "candidate_wrapper": "model.layers.2.proj",
                    "candidate_terminal_launcher": None,
                    "mapping_cardinality": "1:N",
                    "parent_operator": "unresolved",
                }],
            }
            plan["operator_schema_manifest"] = self._write(
                tmp, "schemas.json", {"schemas": [{
                    "name": "aten::mm", "overload_name": "",
                    "qualified_name": "aten::mm",
                    "schema": "aten::mm(Tensor self, Tensor mat2) -> Tensor",
                    "arguments": [
                        {"index": 0, "name": "self", "type": "Tensor",
                         "has_default": False, "alias_info": None},
                        {"index": 1, "name": "mat2", "type": "Tensor",
                         "has_default": False, "alias_info": None},
                    ],
                    "returns": [],
                }]})
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
            self.assertEqual(
                rows[0]["shape"]["input_dims"], [[4, 8], [8, 16]])
            self.assertEqual(
                rows[0]["shape"]["kernel_shape"]["operands"][0]
                ["schema_name"], "self")
            self.assertEqual(rows[1]["semantic_evidence"]["level"], "P")
            self.assertEqual(
                rows[1]["semantic_evidence"]["clean_row_binding"],
                "verified_pattern_position")
            self.assertEqual(
                rows[1]["semantic_evidence"]["probe_scope"], "wrapper")
            self.assertEqual(rows[1]["shape"]["input_dims"],
                             [[4, 8], [16, 8], [4, 16]])
            self.assertEqual(rows[1]["shape"]["input_types"],
                             ["bf16", "fp8", "bf16"])
            linear = rows[1]["semantic_evidence"]["schema"]["linear_interface"]
            self.assertEqual(linear["M"]["value"], 4)
            self.assertEqual(linear["M"]["source"], "clean_trace")
            self.assertEqual(linear["K"]["value"], 8)
            self.assertEqual(linear["N"]["value"], 16)
            with open(result["semantic_table_json"]) as fh:
                layer_io = json.load(fh)["tables"][0]["layer_io"]
            self.assertEqual(layer_io["bucket_match"], "compatible")
            self.assertEqual(layer_io["input"]["shape"], [8, 8])
            self.assertEqual(layer_io["input"]["logger_shape"], [8, 8])
            self.assertEqual(layer_io["input"]["effective_shape"], [4, 8])
            self.assertEqual(
                layer_io["input"]["axis_0_source"], "clean_trace_step")
            self.assertEqual(
                layer_io["input"]["axis_0_alignment"]["status"],
                "axis_0_rewritten")
            with open(result["semantic_table_md"]) as fh:
                markdown = fh.read()
            self.assertIn("pattern layers (3): `[1, 2, 3]`", markdown)
            self.assertIn("shape type", markdown)
            self.assertIn("layer total %", markdown)
            self.assertNotIn("| evidence |", markdown)
            self.assertIn(
                "P(wrapper): x=BF16[4×8]<br>weight=FP8[16×8]"
                "<br><br>y=BF16[4×16]",
                markdown)
            self.assertIn(
                "representative layer I/O: `source=shape_logger, "
                "input=BF16[4×8], output=BF16[4×16], bucket=compatible`",
                markdown)

    def test_axis_0_alignment_preserves_weights_indices_and_ambiguous_1d(self):
        table = {"selected_bucket": {"batch_size": 4, "input_tokens": 0}}
        group = {"batch_size": 1, "input_tokens": 1}
        cases = [
            ({"io": "weight", "tensor_path": "parameters.weight",
              "shape": [1, 8], "dtype": "float8_e4m3fnuz"}, [1, 8]),
            ({"io": "input", "tensor_path": "args.indices",
              "shape": [1, 8], "dtype": "int64"}, [1, 8]),
            ({"io": "input", "tensor_path": "args[0]",
              "shape": [1], "dtype": "bfloat16"}, [1]),
            ({"io": "input", "tensor_path": "args[0]",
              "shape": [1, 8], "dtype": "bfloat16"}, [4, 8]),
            ({"io": "output", "tensor_path": "output.scale",
              "shape": [1, 2], "dtype": "float32"}, [4, 2]),
        ]
        for tensor, expected in cases:
            effective, audit = merge._axis_0_alignment(
                tensor, group, table, exact_bucket=False)
            self.assertEqual(effective, expected)
            self.assertEqual(audit["status"] == "axis_0_rewritten",
                             expected != tensor["shape"])

    def test_verified_pattern_position_allows_cross_layer_shape_group(self):
        groups = [{
            "rank": 0, "phase": "decode", "layer_id": 13,
            "op_instance_id": "capture-op"}]
        target = {"candidate_op_instance_id": "capture-op"}
        table = {"phase": "decode", "representative_layer_id": 32}
        self.assertEqual(
            merge._candidate_groups(
                {}, target, groups, table,
                allow_verified_cross_layer=True),
            groups)
        self.assertEqual(
            merge._candidate_groups({}, target, groups, table), [])

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

    def test_region_context_does_not_publish_layer_shape_as_kernel_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            table = {"tables": [{
                "phase": "decode", "pattern_id": "P",
                "representative_layer_id": 2,
                "selected_bucket": {"batch_size": 4, "input_tokens": 0},
                "rows": [{
                    "pos": 0, "row_id": "event-1", "raw_event_index": 1,
                    "device_seq_index": 1, "raw_name": "copy",
                    "short_name": "copy", "stage": "elementwise",
                    "duration_us": 2.0,
                    "shape": {"source": "unresolved", "input_dims": []},
                    "parent_operator": {"canonical_op": "unresolved"},
                }]}]}
            plan = {"capture_targets": [{
                "row_id": "event-1",
                "candidate_op_instance_id": "layer-2",
                "candidate_op_path": "model.layers.2.self_attn.v_absorb",
                "mapping_cardinality": "1:N",
                "shape_log_layer_evidence": {
                    "scope": "phase_layer_semantic_region",
                    "semantic_region": "v_absorb"},
            }]}
            records = [{
                "phase": "decode", "rank": 0, "layer_id": 2,
                "batch_size": 4, "input_tokens": 4,
                "op_instance_id": "layer-2", "op_name": "2",
                "op_type": "DecoderLayer", "op_path": "model.layers.2",
                "io": "input", "tensor_path": "args[0]",
                "shape": [4, 7168], "dtype": "bf16",
            }]
            result = merge.merge(
                self._write(tmp, "table.json", table),
                self._write(tmp, "plan.json", plan),
                self._write(tmp, "shape.jsonl", records, jsonl=True),
                os.path.join(tmp, "out"))
            with open(result["semantic_table_json"]) as fh:
                row = json.load(fh)["tables"][0]["rows"][0]
            self.assertEqual(
                row["parent_operator"]["canonical_op"],
                "model.layers.2.self_attn.v_absorb")
            self.assertEqual(row["shape"]["input_dims"], [])
            self.assertEqual(
                row["shape"]["source"], "runtime_probe_region_context")

    def test_unmatched_memcpy_has_specific_reason_code(self):
        reason_code, _ = merge._unavailable_reason(
            {"event_type": "gpu_memcpy", "short_name": "Memcpy"},
            {"runtime_marker_mapping_status": "not_found"}, 0)
        self.assertEqual(
            reason_code, "runtime_copy_without_unique_tensor")


class PhaseResolutionGateTest(unittest.TestCase):
    """B8: a phase the shape log covers must resolve at least one table row.

    The eager probe makes the shape log carry decode records, but the merge
    attaches them by `parent_operator`.  A table built from a CUDA-graph
    DECODE trace has `parent_operator: "unresolved"` on every row, so every
    decode row lands at evidence level U while the merge still reports pass.
    """

    def _audit(self, phase, level):
        return {"phase": phase, "evidence": {"level": level}}

    def _groups(self, phases):
        return [{"rank": 0, "phase": phase, "layer": 3} for phase in phases]

    def test_b8_covered_phase_that_resolves_nothing_fails_the_merge(self):
        result = merge._phase_resolution(
            [self._audit("decode", "U"), self._audit("decode", "U"),
             self._audit("prefill", "A"), self._audit("prefill", "U")],
            self._groups(["decode", "prefill"]))
        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["gating"])
        by_phase = {row["phase"]: row for row in result["phases"]}
        self.assertEqual(by_phase["decode"]["status"], "fail")
        self.assertEqual(by_phase["decode"]["rows"], 2)
        self.assertEqual(by_phase["decode"]["resolved"], 0)
        self.assertIn("CUDA-graph", by_phase["decode"]["note"])
        self.assertEqual(by_phase["prefill"]["status"], "pass")
        self.assertEqual(by_phase["prefill"]["resolved_fraction"], 0.5)

    def test_b8_partial_resolution_passes(self):
        result = merge._phase_resolution(
            [self._audit("decode", "U"), self._audit("decode", "B"),
             self._audit("prefill", "A")],
            self._groups(["decode", "prefill"]))
        self.assertEqual(result["status"], "pass")
        by_phase = {row["phase"]: row for row in result["phases"]}
        self.assertEqual(by_phase["decode"]["status"], "pass")
        self.assertEqual(by_phase["decode"]["resolved"], 1)
        self.assertIsNone(by_phase["decode"]["note"])

    def test_b8_phase_absent_from_the_shape_log_is_not_gated(self):
        """A phase the probe never covered has no records to attach; that is
        a coverage gap the capture reports, not a merge failure."""
        result = merge._phase_resolution(
            [self._audit("decode", "U"), self._audit("prefill", "A")],
            self._groups(["prefill"]))
        self.assertEqual(result["status"], "pass")
        by_phase = {row["phase"]: row for row in result["phases"]}
        self.assertFalse(by_phase["decode"]["shape_log_covers_phase"])
        self.assertEqual(by_phase["decode"]["status"], "pass")
        self.assertEqual(result["shape_log_phases"], ["prefill"])

    def test_b8_only_rank_zero_shape_records_define_covered_phases(self):
        """The shape log is keyed to the probed rank; a non-rank-0 group must
        not make the gate claim coverage the merge cannot use."""
        result = merge._phase_resolution(
            [self._audit("decode", "U")],
            [{"rank": 3, "phase": "decode", "layer": 3}])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["shape_log_phases"], [])


if __name__ == "__main__":
    unittest.main()
