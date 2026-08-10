import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import structural_pattern_mapping as mapping


class StructuralPatternMappingTest(unittest.TestCase):
    def _write(self, cfg):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, "config.json")
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        return tmp, path

    def _runtime(self, tmp, value):
        path = os.path.join(tmp.name, "runtime.json")
        with open(path, "w") as fh:
            json.dump(value, fh)
        return path

    def test_dense_moe_formula_dialect_and_mtp_exclusion(self):
        tmp, path = self._write({
            "model_type": "unregistered_mla_model",
            "num_hidden_layers": 6,
            "first_k_dense_replace": 2,
            "moe_layer_freq": 1,
            "q_lora_rank": 16,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 2,
            "intermediate_size": 100,
            "moe_intermediate_size": 20,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "topk_method": "noaux_tc",
            "num_nextn_predict_layers": 1,
        })
        self.addCleanup(tmp.cleanup)
        result = mapping.build(path)
        self.assertEqual(result["patterns"][0]["layer_ids"], [0, 1])
        self.assertEqual(result["patterns"][1]["layer_ids"], [2, 3, 4, 5])
        self.assertEqual(result["excluded_layers_note"]["num_nextn_predict_layers"], 1)
        self.assertTrue(result["coverage_check"]["full_coverage"])
        self.assertEqual(result["pattern_dialect"], "dense_moe_formula")
        self.assertEqual(result["quality"]["status"], "partial")

    def test_formula_dialect_alternating_moe_frequency_keeps_dense_gaps(self):
        tmp, path = self._write({
            "model_type": "another_unregistered_model",
            "num_hidden_layers": 7,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 2,
            "kv_lora_rank": 8,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
        })
        self.addCleanup(tmp.cleanup)
        result = mapping.build(path)
        by_id = {item["pattern_id"]: item for item in result["patterns"]}
        self.assertEqual(by_id["P_MLA_MOE"]["layer_ids"], [2, 4, 6])
        self.assertEqual(by_id["P_MLA_DENSE"]["layer_ids"], [0, 1, 3, 5])

    def test_per_layer_list_dialect_is_mutually_exclusive(self):
        tmp, path = self._write({
            "model_type": "unregistered_hybrid_model",
            "text_config": {
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "full_attention",
                                "linear_attention", "full_attention"],
                "num_experts": 16,
                "num_experts_per_tok": 4,
                "mtp_num_hidden_layers": 1,
            },
        })
        self.addCleanup(tmp.cleanup)
        result = mapping.build(path)
        self.assertEqual([item["layer_ids"] for item in result["patterns"]],
                         [[0, 2], [1, 3]])
        self.assertTrue(result["coverage_check"]["mutually_exclusive"])
        self.assertEqual(result["pattern_dialect"], "per_layer_list")

    def test_any_model_rejects_incomplete_layer_types(self):
        tmp, path = self._write({
            "model_type": "unregistered_model",
            "text_config": {"num_hidden_layers": 2, "layer_types": ["linear_attention"]},
        })
        self.addCleanup(tmp.cleanup)
        with self.assertRaises(ValueError):
            mapping.build(path)

    def test_qwen_like_linear_full_context_and_tp_local_safety(self):
        tmp, path = self._write({
            "model_type": "unregistered_hybrid",
            "text_config": {
                "num_hidden_layers": 4,
                "layer_types": [
                    "linear_attention", "full_attention",
                    "linear_attention", "full_attention"],
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 64,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 1024,
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                },
            },
        })
        self.addCleanup(tmp.cleanup)
        runtime = self._runtime(tmp, {"server_args": {
            "tensor_parallel_size": 8,
            "expert_parallel_size": 8,
        }})
        result = mapping.build(path, runtime_sources=[runtime])
        self.assertEqual(len(result["patterns"]), 2)
        context = result["patterns"][0]["structural_context"]
        static = context["static_model_context"]
        self.assertEqual(static["linear_attention"]["key_heads"]["value"], 16)
        self.assertEqual(
            static["linear_attention"]["key_heads"]["evidence"]["config_path"],
            "text_config.linear_num_key_heads")
        self.assertEqual(static["full_attention"]["num_attention_heads"]["value"], 32)
        self.assertEqual(
            static["quantization"]["weight_block_size"]["value"], [128, 128])
        local = context["runtime_context"]["rank_local_derivations"]
        self.assertEqual(local["attention_heads"]["value"], 4)
        self.assertEqual(local["linear_key_heads"]["value"], 2)
        self.assertEqual(local["linear_value_heads"]["value"], 8)
        self.assertEqual(
            local["key_value_heads"]["status"],
            "replicated_policy_required")
        self.assertNotIn("value", local["key_value_heads"])

    def test_dsr_like_mla_dense_and_moe_context(self):
        tmp, path = self._write({
            "model_type": "unregistered_mla",
            "num_hidden_layers": 5,
            "first_k_dense_replace": 2,
            "moe_layer_freq": 1,
            "hidden_size": 7168,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "intermediate_size": 18432,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "scoring_func": "sigmoid",
        })
        self.addCleanup(tmp.cleanup)
        result = mapping.build(path)
        by_id = {item["pattern_id"]: item for item in result["patterns"]}
        dense = by_id["P_MLA_DENSE"]["structural_context"]
        moe = by_id["P_MLA_MOE"]["structural_context"]
        self.assertEqual(
            dense["static_model_context"]["dense_ffn"]["intermediate_size"]["value"],
            18432)
        self.assertEqual(
            moe["static_model_context"]["moe"]["num_experts"]["value"], 256)
        self.assertEqual(
            moe["static_model_context"]["moe"]["experts_per_token"]["value"], 8)
        self.assertEqual(
            moe["static_model_context"]["mla"]["kv_lora_rank"]["value"], 512)
        self.assertEqual(
            moe["static_model_context"]["mla"]["kv_lora_rank"]
            ["evidence"]["config_field"], "kv_lora_rank")
        self.assertEqual(
            moe["runtime_context"]["rank_local_derivations"]["experts"]["status"],
            "unresolved")

    def test_runtime_cli_parallelism_evidence_is_not_static(self):
        tmp, path = self._write({
            "num_hidden_layers": 1,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 4096,
        })
        self.addCleanup(tmp.cleanup)
        runtime = os.path.join(tmp.name, "server.log")
        with open(runtime, "w") as fh:
            fh.write("launch --tp-size 4 --ep-size 2\n")
        context = mapping.build(
            path, runtime_sources=[runtime])["patterns"][0]["structural_context"]
        self.assertNotIn(
            "tensor_parallel_size",
            context["static_model_context"]["parallelism"])
        self.assertEqual(
            context["runtime_context"]["tensor_parallel_size"]["value"], 4)
        self.assertEqual(
            context["runtime_context"]["rank_local_derivations"]
            ["key_value_heads"]["value"], 2)


if __name__ == "__main__":
    unittest.main()
