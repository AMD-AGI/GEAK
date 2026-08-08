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


if __name__ == "__main__":
    unittest.main()
