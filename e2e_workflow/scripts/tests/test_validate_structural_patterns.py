import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import validate_structural_patterns as validator


class ValidateStructuralPatternsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = os.path.join(self.tmp.name, "config.json")
        self.source = os.path.join(self.tmp.name, "model.py")
        self.patterns = os.path.join(self.tmp.name, "patterns.json")
        with open(self.config, "w") as fh:
            json.dump({
                "model_type": "future_model",
                "num_hidden_layers": 3,
                "layer_kinds": ["a", "b", "a"],
            }, fh)
        with open(self.source, "w") as fh:
            fh.write("class LayerA: pass\nclass LayerB: pass\n")

    def _signature(self, attention):
        return {
            "attention_type": attention,
            "model_native_attention_name": attention,
            "attention_config_fields": {"kind": attention},
            "runtime_attention_module_class": "Layer%s" % attention.upper(),
            "ffn_type": "dense",
            "is_moe": False,
            "num_experts": None,
            "topk": None,
            "shared_expert": False,
            "router_family": "none",
            "special_layer_role": "none",
            "runtime_dispatch_branch": "layer_kinds[layer_id]",
        }

    def _pattern(self, pattern_id, attention, layer_ids):
        return {
            "pattern_id": pattern_id,
            "pattern_display_name": "%s / Dense" % attention,
            "structural_signature": self._signature(attention),
            "layer_ids": layer_ids,
            "representative_candidates": layer_ids,
            "config_evidence": [{
                "config_path": "layer_kinds",
                "value": ["a", "b", "a"],
                "claim": "per-layer structural kind",
            }],
            "source_evidence": [{
                "path": self.source,
                "line_start": 1,
                "line_end": 2,
                "symbol": "LayerA/LayerB",
                "claim": "runtime classes implement both branches",
            }],
        }

    def _write(self, patterns=None, definition=None):
        document = {
            "pattern_definition": definition or {
                "producer": "semantics_mapper_agent",
                "method": "config_runtime_source_analysis",
                "trace_used_for_definition": False,
                "analysis_summary": "Agent joined config kinds to runtime classes.",
            },
            "patterns": patterns or [
                self._pattern("P_A", "a", [0, 2]),
                self._pattern("P_B", "b", [1]),
            ],
        }
        with open(self.patterns, "w") as fh:
            json.dump(document, fh)

    def test_agent_patterns_are_validated_not_generated(self):
        self._write()
        result = validator.validate(
            self.patterns, self.config, [self.source])
        self.assertEqual(result["schema_version"], 2)
        self.assertTrue(result["coverage_check"]["full_coverage"])
        self.assertEqual(
            [item["pattern_id"] for item in result["patterns"]],
            ["P_A", "P_B"])
        self.assertTrue(result["validation"]["definition_preserved"])

    def test_rejects_non_agent_producer(self):
        self._write(definition={
            "producer": "fixed_dialect_script",
            "method": "config_runtime_source_analysis",
            "trace_used_for_definition": False,
            "analysis_summary": "not an Agent",
        })
        with self.assertRaises(ValueError):
            validator.validate(
                self.patterns, self.config, [self.source])

    def test_rejects_trace_defined_pattern(self):
        self._write(definition={
            "producer": "semantics_mapper_agent",
            "method": "config_runtime_source_analysis",
            "trace_used_for_definition": True,
            "analysis_summary": "clustered kernels",
        })
        with self.assertRaises(ValueError):
            validator.validate(
                self.patterns, self.config, [self.source])

    def test_rejects_trace_or_kernel_evidence_hidden_in_pattern(self):
        pattern_a = self._pattern("P_A", "a", [0, 2])
        pattern_a["trace_evidence"] = {
            "kernel_sequence": ["kernel_a", "kernel_b"],
        }
        self._write(patterns=[
            pattern_a,
            self._pattern("P_B", "b", [1]),
        ])
        with self.assertRaisesRegex(
                ValueError, "derived Pattern definition"):
            validator.validate(
                self.patterns, self.config, [self.source])

    def test_rejects_overlap_or_incomplete_coverage(self):
        self._write(patterns=[
            self._pattern("P_A", "a", [0, 1]),
            self._pattern("P_B", "b", [1]),
        ])
        with self.assertRaises(ValueError):
            validator.validate(
                self.patterns, self.config, [self.source])

    def test_rejects_duplicate_signatures_instead_of_splitting(self):
        first = self._pattern("P_A1", "a", [0])
        second = self._pattern("P_A2", "a", [1, 2])
        self._write(patterns=[first, second])
        with self.assertRaises(ValueError):
            validator.validate(
                self.patterns, self.config, [self.source])


if __name__ == "__main__":
    unittest.main()
