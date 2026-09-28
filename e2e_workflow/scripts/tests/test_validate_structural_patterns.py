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
                "num_hidden_layers": 4,
                "layer_kinds": ["linear", "linear", "full", "full"],
            }, fh)
        with open(self.source, "w") as fh:
            fh.write(
                "class LinearLayer: pass\n"
                "class FullLayer: pass\n"
                "def build(kind): return LinearLayer if kind == 'linear' else FullLayer\n")

    def _layer(self, layer_id, kind, context=None, signature_extra=None):
        signature = {
            "layer_implementation": "%sLayer" % kind.title(),
            "body_dispatch": {
                "config_value": kind,
                "implementation": "%sLayer" % kind.title(),
            },
            "semantic_labels": {"attention_type": kind, "ffn_type": "moe"},
            "shape_parameters": {"hidden_size": 16},
        }
        signature.update(signature_extra or {})
        return {
            "layer_id": layer_id,
            "body_display_name": "%s / MoE" % kind.title(),
            "body_signature": signature,
            "instance_context": context or {},
            "config_evidence": [{
                "config_path": "layer_kinds[%d]" % layer_id,
                "value": kind,
                "claim": "per-layer body kind",
            }],
            "source_evidence": [{
                "path": self.source,
                "line_start": 1,
                "line_end": 3,
                "symbol": "build",
                "claim": "runtime selects the layer implementation",
            }],
        }

    def _write(self, layers=None, definition=None):
        document = {
            "schema_version": 3,
            "pattern_definition": definition or {
                "producer": "semantics_mapper_agent",
                "method": "config_runtime_body_analysis",
                "trace_used_for_definition": False,
                "analysis_summary": (
                    "Expanded config layer kinds through runtime construction."),
            },
            "main_layer_scope": {
                "config_path": "",
                "num_hidden_layers": 4,
                "excluded_stacks": [],
            },
            "layers": layers or [
                self._layer(0, "linear", {"is_first_main_layer": True}),
                self._layer(1, "linear"),
                self._layer(2, "full"),
                self._layer(3, "full", {
                    "is_last_main_layer": True,
                    "model_epilogue": True,
                }),
            ],
        }
        with open(self.patterns, "w") as fh:
            json.dump(document, fh)

    def test_validator_derives_patterns_from_body_not_context(self):
        self._write()
        result = validator.validate(
            self.patterns, self.config, [self.source])
        self.assertEqual(result["schema_version"], 3)
        self.assertTrue(result["coverage_check"]["full_coverage"])
        self.assertEqual(len(result["patterns"]), 2)
        self.assertEqual(result["patterns"][0]["layer_ids"], [0, 1])
        self.assertEqual(result["patterns"][1]["layer_ids"], [2, 3])
        self.assertEqual(result["patterns"][0]["representative_candidates"], [1])
        self.assertEqual(result["patterns"][1]["representative_candidates"], [2])
        self.assertTrue(result["layer_contexts"]["3"]["model_epilogue"])
        self.assertTrue(result["validation"]["patterns_derived_deterministically"])

    def test_final_layer_with_same_body_never_creates_a_pattern(self):
        layers = [
            self._layer(0, "linear"),
            self._layer(1, "linear"),
            self._layer(2, "full"),
            self._layer(3, "full", {
                "is_last_main_layer": True,
                "exit_handoff": "standalone_collective",
            }),
        ]
        self._write(layers=layers)
        result = validator.validate(
            self.patterns, self.config, [self.source])
        self.assertEqual(len(result["patterns"]), 2)
        self.assertEqual(result["patterns"][1]["layer_ids"], [2, 3])

    def test_rejects_context_field_inside_body_signature(self):
        layers = [
            self._layer(0, "linear"),
            self._layer(1, "linear"),
            self._layer(2, "full"),
            self._layer(3, "full", signature_extra={"is_last_layer": True}),
        ]
        self._write(layers=layers)
        with self.assertRaisesRegex(ValueError, "instance context"):
            validator.validate(self.patterns, self.config, [self.source])

    def test_rejects_non_agent_producer(self):
        self._write(definition={
            "producer": "fixed_dialect_script",
            "method": "config_runtime_body_analysis",
            "trace_used_for_definition": False,
            "analysis_summary": "not an Agent",
        })
        with self.assertRaises(ValueError):
            validator.validate(self.patterns, self.config, [self.source])

    def test_rejects_trace_defined_pattern(self):
        self._write(definition={
            "producer": "semantics_mapper_agent",
            "method": "config_runtime_body_analysis",
            "trace_used_for_definition": True,
            "analysis_summary": "clustered kernels",
        })
        with self.assertRaises(ValueError):
            validator.validate(self.patterns, self.config, [self.source])

    def test_rejects_trace_or_kernel_evidence_hidden_in_layer(self):
        layers = [
            self._layer(0, "linear"), self._layer(1, "linear"),
            self._layer(2, "full"), self._layer(3, "full"),
        ]
        layers[0]["trace_evidence"] = {"kernel_sequence": ["a", "b"]}
        self._write(layers=layers)
        with self.assertRaisesRegex(ValueError, "derived Pattern definition"):
            validator.validate(self.patterns, self.config, [self.source])

    def test_rejects_incomplete_layer_coverage(self):
        self._write(layers=[
            self._layer(0, "linear"),
            self._layer(1, "linear"),
            self._layer(2, "full"),
        ])
        with self.assertRaisesRegex(ValueError, "cover every main layer"):
            validator.validate(self.patterns, self.config, [self.source])

    def test_real_body_difference_still_splits_a_singleton(self):
        layers = [
            self._layer(0, "linear", signature_extra={
                "feed_forward": {"implementation": "DenseMLP"}}),
            self._layer(1, "linear"),
            self._layer(2, "full"),
            self._layer(3, "full"),
        ]
        self._write(layers=layers)
        result = validator.validate(
            self.patterns, self.config, [self.source])
        self.assertEqual(len(result["patterns"]), 3)
        self.assertEqual(result["patterns"][0]["layer_ids"], [0])

    def test_explicit_sixty_layer_hybrid_config_groups_45_and_15(self):
        layer_types = [
            "full" if (layer_id + 1) % 4 == 0 else "linear"
            for layer_id in range(60)]
        with open(self.config, "w") as fh:
            json.dump({
                "model_type": "future_hybrid",
                "text_config": {
                    "num_hidden_layers": 60,
                    "layer_types": layer_types,
                },
            }, fh)
        layers = []
        for layer_id, kind in enumerate(layer_types):
            item = self._layer(layer_id, kind)
            item["config_evidence"] = [{
                "config_path": "text_config.layer_types[%d]" % layer_id,
                "value": kind,
                "claim": "per-layer runtime construction selector",
            }]
            if layer_id == 59:
                item["instance_context"] = {
                    "is_last_main_layer": True,
                    "model_epilogue": True,
                    "exit_handoff": "runtime-specific",
                }
            layers.append(item)
        self._write(layers=layers)
        with open(self.patterns) as fh:
            document = json.load(fh)
        document["main_layer_scope"]["num_hidden_layers"] = 60
        with open(self.patterns, "w") as fh:
            json.dump(document, fh)
        result = validator.validate(
            self.patterns, self.config, [self.source])
        by_label = {
            pattern["attention_type"]: pattern
            for pattern in result["patterns"]}
        self.assertEqual(len(result["patterns"]), 2)
        self.assertEqual(by_label["linear"]["layer_count"], 45)
        self.assertEqual(by_label["full"]["layer_count"], 15)
        self.assertIn(59, by_label["full"]["layer_ids"])
        self.assertNotIn(59, by_label["full"]["representative_candidates"])


if __name__ == "__main__":
    unittest.main()
