import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_source_mapping as source_mapping


class SemanticSourceMappingTest(unittest.TestCase):
    def test_demangled_symbol_finds_enclosing_runtime_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "runtime.py")
            with open(source, "w") as fh:
                fh.write(
                    "def attention_wrapper(x):\n"
                    "    return add_rmsnorm_quant(x)\n")
            plan_path = os.path.join(tmp, "plan.json")
            with open(plan_path, "w") as fh:
                json.dump({"capture_targets": [{
                    "row_id": "event-1",
                    "raw_name": (
                        "_ZN5aiter24add_rmsnorm_quant_kernelIDF16bEEEv"),
                }]}, fh)
            out = os.path.join(tmp, "mapped.json")
            summary = source_mapping.map_plan(
                plan_path, [source], out)
            self.assertEqual(summary["with_source_candidate"], 1)
            self.assertEqual(summary["unique_wrapper_candidates"], 1)
            with open(out) as fh:
                target = json.load(fh)["capture_targets"][0]
            self.assertEqual(
                target["candidate_terminal_launcher"],
                "add_rmsnorm_quant")
            self.assertTrue(
                target["candidate_wrapper"].endswith(
                    "runtime.py:attention_wrapper"))
            self.assertEqual(
                target["mapping_cardinality"], "probe_required")

    def test_no_source_match_remains_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "runtime.py")
            with open(source, "w") as fh:
                fh.write("def unrelated(x):\n    return x\n")
            plan = os.path.join(tmp, "plan.json")
            with open(plan, "w") as fh:
                json.dump({"capture_targets": [{
                    "row_id": "event-1", "raw_name": "opaque_binary"}]}, fh)
            out = os.path.join(tmp, "mapped.json")
            summary = source_mapping.map_plan(plan, [source], out)
            self.assertEqual(summary["with_source_candidate"], 0)
            with open(out) as fh:
                target = json.load(fh)["capture_targets"][0]
            self.assertEqual(target["source_mapping_status"], "not_found")


if __name__ == "__main__":
    unittest.main()
