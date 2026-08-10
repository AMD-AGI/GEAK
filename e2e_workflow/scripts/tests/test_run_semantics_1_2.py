import json
import os
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import run_semantics_1_2 as runner


class RunSemantics12Test(unittest.TestCase):
    def test_orchestrates_strict_geak_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config.json")
            trace = os.path.join(tmp, "trace.json")
            shape_log = os.path.join(tmp, "shape.log")
            patterns = os.path.join(tmp, "agent_patterns.json")
            for path, value in (
                    (config, "{}"), (trace, "{}"), (shape_log, "shape\n"),
                    (patterns, '{"pattern_definition": {}}')):
                with open(path, "w") as fh:
                    fh.write(value)
            table = os.path.join(tmp, "table.json")
            table_md = os.path.join(tmp, "table.md")
            plan = os.path.join(tmp, "plan.json")
            with open(table, "w") as fh:
                json.dump({"tables": []}, fh)
            with open(table_md, "w") as fh:
                fh.write("# phase 1.1\n")
            with open(plan, "w") as fh:
                json.dump({"capture_targets": []}, fh)
            semantic = {
                "status": "pass",
                "semantic_table_json": table,
                "semantic_table_md": table_md,
                "shape_capture_plan_json": plan,
            }
            merged_json = os.path.join(tmp, "merged.json")
            merged_md = os.path.join(tmp, "merged.md")
            with open(merged_json, "w") as fh:
                json.dump({"tables": []}, fh)
            with open(merged_md, "w") as fh:
                fh.write("# merged\n")
            merged = {
                "status": "pass",
                "semantic_table_json": merged_json,
                "semantic_table_md": merged_md,
            }
            with mock.patch.object(
                    runner.validate_structural_patterns, "validate",
                    return_value={
                        "patterns": [],
                        "validation": {"definition_preserved": True},
                    }), mock.patch.object(
                        runner.semantic_kernel_mapping, "build",
                        return_value=semantic), mock.patch.object(
                            runner.semantic_source_mapping, "map_plan",
                            return_value={}), mock.patch.object(
                                runner.semantic_shape_merge, "merge",
                                return_value=merged), mock.patch.object(
                                    runner.semantic_evidence_ledger, "merge",
                                    return_value=merged):
                result = runner.run(
                    config, trace, shape_log, os.path.join(tmp, "out"),
                    structural_patterns_path=patterns)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                result["evidence_policy"]["levels"], ["K", "P", "U"])
            self.assertTrue(result["evidence_policy"][
                "additive_across_probe_runs"])
            self.assertTrue(os.path.exists(result["result_json"]))
            self.assertTrue(os.path.exists(
                result["published_semantic_table_md"]))
            self.assertTrue(result["structural_pattern_validation"][
                "definition_preserved"])

    def test_rejects_missing_agent_structural_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                    ValueError, "semantics_mapper Agent"):
                runner.run(
                    os.path.join(tmp, "config.json"),
                    os.path.join(tmp, "trace.json"),
                    os.path.join(tmp, "shape.log"),
                    os.path.join(tmp, "out"))


if __name__ == "__main__":
    unittest.main()
