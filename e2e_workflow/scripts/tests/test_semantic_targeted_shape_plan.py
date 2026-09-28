import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_targeted_shape_plan as planner


class SemanticTargetedShapePlanTest(unittest.TestCase):
    def _inputs(self, root):
        source = os.path.join(root, "runtime.py")
        with open(source, "w") as fh:
            fh.write("def launch_allreduce(x):\n    return x\n")
        table_path = os.path.join(root, "table.json")
        rows = [
            {
                "row_id": "u-op", "pos": 0, "raw_name": "kernel_alpha",
                "event_type": "kernel", "semantic_evidence": {
                    "level": "U", "reason_code": "kernel_not_observed_in_probe"},
            },
            {
                "row_id": "u-comm", "pos": 1,
                "raw_name": "allreduce_fusion_kernel", "event_type": "kernel",
                "stage": "communication", "semantic_evidence": {
                    "level": "U", "reason_code": "kernel_not_observed_in_probe"},
            },
            {
                "row_id": "u-copy", "pos": 2, "raw_name": "Memcpy",
                "event_type": "gpu_memcpy", "semantic_evidence": {
                    "level": "U",
                    "reason_code": "runtime_copy_without_unique_tensor"},
            },
            {
                "row_id": "p-op", "pos": 3, "raw_name": "kernel_beta",
                "event_type": "kernel", "semantic_evidence": {"level": "P"},
            },
        ]
        with open(table_path, "w") as fh:
            json.dump({"tables": [{
                "phase": "decode", "pattern_id": "P0",
                "representative_layer_id": 3, "rows": rows,
            }]}, fh)
        plan_path = os.path.join(root, "mapped.json")
        with open(plan_path, "w") as fh:
            json.dump({
                "capture_targets": [{
                    **row, "phase": "decode", "pattern_id": "P0",
                    "representative_layer_id": 3,
                    "runtime_marker_mapping_status": "not_found",
                } for row in rows],
                "target_buckets": [{
                    "phase": "decode", "pattern_id": "P0",
                    "representative_layer_id": 3,
                }],
                "patterns": [{
                    "pattern_id": "P0", "representative_layer_id": 3,
                }],
                "phase_coverage": {"required_phases": ["decode", "prefill"]},
            }, fh)
        agent_path = os.path.join(root, "agent.json")
        evidence = {
            "path": source, "line_start": 1, "line_end": 2,
            "symbol": "launch_allreduce", "claim": "reviewed call path",
        }
        with open(agent_path, "w") as fh:
            json.dump({
                "producer": "semantics_mapper_agent",
                "mapping_claim": "source_call_path_candidates",
                "callable_kernel_map": [{
                    "target": "runtime:launch_allreduce",
                    "kernel_pattern": "kernel_alpha",
                    "scope": "kernel", "source_evidence": [evidence],
                }, {
                    "target": "runtime:launch_allreduce",
                    "kernel_pattern": "allreduce_fusion",
                    "scope": "wrapper", "source_evidence": [evidence],
                }],
            }, fh)
        setup_path = os.path.join(root, "setup.json")
        with open(setup_path, "w") as fh:
            json.dump({"capture_phases": ["prefill", "decode"]}, fh)
        return source, table_path, plan_path, agent_path, setup_path

    def test_selects_only_source_backed_unresolved_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, table, mapped, agent, setup = self._inputs(tmp)
            out_plan = os.path.join(tmp, "targeted.json")
            out_setup = os.path.join(tmp, "targeted_setup.json")
            result = planner.build(
                table, mapped, agent, setup, [source], out_plan, out_setup)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["targeted_row_count"], 2)
            self.assertEqual(result["capture_phases"], ["decode"])
            with open(out_plan) as fh:
                planned = json.load(fh)
            self.assertEqual(
                [row["row_id"] for row in planned["capture_targets"]],
                ["u-op", "u-comm"])
            self.assertNotIn(
                "runtime_marker_mapping_status", planned["capture_targets"][0])
            with open(out_setup) as fh:
                targeted_setup = json.load(fh)
            self.assertEqual(targeted_setup["capture_phases"], ["decode"])
            self.assertEqual(len(targeted_setup["callable_kernel_map"]), 2)
            self.assertEqual(
                result["permanently_ineligible"][0]["row_id"], "u-copy")

    def test_rejects_overlapping_agent_mappings(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, table, mapped, agent, setup = self._inputs(tmp)
            with open(agent) as fh:
                document = json.load(fh)
            duplicate = dict(document["callable_kernel_map"][0])
            duplicate["target"] = "runtime:other"
            document["callable_kernel_map"].append(duplicate)
            with open(agent, "w") as fh:
                json.dump(document, fh)
            with self.assertRaisesRegex(ValueError, "multiple targeted"):
                planner.build(
                    table, mapped, agent, setup, [source],
                    os.path.join(tmp, "out.json"),
                    os.path.join(tmp, "out_setup.json"))

    def test_rejects_unapproved_source_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, table, mapped, agent, setup = self._inputs(tmp)
            with open(agent) as fh:
                document = json.load(fh)
            document["callable_kernel_map"][0]["source_evidence"][0][
                "path"] = os.path.join(tmp, "not-approved.py")
            with open(agent, "w") as fh:
                json.dump(document, fh)
            with self.assertRaisesRegex(ValueError, "unapproved runtime source"):
                planner.build(
                    table, mapped, agent, setup, [source],
                    os.path.join(tmp, "out.json"),
                    os.path.join(tmp, "out_setup.json"))


if __name__ == "__main__":
    unittest.main()
