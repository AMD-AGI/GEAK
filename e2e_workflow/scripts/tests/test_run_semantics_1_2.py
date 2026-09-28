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
    def test_targeted_retry_is_shape_only_and_not_a_boundary_donor(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config.json")
            trace = os.path.join(tmp, "clean.trace.json")
            patterns = os.path.join(tmp, "agent_patterns.json")
            setup = os.path.join(tmp, "capture_setup.json")
            targeted_agent = os.path.join(tmp, "targeted.agent.json")
            for path, value in (
                    (config, "{}"), (trace, "{}"),
                    (patterns, '{"pattern_definition": {}}'),
                    (setup, "{}"), (targeted_agent, "{}")):
                with open(path, "w") as fh:
                    fh.write(value)

            table = os.path.join(tmp, "table.json")
            table_md = os.path.join(tmp, "table.md")
            plan = os.path.join(tmp, "plan.json")
            audit = os.path.join(tmp, "layer_instance_audit.json")
            with open(table, "w") as fh:
                json.dump({"tables": [{
                    "phase": "decode", "pattern_id": "P0",
                    "representative_layer_id": 3,
                    "rows": [{
                        "pos": 0, "row_id": "event-1",
                        "raw_name": "kernel", "duration_us": 1.0,
                        "layer_evidence": (
                            "validated_graph_capture_layer_scope_transfer"),
                    }],
                }]}, fh)
            with open(table_md, "w") as fh:
                fh.write("# semantic\n")
            with open(plan, "w") as fh:
                json.dump({"capture_targets": []}, fh)
            with open(audit, "w") as fh:
                json.dump({"module_scope_count": 0}, fh)
            semantic = {
                "status": "pass",
                "semantic_table_json": table,
                "semantic_table_md": table_md,
                "shape_capture_plan_json": plan,
                "layer_instance_audit_json": audit,
            }
            merged = {
                "status": "pass",
                "semantic_table_json": table,
                "semantic_table_md": table_md,
            }
            primary_trace = os.path.join(tmp, "primary.trace.json")
            targeted_trace = os.path.join(tmp, "targeted.trace.json")
            primary_capture = {
                "shape_capture_execution": "graph_capture",
                "capture_phases": ["decode"],
                "capture_trace": primary_trace,
                "shape_log": os.path.join(tmp, "primary.shape.jsonl"),
            }
            targeted_capture = {
                "shape_capture_execution": "graph_capture",
                "capture_phases": ["prefill"],
                "capture_trace": targeted_trace,
                "shape_log": os.path.join(tmp, "targeted.shape.jsonl"),
            }
            marker_result = {
                "phase_coverage_complete": True,
                "shape_mapping_by_phase": {},
                "clean_table_sequence_audit": {
                    "status": "pass", "groups": []},
            }
            transfer_result = {
                "status": "pass",
                "donor": {"path": primary_trace},
                "mapped_groups": [{"phase": "decode"}],
                "residual_range_count": 0,
                "failures": [],
            }
            with mock.patch.object(
                    runner.validate_structural_patterns, "validate",
                    return_value={"patterns": [], "validation": {}}), \
                    mock.patch.object(
                        runner.semantic_kernel_mapping, "build",
                        side_effect=lambda *args, **kwargs: dict(semantic)), \
                    mock.patch.object(
                        runner.semantic_source_mapping, "map_plan",
                        return_value={}), \
                    mock.patch.object(
                        runner.run_semantic_shape_capture, "capture",
                        side_effect=[primary_capture, targeted_capture]
                    ) as capture_call, \
                    mock.patch.object(
                        runner.semantic_layer_boundary_transfer, "transfer",
                        return_value=transfer_result) as transfer_call, \
                    mock.patch.object(
                        runner.semantic_runtime_marker_mapping, "map_plan",
                        return_value=marker_result), \
                    mock.patch.object(
                        runner.semantic_shape_merge, "merge",
                        return_value=merged), \
                    mock.patch.object(
                        runner.semantic_evidence_ledger, "merge",
                        return_value=merged), \
                    mock.patch.object(
                        runner.semantic_targeted_shape_plan, "build",
                        return_value={
                            "status": "ready",
                            "capture_phases": ["prefill"],
                            "targeted_row_count": 1,
                        }):
                result = runner.run(
                    config, trace, "", os.path.join(tmp, "out"),
                    capture_setup_path=setup,
                    structural_patterns_path=patterns,
                    require_phases=["decode"],
                    targeted_probe_plan_path=targeted_agent)

            self.assertEqual(capture_call.call_count, 2)
            self.assertEqual(transfer_call.call_count, 1)
            self.assertEqual(
                result["targeted_shape_retry"]["status"], "completed")
            self.assertEqual(
                [run["kind"] for run in result["probe_runs"]],
                ["runtime_capture", "targeted_runtime_capture"])
            self.assertEqual(
                result["boundary_rebuild"]["traces"],
                [os.path.abspath(primary_trace)])

    def test_graph_capture_is_wired_after_clean_table_and_replaces_eager_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config.json")
            trace = os.path.join(tmp, "clean.trace.json")
            patterns = os.path.join(tmp, "agent_patterns.json")
            setup = os.path.join(tmp, "capture_setup.json")
            for path, value in (
                    (config, "{}"), (trace, "{}"),
                    (patterns, '{"pattern_definition": {}}'),
                    (setup, "{}")):
                with open(path, "w") as fh:
                    fh.write(value)
            table = os.path.join(tmp, "table.json")
            table_md = os.path.join(tmp, "table.md")
            plan = os.path.join(tmp, "plan.json")
            audit = os.path.join(tmp, "layer_instance_audit.json")
            with open(table, "w") as fh:
                json.dump({"tables": [{
                    "phase": "decode", "pattern_id": "P",
                    "representative_layer_id": 3,
                    "event_count": 1, "layer_total_us": 1.0,
                    "rows": [{"pos": 0, "row_id": "event-1",
                              "short_name": "kernel", "duration_us": 1.0,
                              "layer_evidence": (
                                  "validated_graph_capture_layer_scope_transfer")}],
                }]}, fh)
            with open(table_md, "w") as fh:
                fh.write("# clean\n")
            with open(plan, "w") as fh:
                json.dump({"capture_targets": []}, fh)
            with open(audit, "w") as fh:
                json.dump({"module_scope_count": 0}, fh)
            semantic = {
                "status": "pass", "semantic_table_json": table,
                "semantic_table_md": table_md,
                "shape_capture_plan_json": plan,
                "layer_instance_audit_json": audit,
            }
            merged_json = os.path.join(tmp, "merged.json")
            merged_md = os.path.join(tmp, "merged.md")
            with open(merged_json, "w") as fh:
                json.dump({"tables": []}, fh)
            with open(merged_md, "w") as fh:
                fh.write("# merged\n")
            merged = {"status": "pass", "semantic_table_json": merged_json,
                      "semantic_table_md": merged_md}
            capture_result = {
                "shape_capture_execution": "graph_capture",
                "capture_phases": ["decode"],
                "capture_trace": os.path.join(tmp, "graph.trace.json"),
                "shape_log": os.path.join(tmp, "shape.jsonl"),
            }
            marker_result = {
                "phase_coverage_complete": True,
                "shape_mapping_by_phase": {"decode": {
                    "shape_eligible_target_count": 1,
                    "shape_eligible_matched_target_count": 1,
                }},
                "clean_table_sequence_audit": {
                    "status": "pass", "groups": [
                        {"phase": "decode", "status": "pass"}]},
            }
            with mock.patch.object(
                    runner.validate_structural_patterns, "validate",
                    return_value={"patterns": [], "validation": {}}), \
                    mock.patch.object(
                        runner.semantic_kernel_mapping, "build",
                        side_effect=lambda *args, **kwargs: dict(semantic)), \
                    mock.patch.object(
                        runner.semantic_source_mapping, "map_plan",
                        return_value={}), \
                    mock.patch.object(
                        runner.run_semantic_shape_capture, "capture",
                        return_value=capture_result) as capture_call, \
                    mock.patch.object(
                        runner.semantic_layer_boundary_transfer, "transfer",
                        return_value={
                            "status": "partial",
                            "mapped_groups": [{"phase": "decode"}],
                            "residual_range_count": 1,
                            "failures": [],
                        }), \
                    mock.patch.object(
                        runner.semantic_runtime_marker_mapping, "map_plan",
                        return_value=marker_result) as map_call, \
                    mock.patch.object(
                        runner.semantic_shape_merge, "merge",
                        return_value=merged), \
                    mock.patch.object(
                        runner.semantic_evidence_ledger, "merge",
                        return_value=merged):
                result = runner.run(
                    config, trace, "", os.path.join(tmp, "out"),
                    capture_setup_path=setup,
                    structural_patterns_path=patterns,
                    require_phases=["decode"])
            self.assertTrue(capture_call.called)
            self.assertEqual(
                map_call.call_args[1]["clean_table_path"],
                os.path.join(tmp, "out", "pattern_layer_kernel_table_1_1.json"))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                result["boundary_rebuild"]["verified_phases"], ["decode"])
            self.assertEqual(
                result["boundary_rebuild"]["residual_range_count"], 1)
            self.assertEqual(result["blocking_degraded_boundary_phases"], [])

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
            audit = os.path.join(tmp, "layer_instance_audit.json")
            with open(audit, "w") as fh:
                json.dump({"module_scope_count": 61}, fh)
            semantic = {
                "status": "pass",
                "semantic_table_json": table,
                "semantic_table_md": table_md,
                "shape_capture_plan_json": plan,
                "layer_instance_audit_json": audit,
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

    def _capture_require_phases(self, tmp, env_value):
        """Drive run() far enough to see what require_phases build() was given."""
        config = os.path.join(tmp, "config.json")
        trace = os.path.join(tmp, "trace.json")
        shape_log = os.path.join(tmp, "shape.log")
        patterns = os.path.join(tmp, "agent_patterns.json")
        for path, value in (
                (config, "{}"), (trace, "{}"), (shape_log, "shape\n"),
                (patterns, '{"pattern_definition": {}}')):
            with open(path, "w") as fh:
                fh.write(value)
        seen = {}

        def fake_build(*args, **kwargs):
            seen["require_phases"] = kwargs.get("require_phases")
            raise RuntimeError("stop here")

        env = dict(os.environ)
        env.pop("GEAK_SEMANTICS_REQUIRE_PHASES", None)
        if env_value is not None:
            env["GEAK_SEMANTICS_REQUIRE_PHASES"] = env_value
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(
                    runner.validate_structural_patterns, "validate",
                    return_value={
                        "patterns": [],
                        "validation": {"definition_preserved": True}}), \
                mock.patch.object(
                    runner.semantic_kernel_mapping, "build", fake_build):
            with self.assertRaises(RuntimeError):
                runner.run(
                    config, trace, shape_log, os.path.join(tmp, "out"),
                    structural_patterns_path=patterns)
        return seen["require_phases"]

    def test_two_phase_requirement_is_the_default(self):
        # Regression: this used to read an env var defaulting to EMPTY, so an
        # unset var meant "require no phase" and a prefill-only capture passed.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                sorted(self._capture_require_phases(tmp, None)),
                ["decode", "prefill"])

    def test_env_var_can_narrow_the_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._capture_require_phases(tmp, "prefill"), ["prefill"])

    def test_explicit_argument_still_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(runner.DEFAULT_REQUIRE_PHASES,
                             ("prefill", "decode"))

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
