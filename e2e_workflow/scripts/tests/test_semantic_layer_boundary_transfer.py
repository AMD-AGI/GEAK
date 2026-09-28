import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_kernel_mapping as mapping
import semantic_layer_boundary_transfer as transfer


class SemanticLayerBoundaryTransferTest(unittest.TestCase):
    def _patterns(self, root, num_layers=2):
        path = os.path.join(root, "patterns.json")
        with open(path, "w") as fh:
            json.dump({
                "num_hidden_layers_main": num_layers,
                "patterns": [{
                    "pattern_id": "P0", "pattern_display_name": "body",
                    "layer_ids": list(range(num_layers)),
                    "representative_candidates": list(range(num_layers)),
                }],
                "coverage_check": {"total_main_layers": num_layers,
                                   "covered": num_layers,
                                   "mutually_exclusive": True,
                                   "full_coverage": True},
                "quality": {"status": "pass"},
            }, fh)
        return path

    def _donor(self, root, layers=(0, 1), names=None):
        names = names or {0: ["layer0_a", "layer0_b"],
                          1: ["layer1_a", "layer1_b"]}
        events = []
        correlation = 0
        for layer_id in layers:
            base = 10 + layer_id * 30
            events.append({
                "cat": "user_annotation", "pid": 1, "tid": 2,
                "name": (
                    "GEAK_LAYER_SCOPE|phase=DECODE|bs=4|toks=4|"
                    "layer=%d|path=model.layers.%d" % (layer_id, layer_id)),
                "ts": base, "dur": 20,
            })
            for offset, name in enumerate(names[layer_id]):
                correlation += 1
                events.append({
                    "cat": "hip_runtime", "pid": 1, "tid": 2,
                    "name": "hipLaunchKernel", "ts": base + 2 + offset,
                    "dur": 0.1, "args": {"correlation": correlation,
                                           "kernel": name},
                })
                events.append({
                    "cat": "kernel", "name": name,
                    "ts": 100 + correlation, "dur": 1,
                    "args": {"correlation": correlation},
                })
        path = os.path.join(root, "donor.json")
        with open(path, "w") as fh:
            json.dump({"traceEvents": events}, fh)
        return path

    def _recipient(self, root, body=None):
        body = body or ["layer0_a", "layer0_b", "layer1_a", "layer1_b"]
        names = ["prepare_once"] + body + ["model_epilogue"]
        events = [{
            "cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
            "ts": 0, "dur": 200,
        }]
        events.extend({
            "cat": "kernel", "name": name, "ts": 10 + index * 3,
            "dur": 1, "args": {},
        } for index, name in enumerate(names))
        path = os.path.join(root, "recipient.json")
        with open(path, "w") as fh:
            json.dump({"traceEvents": events}, fh)
        return path

    def test_exact_full_pass_transfers_cuts_and_preserves_outer_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = self._patterns(tmp)
            recipient = self._recipient(tmp)
            boundary = os.path.join(tmp, "boundary.json")
            result = transfer.transfer(
                self._donor(tmp), recipient, patterns, boundary)
            self.assertEqual(result["status"], "pass")
            group = result["mapped_groups"][0]
            self.assertEqual(group["layer_start_positions"], [1, 3])
            self.assertEqual(group["body_end_position"], 5)
            self.assertEqual(group["prefix_row_count"], 1)
            self.assertEqual(group["suffix_row_count"], 1)

            built = mapping.build(
                recipient, patterns, os.path.join(tmp, "out"),
                require_phases=["decode"], boundary_map_paths=[boundary])
            self.assertEqual(built["status"], "pass")
            with open(built["semantic_event_audit_jsonl"]) as fh:
                rows = [json.loads(line) for line in fh]
            self.assertEqual(rows[0]["assignment"], "transition_global")
            self.assertIsNone(rows[0]["layer_id"])
            self.assertEqual(rows[-1]["assignment"], "transition_global")
            self.assertEqual(
                [row["layer_id"] for row in rows[1:5]], [0, 0, 1, 1])
            self.assertTrue(all(
                row["layer_evidence"].startswith(
                    "validated_graph_capture_layer_scope")
                for row in rows[1:5]))

    def test_partial_donor_pass_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = transfer.transfer(
                self._donor(tmp, layers=(0,)), self._recipient(tmp),
                self._patterns(tmp))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                item["reason"] == "donor_has_no_complete_nonempty_layer_pass"
                for item in result["failures"]))

    def test_nonidentical_sequence_is_rejected_instead_of_aligned_by_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = transfer.transfer(
                self._donor(tmp),
                self._recipient(tmp, body=["x", "y", "x", "y"]),
                self._patterns(tmp))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                item["reason"] == "no_validated_stable_projection"
                for item in result["failures"]))

    def test_exact_stable_projection_handles_capture_replay_only_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            donor = self._donor(tmp, names={
                0: ["layer0_a", "layer0_b"],
                1: ["capture_only_fill", "layer1_a", "layer1_b",
                    "Memset (Device)", "capture_only_tail"],
            })
            recipient = self._recipient(tmp, body=[
                "layer0_a", "layer0_b", "replay_only_collective",
                "layer1_a", "layer1_b", "__amd_rocclr_fillBufferAligned",
            ])
            boundary = os.path.join(tmp, "boundary.json")
            result = transfer.transfer(
                donor, recipient, self._patterns(tmp), boundary)
            self.assertEqual(result["status"], "pass")
            group = result["mapped_groups"][0]
            self.assertEqual(
                group["match_rule"],
                "exact_equal_multiplicity_stable_identity_projection")
            self.assertEqual(group["layer_start_positions"], [1, 3])
            self.assertEqual(group["body_end_position"], 7)
            self.assertEqual(group["prefix_row_count"], 1)
            self.assertEqual(group["suffix_row_count"], 1)
            self.assertEqual(
                group["stable_projection"]["boundary_gaps"][0]
                ["assigned_side"], "current_layer_prefix")

            built = mapping.build(
                recipient, self._patterns(tmp), os.path.join(tmp, "out"),
                require_phases=["decode"], boundary_map_paths=[boundary])
            self.assertEqual(built["status"], "pass")
            with open(built["semantic_event_audit_jsonl"]) as fh:
                rows = [json.loads(line) for line in fh]
            self.assertEqual(
                [row["layer_id"] for row in rows[1:7]],
                [0, 0, 1, 1, 1, 1])

    def test_stable_projection_keeps_ambiguous_two_sided_gap_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            donor = self._donor(tmp, names={
                0: ["layer0_a", "layer0_b", "capture_suffix"],
                1: ["capture_prefix", "layer1_a", "layer1_b"],
            })
            recipient = self._recipient(tmp, body=[
                "layer0_a", "layer0_b", "replay_gap",
                "layer1_a", "layer1_b",
            ])
            boundary = os.path.join(tmp, "boundary.json")
            result = transfer.transfer(
                donor, recipient, self._patterns(tmp), boundary)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["residual_range_count"], 1)
            group = result["mapped_groups"][0]
            self.assertEqual(
                group["match_rule"],
                "exact_equal_multiplicity_stable_identity_projection_with_residuals")
            self.assertEqual(group["layer_ranges"], [
                {"layer_id": 0, "start_position": 1, "end_position": 3,
                 "representative_eligible": True},
                {"layer_id": 1, "start_position": 4, "end_position": 6,
                 "representative_eligible": True},
            ])
            self.assertEqual(
                group["residual_ranges"][0]["recipient_gap_identities"],
                ["replay_gap"])

            built = mapping.build(
                recipient, self._patterns(tmp), os.path.join(tmp, "out"),
                require_phases=["decode"], boundary_map_paths=[boundary])
            self.assertEqual(built["status"], "pass")
            with open(built["semantic_event_audit_jsonl"]) as fh:
                rows = [json.loads(line) for line in fh]
            self.assertEqual(
                [row["layer_id"] for row in rows[1:6]],
                [0, 0, None, 1, 1])
            self.assertEqual(rows[3]["layer_region"], "inter_layer_residual")

    def test_unique_first_layer_pattern_is_a_valid_fallback_without_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "num_hidden_layers_main": 2,
                    "patterns": [
                        {
                            "pattern_id": "P0",
                            "pattern_display_name": "first-only",
                            "layer_ids": [0],
                            "representative_candidates": [0],
                        },
                        {
                            "pattern_id": "P1",
                            "pattern_display_name": "ordinary",
                            "layer_ids": [1],
                            "representative_candidates": [1],
                        },
                    ],
                    "coverage_check": {
                        "total_main_layers": 2,
                        "covered": 2,
                        "mutually_exclusive": True,
                        "full_coverage": True,
                    },
                    "quality": {"status": "pass"},
                }, fh)
            recipient = self._recipient(tmp)
            boundary = os.path.join(tmp, "boundary.json")
            result = transfer.transfer(
                self._donor(tmp), recipient, patterns, boundary)
            self.assertEqual(result["status"], "pass")

            built = mapping.build(
                recipient, patterns, os.path.join(tmp, "out"),
                require_phases=["decode"], boundary_map_paths=[boundary])
            self.assertEqual(built["status"], "pass")
            with open(built["semantic_table_json"]) as fh:
                table = json.load(fh)
            by_pattern = {item["pattern_id"]: item for item in table["tables"]}
            self.assertEqual(by_pattern["P0"]["representative_layer_id"], 0)

            with open(built["semantic_event_audit_jsonl"]) as fh:
                rows = [json.loads(line) for line in fh]
            self.assertEqual(rows[0]["raw_name"], "prepare_once")
            self.assertEqual(rows[0]["assignment"], "transition_global")
            self.assertIsNone(rows[0]["layer_id"])

    def test_boundary_artifact_declares_auto_adopted_phase_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = self._recipient(tmp)
            decode = os.path.join(tmp, "run-TP-0-DECODE.trace.json")
            extend = os.path.join(tmp, "run-TP-0-EXTEND.trace.json")
            os.rename(original, decode)
            with open(extend, "w") as fh:
                json.dump({"traceEvents": []}, fh)
            boundary = os.path.join(tmp, "boundary.json")
            result = transfer.transfer(
                self._donor(tmp), decode, self._patterns(tmp), boundary)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                {item["path"] for item in result["recipient"]["traces"]},
                {decode, extend})
            self.assertEqual(
                result["recipient"]["adopted_phase_siblings"],
                [{"phase": "EXTEND", "path": extend}])

            built = mapping.build(
                decode, self._patterns(tmp), os.path.join(tmp, "out"),
                require_phases=["decode"], boundary_map_paths=[boundary])
            self.assertEqual(built["status"], "pass")

    def test_residual_boundary_allows_adjacent_stable_representative(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = {
                0: ["layer0_a", "layer0_b", "capture_suffix"],
                1: ["capture_prefix", "layer1_a", "layer1_b"],
                2: ["layer2_a", "layer2_b"],
                3: ["layer3_a", "layer3_b"],
            }
            donor = self._donor(
                tmp, layers=(0, 1, 2, 3), names=names)
            recipient = self._recipient(tmp, body=[
                "layer0_a", "layer0_b", "replay_gap",
                "layer1_a", "layer1_b",
                "layer2_a", "layer2_b",
                "layer3_a", "layer3_b",
            ])
            patterns = self._patterns(tmp, num_layers=4)
            boundary = os.path.join(tmp, "boundary.json")
            result = transfer.transfer(
                donor, recipient, patterns, boundary)
            self.assertEqual(result["status"], "partial")

            built = mapping.build(
                recipient, patterns, os.path.join(tmp, "out"),
                require_phases=["decode"], boundary_map_paths=[boundary])
            self.assertEqual(built["status"], "pass")
            with open(built["semantic_table_json"]) as fh:
                table = json.load(fh)
            self.assertEqual(
                table["tables"][0]["representative_layer_id"], 0)

    def test_consumer_refuses_failed_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "failed.json")
            with open(path, "w") as fh:
                json.dump({"status": "fail", "failures": ["bad"]}, fh)
            with self.assertRaisesRegex(ValueError, "non-passing"):
                mapping._apply_boundary_map([], path, {})


if __name__ == "__main__":
    unittest.main()
