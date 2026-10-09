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

    def _dispatch_donor(self, root, anchors=True):
        """A cudagraph-off vLLM donor: no markers, one declared dispatch op per layer."""
        events = [
            {"cat": "user_annotation", "name": "step[DECODE bs=4]", "pid": 1, "tid": 2,
             "ts": 5, "dur": 80},
            # As on a real vLLM trace, the device-side step window also spans the
            # host-side dispatch ops of that step.
            {"cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
             "ts": 5, "dur": 120},
        ]
        correlation = 0
        for layer_id in (0, 1):
            base = 10 + layer_id * 30
            if anchors:
                events.append({"cat": "cpu_op", "name": "vllm::layer_core",
                               "pid": 1, "tid": 2, "ts": base, "dur": 5})
            for offset, name in enumerate(("layer%d_a" % layer_id,
                                           "layer%d_b" % layer_id)):
                correlation += 1
                events.append({
                    "cat": "hip_runtime", "pid": 1, "tid": 2,
                    "name": "hipLaunchKernel", "ts": base + 2 + offset,
                    "dur": 0.1, "args": {"correlation": correlation}})
                events.append({
                    "cat": "kernel", "name": name, "ts": 100 + correlation,
                    "dur": 1, "args": {"correlation": correlation}})
        path = os.path.join(root, "dispatch_donor.json")
        with open(path, "w") as fh:
            json.dump({"traceEvents": events}, fh)
        return path

    def _dispatch_patterns(self, root):
        path = self._patterns(root)
        with open(path) as fh:
            doc = json.load(fh)
        doc["patterns"][0]["structural_signature"] = {
            "runtime_dispatch_branch": "vllm::layer_core"}
        with open(path, "w") as fh:
            json.dump(doc, fh)
        return path

    def test_dispatch_op_donor_transfers_cuts_without_markers(self):
        # vLLM: module hooks cannot emit GEAK_LAYER_SCOPE inside torch.compile, but a
        # cudagraph-off capture carries each layer's declared dispatch op.
        with tempfile.TemporaryDirectory() as tmp:
            result = transfer.transfer(
                self._dispatch_donor(tmp), self._recipient(tmp),
                self._dispatch_patterns(tmp), os.path.join(tmp, "b.json"))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["donor"]["scope_source"],
                             "declared_dispatch_op_span")
            self.assertEqual(
                result["mapped_groups"][0]["layer_start_positions"], [1, 3])

    def test_donor_without_markers_or_dispatch_ops_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = transfer.transfer(
                self._dispatch_donor(tmp, anchors=False), self._recipient(tmp),
                self._dispatch_patterns(tmp), os.path.join(tmp, "b.json"))
            self.assertEqual(result["status"], "fail")
            self.assertIn("donor_has_no_layer_scopes",
                          [item.get("reason") for item in result["failures"]])

    def test_dispatch_anchored_step_counts_as_already_authoritative(self):
        # A vLLM prefill step cut by Pattern-declared dispatch ops is already complete;
        # retrying a transfer on it only produced a bucket-mismatch failure that failed
        # the whole map even though every decode step transferred.
        def rows(evidence):
            return [{"layer_instance_id": "s:pass-0:layer-%d" % layer,
                     "layer_id": layer, "layer_evidence": evidence,
                     "device_seq_index": layer} for layer in range(3)]
        self.assertTrue(transfer._complete_authoritative_step(
            rows("declared_dispatch_op_span"), 3))
        self.assertTrue(transfer._complete_authoritative_step(
            rows("python_module_span_external_id"), 3))
        self.assertFalse(transfer._complete_authoritative_step(
            rows("cpu_op_scope"), 3))

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

    def test_step_of_a_phase_the_donor_never_captured_does_not_veto(self):
        # A Profile DECODE trace can hold a prefill step; the graph-construction
        # donor only has decode passes. The decode step must still transfer.
        with tempfile.TemporaryDirectory() as tmp:
            patterns = self._patterns(tmp)
            body = ["layer0_a", "layer0_b", "layer1_a", "layer1_b"]
            events = [
                {"cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
                 "ts": 0, "dur": 200},
                {"cat": "gpu_user_annotation", "name": "step[EXTEND bs=2 toks=16]",
                 "ts": 300, "dur": 100},
            ]
            names = ["prepare_once"] + body + ["model_epilogue"]
            events.extend({"cat": "kernel", "name": name, "ts": 10 + i * 3,
                           "dur": 1, "args": {}} for i, name in enumerate(names))
            events.extend({"cat": "kernel", "name": name, "ts": 310 + i * 3,
                           "dur": 1, "args": {}} for i, name in enumerate(
                               ["prefill_x", "prefill_y", "prefill_z"]))
            recipient = os.path.join(tmp, "recipient.json")
            with open(recipient, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = transfer.transfer(self._donor(tmp), recipient, patterns)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["failures"], [])
            self.assertEqual(result["mapped_step_count"], 1)
            self.assertEqual(
                [(item["phase"], item["reason"]) for item in result["untransferable_steps"]],
                [("prefill", "no_donor_pass_for_phase")])

    def test_unmappable_step_of_a_captured_phase_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = transfer.transfer(
                self._donor(tmp),
                self._recipient(tmp, body=["x", "y", "x", "y"]),
                self._patterns(tmp))
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["untransferable_steps"], [])

    def test_partial_donor_pass_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = transfer.transfer(
                self._donor(tmp, layers=(0,)), self._recipient(tmp),
                self._patterns(tmp))
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any(
                item["reason"] == "donor_has_no_complete_nonempty_layer_pass"
                for item in result["failures"]))

    def test_a_step_without_a_donor_pass_leaves_the_map_partial(self):
        """A batch size the donor never ran must not void the steps it did validate."""
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            for index, (batch, start) in enumerate(((4, 0), (8, 300))):
                events.append({"cat": "gpu_user_annotation",
                               "name": "step[DECODE bs=%d]" % batch,
                               "ts": start, "dur": 200})
                names = ["prepare_once", "layer0_a", "layer0_b",
                         "layer1_a", "layer1_b", "model_epilogue"]
                events.extend({"cat": "kernel", "name": name,
                               "ts": start + 10 + offset * 3, "dur": 1, "args": {}}
                              for offset, name in enumerate(names))
            recipient = os.path.join(tmp, "two_steps.json")
            with open(recipient, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = transfer.transfer(
                self._donor(tmp), recipient, self._patterns(tmp))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["mapped_step_count"], 1)
            self.assertEqual(result["unmapped_step_count"], 1)
            self.assertEqual(result["failures"][0]["batch_size"], 8)
            self.assertEqual(result["failures"][0]["compatible_donor_pass_count"], 0)

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

    def test_reordering_inside_a_layer_passes_on_the_per_layer_multiset(self):
        """Concurrent streams interleave a layer's work by device timing.

        MiniMax-M3 TP8 (default config): routed experts and the shared expert run
        on two streams, so no decode step's projection was byte-equal to the
        donor's while every layer's multiset was.
        """
        with tempfile.TemporaryDirectory() as tmp:
            donor = self._donor(tmp, names={
                0: ["layer0_a", "layer0_b", "layer0_c"],
                1: ["layer1_a", "layer1_b", "layer1_c"],
            })
            recipient = self._recipient(tmp, body=[
                "layer0_b", "layer0_a", "layer0_c",
                "layer1_a", "layer1_b", "layer1_c",
            ])
            result = transfer.transfer(
                donor, recipient, self._patterns(tmp),
                os.path.join(tmp, "boundary.json"))
            self.assertEqual(result["status"], "pass")
            group = result["mapped_groups"][0]
            self.assertEqual(group["match_rule"], transfer.STABLE_PER_LAYER_RULE)
            self.assertEqual(group["layer_start_positions"], [1, 4])
            stable = group["stable_projection"]
            self.assertEqual(stable["order_rule"], "per_layer_multiset")
            self.assertEqual(stable["reordered_layer_count"], 1)
            self.assertEqual(stable["reordered_event_count"], 2)

    def test_reordering_across_a_layer_boundary_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            donor = self._donor(tmp, names={
                0: ["layer0_a", "layer0_b", "layer0_c"],
                1: ["layer1_a", "layer1_b", "layer1_c"],
            })
            recipient = self._recipient(tmp, body=[
                "layer0_a", "layer0_b", "layer1_a",
                "layer0_c", "layer1_b", "layer1_c",
            ])
            result = transfer.transfer(
                donor, recipient, self._patterns(tmp))
            self.assertEqual(result["status"], "fail")
            failure = result["failures"][0]["stable_projection_failures"][0]
            self.assertEqual(failure["reason"], "stable_identity_projection_mismatch")
            self.assertEqual(failure["first_mismatch_layer"], 0)

    def test_byte_equal_projection_keeps_the_exact_rule(self):
        pairs, info = transfer.stable_projection_pairs(
            ["p", "a", "b", "c", "d"], ["a", "b", "c", "d"], [0, 2])
        self.assertEqual(info["order_rule"], "exact_sequence")
        self.assertEqual([(d, r) for _, d, r in pairs],
                         [(0, 1), (1, 2), (2, 3), (3, 4)])

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
