import json
import os
import sys
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_donor_shape_projection as projection  # noqa: E402
import semantic_kernel_mapping as mapping  # noqa: E402
import semantic_layer_boundary_transfer as transfer  # noqa: E402

BODY = ["layer0_a", "layer0_b", "layer1_a", "layer1_b"]


def _write(root, name, events):
    path = os.path.join(root, name)
    with open(path, "w") as fh:
        json.dump({"traceEvents": events}, fh)
    return path


def _patterns(root):
    path = os.path.join(root, "patterns.json")
    with open(path, "w") as fh:
        json.dump({
            "num_hidden_layers_main": 2,
            "patterns": [{
                "pattern_id": "P0", "pattern_display_name": "body",
                "layer_ids": [0, 1], "representative_candidates": [0, 1],
                "structural_signature": {"runtime_dispatch_branch": "vllm::layer_core"},
            }],
            "coverage_check": {"total_main_layers": 2, "covered": 2,
                               "mutually_exclusive": True, "full_coverage": True},
            "quality": {"status": "pass"},
        }, fh)
    return path


def _donor_step(events, step_ts, dims_by_name, correlation, names=BODY):
    """One CUDA-graph-off step: dispatch op per layer, shape-bearing parent ops."""
    events.append({"cat": "user_annotation", "name": "step[DECODE bs=4]",
                   "pid": 1, "tid": 2, "ts": step_ts, "dur": 80})
    events.append({"cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
                   "ts": step_ts, "dur": 120})
    for layer_id in (0, 1):
        base = step_ts + 5 + layer_id * 30
        events.append({"cat": "cpu_op", "name": "vllm::layer_core",
                       "pid": 1, "tid": 2, "ts": base, "dur": 25})
        for offset, name in enumerate(names[layer_id * 2:layer_id * 2 + 2]):
            correlation += 1
            external = 1000 + correlation
            events.append({
                "cat": "cpu_op", "name": "op_%s" % name, "pid": 1, "tid": 2,
                "ts": base + 2 + offset * 5, "dur": 4,
                "args": {"External id": external, "Input Dims": dims_by_name[name],
                         "Input type": ["c10::BFloat16"] * len(dims_by_name[name])}})
            events.append({
                "cat": "hip_runtime", "pid": 1, "tid": 2, "name": "hipLaunchKernel",
                "ts": base + 3 + offset * 5, "dur": 0.1,
                "args": {"correlation": correlation, "External id": external}})
            events.append({
                "cat": "kernel", "name": name, "ts": step_ts + 90 + correlation % 20,
                "dur": 1, "args": {"correlation": correlation, "External id": external}})
    return correlation


def _recipient(root):
    """Graph replay: the same kernels, no parent ops, no shapes."""
    names = ["prepare_once"] + BODY + ["model_epilogue"]
    events = [{"cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
               "ts": 0, "dur": 200}]
    events.extend({"cat": "kernel", "name": name, "ts": 10 + index * 3,
                   "dur": 1, "args": {}} for index, name in enumerate(names))
    return _write(root, "recipient.json", events)


def _dims(scale):
    return {name: [[4, 8 * scale]] for name in BODY}


class DonorShapeProjectionTest(unittest.TestCase):
    def _run(self, root, donor_steps):
        patterns = _patterns(root)
        events, correlation = [], 0
        for index, dims in enumerate(donor_steps):
            correlation = _donor_step(events, 1000 * (index + 1), dims, correlation)
        donor = _write(root, "donor.json", events)
        recipient = _recipient(root)
        boundary = os.path.join(root, "boundary.json")
        result = transfer.transfer(donor, recipient, patterns, boundary)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["donor"]["scope_source"], "declared_dispatch_op_span")
        out = os.path.join(root, "table")
        mapping.build(recipient, patterns, out, require_phases=["decode"],
                      boundary_map_paths=[boundary])
        table = os.path.join(out, "pattern_layer_kernel_table.json")
        summary = projection.project(table, boundary, recipient, donor, patterns,
                                     os.path.join(root, "projected"))
        with open(summary["table_json"]) as fh:
            return summary, json.load(fh)

    def test_projects_shape_and_parent_operator_as_p_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, doc = self._run(tmp, [_dims(1)])
            rows = [row for table in doc["tables"] for row in table["rows"]]
            self.assertTrue(rows)
            self.assertEqual(summary["by_phase"]["decode"]["projected"], len(rows))
            for row in rows:
                self.assertEqual(row["shape"]["source"], projection.SOURCE)
                self.assertEqual(row["shape"]["input_dims"], [[4, 8]])
                self.assertEqual(row["parent_operator"]["canonical_op"],
                                 "op_%s" % row["raw_name"])
                self.assertEqual(row["semantic_evidence"]["level"], "P")
            coverage = doc["phase_coverage"]
            self.assertTrue(coverage["decode_shapes_covered"])
            self.assertEqual(
                coverage["shape_resolution_by_phase"]["decode"]["resolved_fraction"], 1.0)
            with open(summary["table_md"]) as fh:
                self.assertIn("layer0_a", fh.read())

    def test_donor_passes_that_disagree_do_not_project(self):
        # Same kernel sequence and bucket, different dims (e.g. a KV-length-dependent
        # block table): picking either would be a guess.
        with tempfile.TemporaryDirectory() as tmp:
            summary, doc = self._run(tmp, [_dims(1), _dims(2)])
            decode = summary["by_phase"]["decode"]
            self.assertNotIn("projected", decode)
            self.assertEqual(decode["donor_passes_disagree_on_shape"], decode["rows"])
            for table in doc["tables"]:
                for row in table["rows"]:
                    self.assertEqual(row["shape"]["source"], "unresolved")

    def test_rejects_a_boundary_map_for_another_donor(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, [_dims(1)])
            other = _write(tmp, "other.json", [])
            with self.assertRaises(ValueError):
                projection.project(
                    os.path.join(tmp, "table", "pattern_layer_kernel_table.json"),
                    os.path.join(tmp, "boundary.json"),
                    os.path.join(tmp, "recipient.json"), other,
                    os.path.join(tmp, "patterns.json"), os.path.join(tmp, "x"))

    def test_layer_pairing_aliases_only_the_graph_replay_copy(self):
        group = {"layer_ranges": [{"start_position": 0, "end_position": 2},
                                  {"start_position": 2, "end_position": 4}]}
        donor = {"layer_starts": [0, 2],
                 "sequence": ["a", "Memcpy DtoD (Device -> Device)", "b", "c"]}
        pairs = projection._layer_identical_pairs(
            group, donor, ["a", "__amd_rocclr_copyBuffer", "b", "x"])
        # layer 0 differs only by the copy spelling -> paired; layer 1 pairs its
        # agreeing prefix ("b") and stops at the divergence ("c" vs "x").
        self.assertEqual(pairs, {0: 0, 1: 1, 2: 2})

    def test_layer_pairing_skips_work_only_the_eager_donor_launches(self):
        # MiniMax-M3 TP8 decode: eager copies each all-reduce input into the
        # registered buffer; graph replay does not. The prefix rule stopped at the
        # first copy and left every all-reduce and everything after it unpaired.
        group = {"layer_ranges": [{"start_position": 0, "end_position": 5}]}
        donor = {"layer_starts": [0],
                 "sequence": ["rope", "gemm", "pad_copy", "allreduce", "norm",
                              "pad_copy", "allreduce"]}
        pairs = projection._layer_identical_pairs(
            group, donor, ["rope", "gemm", "allreduce", "norm", "allreduce"])
        self.assertEqual(pairs, {0: 0, 1: 1, 3: 2, 4: 3, 6: 4})

    def test_layer_pairing_leaves_an_ambiguous_embedding_unpaired(self):
        # "x" could be either donor "x": only the unambiguous rows pair.
        self.assertEqual(projection._forced_subsequence_pairs(
            ["a", "x", "b"], ["a", "x", "x", "b"]), {0: 0, 2: 3})
        self.assertIsNone(projection._forced_subsequence_pairs(["a", "c"], ["a", "b"]))

    def test_layer_pairing_works_when_graph_replay_adds_the_extra_kernel(self):
        group = {"layer_ranges": [{"start_position": 0, "end_position": 3}]}
        donor = {"layer_starts": [0], "sequence": ["a", "b"]}
        self.assertEqual(projection._layer_identical_pairs(
            group, donor, ["a", "graph_only", "b"]), {0: 0, 1: 2})

    def test_projection_keeps_the_donor_rows_shape_granularity(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, doc = self._run(tmp, [_dims(1)])
            for table in doc["tables"]:
                for row in table["rows"]:
                    # one kernel per parent op in the fixture -> the kernel's own dims
                    self.assertEqual(row["shape"]["granularity"], "kernel")
                    self.assertEqual(row["semantic_evidence"]["probe_scope"], "kernel")
            decode = doc["phase_coverage"]["shape_resolution_by_phase"]["decode"]
            self.assertEqual(decode["kernel_level"], decode["rows"])

    def test_per_layer_rule_pairs_rows_of_the_same_identity(self):
        """A reordered layer must pair each row with its own kind, not its index."""
        donor = ["a", "b", "a", "c", "d"]
        recipient = ["b", "a", "a", "c", "d"]
        pairs = projection._stable_pairs(recipient, donor, [0, 3])
        self.assertEqual(sorted(pairs), [(0, 1), (1, 0), (2, 2), (3, 3), (4, 4)])
        self.assertTrue(all(donor[d] == recipient[r] for d, r in pairs))
        self.assertIn(transfer.STABLE_PER_LAYER_RULE, projection.STABLE_RULES)

    def test_residual_rule_pairs_the_agreeing_prefix_of_each_layer(self):
        # A residual range carved out of layer 0 makes it narrower than the donor's;
        # only the prefix both sides share from the cut is paired.
        group = {"match_rule": projection.STABLE_RULE + "_with_residuals",
                 "stable_projection": {"stable_event_count": 4},
                 "layer_ranges": [{"start_position": 0, "end_position": 1},
                                  {"start_position": 2, "end_position": 4}]}
        donor = {"layer_starts": [0, 2], "sequence": ["a", "r", "b", "c"]}
        pairs = projection._pairs_for(group, donor, ["a", "r", "b", "c"])
        self.assertEqual(pairs, {0: 0, 1: 1, 2: 2, 3: 3})
        # layer 0 (recipient width 1) pairs only "a"; layer 1 is identical.
        self.assertEqual(projection._layer_identical_pairs(
            group, donor, ["a", "r", "b", "c"]), {0: 0, 2: 2, 3: 3})


if __name__ == "__main__":
    unittest.main()
