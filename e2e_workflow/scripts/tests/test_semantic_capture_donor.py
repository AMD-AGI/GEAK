"""Graph-capture launches are the layer-boundary donor (Semantics 1.2).

Graph replay -- the Clean Trace -- executes exactly the kernels launched while
the graph was recorded.  An eager warmup can also run one-shot work: a fused
overlay's first-call self-check ran a reference GEMM plus compare kernels in
the bs=64 warmup of a Qwen3-14B cycle-1 replay and broke the old transfer.
"""
import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_kernel_mapping as mapping
import semantic_layer_boundary_transfer as transfer
import semantic_runtime_capture as capture
import semantic_runtime_marker_mapping as marker_mapping


LAYERS = {0: ["norm", "qkv_gemm", "attn", "o_gemm", "elementwise"],
          1: ["norm", "qkv_gemm", "attn", "o_gemm", "elementwise"]}
SELF_CHECK = ["ref_gemm", "abs", "reduce_max", "clamp"]


def _patterns(root, num_layers=2, representatives=None):
    path = os.path.join(root, "patterns.json")
    with open(path, "w") as fh:
        json.dump({
            "num_hidden_layers_main": num_layers,
            "patterns": [{
                "pattern_id": "P0", "pattern_display_name": "body",
                "layer_ids": list(range(num_layers)),
                "representative_candidates": (
                    representatives or list(range(num_layers))),
            }],
            "coverage_check": {"total_main_layers": num_layers,
                               "covered": num_layers,
                               "mutually_exclusive": True,
                               "full_coverage": True},
            "quality": {"status": "pass"},
        }, fh)
    return path


def _pass(events, source, base, layers, executes, correlation):
    """One marked forward: eager launches execute, capture launches do not."""
    for layer_id, names in sorted(layers.items()):
        start = base + layer_id * 100
        events.append({
            "cat": "user_annotation", "pid": 1, "tid": 2,
            "name": ("GEAK_LAYER_SCOPE|phase=DECODE|bs=4|toks=4|layer=%d|"
                     "src=%s|path=model.layers.%d"
                     % (layer_id, source, layer_id)),
            "ts": start, "dur": 90,
        })
        for offset, name in enumerate(names):
            correlation += 1
            events.append({
                "cat": "hip_runtime", "pid": 1, "tid": 2,
                "name": "hipLaunchKernel", "ts": start + 2 + offset,
                "dur": 0.1,
                "args": {"correlation": correlation, "kernel": name},
            })
            if executes:
                events.append({
                    "cat": "kernel", "name": name,
                    "ts": start + 50 + offset, "dur": 0.5,
                    "args": {"correlation": correlation},
                })
    return correlation


def _donor(root, eager_layers=None, capture_layers=None):
    events = []
    correlation = 0
    if eager_layers:
        correlation = _pass(events, "eager", 1000, eager_layers, True,
                            correlation)
    if capture_layers:
        _pass(events, "capture", 5000, capture_layers, False, correlation)
    path = os.path.join(root, "donor.json")
    with open(path, "w") as fh:
        json.dump({"traceEvents": events}, fh)
    return path


def _recipient(root, names):
    events = [{"cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
               "ts": 0, "dur": 10 + 3 * len(names)}]
    events.extend({"cat": "kernel", "name": name, "ts": 5 + index * 3,
                   "dur": 1, "args": {}} for index, name in enumerate(names))
    path = os.path.join(root, "recipient.json")
    with open(path, "w") as fh:
        json.dump({"traceEvents": events}, fh)
    return path


# Replay with input preparation before the graph and scheduler work for the
# next batch overlapping it; "elementwise" also exists inside the graph.
REPLAY = (["prepare_inputs", "index_put"]
          + LAYERS[0][:3] + ["alloc_decode", "elementwise"] + LAYERS[0][3:]
          + ["elementwise"] + LAYERS[1] + ["sampler"])


class CaptureDonorTransferTest(unittest.TestCase):
    def test_capture_donor_maps_replay_despite_polluted_eager_warmup(self):
        polluted = dict(LAYERS)
        polluted[0] = LAYERS[0][:2] + SELF_CHECK + LAYERS[0][2:]
        with tempfile.TemporaryDirectory() as tmp:
            patterns = _patterns(tmp, representatives=[0])
            recipient = _recipient(tmp, REPLAY)
            boundary = os.path.join(tmp, "boundary.json")
            result = transfer.transfer(
                _donor(tmp, eager_layers=polluted, capture_layers=LAYERS),
                recipient, patterns, boundary)
            self.assertEqual(result["status"], "pass", result["failures"])
            group = result["mapped_groups"][0]
            self.assertEqual(group["match_rule"],
                             "capture_launch_ordered_subsequence")
            self.assertEqual(group["donor"]["source"], "capture")
            self.assertEqual(group["capture_subsequence"]["matched_fraction"],
                             1.0)
            # Capture sets the cuts only: a layer runs from its first found
            # kernel to the next layer's start and drops nothing.
            layer1 = REPLAY.index("norm", 3)
            self.assertEqual(group["layer_start_positions"], [2, layer1])
            self.assertEqual(group["body_end_position"], len(REPLAY) - 1)
            self.assertEqual(group["capture_unmatched_positions"], [5, 6, 9])

            built = mapping.build(
                recipient, patterns, os.path.join(tmp, "out"),
                require_phases=["decode"], boundary_map_paths=[boundary])
            self.assertEqual(built["status"], "pass")
            with open(built["semantic_event_audit_jsonl"]) as fh:
                rows = [json.loads(line) for line in fh]
            owners = [row["layer_id"] for row in rows]
            self.assertEqual(owners, [None, None] + [0] * (layer1 - 2)
                             + [1] * (len(REPLAY) - 1 - layer1) + [None])
            unmatched = [index for index, row in enumerate(rows)
                         if row["layer_evidence"].endswith(
                             "_not_in_capture_launches")]
            self.assertEqual(unmatched, [5, 6, 9])
            # The representative layer keeps every Clean Trace kernel in it.
            with open(built["semantic_table_json"]) as fh:
                table = json.load(fh)["tables"][0]
            self.assertEqual(table["representative_layer_id"], 0)
            self.assertEqual([row["raw_name"] for row in table["rows"]],
                             REPLAY[2:layer1])

    def test_missing_capture_kernels_within_budget_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            # One of ten capture kernels is never seen: 90% still maps.
            names = dict(LAYERS)
            names[1] = LAYERS[1][:4] + ["capture_only_kernel"]
            result = transfer.transfer(
                _donor(tmp, capture_layers=names),
                _recipient(tmp, REPLAY), _patterns(tmp))
            self.assertEqual(result["status"], "pass", result["failures"])
            stats = result["mapped_groups"][0]["capture_subsequence"]
            self.assertEqual(stats["matched_kernel_count"], 9)
            self.assertEqual(stats["skipped_donor_kernels"],
                             [{"layer_id": 1, "identity": "capture_only_kernel"}])

    def test_unmatched_capture_donor_falls_back_to_the_eager_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_names = {0: ["a", "b", "c", "d", "e"],
                             1: ["f", "g", "h", "i", "j"]}
            result = transfer.transfer(
                _donor(tmp, eager_layers=LAYERS, capture_layers=capture_names),
                _recipient(tmp, ["prep"] + LAYERS[0] + LAYERS[1] + ["tail"]),
                _patterns(tmp))
            self.assertEqual(result["status"], "pass", result["failures"])
            group = result["mapped_groups"][0]
            self.assertEqual(group["match_rule"],
                             "exact_contiguous_normalized_device_sequence")

    def test_capture_donor_alone_reports_why_it_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = transfer.transfer(
                _donor(tmp, capture_layers={0: ["a"], 1: ["b"]}),
                _recipient(tmp, ["x", "y", "z"]), _patterns(tmp))
            self.assertEqual(result["status"], "fail")
            failure = result["failures"][0]
            self.assertEqual(failure["capture_donor_failures"][0]["reason"],
                             "capture_donor_not_found_in_step")

    def test_capture_scope_uses_launch_records_not_device_kernels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _donor(tmp, eager_layers=LAYERS, capture_layers=LAYERS)
            events = mapping._load_events(path)
            _, passes = transfer._complete_donor_passes(events, 2)
            by_source = {item["source"]: item for item in passes}
            self.assertEqual(sorted(by_source), ["capture", "eager"])
            self.assertEqual(by_source["capture"]["sequence"],
                             LAYERS[0] + LAYERS[1])
            self.assertEqual(by_source["capture"]["layer_starts"], [0, 5])


class CaptureScopeConsumersTest(unittest.TestCase):
    def test_shape_marker_mapping_ignores_capture_scopes(self):
        events = [
            {"cat": "user_annotation", "pid": 1, "tid": 2, "ts": 1, "dur": 5,
             "name": "GEAK_LAYER_SCOPE|phase=DECODE|bs=4|toks=4|layer=0|"
                     "src=eager|path=model.layers.0"},
            {"cat": "user_annotation", "pid": 1, "tid": 2, "ts": 9, "dur": 5,
             "name": "GEAK_LAYER_SCOPE|phase=DECODE|bs=4|toks=4|layer=0|"
                     "src=capture|path=model.layers.0"},
            {"cat": "user_annotation", "pid": 1, "tid": 2, "ts": 20, "dur": 5,
             "name": "GEAK_LAYER_SCOPE|phase=DECODE|bs=4|toks=4|layer=0|"
                     "path=model.layers.0"},
        ]
        scopes = marker_mapping._layer_scope_markers(events)
        self.assertEqual([item["ts"] for item in scopes], [1.0, 20.0])


class RuntimeLayerScopeSourceTest(unittest.TestCase):
    def _logger(self):
        logger = capture.SemanticRuntimeLogger.__new__(
            capture.SemanticRuntimeLogger)
        logger.layer_scopes = True
        logger.phases = set()
        logger.max_forwards = 1
        logger._bucket_forwards = {}
        logger._context = {"phase": "DECODE", "batch_size": 64,
                           "input_tokens": 64}
        logger.active = lambda: True
        return logger

    def test_first_eager_forward_then_every_capture_forward_is_marked(self):
        logger = self._logger()
        original = capture._stream_capturing
        try:
            capture._stream_capturing = lambda: False
            self.assertEqual(logger._layer_scope_source(0), "eager")
            logger.mark_forward()
            self.assertIsNone(logger._layer_scope_source(0))
            capture._stream_capturing = lambda: True
            self.assertEqual(logger._layer_scope_source(0), "capture")
            self.assertIsNone(logger._layer_scope_source(-1))
        finally:
            capture._stream_capturing = original


if __name__ == "__main__":
    unittest.main()
