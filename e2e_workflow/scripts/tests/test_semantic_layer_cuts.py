"""Layer ownership on hybrid models when the CPU runs ahead of the GPU.

Two historical failure modes, both model- and algorithm-agnostic:

* a flow-linked launch (no External id, e.g. a Triton kernel) was dropped
  whenever its *device* timestamp fell inside a later layer's *CPU* span,
  which is the normal case once the CPU launches ahead of a busy GPU;
* the resulting holes sent the step through a re-cut that split adjacent
  layers at the midpoint of their anchor medians, which is only correct
  when adjacent layers have the same kernel count.
"""
import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_kernel_mapping as mapping


# Hybrid layout: a long layer kind and a short layer kind alternate.
LONG = ["norm_kernel", "in_proj_gemm", "conv_kernel", "recurrent_kernel",
        "out_proj_gemm", "act_kernel", "down_gemm"]
SHORT = ["norm_kernel", "qkv_gemm", "attention_kernel", "down_gemm"]
LAYOUT = [LONG, SHORT, LONG, SHORT]


def _patterns(root):
    path = os.path.join(root, "patterns.json")
    with open(path, "w") as fh:
        json.dump({
            "schema_version": 1,
            "num_hidden_layers_main": len(LAYOUT),
            "patterns": [
                {"pattern_id": "P_LONG", "pattern_display_name": "Long",
                 "attention_type": "linear_attention", "layer_ids": [0, 2]},
                {"pattern_id": "P_SHORT", "pattern_display_name": "Short",
                 "attention_type": "full_attention", "layer_ids": [1, 3]},
            ],
            "coverage_check": {"total_main_layers": len(LAYOUT),
                               "covered": len(LAYOUT),
                               "mutually_exclusive": True,
                               "full_coverage": True},
            "quality": {"status": "pass"},
        }, fh)
    return path


def _cpu_ahead_trace(unlinked=()):
    """One prefill step whose CPU launches finish long before the GPU.

    Each layer's first kernel carries an External id; the rest are linked to
    their launch only by an ac2g flow.  CPU layer spans are 10us apart while
    every kernel takes 3us, so device timestamps land in later CPU spans.
    `unlinked` holds (layer_id, kernel_index) launches with neither link.
    """
    events = [
        {"ph": "X", "cat": "user_annotation", "tid": 1,
         "name": "step[EXTEND bs=1 toks=8]", "ts": 0, "dur": 45},
        {"ph": "X", "cat": "gpu_user_annotation",
         "name": "step[EXTEND bs=1 toks=8]", "ts": 1, "dur": 200},
    ]
    correlation = 0
    gpu_ts = 6
    for layer_id, kernels in enumerate(LAYOUT):
        cpu_ts = layer_id * 10
        cls = "LongDecoderLayer" if kernels is LONG else "ShortDecoderLayer"
        events.append({"ph": "X", "cat": "python_function", "tid": 1,
                       "name": "nn.Module: %s_%d" % (cls, layer_id),
                       "ts": cpu_ts, "dur": 10})
        for index, name in enumerate(kernels):
            correlation += 1
            launch_ts = cpu_ts + 1 + index
            kernel_args = {"stream": 7, "correlation": correlation}
            if (layer_id, index) in unlinked:
                pass
            elif index == 0:
                ext = 1000 + correlation
                kernel_args["External id"] = ext
                events.append({"ph": "X", "cat": "cpu_op", "tid": 1,
                               "name": "aten::%s" % name, "ts": launch_ts,
                               "dur": 1, "args": {"External id": ext}})
            else:
                events.append({"ph": "s", "cat": "ac2g", "name": "ac2g",
                               "id": correlation, "tid": 1, "ts": launch_ts})
                events.append({"ph": "f", "cat": "ac2g", "name": "ac2g",
                               "id": correlation, "bp": "e", "tid": 7,
                               "ts": gpu_ts})
            events.append({"ph": "X", "cat": "kernel", "name": name,
                           "ts": gpu_ts, "dur": 3, "args": kernel_args})
            gpu_ts += 4
    return events


def _owned_names(rows):
    owned = {}
    for row in sorted(rows, key=lambda row: row["device_seq_index"]):
        if row["layer_id"] is not None:
            owned.setdefault(row["layer_id"], []).append(row["raw_name"])
    return owned


class FlowSourceOwnershipTest(unittest.TestCase):
    def test_device_timestamp_in_a_later_cpu_span_keeps_its_launch_layer(self):
        events = _cpu_ahead_trace()
        # Precondition: device timestamps really do land in later CPU spans.
        spans = sorted((e["ts"], e["ts"] + e["dur"], e["name"])
                       for e in events if "nn.Module" in e.get("name", ""))
        layer0_gpu = [e["ts"] for e in events if e.get("ph") == "f"][:3]
        self.assertTrue(any(spans[1][0] <= ts < spans[1][1]
                            for ts in layer0_gpu))

        with tempfile.TemporaryDirectory() as tmp:
            with open(_patterns(tmp)) as fh:
                pattern_doc = json.load(fh)
        rows, _, _, _, _ = mapping._event_rows(events, pattern_doc)
        self.assertEqual(_owned_names(rows),
                         {layer_id: kernels
                          for layer_id, kernels in enumerate(LAYOUT)})
        flow_rows = [row for row in rows if row["raw_name"] != "norm_kernel"]
        self.assertTrue(all(row["layer_evidence"]
                            == "python_module_span_ac2g_flow"
                            for row in flow_rows))

    def test_flow_launched_outside_any_layer_span_stays_unowned(self):
        events = _cpu_ahead_trace()
        # Move one launch before the first layer span.
        launch = next(e for e in events if e.get("ph") == "s")
        launch["ts"] = -5
        with tempfile.TemporaryDirectory() as tmp:
            with open(_patterns(tmp)) as fh:
                pattern_doc = json.load(fh)
        rows, _, _, _, _ = mapping._event_rows(events, pattern_doc)
        target = next(e["ts"] for e in events
                      if e.get("ph") == "f" and e["id"] == launch["id"])
        row = next(row for row in rows if row["timestamp"] == target)
        self.assertIsNone(row["layer_id"])


class HybridLayerCutTest(unittest.TestCase):
    def test_holes_and_gaps_follow_the_layers_own_anchors(self):
        # A hole inside layer 0 and an unowned kernel ending layer 1.
        unlinked = {(0, 3), (1, len(SHORT) - 1)}
        with tempfile.TemporaryDirectory() as tmp:
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": _cpu_ahead_trace(unlinked)}, fh)
            result = mapping.build(trace, _patterns(tmp),
                                   os.path.join(tmp, "out"))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            with open(result["semantic_table_json"]) as fh:
                tables = json.load(fh)["tables"]

        diag = audit["boundary_partition_diagnostics"][0]
        self.assertEqual(diag["status"], "mapped")
        cut = diag["module_scope_ordered_cut"][0]
        self.assertEqual(
            cut["layer_start_device_seq_indices"],
            [1, 1 + len(LONG), 1 + len(LONG) + len(SHORT),
             1 + 2 * len(LONG) + len(SHORT)])
        self.assertEqual(cut["misassigned_anchor_count"], 0)
        self.assertTrue(all(item["method"] == "anchor_gap"
                            for item in cut["layer_boundaries"]))
        for layer_id, kernels in enumerate(LAYOUT):
            instance = next(item for item in audit["instances"]
                            if item["layer_id"] == layer_id)
            self.assertEqual(instance["event_count"], len(kernels))
        by_pattern = {table["pattern_id"]: table for table in tables}
        self.assertEqual(
            [row["raw_name"] for row in by_pattern["P_LONG"]["rows"]], LONG)
        self.assertEqual(
            [row["raw_name"] for row in by_pattern["P_SHORT"]["rows"]], SHORT)


def _rows(owners):
    """Step rows in device order; `owners[i]` is the raw layer of row i."""
    rows = []
    for index, owner in enumerate(owners):
        rows.append({
            "row_id": "event-%d" % index, "device_seq_index": index,
            "layer_id": owner,
            "layer_instance_id": None if owner is None else "L%d" % owner,
            "layer_evidence": ("python_module_span_external_id"
                               if owner is not None else "unresolved"),
            "assignment": "layer_body" if owner is not None
            else "concurrent_unresolved",
        })
    return rows


class NormalizeModuleScopeCutTest(unittest.TestCase):
    def _cut(self, owners, layer_count):
        rows = _rows(owners)
        instances = mapping._authoritative_instances(rows)
        patterns = {layer_id: {"pattern_id": "P"}
                    for layer_id in range(layer_count)}
        ok, audit = mapping._normalize_module_scope_cuts(
            rows, instances, layer_count, patterns)
        return ok, audit, [row["layer_id"] for row in rows]

    def test_unequal_layers_are_cut_at_their_first_anchor(self):
        # 8-kernel layer, 3-kernel layer, 8-kernel layer; one hole in layer 0
        # and one unowned kernel between layers 1 and 2.
        owners = ([0, 0, None, 0, 0, 0, 0, 0] + [1, 1, 1] + [None]
                  + [2] * 8)
        ok, audit, assigned = self._cut(owners, 3)
        self.assertTrue(ok)
        self.assertEqual(assigned, [0] * 8 + [1] * 4 + [2] * 8)
        self.assertEqual(audit[0]["layer_start_device_seq_indices"],
                         [0, 8, 12])

    def test_only_the_overlapping_boundary_uses_the_overlap_rule(self):
        # Layers 0/1 interleave on the device; layers 1/2 do not.
        owners = [0, 0, 0, 1, 0, 1, 1, None, 2, 2, 2, 2]
        ok, audit, assigned = self._cut(owners, 3)
        self.assertTrue(ok)
        first, second = audit[0]["layer_boundaries"]
        self.assertEqual(first["method"], "anchor_overlap_min_misassigned")
        self.assertEqual(first["misassigned_anchor_count"], 1)
        self.assertEqual(second["method"], "anchor_gap")
        self.assertEqual(second["start_device_seq_index"], 8)
        self.assertEqual(assigned[8:], [2, 2, 2, 2])
        self.assertEqual(assigned[7], 1)

    def test_layers_that_cannot_be_ordered_are_refused(self):
        rows = _rows([1, 1, 0, 0])
        instances = mapping._authoritative_instances(rows)
        ok, audit = mapping._normalize_module_scope_cuts(
            rows, instances, 2, {0: {}, 1: {}})
        self.assertFalse(ok)
        self.assertEqual(audit, [])


if __name__ == "__main__":
    unittest.main()
