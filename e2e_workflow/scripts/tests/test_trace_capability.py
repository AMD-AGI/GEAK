import gzip
import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import trace_capability


class TraceCapabilityTest(unittest.TestCase):
    def test_rank_sorted_manifest_and_capabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = [
                {"cat": "gpu_user_annotation",
                 "name": "execute_context_1(8)_generation_0(0)", "ts": 0, "dur": 100},
                {"cat": "gpu_user_annotation",
                 "name": "step[DECODE bs=4]", "ts": 100, "dur": 100},
                {"cat": "python_function",
                 "name": "nn.Module: Qwen3_5LinearDecoderLayer_0",
                 "ts": 110, "dur": 50},
                {"cat": "cpu_op", "name": "model.layers.0",
                 "ts": 1, "dur": 50,
                 "args": {"External id": 7, "Input Dims": [[2, 4]],
                          "Input type": ["Half"]}},
                {"cat": "kernel", "name": "kernel", "ts": 2, "dur": 3,
                 "args": {"External id": 7, "stream": 1}},
                {"cat": "flow", "name": "link", "ph": "s", "ts": 2},
            ]
            for rank in (1, 0):
                path = os.path.join(tmp, "rank_%d.pt.trace.json.gz" % rank)
                with gzip.open(path, "wt") as fh:
                    json.dump({"traceEvents": events}, fh)
            result = trace_capability.build_manifest(tmp)
            self.assertTrue(result["analysis_rank_trace"].endswith("rank_0.pt.trace.json.gz"))
            self.assertEqual([item["rank"] for item in result["trace_files"]], [0, 1])
            caps = result["capability"]["capabilities"]
            self.assertTrue(caps["phase_annotations"])
            self.assertTrue(caps["external_id"])
            self.assertTrue(caps["input_dims_types"])
            self.assertTrue(caps["flow_or_correlation"])
            self.assertTrue(caps["module_layer_spans"])
            self.assertEqual(result["capability"]["phase_annotation_count"], 2)
            self.assertEqual(
                result["capability"]["recommended_layer_mapping"],
                "module_span_plus_flow")

    def test_missing_trace_is_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = trace_capability.build_manifest(tmp)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["analysis_rank_trace"], "")


if __name__ == "__main__":
    unittest.main()
