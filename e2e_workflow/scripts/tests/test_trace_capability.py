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
    @staticmethod
    def _steps(tmp, name, steps):
        """One rank trace whose gpu steps each hold one kernel."""
        events = []
        for index, step in enumerate(steps):
            start = index * 100
            events.append({"cat": "gpu_user_annotation", "name": step,
                           "ts": start, "dur": 50})
            events.append({"cat": "kernel", "name": "k", "ts": start + 10, "dur": 5})
        with gzip.open(os.path.join(tmp, name), "wt") as fh:
            json.dump({"traceEvents": events}, fh)

    PREFILL = "execute_context_1(8)_generation_0(0)"
    DECODE = "execute_context_0(0)_generation_4(4)"

    def test_a_decode_only_trace_fails_a_two_phase_requirement(self):
        """A Profile shaped for steady decode can skip the prefill ramp entirely."""
        with tempfile.TemporaryDirectory() as tmp:
            self._steps(tmp, "rank_0.pt.trace.json.gz", [self.DECODE, self.DECODE])
            plain = trace_capability.build_manifest(tmp, auto_select_rank=True)
            self.assertEqual(plain["status"], "pass")
            required = trace_capability.build_manifest(
                tmp, auto_select_rank=True, require_phases=["prefill", "decode"])
            self.assertEqual(required["status"], "failed")
            self.assertEqual(required["phases_present"], ["decode"])
            self.assertEqual(required["missing_required_phases"], ["prefill"])

    def test_both_phases_in_one_mixed_trace_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._steps(tmp, "rank_0.pt.trace.json.gz", [self.PREFILL, self.DECODE])
            result = trace_capability.build_manifest(
                tmp, auto_select_rank=True, require_phases=["prefill", "decode"])
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["missing_required_phases"], [])

    def test_phases_split_across_a_ranks_files_count_together(self):
        """sglang writes EXTEND and DECODE as two traces of the same rank."""
        with tempfile.TemporaryDirectory() as tmp:
            self._steps(tmp, "rank_0-EXTEND.pt.trace.json.gz", [self.PREFILL])
            self._steps(tmp, "rank_0-DECODE.pt.trace.json.gz", [self.DECODE])
            result = trace_capability.build_manifest(
                tmp, auto_select_rank=True, require_phases=["prefill", "decode"])
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["phases_present"], ["decode", "prefill"])

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

    def test_discovery_ignores_hidden_scratch_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, ".vllm_bench_1_2.json"), "w").write("{}")
            trace = os.path.join(tmp, "dp0_pp0_tp0_dcp0_ep0_rank0.1.pt.trace.json.gz")
            with gzip.open(trace, "wt") as fh:
                json.dump({"traceEvents": []}, fh)
            self.assertEqual(trace_capability.discover(tmp), [trace])

    def test_auto_select_reads_vllm_execute_annotations(self):
        # vLLM has no profile_by_stage: one trace per rank, phases told apart only by the
        # execute_* step annotation. Without reading it every rank looked like it had
        # "missing_decode_annotation" and every vLLM fusion capture failed its manifest.
        with tempfile.TemporaryDirectory() as tmp:
            events = [
                {"cat": "gpu_user_annotation", "ts": 100, "dur": 50,
                 "name": "execute_1024_context_1(sq1024sk1024sqsq1048576sqsk1048576)"
                         "_generation_0(sq0sk0sqsq0sqsk0)"},
                {"cat": "gpu_user_annotation", "ts": 200, "dur": 50,
                 "name": "execute_16_context_0(sq0sk0sqsq0sqsk0)"
                         "_generation_16(sq16sk16409sqsq16sqsk16409)"},
            ]
            events.extend({"cat": "kernel", "name": "k%d" % index, "ts": ts,
                           "dur": 1, "args": {"stream": 1}}
                          for index, ts in enumerate((110, 120, 210, 220, 230)))
            path = os.path.join(
                tmp, "dp0_pp0_tp0_dcp0_ep0_rank0.1790.pt.trace.json.gz")
            with gzip.open(path, "wt") as fh:
                json.dump({"traceEvents": events}, fh)

            result = trace_capability.build_manifest(tmp, auto_select_rank=True)

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["analysis_rank"], 0)
            entry = result["trace_files"][0]
            self.assertEqual(entry["device_events_by_phase"],
                             {"decode": 3, "prefill": 2})

    def test_auto_selects_rank_with_largest_decode_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank, kernel_count in ((0, 2), (1, 5), (2, 3)):
                events = [{
                    "cat": "gpu_user_annotation",
                    "name": "step[DECODE bs=4]",
                    "ts": 100,
                    "dur": 100,
                }]
                events.extend({
                    "cat": "kernel", "name": "kernel_%d" % index,
                    "ts": 110 + index, "dur": 1, "args": {"stream": 1},
                } for index in range(kernel_count))
                path = os.path.join(
                    tmp, "capture-TP-%d-DECODE.trace.json.gz" % rank)
                with gzip.open(path, "wt") as fh:
                    json.dump({"traceEvents": events}, fh)

            result = trace_capability.build_manifest(
                tmp, auto_select_rank=True)

            self.assertEqual(result["analysis_rank"], 1)
            self.assertTrue(result["analysis_rank_trace"].endswith(
                "TP-1-DECODE.trace.json.gz"))
            self.assertEqual(
                result["analysis_rank_selection_reason"],
                "max_decode_step_device_coverage")

    def test_cross_rank_event_max_selects_full_decode_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank, duration, kernel_count in (
                    (0, 6000, 3), (1, 14200, 5), (2, 14100, 4)):
                events = [{
                    "cat": "gpu_user_annotation",
                    "name": "step[DECODE bs=4]",
                    "ts": 100,
                    "dur": duration,
                }]
                events.extend({
                    "cat": "kernel", "name": "kernel_%d" % index,
                    "ts": 110 + index, "dur": 1, "args": {"stream": 1},
                } for index in range(kernel_count))
                path = os.path.join(
                    tmp, "capture-TP-%d-DECODE.trace.json.gz" % rank)
                with gzip.open(path, "wt") as fh:
                    json.dump({"traceEvents": events}, fh)

            result = trace_capability.build_manifest(tmp, auto_select_rank=True)

            self.assertEqual(result["selected_analysis_rank"], 1)
            self.assertEqual(
                [item["eligible"] for item in result["rank_candidates"]],
                [False, True, False])

    def test_all_ranks_without_decode_events_fail_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank in (0, 1):
                events = [{
                    "cat": "gpu_user_annotation",
                    "name": "step[DECODE bs=4]",
                    "ts": 100,
                    "dur": 6000,
                }]
                path = os.path.join(
                    tmp, "capture-TP-%d-DECODE.trace.json.gz" % rank)
                with gzip.open(path, "wt") as fh:
                    json.dump({"traceEvents": events}, fh)

            result = trace_capability.build_manifest(tmp, auto_select_rank=True)

            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["selected_analysis_rank"])
            self.assertEqual(result["analysis_rank_trace"], "")

    def test_retry_uses_best_single_step_event_count_instead_of_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            for step in range(3):
                start = step * 7000
                events.append({
                    "cat": "gpu_user_annotation",
                    "name": "step[DECODE bs=4]",
                    "ts": start,
                    "dur": 6000,
                })
                events.append({
                    "cat": "kernel", "name": "kernel_%d" % step,
                    "ts": start + 10, "dur": 1, "args": {"stream": 1},
                })
            path = os.path.join(tmp, "capture-TP-0-DECODE.trace.json.gz")
            with gzip.open(path, "wt") as fh:
                json.dump({"traceEvents": events}, fh)

            result = trace_capability.build_manifest(tmp, auto_select_rank=True)

            candidate = result["rank_candidates"][0]
            self.assertEqual(candidate["decode_duration_us"], 6000)
            self.assertEqual(candidate["decode_device_events"], 1)
            self.assertTrue(candidate["eligible"])

    def test_equal_maximum_selects_lowest_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank in (2, 1):
                events = [{
                    "cat": "gpu_user_annotation",
                    "name": "step[DECODE bs=4]",
                    "ts": 100,
                    "dur": 100,
                }, {
                    "cat": "kernel", "name": "kernel",
                    "ts": 110, "dur": 1, "args": {"stream": 1},
                }]
                path = os.path.join(
                    tmp, "capture-TP-%d-DECODE.trace.json.gz" % rank)
                with gzip.open(path, "wt") as fh:
                    json.dump({"traceEvents": events}, fh)

            result = trace_capability.build_manifest(tmp, auto_select_rank=True)

            self.assertEqual(result["selected_analysis_rank"], 1)


if __name__ == "__main__":
    unittest.main()
