"""Speculative decoding (EAGLE/NEXTN/MTP) trace recognition for KernelFusion.

With speculative decoding SGLang runs the target model's decode as
`step[TARGET_VERIFY bs=N]`, labels the draft-model prefill `step[EXTEND ...]`
(indistinguishable by name from the target prefill), and replays draft decode
through unannotated draft CUDA graphs.  These tests pin that the fusion trace
manifest and semantic mapping treat TARGET_VERIFY as decode and keep every
draft-model step out of the main-layer tables.
"""
import gzip
import json
import os
import sys
import tempfile
import types
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_topk_harness as topk
import semantic_kernel_mapping as mapping
import semantic_layer_boundary_transfer as boundary
import semantic_runtime_capture as capture
import semantic_runtime_marker_mapping as marker_mapping
import sglang_step_modes
import trace_capability


SPEC = "/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py"


def _frame(func, line, ts, dur, tid=1):
    return {"ph": "X", "cat": "python_function", "tid": tid, "ts": ts,
            "dur": dur, "name": "%s(%d): %s" % (SPEC, line, func)}


def _step(name, cpu_ts, cpu_dur, gpu_ts, gpu_dur, tid=1):
    return [
        {"ph": "X", "cat": "user_annotation", "tid": tid, "name": name,
         "ts": cpu_ts, "dur": cpu_dur},
        {"ph": "X", "cat": "gpu_user_annotation", "name": name,
         "ts": gpu_ts, "dur": gpu_dur},
    ]


def _layers(prefix, ts, count, layer_class="Qwen3_5LinearDecoderLayer",
            external_base=0, tid=1):
    """`count` module-spanned layers, each with one cpu_op + one kernel."""
    events = []
    for index in range(count):
        start = ts + index * 100
        ext = external_base + index
        events.extend([
            {"ph": "X", "cat": "python_function", "tid": tid,
             "name": "nn.Module: %s_%d" % (layer_class, index),
             "ts": start, "dur": 50},
            {"ph": "X", "cat": "cpu_op", "tid": tid, "name": "aten::mm",
             "ts": start + 5, "dur": 5,
             "args": {"External id": ext, "Input Dims": [[4, 8], [8, 8]],
                      "Input type": ["c10::BFloat16", "c10::BFloat16"]}},
            {"ph": "X", "cat": "kernel",
             "name": "%s_gemm_kernel_%d" % (prefix, index),
             "ts": start + 20, "dur": 3,
             "args": {"External id": ext, "stream": 7}},
        ])
    return events


class StepModeParsingTest(unittest.TestCase):
    def test_modes_map_to_model_phases(self):
        cases = {
            "step[EXTEND bs=1 toks=1024]": ("EXTEND", "prefill", 1, 1024),
            "step[DECODE bs=64]": ("DECODE", "decode", 64, 0),
            "step[TARGET_VERIFY bs=8]": ("TARGET_VERIFY", "verify", 8, 0),
            "step[DRAFT_EXTEND bs=8]": ("DRAFT_EXTEND", "draft", 8, 0),
            "step[DRAFT_EXTEND_V2 bs=8]": ("DRAFT_EXTEND_V2", "draft", 8, 0),
            "step[MIXED bs=4]": ("MIXED", "prefill", 4, 0),
        }
        for name, (mode, phase, bs, toks) in cases.items():
            step = sglang_step_modes.parse_step(name)
            self.assertIsNotNone(step, name)
            self.assertEqual(
                (step["mode"], step["phase"], step["batch_size"],
                 step["tokens"]), (mode, phase, bs, toks))
        self.assertTrue(
            sglang_step_modes.parse_step("step[TARGET_VERIFY bs=8]")[
                "speculative"])
        self.assertIsNone(sglang_step_modes.parse_step("step[IDLE bs=0]"))
        self.assertIsNone(sglang_step_modes.parse_step("execute_context_1"))

    def test_phase_aliases_accept_target_verify(self):
        self.assertEqual(sglang_step_modes.canonical_phase("TARGET_VERIFY"),
                         "verify")
        self.assertTrue(sglang_step_modes.is_generation_phase("verify"))
        self.assertTrue(sglang_step_modes.is_generation_phase("decode"))
        self.assertFalse(sglang_step_modes.is_generation_phase("prefill"))
        self.assertEqual(sglang_step_modes.target_phases(["prefill", "verify"]),
                         ["prefill", "verify"])
        self.assertEqual(sglang_step_modes.target_phases(["prefill", "decode"]),
                         ["prefill", "decode"])
        self.assertEqual(sglang_step_modes.canonical_phase("draft_extend"),
                         "draft")
        self.assertEqual(boundary._phase("TARGET_VERIFY"), "verify")
        self.assertEqual(marker_mapping._phase("TARGET_VERIFY"), "verify")


class TraceManifestTest(unittest.TestCase):
    def test_target_verify_decode_trace_is_selected(self):
        """The Qwen3.5-27B NEXTN run failed with EXIT 2 exactly here."""
        with tempfile.TemporaryDirectory() as tmp:
            decode = [
                {"cat": "gpu_user_annotation",
                 "name": "step[TARGET_VERIFY bs=8]", "ts": 100, "dur": 100},
                # Draft decode replays outside any step annotation.
                {"cat": "kernel", "name": "draft_kernel", "ts": 50, "dur": 1},
            ]
            decode.extend({"cat": "kernel", "name": "verify_%d" % index,
                           "ts": 110 + index, "dur": 1}
                          for index in range(4))
            extend = [{"cat": "gpu_user_annotation",
                       "name": "step[EXTEND bs=1 toks=16]",
                       "ts": 0, "dur": 10}]
            for phase, events in (("DECODE", decode), ("EXTEND", extend)):
                path = os.path.join(
                    tmp, "1.0-TP-0-%s.trace.json.gz" % phase)
                with gzip.open(path, "wt") as fh:
                    json.dump({"traceEvents": events}, fh)

            result = trace_capability.build_manifest(
                tmp, auto_select_rank=True)

            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["analysis_rank_trace"].endswith(
                "TP-0-DECODE.trace.json.gz"))
            self.assertEqual(result["rank_candidates"][0]["reason"],
                             "decode_device_events_match_rank_maximum")
            self.assertEqual(
                result["rank_candidates"][0]["decode_device_events"], 4)
            self.assertTrue(result["speculative_decoding"])
            self.assertEqual(result["decode_step_modes"], ["TARGET_VERIFY"])
            self.assertEqual(result["generation_phase"], "verify")
            self.assertEqual(result["target_phases"], ["prefill", "verify"])
            self.assertEqual(result["speculative"]["enabled"], True)
            self.assertEqual(result["speculative"]["evidence"],
                             "trace:TARGET_VERIFY")
            self.assertTrue(result["capability"]["speculative_decoding"])
            self.assertEqual(
                result["capability"]["step_mode_counts"],
                {"TARGET_VERIFY": 1})

    def test_draft_extend_does_not_compete_for_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = [{"cat": "gpu_user_annotation",
                       "name": "step[DRAFT_EXTEND bs=8]",
                       "ts": 0, "dur": 100}]
            events.extend({"cat": "kernel", "name": "draft_%d" % index,
                           "ts": 10 + index, "dur": 1}
                          for index in range(5))
            path = os.path.join(tmp, "1.0-TP-0-DECODE.trace.json.gz")
            with gzip.open(path, "wt") as fh:
                json.dump({"traceEvents": events}, fh)
            result = trace_capability.build_manifest(
                tmp, auto_select_rank=True)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["rank_candidates"][0]["reason"],
                             "missing_decode_annotation")


class SemanticStepSpanTest(unittest.TestCase):
    def test_draft_prefill_labelled_extend_is_excluded_by_stack(self):
        """SGLang names the draft prefill `step[EXTEND bs=1 toks=N]` too."""
        name = "step[EXTEND bs=1 toks=16]"
        events = []
        events.append(_frame("forward_batch_generation", 701, 0, 3000))
        events.extend(_step(name, 10, 900, 20, 900))
        events.append(_frame("_draft_extend_for_prefill", 502, 1000, 500))
        events.extend(_step(name, 1010, 400, 1020, 400))
        spans, drafts = mapping._collect_step_spans_with_draft(events)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0][2], "P")
        self.assertEqual(spans[0][0], 20)
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["ts"], 1020)
        self.assertEqual(drafts[0]["evidence"],
                         "speculative_stack_frame:_draft_extend_for_prefill")

    def test_target_verify_is_decode_and_draft_mode_is_excluded(self):
        events = []
        events.append(_frame("verify", 759, 0, 1000))
        events.extend(_step("step[TARGET_VERIFY bs=8]", 10, 50, 100, 300))
        events.extend(_step("step[DRAFT_EXTEND bs=8]", 1100, 50, 1200, 100))
        spans, drafts = mapping._collect_step_spans_with_draft(events)
        self.assertEqual([span[2] for span in spans], ["V"])
        self.assertEqual(spans[0][4], 8)
        self.assertEqual(spans[0][6], "sglang_step_annotation:TARGET_VERIFY")
        self.assertEqual([item["evidence"] for item in drafts],
                         ["draft_step_mode"])

    def test_speculative_trace_builds_prefill_and_decode_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "schema_version": 1,
                    "num_hidden_layers_main": 2,
                    "patterns": [{
                        "pattern_id": "P_LINEAR_ATTENTION",
                        "pattern_display_name": "Linear",
                        "layer_ids": [0, 1]}],
                    "coverage_check": {
                        "total_main_layers": 2, "covered": 2,
                        "mutually_exclusive": True, "full_coverage": True},
                    "quality": {"status": "pass"},
                }, fh)
            name = "step[EXTEND bs=1 toks=16]"
            extend = [_frame("forward_batch_generation", 701, 0, 2000)]
            # Target prefill: 2 main layers.
            extend.extend(_step(name, 10, 400, 10, 400))
            extend.extend(_layers("target_prefill", 20, 2))
            # Draft (MTP) prefill: 1 layer, same step name, draft frame.
            extend.append(_frame("_draft_extend_for_prefill", 502, 1000, 400))
            extend.extend(_step(name, 1010, 300, 1010, 300))
            extend.extend(_layers("draft_prefill", 1020, 1,
                                  external_base=100))
            decode = [_frame("verify", 759, 5000, 1000)]
            decode.extend(_step("step[TARGET_VERIFY bs=8]",
                                5010, 400, 5010, 400))
            decode.extend(_layers("target_verify", 5020, 2,
                                  external_base=200))
            # Draft decode: unannotated draft CUDA-graph replay.
            decode.append({"ph": "X", "cat": "kernel",
                           "name": "draft_decode_kernel",
                           "ts": 6500, "dur": 2, "args": {"stream": 7}})
            traces = {}
            for phase, events in (("EXTEND", extend), ("DECODE", decode)):
                traces[phase] = os.path.join(
                    tmp, "1.0-TP-0-%s.trace.json.gz" % phase)
                with gzip.open(traces[phase], "wt") as fh:
                    json.dump({"traceEvents": events}, fh)

            result = mapping.build(
                traces["DECODE"], patterns, os.path.join(tmp, "out"),
                require_phases=["prefill", "decode"])

            with open(result["quality_json"]) as fh:
                quality = json.load(fh)
            with open(result["semantic_table_json"]) as fh:
                table = json.load(fh)
            coverage = quality["phase_coverage"]
            # "decode" was requested (legacy default); a speculative trace
            # satisfies it with its generation phase, verify.
            self.assertEqual(coverage["phases_in_tables"],
                             ["prefill", "verify"])
            self.assertEqual(coverage["generation_phase"], "verify")
            self.assertEqual(coverage["required_phases"], ["prefill", "verify"])
            self.assertEqual(coverage["missing_required_phases"], [])
            self.assertTrue(coverage["speculative_decoding"])
            self.assertEqual(coverage["decode_step_modes"], ["TARGET_VERIFY"])
            self.assertEqual(coverage["draft_steps_excluded"], 1)
            names = {row["raw_name"] for item in table["tables"]
                     for row in item["rows"]}
            self.assertFalse(any(value.startswith("draft_")
                                 for value in names), names)
            self.assertIn("target_verify_gemm_kernel_0", names)
            with open(result["semantic_event_audit_jsonl"]) as fh:
                rows = [json.loads(line) for line in fh]
            self.assertFalse(any(row["raw_name"].startswith("draft_")
                                 for row in rows))


class _Mode(object):
    """Mimics sglang ForwardMode: is_extend() is True for speculative modes."""

    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "DECODE"

    def is_target_verify(self):
        return self.name == "TARGET_VERIFY"

    def is_draft_extend(self):
        return self.name in ("DRAFT_EXTEND", "DRAFT_EXTEND_V2")

    def is_extend(self):
        return self.name in (
            "EXTEND", "MIXED", "DRAFT_EXTEND", "TARGET_VERIFY")

    def is_prefill(self):
        return self.is_extend()


class RuntimeCaptureSpeculativeTest(unittest.TestCase):
    def test_phase_of_prefers_speculative_modes(self):
        for name, expected in (("TARGET_VERIFY", "TARGET_VERIFY"),
                               ("DRAFT_EXTEND", "DRAFT_EXTEND"),
                               ("EXTEND", "EXTEND"),
                               ("DECODE", "DECODE")):
            batch = types.SimpleNamespace(
                forward_mode=_Mode(name), batch_size=8, input_ids=None)
            self.assertEqual(capture._phase_of(batch)[0], expected, name)

    def test_canonical_phase_admits_verify_as_decode(self):
        self.assertEqual(capture._canonical_phase("TARGET_VERIFY"), "VERIFY")
        self.assertEqual(capture._canonical_phase("verify"), "VERIFY")
        self.assertEqual(capture._canonical_phase("DECODE"), "DECODE")
        self.assertEqual(capture._canonical_phase("DRAFT_EXTEND"), "DRAFT")

    def test_draft_model_is_not_instrumented(self):
        class Module(object):
            def __init__(self, children=()):
                self._children = list(children)

            def modules(self):
                yield self
                for child in self._children:
                    yield child

        draft = Module([Module(), Module()])
        logger = types.SimpleNamespace(active=lambda: True)
        original = (capture.install_graph_capture_profiler,
                    capture.dump_operator_schema_manifest,
                    capture.get_logger)
        try:
            capture.install_graph_capture_profiler = lambda: None
            capture.dump_operator_schema_manifest = lambda: None
            capture.get_logger = lambda: logger
            self.assertIs(capture.install_on_model(draft, is_draft=True),
                          draft)
        finally:
            (capture.install_graph_capture_profiler,
             capture.dump_operator_schema_manifest,
             capture.get_logger) = original
        self.assertTrue(all(getattr(module, "_geak_semantics_draft", False)
                            for module in draft.modules()))
        self.assertNotIn(Module, capture._INSTALLED_CLASSES)


class SpeculativeStageRuleTest(unittest.TestCase):
    def test_target_verify_attention_and_gdn_kernels_have_stages(self):
        """TARGET_VERIFY runs SGLang's triton extend attention (`_fwd_kernel`)
        and the GDN recurrent update; as `unknown` they were non-donor and a
        fusible region spanned the attention body."""
        for name in ("_fwd_kernel", "_fwd_kernel_stage1", "_fwd_kernel_stage2"):
            self.assertEqual(mapping._stage(name, "kernel"), "attn", name)
        self.assertEqual(mapping._stage(
            "fused_sigmoid_gating_delta_rule_update_kernel", "kernel"),
            "linear_attn")
        # Unrelated *_fwd_kernel* names keep their own stages.
        self.assertEqual(mapping._stage("chunk_fwd_kernel_o", "kernel"),
                         "linear_attn")
        self.assertEqual(mapping._stage("_layer_norm_fwd_1pass_kernel",
                                        "kernel"), "norm")
        self.assertEqual(mapping._stage("l2norm_fwd_kernel", "kernel"), "norm")


class DraftGraphRunnerProfilerTest(unittest.TestCase):
    def test_draft_runners_get_noop_profile_hooks(self):
        """--enable-profile-cuda-graph + NEXTN crashed at server start:
        EAGLEDraftCudaGraphRunner borrows CudaGraphRunner.capture without
        the profile hooks."""
        classes = {}
        saved = {}
        for module_name, class_name in capture._DRAFT_GRAPH_RUNNERS:
            classes[class_name] = type(class_name, (object,), {})
            saved[module_name] = sys.modules.get(module_name)
            sys.modules[module_name] = types.SimpleNamespace(
                **{class_name: classes[class_name]})
        try:
            capture._install_draft_graph_profiler_noop()
        finally:
            for module_name, module in saved.items():
                if module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = module
        for cls in classes.values():
            runner = cls()
            with runner._init_profile_context_and_memory_record() as prof:
                self.assertIsNone(prof)
            self.assertIsNone(runner._post_process_after_profile(None))


class TopKSpeculativeWeightTest(unittest.TestCase):
    def test_decode_weight_uses_accept_length(self):
        info, weights, total = topk._workload_model(
            {"isl": 1024, "osl": 1024, "conc": 64,
             "spec_accept_length": 3.7},
            {"prefill": 100.0, "decode": 10.0})
        self.assertEqual(weights["decode"], 277)  # ceil(1023 / 3.7)
        self.assertTrue(info["speculative_decoding"])
        self.assertEqual(total, 100.0 + 2770.0)

    def test_unknown_accept_length_is_flagged_upper_bound(self):
        info, weights, _ = topk._workload_model(
            {"isl": 1024, "osl": 1024, "speculative_decoding": True},
            {"prefill": 1.0, "decode": 1.0})
        self.assertEqual(weights["decode"], 1023)
        self.assertIn("upper bound", info["decode_forward_model"])

    def test_verify_phase_is_weighted_by_rounds(self):
        info, weights, _ = topk._workload_model(
            {"isl": 1024, "osl": 1024, "spec_accept_length": 3.7},
            {"prefill": 1.0, "verify": 1.0})
        self.assertEqual(weights, {"prefill": 1, "verify": 277})
        self.assertEqual(info["generation_phase"], "verify")

    def test_plain_decode_is_unchanged(self):
        info, weights, _ = topk._workload_model(
            {"isl": 8, "osl": 4}, {"prefill": 1.0, "decode": 1.0})
        self.assertEqual(weights["decode"], 3)
        self.assertNotIn("speculative_decoding", info)



class ServerLogFactsTest(unittest.TestCase):
    """The trace decides speculative on/off; the server log only enriches."""

    LOG = (
        "[2026-09-27 11:01:00] server_args=ServerArgs(model_path='m', "
        "speculative_algorithm='EAGLE', speculative_num_steps=3, "
        "speculative_eagle_topk=1, speculative_num_draft_tokens=4, "
        "max_running_requests=64)\n"
        "[2026-09-27 11:02:00] Decode batch, #running-req: 8, accept len: 2.00, x\n"
        "[2026-09-27 11:02:02] Decode batch, #running-req: 64, accept len: 3.60, x\n"
        "[2026-09-27 11:02:04] Decode batch, #running-req: 63, accept len: 3.80, x\n")

    def _trace_dir(self, tmp, mode="TARGET_VERIFY"):
        events = [{"cat": "gpu_user_annotation", "name": "step[%s bs=64]" % mode,
                   "ts": 100, "dur": 100},
                  {"cat": "kernel", "name": "k", "ts": 110, "dur": 1}]
        path = os.path.join(tmp, "1.0-TP-0-DECODE.trace.json.gz")
        with gzip.open(path, "wt") as fh:
            json.dump({"traceEvents": events}, fh)
        return tmp

    def test_log_enriches_verify_tokens_and_accept_len(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "server.log")
            with open(log, "w") as fh:
                fh.write(self.LOG)
            spec = trace_capability.build_manifest(
                self._trace_dir(tmp), auto_select_rank=True,
                server_log=log)["speculative"]
            self.assertTrue(spec["enabled"])
            self.assertEqual(spec["verify_tokens_per_request"], 4)
            self.assertEqual(spec["algorithm_runtime"], "EAGLE")
            # steady rows only (running >= 0.9 x max seen): 3.6 and 3.8
            self.assertAlmostEqual(spec["accept_len_mean"], 3.7)
            self.assertEqual(spec["warnings"], [])

    def test_missing_or_malformed_log_only_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = self._trace_dir(tmp)
            for log in (os.path.join(tmp, "absent.log"), None):
                doc = trace_capability.build_manifest(
                    trace_dir, auto_select_rank=True, server_log=log)
                self.assertEqual(doc["status"], "pass")
                self.assertTrue(doc["speculative"]["enabled"])
                self.assertIsNone(doc["speculative"]["accept_len_mean"])
            bad = os.path.join(tmp, "bad.log")
            with open(bad, "w") as fh:
                fh.write("garbage\nDecode batch, accept len: x\n")
            doc = trace_capability.build_manifest(
                trace_dir, auto_select_rank=True, server_log=bad)
            self.assertEqual(doc["status"], "pass")
            self.assertTrue(doc["speculative"]["warnings"])

    def test_log_disagreeing_with_trace_is_a_warning_not_a_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "server.log")
            with open(log, "w") as fh:
                fh.write(self.LOG)
            doc = trace_capability.build_manifest(
                self._trace_dir(tmp, mode="DECODE"), auto_select_rank=True,
                server_log=log)
            self.assertEqual(doc["status"], "pass")
            self.assertFalse(doc["speculative"]["enabled"])
            self.assertEqual(doc["generation_phase"], "decode")
            self.assertTrue(any("trace" in w for w in doc["speculative"]["warnings"]))


class AnalysisStepSelectionTest(unittest.TestCase):
    """Several captured steps: every table of a phase comes from ONE step."""

    def test_rule_prefers_workload_bucket_then_median_duration(self):
        # (start, end, tag, tokens, bs, step_id, source)
        spans = [
            (0, 200, "P", 1024, 1, "p-small", "sglang_step_annotation"),
            (300, 2300, "P", 16384, 16, "p-big", "sglang_step_annotation"),
            (3000, 3060, "V", 0, 64, "v-a", "sglang_step_annotation:TARGET_VERIFY"),
            (3100, 3150, "V", 0, 64, "v-b", "sglang_step_annotation:TARGET_VERIFY"),
            (3200, 3270, "V", 0, 64, "v-c", "sglang_step_annotation:TARGET_VERIFY"),
            (3300, 3310, "V", 0, 8, "v-small", "sglang_step_annotation:TARGET_VERIFY"),
        ]
        chosen = mapping._select_analysis_steps(spans)
        self.assertEqual(chosen["prefill"]["step_id"], "p-big")
        self.assertEqual(chosen["prefill"]["input_tokens"], 16384)
        # three bs=64 steps: 60/50/70 us -> median 60 us is v-a
        self.assertEqual(chosen["verify"]["step_id"], "v-a")
        self.assertEqual(chosen["verify"]["batch_size"], 64)
        self.assertEqual(len(chosen["verify"]["same_bucket_step_ids"]), 3)
        self.assertIn("v-small", chosen["verify"]["other_bucket_step_ids"])

    def test_all_pattern_tables_come_from_the_selected_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "schema_version": 1, "num_hidden_layers_main": 2,
                    "patterns": [
                        {"pattern_id": "P0", "pattern_display_name": "A",
                         "layer_ids": [0]},
                        {"pattern_id": "P1", "pattern_display_name": "B",
                         "layer_ids": [1]}],
                    "coverage_check": {"total_main_layers": 2, "covered": 2,
                                       "mutually_exclusive": True,
                                       "full_coverage": True},
                    "quality": {"status": "pass"}}, fh)
            events = []
            # small prefill (bs=1, 16 tok) then the workload prefill (bs=4, 64 tok)
            events.extend(_step("step[EXTEND bs=1 toks=16]", 0, 400, 0, 400))
            events.extend(_layers("small", 10, 2))
            events.extend(_step("step[EXTEND bs=4 toks=64]", 1000, 400, 1000, 400))
            events.extend(_layers("big", 1010, 2, external_base=100))
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"),
                                   require_phases=["prefill"])
            with open(result["semantic_table_json"]) as fh:
                table = json.load(fh)
            buckets = {tb["pattern_id"]: tb["selected_bucket"]
                       for tb in table["tables"]}
            self.assertEqual(set(buckets), {"P0", "P1"})
            for bucket in buckets.values():
                self.assertEqual((bucket["batch_size"], bucket["input_tokens"]),
                                 (4, 64))
            names = {row["raw_name"] for tb in table["tables"] for row in tb["rows"]}
            self.assertTrue(all(name.startswith("big_") for name in names), names)
            with open(result["quality_json"]) as fh:
                quality = json.load(fh)
            chosen = quality["phase_coverage"]["analysis_steps"]["prefill"]
            self.assertEqual((chosen["batch_size"], chosen["input_tokens"]), (4, 64))


class BoundaryBucketPadTest(unittest.TestCase):
    def test_generation_step_matches_padded_graph_bucket(self):
        donors = [{"phase": "verify", "batch_size": bs, "input_tokens": 4 * bs}
                  for bs in (8, 10, 12)]
        # clean verify step bs=9 replays the bs=10 graph
        picked = boundary._compatible_donors(donors, "verify", 9, 0)
        self.assertEqual([d["batch_size"] for d in picked], [10])
        exact = boundary._compatible_donors(donors, "verify", 12, 0)
        self.assertEqual([d["batch_size"] for d in exact], [12])
        self.assertEqual(boundary._compatible_donors(donors, "verify", 13, 0), [])
        # prefill keeps exact batch + token matching
        prefill = [{"phase": "prefill", "batch_size": 1, "input_tokens": 16}]
        self.assertEqual(boundary._compatible_donors(prefill, "prefill", 1, 32), [])


class TopKAutoAcceptLengthTest(unittest.TestCase):
    TABLE = {"phase_coverage": {"generation_phase": "verify",
                                "speculative_decoding": True,
                                "speculative": {"enabled": True,
                                                "accept_len_mean": 3.67}}}

    def test_accept_length_is_read_from_the_semantic_table(self):
        workload = topk._speculative_workload({"isl": 1024, "osl": 1024}, self.TABLE)
        self.assertEqual(workload["spec_accept_length"], 3.67)
        self.assertEqual(workload["spec_accept_length_source"],
                         "semantic_table:phase_coverage.speculative")
        _, weights, _ = topk._workload_model(workload, {"prefill": 1.0, "verify": 1.0})
        self.assertEqual(weights["verify"], 279)

    def test_explicit_value_wins_and_missing_value_is_upper_bound(self):
        workload = topk._speculative_workload(
            {"isl": 1024, "osl": 1024, "spec_accept_length": 3.0}, self.TABLE)
        self.assertEqual(workload["spec_accept_length"], 3.0)
        bare = {"phase_coverage": {"generation_phase": "verify",
                                   "speculative_decoding": True}}
        workload = topk._speculative_workload({"isl": 1024, "osl": 1024}, bare)
        info, weights, _ = topk._workload_model(workload, {"prefill": 1.0, "verify": 1.0})
        self.assertEqual(weights["verify"], 1023)
        self.assertIn("upper bound", info["decode_forward_model"])


class DraftModuleLeakTest(unittest.TestCase):
    """The draft layer's python span must not become a 65th target layer."""

    def test_draft_module_span_inside_long_target_gpu_window_is_ignored(self):
        pattern_doc = {"num_hidden_layers_main": 2, "patterns": [
            {"pattern_id": "P0", "layer_ids": [0, 1]}]}
        name = "step[EXTEND bs=1 toks=16]"
        events = [_frame("forward_batch_generation", 701, 0, 450)]
        # target: CPU 10..410, GPU 10..2000 (GPU runs long after CPU returns)
        events.extend(_step(name, 10, 400, 10, 1990))
        events.extend(_layers("target", 20, 2))
        # draft prefill dispatched by the CPU while the target GPU work runs
        events.append(_frame("_draft_extend_for_prefill", 502, 500, 400))
        events.extend(_step(name, 510, 380, 2010, 90))
        events.append({"ph": "X", "cat": "python_function", "tid": 1,
                       "name": "nn.Module: Qwen3_5AttentionDecoderLayer_0",
                       "ts": 600, "dur": 50})
        _, spans, _, scopes, diagnostics = mapping._event_rows(events, pattern_doc)
        self.assertEqual(len(spans), 1)
        self.assertEqual(len(scopes), 2)
        self.assertEqual([(d["candidate_count"], d["remainder"]) for d in diagnostics],
                         [(2, 0)])


class TransferNonAnalysisFailureTest(unittest.TestCase):
    def test_only_analysis_step_failures_block_the_transfer(self):
        failures = [{"step_id": "s-other", "reason": "no_validated_stable_projection"},
                    {"step_id": "s-main", "reason": "ambiguous"},
                    {"reason": "donor_has_no_geak_layer_scope_markers"}]
        blocking, ignored = boundary._split_step_failures(failures, {"s-main"})
        self.assertEqual([f.get("step_id") for f in blocking], ["s-main", None])
        self.assertEqual([f["step_id"] for f in ignored], ["s-other"])


class RuntimePhaseFilterTest(unittest.TestCase):
    def test_decode_request_admits_verify_and_vice_versa(self):
        self.assertTrue(capture._phase_requested("TARGET_VERIFY", {"DECODE"}))
        self.assertTrue(capture._phase_requested("DECODE", {"VERIFY"}))
        self.assertTrue(capture._phase_requested("EXTEND", {"PREFILL"}))
        self.assertFalse(capture._phase_requested("EXTEND", {"DECODE"}))
        self.assertFalse(capture._phase_requested("DRAFT_EXTEND", {"DECODE", "PREFILL"}))
        self.assertTrue(capture._phase_requested("DRAFT_EXTEND", set()))


class AnalysisStepGateTest(unittest.TestCase):
    def test_incomplete_non_analysis_step_does_not_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({"schema_version": 1, "num_hidden_layers_main": 2,
                           "patterns": [{"pattern_id": "P0",
                                         "pattern_display_name": "A",
                                         "layer_ids": [0, 1]}],
                           "coverage_check": {"total_main_layers": 2, "covered": 2,
                                              "mutually_exclusive": True,
                                              "full_coverage": True},
                           "quality": {"status": "pass"}}, fh)
            events = []
            # profiler armed mid-step: only 1 of 2 layer spans captured
            events.extend(_step("step[EXTEND bs=1 toks=16]", 0, 400, 0, 400))
            events.extend(_layers("partial", 110, 1))
            events.extend(_step("step[EXTEND bs=4 toks=64]", 1000, 400, 1000, 400))
            events.extend(_layers("full", 1010, 2, external_base=100))
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"),
                                   require_phases=["prefill"])
            with open(result["quality_json"]) as fh:
                quality = json.load(fh)
            gate = quality["gates"]["step_layer_order"]
            self.assertEqual(gate["scope"], "analysis_steps")
            self.assertEqual(gate["status"], "pass")
            self.assertEqual(len(gate["non_gating_steps"]), 1)
            self.assertEqual(quality["status"], "pass")


if __name__ == "__main__":
    unittest.main()
