#!/usr/bin/env python3
"""Regression tests for machine-verified live kernel selection."""

import importlib.util
import json
import os
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location(
    "kernel_selection", os.path.join(SCRIPTS, "kernel_selection.py"))
ks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ks)

TARGET = (
    "vllm.v1.attention.ops.chunked_prefill_paged_decode:"
    "chunked_prefill_paged_decode"
)
KERNEL = "kernel_paged_attention_2d"


def trace(kernel=KERNEL, under_target=True):
    marker_start = 100
    kernel_start = 120 if under_target else 250
    events = [
        {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
         "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
        {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
         "ph": "X", "pid": 1, "tid": 2, "ts": marker_start, "dur": 100},
        {"cat": "kernel", "name": f"void vllm::{kernel}<bf16>(int)",
         "ph": "X", "ts": kernel_start, "dur": 25, "args": {"External id": 7}},
    ]
    if under_target:
        events.insert(1, {"cat": "cpu_op", "name": "launch", "ph": "X",
                          "pid": 1, "tid": 2, "ts": 110, "dur": 5,
                          "args": {"External id": 7}})
    return events


class TestCallableSpec(unittest.TestCase):
    def test_only_exact_machine_specs_are_accepted(self):
        self.assertTrue(ks.valid_callable_spec(TARGET))
        self.assertFalse(ks.valid_callable_spec(TARGET + " -> inner kernel"))
        self.assertFalse(ks.valid_callable_spec("vllm/path.py:launcher"))


class TestKernelMatching(unittest.TestCase):
    def test_demangled_kernel_matches_profile_identity(self):
        self.assertTrue(ks.kernel_matches(
            KERNEL, "void vllm::kernel_paged_attention_2d<bf16>(int)"))
        self.assertFalse(ks.kernel_matches(KERNEL, "unrelated_attention_kernel"))


class TestSelectionVerdict(unittest.TestCase):
    def meta(self, calls=7, target=TARGET):
        module, attr = target.split(":", 1)
        return {"module": module, "attr": attr, "total_calls_observed": calls}

    def test_0720_live_inner_launcher_is_selection_success(self):
        verdict = ks.verify(TARGET, KERNEL, self.meta(), trace())
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["matched_kernel_calls"], 1)
        self.assertEqual(verdict["failed"], [])

    def test_kernel_seen_elsewhere_is_not_enough(self):
        verdict = ks.verify(TARGET, KERNEL, self.meta(), trace(under_target=False))
        self.assertFalse(verdict["ok"])
        self.assertIn("device_kernel_not_under_target", verdict["failed"])

    def test_async_kernel_after_marker_is_linked_by_external_id(self):
        events = [
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 40},
            {"cat": "cpu_op", "name": "launch", "ph": "X", "pid": 1, "tid": 2,
             "ts": 110, "dur": 5,
             "args": {"External id": 42}},
            {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 300, "dur": 20,
             "args": {"External id": 42}},
        ]
        verdict = ks.verify(TARGET, KERNEL, self.meta(), events)
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["correlated_external_ids"], 1)

    def test_triton_kernel_without_external_id_is_linked_by_launch_correlation(self):
        # ROCm/kineto emits Triton device rows (hipModuleLaunchKernel) with only `correlation`;
        # `External id` is absent, so the External-id bridge alone reports a false negative.
        events = [
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 40},
            {"cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "ph": "X",
             "pid": 1, "tid": 2, "ts": 110, "dur": 5, "args": {"correlation": 25}},
            {"cat": "kernel", "name": KERNEL, "ph": "X", "pid": 2, "tid": 0,
             "ts": 300, "dur": 20, "args": {"correlation": 25, "stream": 0}},
        ]
        verdict = ks.verify(TARGET, KERNEL, self.meta(), events)
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["matched_kernel_calls"], 1)
        self.assertEqual(verdict["correlated_launch_correlations"], 1)

    def test_launch_correlation_outside_the_marker_span_is_not_enough(self):
        events = [
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 40},
            {"cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "ph": "X",
             "pid": 1, "tid": 2, "ts": 500, "dur": 5, "args": {"correlation": 25}},
            {"cat": "kernel", "name": KERNEL, "ph": "X", "pid": 2, "tid": 0,
             "ts": 600, "dur": 20, "args": {"correlation": 25, "stream": 0}},
        ]
        verdict = ks.verify(TARGET, KERNEL, self.meta(), events)
        self.assertFalse(verdict["ok"])
        self.assertIn("device_kernel_not_under_target", verdict["failed"])

    def test_correlation_ids_do_not_collide_across_merged_call_traces(self):
        # Every per-call trace restarts `correlation` at 1, so an unprefixed merge would let call-2's
        # kernel be attributed to call-1's in-span launch.
        def one(marked, corr, kernel_ts):
            evs = [
                {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + TARGET,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1},
                {"cat": "kernel", "name": KERNEL, "ph": "X", "pid": 2, "tid": 0,
                 "ts": kernel_ts, "dur": 20, "args": {"correlation": corr}},
            ]
            if marked:
                evs += [
                    {"cat": "cpu_op", "name": ks.MARKER_PREFIX + TARGET,
                     "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 40},
                    {"cat": "cuda_runtime", "name": "hipModuleLaunchKernel", "ph": "X",
                     "pid": 1, "tid": 2, "ts": 110, "dur": 5, "args": {"correlation": corr}},
                ]
            return {"traceEvents": evs}

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            # call-1: marked launch, correlation 1.  call-2: NO marker at all, correlation 1 again.
            for i, doc in enumerate((one(True, 1, 300), one(False, 1, 900))):
                p = os.path.join(tmp, "t%d.json" % i)
                json.dump(doc, open(p, "w"))
                paths.append(p)
            merged = ks.merge_process_traces(paths)
            verdict = ks.verify(TARGET, KERNEL, self.meta(), merged)
            self.assertEqual(verdict["matched_kernel_calls"], 1, verdict)

    def test_0802_outer_wrapper_fails_when_0720_launcher_is_marked(self):
        outer = "vllm.v1.attention.layer:unified_attention_with_output"
        inner = TARGET
        events = [
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + outer,
             "ph": "X", "pid": 1, "tid": 2, "ts": 80, "dur": 1},
            {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + inner,
             "ph": "X", "pid": 1, "tid": 2, "ts": 82, "dur": 1},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + outer,
             "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 100},
            {"cat": "cpu_op", "name": ks.MARKER_PREFIX + inner,
             "ph": "X", "pid": 1, "tid": 2, "ts": 120, "dur": 50},
            {"cat": "cpu_op", "name": "launch", "ph": "X", "pid": 1, "tid": 2,
             "ts": 130, "dur": 5, "args": {"External id": 9}},
            {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 300, "dur": 20,
             "args": {"External id": 9}},
        ]
        outer_meta = {"module": "vllm.attention", "attr": "outer", "total_calls_observed": 2}
        outer_verdict = ks.verify(outer, KERNEL, outer_meta, events, [outer, inner])
        self.assertFalse(outer_verdict["ok"])
        self.assertIn("deeper_live_candidate_exists", outer_verdict["failed"])
        self.assertEqual(outer_verdict["deeper_live_candidates"], [inner])

        inner_verdict = ks.verify(inner, KERNEL, self.meta(), events, [outer, inner])
        self.assertTrue(inner_verdict["ok"], inner_verdict)
        self.assertTrue(inner_verdict["deepest_verified"])

    def test_every_declared_probe_candidate_must_have_an_installed_marker(self):
        missing = "vllm.attention:missing_candidate"
        verdict = ks.verify(TARGET, KERNEL, self.meta(), trace(), [TARGET, missing])
        self.assertFalse(verdict["ok"])
        self.assertIn("candidate_marker_not_installed", verdict["failed"])
        self.assertEqual(verdict["missing_candidate_markers"], [missing])

    def test_installed_but_inactive_alternative_branch_does_not_fail(self):
        alternative = "vllm.attention:prefill_only"
        events = trace()
        events.insert(1, {
            "cat": "cpu_op", "name": ks.INSTALL_PREFIX + alternative,
            "ph": "X", "pid": 1, "tid": 2, "ts": 92, "dur": 1,
        })
        verdict = ks.verify(
            TARGET, KERNEL, self.meta(), events, [TARGET, alternative])
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(
            sorted(verdict["candidate_targets_tested"]),
            sorted([TARGET, alternative]),
        )

    def test_a_probed_candidate_that_never_fired_is_reported_not_silently_dropped(self):
        # A marker rebinds one module attribute. A callable dispatched through `torch.ops.*`, or
        # reached through an alias a caller imported before installation, runs with the wrapper
        # bypassed and produces zero spans while being fully live. Reading that as "not deeper"
        # would let a shallower seam collect `deepest_verified`, so the verdict has to say which
        # candidates were probed and never observed, apart from the ones never probed at all.
        invisible = "vllm.model_executor.layers.attention.attention:unified_attention_with_output"
        unprobed = "vllm.attention:never_installed"
        events = trace()
        events.insert(1, {
            "cat": "cpu_op", "name": ks.INSTALL_PREFIX + invisible,
            "ph": "X", "pid": 1, "tid": 2, "ts": 92, "dur": 1,
        })
        verdict = ks.verify(
            TARGET, KERNEL, self.meta(), events, [TARGET, invisible, unprobed])
        self.assertEqual(verdict["installed_but_never_live_candidates"], [invisible])
        self.assertEqual(verdict["missing_candidate_markers"], [unprobed])
        self.assertNotIn(invisible, verdict["deeper_live_candidates"])

    def test_device_projected_annotation_does_not_invert_call_nesting(self):
        # The profiler re-emits each marker on the GPU timeline as `gpu_user_annotation`, where an
        # OUTER seam's short device span can land inside the INNER launcher's long one. Only the
        # host spans describe the real call nesting; the selected inner launcher must still pass.
        outer = "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe:_fused_moe_kernel_sequence"
        events = trace()
        events.insert(1, {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + outer,
                          "ph": "X", "pid": 1, "tid": 2, "ts": 92, "dur": 1})
        events.insert(2, {"cat": "cpu_op", "name": ks.MARKER_PREFIX + outer,
                          "ph": "X", "pid": 1, "tid": 2, "ts": 95, "dur": 200})
        # device projections: outer's is a tiny span inside the target's long one, same pid/tid
        events += [
            {"cat": "gpu_user_annotation", "name": ks.MARKER_PREFIX + TARGET,
             "ph": "X", "pid": 9, "tid": 9, "ts": 500, "dur": 400},
            {"cat": "gpu_user_annotation", "name": ks.MARKER_PREFIX + outer,
             "ph": "X", "pid": 9, "tid": 9, "ts": 600, "dur": 5},
        ]
        verdict = ks.verify(TARGET, KERNEL, self.meta(), events, [TARGET, outer])
        self.assertEqual(verdict["deeper_live_candidates"], [])
        self.assertTrue(verdict["ok"], verdict)
        self.assertTrue(verdict["deepest_verified"])

    def test_capture_of_a_different_callable_is_rejected(self):
        wrong = "vllm.model_executor.layers.attention.attention:outer_wrapper"
        verdict = ks.verify(TARGET, KERNEL, self.meta(target=wrong), trace())
        self.assertFalse(verdict["ok"])
        self.assertIn("capture_target_mismatch", verdict["failed"])

    def test_zero_live_calls_is_rejected(self):
        verdict = ks.verify(TARGET, KERNEL, self.meta(calls=0), trace())
        self.assertFalse(verdict["ok"])
        self.assertIn("target_not_observed", verdict["failed"])

    def test_cli_writes_the_same_machine_verdict(self):
        with tempfile.TemporaryDirectory() as root:
            meta_path = os.path.join(root, "meta.json")
            trace_path = os.path.join(root, "trace.json")
            out_path = os.path.join(root, "selection.json")
            with open(meta_path, "w") as fh:
                json.dump(self.meta(), fh)
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": trace()}, fh)
            rc = ks.main([
                "--target", TARGET,
                "--device-kernel", KERNEL,
                "--capture-meta", meta_path,
                "--torch-trace", trace_path,
                "--out", out_path,
            ])
            self.assertEqual(rc, 0)
            with open(out_path) as fh:
                self.assertTrue(json.load(fh)["ok"])

    def test_cli_fails_closed_when_capture_pid_has_no_trace(self):
        with tempfile.TemporaryDirectory() as root:
            meta_path = os.path.join(root, "capture.pid-111.rank-0", "meta.json")
            os.makedirs(os.path.dirname(meta_path))
            trace_path = os.path.join(root, "selection.pid-999.rank-0.json")
            out_path = os.path.join(root, "selection.json")
            meta = self.meta()
            meta["process_id"] = 111
            with open(meta_path, "w") as fh:
                json.dump(meta, fh)
            with open(trace_path, "w") as fh:
                json.dump({"traceEvents": trace()}, fh)
            rc = ks.main([
                "--target", TARGET,
                "--device-kernel", KERNEL,
                "--capture-meta", meta_path,
                "--torch-trace", trace_path,
                "--out", out_path,
            ])
            self.assertEqual(rc, 1)
            with open(out_path) as fh:
                verdict = json.load(fh)
            self.assertFalse(verdict["ok"])
            self.assertIn("capture_process_trace_missing", verdict["failed"])
            self.assertEqual(verdict["trace_file"], "")

    def test_cli_merges_calls_before_deciding_deepest_candidate(self):
        mid = "vllm.attention:mid"
        inner = TARGET
        with tempfile.TemporaryDirectory() as root:
            meta_path = os.path.join(root, "capture.pid-111.rank-0", "meta.json")
            os.makedirs(os.path.dirname(meta_path))
            meta = self.meta(target=mid)
            meta["process_id"] = 111
            with open(meta_path, "w") as fh:
                json.dump(meta, fh)

            common_installs = [
                {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + mid,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 80, "dur": 1},
                {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + inner,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 82, "dur": 1},
            ]
            call_one = common_installs + [
                {"cat": "cpu_op", "name": ks.MARKER_PREFIX + mid,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 50},
                {"cat": "cpu_op", "name": "launch", "ph": "X",
                 "pid": 1, "tid": 2, "ts": 110, "dur": 5,
                 "args": {"External id": 7}},
                {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 180, "dur": 10,
                 "args": {"External id": 7}},
            ]
            call_two = common_installs + [
                {"cat": "cpu_op", "name": ks.MARKER_PREFIX + mid,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 80},
                {"cat": "cpu_op", "name": ks.MARKER_PREFIX + inner,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 120, "dur": 40},
                {"cat": "cpu_op", "name": "launch", "ph": "X",
                 "pid": 1, "tid": 2, "ts": 130, "dur": 5,
                 "args": {"External id": 9}},
                {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 200, "dur": 10,
                 "args": {"External id": 9}},
            ]
            trace_paths = []
            for index, events in enumerate((call_one, call_two), 1):
                path = os.path.join(
                    root, f"selection.pid-111.rank-0.call-{index}.json")
                with open(path, "w") as fh:
                    json.dump({"traceEvents": events}, fh)
                trace_paths.append(path)
            out_path = os.path.join(root, "selection.json")
            rc = ks.main([
                "--target", mid,
                "--device-kernel", KERNEL,
                "--capture-meta", meta_path,
                "--torch-trace", *trace_paths,
                "--candidate-target", mid,
                "--candidate-target", inner,
                "--out", out_path,
            ])
            self.assertEqual(rc, 1)
            with open(out_path) as fh:
                verdict = json.load(fh)
            self.assertIn("deeper_live_candidate_exists", verdict["failed"])
            self.assertEqual(verdict["deeper_live_candidates"], [inner])

    def test_cli_requires_deepest_selection_on_every_capture_process(self):
        mid = "vllm.attention:mid"
        inner = TARGET

        def process_events(with_inner):
            events = [
                {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + mid,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 80, "dur": 1},
                {"cat": "cpu_op", "name": ks.INSTALL_PREFIX + inner,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 82, "dur": 1},
                {"cat": "cpu_op", "name": ks.MARKER_PREFIX + mid,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 80},
            ]
            if with_inner:
                events.append({
                    "cat": "cpu_op", "name": ks.MARKER_PREFIX + inner,
                    "ph": "X", "pid": 1, "tid": 2, "ts": 120, "dur": 40,
                })
            events.extend([
                {"cat": "cpu_op", "name": "launch", "ph": "X",
                 "pid": 1, "tid": 2, "ts": 130, "dur": 5,
                 "args": {"External id": 9}},
                {"cat": "kernel", "name": KERNEL, "ph": "X", "ts": 200, "dur": 10,
                 "args": {"External id": 9}},
            ])
            return events

        with tempfile.TemporaryDirectory() as root:
            meta_paths, trace_paths = [], []
            for pid, with_inner in ((111, False), (222, True)):
                meta_path = os.path.join(
                    root, f"capture.pid-{pid}.rank-0", "meta.json")
                os.makedirs(os.path.dirname(meta_path))
                meta = self.meta(target=mid)
                meta["process_id"] = pid
                with open(meta_path, "w") as fh:
                    json.dump(meta, fh)
                meta_paths.append(meta_path)
                trace_path = os.path.join(
                    root, f"selection.pid-{pid}.rank-0.call-1.json")
                with open(trace_path, "w") as fh:
                    json.dump({"traceEvents": process_events(with_inner)}, fh)
                trace_paths.append(trace_path)
            out_path = os.path.join(root, "selection.json")
            rc = ks.main([
                "--target", mid,
                "--device-kernel", KERNEL,
                "--capture-meta", *meta_paths,
                "--torch-trace", *trace_paths,
                "--candidate-target", mid,
                "--candidate-target", inner,
                "--out", out_path,
            ])
            self.assertEqual(rc, 1)
            with open(out_path) as fh:
                verdict = json.load(fh)
            self.assertFalse(verdict["ok"])
            self.assertEqual(verdict["deeper_live_candidates"], [inner])
            self.assertEqual(
                sorted(verdict["live_candidate_targets"]), sorted([mid, inner]))


if __name__ == "__main__":
    unittest.main()
