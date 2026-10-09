#!/usr/bin/env python3
"""Unit tests for vllm_semantic_capture.py -- the vLLM eager shape probe's entry point.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v

This module is the ONLY vLLM-specific part of GEAK Semantics 1.2: the capture engine, the
bucketed logger and the `geak.semantics_runtime.v2` record format all live in
semantic_runtime_capture and are framework-agnostic. What differs is where the
(phase, batch_size, input_tokens) context comes from -- sglang reads a ForwardBatch off the
model's own forward, vLLM has to take it from the SchedulerOutput one level up.

Three properties are load-bearing, and each fails silently at serving scale:

  1. INERT WHEN UNARMED. The overlay may sit on the capture PYTHONPATH of a run that is
     also being measured. If install() did anything with GEAK_SEMANTICS_CAPTURE unset, the
     A/B would be measuring the probe.
  2. LOUD WHEN IT CANNOT HOOK. An unhooked model produces an EMPTY shape log, which
     downstream reads as "this phase has no shapes" -- indistinguishable from a phase that
     genuinely has none. Failure has to reach stderr.
  3. ONE PHASE RULE. The phase split must come from vllm_phase_annotate.classify_step, the
     same function that labels the trace. Two copies could disagree about what a decode
     step is, and neither output would reveal it.
"""
import io
import os
import sys
import types
import unittest
from unittest import mock

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import vllm_semantic_capture as vsc                                  # noqa: E402
import vllm_phase_annotate as vpa                                    # noqa: E402


class _SchedulerOutput:
    def __init__(self, per_req):
        self.num_scheduled_tokens = dict(per_req)
        self.total_num_scheduled_tokens = sum(per_req.values())


class _Logger:
    """Stands in for SemanticRuntimeLogger; records the calls that matter."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.layers, self.phases, self.max_forwards = [], [], 1
        self.contexts, self.forwards = [], 0

    def set_context(self, phase, batch_size, input_tokens):
        self.contexts.append((phase, batch_size, input_tokens))

    def mark_forward(self):
        self.forwards += 1


class _CaptureCase(unittest.TestCase):
    def setUp(self):
        vsc._INSTALLED.clear()
        vsc._WARNED.clear()
        self.addCleanup(vsc._INSTALLED.clear)
        self.addCleanup(vsc._WARNED.clear)
        self._env = os.environ.get("GEAK_SEMANTICS_CAPTURE")
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        if self._env is None:
            os.environ.pop("GEAK_SEMANTICS_CAPTURE", None)
        else:
            os.environ["GEAK_SEMANTICS_CAPTURE"] = self._env

    def _runner_class(self):
        class GPUModelRunner:
            uniform_decode_query_len = 1

            def __init__(self):
                self.model = object()
                self.loaded, self.executed = 0, []

            def load_model(self, *a, **kw):
                self.loaded += 1
                return "loaded"

            def execute_model(self, scheduler_output, *a, **kw):
                self.executed.append(scheduler_output)
                return "out"
        return GPUModelRunner

    def _install(self, logger, runner_cls, hooks=None, stderr=None):
        """install() with vllm + the capture core faked out."""
        core = types.SimpleNamespace(
            get_logger=lambda: logger,
            install_hooks_only=hooks or (lambda model: model))
        mod_runner = types.ModuleType("vllm.v1.worker.gpu_model_runner")
        mod_runner.GPUModelRunner = runner_cls
        modules = {
            "semantic_runtime_capture": core,
            "vllm": types.ModuleType("vllm"),
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
            "vllm.v1.worker.gpu_model_runner": mod_runner,
        }
        with mock.patch.dict(sys.modules, modules):
            with mock.patch.object(sys, "stderr", stderr or io.StringIO()):
                return vsc.install()


class TestArming(_CaptureCase):
    def test_inert_when_the_capture_switch_is_off(self):
        logger = _Logger(enabled=False)
        runner_cls = self._runner_class()
        original = runner_cls.execute_model
        self.assertFalse(self._install(logger, runner_cls))
        self.assertIs(runner_cls.execute_model, original,
                      "an unarmed overlay must not wrap anything")

    def test_installs_once_and_is_idempotent(self):
        logger = _Logger()
        runner_cls = self._runner_class()
        self.assertTrue(self._install(logger, runner_cls))
        wrapped = runner_cls.execute_model
        self.assertTrue(self._install(logger, runner_cls))
        self.assertIs(runner_cls.execute_model, wrapped,
                      "re-installing must not double-wrap (each layer would log twice)")

    def test_missing_vllm_degrades_to_false_with_a_warning(self):
        logger = _Logger()
        core = types.SimpleNamespace(get_logger=lambda: logger,
                                     install_hooks_only=lambda m: m)
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"semantic_runtime_capture": core}):
            with mock.patch.object(sys, "stderr", err):
                # no vllm module registered -> import fails
                self.assertFalse(vsc.install())
        self.assertIn("no decode shapes will be captured", err.getvalue())


class TestContextDriving(_CaptureCase):
    def test_decode_step_sets_context_and_marks_the_forward_after_it(self):
        logger, runner_cls = _Logger(), self._runner_class()
        self._install(logger, runner_cls)
        runner = runner_cls()

        order = []
        logger.set_context = lambda *a: order.append(("ctx", a))
        logger.mark_forward = lambda: order.append(("mark", None))
        self.assertEqual(runner.execute_model(_SchedulerOutput({"a": 1, "b": 1})), "out")

        # Context BEFORE the forward, bucket count AFTER: the counter gates _allowed(),
        # so incrementing first would refuse the very forward it should admit.
        self.assertEqual([kind for kind, _ in order], ["ctx", "mark"])
        self.assertEqual(order[0][1], ("DECODE", 2, 2))

    def test_prefill_step_reports_token_count_not_request_count(self):
        logger, runner_cls = _Logger(), self._runner_class()
        self._install(logger, runner_cls)
        runner = runner_cls()
        runner.execute_model(_SchedulerOutput({"a": 4096, "b": 2048}))
        self.assertEqual(logger.contexts, [("EXTEND", 2, 6144)])

    def test_context_comes_from_the_shared_phase_rule(self):
        """Pinned so the probe can never grow its own copy of the prefill/decode split."""
        logger, runner_cls = _Logger(), self._runner_class()
        self._install(logger, runner_cls)
        runner = runner_cls()
        sched = _SchedulerOutput({"a": 1, "b": 1, "c": 1})
        with mock.patch.object(vpa, "classify_step",
                               return_value=("SENTINEL", 7, 9)) as spy:
            runner.execute_model(sched)
        spy.assert_called_once()
        self.assertIs(spy.call_args[0][1], sched)
        self.assertEqual(logger.contexts, [("SENTINEL", 7, 9)])

    def test_unclassifiable_batch_runs_the_forward_without_a_context(self):
        # A dummy/empty batch must neither set a phase nor consume a bucket.
        logger, runner_cls = _Logger(), self._runner_class()
        self._install(logger, runner_cls)
        runner = runner_cls()
        self.assertEqual(runner.execute_model(_SchedulerOutput({})), "out")
        self.assertEqual(logger.contexts, [])
        self.assertEqual(logger.forwards, 0)

    def test_a_raising_phase_rule_does_not_break_the_forward(self):
        logger, runner_cls = _Logger(), self._runner_class()
        err = io.StringIO()
        self._install(logger, runner_cls, stderr=err)
        runner = runner_cls()
        with mock.patch.object(vpa, "classify_step", side_effect=RuntimeError("boom")):
            with mock.patch.object(sys, "stderr", err):
                self.assertEqual(runner.execute_model(_SchedulerOutput({"a": 1})), "out")
        self.assertIn("phase classification failed", err.getvalue())


class TestHookInstallation(_CaptureCase):
    def test_hooks_are_installed_on_the_loaded_model(self):
        logger, runner_cls = _Logger(), self._runner_class()
        seen = []
        self._install(logger, runner_cls, hooks=lambda model: seen.append(model))
        runner = runner_cls()
        self.assertEqual(runner.load_model(), "loaded")
        self.assertEqual(seen, [runner.model])
        self.assertEqual(runner.loaded, 1, "the real load_model must still run")

    def test_a_failing_hook_install_is_reported_not_swallowed(self):
        """An empty shape log reads downstream as 'no shapes exist'. Say it out loud."""
        logger, runner_cls = _Logger(), self._runner_class()
        err = io.StringIO()

        def explode(model):
            raise RuntimeError("no named_modules")

        self._install(logger, runner_cls, hooks=explode, stderr=err)
        runner = runner_cls()
        with mock.patch.object(sys, "stderr", err):
            self.assertEqual(runner.load_model(), "loaded")
        message = err.getvalue()
        self.assertIn("FAILED to install module hooks", message)
        self.assertIn("shape log will be empty", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
