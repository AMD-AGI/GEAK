#!/usr/bin/env python3
"""Tests for marker-only candidate tracing."""

import importlib.util
import os
import sys
import tempfile
import types
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = importlib.util.spec_from_file_location("seam_trace", os.path.join(SCRIPTS, "seam_trace.py"))
st = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(st)


class _Context:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def __enter__(self):
        self.events.append(("enter", self.name))

    def __exit__(self, *unused):
        self.events.append(("exit", self.name))


class _Profiler:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append(("profile", "start"))
        return self

    def __exit__(self, *unused):
        self.events.append(("profile", "stop"))

    def export_chrome_trace(self, path):
        with open(path, "w") as fh:
            fh.write("{}")


class TestSeamTrace(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trace = os.path.join(self.tmp.name, "selection.json")
        os.environ["GEAK_SELECTION_TRACE"] = self.trace
        os.environ["GEAK_SELECTION_TRACE_UNIQUE"] = "0"
        os.environ["GEAK_SELECTION_PROFILE_CALLS"] = "1"
        self.addCleanup(os.environ.pop, "GEAK_SELECTION_TRACE", None)
        self.addCleanup(os.environ.pop, "GEAK_SELECTION_TRACE_UNIQUE", None)
        self.addCleanup(os.environ.pop, "GEAK_SELECTION_PROFILE_CALLS", None)

        torch = types.ModuleType("torch")
        torch.profiler = types.SimpleNamespace(
            ProfilerActivity=types.SimpleNamespace(CPU="cpu", CUDA="cuda"),
            profile=lambda activities: _Profiler(self.events),
            record_function=lambda name: _Context(self.events, name),
        )
        self.saved_torch = sys.modules.get("torch")
        sys.modules["torch"] = torch
        self.addCleanup(self._restore_torch)

        module = types.ModuleType("_seam_trace_fixture")
        module.inner = lambda value: value + 1
        module.outer = lambda value: module.inner(value) * 2
        sys.modules[module.__name__] = module
        self.module = module
        self.addCleanup(sys.modules.pop, module.__name__, None)

        st._INSTALLED.clear()
        st._PROFILE.update(lock=st._PROFILE["lock"], active=False, done=False,
                           owner=None, profiler=None, active_calls=0, root_calls=0, out="",
                           trace_index=0, atexit_registered=False)

    def _restore_torch(self):
        if self.saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self.saved_torch

    def test_first_outer_call_profiles_nested_candidate_markers_once(self):
        st.install("_seam_trace_fixture:outer")
        st.install("_seam_trace_fixture:inner")
        self.assertEqual(self.module.outer(3), 8)
        self.assertTrue(os.path.isfile(self.trace))
        names = [value for action, value in self.events if action == "enter"]
        self.assertEqual(names, [
            st.INSTALL_PREFIX + "_seam_trace_fixture:inner",
            st.INSTALL_PREFIX + "_seam_trace_fixture:outer",
            st.MARKER_PREFIX + "_seam_trace_fixture:outer",
            st.MARKER_PREFIX + "_seam_trace_fixture:inner",
        ])
        self.assertEqual(self.events.count(("profile", "start")), 1)
        self.assertEqual(self.events.count(("profile", "stop")), 1)

    def test_jit_protocol_callable_is_rejected_before_replacement(self):
        class JitLike:
            def __call__(self, value):
                return value

            def run(self, value):
                return value

        original = JitLike()
        self.module.jit_entry = original
        with self.assertRaisesRegex(RuntimeError, "cannot safely mark"):
            st.install("_seam_trace_fixture:jit_entry")
        self.assertIs(self.module.jit_entry, original)

    def test_each_root_call_exports_before_process_exit(self):
        os.environ["GEAK_SELECTION_PROFILE_CALLS"] = "32"
        st.install("_seam_trace_fixture:outer")
        self.assertEqual(self.module.outer(3), 8)
        self.assertTrue(os.path.isfile(self.trace))
        self.assertEqual(self.events.count(("profile", "stop")), 1)

    def test_class_method_candidate_is_marked_and_proven_installed(self):
        class Runner:
            def run(self, value):
                return value + 1

        self.module.Runner = Runner
        st.install("_seam_trace_fixture:Runner.run")
        self.assertEqual(Runner().run(3), 4)
        names = [value for action, value in self.events if action == "enter"]
        self.assertIn(st.INSTALL_PREFIX + "_seam_trace_fixture:Runner.run", names)
        self.assertIn(st.MARKER_PREFIX + "_seam_trace_fixture:Runner.run", names)

    def test_process_local_call_traces_do_not_overwrite(self):
        os.environ.pop("GEAK_SELECTION_TRACE_UNIQUE", None)
        os.environ["GEAK_SELECTION_PROFILE_CALLS"] = "2"
        st.install("_seam_trace_fixture:outer")
        self.module.outer(1)
        self.module.outer(2)
        traces = sorted(
            name for name in os.listdir(self.tmp.name)
            if name.startswith("selection.pid-") and name.endswith(".json"))
        self.assertEqual(len(traces), 2)
        self.assertIn(".call-1.json", traces[0])
        self.assertIn(".call-2.json", traces[1])


if __name__ == "__main__":
    unittest.main()
