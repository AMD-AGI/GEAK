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

    def test_installing_the_same_target_twice_keeps_the_first_wrapper(self):
        """A candidate may be named twice (once by the architect, once by the extractor). Re-wrapping
        would nest a marker inside itself and double-count the seam's calls."""
        st.install("_seam_trace_fixture:outer")
        wrapper = self.module.outer
        st.install("_seam_trace_fixture:outer")
        self.assertIs(self.module.outer, wrapper)
        self.assertEqual(self.module.outer(3), 8)
        markers = [value for action, value in self.events
                   if action == "enter" and value.startswith(st.MARKER_PREFIX)]
        self.assertEqual(markers, [st.MARKER_PREFIX + "_seam_trace_fixture:outer"])

    def test_a_callable_whose_signature_cannot_be_read_is_still_marked(self):
        """`inspect.signature` refuses some C callables reached through a partial. The seam is still a
        pure-Python object to replace, so the probe must install rather than abort the whole capture."""
        import functools

        self.module.opaque = functools.partial(range)
        st.install("_seam_trace_fixture:opaque")
        self.assertEqual(list(self.module.opaque(3)), [0, 1, 2])
        names = [value for action, value in self.events if action == "enter"]
        self.assertIn(st.MARKER_PREFIX + "_seam_trace_fixture:opaque", names)

    def test_the_seam_still_runs_when_the_process_has_no_torch(self):
        """Markers are installed from sitecustomize, which runs before the server imports torch. A
        seam that raised (or silently skipped the call) there would break the server being profiled."""
        st.install("_seam_trace_fixture:outer")
        sys.modules["torch"] = None  # makes `import torch` raise, as it does before torch is built
        self.assertEqual(self.module.outer(3), 8)
        self.assertEqual([value for action, value in self.events if action == "enter"], [])


class TestTracePathIsProcessLocal(unittest.TestCase):
    def setUp(self):
        for key in ("GEAK_SELECTION_TRACE", "GEAK_SELECTION_TRACE_UNIQUE", "RANK"):
            self.addCleanup(os.environ.pop, key, None)
            os.environ.pop(key, None)

    def test_no_trace_env_means_no_path_and_no_profile(self):
        """The marker overlay is also loaded by processes that are not the capture (a tuner, a
        one-shot import check). With no destination they must not start a profiler at all."""
        self.assertEqual(st._trace_path(1), "")
        self.assertFalse(st._start_profile())

    def test_an_explicit_template_is_filled_with_pid_and_rank(self):
        """A TP deployment runs one marked process per rank. Whether the operator places {pid}/{rank}
        by hand or leaves it to the default suffix, two ranks must never be handed one filename."""
        os.environ["GEAK_SELECTION_TRACE"] = "/tmp/sel-{pid}-{rank}.json"
        os.environ["RANK"] = "3"
        self.assertEqual(st._trace_path(7), f"/tmp/sel-{os.getpid()}-3.call-7.json")

    def test_a_path_with_no_placeholders_gets_the_pid_and_rank_appended(self):
        os.environ["GEAK_SELECTION_TRACE"] = "/tmp/sel.json"
        os.environ["RANK"] = "3"
        self.assertEqual(st._trace_path(7), f"/tmp/sel.pid-{os.getpid()}.rank-3.call-7.json")

    def test_the_unique_opt_out_takes_the_path_verbatim(self):
        """The escape hatch for a single-process debug capture: write exactly where I said. It is not
        safe for a multi-rank run, which is why per-process naming is what happens by default."""
        os.environ["GEAK_SELECTION_TRACE"] = "/tmp/sel.json"
        os.environ["GEAK_SELECTION_TRACE_UNIQUE"] = "0"
        self.assertEqual(st._trace_path(7), "/tmp/sel.json")

    def test_the_rank_comes_from_the_launcher_env_when_it_declares_one(self):
        os.environ["RANK"] = "2"
        self.assertEqual(st._rank(), "2")

    def test_an_unranked_process_is_still_named_rather_than_dropped(self):
        self.assertEqual(st._rank(), "unknown")


class TestProfileLifecycleFailsSoft(unittest.TestCase):
    """Every failure here happens inside the SERVER under measurement. A probe that raises would
    take the capture down with it, so each path degrades to "no trace" plus a stderr line."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["GEAK_SELECTION_TRACE"] = os.path.join(self.tmp.name, "selection.json")
        os.environ["GEAK_SELECTION_TRACE_UNIQUE"] = "0"
        self.addCleanup(os.environ.pop, "GEAK_SELECTION_TRACE", None)
        self.addCleanup(os.environ.pop, "GEAK_SELECTION_TRACE_UNIQUE", None)
        self.addCleanup(os.environ.pop, "GEAK_SELECTION_PROFILE_CALLS", None)
        self.saved_torch = sys.modules.get("torch")
        self.addCleanup(self._restore_torch)
        st._PROFILE.update(active=False, done=False, owner=None, profiler=None,
                           active_calls=0, root_calls=0, out="", trace_index=0,
                           atexit_registered=False)

    def _restore_torch(self):
        if self.saved_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self.saved_torch

    def _install_torch(self, profile):
        torch = types.ModuleType("torch")
        torch.profiler = types.SimpleNamespace(
            ProfilerActivity=types.SimpleNamespace(CPU="cpu"),
            profile=profile,
            record_function=lambda name: _Context([], name),
        )
        sys.modules["torch"] = torch

    def test_a_profiler_that_refuses_to_start_disables_further_attempts(self):
        def refuse(activities):
            raise RuntimeError("no profiling on this device")

        self._install_torch(refuse)
        self.assertFalse(st._start_profile())
        self.assertTrue(st._PROFILE["done"])
        self.assertFalse(st._start_profile())

    def test_a_failed_export_does_not_propagate_into_the_served_call(self):
        events = []

        class _Unexportable(_Profiler):
            def export_chrome_trace(self, path):
                raise OSError("disk full")

        self._install_torch(lambda activities: _Unexportable(events))
        self.assertTrue(st._start_profile())
        st._finish_profile()
        self.assertFalse(st._PROFILE["active"])

    def test_finishing_a_profile_that_never_started_is_a_no_op(self):
        st._finish_profile()
        self.assertEqual(st._PROFILE["trace_index"], 0)

    def test_the_call_budget_stops_the_probe_rather_than_tracing_the_whole_run(self):
        """An unbounded probe on a serving process writes a trace per root call for the life of the
        server. The budget is what keeps a capture window bounded."""
        os.environ["GEAK_SELECTION_PROFILE_CALLS"] = "1"
        self._install_torch(lambda activities: _Profiler([]))
        self.assertTrue(st._start_profile())
        st._finish_profile()
        self.assertTrue(st._PROFILE["done"])
        self.assertFalse(st._start_profile())


class TestWrappableRefusesWhatItCannotPreserve(unittest.TestCase):
    def test_a_builtin_is_refused_because_a_python_wrapper_changes_its_identity(self):
        self.assertFalse(st._wrappable(len))

    def test_a_plain_callable_object_is_accepted(self):
        class Seam:
            def __call__(self):
                return 1

        self.assertTrue(st._wrappable(Seam()))

    def test_a_non_callable_attribute_is_refused(self):
        self.assertFalse(st._wrappable(object()))


if __name__ == "__main__":
    unittest.main()
