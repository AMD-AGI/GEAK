#!/usr/bin/env python3
"""Tests for the CUDA/HIP graph-capture guard in seam_trace.py.

Run:  python3 -m unittest discover -s e2e_workflow/scripts/tests -v
  or: python3 e2e_workflow/scripts/tests/test_seam_trace_capture_guard.py

seam_trace wraps a seam with ``torch.profiler.record_function`` and drives a profiler lifecycle
around the outermost call. Every one of those profiler operations -- __enter__, __exit__,
export_chrome_trace -- issues host-side work that is illegal while the current stream is capturing
a graph, and a vLLM/SGLang server captures graphs on every warmup. On ROCm the illegal call does
not simply raise: it poisons the process, and the next library handle creation dies with
``HIPBLAS_STATUS_INTERNAL_ERROR when calling hipblasCreate(handle)``, taking the engine down
mid-warmup. The guard makes the whole marker layer a no-op for the duration of a capture.

test_seam_trace.py covers the marker/profiler behaviour outside capture. These tests cover the
capture path specifically, and each one checks that the guard is what produces the behaviour: the
"neutralised" variants monkeypatch _capturing() back to False and assert the fatal operation
reappears, so a test cannot quietly stop covering its fix.

Also here: a seam traced by torch.compile is named in the trace, and `from X import f` copies are marked.

torch is replaced by a stub whose capture and compile flags are flippable, so this needs no GPU and no torch.
Every test loads its own copy of the module, because _PROFILE is process-global state.
"""
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEAM_TRACE = os.path.join(SCRIPTS, "seam_trace.py")


class _FakeTorch(types.ModuleType):
    """Minimal torch surface: a flippable capture flag, plus recording profiler objects."""

    def __init__(self, capturing=False, has_capture_query=True, compiling=None):
        super().__init__("torch")
        self.entered = []          # record_function labels entered, and EXPORT: paths
        self.profiles_started = 0
        self.capture_queries = 0
        self._capturing = capturing
        outer = self

        class _RecordFunction(contextlib.AbstractContextManager):
            def __init__(self, label):
                self.label = label

            def __enter__(self):
                outer.entered.append(self.label)
                return self

            def __exit__(self, *exc):
                return False

        class _Profile(contextlib.AbstractContextManager):
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                outer.profiles_started += 1
                return self

            def __exit__(self, *exc):
                return False

            def export_chrome_trace(self, path):
                outer.entered.append("EXPORT:" + str(path))
                with open(path, "w") as fh:   # a real file, so the os.replace() downstream works
                    fh.write("{}")

        self.profiler = types.SimpleNamespace(
            record_function=_RecordFunction, profile=_Profile,
            ProfilerActivity=types.SimpleNamespace(CPU=1, CUDA=2))
        self.cuda = types.SimpleNamespace(synchronize=lambda *a, **k: None,
                                          is_available=lambda: True)
        if has_capture_query:
            self.cuda.is_current_stream_capturing = self._capture_query
        if compiling is not None:
            self.compiling = compiling
            self.compiler = types.SimpleNamespace(is_compiling=lambda: self.compiling)

    def _capture_query(self):
        self.capture_queries += 1
        return self._capturing

    def set_capturing(self, value):
        self._capturing = value


@contextlib.contextmanager
def fake_torch(**kwargs):
    saved = {k: v for k, v in sys.modules.items() if k == "torch" or k.startswith("torch.")}
    for key in list(saved):
        del sys.modules[key]
    stub = _FakeTorch(**kwargs)
    sys.modules["torch"] = stub
    sys.modules["torch.profiler"] = stub.profiler
    sys.modules["torch.cuda"] = stub.cuda
    try:
        yield stub
    finally:
        for key in ("torch", "torch.profiler", "torch.cuda"):
            sys.modules.pop(key, None)
        sys.modules.update(saved)


def load_seam():
    """A fresh module object per test, so the global _PROFILE never leaks between tests. Its atexit
    hook is never registered: it would run after the test's fake torch is gone."""
    spec = importlib.util.spec_from_file_location("seam_trace_under_test", SEAM_TRACE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.atexit = types.SimpleNamespace(register=lambda fn: None)
    return mod


_TARGET_SEQ = [0]


def make_target():
    """A fresh module:attr target for install(), with a call log."""
    _TARGET_SEQ[0] += 1
    name = f"seam_victim_{_TARGET_SEQ[0]}"
    mod = types.ModuleType(name)
    calls = []

    def work(x, y=0):
        calls.append((x, y))
        return x + y

    mod.work = work
    sys.modules[name] = mod
    return mod, calls, f"{name}:work"


class TestCapturingQuery(unittest.TestCase):
    def test_reports_the_torch_capture_state(self):
        with fake_torch(capturing=False) as torch:
            seam = load_seam()
            self.assertFalse(seam._capturing())
            torch.set_capturing(True)
            self.assertTrue(seam._capturing())

    def test_older_torch_without_the_query_is_treated_as_not_capturing(self):
        """The query is recent. Its absence must not disable the marker layer entirely."""
        with fake_torch(has_capture_query=False):
            self.assertFalse(load_seam()._capturing())

    def test_absent_torch_is_treated_as_not_capturing(self):
        """Markers install from sitecustomize, which runs before torch is importable."""
        saved = sys.modules.get("torch")
        sys.modules["torch"] = None          # makes `import torch` raise
        try:
            self.assertFalse(load_seam()._capturing())
        finally:
            sys.modules.pop("torch", None)
            if saved is not None:
                sys.modules["torch"] = saved


class TestMarkerIsPassThroughDuringCapture(unittest.TestCase):
    def _call_during_capture(self, neutralise_guard=False):
        with fake_torch(capturing=True) as torch:
            seam = load_seam()
            if neutralise_guard:
                seam._capturing = lambda: False
            mod, calls, target = make_target()
            seam.install(target)
            result = mod.work(2, y=3)
            return result, torch, seam, calls

    def test_wrapped_function_still_runs_and_returns_correctly(self):
        result, _torch, _seam, calls = self._call_during_capture()
        self.assertEqual(result, 5, "the guard must not change what the seam returns")
        self.assertEqual(calls, [(2, 3)], "arguments must pass through untouched")

    def test_no_record_function_node_is_entered_during_capture(self):
        _r, torch, _seam, _c = self._call_during_capture()
        self.assertEqual(torch.entered, [],
                         "record_function was entered mid-capture: this is the call that poisons "
                         "the ROCm process and kills the next hipblasCreate")

    def test_profile_lifecycle_is_untouched_during_capture(self):
        _r, torch, seam, _c = self._call_during_capture()
        self.assertEqual(torch.profiles_started, 0)
        self.assertFalse(seam._PROFILE["active"])
        self.assertEqual(seam._PROFILE["active_calls"], 0,
                         "call depth leaked; a later real call would be misread as nested")

    def test_without_the_guard_the_fatal_call_happens(self):
        """Neutralise _capturing() and the pre-fix behaviour comes straight back."""
        _r, torch, _seam, _c = self._call_during_capture(neutralise_guard=True)
        self.assertNotEqual(torch.entered, [],
                            "NON-DISCRIMINATING: record_function was skipped mid-capture even with "
                            "the guard neutralised, so these tests do not cover the guard")

    def test_marker_resumes_once_capture_ends(self):
        """Scoped to the capture, not a permanent disable: warmup must not cost us the trace."""
        with fake_torch(capturing=True) as torch:
            seam = load_seam()
            mod, _calls, target = make_target()
            seam.install(target)
            mod.work(1)
            self.assertEqual(torch.entered, [])
            torch.set_capturing(False)
            mod.work(1)
            self.assertTrue(any(target in entry for entry in torch.entered),
                            f"marker never resumed after capture ended; entered={torch.entered}")


class TestMarkerIsPassThroughWhileCompiling(unittest.TestCase):
    """Under torch.compile the marker must not reach the capture query, which Dynamo cannot trace."""

    def _call_while_compiling(self, neutralise_check=False):
        with fake_torch(capturing=False, compiling=True) as torch:
            seam = load_seam()
            if neutralise_check:
                seam._compile_probe = lambda: (lambda: False)
            mod, calls, target = make_target()
            seam.install(target)
            result = mod.work(2, y=3)
            return result, torch, seam, calls

    def test_wrapped_function_still_runs_and_returns_correctly(self):
        result, _torch, _seam, calls = self._call_while_compiling()
        self.assertEqual(result, 5)
        self.assertEqual(calls, [(2, 3)])

    def test_neither_the_capture_query_nor_a_marker_runs(self):
        _r, torch, seam, _c = self._call_while_compiling()
        self.assertEqual(torch.capture_queries, 0,
                         "the capture query ran while compiling: Dynamo rejects it and the engine dies")
        self.assertEqual(torch.entered, [])
        self.assertEqual(torch.profiles_started, 0)
        self.assertEqual(seam._PROFILE["active_calls"], 0)

    def test_without_the_check_the_capture_query_runs(self):
        _r, torch, _seam, _c = self._call_while_compiling(neutralise_check=True)
        self.assertGreater(torch.capture_queries, 0,
                           "NON-DISCRIMINATING: the capture query was skipped even with the check "
                           "neutralised, so these tests do not cover the check")

    def test_torch_without_the_compiler_query_is_treated_as_not_compiling(self):
        with fake_torch(capturing=False) as torch:
            seam = load_seam()
            mod, _calls, target = make_target()
            seam.install(target)
            mod.work(1)
            self.assertTrue(any(target in entry for entry in torch.entered))


class TestProfileLifecycleGuards(unittest.TestCase):
    """_start_profile needs GEAK_SELECTION_TRACE set, or it declines for an unrelated reason."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = os.environ.get("GEAK_SELECTION_TRACE")
        os.environ["GEAK_SELECTION_TRACE"] = os.path.join(self.tmp, "selection.json")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("GEAK_SELECTION_TRACE", None)
        else:
            os.environ["GEAK_SELECTION_TRACE"] = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_start_profile_declines_inside_capture(self):
        with fake_torch(capturing=True):
            seam = load_seam()
            self.assertFalse(seam._start_profile())
            self.assertFalse(seam._PROFILE["active"])
            self.assertFalse(seam._PROFILE["done"],
                             "declining must not burn the budget or mark the profile finished")

    def test_start_profile_still_works_outside_capture(self):
        with fake_torch(capturing=False):
            seam = load_seam()
            self.assertTrue(seam._start_profile(), "the guard must not block ordinary profiling")
            self.assertTrue(seam._PROFILE["active"])

    def test_finish_profile_defers_instead_of_exporting_inside_capture(self):
        with fake_torch(capturing=False) as torch:
            seam = load_seam()
            self.assertTrue(seam._start_profile())
            index_before = seam._PROFILE["trace_index"]
            torch.set_capturing(True)
            seam._finish_profile()
            self.assertTrue(seam._PROFILE["active"],
                            "the profile must stay active so a later eager call can export it")
            self.assertEqual(seam._PROFILE["trace_index"], index_before,
                             "trace index advanced without an export")
            self.assertFalse([e for e in torch.entered if str(e).startswith("EXPORT:")],
                             "exported a chrome trace mid-capture: fatal on ROCm")

    def test_deferred_profile_is_exported_by_the_next_eager_finish(self):
        with fake_torch(capturing=False) as torch:
            seam = load_seam()
            seam._start_profile()
            torch.set_capturing(True)
            seam._finish_profile()
            self.assertTrue(seam._PROFILE["active"])
            torch.set_capturing(False)
            seam._finish_profile()
            self.assertFalse(seam._PROFILE["active"])
            self.assertTrue(os.listdir(self.tmp),
                            "the profile deferred during capture was never exported afterwards, "
                            "so the trace is lost")

    def test_a_seam_traced_by_torch_compile_forwards_and_the_next_trace_names_it(self):
        with fake_torch(compiling=True) as torch:
            seam = load_seam()
            inner, calls, inner_target = make_target()
            outer, _calls, outer_target = make_target()
            seam.install(inner_target)
            seam.install(outer_target)
            self.assertEqual(inner.work(1, 2), 3)
            self.assertEqual(calls, [(1, 2)])
            self.assertEqual(torch.entered, [], "the marker body ran while Dynamo was tracing")
            torch.compiling = False
            outer.work(1)                     # an eager root call opens the process-local trace
        self.assertIn(seam.COMPILED_PREFIX + inner_target, torch.entered)
        self.assertNotIn(seam.COMPILED_PREFIX + outer_target, torch.entered)

    def test_a_seam_compiled_while_a_trace_is_open_is_named_in_that_trace(self):
        with fake_torch(compiling=False) as torch:
            seam = load_seam()
            inner, _calls, target = make_target()
            seam.install(target)
            self.assertTrue(seam._start_profile())
            torch.compiling = True
            inner.work(1)
            torch.compiling = False
            seam._finish_profile()
        self.assertIn(seam.COMPILED_PREFIX + target, torch.entered)


class TestCompiledSeamRecord(unittest.TestCase):
    def test_a_failed_compiled_seam_record_is_reported(self):
        with fake_torch():
            seam = load_seam()
            _mod, _calls, target = make_target()
            seam.install(target)
            seam._COMPILED[target].seen = True
            tmp = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, tmp, True)
            err = io.StringIO()
            with mock.patch.dict(os.environ, {"GEAK_SELECTION_TRACE": os.path.join(tmp, "s.json")}), \
                    contextlib.redirect_stderr(err):
                self.assertTrue(seam._start_profile())
                with mock.patch.object(seam, "_record_install_markers", side_effect=RuntimeError("boom")):
                    seam._finish_profile()
        self.assertIn("boom", err.getvalue())


class TestFromImportCopies(unittest.TestCase):
    def test_a_from_import_copy_is_marked_too(self):
        with fake_torch() as torch:
            seam = load_seam()
            mod, _calls, target = make_target()
            importer = types.ModuleType(mod.__name__ + "_importer")
            importer.work, importer.other = mod.work, lambda x: x    # `from mod import work`
            other = importer.other
            sys.modules[importer.__name__] = importer
            self.addCleanup(sys.modules.pop, importer.__name__, None)
            seam.install(target)
            importer.work(2)
        self.assertIs(importer.work, mod.work)
        self.assertIs(importer.other, other)
        self.assertIn(seam.MARKER_PREFIX + target, torch.entered)

    def test_a_seam_declared_under_two_import_paths_leaves_one_span_per_call(self):
        # The caller did `from defining import work`, and both names are candidates. Installing the
        # defining name first rebinds the caller's alias to its marker; installing the caller's name
        # must not wrap that marker, or one call leaves both spans, nested, read as a deeper candidate.
        with fake_torch() as torch:
            seam = load_seam()
            mod, _calls, target = make_target()
            caller = types.ModuleType(mod.__name__ + "_caller")
            caller.work = mod.work
            sys.modules[caller.__name__] = caller
            self.addCleanup(sys.modules.pop, caller.__name__, None)
            seam.install(target)
            seam.install(caller.__name__ + ":work")
            caller.work(1)
        spans = [label for label in torch.entered if label.startswith(seam.MARKER_PREFIX)]
        self.assertEqual(spans, [seam.MARKER_PREFIX + caller.__name__ + ":work"])

    def test_an_alias_that_cannot_be_rebound_is_reported_and_the_marker_still_installs(self):
        class _ReadOnly(types.ModuleType):
            def __setattr__(self, name, value):
                raise AttributeError(name)

        with fake_torch() as torch:
            seam = load_seam()
            mod, _calls, target = make_target()
            frozen = _ReadOnly(mod.__name__ + "_frozen")
            frozen.__dict__["work"] = mod.work
            sys.modules[frozen.__name__] = frozen
            self.addCleanup(sys.modules.pop, frozen.__name__, None)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                seam.install(target)
            mod.work(1)
        self.assertIn(frozen.__name__ + ".work (NOT rebound)", err.getvalue())
        self.assertIn(seam.MARKER_PREFIX + target, torch.entered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
