"""CPU regressions for the vLLM profiler-window release hook (vllm_observer/geak_profiler_release.py).

Observed on a TP8 vLLM capture: every torch profiler window stayed resident in the
TP workers (~100-170 GB host memory per captured point) because the profiler object is
self-referencing and the workers' frozen GC heap seldom runs a full collection.
"""
import gc
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OBSERVER = ROOT / "scripts/vllm_observer"
spec = importlib.util.spec_from_file_location("test_geak_profiler_release",
                                              OBSERVER / "geak_profiler_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class Profile:
    """Stand-in for torch.profiler.profile: a reference cycle holding a large payload."""
    def __init__(self, log):
        self.log = log
        self.action_map = {"stop": self.stop}  # the cycle torch's profiler has
        self.kineto_results = bytearray(1 << 20)

    def stop(self):
        self.log.append("export")  # on_trace_ready runs inside stop()


def wrapper_module():
    class TorchProfilerWrapper:
        def __init__(self):
            self.log = []
            self.profiler = None

        def _start(self):
            self.profiler = Profile(self.log)

        def _stop(self):
            """original doc"""
            self.profiler.stop()
            self.log.append("stopped")
            return "rv"
    return types.SimpleNamespace(TorchProfilerWrapper=TorchProfilerWrapper)


class ProfilerReleaseTest(unittest.TestCase):
    def test_window_is_freed_after_export_even_with_frozen_gc_heap(self):
        mod = wrapper_module()
        self.assertTrue(release.patch_wrapper(mod))
        w = mod.TorchProfilerWrapper()
        gc.freeze()
        self.addCleanup(gc.unfreeze)
        w._start()
        ref = weakref.ref(w.profiler)
        with patch.object(release, "malloc_trim") as trim:
            self.assertEqual(w._stop(), "rv")
        self.assertEqual(w.log, ["export", "stopped"])
        self.assertIsNone(w.profiler)
        self.assertIsNone(ref(), "profiler window still alive after _stop")
        trim.assert_called_once_with()

    def test_unpatched_window_stays_alive_without_collection(self):
        # documents the leak the hook fixes: refcounting alone never frees the window
        mod = wrapper_module()
        w = mod.TorchProfilerWrapper()
        gc.disable()
        self.addCleanup(gc.enable)
        w._start()
        ref = weakref.ref(w.profiler)
        w._stop()
        w._start()  # next window replaces the attribute, the old one is only in its cycle
        self.assertIsNotNone(ref())
        gc.collect()
        self.assertIsNone(ref())

    def test_patch_is_idempotent_and_keeps_metadata(self):
        mod = wrapper_module()
        release.patch_wrapper(mod)
        first = mod.TorchProfilerWrapper._stop
        release.patch_wrapper(mod)
        self.assertIs(mod.TorchProfilerWrapper._stop, first)
        self.assertEqual(first.__name__, "_stop")
        self.assertEqual(first.__doc__, "original doc")

    def test_release_runs_when_stop_raises_and_error_propagates(self):
        mod = wrapper_module()

        def bad(self):
            raise RuntimeError("boom")
        mod.TorchProfilerWrapper._stop = bad
        release.patch_wrapper(mod)
        w = mod.TorchProfilerWrapper()
        w._start()
        with self.assertRaises(RuntimeError):
            w._stop()
        self.assertIsNone(w.profiler)

    def test_release_failure_does_not_break_stop(self):
        mod = wrapper_module()
        release.patch_wrapper(mod)
        w = mod.TorchProfilerWrapper()
        w._start()
        with patch.object(release.gc, "collect", side_effect=MemoryError("x")), \
                patch.object(sys, "stderr", new_callable=io.StringIO) as err:
            self.assertEqual(w._stop(), "rv")
        self.assertIn("profiler release failed", err.getvalue())

    def test_missing_wrapper_warns_without_failing(self):
        with patch.object(sys, "stderr", new_callable=io.StringIO) as err:
            self.assertFalse(release.patch_wrapper(types.SimpleNamespace()))
        self.assertIn("not released", err.getvalue())

    def _import_fake_vllm(self, **env):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "vllm" / "profiler"
            pkg.mkdir(parents=True)
            (pkg.parent / "__init__.py").write_text("")
            (pkg / "__init__.py").write_text("")
            (pkg / "wrapper.py").write_text(
                "class TorchProfilerWrapper:\n    def _stop(self):\n        pass\n")
            environ = dict(os.environ, PYTHONPATH=os.pathsep.join([str(OBSERVER), tmp]), **env)
            environ.pop("GEAK_ITERATION_LOG_DIR", None)
            run = subprocess.run([sys.executable, "-c",
                "import sys; assert 'vllm' not in sys.modules\n"
                "import vllm.profiler.wrapper as w\n"
                "print(getattr(w.TorchProfilerWrapper._stop, '_geak_profiler_release', False))"],
                cwd=tmp, env=environ, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        return run.stdout.strip()

    def test_sitecustomize_patches_wrapper_on_import(self):
        self.assertEqual(self._import_fake_vllm(), "True")

    def test_env_opt_out(self):
        self.assertEqual(self._import_fake_vllm(GEAK_PROFILER_RELEASE="0"), "False")


if __name__ == "__main__":
    unittest.main()
