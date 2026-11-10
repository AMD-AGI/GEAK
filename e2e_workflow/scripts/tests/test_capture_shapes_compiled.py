"""capture_shapes inside a torch.compile(fullgraph=True) region, the way vLLM compiles a model forward.

vLLM's forward calls the hooked dispatch function (e.g. ``rocm_unquantized_gemm``) inside a graph that Dynamo
traces with ``fullgraph=True``, so any construct Dynamo cannot trace in the recording wrapper is an engine-start
failure, not a skipped record. Needs torch; skipped where it is not installed or cannot load its runtime.
"""
import atexit
import contextlib
import importlib
import importlib.util
import io
import itertools
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock


def _import_torch():
    """torch, or None when it is absent or installed with a runtime library that cannot be loaded."""
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError):
        return None


torch = _import_torch()

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "fake_compiled_layer:gemm"
DEVICE_KERNEL = "gemm_kernel"
_LOADS = itertools.count()


def _load_capture_shapes():
    for name in ("CAPTURE_TARGET", "CAPTURE_OUT"):
        os.environ.pop(name, None)
    # A name per load: Dynamo resolves an inlined function's globals through sys.modules by name, so a reused
    # name would let one test's compiled graph write into the previous test's module state.
    spec = importlib.util.spec_from_file_location(
        f"capture_shapes_compiled_{next(_LOADS)}", os.path.join(SCRIPTS_DIR, "capture_shapes.py"))
    mod = importlib.util.module_from_spec(spec)
    # Dynamo resolves an inlined function's globals through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_kernel_selection():
    spec = importlib.util.spec_from_file_location("kernel_selection_compiled", os.path.join(SCRIPTS_DIR, "kernel_selection.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTorchImportGuard(unittest.TestCase):
    def test_a_torch_whose_runtime_library_cannot_load_is_treated_as_absent(self):
        with mock.patch.object(importlib, "import_module", side_effect=OSError("librocm_sysdeps_elf.so.1")):
            self.assertIsNone(_import_torch())


@unittest.skipIf(torch is None, "needs torch")
class TestCaptureUnderCompile(unittest.TestCase):
    def setUp(self):
        self.cs = _load_capture_shapes()
        self.addCleanup(sys.modules.pop, self.cs.__name__, None)
        self.addCleanup(atexit.unregister, self.cs._flush)
        self.out_dir = tempfile.mkdtemp(prefix="capture_compiled_test_")
        self.addCleanup(shutil.rmtree, self.out_dir, True)
        layer = types.ModuleType("fake_compiled_layer")
        layer.gemm = lambda x, w: x @ w.t()
        sys.modules[layer.__name__] = layer
        self.addCleanup(sys.modules.pop, layer.__name__, None)
        self.layer = layer
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        self.cs.install(TARGET, self.out_dir, max_cases=2)
        self.ks = _load_kernel_selection()

    def _forward(self, x, w):
        return self.layer.gemm(x, w) + 1

    def _flush(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.cs._flush()
        with open(os.path.join(self.out_dir, "meta.json")) as fh:
            return json.load(fh)

    def _trace(self, *, target_ran):
        events = [{"cat": "cpu_op", "name": self.ks.INSTALL_PREFIX + TARGET,
                   "ph": "X", "pid": 1, "tid": 2, "ts": 90, "dur": 1}]
        if target_ran:
            events += [
                {"cat": "cpu_op", "name": self.ks.MARKER_PREFIX + TARGET,
                 "ph": "X", "pid": 1, "tid": 2, "ts": 100, "dur": 100},
                {"cat": "cpu_op", "name": "launch", "ph": "X", "pid": 1, "tid": 2,
                 "ts": 110, "dur": 5, "args": {"External id": 7}},
                {"cat": "kernel", "name": f"void {DEVICE_KERNEL}(int)", "ph": "X",
                 "ts": 120, "dur": 25, "args": {"External id": 7}},
            ]
        return events

    def test_the_hooked_callable_compiles_with_fullgraph(self):
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        compiled = torch.compile(self._forward, fullgraph=True, backend="eager")

        torch.testing.assert_close(compiled(x, w), x @ w.t() + 1)

    def test_eager_calls_are_still_recorded(self):
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        self._forward(x, w)

        self.assertEqual(self.cs._STATE["calls"], 1)
        self.assertEqual(list(self.cs._STATE["shape_counts"].values()), [1])

    def test_a_target_reached_only_through_compiled_regions_is_named_unobservable(self):
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        torch.compile(self._forward, fullgraph=True, backend="eager")(x, w)

        meta = self._flush()
        verdict = self.ks.verify(TARGET, DEVICE_KERNEL, meta, self._trace(target_ran=False))

        self.assertEqual(meta["total_calls_observed"], 0)
        self.assertTrue(meta["compiled_passthrough_observed"])
        self.assertEqual(verdict["failed"], ["capture_compiled_unobservable"])

    def test_compiled_passthrough_observation_is_stable_across_calls(self):
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        compiled = torch.compile(self._forward, fullgraph=True, backend="eager")
        for _ in range(3):
            compiled(x, w)

        self.assertTrue(self.cs._STATE["compiled_passthrough_observed"])

    def test_repeated_compiled_calls_reuse_one_graph_and_record_stable_metadata(self):
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        compiled_graphs = []

        def backend(graph, _example_inputs):
            compiled_graphs.append(graph)
            return graph.forward

        compiled = torch.compile(self._forward, fullgraph=True, backend=backend)
        compiled(x, w)
        compiled(x, w)
        graphs_after_warmup = len(compiled_graphs)
        for _ in range(10):
            compiled(x, w)

        self.assertLessEqual(graphs_after_warmup, 2)
        self.assertEqual(len(compiled_graphs), graphs_after_warmup)
        meta = self._flush()
        self.assertTrue(meta["compiled_passthrough_observed"])
        self.assertNotIn("compiled_passthrough_calls", meta)

    def test_an_eager_call_beside_a_compiled_one_is_selected_and_the_capture_states_its_scope(self):
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        torch.compile(self._forward, fullgraph=True, backend="eager")(x, w)
        self._forward(x, w)

        meta = self._flush()
        verdict = self.ks.verify(TARGET, DEVICE_KERNEL, meta, self._trace(target_ran=True))

        self.assertEqual(meta["total_calls_observed"], 1)
        self.assertEqual(meta["capture_scope"], self.cs.CAPTURE_SCOPE)
        self.assertTrue(verdict["ok"], verdict)


if __name__ == "__main__":
    unittest.main()
