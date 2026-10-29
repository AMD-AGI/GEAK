"""capture_shapes inside a torch.compile(fullgraph=True) region, the way vLLM compiles a model forward.

vLLM's forward calls the hooked dispatch function (e.g. ``rocm_unquantized_gemm``) inside a graph that Dynamo
traces with ``fullgraph=True``, so any construct Dynamo cannot trace in the recording wrapper is an engine-start
failure, not a skipped record. Needs torch; skipped where it is not installed.
"""
import atexit
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest

try:
    import torch
except ImportError:
    torch = None

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_capture_shapes():
    for name in ("CAPTURE_TARGET", "CAPTURE_OUT"):
        os.environ.pop(name, None)
    spec = importlib.util.spec_from_file_location("capture_shapes_compiled", os.path.join(SCRIPTS_DIR, "capture_shapes.py"))
    mod = importlib.util.module_from_spec(spec)
    # Dynamo resolves an inlined function's globals through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


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
        self.cs.install("fake_compiled_layer:gemm", self.out_dir, max_cases=2)

    def _forward(self, x, w):
        return self.layer.gemm(x, w) + 1

    def test_the_hooked_callable_compiles_with_fullgraph(self):
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        compiled = torch.compile(self._forward, fullgraph=True, backend="eager")

        torch.testing.assert_close(compiled(x, w), x @ w.t() + 1)

    def test_eager_calls_are_still_recorded(self):
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        self._forward(x, w)

        self.assertEqual(self.cs._STATE["calls"], 1)
        self.assertEqual(list(self.cs._STATE["shape_counts"].values()), [1])


if __name__ == "__main__":
    unittest.main()
