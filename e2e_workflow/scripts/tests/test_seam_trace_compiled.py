"""seam_trace's marker inside a torch.compile(fullgraph=True) region, the way vLLM compiles a model forward.

vLLM's forward calls the marked seam (e.g. ``rocm_unquantized_gemm``) inside a graph that Dynamo traces with
``fullgraph=True``. The marker's capture query, ``torch.cuda.is_current_stream_capturing()``, returns a bool, which
Dynamo cannot put in a graph, so an untraceable marker is an engine-start failure, not a missing marker. Needs
torch; skipped where it is not installed or cannot load its runtime.
"""
import importlib
import importlib.util
import itertools
import os
import sys
import types
import unittest


def _import_torch():
    """torch, or None when it is absent or installed with a runtime library that cannot be loaded."""
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError):
        return None


torch = _import_torch()

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOADS = itertools.count()


def _load_seam_trace():
    # A name per load: Dynamo resolves an inlined function's globals through sys.modules by name, so a reused
    # name would let one test's compiled graph read the previous test's module state.
    spec = importlib.util.spec_from_file_location(
        f"seam_trace_compiled_{next(_LOADS)}", os.path.join(SCRIPTS_DIR, "seam_trace.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipIf(torch is None, "needs torch")
class TestMarkerUnderCompile(unittest.TestCase):
    def setUp(self):
        self.seam = _load_seam_trace()
        self.addCleanup(sys.modules.pop, self.seam.__name__, None)
        layer = types.ModuleType(f"fake_marked_layer_{next(_LOADS)}")
        layer.gemm = lambda x, w: x @ w.t()
        sys.modules[layer.__name__] = layer
        self.addCleanup(sys.modules.pop, layer.__name__, None)
        self.layer = layer
        self.target = f"{layer.__name__}:gemm"
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def _forward(self, x, w):
        return self.layer.gemm(x, w) + 1

    def _install(self):
        self.seam.install(self.target)

    def test_the_marked_callable_compiles_with_fullgraph(self):
        self._install()
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        compiled = torch.compile(self._forward, fullgraph=True, backend="eager")

        torch.testing.assert_close(compiled(x, w), x @ w.t() + 1)

    def test_without_the_compile_check_fullgraph_tracing_fails(self):
        """Neutralise the check and the capture query is traced again, which Dynamo rejects."""
        self.seam._compile_probe = lambda: (lambda: False)
        self._install()
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        compiled = torch.compile(self._forward, fullgraph=True, backend="eager")

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            compiled(x, w)

    def test_eager_calls_still_enter_the_marker(self):
        self._install()
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            self._forward(x, w)

        names = {event.name for event in prof.events()}
        self.assertIn(self.seam.MARKER_PREFIX + self.target, names)


if __name__ == "__main__":
    unittest.main()
