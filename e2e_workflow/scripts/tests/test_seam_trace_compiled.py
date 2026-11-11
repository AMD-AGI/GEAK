"""seam_trace's marker inside a torch.compile(fullgraph=True) region, the way vLLM compiles a model forward.

vLLM's forward calls the marked seam (e.g. ``rocm_unquantized_gemm``) inside a graph that Dynamo traces with
``fullgraph=True``. The marker's capture query, ``torch.cuda.is_current_stream_capturing()``, returns a bool, which
Dynamo cannot put in a graph, so an untraceable marker is an engine-start failure, not a missing marker. The
marker also records that the seam was compiled, in a way vLLM's CUDA-graph bytecode check and its compile cache
accept. Needs torch; skipped where it is not installed or cannot load its runtime.
"""
import atexit
import contextlib
import glob
import importlib
import importlib.util
import io
import itertools
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

        with self.assertRaisesRegex(
            (torch._dynamo.exc.Unsupported, torch._dynamo.exc.TorchRuntimeError),
            "is_current_stream_capturing|_cuda_isCurrentStreamCapturing",
        ):
            compiled(x, w)

    def test_eager_calls_still_enter_the_marker(self):
        self._install()
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            self._forward(x, w)

        names = {event.name for event in prof.events()}
        self.assertIn(self.seam.MARKER_PREFIX + self.target, names)

    def test_fullgraph_reuses_one_graph_and_the_trace_names_the_seam(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        env = mock.patch.dict(os.environ, {"GEAK_SELECTION_TRACE": os.path.join(tmp, "sel.json")})
        env.start()
        self.addCleanup(env.stop)
        self.addCleanup(atexit.unregister, self.seam._finish_profile)
        self._install()
        graphs = []

        def backend(graph, _example_inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(self._forward, fullgraph=True, backend=backend)
        x, w = torch.randn(4, 8), torch.randn(16, 8)
        torch.testing.assert_close(compiled(x, w), x @ w.t() + 1)
        compiled(x, w)
        warm = len(graphs)
        for _ in range(10):
            compiled(x, w)
        self.assertLessEqual(warm, 2)
        self.assertEqual(len(graphs), warm)

        with contextlib.redirect_stderr(io.StringIO()):
            self.layer.gemm(x, w)             # eager: marked, and it exports the trace on return
        spec = importlib.util.spec_from_file_location(f"kernel_selection_compiled_{next(_LOADS)}",
                                                      os.path.join(SCRIPTS_DIR, "kernel_selection.py"))
        ks = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ks
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(ks)
        events = ks.merge_process_traces(glob.glob(os.path.join(tmp, "sel.pid-*.json")))
        verdict = ks.verify(self.target, "no_such_kernel", {"module": self.layer.__name__, "attr": "gemm",
                                                            "total_calls_observed": 1}, events)
        self.assertEqual(verdict["compiled_passthrough_candidates"], [self.target])
        self.assertEqual(verdict["target_marker_calls"], 1)

    def test_the_compiled_frame_passes_vllms_cudagraph_bytecode_check(self):
        # With CUDA graphs on, vLLM refuses a transformed frame that names `update` (what a dict
        # store replays as): the engine would still die at startup, only with another error.
        self._install()
        names = []
        hook = torch._dynamo.convert_frame.register_bytecode_hook(lambda _old, new: names.append(new.co_names))
        self.addCleanup(hook.remove)
        compiled = torch.compile(self._forward, fullgraph=True, backend=lambda graph, _inputs: graph.forward)
        compiled(torch.randn(4, 8), torch.randn(16, 8))
        self.assertTrue(names)
        self.assertFalse(any("update" in frame for frame in names))
        self.assertTrue(self.seam._COMPILED[self.target].seen)

    def test_a_cached_frame_that_names_a_seam_this_process_did_not_mark_still_runs(self):
        # vLLM's AOT compile cache does not key on the marker set, so a run can load a frame compiled
        # where other seams were marked; its replayed record must not raise.
        try:
            from torch._dynamo.aot_compile_types import SerializableCallable
        except ImportError:
            self.skipTest("needs torch's AOT compile (2.10+)")

        class Artifact(SerializableCallable):
            def __init__(self, graph):
                self.graph = graph

            def __call__(self, *args, **kwargs):
                return self.graph(*args, **kwargs)

            @classmethod
            def serialize_compile_artifacts(cls, fn):
                return b""

            @classmethod
            def deserialize_compile_artifacts(cls, data):
                raise NotImplementedError

        self._install()
        layer = self.layer

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return layer.gemm(a, b) + 1

        model, x, w = Model(), torch.randn(4, 8), torch.randn(16, 8)
        with torch._dynamo.config.patch(enable_aot_compile=True):
            compiled = torch.compile(model.forward, fullgraph=True, dynamic=False,
                                     backend=lambda graph, _inputs: Artifact(graph))
            frame = compiled.aot_compile(((x, w), {}))
        self.seam._COMPILED.clear()     # this process marked other seams
        torch.testing.assert_close(frame(model, x, w), x @ w.t() + 1)
        self.assertTrue(self.seam._COMPILED[self.target].seen)


if __name__ == "__main__":
    unittest.main()
