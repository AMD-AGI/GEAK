"""Release each vLLM torch-profiler window's host memory once its trace is written.

vLLM's TorchProfilerWrapper builds a new torch.profiler.profile per window (_start) and
keeps the last one in self.profiler after _stop. That object sits in a reference cycle
(action_map holds its own bound methods), so it and its kineto_results (every recorded
event, with shapes and Python stacks: ~10-20 GB per rank per window with
record_shapes+with_stack) are freed only by a full cyclic GC. Workers gc.freeze() their
heap after warmup, which makes full collections rare, and the C++ event memory adds no GC
pressure, so every window stays resident: host MemAvailable fell ~100-170 GB per captured
point until the capture watchdog restarted the server.

The hook wraps TorchProfilerWrapper._stop. The original runs first: profiler.stop() runs
on_trace_ready, i.e. the trace export, before it returns. Only then is the profiler
dropped, a GC pass run and the freed heap returned to the OS (malloc_trim). What is
recorded and written is unchanged; the cost is a few seconds after each window closes.

GEAK_PROFILER_RELEASE=0 disables it. Loaded by sitecustomize.py in every process that
imports vllm.profiler.wrapper (the TP workers).
"""
import ctypes
import gc
import importlib.abc
import importlib.machinery
import os
import sys

MODULE = "vllm.profiler.wrapper"
MARK = "_geak_profiler_release"


def malloc_trim():
    try:
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (OSError, AttributeError):
        return False


def release(wrapper):
    """Drop the finished window's profiler and collect it; never raises."""
    try:
        wrapper.profiler = None
        gc.collect()
        malloc_trim()
    except Exception as exc:  # serving and the written trace must not depend on this
        print("GEAK: profiler release failed: %r" % (exc,), file=sys.stderr)


def patch_wrapper(module):
    cls = getattr(module, "TorchProfilerWrapper", None)
    original = getattr(cls, "_stop", None)
    if original is None:
        print("GEAK: vllm.profiler.wrapper has no TorchProfilerWrapper._stop; profiler "
              "windows are not released (host memory grows per captured window).",
              file=sys.stderr)
        return False
    if getattr(original, MARK, False):
        return True

    def _stop(self):
        try:
            return original(self)
        finally:
            release(self)

    _stop.__name__ = original.__name__
    _stop.__qualname__ = getattr(original, "__qualname__", "_stop")
    _stop.__doc__ = original.__doc__
    _stop.__wrapped__ = original
    setattr(_stop, MARK, True)
    cls._stop = _stop
    return True


class WrapperFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        original = spec.loader

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                create = getattr(original, "create_module", None)
                return create(spec) if create else None

            def exec_module(self, module):
                original.exec_module(module)
                patch_wrapper(module)

        spec.loader = Loader()
        return spec


def install():
    if os.environ.get("GEAK_PROFILER_RELEASE", "1") == "0":
        return
    if MODULE in sys.modules:
        patch_wrapper(sys.modules[MODULE])
    elif not any(isinstance(finder, WrapperFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, WrapperFinder())
