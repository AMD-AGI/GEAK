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

Version guard. TorchProfilerWrapper is private vLLM API (written against
vllm 0.0.0+mum.14d1ec2336). The hook patches only when the class still has the shape it
relies on: a plain _stop(self) and _start/_stop that use self.profiler (the attribute the
release clears). Otherwise, or if patching fails, nothing is patched, one "GEAK: profiler
release INACTIVE" line names the reason on stderr (the server log) and serving continues;
it never raises into the server. A patched process logs "GEAK: profiler release active".
STATUS holds the same outcome for the importing process.

Remove this module once vLLM releases the finished window itself (TorchProfilerWrapper._stop
dropping self.profiler after the trace export, or breaking the profiler's reference cycle);
that fix is to be proposed upstream. Until then a release on top of such a fix is harmless.
"""
import ctypes
import gc
import importlib.abc
import importlib.machinery
import inspect
import os
import sys

MODULE = "vllm.profiler.wrapper"
MARK = "_geak_profiler_release"
ATTR = "profiler"  # the TorchProfilerWrapper attribute holding the window's torch profiler

# Outcome in this process: active is True once _stop is wrapped; reason says why not.
STATUS = {"active": False, "reason": "not installed"}
_WARNED = set()


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


def shape_problem(cls):
    """None when cls has the shape the release relies on, else the reason it does not."""
    if cls is None:
        return "vllm.profiler.wrapper has no TorchProfilerWrapper"
    stop = getattr(cls, "_stop", None)
    if stop is None:
        return "TorchProfilerWrapper has no _stop"
    if not inspect.isfunction(stop):
        return "TorchProfilerWrapper._stop is a %s, not a plain function" % type(stop).__name__
    try:
        params = list(inspect.signature(stop).parameters.values())
    except (TypeError, ValueError) as exc:
        return "TorchProfilerWrapper._stop signature unreadable (%r)" % (exc,)
    if len(params) != 1 or params[0].kind not in (inspect.Parameter.POSITIONAL_ONLY,
                                                  inspect.Parameter.POSITIONAL_OR_KEYWORD):
        return "TorchProfilerWrapper._stop%s is not _stop(self)" % (inspect.signature(stop),)
    for name in ("_start", "_stop"):
        code = getattr(getattr(cls, name, None), "__code__", None)
        if code is None or ATTR not in code.co_names:
            return "TorchProfilerWrapper.%s does not use self.%s" % (name, ATTR)
    return None


def vllm_version():
    try:
        return str(getattr(sys.modules.get("vllm"), "__version__", None) or "unknown")
    except Exception:  # a lazy module attribute must not break the import being patched
        return "unknown"


def inactive(reason):
    """Record and report (once per reason and process) that nothing was patched."""
    STATUS.update(active=False, reason=reason)
    if reason not in _WARNED:
        _WARNED.add(reason)
        print("GEAK: profiler release INACTIVE (vllm %s): %s; the leak fix is not applied, "
              "profiler windows are not released (host memory grows per captured window). "
              "Serving and traces are unaffected." % (vllm_version(), reason), file=sys.stderr)
    return False


def patch_wrapper(module):
    """Wrap TorchProfilerWrapper._stop; True when patched. Never raises."""
    try:
        cls = getattr(module, "TorchProfilerWrapper", None)
        original = getattr(cls, "_stop", None)
        if getattr(original, MARK, False):
            return True
        problem = shape_problem(cls)
        if problem:
            return inactive(problem)

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
    except Exception as exc:  # a vLLM we do not know must still serve
        return inactive("patching TorchProfilerWrapper._stop failed: %r" % (exc,))
    STATUS.update(active=True, reason=None)
    print("GEAK: profiler release active (vllm %s): TorchProfilerWrapper._stop releases "
          "each window after its trace is written." % vllm_version(), file=sys.stderr)
    return True


class WrapperFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MODULE:
            return None
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        except Exception as exc:  # leave the import to the regular finders
            inactive("cannot locate %s (%r)" % (MODULE, exc))
            return None
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
        STATUS.update(active=False, reason="disabled by GEAK_PROFILER_RELEASE=0")
        return
    if MODULE in sys.modules:
        patch_wrapper(sys.modules[MODULE])
    elif not any(isinstance(finder, WrapperFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, WrapperFinder())
