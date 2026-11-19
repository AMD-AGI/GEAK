"""Opt-in, CPU-only observations of vLLM worker scheduler outputs.

No request IDs, prompt contents, tensor values or object reprs are persisted.
The hook preserves annotate_profile's return value (including context managers).
"""
import atexit
import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import time


MODULE = "vllm.v1.worker.gpu_worker"


def observation(worker, output):
    new = {r.req_id: r for r in output.scheduled_new_reqs}
    cached = output.scheduled_cached_reqs
    computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
    rows = []
    for rid, count in output.num_scheduled_tokens.items():
        context = rid in new or cached.is_context_phase(rid)
        prior = new[rid].num_computed_tokens if rid in new else computed[rid]
        rows.append({"phase": "PREFILL" if context else "DECODE",
                     "query": int(count), "computed": int(prior)})
    return {"time": time.time(), "rank": int(worker.rank),
            "scheduled_tokens": int(output.total_num_scheduled_tokens), "requests": rows}


def patch_worker(module):
    worker = module.Worker
    original = getattr(worker, "annotate_profile", None)
    if original is None:
        print("GEAK: vLLM iteration observation unavailable: this build has no "
              "Worker.annotate_profile; serving continues without iteration evidence. "
              "An observer adapter is required for extraction.", file=sys.stderr)
        return
    if getattr(original, "_geak_iteration_observer", False):
        return
    files, stopped = {}, set()

    def close():
        for fh in files.values():
            fh.close()
    atexit.register(close)

    @functools.wraps(original)
    def observe(self, output, *args, **kwargs):
        root = os.environ.get("GEAK_ITERATION_LOG_DIR")
        if root and output.total_num_scheduled_tokens:
            key = (os.getpid(), int(self.rank))
            if key not in stopped:
                os.makedirs(root, exist_ok=True)
                if key not in files:
                    files[key] = open(os.path.join(root, "rank%d_pid%d.jsonl" % key[::-1]),
                                      "a", encoding="utf-8", buffering=1)
                fh = files[key]
                limit = int(os.environ.get("GEAK_ITERATION_LOG_MAX_BYTES", 256 * 1024 * 1024))
                try:
                    if limit <= 0 or fh.tell() >= limit:
                        raise ValueError("iteration log byte budget exhausted")
                    record = observation(self, output)
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    # Continue serving, but make truncated/incompatible evidence unusable.
                    record = {"error": "vLLM iteration observation failed: " + str(exc),
                              "time": time.time(), "rank": key[1]}
                    stopped.add(key)
                    print("GEAK: " + record["error"], file=sys.stderr)
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        return original(self, output, *args, **kwargs)

    observe._geak_iteration_observer = True
    worker.annotate_profile = observe


PROFILER_MODULE = "vllm.profiler.wrapper"


def patch_profiler(module):
    """Trace every host thread. vLLM's dual-batch overlap (--enable-dbo) runs each
    micro-batch in a thread started per step; torch's profiler records host ops only on
    threads that existed at its start, so those prefill launches had no op, no operand
    shapes and no frames (MiniMax-M3.1-Flash, vLLM 0.33). profile_all_threads restores
    their ops and shapes; their Python frames stay unrecorded."""
    wrapper = getattr(module, "TorchProfilerWrapper", None)
    original = getattr(wrapper, "_start", None)
    if original is None or getattr(original, "_geak_all_threads", False):
        return
    try:
        from torch._C._profiler import _ExperimentalConfig
        config = _ExperimentalConfig(profile_all_threads=True)
    except (ImportError, TypeError) as exc:
        print("GEAK: torch profiler cannot trace all threads (%s); launches from threads "
              "started after the window opens keep no host op" % exc, file=sys.stderr)
        return

    @functools.wraps(original)
    def start(self, *args, **kwargs):
        self._profiler_kwargs.setdefault("experimental_config", config)
        return original(self, *args, **kwargs)

    start._geak_all_threads = True
    wrapper._start = start



def wanted():
    want = {}
    if os.environ.get("GEAK_ITERATION_LOG_DIR"):
        want[MODULE] = patch_worker
    if os.environ.get("GEAK_PROFILE_ALL_THREADS") == "1":
        want[PROFILER_MODULE] = patch_profiler
    return want


class WorkerObserver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        patch = wanted().get(fullname)
        if patch is None:
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
                patch(module)

        spec.loader = Loader()
        return spec


def install():
    want = wanted()
    for name, patch in want.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if (set(want) - set(sys.modules)
            and not any(isinstance(finder, WorkerObserver) for finder in sys.meta_path)):
        sys.meta_path.insert(0, WorkerObserver())
