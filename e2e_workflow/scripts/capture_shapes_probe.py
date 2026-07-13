#!/usr/bin/env python3
"""Per-shape CALL-COUNT probe for hot serving kernels (CUDA-graph-safe shape recovery).

Sibling of `capture_shapes.py`. That module captures a few DISTINCT input shapes + a heavy I/O
oracle (for the single-kernel extractor). THIS module does something different and lighter: it wraps
one or more hot kernels inside a live server process and records, for the WHOLE benchmark window, the
FULL per-shape call-count distribution — the data that the torch/rocprof profiler loses because the
kernels execute inside a replayed CUDA graph (see the plan doc §2: 99% of GPU time shows `dims=[]`).

Why a separate file (do NOT edit capture_shapes.py's semantics):
  - No `max_cases` cap  -> accumulate every distinct shape's count, unbounded in distinct-shape space
    (bounded in practice by the CUDA-graph batch-size buckets).
  - No I/O snapshot     -> never detach/clone tensors (that is heavy and only needed for an oracle);
    only read `.shape`/`.dtype`. Keeps the wrapper cheap enough to sit on a hot path.
  - Structured dims     -> record `[[d0,d1,...], ...]` per tensor arg directly, so downstream gets the
    schema's `dims` without reparsing a string signature.
  - Per-pid flush       -> vLLM runs APIServer/EngineCore as separate processes; each flushes to
    `probe_<pid>.json` so they don't clobber each other. A post-processor merges them.

Hooking mechanism is identical to capture_shapes: `install(target, out_dir)` does
`setattr(module, attr, wrapper)`, so a function-local `from mod import fn` performed LATER by the
server still binds to the wrapped version (plan §6.2 constraint 1). Install from an overlay
sitecustomize.py at interpreter startup (before vLLM imports anything).

Usage (overlay sitecustomize.py):
    import os, capture_shapes_probe as P
    out = os.environ["PROBE_OUT"]
    for tgt in ["triton_kernels.matmul_ogs:matmul_ogs",
                "aiter.ops.triton.unified_attention:unified_attention",
                "vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe:pack_bitmatrix"]:
        P.install(tgt, out)

Each target flushes `<out>/probe_<pid>_<safe_target>.json` on process exit.
"""
import atexit, importlib, json, os, sys, threading

# One record per installed target. Keyed by the "module:attr" target string.
_TARGETS = {}
_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _torch():
    import torch
    return torch


def _iter_tensor_args(args, kwargs):
    """Yield (label, tensor) for every tensor in positional args AND kwargs. kwargs MUST be scanned:
    e.g. vLLM calls unified_attention(q=..., k=..., v=...) entirely by keyword, so a positional-only
    scan records dims=[] (observed). Positional args get index labels, kwargs get their name; kwargs
    are visited in sorted order for a deterministic signature."""
    torch = _torch()
    for i, a in enumerate(args):
        if torch.is_tensor(a):
            yield (f"arg{i}", a)
        elif isinstance(a, (list, tuple)):
            for j, v in enumerate(a):
                if torch.is_tensor(v):
                    yield (f"arg{i}[{j}]", v)
    for k in sorted(kwargs):
        v = kwargs[k]
        if torch.is_tensor(v):
            yield (k, v)
        elif isinstance(v, (list, tuple)):
            for j, vv in enumerate(v):
                if torch.is_tensor(vv):
                    yield (f"{k}[{j}]", vv)


def _shape_sig(args, kwargs):
    """Deterministic key over ALL tensor operands' shapes+dtypes (positional + kwargs)."""
    parts = []
    for label, t in _iter_tensor_args(args, kwargs):
        parts.append(f"{label}=T{tuple(t.shape)}:{t.dtype}")
    return "|".join(parts) if parts else "<no-tensor-args>"


# Per-shape GPU timing via cuda.Event. Only meaningful when CUDA graph is OFF (enforce_eager): with
# graphs ON the kernel runs on graph REPLAY and this Python wrapper isn't even on the replay path, so
# timing here would only see the capture-phase call. Enable with PROBE_TIME=1. Timing adds a
# per-call event-record + a periodic synchronize, so it is opt-in.
_TIME = os.environ.get("PROBE_TIME", "0") == "1"


def _make_wrapper(target):
    st = _TARGETS[target]
    orig = st["orig"]

    def wrapper(*args, **kwargs):
        if _TIME:
            torch = _torch()
            try:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                out = orig(*args, **kwargs)
                ev1.record()
            except Exception:
                # timing unavailable (e.g. no cuda) -> fall back to untimed
                out = orig(*args, **kwargs)
                ev0 = ev1 = None
        else:
            out = orig(*args, **kwargs)  # run the real kernel first; never affect correctness
            ev0 = ev1 = None
        try:
            st["calls"] += 1
            sig = _shape_sig(args, kwargs)
            with _LOCK:
                case = st["cases"].get(sig)
                if case is None:
                    # dims / dtypes / labels are PARALLEL lists, one entry per tensor operand, in the
                    # same order. dtypes is NOT deduplicated: downstream (e.g. random-input generation
                    # for kernel opt verification) needs each tensor's own (shape, dtype) — a merged
                    # dtype set would be ambiguous when operands differ (e.g. bf16 act + fp32 scale).
                    dims, dtypes, labels = [], [], []
                    for _label, t in _iter_tensor_args(args, kwargs):
                        dims.append(list(t.shape))
                        dtypes.append(str(t.dtype))
                        labels.append(_label)
                    case = {"dims": dims, "dtypes": dtypes, "arg_labels": labels, "count": 0,
                            "gpu_ms_sum": 0.0, "timed_count": 0, "_pending": []}
                    st["cases"][sig] = case
                case["count"] += 1
                if ev0 is not None:
                    # can't call elapsed_time until ev1 completes; stash and drain lazily
                    case["_pending"].append((ev0, ev1))
        except Exception as e:  # a probe must NEVER break the server
            sys.stderr.write(f"[probe] capture error on {target} (ignored): {e}\n")
        return out

    return wrapper


def _drain_timing(st):
    """Resolve completed cuda.Event pairs into accumulated GPU ms. Called from the periodic flusher so
    the hot wrapper never blocks on synchronize. Only pairs whose end event is done are consumed.

    The FIRST resolved sample per shape is DISCARDED: the first call to a kernel with a new shape
    pays one-time cost (CUDA context init, kernel JIT/autotune) that would badly skew the average on
    small samples. Steady-state serving runs each shape thousands of times, so dropping one is
    negligible and removes the warmup outlier."""
    for case in st["cases"].values():
        pend = case.get("_pending")
        if not pend:
            continue
        still = []
        for ev0, ev1 in pend:
            try:
                if ev1.query():   # end event finished -> safe to read elapsed_time
                    if not case.get("_warmup_dropped"):
                        case["_warmup_dropped"] = True   # discard the first (warmup) sample
                    else:
                        case["gpu_ms_sum"] += ev0.elapsed_time(ev1)
                        case["timed_count"] += 1
                else:
                    still.append((ev0, ev1))
            except Exception:
                pass  # drop unreadable pair
        case["_pending"] = still


def _flush_one(target):
    st = _TARGETS[target]
    if _TIME:
        try:
            _drain_timing(st)
        except Exception:
            pass
    out_dir = st["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    safe = target.replace(":", "__").replace(".", "_")
    path = os.path.join(out_dir, f"probe_{os.getpid()}_{safe}.json")
    cases = []
    for c in sorted(st["cases"].values(), key=lambda c: c["count"], reverse=True):
        rec = {"dims": c["dims"], "dtypes": c["dtypes"],
               "arg_labels": c.get("arg_labels", []), "count": c["count"]}
        tc = c.get("timed_count", 0)
        if tc:
            rec["gpu_us_avg"] = round(c["gpu_ms_sum"] / tc * 1000.0, 3)  # measured per-call GPU time
            rec["timed_count"] = tc
        cases.append(rec)
    payload = {
        "target": target,
        "pid": os.getpid(),
        "total_calls": st["calls"],
        "num_distinct_shapes": len(cases),
        "timing": bool(_TIME),
        "cases": cases,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    sys.stderr.write(f"[probe] flushed {target} pid={os.getpid()} "
                     f"({len(cases)} distinct shapes, {st['calls']} calls, timing={_TIME}) -> {path}\n")


def _flush_all():
    """Snapshot every target that recorded calls to disk (idempotent overwrite). Safe to call
    repeatedly. Skips targets with 0 calls so the many short-lived helper processes (compile
    workers, mp children) that import the overlay but never run kernels don't flood the output
    dir (observed 46k+ empty files)."""
    for target in list(_TARGETS):
        if _TARGETS[target]["calls"] <= 0:
            continue
        try:
            _flush_one(target)
        except Exception as e:
            sys.stderr.write(f"[probe] flush error on {target} (ignored): {e}\n")


_FLUSHER_STARTED = False


def _start_periodic_flush(interval=5.0):
    """Robust persistence independent of the exit path. vLLM installs its OWN SIGTERM handler on
    EngineCore (core.py:1221) that overwrites ours, and the child process teardown does not reliably
    run atexit — so relying on exit-time flush loses the data. Instead a daemon thread snapshots the
    accumulated counts every few seconds (overwriting the same per-pid file). Whatever the final call
    count, the last snapshot is on disk. atexit is still registered as a best-effort final write."""
    global _FLUSHER_STARTED
    if _FLUSHER_STARTED:
        return
    _FLUSHER_STARTED = True

    def _loop():
        import time
        while True:
            time.sleep(interval)
            try:
                _flush_all()
            except Exception:
                pass

    t = threading.Thread(target=_loop, name="probe-flusher", daemon=True)
    t.start()


# module_name -> list of (attr, target) pending a lazy hook once that module is imported.
_PENDING = {}
_FINDER_INSTALLED = False


def _try_hook(target):
    """setattr-wrap target IF its module is already in sys.modules. Returns True if hooked.
    NEVER imports the module itself — that is the whole point (eager import of a heavy lib like
    aiter on the EngineCore handshake path blocks startup; see plan §6.3)."""
    st = _TARGETS[target]
    if st.get("orig") is not None:
        return True  # already hooked
    mod = sys.modules.get(st["mod_name"])
    if mod is None:
        return False
    try:
        orig = getattr(mod, st["attr"])
    except AttributeError:
        return False
    # Only wrap PLAIN python callables. A triton @jit JITFunction is called as fn[grid](...); a普通
    # function wrapper would break that launch syntax and crash the server. Refuse and disable the
    # target instead (mark hooked so we stop retrying). See #5 pack_bitmatrix.
    if type(orig).__name__ == "JITFunction" or not callable(orig):
        st["orig"] = orig            # mark resolved so the finder stops sweeping it
        st["unhookable"] = True
        sys.stderr.write(f"[probe] SKIP {target}: not a plain callable "
                         f"({type(orig).__name__}); cannot wrap safely\n")
        return True
    st["orig"] = orig
    st["mod"] = mod
    setattr(mod, st["attr"], _make_wrapper(target))
    sys.stderr.write(f"[probe] hooked {target} (lazy) -> {st['out_dir']}\n")
    return True


class _HookFinder:
    """A sys.meta_path finder that never claims to load anything (returns None from find_spec) but
    uses the import event as a trigger to setattr-wrap any pending target whose module just finished
    importing. Passive: it only observes, letting the real importers do the work."""

    def find_module(self, name, path=None):  # legacy API, harmless
        return None

    def find_spec(self, name, path=None, target=None):
        # A target module may become available now (this import) or as a side effect of it.
        try:
            for tgt in list(_PENDING.get(name, [])):
                _try_hook(tgt[1])
            # also sweep any still-unhooked targets (their module may have been imported indirectly)
            for tgt, st in _TARGETS.items():
                if st.get("orig") is None:
                    _try_hook(tgt)
        except Exception as e:
            sys.stderr.write(f"[probe] finder sweep error (ignored): {e}\n")
        return None  # never handle the import


def install(target, out_dir):
    """Register target for a LAZY hook. Does NOT import target's module. The wrap happens the first
    time the module appears in sys.modules (detected via a passive meta-path finder, or immediately
    if it is already imported). Idempotent per target."""
    global _ATEXIT_REGISTERED, _FINDER_INSTALLED
    if target in _TARGETS:
        return
    mod_name, attr = target.split(":")
    _TARGETS[target] = {"out_dir": out_dir, "orig": None, "mod": None,
                        "mod_name": mod_name, "attr": attr, "cases": {}, "calls": 0}
    _PENDING.setdefault(mod_name, []).append((attr, target))

    if not _FINDER_INSTALLED:
        sys.meta_path.insert(0, _HookFinder())
        _FINDER_INSTALLED = True
    if not _ATEXIT_REGISTERED:
        atexit.register(_flush_all)      # best-effort final write on normal interpreter exit
        _start_periodic_flush()          # primary persistence: daemon snapshots every few seconds
        _ATEXIT_REGISTERED = True

    # If the module is ALREADY imported (target came late), hook right now — no import triggered.
    hooked = _try_hook(target)
    state = "hooked now" if hooked else "pending lazy hook"
    sys.stderr.write(f"[probe] registered {target} ({state}) -> {out_dir}\n")


def install_from_env():
    """Env-driven install so a generic overlay can call one function.
    PROBE_TARGETS = comma-separated module:attr list; PROBE_OUT = output dir."""
    tgts = os.environ.get("PROBE_TARGETS", "")
    out = os.environ.get("PROBE_OUT", "")
    if tgts and out:
        for t in [x.strip() for x in tgts.split(",") if x.strip()]:
            try:
                install(t, out)
            except Exception as e:
                sys.stderr.write(f"[probe] install failed for {t}: {e}\n")
