#!/usr/bin/env python3
"""Capture real serving shapes + a reference I/O oracle for a hot kernel.

The Kernel Extractor uses this to turn a profiled hot kernel into a standalone, IMMUTABLE unittest
the single-kernel kernel_workflow can optimize. It hooks the target callable inside a live sglang
server process (via the sitecustomize/monkeypatch overlay mechanism), records (args, kwargs)->output
for the first few DISTINCT input-shape signatures seen during a short bench window, and writes a
torch-loadable `reference_io.pt` + `meta.json`.

This module is meant to be imported at server startup through an overlay PYTHONPATH (it registers the
hook on import), OR called as a function from a custom preimport. It does NOT launch the server
itself — pair it with scripts/bench_e2e.sh (drive the same workload as the profile so shapes match
the regime).

Usage pattern (Extractor writes an overlay sitecustomize.py like):
    import capture_shapes
    capture_shapes.install(
        target="sglang.srt.layers.activation:silu_and_mul",  # module:attr to wrap
        out_dir="/path/exp/<kernel>_task",
        max_cases=5,
    )
Then launch the server with PYTHONPATH=<overlay>:$PYTHONPATH and run a short bench. On process exit
(atexit) the records are flushed to <out_dir>/reference_io.pt + meta.json.

Anti-cheating: the oracle is captured from the UNMODIFIED baseline kernel. The optimizer later must
match it. The unittest + this file's outputs must not be edited during optimization.
"""
import atexit, functools, importlib, json, os, sys, threading

_STATE = {
    "target": None, "out_dir": None, "max_cases": 5, "num_steps": 0,
    "records": [], "seen": set(), "lock": threading.Lock(), "orig": None,
    "mod": None, "attr": None, "installed": False, "calls": 0,
    # regime coverage for the oracle: the classic failure is a single-case oracle (only ONE shape recorded,
    # e.g. one decode step), which under-tests correctness. We guarantee at least one case per regime
    # (decode vs prefill) even if that overshoots max_cases, so the immutable oracle exercises BOTH the q=1
    # decode path and the big-M prefill path. decode_lead_max is the eager decode/prefill cutoff on the
    # leading (token/batch) dim: decode's eager leading dim is the running-BATCH (num_seqs, up to
    # max_num_seqs), NOT 1 — a cutoff of 8 misclassified any batched decode as prefill and never captured a
    # decode oracle case under load. Default 256 (a typical max_num_seqs) catches batched decode while most
    # prefill chunks (>=512) stay prefill; override via CAPTURE_DECODE_LEAD_MAX for a smaller chunk budget.
    "regime_seen": set(), "decode_lead_max": int(os.environ.get("CAPTURE_DECODE_LEAD_MAX", "256")),
    # temporal fidelity (for the graph-replay / interleave UT — hole #2):
    "sequence": [], "seq_cap": 256, "in_graph_calls": 0,
    # complete shape histogram (ALL distinct shapes + call count = real workload weight). Unbounded in
    # distinct shapes (there are only a handful in practice); light — shapes/dtypes only, no tensor data.
    # Separate from the heavy oracle `records` (capped at max_cases for memory): every distinct shape is
    # counted here even when its full I/O is not saved.
    "shape_counts": {}, "shape_meta": {},
    # crash resilience: flush periodically, not only at atexit (OOM/SIGKILL never fires atexit, losing
    # a whole capture). `oracle_records` = records already on disk; a late regime-coverage case appended
    # past max_cases makes the oracle stale and triggers a rewrite so it never disagrees with meta.json.
    "flush_every": 64, "oracle_written": False, "oracle_sha": None, "oracle_records": 0,
}


def _shapes_dtypes(args, kwargs):
    """Light shape/dtype walk (no clone) so we can catalog EVERY distinct shape cheaply, independent of
    the memory-bounded oracle capture."""
    torch = _torch()
    shapes, dtypes = [], []
    def walk(o):
        if torch.is_tensor(o):
            shapes.append(list(o.shape)); dtypes.append(str(o.dtype))
        elif isinstance(o, (list, tuple)):
            for v in o: walk(v)
        elif isinstance(o, dict):
            for v in o.values(): walk(v)
    for a in args: walk(a)
    for v in kwargs.values(): walk(v)
    return {"input_shapes": shapes, "input_dtypes": sorted(set(dtypes))}


def _lead_regime(args, kwargs):
    """Coarse regime (decode vs prefill) of a call, used to tag oracle cases so the oracle covers BOTH
    regimes. Oracle records are captured only EAGERLY (a snapshot during CUDA-graph capture is illegal),
    so under a graph-on regime the recordable decode cases are the eager ones (server warmup / enforce-
    eager / non-graph ops); this classifies them by the first tensor operand's leading (token/batch) dim
    — <= decode_lead_max is decode, larger is prefill. The cutoff is fuzzy: decode's eager leading dim is
    the running BATCH (num_seqs), which can overlap a small prefill chunk, so decode_lead_max defaults to
    a typical max_num_seqs and is env-overridable.

    NOTE: this tag is written onto each oracle record and IS consumed downstream for weighting
    (attribute_weights._distribute splits profiled TIME by the per-case regime for case-based op_kinds),
    not merely for coverage — so a misclassification shifts the decode/prefill weight split."""
    torch = _torch()
    def first(o):
        if torch.is_tensor(o):
            return o
        if isinstance(o, (list, tuple)):
            for v in o:
                t = first(v)
                if t is not None:
                    return t
        return None
    t = first(list(args) + list(kwargs.values()))
    if t is None or t.dim() == 0:
        return "decode"
    return "decode" if int(t.shape[0]) <= _STATE["decode_lead_max"] else "prefill"


def _capturing():
    """True if this call is issued while a CUDA graph is being captured — i.e. the op runs inside the
    server's replayed graph in deployment, so the isolated UT MUST test capture-once/replay-many, not
    just single-shape eager. Guarded: older torch lacks the query."""
    try:
        torch = _torch()
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _torch():
    import torch
    return torch


def _snapshot(x):
    """Detach+clone tensors to CPU so later in-place ops can't corrupt the oracle. Pass scalars/None
    through; summarize unsupported objects by repr so the record stays loadable."""
    torch = _torch()
    if torch.is_tensor(x):
        return {"__tensor__": True, "data": x.detach().to("cpu").clone(),
                "dtype": str(x.dtype), "device": str(x.device),
                "shape": list(x.shape), "contiguous": bool(x.is_contiguous())}
    if isinstance(x, (list, tuple)):
        return type(x)(_snapshot(v) for v in x)
    if isinstance(x, dict):
        return {k: _snapshot(v) for k, v in x.items()}
    if isinstance(x, (int, float, bool)) or x is None:
        return x
    return {"__repr__": repr(x)[:200]}


def _sig(args, kwargs):
    torch = _torch()
    parts = []
    for a in args:
        if torch.is_tensor(a):
            parts.append(f"T{tuple(a.shape)}:{a.dtype}")
        elif isinstance(a, (int, float, bool)) or a is None:
            parts.append(repr(a))
        else:
            parts.append(type(a).__name__)
    for k in sorted(kwargs):
        v = kwargs[k]
        if torch.is_tensor(v):
            parts.append(f"{k}=T{tuple(v.shape)}:{v.dtype}")
        else:
            parts.append(f"{k}={v if isinstance(v,(int,float,bool,type(None))) else type(v).__name__}")
    return "|".join(parts)


def _wrapper(*args, **kwargs):
    s = _STATE
    out = s["orig"](*args, **kwargs)
    s["calls"] += 1
    in_graph = _capturing()
    try:
        sig = _sig(args, kwargs)
        with s["lock"]:
            if in_graph:
                s["in_graph_calls"] += 1
            # complete histogram: count EVERY call at EVERY shape (uncapped) — this is the real weight
            s["shape_counts"][sig] = s["shape_counts"].get(sig, 0) + 1
            if sig not in s["shape_meta"]:
                s["shape_meta"][sig] = _shapes_dtypes(args, kwargs)
            if len(s["sequence"]) < s["seq_cap"]:
                s["sequence"].append({"sig": sig, "in_graph": in_graph})
            # Record the heavy oracle ONLY from eager calls: a snapshot during CUDA-graph capture would
            # (1) do an illegal device sync inside capture and (2) clone placeholder data (capture records
            # ops, not values). Counting/sequence above are sync-free and safe to keep. The same shape
            # recurs eagerly (prefill/warmup), so the oracle is not lost.
            # Guarantee regime coverage: record a distinct sig if there's a free slot OR its regime
            # (decode/prefill) is not yet represented — the latter overrides the max_cases cap so the
            # oracle never freezes on a single regime (the "single-case oracle" bug).
            regime = _lead_regime(args, kwargs)
            need_regime = regime not in s["regime_seen"]
            if sig not in s["seen"] and not in_graph and (len(s["records"]) < s["max_cases"] or need_regime):
                s["seen"].add(sig)
                s["regime_seen"].add(regime)
                s["records"].append({
                    "sig": sig,
                    "regime": regime,
                    "args": _snapshot(args),
                    "kwargs": _snapshot(kwargs),
                    "output": _snapshot(out),
                })
                sys.stderr.write(f"[capture_shapes] recorded case {len(s['records'])} ({regime}): {sig}\n")
    except Exception as e:  # never break the server because capture failed
        sys.stderr.write(f"[capture_shapes] capture error (ignored): {e}\n")
    # crash-resilient incremental flush (OUTSIDE the lock; best-effort, never breaks the server).
    try:
        _maybe_flush(in_graph)
    except Exception as e:
        sys.stderr.write(f"[capture_shapes] periodic flush error (ignored): {e}\n")
    return out


def _maybe_flush(in_graph=False):
    """Called after every wrapped call. Rewrite the light meta.json every `flush_every` calls, and write
    the heavy reference_io.pt whenever the on-disk oracle is behind the in-memory records (even before
    `max_cases` is reached) — so a small workload with fewer distinct shapes than `max_cases` (the common
    single-/few-shape decode case) is NOT left with no oracle on disk until atexit. A later OOM/SIGKILL
    (which never fires atexit) then still leaves a usable partial capture, and a late regime-coverage case
    (appended past max_cases) is not lost. `records` is bounded (max_cases + a couple regime-coverage
    cases), so this rewrites the oracle only a handful of times over the whole capture.

    NEVER flush while the server is capturing a CUDA graph: the oracle write does a device sync / host
    copy, which is ILLEGAL inside graph capture and would corrupt the server's decode-graph capture. We
    just skip this boundary — the next eager call (or atexit) flushes. Cheap, and the window has many
    eager calls."""
    if in_graph:
        return
    s = _STATE
    n = s["calls"]
    if not n or (n % max(1, s["flush_every"])) != 0:
        return
    write_oracle = len(s["records"]) > s["oracle_records"]
    _flush(write_oracle=write_oracle)


def _flush(write_oracle=True):
    s = _STATE
    if not s["records"] and not s["shape_counts"]:
        sys.stderr.write("[capture_shapes] no records captured; nothing to flush\n")
        return
    torch = _torch()
    out_dir = s["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    # Lock-safe snapshots of the concurrently-mutated dicts (this runs OUTSIDE the wrapper lock, so a
    # live serving thread may be adding shapes) — avoids 'dict changed size during iteration'.
    with s["lock"]:
        shape_counts = dict(s["shape_counts"])
        shape_meta = dict(s["shape_meta"])
        records = list(s["records"])
    io_path = os.path.join(out_dir, "reference_io.pt")
    # (Re)freeze the oracle only when the on-disk copy is behind the records, so both an early small-
    # workload capture (< max_cases distinct shapes) and a late regime-coverage case (appended past
    # max_cases) land on disk; records is bounded, so this rewrites only a handful of times.
    if write_oracle and records and len(records) > s["oracle_records"]:
        torch.save({"target": s["target"], "records": records}, io_path)
        import hashlib
        h = hashlib.sha256()
        with open(io_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        s["oracle_sha"] = h.hexdigest()
        s["oracle_written"] = True
        s["oracle_records"] = len(records)
    cases = []
    for r in records:
        shapes, dtypes = [], []
        def walk(o):
            if isinstance(o, dict) and o.get("__tensor__"):
                shapes.append(o["shape"]); dtypes.append(o["dtype"])
            elif isinstance(o, (list, tuple)):
                for v in o: walk(v)
            elif isinstance(o, dict):
                for v in o.values(): walk(v)
        walk(r["args"]); walk(r["kwargs"])
        cases.append({"sig": r["sig"], "regime": r.get("regime", ""),
                      "input_shapes": shapes, "input_dtypes": sorted(set(dtypes)),
                      "count": shape_counts.get(r["sig"], 0)})
    # complete shape histogram (ALL distinct shapes seen, uncapped), sorted by frequency = weight.
    shape_hist = sorted(
        ({"sig": k, "count": v, **shape_meta.get(k, {})} for k, v in shape_counts.items()),
        key=lambda e: e["count"], reverse=True)
    # Temporal fidelity (hole #2): the ordered, WITH-repeats call sequence + whether the op runs inside
    # a captured CUDA graph. num_distinct_shapes>1 means the deployment interleaves shapes
    # (chunked-prefill ⇄ decode); graph_replayed=True means decode runs under the server's replayed
    # graph. The Extractor uses these to decide whether the UT MUST add h.check_correct_sequence
    # (interleave) and h.check_graph_replay (capture-once/replay-many with a reused static buffer),
    # not just single-shape h.check_correct_multi.
    meta = {
        "target": s["target"],
        "module": s["mod"].__name__ if s["mod"] else None,
        "attr": s["attr"],
        "num_cases": len(records),
        "total_calls_observed": s["calls"],
        "regimes_covered": sorted(s["regime_seen"]),
        "cases": cases,
        "shape_counts": shape_hist,
        "num_distinct_shapes": len(shape_counts),
        "call_sequence": s["sequence"],
        "graph_replayed": bool(s["in_graph_calls"] > 0),
        "in_graph_calls": s["in_graph_calls"],
        "reference_io": "reference_io.pt",
        "reference_io_sha256": s["oracle_sha"],   # None until the oracle file is written (partial flush)
        "oracle_complete": bool(s["oracle_written"]),
        "build": False,  # default: pure-python/triton; Extractor flips to True for HIP/CK/asm tasks
        "note": "Oracle captured from baseline. Do NOT edit unittest.py or reference_io.pt during opt.",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    sys.stderr.write(f"[capture_shapes] flushed {len(records)} case(s) "
                     f"(regimes={sorted(s['regime_seen'])}), "
                     f"oracle_complete={s['oracle_written']} -> {out_dir}\n")


def _wrappable(orig):
    """True if `orig` is a plain Python callable we can transparently stand in for. A bare-function
    stand-in for a NATIVE callable (C/builtin `builtin_function_or_method`, or a triton `JITFunction`
    whose caller reads `.fn`/`.cache`/`.warmup` off the object) is the mxfp4 `matmul_ogs` SIGSEGV: the
    native dispatch reads attributes/uses a calling convention a Python wrapper doesn't provide, and the
    missing-attribute access faults in C rather than raising. So we only replace pure-Python functions/
    methods (which introspection follows via __wrapped__); anything else must be hooked at a Python-level
    seam instead."""
    import types
    if isinstance(orig, (types.FunctionType, types.MethodType, functools.partial)):
        return True
    # native C/builtin callable -> a plain-function stand-in changes the calling convention -> unsafe
    if isinstance(orig, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return False
    # unknown object exposing triton-JIT internals -> caller reads them off the object -> unsafe
    if any(hasattr(orig, a) for a in ("fn", "cache", "warmup", "run", "__torch_dispatch__")):
        return False
    # a plain callable instance defined in Python is fine; anything else is treated as unsafe
    return callable(orig) and type(orig).__module__ != "builtins"


def _make_wrapper(orig):
    """Build the recording wrapper, transparently mirroring `orig` so introspection-driven native
    dispatch (inspect.signature via __wrapped__, attribute reads) still works — the root fix for
    'wrapping the callable SIGSEGVs'. functools.wraps copies __name__/__qualname__/__module__/__doc__/
    __dict__ and sets __wrapped__=orig; we also mirror __signature__ when resolvable and copy any extra
    public attributes the original carries so an attribute read on the wrapper doesn't fall through to a
    C-level fault."""
    @functools.wraps(orig)
    def _w(*args, **kwargs):
        return _wrapper(*args, **kwargs)
    try:
        import inspect
        _w.__signature__ = inspect.signature(orig)
    except (ValueError, TypeError):
        pass
    for a in dir(orig):
        if a.startswith("__"):
            continue
        if not hasattr(_w, a):
            try:
                setattr(_w, a, getattr(orig, a))
            except (AttributeError, TypeError):
                pass
    return _w


def install(target, out_dir, max_cases=5):
    """Wrap module:attr to record I/O. Registers an atexit flush. Idempotent.

    Fails FAST at install (server startup) if the target is a native/non-Python callable that a plain
    Python wrapper cannot safely stand in for — converting the old unpredictable mid-run SIGSEGV (which
    took the whole server down and lost the run) into a clear, actionable startup error so the Extractor
    picks a Python-level seam. Override with CAPTURE_WRAP_UNSAFE=1 to force (e.g. when the caller only
    reads shapes, never the JIT internals)."""
    s = _STATE
    if s["installed"]:
        return
    mod_name, attr = target.split(":")
    mod = importlib.import_module(mod_name)
    orig = getattr(mod, attr)
    if not _wrappable(orig) and os.environ.get("CAPTURE_WRAP_UNSAFE", "0") != "1":
        raise RuntimeError(
            f"[capture_shapes] refusing to wrap non-Python callable {target} "
            f"({type(orig).__module__}.{type(orig).__name__}): a plain-function stand-in for a native/"
            f"triton-JIT callable SIGSEGVs the server (e.g. mxfp4 matmul_ogs). Hook a Python-level seam "
            f"(its caller) instead, or set CAPTURE_WRAP_UNSAFE=1 to force.")
    s.update(target=target, out_dir=out_dir, max_cases=int(max_cases),
             orig=orig, mod=mod, attr=attr, installed=True)
    setattr(mod, attr, _make_wrapper(orig))
    atexit.register(_flush)
    sys.stderr.write(f"[capture_shapes] hooked {target}; recording up to {max_cases} cases -> {out_dir}\n")


# Allow configuration purely via env (so a generic overlay sitecustomize can call install()):
#   CAPTURE_TARGET=module:attr  CAPTURE_OUT=/path  CAPTURE_MAX=5
def install_from_env():
    t = os.environ.get("CAPTURE_TARGET")
    o = os.environ.get("CAPTURE_OUT")
    if t and o:
        install(t, o, int(os.environ.get("CAPTURE_MAX", "5")))


if os.environ.get("CAPTURE_TARGET") and os.environ.get("CAPTURE_OUT"):
    try:
        install_from_env()
    except Exception as e:
        sys.stderr.write(f"[capture_shapes] install_from_env failed: {e}\n")
