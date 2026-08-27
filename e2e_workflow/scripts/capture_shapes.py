#!/usr/bin/env python3
"""Capture the real serving SHAPES of a hot kernel (shapes only — no tensor payload).

The Kernel Extractor uses this to turn a profiled hot kernel into a standalone, IMMUTABLE unittest
the single-kernel kernel_workflow can optimize. It hooks the target callable inside a live sglang
server process (via the sitecustomize/monkeypatch overlay mechanism), catalogs the DISTINCT input-shape
signatures / dtypes / regimes / call order seen during a short bench window, and writes a light
`meta.json`.

It does NOT record input or output tensors, and there is no `reference_io.pt`. A stored golden was
redundant (`baseline_overlay/` already IS a runnable reference and has to exist anyway as the timing
denominator), cost 10s-100s of MB per rank per retry, and was only valid while the operands reproduced
bit-for-bit — so a box / torch-build change became a hard failure. Correctness is live parity against
that baseline on fresh in-regime draws (`harness_lib.live_oracle_cases` +
`harness_lib.check_random_vs_baseline`), the rule `kernel_workflow/roles/oracle_freezer.md` already
mandates for the kernel lane.

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
(atexit) the shape catalog is flushed to <out_dir>/meta.json.
"""
import atexit, functools, importlib, json, os, re, shutil, sys, threading

_CAPTURE_DIR_RE = re.compile(r"^capture\.pid-")
_TMP_ARTIFACT_RE = re.compile(r"\.(?:pt|json)\.tmp-\d+")

_STATE = {
    "target": None, "out_dir": None, "max_cases": 5, "num_steps": 0,
    "records": [], "seen": set(), "lock": threading.Lock(), "orig": None,
    "mod": None, "attr": None, "installed": False, "calls": 0,
    # regime coverage: the classic failure is a single-case catalog (only ONE shape recorded,
    # e.g. one decode step), which under-tests correctness. We guarantee at least one case per regime
    # (decode vs prefill) even if that overshoots max_cases, so the UT exercises BOTH the q=1
    # decode path and the big-M prefill path. decode_lead_max is the eager decode/prefill cutoff on the
    # leading (token/batch) dim: decode's eager leading dim is the running-BATCH (num_seqs, up to
    # max_num_seqs), NOT 1 — a cutoff of 8 misclassified any batched decode as prefill and never captured a
    # decode case under load. Default 256 (a typical max_num_seqs) catches batched decode while most
    # prefill chunks (>=512) stay prefill; override via CAPTURE_DECODE_LEAD_MAX for a smaller chunk budget.
    "regime_seen": set(), "decode_lead_max": int(os.environ.get("CAPTURE_DECODE_LEAD_MAX", "256")),
    # temporal fidelity (for the graph-replay / interleave UT — hole #2):
    "sequence": [], "seq_cap": 256, "in_graph_calls": 0,
    # complete shape histogram (ALL distinct shapes + call count = real workload weight). Unbounded in
    # distinct shapes (there are only a handful in practice); light — shapes/dtypes only, no tensor data.
    # Separate from `records` (capped at max_cases): every distinct shape is counted here even when it
    # does not become one of the representative UT cases.
    "shape_counts": {}, "shape_meta": {},
    # crash resilience: flush periodically, not only at atexit (OOM/SIGKILL never fires atexit, losing
    # a whole capture). Everything written is light JSON, so a flush is cheap at any cadence.
    "flush_every": 64, "meta_flush_count": 0,
}


def _rank():
    return next((str(os.environ[key]) for key in
                 ("RANK", "LOCAL_RANK", "TP_RANK", "SLURM_PROCID")
                 if key in os.environ), "unknown")


def _process_out_dir(out_dir):
    """Isolate selection-capture artifacts so TP workers cannot corrupt each other."""
    unique = (os.environ.get("CAPTURE_PROCESS_UNIQUE") == "1"
              or bool(os.environ.get("GEAK_SELECTION_TRACE")))
    if not unique:
        return out_dir
    return os.path.join(
        out_dir, f"capture.pid-{os.getpid()}.rank-{_rank()}")


def _dir_bytes(path):
    """Sum regular-file sizes under ``path`` (symlinks do not count target bytes)."""
    total = 0
    if not path or not os.path.exists(path):
        return 0
    if os.path.islink(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            fp = os.path.join(root, name)
            try:
                if os.path.islink(fp):
                    continue
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def iter_capture_dirs(task_dir):
    """Yield absolute paths of capture.pid-* directories under task_dir (any depth)."""
    task_dir = os.path.abspath(task_dir)
    if not os.path.isdir(task_dir):
        return
    for root, dirs, _files in os.walk(task_dir):
        for name in list(dirs):
            if _CAPTURE_DIR_RE.match(name):
                yield os.path.join(root, name)


def _remove_path(path, telemetry):
    if not path or not os.path.exists(path):
        return 0
    nbytes = _dir_bytes(path)
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.unlink(path)
        telemetry["removed_paths"].append(path)
        telemetry["bytes_reclaimed"] += nbytes
        return nbytes
    except OSError as exc:
        telemetry.setdefault("errors", []).append(f"{path}: {exc}")
        return 0


def _write_json(path, payload):
    tmp = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def promote_selected_meta(task_dir, meta_path):
    """Copy the selected process-local ``meta.json`` into task_dir as the sole authoritative copy.

    Nothing else is promoted: a capture dir holds only light JSON now (no ``reference_io.pt``), so the
    "which rank's multi-GiB oracle wins" question this used to arbitrate no longer exists.
    """
    task_dir = os.path.abspath(task_dir)
    meta_path = os.path.abspath(meta_path)
    os.makedirs(task_dir, exist_ok=True)
    dst_meta = os.path.join(task_dir, "meta.json")
    result = {"promoted_meta": False, "meta_path": dst_meta}
    if os.path.isfile(meta_path):
        if os.path.abspath(meta_path) != os.path.abspath(dst_meta):
            shutil.copy2(meta_path, dst_meta)
        result["promoted_meta"] = True
    return result


def cleanup_task_capture_artifacts(task_dir, keep_meta_path=None, promote=None,
                                   remove_tmp=True):
    """Reclaim process-local capture dirs under task_dir.

    When ``promote`` is true (default if ``keep_meta_path`` is set), copy the selected ``meta.json``
    into ``task_dir`` first, then delete every ``capture.pid-*`` directory (including nested
    ``_selcap*`` attempts) and leftover ``*.tmp-*`` atomic-write files. Writes
    ``capture_telemetry.json`` into ``task_dir``.

    Returns a telemetry dict with ``bytes_reclaimed``, ``removed_paths``, etc.
    """
    task_dir = os.path.abspath(task_dir)
    telemetry = {
        "task_dir": task_dir,
        "bytes_reclaimed": 0,
        "bytes_before": 0,
        "bytes_after": 0,
        "capture_dirs_seen": [],
        "removed_paths": [],
        "promoted": False,
        "keep_meta_path": keep_meta_path,
        "errors": [],
    }
    if not os.path.isdir(task_dir):
        telemetry["errors"].append(f"missing task_dir: {task_dir}")
        return telemetry

    capture_dirs = sorted(iter_capture_dirs(task_dir))
    telemetry["capture_dirs_seen"] = capture_dirs
    telemetry["bytes_before"] = sum(_dir_bytes(d) for d in capture_dirs)

    do_promote = bool(promote) if promote is not None else bool(keep_meta_path)
    if do_promote and keep_meta_path and os.path.isfile(keep_meta_path):
        promoted = promote_selected_meta(task_dir, keep_meta_path)
        telemetry["promoted"] = True
        telemetry["promote"] = promoted

    for cap_dir in capture_dirs:
        _remove_path(cap_dir, telemetry)

    if remove_tmp:
        for root, _dirs, files in os.walk(task_dir):
            for name in files:
                if ".tmp-" in name:
                    _remove_path(os.path.join(root, name), telemetry)

    remaining = sorted(iter_capture_dirs(task_dir))
    telemetry["capture_dirs_remaining"] = remaining
    telemetry["bytes_after"] = sum(_dir_bytes(d) for d in remaining)

    try:
        _write_json(os.path.join(task_dir, "capture_telemetry.json"), telemetry)
    except OSError as exc:
        telemetry["errors"].append(f"telemetry write failed: {exc}")
    return telemetry


def promote_and_reclaim(task_dir, keep_meta_path=None, promote=True):
    """Convenience wrapper used by kernel_selection after a verdict."""
    return cleanup_task_capture_artifacts(
        task_dir, keep_meta_path=keep_meta_path, promote=promote and bool(keep_meta_path))


def reclaim_workspace_captures(eval_dir):
    """Finalize-time sweep: reclaim capture.pid-* under every kernels/*_task."""
    eval_dir = os.path.abspath(eval_dir)
    kernels = os.path.join(eval_dir, "kernels")
    telemetry = {
        "eval_dir": eval_dir,
        "tasks": [],
        "bytes_reclaimed": 0,
        "errors": [],
    }
    if not os.path.isdir(kernels):
        telemetry["errors"].append(f"missing kernels dir: {kernels}")
        return telemetry
    for name in sorted(os.listdir(kernels)):
        task = os.path.join(kernels, name)
        if not os.path.isdir(task) or not name.endswith("_task"):
            continue
        task_tel = cleanup_task_capture_artifacts(task, promote=False)
        telemetry["bytes_reclaimed"] += int(task_tel.get("bytes_reclaimed") or 0)
        telemetry["tasks"].append({
            "task": task,
            "bytes_reclaimed": task_tel.get("bytes_reclaimed"),
            "capture_dirs_removed": len(task_tel.get("removed_paths") or []),
        })
    try:
        _write_json(os.path.join(eval_dir, "capture_workspace_telemetry.json"), telemetry)
    except OSError as exc:
        telemetry["errors"].append(str(exc))
    return telemetry


def _shapes_dtypes(args, kwargs):
    """Light shape/dtype walk (no clone) so we can catalog EVERY distinct shape cheaply, independent of
    the max_cases-bounded representative case list."""
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
    """Coarse regime (decode vs prefill) of a call, used to tag cases so the case list covers BOTH
    regimes. Cases are taken only from EAGER calls, so under a graph-on regime the recordable decode
    cases are the eager ones (server warmup / enforce-eager / non-graph ops); this classifies them by
    the first tensor operand's leading (token/batch) dim
    — <= decode_lead_max is decode, larger is prefill. The cutoff is fuzzy: decode's eager leading dim is
    the running BATCH (num_seqs), which can overlap a small prefill chunk, so decode_lead_max defaults to
    a typical max_num_seqs and is env-overridable.

    NOTE: this tag is written onto each recorded case and IS consumed downstream for weighting
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
    # This marker is consumed by kernel_selection.py from the capture run's torch trace.  It turns
    # "the hook saw calls" into stronger evidence: the GPU kernel selected from the baseline profile
    # must actually execute while THIS callable is active.  record_function is effectively a no-op
    # when no profiler is collecting, so existing non-profiled capture users keep the same behavior.
    marker = f"GEAK_TARGET::{s['target']}"
    try:
        record_function = _torch().profiler.record_function
    except Exception:
        record_function = None
    if record_function:
        with record_function(marker):
            out = s["orig"](*args, **kwargs)
    else:
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
            # Representative cases are taken ONLY from eager calls: a call observed during CUDA-graph
            # CAPTURE reflects the trace, not a real serving invocation, and its regime tag would skew
            # the decode/prefill weight split. Counting/sequence above stay uncapped and cover both.
            # Guarantee regime coverage: record a distinct sig if there's a free slot OR its regime
            # (decode/prefill) is not yet represented — the latter overrides the max_cases cap so the
            # case list never freezes on a single regime (the "single-case oracle" bug).
            regime = _lead_regime(args, kwargs)
            need_regime = regime not in s["regime_seen"]
            if sig not in s["seen"] and not in_graph and (len(s["records"]) < s["max_cases"] or need_regime):
                s["seen"].add(sig)
                s["regime_seen"].add(regime)
                # Shapes/dtypes only — no tensor is retained. See the module docstring: the golden is
                # the live baseline leg, not a recorded blob.
                s["records"].append(dict({"sig": sig, "regime": regime},
                                         **_shapes_dtypes(args, kwargs)))
                sys.stderr.write(
                    f"[capture_shapes] recorded case {len(s['records'])} ({regime}): {sig}\n")
    except Exception as e:  # never break the server because capture failed
        sys.stderr.write(f"[capture_shapes] capture error (ignored): {e}\n")
    # crash-resilient incremental flush (OUTSIDE the lock; best-effort, never breaks the server).
    try:
        _maybe_flush(in_graph)
    except Exception as e:
        sys.stderr.write(f"[capture_shapes] periodic flush error (ignored): {e}\n")
    return out


def _maybe_flush(in_graph=False):
    """Called after every wrapped call: rewrite the light ``meta.json`` every ``flush_every`` calls.

    NEVER flush while the server is capturing a CUDA graph — keep the capture window free of any
    incidental host work.
    """
    if in_graph:
        return
    s = _STATE
    n = s["calls"]
    if not n or (n % max(1, s["flush_every"])) != 0:
        return
    _flush()


def _flush():
    s = _STATE
    if not s["records"] and not s["shape_counts"]:
        sys.stderr.write("[capture_shapes] no records captured; nothing to flush\n")
        return
    out_dir = s["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    # Lock-safe snapshots of the concurrently-mutated dicts (this runs OUTSIDE the wrapper lock, so a
    # live serving thread may be adding shapes) — avoids 'dict changed size during iteration'.
    with s["lock"]:
        shape_counts = dict(s["shape_counts"])
        shape_meta = dict(s["shape_meta"])
        records = list(s["records"])
    cases = [{"sig": r["sig"], "regime": r.get("regime", ""),
              "input_shapes": r.get("input_shapes") or [],
              "input_dtypes": r.get("input_dtypes") or [],
              "count": shape_counts.get(r["sig"], 0)}
             for r in records]
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
        "process_id": os.getpid(),
        "rank": _rank(),
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
        "build": False,  # default: pure-python/triton; Extractor flips to True for HIP/CK/asm tasks
        "note": ("Shapes captured from the baseline server; NO tensor oracle is recorded. Correctness "
                 "is live parity vs the baseline leg (h.live_oracle_cases). Do NOT edit unittest.py "
                 "or meta.json during optimization."),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    tmp_meta = f"{meta_path}.tmp-{os.getpid()}-{threading.get_ident()}"
    try:
        with open(tmp_meta, "w") as fh:
            json.dump(meta, fh, indent=2)
        os.replace(tmp_meta, meta_path)
    except Exception:
        if os.path.exists(tmp_meta):
            try:
                os.unlink(tmp_meta)
            except OSError:
                pass
        raise
    s["meta_flush_count"] = int(s.get("meta_flush_count") or 0) + 1
    sys.stderr.write(f"[capture_shapes] flushed {len(records)} case(s) "
                     f"(regimes={sorted(s['regime_seen'])}, "
                     f"distinct_shapes={len(shape_counts)}) -> {out_dir}\n")


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
    """Wrap module:attr to catalog input shapes. Registers an atexit flush. Idempotent.

    Fails FAST at install (server startup) if the target is a native/non-Python callable that a plain
    Python wrapper cannot safely stand in for — converting the old unpredictable mid-run SIGSEGV (which
    took the whole server down and lost the run) into a clear, actionable startup error so the Extractor
    picks a Python-level seam. Override with CAPTURE_WRAP_UNSAFE=1 to force (e.g. when the caller only
    reads shapes, never the JIT internals)."""
    s = _STATE
    if s["installed"]:
        return
    mod_name, attr = target.split(":", 1)
    mod = importlib.import_module(mod_name)
    # attr may be dotted (e.g. Class.method): resolve the binding owner + leaf, but keep the full
    # module path + dotted attr in meta so kernel_selection's f"{module}:{attr}" == target check holds.
    owner = mod
    for part in attr.split(".")[:-1]:
        owner = getattr(owner, part)
    leaf = attr.split(".")[-1]
    orig = getattr(owner, leaf)
    if not _wrappable(orig) and os.environ.get("CAPTURE_WRAP_UNSAFE", "0") != "1":
        raise RuntimeError(
            f"[capture_shapes] refusing to wrap non-Python callable {target} "
            f"({type(orig).__module__}.{type(orig).__name__}): a plain-function stand-in for a native/"
            f"triton-JIT callable SIGSEGVs the server (e.g. mxfp4 matmul_ogs). Hook a Python-level seam "
            f"(its caller) instead, or set CAPTURE_WRAP_UNSAFE=1 to force.")
    out_dir = _process_out_dir(out_dir)
    s.update(target=target, out_dir=out_dir, max_cases=int(max_cases),
             orig=orig, mod=mod, attr=attr, installed=True, meta_flush_count=0)
    if os.environ.get("GEAK_SELECTION_TRACE"):
        # Selection runs are intentionally short and server teardown may use SIGTERM, which skips
        # Python atexit. Persist meta immediately.
        s["flush_every"] = 1
    elif os.environ.get("CAPTURE_FLUSH_EVERY"):
        s["flush_every"] = max(1, int(os.environ["CAPTURE_FLUSH_EVERY"]))
    setattr(owner, leaf, _make_wrapper(orig))
    atexit.register(_flush)
    sys.stderr.write(
        f"[capture_shapes] hooked {target}; cataloging up to {max_cases} shape case(s) -> {out_dir}"
        f" (shapes only, no tensor oracle)\n")


# Allow configuration purely via env (so a generic overlay sitecustomize can call install()):
#   CAPTURE_TARGET=module:attr  CAPTURE_OUT=/path  CAPTURE_MAX=5
def install_from_env():
    t = os.environ.get("CAPTURE_TARGET")
    o = os.environ.get("CAPTURE_OUT")
    if t and o:
        install(t, o, int(os.environ.get("CAPTURE_MAX", "5")))


def _cli(argv=None):
    """CLI for capture reclaim (used by extractor retries / kernel_selection / finalize)."""
    import argparse
    parser = argparse.ArgumentParser(description="capture_shapes utilities")
    parser.add_argument("--cleanup-task-dir", default="",
                        help="reclaim capture.pid-* dirs under this task dir")
    parser.add_argument("--reclaim-workspace", default="",
                        help="reclaim capture.pid-* under eval_dir/kernels/*_task")
    parser.add_argument("--keep-meta", default="",
                        help="optional selected capture meta.json to promote before cleanup")
    parser.add_argument("--promote", action="store_true",
                        help="promote --keep-meta meta.json into the task root before reclaim")
    parser.add_argument("--no-promote", action="store_true",
                        help="reclaim without promoting (retry / failure path)")
    args = parser.parse_args(argv)
    if args.reclaim_workspace:
        telemetry = reclaim_workspace_captures(args.reclaim_workspace)
        print(json.dumps(telemetry, indent=2))
        return 0 if not telemetry.get("errors") else 1
    if not args.cleanup_task_dir:
        parser.error("pass --cleanup-task-dir or --reclaim-workspace")
    promote = False if args.no_promote else (args.promote or bool(args.keep_meta))
    telemetry = cleanup_task_capture_artifacts(
        args.cleanup_task_dir,
        keep_meta_path=args.keep_meta or None,
        promote=promote,
    )
    print(json.dumps(telemetry, indent=2))
    return 0 if not telemetry.get("errors") else 1


if os.environ.get("CAPTURE_TARGET") and os.environ.get("CAPTURE_OUT"):
    try:
        install_from_env()
    except Exception as e:
        sys.stderr.write(f"[capture_shapes] install_from_env failed: {e}\n")


if __name__ == "__main__":
    sys.exit(_cli())
