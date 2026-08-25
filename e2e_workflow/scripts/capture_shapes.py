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
import atexit, functools, importlib, json, os, re, shutil, sys, threading

# Issue #429: process-local MXFP4 MoE oracles can retain multi-GiB tensors per rank/retry until the
# workspace is evicted. Defaults bound heavy oracle persistence; set CAPTURE_BYTE_BUDGET=0 for unlimited.
_DEFAULT_BYTE_BUDGET = os.environ.get("CAPTURE_BYTE_BUDGET", "8GiB")
_DEFAULT_CASE_BYTE_LIMIT = os.environ.get("CAPTURE_CASE_BYTE_LIMIT", "2GiB")
_DEFAULT_PERSIST_POLICY = os.environ.get("CAPTURE_PERSIST_POLICY", "share_large")
_DEFAULT_SHARE_MIN_BYTES = os.environ.get("CAPTURE_SHARE_MIN_BYTES", "16MiB")
_CAPTURE_DIR_RE = re.compile(r"^capture\.pid-")
_TMP_ARTIFACT_RE = re.compile(r"\.(?:pt|json)\.tmp-\d+")
# Heuristic names for expert/static parameter tensors (used by moe_slim / share_large).
_WEIGHT_KEY_RE = re.compile(
    r"(^w[123]$|^weight$|expert_w|gate_up|down_proj|up_proj|_weight$)", re.I)

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
    # storage bound (#429): estimated serialized tensor bytes already accepted into `records` + shared.
    "byte_budget": 0, "case_byte_limit": 0, "persist_policy": "share_large",
    "share_min_bytes": 16 << 20, "shared_tensors": {}, "shared_bytes_est": 0,
    "oracle_bytes_est": 0, "budget_exceeded": False,
    "budget_skip_count": 0, "oracle_save_count": 0, "meta_flush_count": 0,
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


def parse_byte_budget(value):
    """Parse CAPTURE_BYTE_BUDGET: int bytes, or strings like 8GiB / 512M / 0 (unlimited)."""
    if value is None:
        return 0
    text = str(value).strip().lower().replace(" ", "")
    if not text or text in ("0", "none", "unlimited", "inf"):
        return 0
    units = {
        "b": 1, "k": 1024, "kb": 1024, "ki": 1024, "kib": 1024,
        "m": 1024 ** 2, "mb": 1024 ** 2, "mi": 1024 ** 2, "mib": 1024 ** 2,
        "g": 1024 ** 3, "gb": 1024 ** 3, "gi": 1024 ** 3, "gib": 1024 ** 3,
        "t": 1024 ** 4, "tb": 1024 ** 4, "ti": 1024 ** 4, "tib": 1024 ** 4,
    }
    for suffix, mult in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if text.endswith(suffix):
            num = text[:-len(suffix)]
            if num and num.replace(".", "", 1).isdigit():
                return int(float(num) * mult)
    if text.isdigit():
        return int(text)
    raise ValueError(f"invalid CAPTURE_BYTE_BUDGET: {value!r}")


def _dtype_itemsize(dtype):
    name = str(dtype).split(".")[-1].lower()
    table = {
        "float64": 8, "double": 8, "complex128": 16,
        "float32": 4, "float": 4, "complex64": 8,
        "float16": 2, "half": 2, "bfloat16": 2,
        "int64": 8, "long": 8, "uint64": 8,
        "int32": 4, "int": 4, "uint32": 4,
        "int16": 2, "short": 2, "uint16": 2,
        "int8": 1, "uint8": 1, "bool": 1,
        "float8_e4m3fn": 1, "float8_e5m2": 1, "float8_e4m3fnuz": 1,
    }
    return table.get(name, 2)


def _tensor_nbytes(tensor):
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        try:
            n = 1
            for dim in tensor.shape:
                n *= int(dim)
            return n * _dtype_itemsize(getattr(tensor, "dtype", "float16"))
        except Exception:
            return 0


def _estimate_object_bytes(obj):
    """Pre-serialization tensor-byte estimate (no clone). Nested containers walked recursively."""
    torch = _torch()
    if torch.is_tensor(obj):
        return _tensor_nbytes(obj)
    if isinstance(obj, dict) and obj.get("__tensor__"):
        shape = obj.get("shape") or []
        n = 1
        for dim in shape:
            n *= int(dim)
        return n * _dtype_itemsize(obj.get("dtype", "float16"))
    if isinstance(obj, (list, tuple)):
        return sum(_estimate_object_bytes(v) for v in obj)
    if isinstance(obj, dict):
        return sum(_estimate_object_bytes(v) for v in obj.values())
    return 0


def _estimate_call_bytes(args, kwargs, out):
    return (_estimate_object_bytes(args)
            + _estimate_object_bytes(kwargs)
            + _estimate_object_bytes(out))


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


def promote_selected_oracle(task_dir, meta_path):
    """Move/copy the selected process-local oracle into task_dir as the sole authoritative copy."""
    task_dir = os.path.abspath(task_dir)
    meta_path = os.path.abspath(meta_path)
    os.makedirs(task_dir, exist_ok=True)
    cap_dir = os.path.dirname(meta_path)
    src_io = os.path.join(cap_dir, "reference_io.pt")
    dst_io = os.path.join(task_dir, "reference_io.pt")
    dst_meta = os.path.join(task_dir, "meta.json")
    result = {"promoted_meta": False, "promoted_oracle": False, "bytes_promoted": 0,
              "oracle_path": dst_io, "meta_path": dst_meta}
    if os.path.isfile(meta_path):
        if os.path.abspath(meta_path) != os.path.abspath(dst_meta):
            shutil.copy2(meta_path, dst_meta)
        result["promoted_meta"] = True
    if os.path.isfile(src_io):
        if os.path.abspath(src_io) != os.path.abspath(dst_io):
            # Prefer rename to avoid retaining two full copies when on the same filesystem.
            if not os.path.exists(dst_io):
                try:
                    os.replace(src_io, dst_io)
                except OSError:
                    shutil.copy2(src_io, dst_io)
            else:
                shutil.copy2(src_io, dst_io)
        result["promoted_oracle"] = True
        result["bytes_promoted"] = os.path.getsize(dst_io)
    return result


def cleanup_task_capture_artifacts(task_dir, keep_meta_path=None, promote=None,
                                   remove_tmp=True):
    """Reclaim process-local capture dirs under task_dir (issue #429).

    When ``promote`` is true (default if ``keep_meta_path`` is set), copy/move the selected
    ``meta.json`` + ``reference_io.pt`` into ``task_dir`` first, then delete every
    ``capture.pid-*`` directory (including nested ``_selcap*`` attempts) and leftover ``*.tmp-*``
    atomic-write files. Writes ``capture_telemetry.json`` into ``task_dir``.

    Returns a telemetry dict with ``bytes_reclaimed``, ``bytes_promoted``, ``removed_paths``, etc.
    """
    task_dir = os.path.abspath(task_dir)
    telemetry = {
        "task_dir": task_dir,
        "bytes_reclaimed": 0,
        "bytes_promoted": 0,
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
        promoted = promote_selected_oracle(task_dir, keep_meta_path)
        telemetry["promoted"] = True
        telemetry["bytes_promoted"] = promoted["bytes_promoted"]
        telemetry["promote"] = promoted

    for cap_dir in capture_dirs:
        _remove_path(cap_dir, telemetry)

    if remove_tmp:
        for root, _dirs, files in os.walk(task_dir):
            for name in files:
                if ".tmp-" in name:
                    _remove_path(os.path.join(root, name), telemetry)

    # After reclaim, only the authoritative root oracle (if promoted) should remain.
    remaining = sorted(iter_capture_dirs(task_dir))
    telemetry["capture_dirs_remaining"] = remaining
    telemetry["bytes_after"] = sum(_dir_bytes(d) for d in remaining)
    root_io = os.path.join(task_dir, "reference_io.pt")
    telemetry["authoritative_oracle_bytes"] = (
        os.path.getsize(root_io) if os.path.isfile(root_io) else 0)
    telemetry["authoritative_oracle_present"] = os.path.isfile(root_io)

    try:
        _write_json(os.path.join(task_dir, "capture_telemetry.json"), telemetry)
    except OSError as exc:
        telemetry["errors"].append(f"telemetry write failed: {exc}")
    return telemetry


def promote_and_reclaim(task_dir, keep_meta_path=None, promote=True):
    """Convenience wrapper used by kernel_selection after a verdict."""
    return cleanup_task_capture_artifacts(
        task_dir, keep_meta_path=keep_meta_path, promote=promote and bool(keep_meta_path))


def reclaim_workspace_captures(eval_dir, workspace_budget=0):
    """Finalize-time sweep: reclaim capture.pid-* under every kernels/*_task.

    If ``workspace_budget`` > 0 and the sum of authoritative ``reference_io.pt`` sizes exceeds it,
    the largest oracles are reported in telemetry (caller may drop editable heads); this function
    does not delete authoritative oracles.
    """
    eval_dir = os.path.abspath(eval_dir)
    kernels = os.path.join(eval_dir, "kernels")
    telemetry = {
        "eval_dir": eval_dir,
        "tasks": [],
        "bytes_reclaimed": 0,
        "authoritative_oracle_bytes": 0,
        "workspace_budget": int(workspace_budget or 0),
        "over_budget": False,
        "errors": [],
    }
    if not os.path.isdir(kernels):
        telemetry["errors"].append(f"missing kernels dir: {kernels}")
        return telemetry
    oracle_sizes = []
    for name in sorted(os.listdir(kernels)):
        task = os.path.join(kernels, name)
        if not os.path.isdir(task) or not name.endswith("_task"):
            continue
        # Drop nested attempt dirs' heavy captures but keep task root oracle.
        task_tel = cleanup_task_capture_artifacts(task, promote=False)
        telemetry["bytes_reclaimed"] += int(task_tel.get("bytes_reclaimed") or 0)
        root_io = os.path.join(task, "reference_io.pt")
        nbytes = os.path.getsize(root_io) if os.path.isfile(root_io) else 0
        telemetry["authoritative_oracle_bytes"] += nbytes
        oracle_sizes.append((nbytes, task))
        telemetry["tasks"].append({
            "task": task,
            "bytes_reclaimed": task_tel.get("bytes_reclaimed"),
            "oracle_bytes": nbytes,
            "capture_dirs_removed": len(task_tel.get("removed_paths") or []),
        })
    budget = int(workspace_budget or 0)
    if budget > 0 and telemetry["authoritative_oracle_bytes"] > budget:
        telemetry["over_budget"] = True
        telemetry["largest_oracles"] = [
            {"task": t, "bytes": b} for b, t in sorted(oracle_sizes, reverse=True)[:8]
        ]
    try:
        _write_json(os.path.join(eval_dir, "capture_workspace_telemetry.json"), telemetry)
    except OSError as exc:
        telemetry["errors"].append(str(exc))
    return telemetry


def _write_capture_manifest(out_dir, extra=None):
    """Lightweight size/shape manifest written when the heavy oracle is skipped or reclaiming."""
    s = _STATE
    payload = {
        "target": s.get("target"),
        "process_id": os.getpid(),
        "rank": _rank(),
        "byte_budget": s.get("byte_budget", 0),
        "oracle_bytes_est": s.get("oracle_bytes_est", 0),
        "budget_exceeded": bool(s.get("budget_exceeded")),
        "budget_skip_count": int(s.get("budget_skip_count") or 0),
        "num_records": len(s.get("records") or []),
        "oracle_complete": bool(s.get("oracle_written")),
        "note": "Heavy reference_io.pt omitted or bounded; see shapes in meta.json.",
    }
    if extra:
        payload.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    _write_json(os.path.join(out_dir, "capture_manifest.json"), payload)


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


def _is_weight_key(key):
    return bool(_WEIGHT_KEY_RE.search(str(key)))


def _should_share_kwarg(key, value, policy, share_min_bytes):
    """Whether this kwarg should be stored once in the oracle's shared pool (not per-case)."""
    nbytes = _estimate_object_bytes(value)
    if policy in ("moe_slim", "share_large") and _is_weight_key(key) and nbytes >= (1 << 20):
        return True
    if policy == "share_large" and nbytes >= share_min_bytes:
        return True
    return False


def _snapshot_kwargs(kwargs, shared_store, policy, share_min_bytes):
    """Snapshot kwargs; large/static weight tensors are stored once in ``shared_store``."""
    if policy == "full" or not isinstance(kwargs, dict):
        return _snapshot(kwargs), 0, []
    snap = {}
    shared_added = 0
    shared_keys = []
    for key, value in kwargs.items():
        if _should_share_kwarg(key, value, policy, share_min_bytes):
            if key not in shared_store:
                shared_store[key] = _snapshot(value)
                shared_added += _estimate_object_bytes(value)
            snap[key] = {"__shared__": key}
            shared_keys.append(key)
        else:
            snap[key] = _snapshot(value)
    return snap, shared_added, shared_keys


def resolve_shared_refs(obj, shared):
    """Replace ``{"__shared__": key}`` leaves with entries from the oracle shared pool."""
    if isinstance(obj, dict) and set(obj.keys()) == {"__shared__"}:
        key = obj["__shared__"]
        if key not in shared:
            raise KeyError(f"oracle shared ref missing: {key!r}")
        return shared[key]
    if isinstance(obj, dict):
        return {k: resolve_shared_refs(v, shared) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(resolve_shared_refs(v, shared) for v in obj)
    return obj


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
                call_bytes = _estimate_call_bytes(args, kwargs, out)
                budget = int(s.get("byte_budget") or 0)
                case_limit = int(s.get("case_byte_limit") or 0)
                used = int(s.get("oracle_bytes_est") or 0) + int(s.get("shared_bytes_est") or 0)
                policy = s.get("persist_policy") or "share_large"
                share_min = int(s.get("share_min_bytes") or (16 << 20))
                # Estimate after sharing: weight kwargs counted once into shared pool.
                shared_store = s.setdefault("shared_tensors", {})
                est_shared_add = 0
                est_case = call_bytes
                if isinstance(kwargs, dict) and policy != "full":
                    for key, value in kwargs.items():
                        if _should_share_kwarg(key, value, policy, share_min):
                            nbytes = _estimate_object_bytes(value)
                            est_case -= nbytes
                            if key not in shared_store:
                                est_shared_add += nbytes
                effective_need = max(0, est_case) + est_shared_add
                if case_limit > 0 and effective_need > case_limit:
                    s["seen"].add(sig)
                    s["budget_exceeded"] = True
                    s["budget_skip_count"] = int(s.get("budget_skip_count") or 0) + 1
                    sys.stderr.write(
                        f"[capture_shapes] case byte limit exceeded "
                        f"(need={effective_need} limit={case_limit}); "
                        f"skipping heavy oracle for {sig}\n")
                    try:
                        _write_capture_manifest(s["out_dir"], {
                            "skipped_sig": sig,
                            "skipped_bytes_est": effective_need,
                            "reason": "case_byte_limit",
                        })
                    except Exception as manifest_exc:
                        sys.stderr.write(
                            f"[capture_shapes] manifest write failed: {manifest_exc}\n")
                elif budget > 0 and used + effective_need > budget:
                    s["seen"].add(sig)
                    s["budget_exceeded"] = True
                    s["budget_skip_count"] = int(s.get("budget_skip_count") or 0) + 1
                    sys.stderr.write(
                        f"[capture_shapes] byte budget exceeded "
                        f"(used={used} need={effective_need} budget={budget}); "
                        f"skipping heavy oracle for {sig}\n")
                    try:
                        _write_capture_manifest(s["out_dir"], {
                            "skipped_sig": sig,
                            "skipped_bytes_est": effective_need,
                            "reason": "byte_budget",
                        })
                    except Exception as manifest_exc:
                        sys.stderr.write(
                            f"[capture_shapes] manifest write failed: {manifest_exc}\n")
                else:
                    snap_kwargs, shared_added, shared_keys = _snapshot_kwargs(
                        kwargs, shared_store, policy, share_min)
                    s["seen"].add(sig)
                    s["regime_seen"].add(regime)
                    s["records"].append({
                        "sig": sig,
                        "regime": regime,
                        "args": _snapshot(args),
                        "kwargs": snap_kwargs,
                        "output": _snapshot(out),
                        "shared_keys": shared_keys,
                    })
                    s["shared_bytes_est"] = int(s.get("shared_bytes_est") or 0) + shared_added
                    s["oracle_bytes_est"] = int(s.get("oracle_bytes_est") or 0) + max(0, est_case)
                    sys.stderr.write(
                        f"[capture_shapes] recorded case {len(s['records'])} ({regime}): {sig}"
                        f" shared={shared_keys or []}\n")
    except Exception as e:  # never break the server because capture failed
        sys.stderr.write(f"[capture_shapes] capture error (ignored): {e}\n")
    # crash-resilient incremental flush (OUTSIDE the lock; best-effort, never breaks the server).
    try:
        _maybe_flush(in_graph)
    except Exception as e:
        sys.stderr.write(f"[capture_shapes] periodic flush error (ignored): {e}\n")
    return out


def _maybe_flush(in_graph=False):
    """Called after every wrapped call.

    Rewrites light ``meta.json`` every ``flush_every`` calls. Heavy ``reference_io.pt`` is written only
    when in-memory records outpace the on-disk oracle — so selection mode (``flush_every=1``) does not
    rewrite multi-GiB tensors on every call, only when a new case is recorded (#429).

    NEVER flush while the server is capturing a CUDA graph: the oracle write does a device sync / host
    copy, which is ILLEGAL inside graph capture.
    """
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
        tmp_io = f"{io_path}.tmp-{os.getpid()}-{threading.get_ident()}"
        try:
            with s["lock"]:
                shared = dict(s.get("shared_tensors") or {})
            torch.save({
                "target": s["target"],
                "records": records,
                "shared": shared,
                "persist_policy": s.get("persist_policy") or "share_large",
            }, tmp_io)
            os.replace(tmp_io, io_path)
        except Exception:
            if os.path.exists(tmp_io):
                try:
                    os.unlink(tmp_io)
                except OSError:
                    pass
            raise
        import hashlib
        h = hashlib.sha256()
        with open(io_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        s["oracle_sha"] = h.hexdigest()
        s["oracle_written"] = True
        s["oracle_records"] = len(records)
        s["oracle_save_count"] = int(s.get("oracle_save_count") or 0) + 1
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
        "reference_io": "reference_io.pt",
        "reference_io_sha256": s["oracle_sha"],   # None until the oracle file is written (partial flush)
        "oracle_complete": bool(s["oracle_written"]),
        "byte_budget": int(s.get("byte_budget") or 0),
        "case_byte_limit": int(s.get("case_byte_limit") or 0),
        "persist_policy": s.get("persist_policy") or "share_large",
        "shared_tensor_keys": sorted((s.get("shared_tensors") or {}).keys()),
        "oracle_bytes_est": int(s.get("oracle_bytes_est") or 0) + int(s.get("shared_bytes_est") or 0),
        "budget_exceeded": bool(s.get("budget_exceeded")),
        "budget_skip_count": int(s.get("budget_skip_count") or 0),
        "oracle_save_count": int(s.get("oracle_save_count") or 0),
        "build": False,  # default: pure-python/triton; Extractor flips to True for HIP/CK/asm tasks
        "note": "Oracle captured from baseline. Do NOT edit unittest.py or reference_io.pt during opt.",
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
    out_dir = _process_out_dir(out_dir)
    try:
        byte_budget = parse_byte_budget(os.environ.get("CAPTURE_BYTE_BUDGET", _DEFAULT_BYTE_BUDGET))
        case_byte_limit = parse_byte_budget(
            os.environ.get("CAPTURE_CASE_BYTE_LIMIT", _DEFAULT_CASE_BYTE_LIMIT))
        share_min_bytes = parse_byte_budget(
            os.environ.get("CAPTURE_SHARE_MIN_BYTES", _DEFAULT_SHARE_MIN_BYTES))
    except ValueError as exc:
        raise RuntimeError(f"[capture_shapes] {exc}") from exc
    persist_policy = (os.environ.get("CAPTURE_PERSIST_POLICY", _DEFAULT_PERSIST_POLICY)
                      or "share_large").strip().lower()
    if persist_policy not in ("full", "share_large", "moe_slim"):
        raise RuntimeError(
            f"[capture_shapes] invalid CAPTURE_PERSIST_POLICY={persist_policy!r} "
            f"(expected full|share_large|moe_slim)")
    s.update(target=target, out_dir=out_dir, max_cases=int(max_cases),
             orig=orig, mod=mod, attr=attr, installed=True,
             byte_budget=byte_budget, case_byte_limit=case_byte_limit,
             persist_policy=persist_policy, share_min_bytes=share_min_bytes or (16 << 20),
             shared_tensors={}, shared_bytes_est=0,
             oracle_bytes_est=0, budget_exceeded=False,
             budget_skip_count=0, oracle_save_count=0, meta_flush_count=0)
    if os.environ.get("GEAK_SELECTION_TRACE"):
        # Selection runs are intentionally short and server teardown may use SIGTERM, which skips
        # Python atexit. Persist meta immediately; heavy oracle still only rewrites when records grow.
        s["flush_every"] = 1
    elif os.environ.get("CAPTURE_FLUSH_EVERY"):
        s["flush_every"] = max(1, int(os.environ["CAPTURE_FLUSH_EVERY"]))
    setattr(mod, attr, _make_wrapper(orig))
    atexit.register(_flush)
    sys.stderr.write(
        f"[capture_shapes] hooked {target}; recording up to {max_cases} cases -> {out_dir}"
        f" (byte_budget={byte_budget or 'unlimited'}"
        f" case_limit={case_byte_limit or 'unlimited'}"
        f" policy={persist_policy})\n")


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
    parser.add_argument("--workspace-budget", default="0",
                        help="optional total authoritative-oracle byte budget for telemetry")
    parser.add_argument("--keep-meta", default="",
                        help="optional selected capture meta.json to promote before cleanup")
    parser.add_argument("--promote", action="store_true",
                        help="promote --keep-meta oracle into the task root before reclaim")
    parser.add_argument("--no-promote", action="store_true",
                        help="reclaim without promoting (retry / failure path)")
    args = parser.parse_args(argv)
    if args.reclaim_workspace:
        budget = parse_byte_budget(args.workspace_budget)
        telemetry = reclaim_workspace_captures(args.reclaim_workspace, workspace_budget=budget)
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
