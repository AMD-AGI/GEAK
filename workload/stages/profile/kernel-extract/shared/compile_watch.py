#!/usr/bin/env python3
"""Record every Triton compile variant a process builds, and the call that built it.

Coverage measured in M (tokens, batch) says nothing about which kernel a Triton entry
compiles: the variant also depends on argument dtypes, which optional arguments are None,
integer specialization (an int argument equal to 1 becomes a constexpr; others get a
divisibility-by-16 attribute), constexpr values and the autotune config. A UT can cover every
served M and still never compile the variant the server compiles. Muse Spark's
attention_2d_prefill candidate passed its UT on 3419 cases and crashed the server on its
first request: a warmup call with max_seqlen_q=1 (the decode branch of the prefill kernel,
a variant no UT case built), which the profile never saw because its windows skip warmup.

This module hooks Triton's JIT cache (`knobs.runtime.jit_cache_hook`, or
`JITFunction.cache_hook` before Triton 3.3). The hook fires once per in-process cache miss,
before compilation, so recording costs nothing on the launch path and sees startup, warmup
and CUDA-graph capture as well as served traffic. Each event holds the variant's identity
(kernel, argument types, constexprs, specialization attributes, warps/stages), the Python
stack that launched it and, optionally, the arguments of the first non-Triton frames: shapes,
dtypes, strides, scalars, None, and the contents of integer and small tensors (sequence
bookkeeping, page tables, scales). Bulk float payloads are kept as descriptors and redrawn
from a seed when the call is replayed.

  server   probe_server.py puts an overlay holding this file and a sitecustomize on
           PYTHONPATH (COMPILE_WATCH_DIR, COMPILE_WATCH_FRAMES); every process appends
           events to variants.pid<pid>.jsonl and, for the first process to compile a
           variant, saves the frame tensors under calls/
  collect  python3 compile_watch.py collect --dir D --out compile_variants.json
           [--ready-epoch T] merges processes, one record per variant
  UT       leg_runner.py installs it in the baseline leg (COMPILE_WATCH_OUT) and the
           ut_driver compares the baseline's variants with the server's

Installation is deferred: nothing is imported until the application imports triton itself,
so the import order of the observed process is unchanged (an early `import aiter` from a
sitecustomize once made FlyDSL's JIT abort). Stdlib at import time; torch lazily.
"""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
import re
import sys
import threading
import time

_STATE = {"installed": False, "bound": False, "events": [], "seen": set(), "lock": threading.Lock(),
          "dir": None, "frames": 0, "fh": None, "t0": time.time(), "n": 0, "prev": None}
# Frames that are plumbing, not a caller: Triton itself, torch's op/module/decorator wrappers.
_SKIP_FILES = re.compile(r"[/\\](triton|_dynamo|_inductor|_library)[/\\]|compile_watch\.py$|"
                         r"[/\\]torch[/\\](_ops\.py|utils[/\\]_contextlib\.py|"
                         r"nn[/\\]modules[/\\]module\.py)")
INT_DATA_CAP = int(os.environ.get("COMPILE_WATCH_INT_CAP", str(512 << 20)))
FLOAT_DATA_CAP = int(os.environ.get("COMPILE_WATCH_FLOAT_CAP", "4096"))
EVENT_DATA_CAP = int(os.environ.get("COMPILE_WATCH_EVENT_CAP", str(1 << 30)))
IDENTITY_FIELDS = ("fn", "signature", "constexprs", "attrs", "num_warps", "num_stages")


def _rank():
    return next((str(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "TP_RANK")
                 if k in os.environ), "")


# ----------------------------------------------------------------------------- identity
def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    val = getattr(v, "value", v)          # tl.constexpr wraps its value
    if val is not v:
        return _jsonable(val)
    return repr(v)


def _by_name(paths, names):
    """{(i,): x} -> {param_name: x}; deeper paths keep their tuple text."""
    out = {}
    for path, val in (paths or {}).items():
        if isinstance(path, tuple) and len(path) == 1 and isinstance(path[0], int) \
                and path[0] < len(names):
            key = names[path[0]]
        elif isinstance(path, str):
            key = path
        else:
            key = str(path)
        out[key] = _jsonable(val)
    return out


def variant_identity(fn, compile_info):
    """The fields that select a compiled kernel, as plain JSON."""
    sig = compile_info.get("signature") or {}
    names = [str(k) for k in sig]
    attrs = (compile_info.get("configs") or [{}])[0]
    if not isinstance(attrs, dict):        # pre-3.3 Triton: an AttrsDescriptor
        attrs = {"descriptor": repr(attrs)}
    return {"fn": fn,
            "signature": {str(k): str(v) for k, v in sig.items()},
            "constexprs": _by_name(compile_info.get("constants") or {}, names),
            "attrs": {k: v for k, v in _by_name(attrs, names).items() if v not in ([], None)},
            "num_warps": compile_info.get("num_warps"),
            "num_stages": compile_info.get("num_stages")}


def variant_id(ident):
    blob = json.dumps({k: ident.get(k) for k in IDENTITY_FIELDS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def short_name(fn):
    return str(fn).rsplit(".", 1)[-1]


# ----------------------------------------------------------------------------- frames
def _torch():
    return sys.modules.get("torch")


def _capturing(torch):
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _describe(x, tensors, budget, in_capture, depth=0):
    torch = _torch()
    if torch is not None and isinstance(x, torch.Tensor):
        d = {"__tensor__": 1, "shape": list(x.shape), "stride": list(x.stride()),
             "dtype": str(x.dtype), "device": str(x.device), "offset": int(x.storage_offset()),
             "nbytes": int(x.numel() * x.element_size()),
             # the storage, not the view, decides AMD's buffer-op specialization (<= 2 GiB)
             "storage_nbytes": int(x.untyped_storage().nbytes())}
        floating = x.is_floating_point() or x.is_complex()
        cap = FLOAT_DATA_CAP if floating else INT_DATA_CAP
        if in_capture:
            d["data"] = None
            d["why_no_data"] = "launched inside CUDA graph capture: contents are placeholders"
        elif d["nbytes"] <= cap and d["nbytes"] <= budget[0]:
            t = x.detach().to("cpu").clone()
            # Request tables are mostly unused rows (req_to_token is [max_reqs, max_ctx]):
            # keep a large, mostly-zero integer tensor sparse. Rebuilt dense on replay.
            if not floating and d["nbytes"] > (1 << 20) and int((t != 0).sum()) * 4 < t.numel():
                t = t.to_sparse()
                d["sparse"] = True
            tensors.append(t)
            budget[0] -= d["nbytes"]
            d["data"] = len(tensors) - 1
        else:
            d["data"] = None
            d["why_no_data"] = ("bulk float payload: redrawn from a seed on replay" if floating
                                else "integer tensor over the capture cap (%d bytes)" % cap)
        return d
    if torch is not None and isinstance(x, torch.dtype):
        return {"__dtype__": str(x)}
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if x == x and abs(x) != float("inf") else {"__float__": repr(x)}
    if isinstance(x, (list, tuple)) and depth < 2 and len(x) <= 64:
        return {"__seq__": type(x).__name__,
                "items": [_describe(v, tensors, budget, in_capture, depth + 1) for v in x]}
    return {"__object__": "%s.%s" % (type(x).__module__, type(x).__qualname__),
            "repr": repr(x)[:120]}


def _frame_name(frame):
    code = frame.f_code
    qual = getattr(code, "co_qualname", None)
    if not qual:
        owner = frame.f_locals.get("self") if code.co_argcount else None
        qual = ("%s.%s" % (type(owner).__qualname__, code.co_name) if owner is not None
                and code.co_varnames[:1] == ("self",) else code.co_name)
    return "%s:%s" % (frame.f_globals.get("__name__", "?"), qual)


def _frame_args(frame):
    code = frame.f_code
    n = code.co_argcount + code.co_kwonlyargcount
    names = list(code.co_varnames[:n])
    flags = code.co_flags
    extra = {}
    if flags & 0x04:        # *args
        extra["*"] = code.co_varnames[n]
        n += 1
    if flags & 0x08:        # **kwargs
        extra["**"] = code.co_varnames[n]
    return names, code.co_argcount, extra


def _stack(limit=48):
    out, f = [], sys._getframe(2)
    while f is not None and len(out) < limit:
        out.append(f)
        f = f.f_back
    return out


def _record_frames(frames, k, tensors, in_capture):
    torch = _torch()
    budget = [EVENT_DATA_CAP]
    rows = []
    for f in frames:
        if len(rows) >= k:
            break
        if _SKIP_FILES.search(f.f_code.co_filename) or f.f_code.co_name == "<lambda>":
            continue
        names, npos, extra = _frame_args(f)
        loc = f.f_locals
        args = {}
        for name in names:
            if name == "self" and name in loc and not (torch and isinstance(loc[name], torch.Tensor)):
                args[name] = {"__object__": type(loc[name]).__qualname__}
                continue
            args[name] = _describe(loc.get(name), tensors, budget, in_capture) \
                if name in loc else {"__missing__": 1}
        if "*" in extra:
            args["*" + extra["*"]] = _describe(tuple(loc.get(extra["*"]) or ()), tensors, budget,
                                               in_capture)
        if "**" in extra:
            kw = loc.get(extra["**"]) or {}
            args["**" + extra["**"]] = {k2: _describe(v, tensors, budget, in_capture)
                                        for k2, v in kw.items()}
        rows.append({"callable": _frame_name(f), "file": f.f_code.co_filename,
                     "line": f.f_lineno, "positional": names[:npos], "args": args})
    return rows


# ----------------------------------------------------------------------------- import environment
ENV_MODULES = ("triton", "triton.language", "torch", "torch.nn.functional")


def env_fingerprint():
    """What the process's Triton/torch namespaces resolve to, by attribute.

    A serving stack can rebind library functions at import time (sglang replaces
    `triton.next_power_of_2`, a ConstexprFunction, with a plain function, so a kernel that
    calls it in a jit body compiles in a bare UT and fails in the server). Comparing this
    fingerprint between the server and the UT names every such difference."""
    out = {}
    for name in ENV_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        for attr, v in list(vars(mod).items()):
            if attr.startswith("_") or not callable(v) or isinstance(v, type(sys)):
                continue
            out["%s.%s" % (name, attr)] = "%s:%s" % (type(v).__name__,
                                                     getattr(v, "__module__", None))
    return out


def env_diff(server_env, ut_env):
    """Attributes bound differently in the server than in the UT (missing ones are not)."""
    return [{"attr": k, "server": v, "ut": ut_env[k]} for k, v in sorted((server_env or {}).items())
            if k in (ut_env or {}) and ut_env[k] != v]


# ----------------------------------------------------------------------------- hook
def _emit(ev):
    s = _STATE
    s["events"].append(ev)
    if s["dir"]:
        if s["fh"] is None:
            os.makedirs(s["dir"], exist_ok=True)
            s["fh"] = open(os.path.join(s["dir"], "variants.pid%d.jsonl" % os.getpid()), "a",
                           buffering=1)
        s["fh"].write(json.dumps(ev, default=repr) + "\n")


def _claim(vid):
    """True for the first process on this node to record a variant's call data."""
    path = os.path.join(_STATE["dir"], "claims", vid)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False
    except OSError:
        return True


def _hook(*_a, **kw):
    s = _STATE
    try:
        fninfo, comp = kw.get("fn"), kw.get("compile") or {}
        fn = "%s.%s" % (getattr(fninfo, "module", "?"), getattr(fninfo, "name", "?"))
        ident = variant_identity(fn, comp)
        vid = variant_id(ident)
        torch = _torch()
        in_capture = bool(torch is not None and _capturing(torch))
        frames = _stack()
        ev = dict(ident, id=vid, name=short_name(fn), pid=os.getpid(), rank=_rank(),
                  t=round(time.time(), 3), t_rel=round(time.time() - s["t0"], 3),
                  is_warmup=bool(kw.get("is_manual_warmup")), in_graph_capture=in_capture,
                  stack=[_frame_name(f) for f in frames
                         if not _SKIP_FILES.search(f.f_code.co_filename)][:32])
        with s["lock"]:
            if not s.get("env_done"):
                ev["env"] = env_fingerprint()
                s["env_done"] = True
            s["n"] += 1
            first = vid not in s["seen"]
            s["seen"].add(vid)
            if first and s["frames"] > 0 and (not s["dir"] or _claim(vid)):
                tensors = []
                ev["frames"] = _record_frames(frames, s["frames"], tensors, in_capture)
                if tensors and s["dir"]:
                    os.makedirs(os.path.join(s["dir"], "calls"), exist_ok=True)
                    rel = os.path.join("calls", "%s.pid%d.pt" % (vid, os.getpid()))
                    torch.save(tensors, os.path.join(s["dir"], rel))
                    ev["tensors"] = rel
                elif tensors:
                    ev["tensor_objects"] = tensors
            _emit(ev)
    except Exception as e:             # never break the observed process
        sys.stderr.write("[compile_watch] hook error (ignored): %r\n" % (e,))
    prev = s["prev"]
    return prev(*_a, **kw) if prev else None


def _bind(triton_mod=None):
    s = _STATE
    if s["bound"]:
        return
    knobs = getattr(triton_mod or sys.modules.get("triton"), "knobs", None)
    rt = getattr(knobs, "runtime", None) if knobs is not None else None
    if rt is not None and hasattr(rt, "jit_cache_hook"):
        s["prev"] = rt.jit_cache_hook
        rt.jit_cache_hook = _hook
    else:
        from triton.runtime.jit import JITFunction
        s["prev"] = getattr(JITFunction, "cache_hook", None)
        JITFunction.cache_hook = _hook
    s["bound"] = True


class _BindOnImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Claims `triton` once, lets the real loader execute it, then binds the hook."""

    def __init__(self):
        self._busy = False

    def find_spec(self, name, path=None, target=None):
        if name != "triton" or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        self._real = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        module.__spec__.loader = self._real
        self._real.exec_module(module)
        try:
            _bind(module)
        except Exception as e:
            sys.stderr.write("[compile_watch] bind failed: %r\n" % (e,))
        sys.meta_path[:] = [f for f in sys.meta_path if f is not self]


def install(out_dir=None, frames=None):
    """Start recording (idempotent). Defaults from COMPILE_WATCH_DIR / COMPILE_WATCH_FRAMES."""
    s = _STATE
    if s["installed"]:
        return
    s["dir"] = out_dir if out_dir is not None else (os.environ.get("COMPILE_WATCH_DIR") or None)
    s["frames"] = int(frames if frames is not None else os.environ.get("COMPILE_WATCH_FRAMES", "0"))
    s["installed"] = True
    if "triton" in sys.modules:
        _bind(sys.modules["triton"])
    else:
        sys.meta_path.insert(0, _BindOnImport())


def events():
    return list(_STATE["events"])


def dump(path):
    """Write this process's events (identity + stack, no tensors) to `path`."""
    rows = [{k: v for k, v in e.items() if k != "tensor_objects"} for e in _STATE["events"]]
    with open(path, "w") as f:
        json.dump({"events": rows, "pid": os.getpid()}, f, default=repr)


# ----------------------------------------------------------------------------- collect / compare
def collect(watch_dir, ready_epoch=None, server_mode=None):
    """Per-process JSONL -> one record per variant (the event with call frames wins)."""
    by_id, procs, env = {}, set(), None
    for name in sorted(os.listdir(watch_dir)) if os.path.isdir(watch_dir) else []:
        if not (name.startswith("variants.pid") and name.endswith(".jsonl")):
            continue
        with open(os.path.join(watch_dir, name)) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                procs.add(ev.get("pid"))
                if ev.get("env") and env is None:
                    env = ev["env"]
                ev.pop("env", None)
                rec = by_id.get(ev["id"])
                if rec is None:
                    rec = by_id[ev["id"]] = dict(ev, processes=0, first_t=ev["t"],
                                                 eager=False, in_graph_capture=False)
                    rec.pop("pid", None)
                    rec.pop("t", None)
                rec["processes"] += 1
                rec["first_t"] = min(rec["first_t"], ev["t"])
                rec["eager"] = rec["eager"] or not ev.get("in_graph_capture")
                rec["in_graph_capture"] = rec["in_graph_capture"] or bool(ev.get("in_graph_capture"))
                if ev.get("frames") and not rec.get("frames"):
                    rec["frames"] = ev["frames"]
                    if ev.get("tensors"):
                        rec["tensors"] = ev["tensors"]
    for rec in by_id.values():
        if ready_epoch:
            rec["before_ready"] = rec["first_t"] < float(ready_epoch)
    return {"contract": "kernel_extract/compile_variants/v1", "watch_dir": os.path.abspath(watch_dir),
            "server_mode": server_mode, "ready_epoch": ready_epoch, "processes": len(procs),
            "env": env or {},
            "variants": sorted(by_id.values(), key=lambda r: (r["name"], r["first_t"]))}


def frame_for(rec, target_callable):
    """The recorded frame of `module:qualname` (or of its bare function name) in a variant."""
    mod, _, qual = (target_callable or "").partition(":")
    leaf = qual.rsplit(".", 1)[-1]
    for fr in rec.get("frames") or []:
        fmod, _, fqual = fr["callable"].partition(":")
        if fmod == mod and (fqual == qual or fqual.rsplit(".", 1)[-1] == leaf):
            return fr
    return None


def belongs(rec, kernel_names, target_callable=None):
    """A server variant is the task's when its kernel is one of the task's device kernels or
    its launch stack passes through the task's entry."""
    if rec.get("name") in kernel_names:
        return True
    if target_callable:
        mod, _, qual = target_callable.partition(":")
        leaf = qual.rsplit(".", 1)[-1]
        for fr in rec.get("stack") or []:
            fmod, _, fqual = fr.partition(":")
            if fmod == mod and (fqual == qual or fqual.rsplit(".", 1)[-1] == leaf):
                return True
    return False


def _fields(rec):
    out = {}
    for k in ("signature", "constexprs", "attrs"):
        for name, v in (rec.get(k) or {}).items():
            out["%s[%s]" % (k, name)] = v
    out["num_warps"], out["num_stages"] = rec.get("num_warps"), rec.get("num_stages")
    return out


def diff_fields(server_rec, ut_rec):
    a, b = _fields(server_rec), _fields(ut_rec)
    return [{"field": k, "server": a.get(k), "ut": b.get(k)}
            for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def coverage(server_recs, ut_events, waived=None):
    """Which of the server's variants the UT baseline compiled; for each one it did not, the
    nearest variant of the same kernel the UT did compile and the fields that differ."""
    waived = {w["id"]: w for w in (waived or []) if w.get("id") and (w.get("why") or "").strip()}
    have = {}
    for e in ut_events:
        have.setdefault(e["id"], e)
    by_name = {}
    for e in have.values():
        by_name.setdefault(e["name"], []).append(e)
    covered, missing, waived_rows = [], [], []
    for r in server_recs:
        row = {"id": r["id"], "kernel": r["name"], "eager": r.get("eager"),
               "in_graph_capture": r.get("in_graph_capture"),
               "before_ready": r.get("before_ready")}
        if r["id"] in have:
            covered.append(row)
            continue
        cands = by_name.get(r["name"]) or []
        if cands:
            near = min(cands, key=lambda e: len(diff_fields(r, e)))
            row["nearest_ut_variant"] = near["id"]
            row["differs"] = diff_fields(r, near)
        else:
            row["differs"] = "the UT never compiled %s at all" % r["name"]
        if r["id"] in waived:
            row["waived"] = waived[r["id"]]["why"]
            waived_rows.append(row)
        else:
            missing.append(row)
    return {"server_variants": len(server_recs), "covered": len(covered), "missing": missing,
            "waived": waived_rows}


# ----------------------------------------------------------------------------- replay
class Unreplayable(Exception):
    """A recorded call that cannot be rebuilt outside the server (an object argument, an
    integer tensor whose contents were not kept, a pool over the memory budget)."""


# Integer operands that address rows of a cache pool. Used only to compact a pool that does
# not fit: the ids they hold are renumbered densely and the pool shrinks to the rows touched.
INDEX_ARG_RE = re.compile(r"(block_table|page_table|req_to_token|kv_indices|token_indices|"
                          r"page_indices|topk_indices|^indices$)", re.I)
SNAPSHOT_CAP = 256 << 20


def _tensor_leaves(args):
    """[(name, desc)] for every tensor descriptor in a recorded frame's arguments."""
    out = []

    def walk(name, d):
        if isinstance(d, dict) and d.get("__tensor__"):
            out.append((name, d))
        elif isinstance(d, dict) and "__seq__" in d:
            for i, v in enumerate(d["items"]):
                walk("%s[%d]" % (name, i), v)
        elif isinstance(d, dict) and name.startswith("**"):
            for k, v in d.items():
                walk(k, v)
    for name, d in args.items():
        walk(name, d)
    return out


def _span(d):
    if any(int(n) == 0 for n in d["shape"]):
        return int(d["offset"])
    return int(d["offset"]) + sum((int(n) - 1) * int(st)
                                  for n, st in zip(d["shape"], d["stride"])) + 1


def _dtype(torch, name):
    return getattr(torch, str(name).replace("torch.", ""))


def _itemsize(torch, name):
    return torch.empty((), dtype=_dtype(torch, name)).element_size()


def plan_call(call, data):
    """Check a recorded call can be rebuilt; returns the per-tensor plan and its notes."""
    torch = __import__("torch")
    notes = []
    for name, d in call["args"].items():
        if isinstance(d, dict) and "__object__" in d and name != "self":
            raise Unreplayable("argument %s is a %s" % (name, d["__object__"]))
        if isinstance(d, dict) and "__missing__" in d:
            raise Unreplayable("argument %s was not bound when the kernel compiled" % name)
    if call["args"].get("self") is not None and "self" in call.get("positional", []):
        raise Unreplayable("the entry is a bound method; replay needs its instance")
    leaves = _tensor_leaves(call["args"])
    plan = {}
    for name, d in leaves:
        floating = "float" in d["dtype"] or "complex" in d["dtype"]
        if d.get("data") is None and not floating:
            raise Unreplayable("integer operand %s has no recorded contents (%s)"
                               % (name, d.get("why_no_data", "")))
        if d.get("data") is not None and d["data"] >= len(data):
            raise Unreplayable("operand %s refers to missing recorded data" % name)
        plan[name] = dict(d, rows=int(d["shape"][0]) if d["shape"] else None)
    return plan, notes


def _compact(plan, data, budget, notes):
    """Shrink pools (bulk float operands with no recorded contents) to the rows the index
    operands touch, renumbering those ids. Refuses whatever it cannot do exactly."""
    torch = __import__("torch")
    pools = [n for n, p in plan.items() if p.get("data") is None and p["shape"]
             and p["nbytes"] >= (64 << 20)]
    idx = [n for n, p in plan.items() if p.get("data") is not None
           and INDEX_ARG_RE.search(n.split("[")[0]) and "float" not in p["dtype"]]
    if not pools or not idx:
        raise Unreplayable("needs %.1f GiB > budget %.1f GiB and has no pool/index pair to "
                           "compact" % (sum(_bytes(p) for p in plan.values()) / 2**30, budget / 2**30))
    rows = {int(plan[n]["shape"][0]) for n in pools}
    if len(rows) != 1:
        raise Unreplayable("pools %s differ in leading size %s; cannot compact them with one "
                           "renumbering" % (pools, sorted(rows)))
    n_rows = rows.pop()
    vals = torch.cat([data[plan[n]["data"]].reshape(-1).to(torch.int64) for n in idx])
    vals = vals[vals >= 0]
    top = int(vals.max()) if vals.numel() else -1
    page = int(plan[pools[0]]["shape"][1]) if len(plan[pools[0]]["shape"]) > 1 else 1
    if top < n_rows:
        unit = 1
    elif top < n_rows * page and all(len(plan[n]["shape"]) > 1 and int(plan[n]["shape"][1]) == page
                                     for n in pools):
        unit = page
    else:
        raise Unreplayable("index operands reach %d, past the pools' %d rows" % (top, n_rows))
    used = torch.unique(vals // unit)
    if used.numel() >= n_rows:
        raise Unreplayable("index operands touch every pool row; nothing to compact")
    for n in idx:
        t = data[plan[n]["data"]]
        v = t.to(torch.int64)
        hit = v >= 0
        new = torch.searchsorted(used, (v // unit).clamp(min=0)) * unit + v % unit
        data[plan[n]["data"]] = torch.where(hit, new, v).to(t.dtype)
    for n in pools:
        plan[n]["shape"] = [int(used.numel())] + list(plan[n]["shape"][1:])
        plan[n]["compacted"] = True
    notes.append("pools %s compacted from %d to %d rows (unit %d) through %s"
                 % (pools, n_rows, used.numel(), unit, idx))


MAX_INT_32 = 2**31 - 1


def _elems(p):
    """Elements to allocate: the recorded storage, so storage-dependent specialization (AMD's
    `tt.pointer_range` for storages <= 2 GiB) matches the server. A compacted pool keeps
    only the side of the 2 GiB line its original storage was on."""
    torch = __import__("torch")
    size = _itemsize(torch, p["dtype"])
    if p.get("compacted"):
        floor = (MAX_INT_32 // size + 1) if int(p.get("storage_nbytes") or 0) > MAX_INT_32 else 0
        return max(_span(p), floor)
    return max(_span(p), int(p.get("storage_nbytes") or 0) // size)


def _bytes(p):
    torch = __import__("torch")
    return _elems(p) * _itemsize(torch, p["dtype"])


def materialize_call(call, data, seed=0, budget=16 << 30):
    """(args, kwargs, named_tensors, notes) rebuilding a recorded call on this process's
    device. Recorded contents are restored exactly, with the recorded strides and storage
    offset; bulk float operands are drawn from N(0, 1) with a per-operand seed, so both legs
    of a replay see identical operands."""
    torch = __import__("torch")
    data = [t.to_dense() if getattr(t, "is_sparse", False) else t for t in data]
    plan, notes = plan_call(call, data)
    if sum(_bytes(p) for p in plan.values()) > budget:
        _compact(plan, data, budget, notes)
        if sum(_bytes(p) for p in plan.values()) > budget:
            raise Unreplayable("still %.1f GiB after compaction" %
                               (sum(_bytes(p) for p in plan.values()) / 2**30))
    built = {}
    for i, (name, p) in enumerate(sorted(plan.items())):
        dt = _dtype(torch, p["dtype"])
        dev = "cuda" if p["device"].startswith("cuda") and torch.cuda.is_available() else p["device"]
        n = _elems(p)
        buf = torch.empty(n, dtype=dt, device=dev)
        if p.get("data") is None:
            g = torch.Generator(device=dev).manual_seed(int(seed) * 1000003 + i)
            step = 1 << 26
            for lo in range(0, n, step):
                hi = min(n, lo + step)
                buf[lo:hi].copy_(torch.randn(hi - lo, generator=g, device=dev,
                                             dtype=torch.float32).to(dt))
            built[name] = buf.as_strided(p["shape"], p["stride"], p["offset"])
        else:
            src = data[p["data"]].to(dev)
            if 0 in [int(st) for st, sz in zip(p["stride"], p["shape"]) if int(sz) > 1]:
                built[name] = src.contiguous()
                notes.append("%s: broadcast (stride 0) operand rebuilt contiguous" % name)
            else:
                view = buf.as_strided(p["shape"], p["stride"], p["offset"])
                view.copy_(src.reshape(p["shape"]))
                built[name] = view

    def value(name, d):
        if isinstance(d, dict) and d.get("__tensor__"):
            return built[name]
        if isinstance(d, dict) and "__dtype__" in d:
            return _dtype(torch, d["__dtype__"])
        if isinstance(d, dict) and "__float__" in d:
            return float(d["__float__"])
        if isinstance(d, dict) and "__seq__" in d:
            items = [value("%s[%d]" % (name, i), v) for i, v in enumerate(d["items"])]
            return tuple(items) if d["__seq__"] == "tuple" else items
        return d
    args, kwargs = [], {}
    for name in call.get("positional") or []:
        if name != "self":
            args.append(value(name, call["args"][name]))
    for name, d in call["args"].items():
        if name in (call.get("positional") or []) or name == "self":
            continue
        if name.startswith("**"):
            kwargs.update({k: value(k, v) for k, v in d.items()})
        elif name.startswith("*"):
            args.extend(value(name, d))
        else:
            kwargs[name] = value(name, d)
    return args, kwargs, built, notes


def storage_fingerprint(t):
    """(byte sum, byte square sum) over a tensor's whole storage: cheap, on device."""
    torch = __import__("torch")
    u8 = torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage())
    s = q = 0
    for chunk in u8.split(1 << 26):
        c = chunk.to(torch.int32)
        s += int(c.sum(dtype=torch.int64))
        q += int((c * c).sum(dtype=torch.int64))
    return s, q


def snapshot_output(x):
    """CPU copy of an output; a tensor over SNAPSHOT_CAP becomes per-row sums (float64)."""
    torch = __import__("torch")
    if isinstance(x, torch.Tensor):
        if x.numel() * x.element_size() <= SNAPSHOT_CAP or x.dim() == 0:
            return x.detach().cpu().clone()
        sums, abss = [], []
        step = max(1, (64 << 20) // max(1, x[0].numel() * 4))
        for lo in range(0, x.shape[0], step):
            blk = x[lo:lo + step].detach().float().reshape(min(step, x.shape[0] - lo), -1)
            sums.append(blk.sum(1).double().cpu())
            abss.append(blk.abs().sum(1).double().cpu())
        return {"__rows__": True, "shape": list(x.shape), "dtype": str(x.dtype),
                "sum": torch.cat(sums), "abs": torch.cat(abss)}
    if isinstance(x, (list, tuple)):
        return [snapshot_output(v) for v in x]
    if isinstance(x, dict):
        return {k: snapshot_output(v) for k, v in x.items()}
    return x if x is None or isinstance(x, (int, float, bool, str)) else repr(x)


def _flatten(x, path, out):
    torch = __import__("torch")
    if isinstance(x, torch.Tensor) or (isinstance(x, dict) and x.get("__rows__")):
        out[path] = x
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _flatten(v, "%s[%d]" % (path, i), out)
    elif isinstance(x, dict):
        for k, v in x.items():
            _flatten(v, "%s.%s" % (path, k), out)
    else:
        out[path] = x


def compare_call_outputs(base, cand, tol, ignore=()):
    """Every output of one replayed call: integers exactly, floats to `tol` of the baseline's
    largest magnitude, NaN where the baseline has NaN, row sums for snapshotted pools."""
    torch = __import__("torch")
    fb, fc = {}, {}
    _flatten(base, "out", fb)
    _flatten(cand, "out", fc)
    rows, ok = [], True
    for k in sorted(set(fb) | set(fc)):
        if any(k == "out.mutated." + i or k.startswith("out.mutated.%s[" % i) for i in ignore):
            continue
        b, c = fb.get(k), fc.get(k)
        r = {"output": k}
        if k not in fb or k not in fc:
            r.update(ok=False, why="written by the %s only" % ("baseline" if k in fb else "candidate"))
        elif isinstance(b, dict) and b.get("__rows__"):
            if not (isinstance(c, dict) and c.get("__rows__")) or c["shape"] != b["shape"]:
                r.update(ok=False, why="shape differs")
            else:
                err = float(((c["sum"] - b["sum"]).abs() / (b["abs"] + 1e-6)).max())
                r.update(ok=err <= tol, row_sum_rel_err=err)
        elif isinstance(b, torch.Tensor):
            if not isinstance(c, torch.Tensor) or c.shape != b.shape or c.dtype != b.dtype:
                r.update(ok=False, why="shape/dtype %s vs %s" % (
                    (tuple(b.shape), str(b.dtype)),
                    (tuple(c.shape), str(c.dtype)) if isinstance(c, torch.Tensor) else type(c)))
            elif not (b.is_floating_point() or b.is_complex()):
                bad = int((b != c).sum())
                r.update(ok=bad == 0, mismatched=bad, numel=b.numel())
            else:
                bf, cf = b.double(), c.double()
                nan_b, nan_c = torch.isnan(bf), torch.isnan(cf)
                same_nan = bool((nan_b == nan_c).all())
                fin = torch.isfinite(bf) & torch.isfinite(cf)
                scale = float(bf[fin].abs().max()) if bool(fin.any()) else 0.0
                err = float((cf - bf)[fin].abs().max()) / max(scale, 1e-6) if bool(fin.any()) else 0.0
                inf_ok = bool((torch.isinf(bf) == torch.isinf(cf)).all())
                r.update(ok=same_nan and inf_ok and err <= tol, rel_err=err, nan_match=same_nan)
        else:
            r.update(ok=b == c, base=b, cand=c)
        ok = ok and r["ok"]
        rows.append(r)
    return ok, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ready-epoch", type=float, default=None)
    p.add_argument("--server-mode", default=None)
    a = ap.parse_args(argv)
    doc = collect(a.dir, a.ready_epoch, a.server_mode)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    print("compile_watch: %d variant(s) from %d process(es) -> %s"
          % (len(doc["variants"]), doc["processes"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
