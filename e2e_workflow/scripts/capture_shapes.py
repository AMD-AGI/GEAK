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
import atexit, importlib, json, os, sys, threading

_STATE = {
    "target": None, "out_dir": None, "max_cases": 5, "num_steps": 0,
    "records": [], "seen": set(), "lock": threading.Lock(), "orig": None,
    "mod": None, "attr": None, "installed": False, "calls": 0,
}


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
    try:
        sig = _sig(args, kwargs)
        with s["lock"]:
            if sig not in s["seen"] and len(s["records"]) < s["max_cases"]:
                s["seen"].add(sig)
                s["records"].append({
                    "sig": sig,
                    "args": _snapshot(args),
                    "kwargs": _snapshot(kwargs),
                    "output": _snapshot(out),
                })
                sys.stderr.write(f"[capture_shapes] recorded case {len(s['records'])}: {sig}\n")
    except Exception as e:  # never break the server because capture failed
        sys.stderr.write(f"[capture_shapes] capture error (ignored): {e}\n")
    return out


def _flush():
    s = _STATE
    if not s["records"]:
        sys.stderr.write("[capture_shapes] no records captured; nothing to flush\n")
        return
    torch = _torch()
    out_dir = s["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    io_path = os.path.join(out_dir, "reference_io.pt")
    torch.save({"target": s["target"], "records": s["records"]}, io_path)
    # meta.json: shapes/dtypes + a checksum of the oracle so the validator can detect tampering.
    import hashlib
    h = hashlib.sha256()
    with open(io_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    cases = []
    for r in s["records"]:
        shapes, dtypes = [], []
        def walk(o):
            if isinstance(o, dict) and o.get("__tensor__"):
                shapes.append(o["shape"]); dtypes.append(o["dtype"])
            elif isinstance(o, (list, tuple)):
                for v in o: walk(v)
            elif isinstance(o, dict):
                for v in o.values(): walk(v)
        walk(r["args"]); walk(r["kwargs"])
        cases.append({"sig": r["sig"], "input_shapes": shapes, "input_dtypes": sorted(set(dtypes))})
    meta = {
        "target": s["target"],
        "module": s["mod"].__name__ if s["mod"] else None,
        "attr": s["attr"],
        "num_cases": len(s["records"]),
        "total_calls_observed": s["calls"],
        "cases": cases,
        "reference_io": "reference_io.pt",
        "reference_io_sha256": h.hexdigest(),
        "build": False,  # default: pure-python/triton; Extractor flips to True for HIP/CK/asm tasks
        "note": "Oracle captured from baseline. Do NOT edit unittest.py or reference_io.pt during opt.",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    sys.stderr.write(f"[capture_shapes] flushed {len(s['records'])} case(s) -> {io_path}\n")


def install(target, out_dir, max_cases=5):
    """Wrap module:attr to record I/O. Registers an atexit flush. Idempotent."""
    s = _STATE
    if s["installed"]:
        return
    mod_name, attr = target.split(":")
    mod = importlib.import_module(mod_name)
    orig = getattr(mod, attr)
    s.update(target=target, out_dir=out_dir, max_cases=int(max_cases),
             orig=orig, mod=mod, attr=attr, installed=True)
    setattr(mod, attr, _wrapper)
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
