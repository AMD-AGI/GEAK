#!/usr/bin/env python3
"""I/O-oracle capture for the sglang native MXFP8 fused-MoE seam
`sglang.kernels.ops.moe.mxfp8_moe_amd_gfx95:fused_experts_mxfp8`.

Why a custom hook (not the generic scripts/capture_shapes.py): the fused MoE receives the FULL
per-rank expert weight set (w1/w2 fp8 + w1_scale/w2_scale uint8 E8M0) on EVERY call. Those weights are
CONSTANT across calls, but the generic hook snapshots ALL args PER case -> the oracle balloons to
max_cases x (hundreds of MB of weights). This hook stores the constant weights ONCE in a shared
`weights` slot and records only the VARYING inputs (hidden_states, topk_weights, topk_ids + the scalar
config) + the golden output per case. All operands here are PLAIN torch tensors (no proprietary swizzle),
so a straight detach->cpu->clone reproduces the oracle exactly.

Exposes the SAME `install(target, out_dir, max_cases)` entrypoint as capture_shapes.py, so the overlay
sitecustomize (which does `import capture_shapes; capture_shapes.install(...)`) drives it unchanged when
this file is copied in as the overlay's capture_shapes.py (via overlay_setup add-capture --capture-file).

Env overrides: CAPTURE_OUT, CAPTURE_MAX, CAPTURE_DECODE_LEAD_MAX (decode/prefill cutoff on T).
"""
import atexit
import hashlib
import inspect
import json
import os
import sys
import threading

_S = {
    "target": None,
    "out_dir": os.environ.get("CAPTURE_OUT"),
    "max_cases": int(os.environ.get("CAPTURE_MAX", "6")),
    "decode_lead_max": int(os.environ.get("CAPTURE_DECODE_LEAD_MAX", "256")),
    "records": [],          # per-case varying inputs + golden
    "weights": None,        # constant: w1, w2, w1_scale, w2_scale, expert_map, config scalars
    "seen": set(),
    "regime_seen": set(),
    "shape_counts": {},
    "lock": threading.Lock(),
    "calls": 0,
    "in_graph_calls": 0,
    "sequence": [],
    "seq_cap": 256,
    "installed": False,
    "orig": None,
    "mod": None,
    "attr": None,
    "sig": None,
    "written": False,
    "oracle_sha": None,
    "oracle_records": 0,
}


def _torch():
    import torch
    return torch


def _cpu(x):
    torch = _torch()
    if torch.is_tensor(x):
        return x.detach().to("cpu").clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    return x


def _lead_regime(hidden_states):
    torch = _torch()
    try:
        m = int(hidden_states.shape[0])
    except Exception:
        return "decode"
    return "decode" if m <= _S["decode_lead_max"] else "prefill"


def _capturing():
    try:
        return bool(_torch().cuda.is_current_stream_capturing())
    except Exception:
        return False


def _bind(args, kwargs):
    """Normalize positional+kwargs into a name->value dict using the original signature."""
    ba = _S["sig"].bind(*args, **kwargs)
    ba.apply_defaults()
    return dict(ba.arguments)


def _case_sig(hidden_states, topk_ids):
    return f"T{int(hidden_states.shape[0])}|topk{tuple(topk_ids.shape)}|{hidden_states.dtype}"


def _wrapper(*args, **kwargs):
    # run the real op FIRST so its return value is the golden output
    ret = _S["orig"](*args, **kwargs)
    _S["calls"] += 1
    in_graph = _capturing()
    try:
        a = _bind(args, kwargs)
        hidden_states = a["hidden_states"]
        topk_ids = a["topk_ids"]
        sig = _case_sig(hidden_states, topk_ids)
        regime = _lead_regime(hidden_states)
        added = False
        with _S["lock"]:
            if in_graph:
                _S["in_graph_calls"] += 1
            _S["shape_counts"][sig] = _S["shape_counts"].get(sig, 0) + 1
            if len(_S["sequence"]) < _S["seq_cap"]:
                _S["sequence"].append({"sig": sig, "in_graph": in_graph})
            need_regime = regime not in _S["regime_seen"]
            room = len(_S["records"]) < _S["max_cases"]
            new = sig not in _S["seen"]
            # Oracle snapshot is EAGER-ONLY (a clone during CUDA-graph capture is illegal).
            if new and not in_graph and (room or need_regime):
                _S["seen"].add(sig)
                _S["regime_seen"].add(regime)
                if _S["weights"] is None:
                    _S["weights"] = {
                        "w1": _cpu(a["w1"]),
                        "w2": _cpu(a["w2"]),
                        "w1_scale": _cpu(a["w1_scale"]),
                        "w2_scale": _cpu(a["w2_scale"]),
                        "b1": _cpu(a.get("b1")),
                        "b2": _cpu(a.get("b2")),
                        "expert_map": _cpu(a.get("expert_map")),
                        # scalar / config kwargs (constant across calls) — replayed verbatim
                        "activation": a.get("activation"),
                        "is_gated": a.get("is_gated"),
                        "no_combine": a.get("no_combine"),
                        "inplace": a.get("inplace"),
                        "apply_router_weight_on_input": a.get("apply_router_weight_on_input"),
                        "routed_scaling_factor": a.get("routed_scaling_factor"),
                        "gemm1_alpha": a.get("gemm1_alpha"),
                        "gemm1_limit": a.get("gemm1_limit"),
                        "swiglu_limit": a.get("swiglu_limit"),
                        "gate_up_interleaved": a.get("gate_up_interleaved"),
                    }
                    sys.stderr.write("[moe_capture_mxfp8] stored constant weights slot\n")
                rec = {
                    "sig": sig,
                    "regime": regime,
                    "T": int(hidden_states.shape[0]),
                    "hidden_states": _cpu(hidden_states),
                    "topk_weights": _cpu(a["topk_weights"]),
                    "topk_ids": _cpu(topk_ids),
                    "golden": _cpu(ret),
                }
                _S["records"].append(rec)
                added = True
                sys.stderr.write(
                    f"[moe_capture_mxfp8] recorded case {len(_S['records'])} ({regime}) {sig}\n")
        if added:
            _flush()
    except Exception as e:
        sys.stderr.write(f"[moe_capture_mxfp8] capture error (ignored): {e!r}\n")
    return ret


def _flush():
    torch = _torch()
    out_dir = _S["out_dir"]
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with _S["lock"]:
        records = list(_S["records"])
        weights = _S["weights"]
        shape_counts = dict(_S["shape_counts"])
        sequence = list(_S["sequence"])
    if not records or weights is None:
        return
    io_path = os.path.join(out_dir, "reference_io.pt")
    tmp = io_path + ".tmp"
    torch.save({"target": _S["target"], "weights": weights, "records": records}, tmp)
    os.replace(tmp, io_path)
    h = hashlib.sha256()
    with open(io_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    _S["oracle_sha"] = h.hexdigest()
    _S["written"] = True
    _S["oracle_records"] = len(records)
    cases = [{"sig": r["sig"], "regime": r["regime"], "T": r["T"],
              "count": shape_counts.get(r["sig"], 0)} for r in records]
    meta = {
        "target": _S["target"],
        "num_cases": len(records),
        "total_calls_observed": _S["calls"],
        "regimes_covered": sorted(_S["regime_seen"]),
        "cases": cases,
        "shape_counts": sorted(({"sig": k, "count": v} for k, v in shape_counts.items()),
                               key=lambda e: e["count"], reverse=True),
        "num_distinct_shapes": len(shape_counts),
        "call_sequence": sequence,
        "graph_replayed": bool(_S["in_graph_calls"] > 0),
        "in_graph_calls": _S["in_graph_calls"],
        "reference_io": "reference_io.pt",
        "reference_io_sha256": _S["oracle_sha"],
        "oracle_complete": True,
        "build": False,
        "note": "MoE oracle captured from baseline fused_experts_mxfp8. IMMUTABLE. weights stored once.",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    sys.stderr.write(f"[moe_capture_mxfp8] flushed {len(records)} case(s) "
                     f"(regimes={sorted(_S['regime_seen'])}) -> {out_dir}\n")


def install(target, out_dir, max_cases=6):
    """Same signature as capture_shapes.install so the overlay sitecustomize drives it unchanged."""
    if _S["installed"]:
        return
    import importlib
    _S["target"] = target
    _S["out_dir"] = out_dir
    _S["max_cases"] = int(max_cases)
    mod_name, attr = target.split(":")
    mod = importlib.import_module(mod_name)
    orig = getattr(mod, attr)
    _S["orig"] = orig
    _S["mod"] = mod
    _S["attr"] = attr
    _S["sig"] = inspect.signature(orig)
    _S["installed"] = True
    setattr(mod, attr, _wrapper)
    atexit.register(_flush)
    sys.stderr.write(
        f"[moe_capture_mxfp8] hooked {target}; up to {_S['max_cases']} cases -> {out_dir}\n")


def install_from_env():
    t = os.environ.get("CAPTURE_TARGET")
    o = os.environ.get("CAPTURE_OUT")
    if t and o:
        install(t, o, int(os.environ.get("CAPTURE_MAX", "6")))


if os.environ.get("CAPTURE_TARGET") and os.environ.get("CAPTURE_OUT"):
    try:
        install_from_env()
    except Exception as e:
        sys.stderr.write(f"[moe_capture_mxfp8] install_from_env failed: {e!r}\n")
