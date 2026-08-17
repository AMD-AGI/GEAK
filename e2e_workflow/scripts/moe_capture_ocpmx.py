#!/usr/bin/env python3
"""I/O-oracle capture for the FUSED mxfp4 EMULATION grouped-MoE seam
`vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe:OCP_MXQuantizationEmulationTritonExperts.apply`.

Why a custom hook (not scripts/capture_shapes.py):
  * `apply` is an INSTANCE METHOD whose fused body reads ~20 attributes off `self`
    (quant_config, moe_config, w1_scale_val, w2_scale_val, ocp_mx_scheme, gemm1_*, block_shape, ...).
    A standalone replay therefore needs the WHOLE `self`, not just the call args. We pickle `self` ONCE
    (torch.save stores nested tensors with device info; torch.load(map_location="cpu") remaps them),
    so the single-GPU isolated harness can reconstruct the op with `Class.__new__` + `__dict__.update`
    and dispatch `.apply` to whichever module leg (baseline install vs kernel_src overlay) is on
    PYTHONPATH — pickle re-imports the class by dotted name at load, so the SAME captured state runs the
    baseline code on the baseline leg and the candidate code on the candidate leg.
  * `apply` writes its result IN-PLACE into the pre-allocated `output` buffer and returns None. The
    golden is `output` AFTER the call; we also record the pre-call buffer so the harness restores it.
  * The packed mxfp4 expert weights w1/w2 (uint8) are CONSTANT across calls -> stored once in a shared
    slot alongside `self`. Per-case records hold only the varying activation + routing + golden.
  * Routing (topk_ids/topk_weights) is NOT value-independent for MoE (skew drives block/padding counts),
    so it is captured REAL from the live server, never synthesized.

Snapshot is EAGER-ONLY: a clone during CUDA-graph capture is illegal, so calls made while a HIP/CUDA
graph is capturing are counted but not recorded (matches the mxfp8 hook contract).

Exposes `install(target, out_dir, max_cases)` (same as capture_shapes.py) so the overlay sitecustomize
`import capture_shapes; capture_shapes.install(...)` drives it unchanged when copied in as the overlay's
capture_shapes.py (via `overlay_setup.py add-capture --capture-file`).

Env overrides: CAPTURE_OUT, CAPTURE_MAX, CAPTURE_DECODE_LEAD_MAX (decode/prefill cutoff on M).
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
    "const": None,          # constant: self (pickled), w1, w2
    "self_pickle_err": None,
    "seen": set(),
    "regime_seen": set(),
    "shape_counts": {},
    "lock": threading.Lock(),
    "calls": 0,
    "in_graph_calls": 0,
    "sequence": [],
    "seq_cap": 256,
    "installed": False,
    "orig": None,           # unbound original apply
    "cls": None,
    "sig": None,
    "written": False,
    "oracle_sha": None,
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


def _capturing():
    try:
        return bool(_torch().cuda.is_current_stream_capturing())
    except Exception:
        return False


def _lead_regime(m):
    return "decode" if int(m) <= _S["decode_lead_max"] else "prefill"


def _meta_to_dict(etm):
    """ExpertTokensMetadata -> plain dict of cpu tensors (or None)."""
    if etm is None:
        return None
    d = {}
    for k in ("expert_num_tokens", "expert_num_tokens_cpu"):
        v = getattr(etm, k, None)
        d[k] = _cpu(v)
    return d


def _wrapper(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
             activation, global_num_experts, expert_map, a1q_scale, a2_scale,
             workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
    torch = _torch()
    in_graph = _capturing()
    out_pre = None
    if not in_graph:
        try:
            out_pre = output.detach().to("cpu").clone()
        except Exception:
            out_pre = None
    # run the REAL op first (writes output in place) so output becomes the golden
    ret = _S["orig"](self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                     activation, global_num_experts, expert_map, a1q_scale, a2_scale,
                     workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input)
    _S["calls"] += 1
    try:
        m = int(hidden_states.shape[0])
        sig = f"M{m}|topk{tuple(topk_ids.shape)}|{hidden_states.dtype}"
        regime = _lead_regime(m)
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
            if new and not in_graph and (room or need_regime):
                _S["seen"].add(sig)
                _S["regime_seen"].add(regime)
                if _S["const"] is None:
                    _S["const"] = {"self": self, "w1": _cpu(w1), "w2": _cpu(w2)}
                    sys.stderr.write("[moe_capture_ocpmx] stored constant slot (self+w1+w2)\n")
                rec = {
                    "sig": sig,
                    "regime": regime,
                    "m": m,
                    "hidden_states": _cpu(hidden_states),
                    "topk_weights": _cpu(topk_weights),
                    "topk_ids": _cpu(topk_ids),
                    "activation": activation,
                    "global_num_experts": int(global_num_experts),
                    "expert_map": _cpu(expert_map),
                    "a1q_scale": _cpu(a1q_scale),
                    "a2_scale": _cpu(a2_scale),
                    "ws13_shape": list(workspace13.shape), "ws13_dtype": str(workspace13.dtype),
                    "ws2_shape": list(workspace2.shape), "ws2_dtype": str(workspace2.dtype),
                    "expert_tokens_meta": _meta_to_dict(expert_tokens_meta),
                    "apply_router_weight_on_input": bool(apply_router_weight_on_input),
                    "output_shape": list(output.shape), "output_dtype": str(output.dtype),
                    "output_pre": out_pre,
                    "golden": output.detach().to("cpu").clone(),
                }
                _S["records"].append(rec)
                added = True
                sys.stderr.write(
                    f"[moe_capture_ocpmx] recorded case {len(_S['records'])} ({regime}) {sig}\n")
        if added:
            _flush()
    except Exception as e:
        sys.stderr.write(f"[moe_capture_ocpmx] capture error (ignored): {e!r}\n")
    return ret


def _flush():
    torch = _torch()
    out_dir = _S["out_dir"]
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with _S["lock"]:
        records = list(_S["records"])
        const = _S["const"]
        shape_counts = dict(_S["shape_counts"])
        sequence = list(_S["sequence"])
    if not records or const is None:
        return
    io_path = os.path.join(out_dir, "reference_io.pt")
    # per-pid tmp so concurrent TP workers writing the SAME out dir can't corrupt each other's file;
    # the final os.replace is atomic, so reference_io.pt is always ONE complete single-rank oracle.
    tmp = io_path + f".{os.getpid()}.tmp"
    # try to pickle self; if it fails, drop self and record the error (harness then reconstructs
    # from a curated attr subset — but normally self pickles fine).
    try:
        torch.save({"target": _S["target"], "const": const, "records": records}, tmp)
    except Exception as e:
        _S["self_pickle_err"] = repr(e)
        sys.stderr.write(f"[moe_capture_ocpmx] FULL-self pickle failed: {e!r}; retrying w/ attr subset\n")
        self_obj = const["self"]
        curated = {}
        probe = io_path + f".{os.getpid()}.probe"
        for k, v in vars(self_obj).items():
            try:
                torch.save(v, probe); curated[k] = v
            except Exception:
                curated[k] = ("__UNPICKLABLE__", repr(type(v)))
        try:
            os.remove(probe)
        except Exception:
            pass
        const2 = {"self_dict": curated, "self_class": type(self_obj).__module__ + ":" + type(self_obj).__qualname__,
                  "w1": const["w1"], "w2": const["w2"]}
        torch.save({"target": _S["target"], "const": const2, "records": records,
                    "self_pickle_err": _S["self_pickle_err"]}, tmp)
    os.replace(tmp, io_path)
    h = hashlib.sha256()
    with open(io_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    _S["oracle_sha"] = h.hexdigest()
    _S["written"] = True
    cases = [{"sig": r["sig"], "regime": r["regime"], "m": r["m"],
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
        "self_pickle_err": _S["self_pickle_err"],
        "oracle_complete": True,
        "build": False,
        "note": "OCP-MX mxfp4 emulation fused-MoE oracle from baseline apply(). self captured once. IMMUTABLE.",
    }
    with open(os.path.join(out_dir, "capture_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    sys.stderr.write(f"[moe_capture_ocpmx] flushed {len(records)} case(s) "
                     f"(regimes={sorted(_S['regime_seen'])}) -> {out_dir}\n")


def install(target, out_dir, max_cases=6):
    if _S["installed"]:
        return
    import importlib
    _S["target"] = target
    _S["out_dir"] = out_dir
    _S["max_cases"] = int(max_cases)
    mod_name, attr = target.split(":")
    cls_name, meth = attr.split(".")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    orig = getattr(cls, meth)
    _S["orig"] = orig
    _S["cls"] = cls
    _S["sig"] = inspect.signature(orig)
    _S["installed"] = True
    setattr(cls, meth, _wrapper)
    atexit.register(_flush)
    sys.stderr.write(
        f"[moe_capture_ocpmx] hooked {target}; up to {_S['max_cases']} cases -> {out_dir}\n")


def install_from_env():
    t = os.environ.get("CAPTURE_TARGET")
    o = os.environ.get("CAPTURE_OUT")
    if t and o:
        install(t, o, int(os.environ.get("CAPTURE_MAX", "6")))


if os.environ.get("CAPTURE_TARGET") and os.environ.get("CAPTURE_OUT"):
    try:
        install_from_env()
    except Exception as e:
        sys.stderr.write(f"[moe_capture_ocpmx] install_from_env failed: {e!r}\n")
