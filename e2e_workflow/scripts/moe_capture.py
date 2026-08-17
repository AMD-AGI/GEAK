#!/usr/bin/env python3
"""Custom I/O-oracle capture for the FUSED mxfp4 grouped MoE seam
`UnfusedOAITritonExperts.apply` (DeepSeek-V4 / gpt-oss TRITON_UNFUSED path).

The generic scripts/capture_shapes.py drops the mxfp4 expert weights: they arrive as
triton_kernels.tensor.Tensor objects (proprietary swizzle) which its _snapshot() turns into a
`{"__repr__":...}` placeholder, so the oracle can never be replayed. This hook instead pickles the
REAL objects (weights, quant_config, routing) via torch.save; nested torch tensors (including those
inside the triton Tensor storage and the FusedMoEQuantConfig) are remapped to CPU at load time via
`map_location`, so the file is portable to the single-GPU isolated harness.

The mxfp4 expert weights + quant_config + expert_map are CONSTANT across calls, so they are stored
ONCE in a shared `weights` slot; per-case records hold only the varying activation/routing + the
golden output. Hooked as a class-attribute monkeypatch (apply is a method).

Env:  CAPTURE_OUT=<task_dir>  [CAPTURE_MAX=6]  [CAPTURE_DECODE_LEAD_MAX=256]
Registered from an overlay sitecustomize that does `import moe_capture` (the overlay copies this file
in via overlay_setup add-capture --capture-file).
"""
import atexit
import hashlib
import inspect
import json
import os
import sys
import threading

_TARGET = "vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe:UnfusedOAITritonExperts.apply"

_S = {
    "out_dir": os.environ.get("CAPTURE_OUT"),
    "max_cases": int(os.environ.get("CAPTURE_MAX", "6")),
    "decode_lead_max": int(os.environ.get("CAPTURE_DECODE_LEAD_MAX", "256")),
    "records": [],            # per-case varying inputs + golden
    "weights": None,          # constant: w1, w2, quant_config, expert_map, ...
    "seen": set(),
    "regime_seen": set(),
    "lock": threading.Lock(),
    "calls": 0,
    "installed": False,
    "orig": None,
    "cls": None,
    "shape_counts": {},
    "written": False,
    "oracle_sha": None,
}


def _torch():
    import torch
    return torch


def _cpu(x):
    """Recursively move plain torch tensors to detached CPU clones. Leave triton Tensor / dataclass /
    other objects intact — torch.save pickles them, and map_location='cpu' at load handles their nested
    tensors."""
    torch = _torch()
    if torch.is_tensor(x):
        return x.detach().to("cpu").clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    return x


def _try_pickle(obj):
    """Attempt to pickle-round-trip a single object (moving nested tensors to CPU). Returns
    (ok, value_or_reprplaceholder)."""
    import pickle
    try:
        # only move top-level / nested torch tensors to cpu; keep the object graph
        val = _cpu_deep_objects(obj)
        pickle.dumps(val)
        return True, val
    except Exception as e:
        return False, {"__unpicklable__": True, "type": type(obj).__name__, "err": repr(e)[:300]}


def _cpu_deep_objects(x):
    """Like _cpu but also descends into triton Tensor.storage.data and dataclass instances so nested
    CUDA tensors become CPU without needing map_location. Falls back to returning x unchanged for
    opaque objects (torch.save + map_location will still remap their tensors)."""
    torch = _torch()
    if torch.is_tensor(x):
        return x.detach().to("cpu").clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu_deep_objects(v) for v in x)
    if isinstance(x, dict):
        return {k: _cpu_deep_objects(v) for k, v in x.items()}
    return x


def _lead_regime(hidden_states):
    torch = _torch()
    try:
        m = int(hidden_states.shape[0])
    except Exception:
        return "decode"
    return "decode" if m <= _S["decode_lead_max"] else "prefill"


def _sig(hidden_states, topk_ids):
    return f"M{int(hidden_states.shape[0])}|topk{tuple(topk_ids.shape)}|{hidden_states.dtype}"


def _wrapper(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
             global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
             expert_tokens_meta, apply_router_weight_on_input):
    torch = _torch()
    # run the real op first so `output` holds the golden result
    ret = _S["orig"](self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                     global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
                     expert_tokens_meta, apply_router_weight_on_input)
    _S["calls"] += 1
    try:
        in_graph = torch.cuda.is_current_stream_capturing()
    except Exception:
        in_graph = False
    try:
        sig = _sig(hidden_states, topk_ids)
        regime = _lead_regime(hidden_states)
        added = False
        with _S["lock"]:
            _S["shape_counts"][sig] = _S["shape_counts"].get(sig, 0) + 1
            need_regime = regime not in _S["regime_seen"]
            room = len(_S["records"]) < _S["max_cases"]
            new = sig not in _S["seen"]
            # Oracle snapshot is EAGER-ONLY: a clone during CUDA-graph capture is illegal.
            if new and not in_graph and (room or need_regime):
                _S["seen"].add(sig)
                _S["regime_seen"].add(regime)
                # constant weights slot: store ONCE
                if _S["weights"] is None:
                    ok_qc, qc = _try_pickle(self.quant_config)
                    _S["weights"] = {
                        "w1": _cpu_deep_objects(w1),
                        "w2": _cpu_deep_objects(w2),
                        "quant_config": qc,
                        "quant_config_picklable": ok_qc,
                        "expert_map": _cpu(expert_map),
                        "lora_context": getattr(self, "_lora_context", None),
                        "self_class": type(self).__qualname__,
                        "self_module": type(self).__module__,
                    }
                    sys.stderr.write(
                        f"[moe_capture] stored constant weights slot; quant_config picklable={ok_qc}\n")
                rec = {
                    "sig": sig,
                    "regime": regime,
                    "M": int(hidden_states.shape[0]),
                    "hidden_states": _cpu(hidden_states),
                    "topk_weights": _cpu(topk_weights),
                    "topk_ids": _cpu(topk_ids),
                    "activation": activation,
                    "global_num_experts": int(global_num_experts),
                    "a1q_scale": _cpu(a1q_scale),
                    "a2_scale": _cpu(a2_scale),
                    "apply_router_weight_on_input": bool(apply_router_weight_on_input),
                    "output_shape": list(output.shape),
                    "output_dtype": str(output.dtype),
                    "workspace13_shape": list(workspace13.shape),
                    "workspace13_dtype": str(workspace13.dtype),
                    "workspace2_shape": list(workspace2.shape),
                    "workspace2_dtype": str(workspace2.dtype),
                    "expert_tokens_meta": None if expert_tokens_meta is None else "PRESENT",
                    "golden": output.detach().to("cpu").clone(),
                }
                _S["records"].append(rec)
                added = True
                sys.stderr.write(
                    f"[moe_capture] recorded case {len(_S['records'])} ({regime}) {sig}\n")
        # heavy torch.save happens OUTSIDE the lock so a forward is not blocked while another thread
        # writes; only when a NEW case was actually added (bounded by max_cases + regime coverage).
        if added:
            _flush()
    except Exception as e:
        sys.stderr.write(f"[moe_capture] capture error (ignored): {e!r}\n")
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
    if not records or weights is None:
        return
    # 8 TP workers share out_dir; write a PID-unique file ATOMICALLY (temp + os.replace) so concurrent
    # torch.save calls never interleave/corrupt a shared path. A post-capture step picks one complete
    # rank file and renames it to reference_io.pt. Each rank's file is internally consistent (its own
    # 48 local experts' weights + the routing/golden observed on that rank).
    pid = os.getpid()
    io_path = os.path.join(out_dir, f"reference_io.pt.{pid}")
    tmp = io_path + ".tmp"
    torch.save({"target": _TARGET, "weights": weights, "records": records}, tmp)
    os.replace(tmp, io_path)
    h = hashlib.sha256()
    with open(io_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    _S["oracle_sha"] = h.hexdigest()
    _S["written"] = True
    cases = [{"sig": r["sig"], "regime": r["regime"], "M": r["M"],
              "count": shape_counts.get(r["sig"], 0)} for r in records]
    meta = {
        "target": _TARGET,
        "num_cases": len(records),
        "total_calls_observed": _S["calls"],
        "regimes_covered": sorted(_S["regime_seen"]),
        "cases": cases,
        "shape_counts": sorted(({"sig": k, "count": v} for k, v in shape_counts.items()),
                               key=lambda e: e["count"], reverse=True),
        "reference_io": os.path.basename(io_path),
        "pid": pid,
        "reference_io_sha256": _S["oracle_sha"],
        "oracle_complete": True,
        "quant_config_picklable": bool(weights.get("quant_config_picklable")),
        "note": "MoE oracle captured from baseline UnfusedOAITritonExperts.apply. IMMUTABLE.",
    }
    with open(os.path.join(out_dir, f"capture_meta.json.{pid}"), "w") as fh:
        json.dump(meta, fh, indent=2)
    sys.stderr.write(f"[moe_capture] flushed {len(records)} case(s) "
                     f"(regimes={sorted(_S['regime_seen'])}) -> {out_dir}\n")


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def install(target=None, out=None, max=None):
    """Install the class-attribute hook. Callable two ways:
      (a) auto on import when CAPTURE_OUT is set (target/out/max come from env / _TARGET);
      (b) from the overlay sitecustomize as `install(target, out, max)` (manifest-driven) — in that
          case override _TARGET/out_dir/max_cases from the manifest so no env is required."""
    global _TARGET
    if target:
        _TARGET = target
    if out:
        _S["out_dir"] = out
    if max is not None:
        _S["max_cases"] = int(max)
    if _S["installed"]:
        return
    import importlib
    mod_name, attr = _TARGET.split(":")
    cls_name, meth = attr.split(".")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    orig = getattr(cls, meth)
    _S["orig"] = orig
    _S["cls"] = cls
    _S["installed"] = True
    setattr(cls, meth, _wrapper)
    atexit.register(_flush)
    sys.stderr.write(f"[moe_capture] hooked {_TARGET}; up to {_S['max_cases']} cases -> {_S['out_dir']}\n")


if _S["out_dir"]:
    try:
        install()
    except Exception as e:
        sys.stderr.write(f"[moe_capture] install failed: {e!r}\n")
