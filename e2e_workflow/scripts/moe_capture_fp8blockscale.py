#!/usr/bin/env python3
"""I/O-oracle capture for the FUSED fp8-blockscale grouped MoE seam
`vllm.model_executor.layers.fused_moe.experts.triton_moe:TritonExperts.apply`
(the AITER-OFF live path for fp8 blockscale MoE on ROCm).

Same class-method-hook mechanism as moe_capture.py (mxfp4 UnfusedOAITritonExperts). Differences for
fp8 blockscale:
  * The fp8 expert weights (w1/w2) are PLAIN torch tensors (float8_e4m3*), and the per-block weight
    scales + activation scale live inside `self.quant_config` (a FusedMoEQuantConfig) — apply() reads
    them via the `w1_scale`/`w2_scale`/`a1_scale`/`block_shape`/`per_act_token_quant`/`quant_dtype`
    PROPERTIES (all `return self.quant_config.<x>`). So capturing `self.quant_config` captures ALL the
    scales/flags needed to replay apply() faithfully — no per-arg scale plumbing.
  * apply() also reads three __init__-set scalars (gemm1_clamp_limit / gemm1_alpha / gemm1_beta) and
    `self._lora_context` (None here). We store those in the constant `weights` slot so the isolated
    harness can rebuild a `__new__` shim without the full FusedMoEConfig/model wiring.

The constant slot (w1, w2, quant_config, expert_map, gemm1_* , lora_context) is stored ONCE; per-case
records hold only the varying activation/routing + golden output. 8/2 TP workers each write a
PID-unique file atomically; a post step picks one complete rank shard as reference_io.pt.

Env: CAPTURE_OUT=<task_dir> [CAPTURE_MAX=6] [CAPTURE_DECODE_LEAD_MAX=256]
Registered from an overlay sitecustomize that does `import capture_shapes` (overlay_setup add-capture
copies THIS file in as capture_shapes.py).
"""
import atexit
import hashlib
import json
import os
import sys
import threading

_TARGET = "vllm.model_executor.layers.fused_moe.experts.triton_moe:TritonExperts.apply"

_S = {
    "out_dir": os.environ.get("CAPTURE_OUT"),
    "max_cases": int(os.environ.get("CAPTURE_MAX", "6")),
    "decode_lead_max": int(os.environ.get("CAPTURE_DECODE_LEAD_MAX", "256")),
    "records": [],
    "weights": None,
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
    torch = _torch()
    if torch.is_tensor(x):
        return x.detach().to("cpu").clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    return x


def _cpu_deep_objects(x):
    torch = _torch()
    if torch.is_tensor(x):
        return x.detach().to("cpu").clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu_deep_objects(v) for v in x)
    if isinstance(x, dict):
        return {k: _cpu_deep_objects(v) for k, v in x.items()}
    return x


def _try_pickle(obj):
    import pickle
    try:
        val = _cpu_deep_objects(obj)
        pickle.dumps(val)
        return True, val
    except Exception as e:
        # last-ditch: keep the object; torch.save + map_location remaps its tensors
        try:
            return True, _cpu_deep_objects(obj)
        except Exception:
            return False, {"__unpicklable__": True, "type": type(obj).__name__, "err": repr(e)[:300]}


def _lead_regime(hidden_states):
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
            if new and not in_graph and (room or need_regime):
                _S["seen"].add(sig)
                _S["regime_seen"].add(regime)
                if _S["weights"] is None:
                    ok_qc, qc = _try_pickle(self.quant_config)
                    _S["weights"] = {
                        "w1": _cpu_deep_objects(w1),
                        "w2": _cpu_deep_objects(w2),
                        "quant_config": qc,
                        "quant_config_picklable": ok_qc,
                        "expert_map": _cpu(expert_map),
                        "lora_context": None,   # no LoRA on this serve
                        # __init__-set scalars apply() reads (via self.*):
                        "gemm1_clamp_limit": getattr(self, "gemm1_clamp_limit", None),
                        "gemm1_alpha": getattr(self, "gemm1_alpha", 1.0),
                        "gemm1_beta": getattr(self, "gemm1_beta", 0.0),
                        "quantization_emulation": bool(getattr(self, "quantization_emulation", False)),
                        "self_class": type(self).__qualname__,
                        "self_module": type(self).__module__,
                        "w1_shape": list(w1.shape),
                        "w2_shape": list(w2.shape),
                        "w1_dtype": str(w1.dtype),
                        "w2_dtype": str(w2.dtype),
                    }
                    sys.stderr.write(
                        f"[moe_capture_fp8bs] stored constant weights slot (cls={type(self).__qualname__}); "
                        f"quant_config picklable={ok_qc}; w1{list(w1.shape)}{w1.dtype} w2{list(w2.shape)}{w2.dtype}\n")
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
                    f"[moe_capture_fp8bs] recorded case {len(_S['records'])} ({regime}) {sig}\n")
        if added:
            _flush()
    except Exception as e:
        sys.stderr.write(f"[moe_capture_fp8bs] capture error (ignored): {e!r}\n")
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
        "note": "fp8-blockscale MoE oracle captured from baseline TritonExperts.apply. IMMUTABLE.",
    }
    with open(os.path.join(out_dir, f"capture_meta.json.{pid}"), "w") as fh:
        json.dump(meta, fh, indent=2)
    sys.stderr.write(f"[moe_capture_fp8bs] flushed {len(records)} case(s) "
                     f"(regimes={sorted(_S['regime_seen'])}) -> {out_dir}\n")


def install(target=None, out=None, max=None):
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
    sys.stderr.write(f"[moe_capture_fp8bs] hooked {_TARGET}; up to {_S['max_cases']} cases -> {_S['out_dir']}\n")


if _S["out_dir"]:
    try:
        install()
    except Exception as e:
        sys.stderr.write(f"[moe_capture_fp8bs] install failed: {e!r}\n")
