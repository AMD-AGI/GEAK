#!/usr/bin/env python3
"""Offline (no long run) validator for the vendored FlyDSL int4-W4A16 MoE shim.

Run inside a container that has flydsl + kernels.moe_gemm_2stage importable
(`source $FLYDSL_ROOT/flydsl_env.sh` or the `ensure_flydsl` skill first):

    python3 validate_shim_offline.py

Checks the two vendored files IN THIS DIRECTORY (flydsl_moe_shim.py +
flydsl_capture_precompile.py) at the structure/import/contract level:
  - both byte-compile
  - the shim has NO hardcoded machine / run-dir paths (publish requirement)
  - the memory contract holds: stage-2 accumulate=True, and the old
    accumulate=False + repeat_interleave + host-moe_sum expanded-output path is absent
  - both weight AND scale params are re-homed (no scale-dup KV-OOM)
  - the shim imports (exercises flydsl bindings) and exposes the public API
  - flydsl_fused_experts_impl's signature matches the two vLLM seam edits
  - the precompile file imports and exposes install_capture_precompile

Exit 0 = PASS, 1 = FAIL. This is a deploy-time gate, NOT a numeric/e2e test
(correctness is gated by in-server GSM8K + the same-session A/B).
"""
import importlib
import inspect
import os
import py_compile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
shim_path = os.path.join(HERE, "flydsl_moe_shim.py")
pre_path = os.path.join(HERE, "flydsl_capture_precompile.py")
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    ok = ok and bool(cond)


# 1) byte-compile both files
for p in (shim_path, pre_path):
    try:
        py_compile.compile(p, doraise=True)
        check(f"py_compile {os.path.basename(p)}", True)
    except Exception as e:  # noqa: BLE001
        check(f"py_compile {os.path.basename(p)}", False, repr(e))

# 2) no hardcoded machine / run-dir paths (self-contained publish requirement)
src = open(shim_path).read()
bad = [t for t in ("/nfs", "geak_exp", "/home/", "20260719_162636") if t in src]
check("shim: no hardcoded run-dir/machine paths", not bad, f"found {bad}" if bad else "clean")

# 3) memory contract: accumulate=True present; old expanded-output path absent
check("shim: accumulate=True present", "accumulate=True" in src)
check(
    "shim: old accumulate=False expanded-output path absent",
    not ("accumulate=False" in src
         and "repeat_interleave(top_k" in src
         and ".view(M, top_k, hidden).sum(dim=1)" in src),
)
check("shim: scale re-home (w13 & w2)",
      "w13_weight_scale.data" in src and "w2_weight_scale.data" in src)

# 4) real import (needs flydsl on PYTHONPATH) + public API + signature
sys.path.insert(0, HERE)
try:
    m = importlib.import_module("flydsl_moe_shim")
    check("shim: import OK", True, m.__file__)
except Exception as e:  # noqa: BLE001
    check("shim: import OK", False, repr(e))
    print("RESULT FAIL (import failed -- source flydsl_env.sh / run ensure_flydsl first)")
    sys.exit(1)

for fn in ("convert_layer_inplace", "flydsl_fused_experts_impl"):
    check(f"shim: has {fn}", hasattr(m, fn))

sig = inspect.signature(m.flydsl_fused_experts_impl)
params = list(sig.parameters)
need_positional = ["hidden_states", "w1", "w2", "topk_weights", "topk_ids", "inplace"]
need_kw = ["activation", "apply_router_weight_on_input",
           "global_num_experts", "expert_map", "w1_scale", "w2_scale"]
check("shim: positional params match the vLLM seam", params[:6] == need_positional, str(params[:6]))
check("shim: kw params present", all(k in sig.parameters for k in need_kw),
      "missing: " + ",".join(k for k in need_kw if k not in sig.parameters))
check("shim: absorbs extras via **kwargs",
      any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))

# 5) precompile seam API
try:
    pre = importlib.import_module("flydsl_capture_precompile")
    check("precompile: import OK", True, pre.__file__)
    check("precompile: has install_capture_precompile", hasattr(pre, "install_capture_precompile"))
except Exception as e:  # noqa: BLE001
    check("precompile: import OK", False, repr(e))

print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
