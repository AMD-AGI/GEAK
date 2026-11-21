"""Real launcher + benchmark for the raw @triton.jit kernel
`_per_token_group_quant_fp8` (vllm fp8_utils).

Kernel role: per-token-group FP8 quantization. `y` has shape
[num_tokens, hidden]; each row is split into hidden//group_size groups of
`group_size` columns. One Triton program quantizes one group, producing a
fp8 output tile (`y_q`) and a per-group float32 scale (`y_s`).
grid = (num_tokens * (hidden // group_size),)  -- one program per group.

WORKLOAD REGIME (token-parallel quant op):
  num_tokens M = B * 1024 for B in {2, 32, 64}.
  Model dims kept from captured base case: hidden=4096, group_size=128,
  BLOCK=128, fp8 dtype=float8_e4m3fnuz, use_ue8m0=False, eps=1e-10.
"""
import os
import json
import importlib.util

import torch

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_EDIT = os.path.join(TASK_DIR, "source", "triton__per_token_group_quant_fp8.py")
SRC_GOLD = os.path.join(TASK_DIR, "source_golden", "triton__per_token_group_quant_fp8.py")
KERNEL_NAME = "_per_token_group_quant_fp8"

# ---- regime ----------------------------------------------------------------
HIDDEN = 4096
GROUP_SIZE = 128
BLOCK = 128
FP8_MAX = 224.0
FP8_MIN = -224.0
EPS = 1e-10
USE_UE8M0 = False
SEQLEN = 1024
FP8_DTYPE = getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn)

CONCURRENCY = {"c2": 2, "c32": 32, "c64": 64}


def regime_cases():
    """Return list of (id, num_tokens) for B in {2,32,64}."""
    return [(cid, B * SEQLEN) for cid, B in CONCURRENCY.items()]


def _load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, KERNEL_NAME)


def build_inputs(num_tokens, seed=42):
    """Deterministic seeded inputs for one regime case."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    y = torch.randn((num_tokens, HIDDEN), dtype=torch.bfloat16,
                    device="cuda", generator=g)
    groups_per_row = HIDDEN // GROUP_SIZE
    y_q = torch.empty((num_tokens, HIDDEN), dtype=FP8_DTYPE, device="cuda")
    y_s = torch.empty((num_tokens, groups_per_row), dtype=torch.float32,
                      device="cuda")
    return y, y_q, y_s


def grid_for(num_tokens):
    groups_per_row = HIDDEN // GROUP_SIZE
    return (num_tokens * groups_per_row,)


def launch(kern, y, y_q, y_s):
    """Launch the jit kernel. y_q and y_s are mutated in place."""
    num_tokens = y.shape[0]
    grid = grid_for(num_tokens)
    kern[grid](
        y,
        y_q,
        y_s,
        GROUP_SIZE,
        HIDDEN,            # y_num_columns
        y.stride(0),       # y_row_stride
        EPS,
        fp8_min=FP8_MIN,
        fp8_max=FP8_MAX,
        use_ue8m0=USE_UE8M0,
        BLOCK=BLOCK,
        num_warps=1,
        num_stages=1,
    )


def _run_one(kern, num_tokens, seed=42):
    y, y_q, y_s = build_inputs(num_tokens, seed=seed)
    launch(kern, y, y_q, y_s)
    torch.cuda.synchronize()
    return y_q, y_s


def correctness():
    kern_e = _load(SRC_EDIT, "kern_edit")
    kern_g = _load(SRC_GOLD, "kern_gold")
    for cid, num_tokens in regime_cases():
        qe, se = _run_one(kern_e, num_tokens)
        qg, sg = _run_one(kern_g, num_tokens)
        # scale: float32, compare directly
        if se.shape != sg.shape:
            return False, f"{cid}: scale shape mismatch {se.shape} vs {sg.shape}"
        sd = (se.float() - sg.float()).abs().max().item()
        if sd > 1e-5:
            return False, f"{cid}: scale max-abs-diff {sd}"
        # fp8 output: compare in float32, cosine + max rel
        ae = qe.float().flatten()
        ag = qg.float().flatten()
        cos = torch.nn.functional.cosine_similarity(
            ae.unsqueeze(0), ag.unsqueeze(0)).item()
        denom = ag.abs().clamp_min(1e-6)
        maxrel = ((ae - ag).abs() / denom).max().item()
        if cos < 0.99:
            return False, f"{cid}: cosine {cos} < 0.99"
        if maxrel > 1e-3:
            return False, f"{cid}: max-rel-err {maxrel}"
    return True, None


def performance(n_warmup=10, n_iter=100):
    kern_e = _load(SRC_EDIT, "kern_edit_perf")
    out = []
    for cid, num_tokens in regime_cases():
        y, y_q, y_s = build_inputs(num_tokens)
        for _ in range(n_warmup):
            launch(kern_e, y, y_q, y_s)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
        for j in range(n_iter):
            starts[j].record()
            launch(kern_e, y, y_q, y_s)
            ends[j].record()
        torch.cuda.synchronize()
        avg = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_iter
        out.append({
            "test_case_id": cid,
            "execution_time_ms": avg,
            "params": {"num_tokens": num_tokens, "hidden": HIDDEN,
                       "group_size": GROUP_SIZE, "B": CONCURRENCY[cid],
                       "seqlen": SEQLEN},
        })
    return out


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        ok, err = correctness()
        print("Correctness:", "PASS" if ok else "FAIL", err or "")
    else:
        for c in performance():
            print(f"Performance: {c['execution_time_ms']:.4f} ms ({c['test_case_id']})")
