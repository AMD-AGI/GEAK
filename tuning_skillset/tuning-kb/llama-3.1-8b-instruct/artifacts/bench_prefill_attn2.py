#!/usr/bin/env python3
"""Race prefill attention at the shape the *sealed benchmark* actually produces.

analysis/bench_prefill_attn.py raced the shape my probe driver produces -- 2 whole 8192-token
requests, no prefix -- and ragged beat paged by 34.71%. The sealed benchmark never produces that
shape. Its prompts are 8192 tokens, the server's tokenizer prepends BOS, and 2*8193 > the
16384-token chunked-prefill budget, so from the second batch onward every prefill batch is
three sequences: the 2-token tail of the previous request, one whole 8193-token request, and the
~8189-token head of the next. The tail carries a prefix, so a batch-level "no prefix anywhere"
gate never fires -- which is exactly why the end-to-end A/B moved 0.0%.

So: can the fast ragged kernel serve a batch that *has* prefixes? Two ways to feed it the prefix
KV, which lives in the paged pool rather than in the k/v the layer just produced:

  paged        mha_batch_prefill_func + page table            <- production today
  gather       index_select the full KV out of the pool into one contiguous buffer, then
               flash_attn_varlen_func with cu_seqlens_k != cu_seqlens_q (causal is bottom-right
               aligned, which is the correct chunked-prefill mask)
  blocktable   flash_attn_varlen_func reading the pool directly through its own block_table
  paged_self   byte-identical copy of `paged`, registered last, as the arm-position control

The gather is not free -- it moves ~100 MB per layer -- and that cost is inside the timed region,
where it belongs.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

sys.path.insert(0, "/sgl-workspace/aiter")
from aiter import flash_attn_varlen_func, mha_batch_prefill_func  # noqa: E402

import gbench  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--qlens", default="2,8193,8189")
AP.add_argument("--kvlens", default="8193,8193,8189")
AP.add_argument("--pool-tokens", type=int, default=1_236_694)
AP.add_argument("--qh", type=int, default=32)
AP.add_argument("--kvh", type=int, default=8)
AP.add_argument("--d", type=int, default=128)
AP.add_argument("--rounds", type=int, default=5)
AP.add_argument("--reps", type=int, default=4)
AP.add_argument("--iters", type=int, default=10)
AP.add_argument("--out", default="analysis/prefill_attn2.json")
A = AP.parse_args()

dev, dt = "cuda", torch.bfloat16
D = A.d
qlens = [int(x) for x in A.qlens.split(",")]
kvlens = [int(x) for x in A.kvlens.split(",")]
assert len(qlens) == len(kvlens) and all(q <= k for q, k in zip(qlens, kvlens))
B = len(qlens)
NQ, NKV = sum(qlens), sum(kvlens)
scale = 1.0 / (D**0.5)
print(f"[shape] {B} seqs  q={qlens} (total {NQ})  kv={kvlens} (total {NKV})")

torch.manual_seed(0)
q = torch.randn(NQ, A.qh, D, device=dev, dtype=dt) * 0.1
kv_flat_k = torch.randn(NKV, A.kvh, D, device=dev, dtype=dt) * 0.1
kv_flat_v = torch.randn(NKV, A.kvh, D, device=dev, dtype=dt) * 0.1

# Paged pool at the production size, each sequence's full KV one contiguous run -- what SGLang's
# allocator produces here (round 1 measured the production layout as ~contiguous).
pool = A.pool_tokens
k_cache = torch.zeros(pool, 1, A.kvh, D, device=dev, dtype=dt)
v_cache = torch.zeros(pool, 1, A.kvh, D, device=dev, dtype=dt)
starts, s = [], 1024
for L in kvlens:
    starts.append(s)
    s += L
page_idx = torch.cat(
    [
        torch.arange(st, st + L, device=dev, dtype=torch.int32)
        for st, L in zip(starts, kvlens)
    ]
)
k_cache.view(pool, A.kvh, D)[page_idx.long()] = kv_flat_k
v_cache.view(pool, A.kvh, D)[page_idx.long()] = kv_flat_v


def _indptr(lens):
    out = [0]
    for L in lens:
        out.append(out[-1] + L)
    return torch.tensor(out, device=dev, dtype=torch.int32)


qo_indptr = _indptr(qlens)
kv_indptr = _indptr(kvlens)
max_q, max_kv = max(qlens), max(kvlens)

# dense block table for the block_table arm (page_block_size == 1, so one entry per token)
block_table = torch.zeros(B, max_kv, device=dev, dtype=torch.int32)
for i, (st, L) in enumerate(zip(starts, kvlens)):
    block_table[i, :L] = torch.arange(st, st + L, device=dev, dtype=torch.int32)

out_a = torch.empty(NQ, A.qh, D, device=dev, dtype=dt)
out_b = torch.empty(NQ, A.qh, D, device=dev, dtype=dt)
out_c = torch.empty(NQ, A.qh, D, device=dev, dtype=dt)
out_d = torch.empty(NQ, A.qh, D, device=dev, dtype=dt)
kg = torch.empty(NKV, A.kvh, D, device=dev, dtype=dt)
vg = torch.empty(NKV, A.kvh, D, device=dev, dtype=dt)
pidx_l = page_idx.long()


def paged(out=out_a):
    return mha_batch_prefill_func(
        q, k_cache, v_cache, qo_indptr, kv_indptr, page_idx, max_q, max_kv,
        causal=True, logits_soft_cap=0.0, alibi_slopes=None, return_lse=False,
        return_attn_probs=False, window_size=(-1, -1), out=out,
    )


def paged_self():
    return paged(out_d)


def gather(out=out_b):
    torch.index_select(k_cache.view(pool, A.kvh, D), 0, pidx_l, out=kg)
    torch.index_select(v_cache.view(pool, A.kvh, D), 0, pidx_l, out=vg)
    return flash_attn_varlen_func(
        q, kg, vg, qo_indptr, kv_indptr, max_q, max_kv,
        softmax_scale=scale, causal=True, out=out,
    )


def blocktable(out=out_c):
    return flash_attn_varlen_func(
        q, k_cache, v_cache, qo_indptr, kv_indptr, max_q, max_kv,
        softmax_scale=scale, causal=True, block_table=block_table, out=out,
    )


cands = {"paged": paged, "gather": gather, "blocktable": blocktable, "paged_self": paged_self}

# --- correctness against the production arm ----------------------------------------------
paged()
torch.cuda.synchronize()
ref = out_a.float()
checks = {}
for name, fn in (("gather", gather), ("blocktable", blocktable)):
    try:
        fn()
        torch.cuda.synchronize()
        got = (out_b if name == "gather" else out_c).float()
        rel = ((ref - got).abs().max() / ref.abs().max()).item()
        cos = torch.nn.functional.cosine_similarity(
            ref.flatten(), got.flatten(), dim=0
        ).item()
        checks[name] = {"max_rel_err": rel, "cos": cos}
        print(f"[check] {name:11s} max_rel_err={rel:.3e}  cos={cos:.8f}")
    except Exception as e:  # noqa: BLE001
        checks[name] = {"error": repr(e)[:200]}
        print(f"[check] {name:11s} FAILED: {repr(e)[:200]}")
        cands.pop(name, None)

res = gbench.race(cands, rounds=A.rounds, reps=A.reps, warmup=3, iters=A.iters)
base = res["paged"]["us"]
for n, r in res.items():
    if "us" in r:
        r["vs_paged_pct"] = 100 * (base - r["us"]) / base
        print(
            f"{n:12s} {r['us']:9.2f} us  spread {r['spread_pct']:.2f}%  "
            f"vs paged {r['vs_paged_pct']:+.2f}%"
        )
    else:
        print(f"{n:12s} {r['mode']}")

json.dump(
    {"shape": vars(A), "qlens": qlens, "kvlens": kvlens, "checks": checks, "result": res},
    open(A.out, "w"),
    indent=2,
)
print("->", A.out)
