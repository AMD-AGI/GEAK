#!/usr/bin/env python3
"""Race the prefill attention call at the production extend shape.

Production (from the torch-profiler window of a live server, /tmp/r2prof):
  chunked-prefill-size 16384 and ISL 8192 => every prefill batch is 2 whole requests,
  16384 q tokens, no prefix (radix cache disabled), 32 q heads / 8 kv heads, head_dim 128,
  bf16, causal. SGLang's aiter backend serves that with `mha_batch_prefill_func` reading K/V
  back out of the paged pool through a page table at page_size 1.

Arms:
  paged        mha_batch_prefill_func + page table   <- what the server runs today
  ragged       flash_attn_varlen_func on the contiguous k/v the layer just produced
  ragged_self  a byte-identical copy of `ragged`, registered second, as the
               arm-position-bias control round 1 (its FINDINGS.md, conclusion 5) says every
               new harness needs.

The KV pool is allocated at the production size and each sequence's 8192 tokens are laid
out as one contiguous run, which is what SGLang's allocator actually produces here.
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
AP.add_argument("--seqs", type=int, default=2)
AP.add_argument("--slen", type=int, default=8192)
AP.add_argument("--pool-tokens", type=int, default=1_236_694)
AP.add_argument("--qh", type=int, default=32)
AP.add_argument("--kvh", type=int, default=8)
AP.add_argument("--d", type=int, default=128)
AP.add_argument("--rounds", type=int, default=5)
AP.add_argument("--reps", type=int, default=4)
AP.add_argument("--iters", type=int, default=10)
AP.add_argument("--out", default="analysis/prefill_attn.json")
A = AP.parse_args()

dev = "cuda"
dt = torch.bfloat16
B, S, D = A.seqs, A.slen, A.d
NQ = B * S
scale = 1.0 / (D**0.5)

torch.manual_seed(0)
q = torch.randn(NQ, A.qh, D, device=dev, dtype=dt) * 0.1
k_rag = torch.randn(NQ, A.kvh, D, device=dev, dtype=dt) * 0.1
v_rag = torch.randn(NQ, A.kvh, D, device=dev, dtype=dt) * 0.1

# Paged pool at the production size, with each sequence's prefill run contiguous.
pool = A.pool_tokens
k_cache = torch.zeros(pool, 1, A.kvh, D, device=dev, dtype=dt)
v_cache = torch.zeros(pool, 1, A.kvh, D, device=dev, dtype=dt)
# start the two runs where the allocator would: back to back from a low water mark
starts = [1024 + i * S for i in range(B)]
page_idx = torch.cat(
    [torch.arange(s, s + S, device=dev, dtype=torch.int32) for s in starts]
)
k_cache.view(pool, A.kvh, D)[page_idx.long()] = k_rag
v_cache.view(pool, A.kvh, D)[page_idx.long()] = v_rag

qo_indptr = torch.arange(0, (B + 1) * S, S, device=dev, dtype=torch.int32)
kv_indptr = qo_indptr.clone()
kv_last_page = torch.ones(B, device=dev, dtype=torch.int32)

out_paged = torch.empty(NQ, A.qh, D, device=dev, dtype=dt)
out_ragged = torch.empty(NQ, A.qh, D, device=dev, dtype=dt)
out_ragged2 = torch.empty(NQ, A.qh, D, device=dev, dtype=dt)


def paged(out=out_paged):
    return mha_batch_prefill_func(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        page_idx,
        S,
        S,
        causal=True,
        logits_soft_cap=0.0,
        alibi_slopes=None,
        return_lse=False,
        return_attn_probs=False,
        window_size=(-1, -1),
        out=out,
    )


def ragged(out=out_ragged):
    return flash_attn_varlen_func(
        q,
        k_rag,
        v_rag,
        qo_indptr,
        qo_indptr,
        S,
        S,
        softmax_scale=scale,
        causal=True,
        out=out,
    )


def ragged_self():
    return ragged(out_ragged2)


# --- correctness: the two arms must agree ------------------------------------------------
paged()
ragged()
torch.cuda.synchronize()
a = out_paged.float()
b = out_ragged.float()
rel = ((a - b).abs().max() / a.abs().max()).item()
cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
print(f"[check] max_rel_err={rel:.3e}  cos={cos:.8f}")

res = gbench.race(
    {"paged": paged, "ragged": ragged, "ragged_self": ragged_self},
    rounds=A.rounds,
    reps=A.reps,
    warmup=3,
    iters=A.iters,
)
base = res["paged"]["us"]
for n, r in res.items():
    if "us" in r:
        r["vs_paged_pct"] = 100 * (base - r["us"]) / base
        print(
            f"{n:14s} {r['us']:9.2f} us  spread {r['spread_pct']:.2f}%  "
            f"vs paged {r['vs_paged_pct']:+.2f}%"
        )
    else:
        print(f"{n:14s} {r['mode']}")

json.dump(
    {"shape": vars(A), "max_rel_err": rel, "cos": cos, "result": res},
    open(A.out, "w"),
    indent=2,
)
print("->", A.out)
