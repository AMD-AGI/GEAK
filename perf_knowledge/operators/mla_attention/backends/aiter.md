---
title: mla_attention on aiter — SOTA card
kind: sota_card
operator: mla_attention
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: sota
updated: 2026-06-08
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
  - https://vllm.ai/blog/2026-02-27-rocm-attention-backend
---

# mla_attention × aiter

## TL;DR (one-line decision)
> aiter is **the** MLA backend on AMD: `mla_decode_fwd` (hand-tuned asm, **matrix-absorbed MQA on the
> latent**) is the headline kernel — **up to 17× vs naive decode** (vendor, MI300X) — and
> `mla_prefill_fwd` / `mla_prefill_ps_fwd` (persistent) cover prefill. AITER MLA vs Triton-MLA TPS:
> **MI300X 1.33×, MI325X 1.41×, MI355X 1.52×** (vLLM ROCm, Feb 2026); TPOT **1.2–1.6×**. Gen nuance:
> on **gfx942** the Triton-MLA variant edges **+2–3% TPS**; on **gfx950** `ROCM_AITER_MLA` wins (uses
> AITER asm MHA prefill, best TTFT) and is the auto-selected, recommended-for-all-workloads default.
> This is the default; everything else is a fallback or reference.

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes / shapes | measured perf (`value @ hw, date`) | when it's best |
|---|---|---|---|---|
| `mla_decode_fwd` (asm, absorbed MQA) | `ROCm/aiter@a6bb49937:aiter/mla.py:156` | gfx942/950; bf16/fp16/fp8; nhead 16/64/128; **kv_lora_rank 512 + rope 64**; auto `num_kv_splits` | **17× vs naive decode** @ MI300X, 2025-03 (vendor); **AITER MLA vs Triton-MLA TPS: MI300X 1.33×, MI325X 1.41×, MI355X 1.52×; TPOT 1.2–1.6×** @ vLLM ROCm blog, Feb 2026 (vendor) | DeepSeek decode |
| `mla_prefill_fwd` | `aiter/mla.py:536` | long sq; `num_kv_splits=1` | prefill MLA | DeepSeek prefill |
| `mla_prefill_ps_fwd` (persistent, `tile_q=256`) | `aiter/mla.py:583` | many short prefills | persistent scheduler amortizes launches | batched short prefills |

**Real signature + auto-split + fp8 scales + persistent gate** (`aiter/mla.py`):
```python
def mla_decode_fwd(q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
    max_seqlen_q, page_size=1, nhead_kv=1, sm_scale=None, logit_cap=0.0,
    num_kv_splits=None, num_kv_splits_indptr=None,        # "for experts only!!!"
    work_meta_data=None, ...,                              # persistent mode
    q_scale=None, kv_scale=None, ...):
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    if sm_scale is None: sm_scale = 1.0 / (qk_head_dim**0.5)
    persistent_mode = work_meta_data is not None
    if not persistent_mode:
        if num_kv_splits is None or num_kv_splits_indptr is None:
            num_kv_splits, num_kv_splits_indptr = get_meta_param(num_kv_splits, bs, total_kv,
                nhead, max_seqlen_q, q.dtype)        # CU-aware split count, fp8 block_n table
    aiter.mla_decode_stage1_asm_fwd(q, kv_buffer, qo_indptr, kv_indptr, kv_indices, ...,
        q_scale, kv_scale)                           # asm stage-1, then stage-2 reduce
```
`get_meta_param` picks `num_kv_splits` from a CU-occupancy score over `range(1,17)`; for fp8 it clamps to
a per-head `min_block_n` table (`{16:128, 32:128, 48:64, 64:64, 128:32, 256:32, 384:32, 512:32}`).
The `kv_buffer` is the absorbed latent cache `[num_page, page_size, nhead_kv, kv_lora_rank + qk_rope]`.

## Config space / knobs
| param | range / values | effect | default |
|---|---|---|---|
| `num_kv_splits` | auto (`get_meta_param`, 1–16) — "experts only" | flash-decoding parallelism | None (auto) |
| `page_size` | 1 (fast unpaged) / >1 | KV paging | 1 |
| `nhead_kv` | 1 (decode contract) | latent is MQA → 1 kv head | 1 |
| persistent mode | `work_meta_data`… / `SGLANG_AITER_MLA_PERSIST=1` | amortize many short seqs | off |
| `q_scale` / `kv_scale` | fp32 tensors | fp8 (de)scale | None |
| `sm_scale` | float | QK scale | `1/sqrt(qk_head_dim)` |
| `logit_cap` | **must be ≤0** (`assert`) | NOT yet supported | 0 |
| `mgc` | 16 / 32 / 64 (auto by nhead/dtype) | inner tiling tuned for nhead+dtype | auto |

Framework: sglang `SGLANG_ROCM_FUSED_DECODE_MLA=1`, `SGLANG_AITER_MLA_PERSIST=1`; vLLM
`VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_MLA=1`. **Gen ranking**: gfx942 favors
`ROCM_AITER_TRITON_MLA` (+2–3% TPS); gfx950 favors `ROCM_AITER_MLA` (asm prefill). See
[../tuning.md](../tuning.md).

## Numerics / parity
Matrix absorption is algebraically exact → bf16 parity-safe. fp8 latent/KV is accuracy-sensitive — AITER
MLA has shown eval regressions (gsm8k loss, aiter #1455) → task-accuracy gate. **aiter MLA decode does
not support fp8 KV-cache in vLLM upstream** (vendor caveat). fp8 dialect: FNUZ gfx942 / OCP gfx950.
`logit_cap` unsupported (`assert logit_cap <= 0`). See [../numerics.md](../numerics.md).

## Integration (rebind seam)
- **sglang:** `--attention-backend aiter` → `from aiter.mla import mla_decode_fwd, mla_prefill_fwd` (the
  literal dispatch surface in `aiter_backend.py`).
- **vLLM:** `--attention-backend ROCM_AITER_MLA` (+ master `VLLM_ROCM_USE_AITER=1`).
- **Verify it engaged:** `AITER_LOG_MORE=1` shows `mla_decode_stage1_asm_fwd`; rocprofv3 shows the asm MLA
  kernel (not Triton `_fwd_kernel_*` / `_fwd_grouped_kernel_*`).

## Pitfalls & anti-patterns
- gfx942 may fall back to Triton for the newest MLA variants (sparse MLA) → several × slower; confirm with
  `AITER_LOG_MORE=1`.
- `nhead_kv` must be 1 and `page_size==1` to hit the fast unpaged decode path.
- Don't hand-set `num_kv_splits` — it's auto and shape/CU-aware ("experts only").
- `logit_cap > 0` will `assert` — MLA softcap not yet supported here.
- fp8 MLA accuracy regressions — gate; and remember fp8 KV is unsupported in vLLM upstream MLA.

## Worked example
DeepSeek-V3 decode on MI300X, bf16, ctx 8K, batch 64:
1. `--attention-backend ROCM_AITER_MLA` (vLLM) + `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=1`.
2. `kv_buffer` is the absorbed latent `[…, 512+64]`; `nhead_kv=1`, `page_size=1`.
3. `get_meta_param` picks `num_kv_splits` (CU-aware) — leave `None`.
4. `AITER_LOG_MORE=1` → `mla_decode_stage1_asm_fwd`; bench TPOT vs `ROCM_AITER_TRITON_MLA` (on gfx942 the
   Triton variant may edge +2–3%); greedy temp=0 parity vs Triton MLA over ≥10 prompts.

## How to verify (bench + oracle)
`AITER_LOG_MORE=1` confirms asm MLA (not Triton); AI-Developer-Hub `aiter_mla_decode_kernel` notebook for
a runnable example; isolated decode bench + e2e TPOT + gsm8k accuracy gate (especially fp8).

## Alternatives / cross-links
[[./triton.md]] (reference / gfx942 contender) · [[./ck.md]] · [[../../attention_decode_paged/backends/aiter.md]] ·
[[../../attention_prefill_fmha/backends/aiter.md]] · `backends/aiter/attn_mla.md` ·
`backends/sglang_kernels/attention_backends.md` · [[../overview.md]].

## Sources
- On-box `ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/mla.py` (`mla_decode_fwd`:156, `mla_prefill_fwd`:536, `mla_prefill_ps_fwd`:583 tile_q=256, `get_meta_param`:109 auto num_kv_splits + fp8 block_n table, persistent mode, `q_scale`/`kv_scale`, `assert logit_cap<=0`).
- 17× decode (vendor, MI300X, 2025-03): https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- AITER MLA vs Triton-MLA TPS MI300X 1.33× / MI325X 1.41× / MI355X 1.52×, TPOT 1.2–1.6×; gfx942 Triton-MLA +2–3% TPS, gfx950 ROCM_AITER_MLA wins (best TTFT); fp8 KV unsupported in vLLM upstream (vendor, Feb 2026): https://vllm.ai/blog/2026-02-27-rocm-attention-backend
