---
title: DeepSeek V3 / V3.2 / V4 attention on MI300X — bring-up & sparse-MLA case study
kind: case_study
updated: 2026-06-09
note: migrated from perf_knowledge v1; cross-refs repointed to perf_knowledge operator dirs
---

# DeepSeek-V3 / V3.2 / V4 Attention at Scale on AMD MI300X (CDNA3 / gfx942)

> Scope: the **DeepSeek family attention stack** as it runs on MI300X — V3's **MLA + MoE** and its serving kernels, the **V3.2/V4 sparse attention (DSA / NSA-style + lightning indexer)**, the concrete MI300X optimizations in **sglang / vllm / aiter**, the **FP8 (`fnuz`)** paths, the relevant **PRs / bring-up bugs**, and a ranked **"what to optimize for DeepSeek on MI300X"** plan. Deep mechanics live in the focused docs: MLA math/absorption → [[operators/mla_attention/overview.md]]; sparse selection/indexer → [[operators/sparse_attention_nsa/overview.md]]; paged decode/split-KV → [[operators/attention_decode_paged/overview.md]]; prefill FMHA → [[operators/attention_prefill_fmha/overview.md]].
>
> **AMD-only.** Targets MI300X (304 CU, 5.3 TB/s HBM3, 192 GB, fp8 `fnuz`).

---

## 0. TL;DR — the DeepSeek-on-MI300X picture

| Model | Attention | KV cache | Serving on MI300X |
|---|---|---|---|
| **V3 / R1** (Dec'24–Jan'25) | **MLA** (latent 576-d) + MoE | ~57× smaller than MHA | aiter MLA decode + MHA prefill + fused MoE; **fp8 W8A8**; sglang/vllm default |
| **V3.2-Exp / V3.2** (Sep–Dec'25) | MLA + **DSA** (lightning indexer + top-k) | latent + **separate indexer K-cache** | sparse MLA kernels (FlashMLA); fp8 indexer; vllm/sglang day-0 |
| **V4 / V4-Flash** (2025-26) | sparse attention: learned indexer top-k + sliding window, FP8 caches | compressed + sparse | **rough on gfx942** — several aiter sparse-MLA paths missing/broken → Triton fallback |

**MI300X's structural advantage for DeepSeek:** 192 GB HBM3 (vs H100 80 GB) + comparable FP8 compute (2615 TFLOP/s) at ~half the list price. MLA already shrinks KV ~57×; on 192 GB that means very long context / big batch fit on **one** GPU. The catch: **aiter's tuned coverage favors CDNA4 (MI350) over CDNA3 (gfx942)**, so the newest sparse paths need fallbacks.

---

## 1. V3/R1 attention: MLA + the serving kernels

DeepSeek-V3 attention is **MLA** (see [[operators/mla_attention/overview.md]] for the math): down-project hidden → 576-d latent (`kv_lora_rank=512` + decoupled-RoPE `64`), cache only that, 128 query heads. Two kernels:

| Phase | Kernel on MI300X | Form |
|---|---|---|
| Prefill | aiter MHA / `ck_tile` FMHA (hdim q=192, v=128) over up-projected K/V | MHA |
| Decode | **aiter `mla_decode_fwd`** (absorbed) | MQA over the 576-d latent |

AMD's AITER integration into sglang for R1 reports the kernel-level wins: **MLA decode up to ~17×, MHA prefill up to ~14×, block-scale GEMM up to ~2×, block-scale fused MoE up to ~3×** — and ~2× end-to-end MLA layer inference. The decode path is the absorbed **MQA** form (one latent head shared by all 128 query heads), which is memory-bound and lives or dies on **`num_kv_splits`** filling the 304 CUs ([[operators/attention_decode_paged/overview.md]] §4, [[operators/mla_attention/overview.md]] §7).

**FP8 W8A8** is the production precision: weights fp8, activations fp8, latent/KV optionally fp8 — all in **`e4m3fnuz`** on CDNA3 (see §4). Projections go through `aiter.tuned_gemm`; MoE through aiter block-scale fused MoE.

### Disaggregated prefill/decode
MI300X serving for DeepSeek commonly **disaggregates prefill and decode** (separate workers): prefill is compute-bound MHA (wants big tiles, full CUs), decode is bandwidth-bound MQA (wants split-KV + HIP-graph). Splitting them lets each side use its ideal kernel config and batching — a key sglang-on-MI300X pattern.

---

## 2. V3.2 / V4 attention: DSA (sparse) + lightning indexer

V3.2 adds **DeepSeek Sparse Attention (DSA)** on top of MLA — the *only* architectural change from V3.1. V4/V4-Flash extends the same idea. Two pieces (full detail in [[operators/sparse_attention_nsa/overview.md]] §5):

1. **Lightning indexer** — a small, few-head, **FP8 ReLU** scorer that ranks preceding tokens for the current query. It maintains its **own K-cache** (the "indexer K cache"), *separate* from the MLA latent cache → on MI300X you allocate two cache pools.
2. **Top-k token selection** — attention runs only over the top-k (k=2048 in V3.2 training); drops core attention from `O(L²)` to `O(L·k)`. Because DSA sits on MLA's **MQA mode**, each selected latent entry is shared across all heads (max reuse).

V4-Flash specifically: "each query attends to a top-k subset of the KV cache picked by a learned indexer, with sliding-window context handled separately," with **FP8 caches feeding the compressor, indexer, and sliding-window paths**. For short prefill, DeepSeek uses a **masked-MHA mode** that simulates DSA (cheaper at small L).

### Kernel surface
- **sparse MLA prefill** + **sparse MLA decode** (FlashMLA sparse kernels upstream; ~640 TFLOP/s prefill / ~410 TFLOP/s decode on H800 — AMD equivalents via aiter where present).
- **paged MQA logits** (the indexer/selection logits over the paged latent).
- **lightning-indexer GEMM** (DeepGEMM upstream; on ROCm a small fp8 GEMM + top-k).

---

## 3. MI300X / gfx942 bring-up: the real bugs and fixes (V4-Flash)

This is the part you must know to write DeepSeek attention on MI300X. From the public V4-Flash MI300X bring-up (vllm-amd fork) — **aiter coverage targets CDNA4, so gfx942 has gaps**:

| Issue | gfx942 status | Fix / fallback | PR |
|---|---|---|---|
| **paged MQA logits** | aiter path **missing** | ROCm helper → aiter where available, else **Triton** | `cb8a18556` |
| **sparse MLA prefill** | aiter path **missing** | Triton fallback | `cb8a18556` |
| **sparse MLA decode** | aiter path **missing** | Triton fallback | `cb8a18556` |
| AITER **prefill MQA logits** | exists but **broken on gfx942** | refuse dispatch when platform reports gfx942 → Triton | `cb8a18556` |
| AITER **sparse prefill logits** | **broken on gfx942** | same guard → Triton | `cb8a18556` |
| FP8 dialect mismatch | values **2× off** | force **platform fp8 (`fnuz`)** in compressor + fused compress/quant/cache write | `236de4e64` |
| sliding-window K-cache fp8 | dialect | fnuz-aware fused **quantise-and-insert** | `bd06e5d87` |
| **HIP-graph capture** of sparse MLA decode | dynamic ragged metadata not capture-safe | rebuild metadata as **static, capture-safe tensors** (no host→device scalar writes under capture) | `22cc02230` |

Note the Triton fallback is "several times slower" than aiter — so on gfx942 the *missing kernels are the optimization target*. Throughput after bring-up improved 2485 → 2699 output tok/s/GPU (~+8.6%) with bottlenecks remaining in ragged metadata rebuilds, redundant bf16 projection-weight materialization, scratch→output copies, and single static Triton/MXFP4 tile shapes across mismatched regimes.

(Also seen in this era: `fmha_v3` **MI300 kernel hang at large prefill** ≥ ~20480 tokens because the aiter dispatcher keys on `multiProcessorCount==304` + `gfx942` shared by MI300X/MI325X and picks a broken hsaco — pin a known-good aiter.)

---

## 4. FP8 on CDNA3 — the `fnuz` rule for DeepSeek

DeepSeek is an FP8-native model family, and CDNA3 FP8 has a trap:

- MI300X (CDNA3) supports only **`fnuz`** fp8 (`e4m3fnuz`/`e5m2fnuz`): finite, NaN, unsigned-zero, **no `-0`/`inf`**.
- It is **not** OCP fp8 (MI325/MI350/MI355X). The bit layout matches OCP `e4m3`/`e5m2` but the **exponent bias differs by 1** → reading the wrong dialect gives values **exactly 2× off**.
- Many vLLM FP8 paths distinguish `e4m3`/`e5m2` but **not** `fnuz` vs OCP → the V4 bring-up had to force the platform fp8 dtype in the compressor, the fused compress/quant/cache writes, and the sliding-window K-cache (PRs `236de4e64`, `bd06e5d87`).
- Use `torch.float8_e4m3fnuz` everywhere on MI300X: weights (W8A8), latent KV, indexer K-cache, sliding-window K-cache. **block-scale** quant for outlier-heavy activations.

---

## 5. The key kernel logic (DeepSeek decode, MI300X)

DeepSeek decode = **absorbed MLA (MQA) + DSA selection**. Skeleton (combines [[operators/mla_attention/overview.md]] §4 absorption, [[operators/sparse_attention_nsa/overview.md]] §2.2 gathered-block flash, [[operators/attention_decode_paged/overview.md]] split-KV):

```python
# ---- per decode step, per sequence ----
# 1. project + absorb  (build MQA query over the 576-d latent)
q_nope, q_pe, c_KV_new, k_pe_new = mla_project(h_t)        # down/up + RoPE
q_nope = bmm(q_nope, W_UK_absorb)                          # absorb W_UK -> latent space
q_input = concat(q_nope, q_pe)                            # [num_heads, 576]
append_to_latent_cache(c_KV_new, k_pe_new)               # fp8 e4m3fnuz

# 2. DSA: lightning indexer picks top-k blocks (fp8 ReLU scorer, own K-cache)
idx_scores = relu(q_idx @ indexer_K_cache.T)             # fp8 GEMM (small)
sel_blocks = topk(idx_scores, k).indices                  # block ids (the gather)
sel_blocks = sort(sel_blocks)                             # cache locality + causal

# 3. sparse MLA decode = split-KV flash over the SELECTED latent blocks (MQA)
for split in kv_splits(sel_blocks):                        # fill 304 CUs
    for blk in split:                                      # gathered active blocks only
        c = gather_latent(latent_cache, blk)              # [page, 576] coalesced buffer_load
        s = sm_scale * (q_input @ c.T)                    # MQA GEMV over latent
        online_softmax(s, c[:, :512])                     # accumulate over 512-d value part
    write_partial(acc, lse)
out_latent = lse_merge(partials)                          # [num_heads, 512]

# 4. absorb W_UV, output proj
out = bmm(out_latent, W_UV_absorb); out = o_proj(out)
```

Every line maps to a focused doc: absorption ([[operators/mla_attention/overview.md]]), indexer+top-k+gather ([[operators/sparse_attention_nsa/overview.md]]), split-KV+LSE merge ([[operators/attention_decode_paged/overview.md]]). The MI300X-specific bits: **fp8 `fnuz`** caches, **sorted gathered block ids**, **`num_kv_splits`** to fill CUs, **capture-safe static metadata** for HIP-graph.

---

## 6. Ranked plan — "what to optimize for DeepSeek on MI300X"

Ordered by typical impact-per-effort on gfx942:

1. **Use aiter MLA decode (`mla_decode_fwd`) + tune `num_kv_splits`.** The single biggest decode lever; absorbed MQA over one latent head under-fills 304 CUs without enough splits. (up to ~17× kernel, ~2× e2e). → [[operators/mla_attention/overview.md]]
2. **FP8 `e4m3fnuz` everywhere** (W8A8 + latent + indexer + window caches), block-scale quant. Halves KV bytes (the decode bottleneck) and feeds the 2615 TFLOP/s fp8 cores. Guard against the **2× dialect bug**. → §4
3. **Fill the gfx942 sparse gaps.** Write/port the **missing** kernels (paged MQA logits, sparse MLA prefill/decode) so you stop falling back to "several× slower" Triton; guard the **broken** aiter paths on gfx942. → §3, [[operators/sparse_attention_nsa/overview.md]]
4. **HIP-graph everything in decode** with **capture-safe static metadata** (no dynamic ragged allocs / host→device scalar writes under capture). Decode is one tiny kernel per token; launch overhead and capture failures dominate otherwise. → §3
5. **Disaggregate prefill/decode** so each uses its ideal config (MHA big-tile prefill vs split-KV MQA decode). → §1
6. **Prefill MLA via `ck_tile` hdim%32** (q=192, v=128) MHA; avoid the `fmha_v3` large-prefill hang. → [[operators/attention_prefill_fmha/overview.md]]
7. **DSA selection hygiene:** sort selected block ids (cache locality + causal), GQA-pack so one selected latent block feeds all heads, split-KV over the selected set. → [[operators/sparse_attention_nsa/overview.md]]
8. **MoE side** (not attention, but co-bottleneck): aiter block-scale fused MoE (up to ~3×), block-scale GEMM (up to ~2×); watch the MXFP4 emulation/expert-mask routing bugs.
9. **Reduce host overhead:** kill redundant bf16 projection-weight materialization, scratch→output copies, ragged metadata rebuilds (the post-bring-up bottlenecks).
10. **Track aiter releases** — sparse/MLA gfx942 coverage is actively landing; re-benchmark Triton-fallback paths against new aiter each release.

---

## 7. Checklist

1. Decode = **absorbed MLA MQA** + DSA top-k; tune **`num_kv_splits`** to fill 304 CUs.
2. **fp8 `e4m3fnuz`** for all weights/caches; block-scale; verify no 2× dialect error.
3. **Two cache pools**: MLA latent (576-d) + separate **indexer K-cache**.
4. **Lightning indexer** in fp8 ReLU; **sort** selected block ids.
5. **Sparse flash** = gathered-block split-KV over the latent (MQA GEMV + LSE merge).
6. **HIP-graph** with **static capture-safe** metadata.
7. **gfx942 guards + Triton fallbacks** for missing/broken aiter sparse paths; fill those kernels as the top optimization target.
8. **Prefill** via ck_tile MHA (hdim%32, q=192/v=128); avoid `fmha_v3` large-prefill hang.
9. **Disaggregate** prefill/decode; pair with aiter fused MoE.
10. Re-benchmark against each new aiter release.

---

## Sources

- DeepSeek-V3 Technical Report (MLA + MoE, fp8) — https://arxiv.org/abs/2412.19437
- DeepSeek-V3.2 Technical Report (DSA, lightning indexer, arXiv:2512.02556) — https://arxiv.org/abs/2512.02556
- Native Sparse Attention (DeepSeek, arXiv:2502.11089) — https://arxiv.org/abs/2502.11089
- Bringing up DeepSeek-V4-Flash on AMD MI300X (gfx942 gaps, fp8 fnuz, capture-safe metadata, PRs) — https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
- Accelerate DeepSeek-R1 Inference: Integrate AITER into SGLang (kernel-level speedups) — https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
- Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X (MLA + fp8 serving) — https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html
- AITER-Enabled MLA Layer Inference on AMD Instinct MI300X (`mla_decode_fwd`, absorption) — https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention (separate indexer cache, FlashMLA/DeepGEMM) — https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html
- Unleashing MI300X for LLM Serving: Disaggregating Prefill & Decode with SGLang (ROCm Blog) — https://rocm.blogs.amd.com/software-tools-optimization/disaggregation/README.html
- ROCm/aiter (MLA, paged-attn, fused MoE; gfx942 dispatch) — https://github.com/ROCm/aiter
