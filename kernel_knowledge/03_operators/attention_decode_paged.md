# Decode / Paged Attention on AMD MI300X (CDNA3 / gfx942)

> Scope: the **decode** regime — one new query token per sequence (`M = 1`), attending to a long, growing KV cache. This is **memory-bandwidth-bound**, dominated by streaming the paged KV cache from HBM. Covers paged KV layout & block tables, KV gather, **split-KV / flash-decoding**, GQA head packing, fp8 KV cache, and the **aiter decode attention** that is the sglang/vllm default on ROCm. Prefill is in `attention_prefill.md`; MLA decode in `mla.md`.
>
> **AMD-only.** Numbers target MI300X (304 CU, 5.3 TB/s HBM3, 64 KiB LDS/CU).

---

## 0. TL;DR

| Situation | First choice | Why |
|---|---|---|
| Production decode, GQA, paged KV (bf16/fp16) | **aiter decode attention** (sglang/vllm ROCm default) | up to ~17× over naive MLA decode; tuned gfx942 paged-attn |
| You must write/modify the kernel | **Triton paged decode** (split-KV / flash-decoding) | readable, autotunable; the sglang triton fallback |
| Low-level control / new KV layout | **CK paged attention** (`ck_tile` splitkv) | source of tuned kernels |
| MLA decode (DeepSeek) | aiter MLA decode → see `mla.md` | MQA-form, latent KV |
| KV cache too big for HBM bandwidth | **fp8 KV cache** + split-KV | halves KV bytes streamed |

**The one number that matters:** decode attention reads ~`2 · N · d_kv · sizeof(dtype)` bytes per head-group per step from HBM and does almost no FLOPs. At 5.3 TB/s, a 32k-token bf16 KV cache (`d=128`) is ~16 MB/head → the kernel is a **bandwidth** problem. Everything below exists to (a) stream KV coalesced, (b) cut KV bytes (fp8, MLA), and (c) **fill all 304 CUs even though M=1** (split-KV).

---

## 1. Why decode is a different kernel from prefill

| | Prefill | Decode |
|---|---|---|
| Q length M | large (full prompt) | **1** (per sequence) |
| Bound by | matrix throughput (MFMA) | **HBM bandwidth** |
| KV access | contiguous, freshly written | **paged, scattered** (block table) |
| Parallelism | `M/BLOCK_M × H × B` workgroups → fills CUs | only `H × B` → **starves 304 CUs** |
| Softmax cost | amortized over big tiles | dominates relative to tiny matmul |
| Key trick | tiling / causal skip | **split-KV (flash-decoding)** + paging |

Because M=1, the QKᵀ is a **matrix-vector** (GEMV-like) per head; MFMA is poorly utilized. The win is not FLOPs — it's reading KV once, coalesced, with enough workgroups to saturate bandwidth.

---

## 2. Paged KV cache layout & block tables

vLLM-style **PagedAttention**: the KV cache is split into fixed-size **pages** (a.k.a. blocks) of `page_size` tokens (commonly 16, also 1, 32, 64, 128). Pages live in two big pools `key_cache`, `value_cache`; each sequence owns a list of physical page ids stored in a **block table**.

```
block_table[seq, logical_block]  ->  physical_page_id
# token t of seq s lives at:
#   page  = block_table[s, t // page_size]
#   slot  = t %  page_size
```

Typical physical layout on ROCm (vLLM/aiter):

```
key_cache:   [num_pages, num_kv_heads, head_dim/x, page_size, x]   # x = 8 or 16 (vectorized inner)
value_cache: [num_pages, num_kv_heads, head_dim, page_size]
```

The `head_dim/x, …, x` split on K makes the inner `x` contiguous so the QKᵀ GEMV reads K with **128-bit `buffer_load`** (coalesced). Value is laid out so the PV step reads V columns contiguously.

**Page size trade-off on MI300X:**

| page_size | Pro | Con |
|---|---|---|
| 1 | minimal padding waste, max sharing | many tiny gathers, poor coalescing |
| 16 | vLLM default, good coalescing | small internal frag |
| 32-64 | best HBM burst efficiency on MI300X | more wasted slots for short seqs |
| 128 | aiter/CK fast path; great bandwidth | frag for many short seqs |

aiter's CK FMHA paged path historically supports page sizes **1, 16, 1024**; aiter's native paged decode supports the common 16/32/64/128. Match `page_size` to the kernel you dispatch to.

---

## 3. KV gather — the paged inner loop

For each KV page in the sequence's block table, gather K/V and run online softmax (same recurrence as prefill, but M=1):

```python
# decode: one query vector q[H, d] for this sequence
m = -inf; l = 0; acc = zeros(d)        # per (head) running state, M=1
for lb in range(num_logical_blocks(seq)):
    pid = block_table[seq, lb]                 # physical page id (the "gather")
    k_pg = key_cache[pid, kv_head]             # [page_size, d]  buffer_load, coalesced
    v_pg = value_cache[pid, kv_head]           # [page_size, d]
    s = sm_scale * (q @ k_pg.T)                # [page_size]   GEMV
    mask invalid slots (last page partial / causal)
    m_new = max(m, max(s))
    p = exp2(s - m_new); alpha = exp2(m - m_new)
    l = l*alpha + sum(p)
    acc = acc*alpha + p @ v_pg                 # [d]
    m = m_new
out = acc / l
```

The **gather** is `block_table[seq, lb] → pid`; the indirection means K/V addresses are scattered across HBM, so coalescing *within* a page (the `x`-vectorized layout) is what preserves bandwidth.

---

## 4. Split-KV / flash-decoding — filling 304 CUs

With M=1 there are only `B·H` workgroups — for a single request with 32 KV heads that's **32 workgroups on a 304-CU GPU** → 90% idle and bandwidth-starved. **Flash-decoding / split-KV** fixes this by partitioning the KV sequence into `num_kv_splits` chunks computed by **separate workgroups in parallel**, then a small **reduction** kernel combines the partial `(acc, m, l)` using the same online-softmax merge.

### 4.1 Phase 1 — partial attention per split

```python
# grid = (B, H, num_kv_splits)
split = program_id(2)
n_per_split = cdiv(seq_len, num_kv_splits)
n_lo = split*n_per_split; n_hi = min(seq_len, n_lo+n_per_split)
m=-inf; l=0; acc=zeros(d)
for t in range(n_lo, n_hi, page_size):
    ... gather page, online softmax (as §3) ...
# write partial state for this split:
partial_o[B,H,split]   = acc          # unnormalized
partial_lse[B,H,split] = m + log2(l)  # log-sum-exp of the split
```

### 4.2 Phase 2 — split reduction (LSE merge)

```python
# grid = (B, H); combine num_kv_splits partials
m_global = max_s partial_lse[...,s]
acc=0; l=0
for s in range(num_kv_splits):
    scale = exp2(partial_lse[...,s] - m_global)
    acc += scale * partial_o[...,s]
    l   += scale
out = acc / l
```

This is exactly FA's online-softmax associativity applied across splits. `num_kv_splits` is the **key decode knob**: enough to fill 304 CUs (`B·H·splits ≳ 600-1200`) but not so many that the reduction + extra HBM writes dominate.

| Regime | num_kv_splits |
|---|---|
| 1 request, long context | 8-32 (need splits to fill CUs) |
| large batch (B·H already ≫ 304) | 1-2 (no splits needed) |
| medium batch, medium context | 4-8 |

aiter/sglang auto-pick `num_kv_splits` from `B, H, seq_len`; the Triton fallback exposes it as a launch param.

---

## 5. GQA head packing in decode

With GQA, `g = H_q / H_kv` query heads share one KV head. In decode this is gold: **load each KV page once, reuse for all `g` query heads**. Pack the `g` query vectors into the rows of the QKᵀ so a single K page load feeds `g` GEMVs → turns the per-KV-byte arithmetic from 1 to `g`, easing the bandwidth bottleneck. For DeepSeek MLA decode the absorbed form makes this an extreme case (`g` = all heads share **one** latent KV head → pure MQA; see `mla.md`).

Map `program_id` to `(batch, kv_head, split)` and vectorize the `g` query heads inside the workgroup.

---

## 6. FP8 KV cache in attention

Halving KV bytes directly relieves the bandwidth bound — the single biggest decode lever after split-KV.

- Store K/V pages in **fp8 `e4m3fnuz`** (CDNA3 dialect — see the `fnuz` note below), keep Q in bf16.
- Dequant K/V to bf16/f32 *inside* the kernel right before the GEMV, or use `v_mfma_*_fp8` directly with f32 accumulate.
- Per-page or per-channel **scales** stored alongside pages; recent aiter PA adds `stride_scale_page` for the scale write.
- Accuracy: fp8 KV is generally safe for decode (values already softmaxed-weighted); use per-channel scales if you see degradation.

**`fnuz` trap (CDNA3-only):** MI300X fp8 is `e4m3fnuz` / `e5m2fnuz` — finite, NaN, unsigned-zero, **bias differs by 1** from OCP fp8 (MI325/350/355X). Use `torch.float8_e4m3fnuz`; misreading the dialect yields values off by exactly **2×**.

Bytes streamed per token (d=128, one KV head): bf16 = 512 B, **fp8 = 256 B**. On a 5.3 TB/s GPU that's a ~2× ceiling lift on the dominant cost.

---

## 7. aiter decode attention — the ROCm default

aiter (AI Tensor Engine for ROCm) is the **default attention backend in both vLLM (V1) and sglang on ROCm**. For decode it provides tuned paged-attention and an **MLA decode** kernel (the headline DeepSeek path, up to ~17× over the naive MLA decode). Enable in vLLM with `VLLM_ROCM_USE_AITER=1` (master switch; sub-flags default on).

Recent aiter paged-attention work (release notes):
- **runtime dispatch for >4 GB KV cache** in batch prefill;
- `top_k_per_row` prefill fix for `batched_token_num > 4096`;
- **gfx942/gfx950 PA "PS" kernel** update with `stride_scale_page` write (fp8 scale);
- MLA `nhead=32` non-persistent decode crash fix (gfx950).

aiter decode for MLA runs the **MQA form** of MLA (one latent KV head shared across all query heads) — the structurally-ideal shape for a memory-bound decode GEMV. See `mla.md` §decode.

### gfx942 coverage caveats (DeepSeek V3.2/V4 era)
For sparse/MLA decode on **gfx942 specifically**, some aiter paths are **missing** (paged MQA logits, sparse MLA decode) or **broken** (sparse prefill logits) and fall back to Triton (several× slower). When writing DeepSeek decode kernels, guard on `gfx942` and provide a Triton fallback. (Details in `deepseek_v3_v4_attention.md`.)

---

## 8. CK paged attention (`ck_tile` splitkv)

The CK `ck_tile` FMHA `splitkv` pipeline is the prefill/decode-unified split-KV implementation; the paged/decode variants add:
- batch-prefill + paged KV with **flexible page sizes** and lookup tables;
- **fp8 KV cache**;
- **streamingllm / gpt-oss sink** support in the splitkv pipeline (sliding-window decode with attention sink).

CK is the lowest-level option; you typically reach aiter (which wraps CK) unless you're adding a new KV layout or mask.

---

## 9. Tuning-knob table (paged decode on MI300X)

| Knob | Typical | Effect | MI300X guidance |
|---|---|---|---|
| `num_kv_splits` | 1-32 | parallelism to fill 304 CUs | scale so `B·H·splits ≈ 600-1200`; long-ctx/low-batch → 8-32 |
| `page_size` | 1,16,32,64,128 | coalescing vs frag | 16 default; 64/128 for long ctx + bandwidth; match dispatched kernel |
| KV dtype | bf16 / **fp8 fnuz** | bytes streamed | fp8 ≈ 2× bandwidth headroom |
| `BLOCK_N` (tokens/iter) | = page_size or 2-4 pages | gather granularity | multiple pages per iter improves burst |
| GQA group packing `g` | model-fixed | reuse KV per query head | always pack all `g` heads per KV load |
| K layout inner `x` | 8 / 16 | 128-bit `buffer_load` | x=8 (bf16) / 16 (fp8) for coalesced K |
| `waves_per_eu` | 2-4 | occupancy to hide HBM latency | high (decode is latency-bound); raise if no spills |
| reduction kernel fuse | on/off | avoid extra launch | fuse phase-2 when splits small |
| CUDA/HIP-graph capture | on | kill launch overhead | essential for decode (many tiny kernels) |

**Latency-hiding note:** decode is HBM-latency-bound, so **high occupancy** (many in-flight waves per EU) matters more than in prefill — you want enough waves to keep `buffer_load` requests in flight. Use HIP graphs to remove per-step CPU launch overhead (decode launches one attn kernel per token).

---

## 10. Checklist for a high-quality MI300X paged-decode kernel

1. **Paged KV** with `x`-vectorized K layout for 128-bit `buffer_load`.
2. **Split-KV** with auto `num_kv_splits` to fill 304 CUs; LSE-merge reduction.
3. **Online softmax** GEMV (M=1), `exp2`, f32 accum.
4. **GQA pack** all `g` query heads per KV load.
5. **fp8 KV** (`e4m3fnuz`) + per-page scales to halve bandwidth.
6. **HIP-graph** capture; static/capture-safe metadata (no host→device scalar writes under capture).
7. High `waves_per_eu`; verify coalesced K/V loads in the profiler (HBM BW utilization should approach 5.3 TB/s).
8. **Dispatch to aiter** when available; Triton fallback for gfx942 gaps. If within ~15% of aiter, ship your own; else use aiter.

---

## Sources

- vLLM PagedAttention (Efficient Memory Management for LLM Serving, SOSP 2023) — https://arxiv.org/abs/2309.06180
- Flash-Decoding for long-context inference (Tri Dao et al.) — https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- AITER-Enabled MLA Layer Inference on AMD Instinct MI300X (ROCm Blog) — https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- ROCm/aiter releases (paged-attention / FMHA / fp8 scale updates) — https://github.com/ROCm/aiter/releases
- vLLM V1 performance optimization on ROCm (AITER attention backend defaults) — https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- ROCm/composable_kernel `ck_tile` FMHA splitkv + paged + sink (CHANGELOG) — https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
- Accelerate DeepSeek-R1: Integrate AITER into SGLang (ROCm Blog) — https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
- AMD Instinct MI300X architecture (Hot Chips 2024, bandwidth/LDS specs) — https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf
