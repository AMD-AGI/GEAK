# Multi-head Latent Attention (MLA) on AMD MI300X (CDNA3 / gfx942)

> Scope: DeepSeek **Multi-head Latent Attention** — the latent-KV math (down/up projection), **weight absorption / matrix merging** for decode, why MLA is KV-cache-efficient, the **decoupled RoPE** trick, prefill-vs-decode MLA kernels, and the **aiter / sglang / CK** MLA implementations on MI300X (incl. fp8). Includes the absorption + decode kernel logic and a tuning table. The DeepSeek-V3/V4 serving stack (MoE, sparse DSA, PRs) is in `deepseek_v3_v4_attention.md`; generic paged decode in `attention_decode_paged.md`; sparse selection in `sparse_attention.md`.
>
> **AMD-only.** Targets MI300X (304 CU, 5.3 TB/s HBM3, 64 KiB LDS). Concrete shapes use DeepSeek-V3/R1 config.

---

## 0. TL;DR

| Phase | Use on MI300X | Form | Why |
|---|---|---|---|
| **Decode** | **aiter `mla_decode_fwd`** (sglang/vllm ROCm default) | **MQA** (absorbed) | one latent KV head (576-d) shared across all 128 query heads → up to ~17× over naive; ~2× e2e |
| **Prefill** | aiter MHA / ck_tile FMHA over up-projected K/V | **MHA** (non-absorbed) | many new tokens → compute-bound, absorption not worth it |
| fp8 | aiter MLA fp8 / ck_tile fp8 (`e4m3fnuz`) | both | latent + KV in fp8, halves the (already small) KV bytes |
| write/port | Triton MLA decode (sglang triton backend) | MQA | readable; fallback when aiter lacks a gfx942 path |

**The MLA decode shapes you must memorize (DeepSeek-V3):**

| Symbol | Value | Meaning |
|---|---|---|
| `num_heads` | 128 | query heads |
| `kv_lora_rank` | **512** | latent KV dim (the cached vector) |
| `qk_rope_head_dim` | **64** | decoupled-RoPE dim |
| **cached latent width** | **512 + 64 = 576** | *bytes per token in KV cache* |
| `qk_nope_head_dim` | 128 | content part of Q/K per head |
| `q_head_dim` | 128 + 64 = **192** | full Q head dim (nope+rope) |
| `v_head_dim` | 128 | value head dim |
| `q_lora_rank` | 1536 | query down-proj rank |

The headline: **576 numbers cached per token, total, shared by all 128 heads** — vs MHA's `2·128·128 = 32768`. That ~**57× KV-cache shrink** is the whole point; it's what lets DeepSeek serve long context on 192 GB MI300X.

---

## 1. The MLA math (what is cached, and why it's small)

Standard MHA caches per-head `K,V ∈ [H, d]` → huge. MLA instead **down-projects** the hidden state to a single low-rank **latent** `c_KV ∈ [kv_lora_rank=512]` and caches only that (plus a tiny decoupled-RoPE key). At use time it **up-projects** the latent back to per-head K and V.

```
# --- per token t ---
c_KV   = W_DKV · h_t                 # down-proj: hidden -> latent [512]   (CACHED)
k_pe   = RoPE(W_KR · h_t)            # decoupled rope key [64]             (CACHED)
# cached KV per token = concat(c_KV, k_pe) = 576 dims, ONE head

# query side
c_Q    = W_DQ · h_t                  # query down-proj -> [q_lora_rank=1536]
q_nope = W_UQ · c_Q                  # per-head content query [H,128]
q_pe   = RoPE(W_QR · c_Q)            # per-head rope query   [H,64]

# up-projection of latent (only conceptually; absorbed away at decode)
k_nope = W_UK · c_KV                 # -> per-head content key [H,128]
v      = W_UV · c_KV                 # -> per-head value       [H,128]
```

Attention score = **content term + position term**:

```
S = q_nope · k_nopeᵀ   +   q_pe · k_peᵀ
#   (uses latent via W_UK)   (decoupled RoPE, shared single k_pe head)
O = softmax(S) · v        # v via W_UV
```

**KV-cache efficiency:** you store `c_KV(512) + k_pe(64) = 576` per token, once, regardless of head count. With 128 heads that's the ~57× reduction over MHA. On MI300X this turns the decode KV stream from bandwidth-crushing to manageable — the dominant decode cost (`attention_decode_paged.md`) shrinks directly with bytes-per-token.

---

## 2. Decoupled RoPE — why MLA needs two key parts

RoPE is **position-dependent**, but the latent `c_KV` is position-agnostic (so it can be cached and absorbed). These conflict: if you applied RoPE to the up-projected `k_nope`, you couldn't absorb `W_UK` (the rotation sits between Q and K and breaks the matrix merge).

**Fix (decoupled RoPE):** split each head into two sub-heads:
- a **content** part (`q_nope`/`k_nope`, dim 128) — *no RoPE*, position-agnostic → absorbable;
- a **position** part (`q_pe`/`k_pe`, dim 64) — *RoPE applied here only*, and `k_pe` is a **single shared head** cached alongside the latent.

Final score sums the two. The elegance: absorption applies to the content half; RoPE rides on the position half independently. That's why the cached width is `512 + 64` and the Q head is `128 + 64 = 192`.

---

## 3. Weight absorption (matrix merge) — the decode accelerator

At **decode** (one new token, attending to a long latent cache), you do **not** want to up-project the whole cache to per-head K/V every step — that's `O(L·H·d)` work and re-materializes the big tensors you compressed away. **Weight absorption** folds the up-projections into the query/output matrices so K/V are *never materialized*:

- Absorb `W_UK` into the query: `q_nope · (W_UK · c_KV)ᵀ = (q_nope · W_UK) · c_KVᵀ`. Precompute `q_absorb = q_nope @ W_UK` → the query now attends **directly to the cached latent `c_KV`**.
- Absorb `W_UV` into the output: apply `W_UV` after the `softmax · c_KV` instead of before.

After absorption the cache has **one** KV head (the 576-d latent), and all 128 query heads attend to it → the computation **is structurally MQA**. This is ideal for memory-bound decode: load the latent once, reuse for every head (max GQA-pack, `attention_decode_paged.md` §5). It also *raises arithmetic intensity* (~2× over plain MQA), which suits MI300X's matrix cores.

The crossover: **decode → absorbed (MQA)**; **prefill → non-absorbed (MHA)**. With many new tokens, materializing K/V once and doing dense MHA beats absorbing (the absorbed form trades extra per-head FLOPs for the cache saving, which only pays when L_new is small).

---

## 4. aiter MLA decode on MI300X — the real kernel

aiter's `mla_decode_fwd` is the **sglang/vllm default MLA decode** on ROCm — the headline DeepSeek path (AMD reports up to **~17×** over the naive MLA decode and ~**2×** end-to-end on MI300X). It runs the **absorbed MQA** form.

### 4.1 Host-side absorption (sglang/aiter pattern)

```python
# split kv_b_proj into the two absorbable halves (once, at load)
w = kv_b_proj.weight.view(num_heads, -1, kv_lora_rank)   # [128, 256, 512]
q_absorb   = w[:, :qk_nope_head_dim, :]                   # W_UK part  [128,128,512]
out_absorb = w[:, qk_nope_head_dim:, :]                   # W_UV part  [128,128,512]

# per step: build the MQA query that attends to the latent directly
q_nope = q_nope.transpose(0,1)                            # [128, bs, 128]
q_nope = torch.bmm(q_nope, q_absorb)                      # absorb W_UK -> [128, bs, 512]
q_input = concat(q_nope.transpose(0,1), q_pe, dim=-1)     # [bs, 128, 576]  (latent + rope)
```

### 4.2 The decode kernel call (latent MQA + paged/varlen)

```python
from aiter.mla import mla_decode_fwd
mla_decode_fwd(
    q_input,            # [bs*sq, num_heads, 576]   (q attends to 576-d latent)
    total_kv,           # paged latent KV cache [.., 576]
    out_asm,            # [bs*sq, num_heads, kv_lora_rank=512]
    qo_indptr,          # [bs+1] varlen query offsets
    kv_indptr,          # [bs+1] varlen KV offsets
    kv_indices,         # [kv_indptr[-1]] page/token indices  (the gather)
    kv_last_page_lens,  # [bs] last-page size per seq
    num_kv_splits,      # split-KV factor to fill 304 CUs
    sm_scale = 1.0 / sqrt(q_head_dim),    # q_head_dim = 192
)
# then apply W_UV and o_proj:
out = torch.bmm(out_asm.transpose(0,1), out_absorb).transpose(0,1)   # absorb W_UV
out = o_proj(out)
```

Internally it is the **split-KV paged decode** of `attention_decode_paged.md` specialized to a single 576-d KV head: gather latent pages via `kv_indices`, online-softmax GEMV over the latent, LSE-merge across `num_kv_splits`. The `indptr/indices/last_page_lens` give varlen + paging. Output is the 512-d `softmax·c_KV` *before* `W_UV` (kept latent until the bmm).

Reference config from the AMD blog: AITER v0.1.4, ROCm 6.4, bf16; ~2× e2e, up to 1.47× scaling with batch and up to 2× with context length (the gain grows with context — exactly because the latent KV stream is the cost).

---

## 5. Prefill MLA on MI300X

Prefill is compute-bound and processes many new tokens, so **don't absorb** — instead:
1. up-project the cached/incoming latent to per-head `k_nope` (`W_UK`) and `v` (`W_UV`);
2. assemble full 192-d Q (`q_nope`‖`q_pe`) and K (`k_nope`‖`k_pe`);
3. run a normal **MHA flash prefill** (aiter MHA / ck_tile FMHA) with **`hdim` = multiple of 32** support (192 is `6·32`) — recent `ck_tile` added the `hdim % 32` path precisely to cover MLA's 192/128 split heads.

So prefill MLA ≈ standard FMHA (`attention_prefill.md`) over the up-projected tensors; the only MLA-specific work is the down/up projections and the decoupled-RoPE assembly. CK FMHA's MLA head shape is `hdim_q=192, hdim_v=128`.

---

## 6. fp8 MLA on CDNA3

- Cache the latent `c_KV` and `k_pe` in **fp8 `e4m3fnuz`** (CDNA3 dialect — bias differs by 1 vs OCP fp8 on MI325/350; misreading → 2× error). Halves the already-small 576-byte/token cache.
- Up/down projection GEMMs in fp8 (`v_mfma_*_fp8`, f32 accum) via `aiter.tuned_gemm`.
- DeepSeek-V3 ships **fp8 (W8A8)** weights; sglang/vllm MLA on ROCm support the fp8 path. Use **block-scale** quant for outlier-heavy activations.
- FlashMLA (upstream) added fp8 + sparse MLA kernels (V3.2 DSA); on gfx942 the sparse-MLA paths are partly missing/broken in aiter — see `deepseek_v3_v4_attention.md`.

---

## 7. Tuning-knob table (MLA on MI300X)

| Knob | Typical | Effect | MI300X guidance |
|---|---|---|---|
| `num_kv_splits` (decode) | 1-32 | fill 304 CUs (single-head latent MQA) | latent MQA = few workgroups → **need splits**; long ctx/low batch → 8-32 |
| absorb vs non-absorb | decode/prefill | MQA vs MHA | absorb for decode, MHA for prefill (crossover at small L_new) |
| latent dtype | bf16 / **fp8 fnuz** | KV bytes | fp8 to shrink 576B/token further |
| page_size (latent cache) | 16/32/64/128 | coalescing | match aiter MLA paged kernel; bigger pages for long ctx |
| `q_head_dim` path | 192 (hdim%32) | prefill FMHA | ensure ck_tile hdim%32 kernel selected |
| GQA-pack | all 128 heads → 1 latent | reuse latent | inherent in absorbed MQA; the max-reuse case |
| `waves_per_eu` | 2-4 | occupancy (latency-bound decode) | high; raise if no VGPR spills |
| HIP-graph | on | launch overhead | essential (one MLA decode kernel per token) |
| tuned_gemm | aiter | down/up projections | use `aiter.tuned_gemm`; tune for 512/1536 ranks |

**Filling CUs is the MLA-decode pitfall:** because the absorbed form is a *single* 576-d KV head, you get even fewer base workgroups than ordinary GQA decode → `num_kv_splits` is the most important knob. Set it so `bs · 128_heads · splits` saturates 304 CUs without over-fragmenting the LSE reduction.

---

## 8. Checklist for a high-quality MI300X MLA kernel

1. **Cache only the 576-d latent** (`c_KV`512 + `k_pe`64), one head, all query heads share it.
2. **Decoupled RoPE**: content (128, no RoPE, absorbable) + position (64, RoPE, shared key).
3. **Decode = absorb** (`W_UK`→query, `W_UV`→output) → MQA GEMV over the latent; never materialize per-head K/V.
4. **Prefill = MHA** over up-projected K/V; use the `hdim%32` FMHA (q=192, v=128).
5. **Split-KV** with `num_kv_splits` tuned to fill 304 CUs — the key MLA-decode lever.
6. **Paged + varlen** via `qo_indptr/kv_indptr/kv_indices/kv_last_page_lens`.
7. **fp8 `e4m3fnuz`** latent + projections to shrink bytes/token; block-scale quant.
8. **HIP-graph** decode; high `waves_per_eu`.
9. **Dispatch to aiter `mla_decode_fwd`** (sglang/vllm default); Triton MLA as gfx942 fallback. Beat it only if within ~15%.

---

## Sources

- AITER-Enabled MLA Layer Inference on AMD Instinct MI300X (ROCm Blog — shapes, `mla_decode_fwd`, absorption, speedups) — https://rocm.blogs.amd.com/software-tools-optimization/aiter-mla/README.html
- DeepSeek-V3 Technical Report (MLA architecture, kv_lora_rank/rope dims, fp8) — https://arxiv.org/abs/2412.19437
- deepseek-ai/FlashMLA (MLA kernels; MQA mode head_dim_k=576/head_dim_v=512; fp8 + sparse) — https://github.com/deepseek-ai/FlashMLA
- TransMLA: MLA Is All You Need (absorb/MHA↔MQA dual-mode analysis) — https://arxiv.org/abs/2502.07864
- ROCm/aiter (`aiter.mla`, `aiter.tuned_gemm`, MLA decode kernels) — https://github.com/ROCm/aiter
- ROCm/composable_kernel `ck_tile` FMHA hdim%32 + MLA shape (q=192,v=128) (CHANGELOG) — https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
- Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X (MLA + fp8 serving) — https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html
