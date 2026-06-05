# Linear / Recurrent Attention on AMD MI300X (CDNA3 / gfx942)

> Scope: **linear / recurrent attention** — Mamba2 **SSD** (state-space duality), **gated DeltaNet** (Qwen3-Next / Qwen3.5 workhorse), RWKV, and the **chunked-scan / chunkwise-parallel** algorithm that makes them train- and prefill-efficient (intra-chunk parallel matmuls + inter-chunk recurrence). Includes the chunked-scan core kernel logic, state passing, why these are **Triton** kernels on MI300X, and a tuning table. Softmax attention is in the other operator docs; hybrid models (e.g. Qwen3-Next 3:1) mix this with full attention layers (`attention_prefill.md`/`attention_decode_paged.md`).
>
> **AMD-only.** Targets MI300X (304 CU, 5.3 TB/s HBM3, 64 KiB LDS, 1216 SIMDs).

---

## 0. TL;DR

| Want | Use on MI300X | Why |
|---|---|---|
| Mamba2 layer (SSD) | `mamba_chunk_scan_combined` (Triton, from `mamba_ssm` / vLLM) | chunked scan → uses MFMA via matmuls |
| Gated DeltaNet (Qwen3-Next/3.5) | **flash-linear-attention (`fla`)** `chunk_gated_delta_rule` (Triton) | the production GDN kernel; runs on ROCm Triton |
| RWKV | fused recurrence/chunk Triton kernels | linear-time state recurrence |
| Hybrid model serving | vLLM hybrid KV-cache manager + full CUDA/HIP-graph | manages linear-state + full-attn layers together |
| Write/port a new variant | **Triton** | no tuned CK/aiter linear-attention kernels on gfx942; Triton is portable |

**Why Triton, not CK/aiter:** linear-attention kernels are **not** in aiter's or CK's tuned attention set on gfx942 — the ecosystem (Mamba2, GDN, GLA, RWKV) standardized on **`flash-linear-attention` / `mamba_ssm` Triton kernels**, which compile to MI300X via the ROCm Triton backend. So on MI300X you optimize *Triton*: MFMA tile sizes, LDS state tiling, `num_warps/waves_per_eu`, and chunk size.

---

## 1. The recurrence, and why naive scan is bad on MI300X

A linear/recurrent attention layer maintains a **matrix state** `S ∈ [d_k, d_v]` (the "fast weights" / SSM state) updated per token:

```
# generic gated linear-attention recurrence (decode form)
S_t = diag(a_t) · S_{t-1} + k_t v_tᵀ      # state update  (a_t = gate/decay)
o_t = S_tᵀ q_t                             # output read
```

This is `O(L)` but **purely sequential** and **not matmul** — it's element-wise + outer-products on the vector ALU. On CDNA3 that means it **never touches the 1307 TFLOP/s matrix cores** and is bottlenecked by non-matmul throughput (the same gap Tri Dao notes: ~16× between matmul and non-matmul FLOPs). Fine for *decode* (one token, state is tiny), catastrophic for *prefill/training* (sequential over thousands of tokens).

**Chunking is the fix.** Split the sequence into chunks of size `C`; do the heavy work as **matmuls inside chunks** (parallel, MFMA), and only a **short recurrence over chunk boundaries** (sequential, but `L/C` steps instead of `L`).

---

## 2. Mamba2 SSD — the chunked-scan algorithm

Mamba2's **State Space Duality** views the SSM as a structured (semiseparable) matrix `M` where `y = M x`. Block-decompose `M` into `C×C` blocks: **diagonal blocks** use the quadratic (attention-like) form; **off-diagonal blocks** are low-rank and factor into `B` (right), `A` (decay/middle), `C` (left) terms. This yields a **4-step** algorithm — three of the four steps are pure matmuls (tensor cores), only one is a scan over chunks.

| Step | Name | Compute | Parallel? |
|---|---|---|---|
| 1 | **Intra-chunk outputs** | diagonal blocks: `Y_diag = (C Bᵀ ∘ L) X` per chunk | ✅ matmul, all chunks parallel |
| 2 | **Chunk states** | final state of each chunk assuming zero init: `states = (B ∘ decay)ᵀ X` | ✅ matmul, all chunks parallel |
| 3 | **Pass states (inter-chunk recurrence)** | scan over chunk-final states: `new_states = scan(A_chunk, states)` | ⚠️ sequential, but only `L/C` steps |
| 4 | **Output from initial states** | off-diagonal: `Y_off = (C ∘ decay) · state_in` per chunk | ✅ matmul, all chunks parallel |
| — | combine | `Y = Y_diag + Y_off` | ✅ |

`L = exp(segsum(A))` is the within-chunk cumulative-decay matrix; `segsum` = segmented cumulative sum of the log-decays. Steps 1,2,4 are GEMMs → **MFMA on MI300X**; step 3 is the only sequential part and runs on `L/C` chunk states (≈100× shorter), so its `O(L²)`-if-materialized or `O(L)`-if-scanned cost is negligible.

### 2.1 Chunked-scan core logic (the math the Triton kernel implements)

```python
# X:[B,L,H,d_v]  A:[B,L,H] (log-decay)  B,C:[B,L,H,d_state]
# reshape L -> (n_chunks, chunk_len = Cn)
A = A.reshape(b, n_chunks, Cn, h)
A_cumsum = A.cumsum(dim=2)                       # within-chunk cumulative decay

# STEP 1: intra-chunk (diagonal) — attention-like quadratic form
Lmat = exp(segsum(A))                            # [b,h,nc,Cn,Cn] lower-tri decay
scores = einsum('bclhn,bcshn->bhcls', C, B)     # C·Bᵀ  (MFMA)
Y_diag = einsum('bhcls,bhcls,bcshv->bclhv', scores, Lmat, X)   # ∘L then ·X

# STEP 2: each chunk's end-state assuming zero init-state
decay_states = exp(A_cumsum[...,-1:] - A_cumsum) # decay to chunk end
states = einsum('bclhn,bhcl,bclhv->bchnv', B, decay_states, X)  # (MFMA)

# STEP 3: INTER-CHUNK RECURRENCE — sequential over n_chunks
decay_chunk = exp(segsum(pad(A_cumsum[...,-1])))  # cross-chunk decay
new_states  = einsum('bhzc,bchnv->bzhnv', decay_chunk, states)  # prefix over chunks
state_in    = new_states[:, :-1]                  # init-state fed to each chunk

# STEP 4: off-diagonal — contribution from incoming state
state_decay_out = exp(A_cumsum)
Y_off = einsum('bclhn,bchnv,bhcl->bclhv', C, state_in, state_decay_out)  # (MFMA)

Y = (Y_diag + Y_off).reshape(b, L, h, d_v)
```

The production `mamba_chunk_scan_combined` Triton kernel differs from this reference in **Step 3**: instead of materializing the `L/C × L/C` decay matrix and multiplying, it runs a real **associative scan** over chunk states, keeping it truly `O(L)`. The reference materializes (simpler, fine because `L/C ≈ 100`).

### 2.2 Fusion on MI300X

PyTorch/IBM fused the whole SSD prefill (conv1d + projection + the 4 SSD steps) into **one Triton kernel** → 1.5×-2.5× over the unfused path, *and changed the optimal chunk size from 256 → 128* (smaller chunk = less register pressure, less sensitive to masks/precision). On MI300X the same logic holds: fewer launches, state kept in LDS/VGPR across steps. `mamba_split_conv1d_scan_combined` is the fused entry (`mamba_ssm`); vLLM uses the chunked-scan path with a **hybrid KV-cache manager** for mixed linear/full-attention models, and **full HIP/CUDA-graph mode** because Triton launch overhead disproportionately hurts decode-only batches.

---

## 3. Gated DeltaNet — Mamba2 gating + delta rule

Gated DeltaNet (NVIDIA, ICLR 2025) = DeltaNet's **delta rule** (correct the state by the prediction error) + Mamba2-style **gated decay**. It is the **workhorse linear layer of Qwen3-Next-80B-A3B and the Qwen3.5 / 3.6 series** (interleaved 3:1 with full attention; Kimi Linear uses the same ratio).

State update (per token, decode form):

```
S_t = α_t · (I − β_t k_t k_tᵀ) S_{t-1} + β_t k_t v_tᵀ      # gated delta rule
o_t = S_tᵀ q_t
#  α_t = scalar gate (decay),  β_t = per-token write strength (delta gate)
```

The `(I − β k kᵀ)` term is the **delta correction** — it removes the old value associated with `k` before writing the new one (associative-memory update), which plain linear attention lacks.

### 3.1 Chunked form (WY representation)

The sequential `(I − β k kᵀ)` products are turned chunk-parallel via the **WY representation** (the same trick as Householder/block-QR): within a chunk, the product of rank-1 updates is expressed as a structured `T = (I − tril(β k kᵀ))⁻¹` solve, after which intra-chunk outputs and chunk-end states are **matmuls**, and only chunk states pass between chunks recurrently. This is what `fla`'s `chunk_gated_delta_rule(q, k, v, g, beta, scale)` computes (`g` = gate, `beta` = delta strength). Same 4-phase skeleton as SSD: intra-chunk parallel + inter-chunk recurrence.

```python
# flash-linear-attention API (Triton; runs on ROCm)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
o, recurrent_state = chunk_gated_delta_rule(
    q, k, v, g=gate, beta=beta, scale=scale,
    initial_state=state,          # for chunked prefill / cross-request carry
    output_final_state=True)      # returns final S for the next chunk/decode
```

### 3.2 MI300X considerations
- `fla` Triton kernels compile on ROCm; `tl.dot` → MFMA. The intra-chunk matmuls (`C×C` and `C×d`) are the MFMA-bound part.
- **Low occupancy risk:** the chunked GDN forward can only launch `batch · num_heads` thread-blocks (the recurrent state is per (b,h)), so for small batch / few heads / TP it **under-fills 304 CUs**. Qwen's **FlashQLA** (TileLang) fixes this on NVIDIA via gate-driven intra-card context parallelism (split the chunk dim further); on MI300X the equivalent lever is to **increase parallel work** — larger batch, or split the head-state/chunk dimension across more workgroups, or use a smaller chunk to expose more blocks (watch the recurrence cost).
- **Decode** GDN is the pure recurrence of §1 — tiny, memory-bound; keep state resident, batch across requests, HIP-graph the step.

---

## 4. RWKV (and the general shape)

RWKV (v5/v6 "Eagle/Finch", v7) is another recurrent/linear family: a per-channel decay state with token-shift and a WKV recurrence. Same MI300X story — fused **Triton WKV** kernels (chunked for prefill/training, pure recurrence for decode). The state is per-channel (vector, not matrix in older versions; matrix-valued in v7's delta-rule-like update), so state tiling/LDS budgeting is the key knob. All linear families share: **chunk for prefill (matmul), recur for decode (vector ALU)**.

---

## 5. State passing (chunked prefill, cross-request, hybrid cache)

The defining feature vs softmax attention: a **compact recurrent state** flows forward, so you don't keep a growing KV cache for these layers.

- **Within a sequence:** chunk `i`'s final state = chunk `i+1`'s `initial_state` (Step 3 / `output_final_state=True`).
- **Chunked prefill (serving):** a long prompt processed in chunks carries `recurrent_state` between chunks — pass `initial_state` in, take final state out.
- **Decode:** the state *is* the cache — fixed-size `[d_k, d_v]` per (b,h), independent of context length (the big memory win over softmax KV cache).
- **Hybrid models (Qwen3-Next 3:1, Mamba-Transformer):** vLLM's **hybrid KV-cache manager** allocates softmax-KV pages for full-attn layers **and** fixed-size recurrent-state slots for linear layers in one allocator → no fragmentation, both managed under one HIP-graph.

---

## 6. Tuning-knob table (linear attention on MI300X)

| Knob | Typical | Effect | MI300X guidance |
|---|---|---|---|
| **chunk size `C`** | 64, 128, 256 | matmul efficiency vs recurrence length vs register pressure | **128** is the modern sweet spot (post-fusion); 256 only if VGPR allows; smaller `C` exposes more blocks to fill CUs but lengthens the scan |
| `BLOCK` (state tile `d_state`/`d_v`) | 32, 64, 128 | LDS/VGPR for state `S` | tile so `S` + chunk tiles fit 64 KiB LDS |
| `num_warps` | 4, 8 | wavefronts/WG (64 lanes each) | 4 for C=128; 8 for larger C/state |
| `num_stages` | 1, 2 | SW pipeline of loads vs MFMA | 2 to overlap `global_load` with `v_mfma` if LDS allows |
| `waves_per_eu` | 1-3 | occupancy | raise to hide latency; lower if register spills |
| fuse conv1d+proj+SSD | on | one launch | use `*_combined` fused kernel (Mamba2) |
| HIP/CUDA-graph | on | kill Triton launch overhead | essential for decode-only batches |
| state dtype | bf16 state / f32 accum | accuracy | f32 accumulate the recurrence; bf16 store |
| parallelize (b·h small) | split chunk/head | fill 304 CUs | add CP-style split when `b·h ≪ 304` (the FlashQLA insight) |

---

## 7. Checklist for a high-quality MI300X linear-attention kernel

1. **Chunk it** — intra-chunk matmuls (MFMA) + inter-chunk recurrence; never a token-level scan for prefill.
2. **Map the 3 parallel steps to `tl.dot`** (MFMA); keep only the short chunk-state pass sequential.
3. **Chunk size 128** as the default; autotune 64/128/256, watch VGPR spills.
4. **Tile the state** to fit LDS; f32 accumulate, bf16 store.
5. **Fuse** conv1d + projection + scan into one kernel (Mamba2 `*_combined`).
6. **State passing** via `initial_state`/`output_final_state` for chunked prefill and decode-as-cache.
7. **Fill 304 CUs**: if `batch·heads` is small, split chunk/head dims (FlashQLA-style CP) — the GDN low-occupancy trap.
8. **HIP-graph** decode; the recurrence step is tiny and latency-bound.
9. Use **`fla` / `mamba_ssm`** Triton kernels as the baseline and the thing to beat — there is no tuned CK/aiter linear-attention on gfx942.

---

## Sources

- State Space Duality (Mamba-2) Part III — The Algorithm (Tri Dao / Goomba Lab) — https://tridao.me/blog/2024/mamba2-part3-algorithm/
- Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba2, arXiv:2405.21060) — https://arxiv.org/abs/2405.21060
- Accelerating Mamba2 with Kernel Fusion (PyTorch blog, fused Triton SSD, chunk 256→128) — https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/
- Gated Delta Networks: Improving Mamba2 with Delta Rule (NVIDIA, ICLR 2025) — https://github.com/NVlabs/GatedDeltaNet
- flash-linear-attention (`fla`): chunk_gated_delta_rule and friends (Triton) — https://github.com/fla-org/flash-linear-attention
- vLLM Now Supports Qwen3-Next: Hybrid Architecture (GDN+full attn, hybrid KV cache, FLA Triton) — https://blog.vllm.ai/2025/09/11/qwen3-next.html
- QwenLM/FlashQLA: CP-/Bwd-friendly fused GDN kernels (TileLang) — https://github.com/QwenLM/FlashQLA
- state-spaces/mamba (`mamba_chunk_scan_combined`, `mamba_split_conv1d_scan_combined`) — https://github.com/state-spaces/mamba
