# Sparse Attention on AMD MI300X (CDNA3 / gfx942)

> Scope: **sparse attention** — block-sparse masks, sliding-window + attention sink (StreamingLLM / gpt-oss), DeepSeek **NSA** (Native Sparse Attention) and **DSA** (DeepSeek Sparse Attention, V3.2/V4), top-k / landmark block selection, and **how sparsity becomes block-skipping inside a flash kernel** on MI300X. The dense kernels are in `attention_prefill.md` / `attention_decode_paged.md`; the DeepSeek-specific serving stack is in `deepseek_v3_v4_attention.md`; the MLA interaction in `mla.md`.
>
> **AMD-only.** Numbers/instructions/backends target MI300X (304 CU, 5.3 TB/s, 64 KiB LDS).

---

## 0. TL;DR — the three families and the one kernel idea

| Family | Pattern | Decided by | MI300X mapping |
|---|---|---|---|
| **Static block-sparse** | fixed mask (sliding window, dilated, A-shape) | layout, not data | skip KV blocks by mask at compile/launch |
| **Sliding-window + sink** | last `w` tokens + first few "sink" tokens | position | clamp KV range + always-include sink block |
| **Dynamic top-k (NSA/DSA)** | per-query top-`n` blocks chosen by a cheap scorer | data (indexer / pooled scores) | gather selected block ids → flash over the gathered set |

**The single unifying idea:** a flash attention kernel already loops over KV blocks. *Sparsity = don't run the QKᵀ/PV MFMA for blocks the mask says are empty.* You either (a) **never iterate** skipped blocks (static / windowed → cheap, deterministic) or (b) **iterate a gathered list of block ids** (dynamic top-k → an extra indirection like paged attention's block table). Both keep the dense online-softmax core unchanged.

---

## 1. Why sparse attention, and why it's an MI300X *block* problem

For 64k-context decode, softmax attention is **70-80% of total latency**. Sparsity cuts core-attention complexity from `O(L²)` to `O(L·k)` (k = selected tokens ≪ L). But naive *token-level* sparsity destroys GPU throughput: MI300X (like all GPUs) wants **contiguous, coalesced `buffer_load`** and **dense MFMA tiles**. Scattered single tokens give neither.

So every production sparse-attention design is **block-wise**: select/skip in units of `BLOCK_N` (32-128 tokens) so each retained unit is still a dense MFMA tile with coalesced loads. NSA's own paper states it bluntly: blockwise selection is chosen "because modern GPU architectures exhibit significantly higher throughput for contiguous access," and "attention scores exhibit blockwise clustering." This is doubly true on CDNA3 where the 64-lane wavefront + MFMA 16×16/32×32 tiles reward block granularity ≥ 64.

---

## 2. Block-sparse mask + block-skipping kernel logic

The canonical structure: a **block mask** `M[q_block, k_block] ∈ {0,1}` (or a per-q-block list of active k-blocks). The kernel iterates only active blocks.

### 2.1 With an explicit block mask

```python
@triton.jit
def block_sparse_attn(Q, K, V, BlockMask, sm_scale, Out,
                      stride_..., NUM_KBLK: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    qb   = tl.program_id(0)      # query block
    hz   = tl.program_id(1)
    q    = tl.load(Q_ptr(qb))    # [BLOCK_M, d] resident in VGPR
    m_i  = -inf; l_i = 0; acc = zeros([BLOCK_M, BLOCK_D])
    for kb in range(0, NUM_KBLK):
        if tl.load(BlockMask + qb*NUM_KBLK + kb) == 0:
            continue             # <-- BLOCK SKIP: no buffer_load, no MFMA at all
        k = tl.load(K_ptr(kb))   # only loaded for active blocks
        s = sm_scale * tl.dot(q, k)            # MFMA
        # (optional) element mask only on partial/diagonal blocks
        m_ij  = tl.maximum(m_i, tl.max(s,1))
        p     = tl.math.exp2(s - m_ij[:,None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i   = l_i*alpha + tl.sum(p,1)
        acc   = acc*alpha[:,None] + tl.dot(p.to(V.dtype), tl.load(V_ptr(kb)))
        m_i   = m_ij
    tl.store(Out_ptr(qb), (acc / l_i[:,None]).to(Out.dtype.element_ty))
```

The `continue` is the whole point: a skipped block costs **zero** `buffer_load` and **zero** `v_mfma`. With 90% sparsity you do ~10% of the dense work.

### 2.2 With a gathered active-block list (better — no wasted iterations)

For high sparsity, iterating all `NUM_KBLK` just to `continue` wastes loop overhead. Instead precompute, per q-block, a **list of active k-block ids** (`active_ids[qb, 0:n_active]`), exactly like a paged block table, and iterate that:

```python
n_active = tl.load(NumActive + qb)
for i in range(0, n_active):
    kb = tl.load(ActiveIds + qb*MAX_ACTIVE + i)   # gather the block id
    k  = tl.load(K_ptr(kb)); v = tl.load(V_ptr(kb))
    ... online softmax (dense MFMA over this block) ...
```

This is how NSA/DSA selected-block attention and paged sliding-window decode are implemented: **selection produces block ids; the flash loop consumes them** — structurally identical to PagedAttention's gather (see `attention_decode_paged.md` §3). Coalescing is preserved because each gathered block is contiguous.

---

## 3. Sliding-window + attention sink (StreamingLLM / gpt-oss)

**Sliding window (SWA):** query `q_idx` attends only to `[q_idx - w + 1, q_idx]`. In a flash kernel this is just a **tighter KV-block range** + an element mask on the window edges:

```python
k_lo = max(0, (q_idx - w + 1)) // BLOCK_N
k_hi = (q_idx) // BLOCK_N + 1
for kb in range(k_lo, k_hi):    # only window blocks
    ... mask edges where (q_idx - k_idx) >= w or k_idx > q_idx ...
```

**Attention sink (StreamingLLM):** keeping only a window collapses quality because softmax dumps probability mass on the first tokens. Fix: always also attend to the first few **"sink" tokens** (block 0). So the active set = `{sink block} ∪ {window blocks}`:

```python
# always include sink
process_block(0, mask=causal)
# then the sliding window
for kb in range(k_lo, k_hi): process_block(kb, mask=window∧causal)
```

On MI300X this is directly supported in the tuned kernels: **`ck_tile` FMHA added `streamingllm` sink and `gpt-oss` sink** across the `qr_ks_vs`, `qr_async`, `qr_async_trload`, and **`splitkv`** pipelines (so both prefill and split-KV decode get sink support). gpt-oss uses SWA+sink in alternating layers; this is the path you dispatch to on ROCm. KV cache for SWA is bounded to `w` tokens → constant memory regardless of context length (the StreamingLLM win).

---

## 4. NSA — Native Sparse Attention (DeepSeek, Feb 2025)

NSA is **natively trainable** (sparse in training and inference, not a post-hoc mask) and **hardware-aligned** (blockwise, GQA-grouped). Three parallel branches, combined by a learned gate:

| Branch | Mechanism | Hyperparam (paper) |
|---|---|---|
| **Compression** | group consecutive tokens into blocks, compress each block → one vector (coarse global context) | block `l=32`, stride `d=16` |
| **Selection** | score compressed blocks, keep **top-n** raw blocks → fine attention over them | sel block `l'=64`, **n=16** |
| **Sliding window** | last `w` tokens (local context), kept independent so it doesn't dominate training | `w=512` |

```
out = gate_cmp · Attn(q, K_cmp, V_cmp)        # compressed (coarse)
    + gate_sel · Attn(q, gather(K, topN), gather(V, topN))   # selected (fine)
    + gate_win · Attn(q, K_window, V_window)   # local
```

**Hardware-aligned kernel design** (the part that matters for MI300X): NSA's kernel loads queries **by GQA group** (grid loop), fetches the corresponding **sparse KV blocks** (inner loop), and computes on-chip. Because *all heads in a GQA group share the same selected blocks*, one block load feeds the whole group — exactly the GQA-pack pattern that raises arithmetic intensity (see prefill doc §5). The selection top-n is done at **block granularity** so each fetched block is a dense, coalesced MFMA tile.

Mapping to MI300X: the selection branch = the **gathered active-block flash loop** of §2.2; the compression branch = a tiny dense attention over `L/d` compressed vectors; the window branch = §3. There is no upstream tuned aiter NSA kernel for gfx942 yet — you build it from the block-skip primitives above (Triton), reusing the GQA-group grid loop.

---

## 5. DSA — DeepSeek Sparse Attention (V3.2 / V4)

DSA is the production sparse mechanism shipped in **DeepSeek-V3.2-Exp (Sep 2025)** and **V3.2 / V4 (Dec 2025)**. It is **MLA-based** and has two parts:

1. **Lightning indexer** — a cheap, few-head, **FP8** scorer (ReLU activation, chosen for throughput) that computes an index score between the current query and each preceding token → decides which tokens to select. It keeps its **own separate K-cache** (the "indexer K cache"), distinct from the MLA latent KV cache.
2. **Top-k token selection** — attention runs only over the top-k selected tokens, dropping core-attention from `O(L²)` to `O(L·k)` (k=2048 in V3.2 sparse training).

Because it's built on MLA's **MQA mode**, each latent KV entry is shared across **all** query heads → one gather feeds every head (the strongest possible GQA-pack). For short-prefill, DeepSeek instead uses a **masked-MHA mode** that simulates DSA (cheaper when L is small).

Kernel surface (what to dispatch to):
- prefill/decode **sparse MLA** attention (FlashMLA sparse kernels upstream);
- the **lightning-indexer** GEMM (DeepGEMM upstream; on ROCm a small fp8 GEMM + top-k).

**On MI300X / gfx942 this is the rough edge.** aiter coverage favors CDNA4; on gfx942 several DSA pieces are **missing** (paged MQA logits, **sparse MLA prefill**, **sparse MLA decode**) and some are **broken** (AITER prefill MQA logits, sparse prefill logits → must refuse dispatch on gfx942 and fall back to Triton, which is several× slower). See `deepseek_v3_v4_attention.md` for the exact PRs/fallbacks. The lightning indexer must use **`e4m3fnuz`** fp8 on CDNA3.

---

## 6. Top-k / landmark block selection — how the block ids are produced

The selection step is a separate, cheap kernel that runs *before* the sparse flash loop and emits the active-block ids:

```python
# 1. score each candidate KV block against the query (cheap, low-dim or compressed)
scores[qb, kb] = reduce( q_pool @ K_block_repr[kb] )   # pooled / compressed / indexer score
# 2. top-n block ids per query block
active_ids[qb, :] = topk(scores[qb, :], n).indices
# 3. (optional) sort ids ascending for monotone causal + better cache locality
```

Variants:
- **NSA compression-as-score:** reuse the compressed block vectors to score blocks → top-n.
- **DSA lightning indexer:** dedicated fp8 ReLU scorer with its own K-cache.
- **Landmark / pooled:** mean/max-pool each KV block → a representative vector; score query against landmarks (a la block-sparse "estimate then select").

MI300X notes: this scorer is small and often **memory-bound** (reads compressed reprs); do it in **fp8** (indexer) or bf16, fuse the top-k where possible, and **sort the selected ids** so the downstream flash loop reads KV monotonically (better L2/Infinity-Cache reuse and causal validity). The top-k itself is a small kernel; for n≤32 a per-row partial sort in LDS is fine.

---

## 7. Static sparse patterns worth knowing (cheap, no scorer)

| Pattern | Mask | Use |
|---|---|---|
| Sliding window | band around diagonal | local LMs, gpt-oss layers |
| Window + sink | band + column 0 block | StreamingLLM, long-stream decode |
| Dilated / strided | every k-th block | Longformer-style |
| A-shape / global+local | first block (global) + window | BigBird-ish |
| Block-diagonal (varlen) | per-document blocks | packed multi-doc prefill |

These need **no selection kernel** — the mask is a pure function of position, so block-skip ranges are computed at launch. Cheapest sparsity; always prefer when the pattern is fixed.

---

## 8. Tuning-knob table (sparse attention on MI300X)

| Knob | Typical | Effect | MI300X guidance |
|---|---|---|---|
| selection `BLOCK_N` | 32, 64, 128 | block granularity | ≥64 for coalesced `buffer_load` + dense MFMA; 64 (NSA sel) is a good default |
| top-n selected blocks | 8-32 (NSA n=16) | accuracy vs work | more = closer to dense; tune per quality budget |
| window `w` | 256-1024 (NSA 512) | local span | bounds SWA KV memory |
| sink size | 1-4 blocks | quality of windowed | 1 block (first tokens) usually enough |
| active-list vs mask | list for high sparsity | loop overhead | use gathered ids when sparsity >~50% |
| id sorting | on | cache locality + causal | sort ascending before flash loop |
| indexer dtype | **fp8 fnuz** | scorer cost | fp8 for DSA indexer on CDNA3 |
| GQA group pack | model-fixed | one block load feeds group | always pack (NSA grid-loop pattern) |
| compression block/stride | l=32,d=16 (NSA) | coarse-branch cost | model-defined |
| split-KV over selected | 1-8 | fill 304 CUs in sparse decode | needed when n·heads small |

---

## 9. Checklist for a high-quality MI300X sparse-attention kernel

1. **Stay block-wise** (`BLOCK_N ≥ 64`): never gather single tokens.
2. **Reuse the dense flash core** — sparsity only changes *which* blocks enter the online-softmax loop.
3. **Skip cleanly**: `continue` (low sparsity) or a **gathered active-id list** (high sparsity); no wasted `buffer_load`/MFMA.
4. **Sort selected block ids** for cache locality and causal monotonicity.
5. **GQA-pack** so one selected-block load feeds the whole group (NSA grid-loop).
6. **Sink + window** via tighter KV range + always-include block 0; dispatch to `ck_tile` sink pipelines when the pattern is static.
7. **DSA indexer in fp8 `e4m3fnuz`**; separate indexer K-cache.
8. **Split-KV over the selected set** in sparse decode to fill 304 CUs.
9. On **gfx942**, guard DeepSeek sparse-MLA paths and provide Triton fallbacks (aiter gaps).

---

## Sources

- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (DeepSeek, arXiv:2502.11089, ACL 2025) — https://arxiv.org/abs/2502.11089
- DeepSeek-V3.2-Exp in vLLM: Fine-Grained Sparse Attention in Action (vLLM Blog) — https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html
- SGLang Day-0 Support for DeepSeek-V3.2 with Sparse Attention (LMSYS Blog) — https://www.lmsys.org/blog/2025-09-29-deepseek-V32/
- StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453) — https://arxiv.org/abs/2309.17453
- ROCm/composable_kernel `ck_tile` FMHA streamingllm + gpt-oss sink (splitkv/qr_*) — https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
- Bringing up DeepSeek-V4-Flash on AMD MI300X (sparse MLA / indexer, gfx942 gaps) — https://fergusfinn.com/blog/deepseek-v4-flash-mi300x/
- lucidrains/native-sparse-attention-pytorch (reference NSA impl) — https://github.com/lucidrains/native-sparse-attention-pytorch
- DeepSeek-V3.2 technical report (arXiv:2512.02556) — https://arxiv.org/abs/2512.02556
