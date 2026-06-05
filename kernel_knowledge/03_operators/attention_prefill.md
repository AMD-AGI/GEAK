# Prefill (Full / Causal) Attention on AMD MI300X (CDNA3 / gfx942)

> Scope: the **prefill** regime of LLM inference — large `Q` length (the full prompt), full or causal masking, compute-bound, MFMA-driven. This is where FlashAttention-2/3 math, Q/K/V tiling, online softmax and the CK `ck_tile` FMHA / Triton-FA / aiter MHA backends all live. Decode (M=1) is covered in `attention_decode_paged.md`; MLA in `mla.md`; sparse in `sparse_attention.md`.
>
> **AMD-only.** All numbers, instructions and backend names target MI300X. CUDA names appear only as contrast.

---

## 0. TL;DR — what to reach for on MI300X

| Situation | First choice | Why |
|---|---|---|
| Production serving, bf16/fp16 prefill, GQA, varlen | **aiter MHA (CK fmha_v3 under the hood)** — sglang/vllm default on ROCm | Tuned for gfx942, up to ~14× over naive; varlen + paged-prefill support |
| You need to *write/modify* a kernel | **Triton FA** (`triton` flash attention, ROCm fork) | Portable, autotunable, readable; ~80-90% of CK at good tiles |
| You need full control of MFMA layout / new mask | **`ck_tile` 01_fmha** (Composable Kernel) | Lowest-level tile framework, the source of the tuned kernels |
| FP8 prefill (DeepSeek/Qwen FP8) | aiter MHA fp8 / `ck_tile` fp8 fmha fwd (tensor-wise or block-scale quant) | CDNA3 needs **`fnuz`** fp8 dialect (see §7) |
| Long-context, sparsity available | block-sparse / NSA path → see `sparse_attention.md` | skip KV blocks, not just tile faster |

**MI300X reality check:** CDNA3 has **no warp-specialization / no `wgmma` / no TMA**. The FA-3 producer-consumer async-pipeline trick (Hopper) does **not** port directly. On MI300X the levers are: **MFMA tiling, LDS (64 KiB/CU) double-buffering, `waves_per_eu` occupancy, `buffer_load` for K/V streaming, and software pipelining of `ds_read`/`global_load` against `v_mfma`** — not hardware async warpgroups.

---

## 1. The MI300X hardware the prefill kernel runs on

| Resource | Per-CU | Per-GPU | Implication for FMHA |
|---|---|---|---|
| Compute Units | — | **304** (8 XCD × 38) | Grid must produce ≥ ~600-1200 workgroups to fill |
| SIMDs | 4 | 1216 | wavefront = **64 lanes** (not 32) |
| VGPR | 128 KiB (512 × 256-lane? → 65536 × 32-bit) | 128 KiB ×1216 | register pressure caps `BLOCK_M×BLOCK_N` |
| LDS | **64 KiB** | ×304 | holds K/V/Q tiles + softmax scratch; the binding constraint |
| L1 D$ | 32 KiB | ×304 | |
| L2 | 4 MiB | ×8 (per XCD) | K/V reuse across heads |
| Infinity Cache | — | 256 MiB | |
| HBM3 | — | 192 GiB @ **5.3 TB/s** | prefill is compute-bound, but K/V streaming still matters |
| BF16/FP16 matrix | — | **1307 TFLOP/s** | the FMHA roofline ceiling |
| FP8 matrix | — | **2615 TFLOP/s** | fp8 fmha can ~2× bf16 |

Key MFMA instructions for FMHA (CDNA3):

| Instruction | Shape (M×N×K) | dtype | Use |
|---|---|---|---|
| `v_mfma_f32_16x16x16_f16` | 16×16×16 | fp16→f32 | small-tile QK / PV |
| `v_mfma_f32_32x32x8_f16` | 32×32×8 | fp16→f32 | larger tile, fewer issues |
| `v_mfma_f32_16x16x16_bf16` | 16×16×16 | bf16→f32 | bf16 path |
| `v_mfma_f32_16x16x32_fp8` | 16×16×32 | fp8(fnuz)→f32 | fp8 fmha (double K) |

Accumulation is always **f32** in VGPR; softmax (`exp`, max, sum) runs on the **vector ALU** (transcendental `v_exp_f32`), not matrix cores. Unlike Hopper, CDNA3 has **no separate SFU throughput cliff documented as a hard 256× imbalance**, but `exp` still competes with `v_mfma` issue slots — minimizing rescales matters.

---

## 2. The FlashAttention math (what every prefill kernel computes)

Given `Q ∈ [M,d]`, `K,V ∈ [N,d]`, scale `sm_scale = 1/√d`:

```
S = sm_scale · Q Kᵀ        # [M, N]
P = softmax(S, axis=-1)     # row-wise
O = P V                     # [M, d]
```

FlashAttention never materializes `S` or `P` in HBM. It tiles `N` into blocks of `BLOCK_N`, and for each block keeps a running **max `m`**, running **sum `l`**, and running output accumulator **`acc`**, applying the **online softmax** rescale.

### 2.1 Online softmax (the core recurrence)

For each KV block `j` with partial scores `S_j`:

```
m_new   = max(m, rowmax(S_j))           # new running max
α       = exp(m - m_new)                # rescale factor for old state
P_j     = exp(S_j - m_new)              # unnormalized probs for this block
l       = α·l + rowsum(P_j)             # update running denominator
acc     = α·acc + P_j @ V_j             # update running output (still unnormalized)
m       = m_new
```

After the last block: `O = acc / l`. This is numerically stable (subtract running max) and needs only `O(M·d)` state — the whole reason FA fits in LDS/registers.

### 2.2 Causal masking

For causal attention, a query at position `q_idx` may attend only to `k_idx ≤ q_idx`. Two optimizations:

1. **Block skipping:** if `block_k_start > block_q_end`, the entire KV block is masked → **skip the matmul entirely** (≈ halves causal FLOPs).
2. **Diagonal blocks only** need an element-wise mask (`S += (k_idx > q_idx) ? -inf : 0`); off-diagonal lower-triangle blocks are fully visible (no mask, no skip).

```
# pseudo: only iterate KV blocks that can contribute
n_block_max = ((q_block_idx+1)*BLOCK_M) // BLOCK_N + 1   # causal upper bound
for j in range(0, n_block_max):
    is_diagonal = (j*BLOCK_N) <= (q_block_idx+1)*BLOCK_M and ...
    S = sm_scale * (q_tile @ k_tile_j.T)
    if is_diagonal: S += causal_mask_tile     # else no mask
    ... online softmax ...
```

---

## 3. Triton FlashAttention prefill kernel — annotated inner loop (MI300X)

This is the real structure of the ROCm/Triton FA-2 forward (the kind that ships in the Triton tutorials / sglang's triton backend). The MI300X-specific parts are flagged.

```python
import triton, triton.language as tl

@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,
              stride_qz, stride_qh, stride_qm, stride_qk,
              stride_kz, stride_kh, stride_kn, stride_kk,
              stride_vz, stride_vh, stride_vn, stride_vk,
              stride_oz, stride_oh, stride_om, stride_on,
              Z, H, N_CTX,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_DMODEL: tl.constexpr, IS_CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)          # which BLOCK_M of queries
    off_hz  = tl.program_id(1)          # batch*head index

    # ---- pointers into Q/K/V for this (batch,head) ----
    q_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(Q + q_offset, (N_CTX, BLOCK_DMODEL),
        (stride_qm, stride_qk), (start_m*BLOCK_M, 0), (BLOCK_M, BLOCK_DMODEL), (1,0))
    K_block_ptr = tl.make_block_ptr(K + off_hz*stride_kh, (BLOCK_DMODEL, N_CTX),
        (stride_kk, stride_kn), (0,0), (BLOCK_DMODEL, BLOCK_N), (0,1))   # K is loaded transposed
    V_block_ptr = tl.make_block_ptr(V + off_hz*stride_vh, (N_CTX, BLOCK_DMODEL),
        (stride_vn, stride_vk), (0,0), (BLOCK_N, BLOCK_DMODEL), (1,0))

    # ---- online-softmax running state (kept in VGPR) ----
    m_i = tl.zeros([BLOCK_M], tl.float32) - float("inf")   # running max
    l_i = tl.zeros([BLOCK_M], tl.float32)                  # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)    # f32 accumulator

    q = tl.load(Q_block_ptr)                # Q tile stays resident in registers
    qk_scale = sm_scale * 1.44269504        # fold log2(e) so we can use exp2 (faster on CDNA3)

    # causal upper bound on KV blocks (block-skipping)
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        k = tl.load(K_block_ptr)            # [d, BLOCK_N]  (buffer_load on MI300X)
        # ---- S = QKᵀ via MFMA (tl.dot lowers to v_mfma_f32_16x16x16) ----
        qk = tl.dot(q, k)                   # [BLOCK_M, BLOCK_N] f32 accum
        if IS_CAUSAL:
            offs_m = start_m*BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n      + tl.arange(0, BLOCK_N)
            qk = tl.where(offs_m[:,None] >= offs_n[None,:], qk, float("-inf"))
        # ---- online softmax ----
        m_ij  = tl.maximum(m_i, tl.max(qk,1)*qk_scale)
        qk    = qk*qk_scale - m_ij[:,None]
        p     = tl.math.exp2(qk)            # exp2, not exp → maps to v_exp_f32 cheaply
        alpha = tl.math.exp2(m_i - m_ij)
        l_i   = l_i*alpha + tl.sum(p,1)
        acc   = acc*alpha[:,None]           # rescale old O
        v     = tl.load(V_block_ptr)        # [BLOCK_N, d]
        acc  += tl.dot(p.to(v.dtype), v)    # PV matmul (second MFMA)
        m_i   = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:,None]                 # final normalize
    tl.store(M + off_hz*N_CTX + start_m*BLOCK_M + tl.arange(0,BLOCK_M),
             m_i + tl.math.log2(l_i))       # logsumexp, for bwd
    O_block_ptr = tl.make_block_ptr(Out + q_offset, (N_CTX, BLOCK_DMODEL),
        (stride_om, stride_on), (start_m*BLOCK_M,0), (BLOCK_M, BLOCK_DMODEL),(1,0))
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))
```

**MI300X-specific notes inside this kernel:**
- `tl.dot` lowers to **`v_mfma_f32_16x16x16`** (or 32x32x8 for big tiles). The Triton ROCm backend picks the MFMA based on `BLOCK_M/N` and dtype.
- Use **`exp2`** (`tl.math.exp2`) and fold `log2(e)` into `qk_scale`; CDNA3's `v_exp_f32` computes base-2, so `exp2` avoids an extra multiply.
- `K` is loaded **pre-transposed** (`[d, BLOCK_N]`) so the QKᵀ MFMA gets the contraction dim contiguous → coalesced `buffer_load`.
- The grid is `(triton.cdiv(N_CTX, BLOCK_M), Z*H)`. To fill 304 CUs you want `Z*H*ceil(M/BLOCK_M)` ≳ 1200; for short prompts / few heads this **under-fills** — that's the motivation for split-K style prefill or smaller `BLOCK_M`.

---

## 4. CK `ck_tile` FMHA — the tuned backend

`ck_tile` (Composable Kernel's tile framework) is where AMD's *production* FMHA kernels are written; aiter MHA and the ROCm flash-attention CK backend dispatch into these. Source: `example/ck_tile/01_fmha` in ROCm/composable_kernel (now mirrored under ROCm/rocm-libraries).

### 4.1 Pipeline structure

`ck_tile` FMHA maps the FA pipeline onto explicit **tile windows** + a **block-GEMM pipeline**:

| Stage | ck_tile primitive | What it does |
|---|---|---|
| Load Q tile | `tile_window` + `load_tile` | Q[BlockM, d] → LDS/VGPR once |
| Stream K | `tile_window` over N | K[BlockN, d] → LDS, double-buffered |
| `S = QKᵀ` | `BlockGemm` (gemm0) | MFMA, f32 accum in VGPR |
| Softmax | `BlockReduce` (max/sum) + element-wise | online rescale |
| `O += PV` | `BlockGemm` (gemm1) | second MFMA |
| Store O | `store_tile` | O[BlockM, d] → HBM |

Pipeline variants you choose between (these are real ck_tile pipeline names):
- **`qr_ks_vs`** — Q in registers, K/V streamed through LDS. The general workhorse.
- **`qr_async`** — uses `buffer_load` async to overlap K/V global loads with MFMA (CDNA3's substitute for Hopper TMA/async-copy).
- **`qr_async_trload`** — adds transposed load for V.
- **`splitkv`** — splits the KV loop across workgroups + a reduction (used for short-M / long-N to fill CUs; this is the prefill cousin of flash-decoding, see decode doc).

### 4.2 Feature coverage (from CK CHANGELOG, 2024-2026)

- Batch-prefill kernel with multiple **KV cache layouts**, flexible **page sizes**, lookup-table configs.
- **FP8 KV cache** for batch prefill.
- **streamingllm sink** + **gpt-oss sink** support across `qr_ks_vs`, `qr_async`, `splitkv`.
- **FP8 dynamic tensor-wise quant** and **FP8 block-scale quant** for the fwd kernel.
- `hdim` supported as a **multiple of 32** (not only 64/128) → covers MLA's 192/128 split heads.
- Multi-arch build: `Arch` template param so one binary holds gfx942 + gfx950 + gfx12 kernels.

### 4.3 Known gfx942 gotchas
- `fmha_v3` "MI300 kernel set" can **hang at large prefill** (≥ ~20480 tokens) on MI325X because the aiter dispatcher keys on `multiProcessorCount==304` (shared by MI300X/MI325X) + `gfx942` and picks a broken hsaco. Pin to a known-good aiter / route around it.
- Building upstream Dao-AILab flash-attention CK backend on ROCm can fail with `getCurrentHIPStream` vs `getCurrentCUDAStream` — use the ROCm fork / aiter, not upstream FA's CK path.
- AITER batch-prefill via CK FMHA historically limited to **page sizes 1, 16, 1024**.

---

## 5. GQA, varlen / packed, and head layout

**GQA (grouped-query attention).** `H_q` query heads share `H_kv` KV heads (`g = H_q/H_kv`). In prefill, the efficient layout loads **one KV head once** and reuses it for all `g` query heads — pack the `g` query heads into the `BLOCK_M` dimension so a single K/V LDS load feeds `g` QKᵀ MFMAs. Map `program_id(1)` to `(batch, kv_head)` and loop/vectorize over the group. This raises arithmetic intensity (more MFMA per byte of K/V) — exactly what you want on a compute-bound prefill.

**Varlen / packed (the serving case).** Requests have different lengths; they are concatenated into one packed tensor with a `cu_seqlens` prefix-sum array (`[0, len0, len0+len1, …]`). The kernel:
- grid over `(num_q_blocks_total, H)` where blocks are assigned per-sequence via `cu_seqlens`;
- each block looks up its sequence id, clamps the causal/KV range to that sequence;
- **no padding waste** — short and long requests share the launch.

This is what aiter MHA / ck_tile batch-prefill implement; it's the default path in sglang/vllm prefill on ROCm.

---

## 6. Tuning-knob table (prefill FMHA on MI300X)

| Knob | Typical values | Effect | MI300X guidance |
|---|---|---|---|
| `BLOCK_M` (Q tile) | 64, 128, 256 | larger ⇒ more MFMA reuse, more VGPR | 128 is the sweet spot for d=128 bf16; 256 only if VGPR allows |
| `BLOCK_N` (KV tile) | 32, 64, 128 | larger ⇒ fewer softmax rescales, more LDS | 64-128; bounded by 64 KiB LDS (K+V tiles + scratch) |
| `BLOCK_DMODEL` (head dim) | 64, 128, (192/256) | fixed by model | d=128 is the tuned default; 192 (MLA) uses hdim%32 path |
| `num_warps` (Triton) | 4, 8 | wavefronts per WG; 1 warp = 64 lanes | **4** for BLOCK_M=128; 8 for 256 |
| `num_stages` (Triton) | 1, 2 | SW pipeline depth (LDS double-buffer) | 2 to overlap `global_load` K/V with MFMA; costs LDS |
| `waves_per_eu` | 1, 2, 3, 4 | occupancy hint (VGPR pressure) | start 2; raise to hide latency if VGPR/LDS allow, lower if spilling |
| `matrix_instr_nonkdim` (Triton ROCm) | 16, 32 | pick mfma 16x16 vs 32x32 | 16 for small tiles/causal diag; 32 for big square tiles |
| `kpack` | 1, 2 | K packing for mfma feed | 2 improves QKᵀ K-dim throughput at d=128 |
| ck_tile pipeline | `qr_ks_vs` / `qr_async` | LDS streaming vs async overlap | `qr_async` for long N to hide HBM latency |
| causal block-skip | on | skip masked KV blocks | always on for causal |
| fp8 quant mode | tensor-wise / block-scale | accuracy vs speed | block-scale for outlier-heavy LLMs |

**Occupancy rule of thumb:** on CDNA3 an FMHA WG with `BLOCK_M=128, d=128, num_stages=2` uses ~half the 64 KiB LDS and a large VGPR slice → ~2 waves/EU. If the Triton dump shows VGPR spills, drop `BLOCK_M` to 64 or `num_stages` to 1 before anything else.

---

## 7. FP8 prefill on CDNA3 — the `fnuz` trap

MI300X (CDNA3) FP8 is the **`fnuz`** dialect: *finite, NaN, unsigned-zero* — no `-0`, no `inf`. It is **not** OCP-standard FP8 (which MI325/MI350/MI355X use). `e4m3fnuz`/`e5m2fnuz` share the bit layout with OCP `e4m3`/`e5m2` but the **exponent bias differs by one** → mis-reading the dialect produces values off by exactly **2×**. Implications:
- Use `torch.float8_e4m3fnuz` (not `e4m3fn`) for activations/KV on MI300X.
- The QKᵀ uses `v_mfma_f32_16x16x32_fp8` (K-dim 32, double the bf16 K). Accumulate f32.
- Prefer **block-scale** (per-128-block) quantization for LLM activations with outliers; tensor-wise is faster but less accurate.
- ck_tile fwd supports both **dynamic tensor-wise** and **block-scale** fp8; aiter MHA exposes the fp8 path. fp8 prefill can approach the **2615 TFLOP/s** matrix ceiling (~2× bf16) when softmax/exp doesn't bottleneck.

---

## 8. Why FA-3's Hopper tricks don't port (and the CDNA3 equivalents)

| FA-3 (Hopper) technique | Ports to MI300X? | CDNA3 equivalent |
|---|---|---|
| Warp-specialization (producer/consumer warpgroups) | **No** (no `wgmma` async warpgroups) | software-pipeline `buffer_load` (K/V) vs `v_mfma`; `qr_async` pipeline |
| TMA bulk async copy | No (no TMA) | `buffer_load` + LDS double-buffer (`num_stages=2`) |
| GEMM-softmax 2-stage ping-pong (overlap exp under WGMMA) | Partially | interleave `v_exp_f32` with `v_mfma` issue; rely on 4 SIMD/CU + multiple waves to hide exp |
| FP8 incoherent processing (Hadamard) | Concept ports | block-scale quant + (optional) Hadamard pre-rotate; `fnuz` dialect |
| Occupancy via async | Differently | `waves_per_eu`, LDS budget, VGPR |

The net: on MI300X you get FA-2-class *algorithm* + CDNA3 MFMA throughput, and the "FA-3" wins come from **fp8 matrix cores (2615 TFLOP/s)** and **async `buffer_load` pipelining**, not from warpgroup specialization.

---

## 9. Practical checklist for writing a high-quality MI300X prefill kernel

1. **Tile for MFMA first:** `BLOCK_M=128, BLOCK_N=64, d=128`, f32 accumulate, `v_mfma_f32_16x16x16`.
2. **Online softmax with `exp2`**, fold `log2(e)` into the scale; keep `m_i/l_i/acc` in VGPR.
3. **Causal block-skip** (`hi = (start_m+1)*BLOCK_M`) + element mask only on the diagonal block.
4. **Pre-transpose K** for coalesced contraction; `buffer_load` K/V.
5. **GQA:** pack `g` query heads into `BLOCK_M`, load each KV head once.
6. **Varlen** via `cu_seqlens`; never pad.
7. **Double-buffer** K/V in LDS (`num_stages=2`) staying under 64 KiB.
8. **Autotune** `BLOCK_M/N, num_warps, waves_per_eu, kpack, matrix_instr_nonkdim`; verify no VGPR spills in the ISA dump.
9. **FP8:** `e4m3fnuz`, block-scale, `v_mfma_*_fp8`.
10. **Validate against aiter MHA / ck_tile** — if you're not within ~10-15%, dispatch to aiter instead.

---

## Sources

- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (NeurIPS 2024) — https://arxiv.org/abs/2407.08608
- FlashAttention-2 (Tri Dao, 2023) — https://arxiv.org/abs/2307.08691
- ROCm Blog — "From Theory to Kernel: Implement FlashAttention-v2 with CK-Tile" — https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
- ROCm/composable_kernel `ck_tile` FMHA examples + CHANGELOG — https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
- ROCm/aiter (AI Tensor Engine for ROCm) — https://github.com/ROCm/aiter
- AITER: AI Tensor Engine For ROCm (ROCm Blog) — https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
- AMD Instinct MI300X architecture (Hot Chips 2024) — https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith(MI300X).v1.Final.20240817.pdf
- vLLM V1 performance optimization on ROCm (AITER attention backend) — https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
- aiter issue #3139 — fmha_v3 MI300 kernel hang at large prefill (gfx942 dispatch bug) — https://github.com/ROCm/aiter/issues/3139
