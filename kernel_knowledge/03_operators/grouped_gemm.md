# Grouped GEMM on AMD MI300X (CDNA3 / gfx942)

> Scope: **grouped GEMM** — a batch of independent GEMMs with **variable per-group shapes**, fused into
> a single kernel launch. This is the compute backbone of **MoE expert FFNs**: after routing, each
> expert `e` does `Y_e = X_e · W_e` where `X_e` has a *different, data-dependent* row count `M_e`
> (tokens routed to that expert). AMD-only (gfx942); see `gemm.md` for the underlying tiled-MFMA math
> and `moe.md` for the full MoE pipeline that wraps this.

The defining property: **N and K are uniform across groups (the FFN weight shape is fixed), but M varies
per group and is unknown until routing runs.** That ruling-out of static shapes is exactly why batched
GEMM (`batched_gemm.md`, uniform M) is the wrong tool and a per-expert loop of GEMMs is too launch-bound.

---

## 1. The problem & why a single kernel

For `E` experts, top-`k` routing, `T` tokens, hidden `H`, intermediate `I`:

- Total rows processed = `T·k` (each token goes to `k` experts), partitioned into `E` groups of size
  `M_e` with `sum_e M_e = T·k`. `M_e` is **skewed and dynamic** — a hot expert may get 10× the tokens
  of a cold one, and some experts may get **zero**.
- Two GEMMs per expert in a gated MLP: up/gate `[M_e,H]·[H,2I]` then down `[M_e,I]·[I,H]`.

| Naive approach | Cost on MI300X |
|---|---|
| Python loop of `E` separate hipBLASLt calls | `E` launches (E=8..256) → launch-bound; tiny skinny GEMMs each leave most of 304 CUs idle; no cross-expert load balance |
| Pad every group to `max(M_e)` + batched GEMM | wastes compute on padding; with skew, `max` ≫ mean → 2–10× waste |
| **Grouped GEMM (one launch)** | one launch; tiles from all experts share the 304-CU pool; load balance is implicit in the tile schedule |

So grouped GEMM = "do all `E` variable-M GEMMs in one launch, with a tile scheduler that maps a flat
tile index to `(expert, m_tile, n_tile)` and balances skew across CUs."

---

## 2. Segment offsets — the data layout

Tokens are first **sorted/permuted by expert** (see `moe.md` MoE-align-&-sort) so each expert's rows are
contiguous. Grouped GEMM then consumes three small device arrays:

```
X_sorted : [T*k, H]          # activations, rows grouped contiguously by expert
group_sizes / offsets:
    m_sizes[e]   = M_e                          # rows for expert e
    m_offsets[e] = prefix_sum(M_e) (cumsum)     # start row of expert e in X_sorted
W        : [E, H, N]         # per-expert weights (3D), or pointer array group_b_ptrs[e]
Y_sorted : [T*k, N]          # output, same row layout as X_sorted
```

Two API shapes in the wild:
- **2D + offsets** (`torch._grouped_mm`, DeepGEMM `m_grouped`): A packed `[sum M_e, K]`, a single B (or
  per-group B), `offs` = int32 cumulative row offsets `[M_0, M_0+M_1, ...]`. Constraint of the native
  torch 2D path: B shared across groups; truly per-expert B needs the 3D variant.
- **pointer arrays** (Triton tutorial style): device tensors `group_a_ptrs[e]`, `group_b_ptrs[e]`,
  `group_c_ptrs[e]`, plus per-group `[gm,gn,gk]` sizes and `[lda,ldb,ldc]` leading dims.

**Alignment matters.** Each expert's segment is padded up to a multiple of `BLOCK_M` (the MoE-align step)
so no tile straddles two experts — a tile belongs to exactly one expert's weight. Padding rows carry a
sentinel token id and are masked in the epilogue.

---

## 3. Grouped-GEMM kernel logic — the two scheduling styles

### Style A: device-side strided scheduler (Triton tutorial / fixed-grid)

Launch a fixed number of workgroups (`NUM_SM`, e.g. 304). Each strides through the global tile space; a
running `last_problem_end` finds which group the current `tile_idx` falls into.

```python
@triton.autotune(configs=[
    triton.Config({'BLOCK_M':128,'BLOCK_N':128,'BLOCK_K':32,'NUM_SM':304}),  # gfx942: 1 wg/CU
    triton.Config({'BLOCK_M':64, 'BLOCK_N':64, 'BLOCK_K':64,'NUM_SM':304}),
], key=['group_size'])
@triton.jit
def grouped_matmul_kernel(group_a_ptrs, group_b_ptrs, group_c_ptrs,
                          group_gemm_sizes, g_lds, group_size,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr, NUM_SM: tl.constexpr):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):                              # walk groups (experts)
        gm = tl.load(group_gemm_sizes + g*3)                # M_e (variable!)
        gn = tl.load(group_gemm_sizes + g*3 + 1)            # N (uniform)
        gk = tl.load(group_gemm_sizes + g*3 + 2)            # K (uniform)
        num_m_tiles = tl.cdiv(gm, BLOCK_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles
        # does this workgroup's current tile fall inside expert g?
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            lda = tl.load(g_lds + g*3); ldb = tl.load(g_lds + g*3+1); ldc = tl.load(g_lds + g*3+2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            # flat tile -> (m,n) tile coords within this expert
            t = tile_idx - last_problem_end
            tm = t // num_n_tiles
            tn = t %  num_n_tiles
            offs_am = tm*BLOCK_M + tl.arange(0, BLOCK_M)
            offs_bn = tn*BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k  = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_am[:,None]*lda + offs_k[None,:]
            b_ptrs = b_ptr + offs_k[:,None]*ldb + offs_bn[None,:]
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(gk, BLOCK_K)):       # MFMA inner loop, fp32 accumulate
                a = tl.load(a_ptrs); b = tl.load(b_ptrs)
                acc += tl.dot(a, b)                          # -> v_mfma_f32_16x16x16_*
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * ldb
            c = acc.to(tl.float16)
            c_ptrs = c_ptr + ldc*offs_am[:,None] + offs_bn[None,:]
            tl.store(c_ptrs, c)
            tile_idx += NUM_SM                               # stride to next tile this wg owns
        last_problem_end += num_tiles
```

Load balance is implicit: the `tile_idx += NUM_SM` stride spreads all experts' tiles round-robin over
the fixed workgroup pool, so a hot expert's many tiles are shared across CUs instead of serializing.

### Style B: persistent, cache-aware schedule (production / DeepSeek-V3 style)

Precompute a flat tile→(expert,m_tile,n_tile) **schedule on host/device once**, launch exactly `#CU`
persistent workgroups, and order the schedule so consecutive tiles **reuse the same expert weight panel**
(temporal locality → L2/Infinity-Cache hits). On every `expert_id` change the workgroup re-points its
weight base (on CUDA: rewrites a TMA descriptor; on gfx942: recomputes the `W[e]` base pointer + reloads
the weight tile into LDS). PyTorch's persistent cache-aware grouped GEMM reports up to 2.62× over the
PyTorch loop on DeepSeek-V3.

> **Critical correctness gotcha (block-scheduled grouped GEMM):** the schedule's `BLOCK_M` and the
> kernel's `BLOCK_M` **must be identical**. If you autotune `BLOCK_M` while the schedule was built with
> `BLOCK_M=64`, picking 128 makes tiles overlap and picking 32 skips rows — output looks plausible but
> ~30–45% of values are silently wrong. **Fix: pin `BLOCK_M`, do not autotune it** once a schedule exists.

---

## 4. Load balancing across groups (the skew problem)

Routing is highly skewed, *especially in decode* (small token count, a few hot experts). Implications:

| Symptom | Cause | Mitigation on MI300X |
|---|---|---|
| Long tail / a few CUs run long | one expert got most tokens; its tiles dominate | strided/persistent schedule shares tiles across CUs; split-K within a hot expert's GEMM |
| Many empty experts | top-k concentration / small batch | `M_e=0` → `num_tiles=0`, the scheduler skips it for free; no wasted launch |
| Tiny `M_e` (1–2 rows) | cold experts | small `BLOCK_M` (16/32) so the tile isn't 90% padding |
| Tile/schedule mismatch | autotuned BLOCK_M vs prebuilt schedule | pin BLOCK_M (§3 gotcha) |

Decode regime: `M_e` averages `T·k/E` which for small batch and E=256 is often **< 1** per expert → most
experts empty, a few have a handful of rows. This is brutally memory-bound on weight loads (read a full
expert weight to multiply a couple of rows), so decode MoE favors: small BLOCK_M, fp8/fp4 expert weights
to cut bytes, and the **batched/masked** layout (see `moe.md`, DeepEP-low-latency / BatchedTriton path)
that fixes a per-expert token capacity so the schedule is static.

---

## 5. When grouped vs batched vs loop

| Use | Condition |
|---|---|
| **Grouped GEMM** | variable M per group (MoE experts), uniform N,K; want one launch + implicit balance |
| **Batched GEMM** (`batched_gemm.md`) | *uniform* M,N,K across batch (multi-head proj, fixed-capacity MoE); strided/array batched |
| **Loop of GEMMs** | very few large groups where launch overhead is negligible and each fills the chip |
| **Masked/padded batched** | dynamic M but you fix a per-expert capacity → static shapes, drop tokens over capacity (DeepEP-LL) |

---

## 6. Backend ladder (MI300X grouped GEMM)

| Tier | Mechanism | Edit? |
|---|---|---|
| A — backend select | bench available grouped-GEMM impls on the routed shapes | no |
| B — per-backend tune | autotune tiles/instance within the winner | no |
| C — code rewrite | edit the Triton/CK grouped kernel (persistent schedule, fp8, fusion) | yes |
| D — quantization | fp8/fp4 expert weights | flag → accuracy gate |

| Backend | Notes (gfx942) |
|---|---|
| **aiter grouped/fused MoE GEMM** | AMD's tuned path; block-scale fp8 grouped GEMM; first try for MoE expert compute |
| **CK / ck_tile grouped GEMM** | `DeviceGroupedGemm*_Xdl` + ck_tile fused-MoE; tunable instance (tile, pipeline, MFMA 16/32); hand-asm inner loop → strong but rigid |
| **Triton grouped GEMM** | the editable path; the kernel in §3; autotune `BLOCK_M/N/K`, `matrix_instr_nonkdim=16`, `waves_per_eu`, `kpack`, `NUM_SM=304`; pin BLOCK_M if persistent |
| **DeepGEMM `m_grouped`** | fp8 block-scale (G128) grouped GEMM; contiguous + masked variants; ROCm port maturity varies |
| **hipBLASLt grouped ext** | grouped-GEMM extension; tune solution per (E,N,K) profile |

**Ranked first-moves:** prefill MoE: aiter fp8 grouped → CK → Triton. Decode MoE: aiter masked/batched
fp8 → Triton batched(masked) → CK.

**Triton Tier-B knobs:** same as `gemm.md` plus `NUM_SM` (= 304 to put one persistent wg per CU) and the
**pinned** `BLOCK_M` matching the align block_size used by MoE-align-&-sort (commonly 32 or 64 on gfx942).

---

## 7. Profiling checklist

- Confirm **all 304 CUs** active during the grouped launch (skew can leave CUs idle → check omniperf
  CU-occupancy timeline; if tail-heavy, raise NUM_SM stride sharing or split-K the hot expert).
- Verify **no tile straddles two experts** (alignment) — a correctness bug, not just perf.
- For persistent schedule: watch **L2/Infinity-Cache hit rate** — the whole point is weight reuse across
  consecutive same-expert tiles; a poor schedule order kills it.
- fp8 path: check the scale layout (block-128) is applied in the fp32 epilogue, and run an accuracy gate.
- Re-confirm the BLOCK_M ↔ schedule invariant after any autotune (the silent-wrong-output trap).

---

## Sources
- Triton Group GEMM tutorial (device-side strided scheduler, code): https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
- PyTorch: Accelerating MoEs with a Triton Persistent Cache-Aware Grouped GEMM Kernel (DeepSeek-V3, 2.62×): https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/
- Subhadip Mitra — Fused MoE Dispatch in Triton (block_id→(expert,offset), BLOCK_M pin gotcha): https://subhadipmitra.com/blog/2026/fused-moe-dispatch-triton/
- Ian Barber — Grouped GEMMs and MoE: https://ianbarber.blog/2025/02/11/grouped-gemms-and-moe/
- pytorch-labs/applied-ai Grouped GEMM (DeepWiki): https://deepwiki.com/pytorch-labs/applied-ai/3.3-grouped-gemm
- Composable Kernel ck_tile fused-MoE / grouped GEMM docs: https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/
- AITER: AI Tensor Engine for ROCm (block-scale fused MoE grouped GEMM, 3×): https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
