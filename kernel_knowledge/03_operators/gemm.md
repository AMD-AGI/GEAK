# Dense GEMM on AMD MI300X (CDNA3 / gfx942)

> Scope: dense matrix-multiply `C = A·B [+bias] [+act]` for LLM inference on **MI300X (CDNA3, gfx942)**.
> AMD-only. CUDA mentioned only as contrast. Numbers target gfx942; gfx950 (MI350/CDNA4) noted where
> it changes the answer. This is the single highest-`pct_gpu_time` op class in dense LLM inference
> (QKV/O proj, MLP up/gate/down, LM head) — getting it right is the largest Amdahl lever you have.

GEMM is **not** "just call a library and move on." On MI300X the gap between theoretical and achieved
FLOPs is large: the chip has ~1.5× H100's peak bf16/fp8 FLOP rate but sustained efficiency on real
shapes is often **45–55%** of peak (vs >90% on a mature H100 stack), because the kernel/compiler stack
is younger and because power/clock management throttles under sustained MFMA load. That gap *is* the
optimization opportunity: a fixed-shape library GEMM is one of the most tunable things on the chip.

---

## 1. Hardware budget you are spending (gfx942)

| Resource | MI300X value | GEMM relevance |
|---|---|---|
| Compute Units | **304** (8 XCD × 38 active; 40 physical) | grid must cover ≥304 tiles to fill the chip |
| Wavefront | **64 lanes** (NOT 32) | MFMA operand layouts, `__shfl`/ballot are 64-wide |
| VGPR / EU (SIMD) | **512** (alloc granularity **16**) | accumulator tile + double-buffer must fit; spills are death |
| SGPR | 102–106 usable | pointer/scalar bookkeeping |
| LDS / CU | **64 KB** (32 banks × 4 B) | A/B staging tiles + double buffer |
| L2 (per XCD) | 4 MB; **256 MB** aggregate Infinity Cache | weight reuse across tiles, persistent-kernel locality |
| HBM3 | **192 GB @ 5.3 TB/s** | decode (skinny-M) is bandwidth-bound on weights |
| Peak bf16/fp16 MFMA | **1307 TFLOPS** (2048 FLOP/CU/clk) | prefill ceiling |
| Peak fp8/int8 MFMA | **2614 TFLOPS** (4096 FLOP/CU/clk) | 2× fp16; the reason fp8 wins prefill |
| Peak fp6/fp4 (gfx950) | up to ~2× fp8 again on CDNA4 | not on gfx942; design for it portably |
| Engine clock | ~2.1 GHz (throttles under load) | sustained ≠ peak |

**Roofline rule of thumb.** Arithmetic intensity to be compute-bound on bf16:
`AI* = 1307e12 / 5.3e12 ≈ 247 FLOP/byte`. A GEMM with `M,N,K` does `2·M·N·K` FLOP and moves
`2·(M·K + K·N + M·N)` bytes (bf16). Prefill (large M,N,K) sits far right of the knee → **compute-bound**.
Decode (M = batch, often 1–256; weights huge) sits left → **memory-bound on weight load** → the goal
shifts from MFMA utilization to *bytes of weight read per token* (favoring fp8/fp4 weights and split-K).

---

## 2. The tiled-MFMA algorithm (the thing every backend implements)

GEMM is a 3-level tiling of the iteration space onto the (grid → workgroup → wave → MFMA) hierarchy.

```
GLOBAL:    C[M,N] = sum_k A[M,k] B[k,N]
  grid tile:    each workgroup owns a BLOCK_M × BLOCK_N output tile         (covers CUs)
    wave tile:    each of the 4 waves owns a WAVE_M × WAVE_N sub-tile        (4 SIMDs/CU)
      mfma tile:    each MFMA does 16×16×{16|32} (or 32×32×{8|16})           (Matrix Core)
  K-loop:    stream BLOCK_K slabs of A,B from HBM → LDS → VGPR → MFMA, accumulate in fp32 VGPRs
```

Pseudocode of the canonical "block GEMM on MI300" (matches CK-tile / hipBLASLt structure):

```text
for k0 in range(0, K, BLOCK_K):                       # main K loop (software-pipelined)
    gmem -> LDS:  load A_tile[BLOCK_M, BLOCK_K], B_tile[BLOCK_K, BLOCK_N]   (vectorized, coalesced)
    cp.async / buffer_load_dwordx4 into the *next* LDS buffer  (double buffer)
    __syncthreads()
    for kk in range(0, BLOCK_K, MFMA_K):              # inner MFMA loop, fully unrolled
        a_frag = LDS -> VGPR  (ds_read, swizzled to MFMA lane layout)
        b_frag = LDS -> VGPR
        acc   += v_mfma_f32_16x16x16_f16(a_frag, b_frag, acc)   # fp32 accumulate
    swap LDS buffers
epilogue: acc(fp32) -> [scale][+bias][act][cast to out dtype] -> coalesced global store
```

The four things that make or break it on CDNA3:
1. **MFMA instruction choice** (16×16 vs 32×32) — see §3.
2. **LDS double-buffering** to hide HBM latency behind MFMA (the `num_stages` knob).
3. **Coalesced 128-bit `buffer_load`** (`global_load_dwordx4`) gmem→reg, swizzled ds_write→LDS with no
   bank conflicts (stride 128 B = no conflict; stride 4 B = 32-way conflict).
4. **fp32 accumulators staying in VGPRs** without spilling → constrains tile size vs the 512-VGPR/EU budget.

---

## 3. MFMA inner loop — the Matrix Core ISA (gfx942)

CDNA3 Matrix Cores expose `v_mfma_*` ops; each consumes A/B fragments from VGPRs and accumulates into
fp32 VGPRs. Per-CU per-cycle throughput is fixed; choice is driven by **shape, occupancy, register pressure**.

| Instruction (intrinsic) | dtype | tile M×N×K | FLOP/instr | cycles | FLOP/CU/clk | acc VGPRs/lane |
|---|---|---|---|---|---|---|
| `v_mfma_f32_16x16x16_f16` | fp16→fp32 | 16×16×16 | 8192 | 16 | 2048 | 4 |
| `v_mfma_f32_32x32x8_f16` | fp16→fp32 | 32×32×8 | 16384 | 32 | 2048 | 16 |
| `v_mfma_f32_16x16x16_bf16` | bf16→fp32 | 16×16×16 | 8192 | 16 | 2048 | 4 |
| `v_mfma_f32_16x16x32_fp8_fp8` | fp8→fp32 | 16×16×32 | 16384 | 16 | **4096** | 4 |
| `v_mfma_f32_32x32x16_fp8_fp8` | fp8→fp32 | 32×32×16 | 32768 | 32 | **4096** | 16 |
| `v_mfma_i32_16x16x32_i8` | int8→int32 | 16×16×32 | — | 16 | 4096 | 4 |

**The decisive rule on MI300X: prefer 16×16 over 32×32, even for large GEMMs.** Two reasons:
- **Register pressure / double-buffering.** 32×32×16 fp8 needs 16 acc VGPRs/lane for the master frag
  + 16 for the block frag → no headroom for double buffering. 16×16×32 needs only **4 VGPRs/lane**,
  leaving registers free to double-buffer the LDS→VGPR loads (the real latency hider).
- **Occupancy on small problems.** Example K=7168, M=1024, N=512: with 32×32 tiles computing 128×128
  per block you launch ~32 blocks → 16/304 CUs busy. With 16×16 computing 64×64 per block you get
  64/304 CUs → **4× occupancy**. The ROCm tuning guide and CK both confirm `mfma_16x16` typically wins.
- 32×32 only earns its keep when M,N are both large *and* K is short (better arithmetic intensity per
  warp), and register pressure is not the binder.

In Triton this knob is `matrix_instr_nonkdim ∈ {16, 32}`; in CK it is the `MfmaInstr` of the instance.

Use the **ROCm Matrix Instruction Calculator** (`--detail-instruction` / `-d`, supports gfx942) to get
exact operand→lane register layouts before hand-writing HIP MFMA.

---

## 4. Annotated Triton GEMM for gfx942

A production-shaped Triton matmul with the AMD-specific autotune knobs. The core is the K-loop with
fp32 accumulation; the AMD wins are in the `Config` kwargs (`matrix_instr_nonkdim`, `waves_per_eu`,
`kpack`) and the `GROUP_M` L2-locality reordering.

```python
import triton, triton.language as tl

@triton.autotune(
    configs=[
        # MI300X sweet spots: 16x16 MFMA, 2-stage pipeline, 8 warps for big tiles
        triton.Config({'BLOCK_M':256,'BLOCK_N':256,'BLOCK_K':64,'GROUP_M':8,
                       'matrix_instr_nonkdim':16,'waves_per_eu':2,'kpack':2},
                       num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M':128,'BLOCK_N':256,'BLOCK_K':64,'GROUP_M':8,
                       'matrix_instr_nonkdim':16,'waves_per_eu':2,'kpack':2},
                       num_warps=8, num_stages=2),
        # decode / skinny-M: small BLOCK_M, SPLIT_K to recover parallelism
        triton.Config({'BLOCK_M':16,'BLOCK_N':128,'BLOCK_K':128,'GROUP_M':1,'SPLIT_K':8,
                       'matrix_instr_nonkdim':16,'waves_per_eu':4,'kpack':2},
                       num_warps=4, num_stages=2),
    ],
    key=['M','N','K'],
)
@triton.jit
def gemm_kernel(A, B, C, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr = 1):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)                       # split-K reduction dimension
    # ---- L2-locality "grouped" launch order: process a GROUP_M-tall super-row of tiles ----
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None]*stride_am + offs_k[None, :]*stride_ak
    b_ptrs = B + offs_k[:, None]*stride_bk + offs_n[None, :]*stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)     # fp32 accumulate in VGPRs
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        a = tl.load(a_ptrs)                                   # coalesced HBM->reg, staged via LDS
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)                                   # lowers to v_mfma_f32_16x16x16_*
        a_ptrs += BLOCK_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_K * SPLIT_K * stride_bk

    c = acc.to(C.dtype.element_ty)                            # epilogue: cast (+bias/+act fused here)
    c_ptrs = C + offs_m[:, None]*stride_cm + offs_n[None, :]*stride_cn
    if SPLIT_K == 1:
        tl.store(c_ptrs, c)
    else:
        tl.atomic_add(c_ptrs, c)                             # split-K partials reduce in global mem
```

**AMD-specific Triton knobs (the Tier-B search space):**

| Knob | Range | Effect on gfx942 |
|---|---|---|
| `BLOCK_M/N/K` | 16–256 | tile vs occupancy vs LDS/VGPR; 256×256×64 typical prefill |
| `matrix_instr_nonkdim` | **16**, 32 | MFMA size; **16 wins** (see §3) |
| `num_warps` | 4, 8 | 8 for big tiles, 4 for skinny-M |
| `num_stages` | 0,1,**2** | LDS pipeline depth; 2 for single-GEMM, 0 if no benefit / register-bound |
| `waves_per_eu` | 1–4 | nudge compiler to lower VGPR to hit a higher occupancy band |
| `kpack` | 1, **2** | pack 2 K-elements per ds_read → fewer LDS instrs, better MFMA feed |
| `GROUP_M` | 1–8 | reorder tile launch for L2 weight reuse (huge on 256 MB Infinity Cache) |
| `SPLIT_K` | 1–16 | recover parallelism for skinny-M decode (atomic-add reduce) |
| `OPTIMIZE_EPILOGUE` | 0/1 | keep MFMA-layout store (skip layout convert); minor net effect |

> Triton-on-ROCm gotcha: `num_stages` semantics differ from CUDA. For a single GEMM, `num_stages=2`
> (or even `0` to let the backend choose) typically beats large values; the LLVM/ROCm pipeliner does
> not benefit from deep `cp.async`-style staging the way CUDA does.

---

## 5. Shape regimes — prefill vs decode

LLM inference has two utterly different GEMM regimes; tune them separately.

| Regime | M (rows) | N,K | Bottleneck | Strategy | Best tiles |
|---|---|---|---|---|---|
| **Prefill** (large-M) | seq_len·batch (1k–32k) | model dims (4k–28k) | **compute** (MFMA) | maximize MFMA util, big tiles, L2 reuse, fp8 | 256×256×64, 16×16 MFMA, GROUP_M=8 |
| **Decode** (skinny-M) | batch (1–256) | model dims | **HBM bandwidth** on weights | minimize weight bytes/token, split-K, fp8/fp4 weights, persistent | 16–64 × 128 ×128, SPLIT_K=4–16 |
| **LM head** | M·vocab (≥128k N) | huge N | compute + store | split-N, fuse top-k/argmax in epilogue | big N tiles |

**Decode is the trap.** With M=1..32, a 256×256 tile wastes ~99% of the M dimension; you launch a
handful of workgroups and leave 290/304 CUs idle, *and* you re-read the entire weight matrix once with
no row reuse → pure bandwidth. Fixes, in order: (1) **split-K** to spread the single output tile's K
reduction across many CUs; (2) small BLOCK_M (16/32) to stop wasting MFMA lanes; (3) **fp8/fp4 weights**
to halve/quarter the bytes read (the actual win on decode); (4) hipBLASLt TensileLite custom skinny
kernels (1.6–2.6× on shapes like [3,14400,64]: 23.2µs → 8.8µs).

---

## 6. Split-K vs Stream-K vs Persistent

| Scheme | What | When on MI300X |
|---|---|---|
| **Data-parallel** (default) | 1 workgroup per output tile, full K loop | prefill where #tiles ≥ 304 |
| **Split-K** | partition K across G workgroups, atomic/2-pass reduce | skinny-M / few output tiles (decode); G=4–16 |
| **Stream-K** | tiles share K work via a global work queue → near-perfect load balance for any M,N,K | irregular shapes / when split-K's fixed G mis-balances; available in CK & rocBLAS Tensile |
| **Persistent** | launch exactly #CU (304) workgroups; each loops over many tiles | amortize launch overhead, keep weights hot in L2/registers; pairs with GROUP_M locality |

Persistent + grouped scheduling is the standard high-throughput layout: 304 long-lived workgroups
stride through the tile space in an L2-friendly order so an expert/weight panel loaded once is reused
across the tiles that need it. (Same idea as the persistent grouped-GEMM in `grouped_gemm.md`.)

---

## 7. Epilogue fusion

The epilogue runs on the fp32 accumulator still in VGPRs — fusing here is free bandwidth (the output
is written once, with the extra op folded in). Fuse aggressively to delete neighbor kernels:

| Fusion | Use | Backend support |
|---|---|---|
| `+ bias` | every Linear with bias | hipBLASLt epilogue, aiter, Triton, CK |
| `+ act` (GELU/SiLU/ReLU) | MLP up/gate | hipBLASLt(GELU), aiter, Triton, CK |
| `SwiGLU` (gate·SiLU(up)) | gated MLP | aiter fused, Triton (compute both halves, fuse) |
| `+ residual add` | post-proj | Triton/aiter; folds an elementwise kernel away |
| **per-tensor/row fp8 quant** | output → next fp8 GEMM | aiter `gemm+quant`, Triton scaled epilogue |
| `+ RMSNorm` (next layer) | rarely; usually its own kernel | aiter fused norm+quant more common |

Rule: a fused epilogue that **collapses a separate elementwise/quant kernel** is almost always a net
e2e win even if it slightly lowers the GEMM's own MFMA utilization — you delete a whole HBM round-trip.

---

## 8. dtype variants

| dtype | Peak TFLOPS | Notes (gfx942) |
|---|---|---|
| **bf16** | 1307 | default prefill; numerically robust; `v_mfma_*_bf16` |
| **fp16** | 1307 | same rate; watch overflow in long-K accum (fp32 acc protects) |
| **fp8 e4m3 / e5m2** | **2614** | 2× throughput + half weight bytes → wins prefill *and* decode; needs scaling (per-tensor/row/block-128); accuracy gate required |
| **int8** | 2614 | SmoothQuant-style; CK `DeviceBatchedGemmMultiD_Xdl` path |
| **mxfp4 / fp4** | ~2× fp8 (gfx950) | **CDNA4 (MI350) only** for native MFMA; on gfx942 dequant-to-bf16/fp8 path |
| **fp6** | gfx950 | not native on gfx942 |

fp8 GEMM on MI300X needs scale management. Common formats: **per-tensor** (one scale, cheapest),
**per-row/token** (A) + **per-channel** (B), and **block-scale G(128)** (DeepSeek-style 128-elem blocks,
best accuracy). The scale multiply lives in the fp32 epilogue. Block-scale is the format DeepGEMM /
aiter block-scale MoE use; it costs a few % vs per-tensor but recovers most of the accuracy.

---

## 9. Backend ladder (cheapest-first; the op unittest/oracle is the judge)

| Tier | Mechanism | Source edit? | Parity |
|---|---|---|---|
| A — backend select | run each backend on the exact shape, keep fastest-correct | no | same-dtype safe* |
| B — per-backend tune | autotune *within* the chosen backend | no | safe* |
| C — code rewrite | edit the Triton/HIP/CK kernel (split-K, persistent, fusion) | yes | safe* |
| D — quantization | change dtype (fp8/fp4) | flag | **breaks byte-parity → accuracy gate** |

`*` bf16 reduction order differs across backends → not byte-identical; re-check e2e parity.

**Backend menu (MI300X dense GEMM):**

| Backend | Strength | First-try regime |
|---|---|---|
| **PyTorch TunableOp** | runtime auto-picks rocBLAS+hipBLASLt per shape, caches to CSV; pure env, parity-safe | the easiest first move, always |
| **hipBLASLt / Tensile** | sglang/vLLM default; strong *when shape is in tuning DB*; TensileLite generates custom skinny kernels (1.6–2.6×) | prefill large-M; decode after TensileLite tune |
| **rocBLAS** | alternate library; sometimes wins odd/skinny shapes hipBLASLt mistunes | fallback compare |
| **aiter GEMM** | AMD fused GEMM (+bias/act/quant epilogue); often wins decode/skinny + fuses | decode/skinny + fusion |
| **CK / ck_tile** | template GEMM, tunable by instance (tile, pipeline v1/v2, MFMA, pad); stream-K | when you need a fused/odd path or stream-K balance |
| **Triton** | editable, autotunable; path to split-K/persistent/epilogue-fusion rewrites | when fusion collapses a neighbor or library mistunes |

**Ranked first-moves:**
- **Prefill (large-M):** TunableOp → hipBLASLt(tuned DB) → CK → Triton → aiter, then fp8 (Tier D).
- **Decode (M=batch):** aiter → TunableOp → hipBLASLt(TensileLite) → Triton(split-K), then fp8/fp4.

**Tier-B knobs per backend:**
- hipBLASLt: enumerate solution indices for exact (M,N,K,dtype,transpose,bias); pin best via
  `HIPBLASLT_TUNING_FILE`; offline `hipblaslt-bench`; TensileLite for skinny decode shapes.
- TunableOp: `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 PYTORCH_TUNABLEOP_FILENAME=<csv>`,
  warmup to populate, ship with `TUNING=0`.
- CK: pick instance (tile, pipeline v1/v2, padded vs not, MFMA 16 vs 32).
- Triton: the §4 knob table.

> Coverage caveat: aiter's tuned coverage skews to newer parts (CDNA4); on gfx942 some shapes fall back
> to generic Triton, which is several× slower than a tuned kernel. Always Tier-A bench — do not assume
> "aiter is installed" means "aiter is fast for this shape." Watch hipBLASLt's
> `not found tuned config ... using default config` log line: that means you're on the slow generic path.

---

## 10. Validation & profiling checklist

- **Correctness:** compare against an immutable bf16/fp32 oracle; fp8 needs an *accuracy* gate (rel-err
  / task eval), not byte parity.
- **`rocprofv2` / omniperf:** check MFMA-busy %, VGPR/LDS occupancy band, vL1D/L2 hit rate, no VGPR spills.
- **Occupancy band:** if VGPR usage is just over a 16-VGPR boundary, nudge `waves_per_eu` to drop into a
  higher band (e.g. 170→176 VGPR caps you at 2 waves/EU; trimming under 170 → 3 waves).
- **Power/clock:** sustained MFMA throttles clocks; benchmark with realistic duty cycle, not a 1-shot.
- Beware NVIDIA-derived **wave-specialization** ports: AMD's static register allocation makes producer
  waves waste registers; **8-wave ping-pong / 4-wave interleave** (HipKittens) outperform on CDNA3/4.

---

## Sources
- AMD Instinct MI300 (CDNA3) ISA Reference Guide (Aug 2025): https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- Matrix Core Programming on AMD CDNA3 and CDNA4 (ROCm Blogs): https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AMD MI300X workload optimization (ROCm docs, MFMA 16×16 vs 32×32, occupancy/VGPR): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Customizing GEMM with hipBLASLt TensileLite tuning (skinny decode 1.6–2.6×): https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html
- A Block GEMM on MI300 — Composable Kernel ck_tile docs: https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- Hands-On with CK-Tile: optimized GEMM on AMD GPUs (ROCm Blogs): https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html
- ROCm amd_matrix_instruction_calculator (per-instruction layouts/throughput): https://github.com/ROCm/amd_matrix_instruction_calculator
- HipKittens: Fast and Furious AMD Kernels (ping-pong/interleave vs wave-spec): https://arxiv.org/html/2511.08083v1
- Nscale: MI300X GEMM tuning up to 7.2× throughput/latency: https://www.nscale.com/blog/nscale-benchmarks-amd-mi300x-gpus-with-gemm-tuning-improves-throughput-and-latency-by-up-to-7-2x
- Triton matmul tutorial (GROUP_M L2 reorder, autotune): https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
