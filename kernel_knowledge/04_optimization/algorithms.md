# Algorithm-Level Optimization Catalog — AMD Instinct MI300X (CDNA3 / gfx942)

> Scope: AMD only. Target MI300X (gfx942, CDNA3); gfx950 (CDNA4, MI350/MI355) notes inline. This is the catalog of *algorithmic* (not micro-arch) techniques that move LLM-inference kernels toward peak on MI300X. For each: **what it is, when it helps, MI300X-specific tradeoffs**, and pseudocode for the load-bearing ones.
>
> The recurring CDNA3 facts that shape every choice:
> - **304 CUs = 8 XCDs × 38 CUs.** Grids should produce **≥1024 workgroups** to fill the machine and tolerate the round-robin XCD scheduler.
> - **512×32-bit registers/SIMD** = 256 VGPR + 256 AGPR at 1 wave/SIMD. MFMA accumulators consume **AGPRs**. Static register allocation means *producer-only* waves waste registers.
> - Best MFMA tile is **`mfma_16x16`** (`matrix_instr_nonkdim=16`).
> - **NVIDIA-style warp/wave specialization underperforms on CDNA3.** Prefer **8-wave ping-pong** or **4-wave interleave** (HipKittens, Nov 2025).
> - 256 KB LDS/CU, 32 MB shared L2 (per-XCD slices), ~5.3 TB/s HBM3.

---

## Quick decision table

| Technique | Primary win | Best regime on MI300X | Risk / cost |
|---|---|---|---|
| **Split-K** | fill CUs on skinny GEMM | small M or N, large K (decode) | atomic/reduction overhead |
| **Stream-K** | load-balance partial tiles across CUs | tall-skinny / odd shapes on multi-chiplet | fixup complexity |
| **Persistent kernel** | amortize launch + reuse L2/regs | many tiles, repeated launches | manual tile loop, scheduling |
| **Software pipelining (multi-stage)** | hide HBM latency behind MFMA | compute-bound GEMM/attention | LDS + register pressure |
| **Online softmax** | O(1) memory softmax, enables FA | any attention | numerical care |
| **FlashAttention** | no N² scores in HBM | prefill / long context | tiling, masking complexity |
| **Chunked prefill** | overlap prefill with decode, bound TTFT | mixed serving batches | scheduler logic |
| **Speculative decoding** | fewer target-model steps | low-batch latency-bound decode | verify GEMM shapes shift |
| **GEMM + epilogue fusion** | kill memory-bound neighbor kernels | bias/act/quant after matmul | epilogue register pressure |
| **Comm–compute overlap** | hide RCCL behind GEMM | tensor/expert parallel | stream/chunk orchestration |
| **Ping-pong / 4-wave interleave** | peak MFMA util on CDNA3 | hand-tuned GEMM/attention | hard to write; replaces wave-spec |

---

## 1. Split-K vs Stream-K

### Split-K
**What:** partition the K (contraction) dimension across multiple workgroups; each computes a partial `C` tile, then results are summed (atomics or a second reduction kernel).

**When it helps on MI300X:** decode-time GEMMs where M is tiny (1–256). A single tile of C = one workgroup → grid far below 1024 → most of 304 CUs idle. Split-K multiplies the grid by `SPLIT_K`, restoring occupancy.

**Tradeoff:** the partial-sum reduction costs HBM traffic + atomics. Too-large `SPLIT_K` makes the reduction dominate. Sweep `SPLIT_K ∈ {1,2,4,8,16}` per shape. Atomic-add accumulation needs fp32 accumulators for accuracy.

```python
# Split-K: grid = (ceil(M/BM)*ceil(N/BN), SPLIT_K)
pid_k = tl.program_id(1)
k_start = pid_k * (K // SPLIT_K)
acc = tl.zeros((BM, BN), tl.float32)
for k in range(k_start, k_start + K // SPLIT_K, BK):
    acc += tl.dot(load_a(k), load_b(k))          # mfma_16x16 accumulates in fp32/AGPR
if SPLIT_K == 1:
    store(c_ptr, acc)
else:
    tl.atomic_add(c_ptr, acc)                    # partial-sum across K splits
```

### Stream-K
**What:** instead of statically partitioning K per output tile, assign each workgroup a contiguous **slice of the total MAC work** (across all tiles), so every CU does ~equal work even when the tile count doesn't divide evenly by CU count. Partial tiles are reconciled with a "fixup" pass.

**When it helps on MI300X:** the multi-chiplet 304-CU layout makes static tiling produce **quantization waves** (a few CUs finish a leftover wave while the rest idle). Stream-K eliminates the tail. It is the leading distribution scheme for tall-skinny / awkward GEMM on MI300X. Stream-K++ (arXiv 2408.11417) adaptively chooses among basic Stream-K, data-parallel→one-batch, and two-batch→data-parallel schedules.

**Tradeoff:** the fixup/reduction logic is more complex than split-K; small uniform GEMMs that already fill the machine don't benefit. Use Stream-K when tile count is *not* a clean multiple of available CUs.

> **Rule:** skinny + clean K → Split-K. Awkward tile-count vs 304 CUs → Stream-K. Already ≥1024 balanced tiles → plain data-parallel.

---

## 2. Persistent kernels

**What:** launch exactly one workgroup per CU (≈304–608 total) that loops over output tiles via an internal work counter, rather than launching one workgroup per tile. The grid is "persistent."

**When it helps on MI300X:**
- Amortizes kernel launch + prologue across many tiles (matters for many small GEMMs / MoE expert loops).
- Keeps weights/scales resident in registers/LDS across tiles → **L2 reuse** improves, HBM re-reads drop.
- Natural host for Stream-K (the work counter *is* the Stream-K schedule) and for **ping-pong** scheduling.

**Tradeoff:** you hand-write the tile loop and the cache-aware tile ordering. The round-robin XCD scheduler means tile-to-XCD mapping affects L2 hit rate — order tiles to keep an XCD's tiles sharing operands.

```text
persistent_gemm():
  wg_id = blockIdx.x                       # one wg per CU, total ~= num_CUs
  for tile in tile_iter(wg_id, num_tiles): # strided / Stream-K assignment
      m, n = tile_to_mn(tile)
      acc = 0
      for k in range(0, K, BK):
          acc += mfma_16x16(A[m,k], B[k,n])
      epilogue_store(C[m,n], acc)          # fused bias/act/quant here (see fusion file)
```

---

## 3. Software pipelining / multi-stage prefetch (global → LDS → reg)

**What:** overlap memory movement of iteration *i+1* (and *i+2*) with the MFMA compute of iteration *i*, by staging operands through LDS double/triple buffers and registers.

```text
prologue:  load A0,B0 -> LDS buf0 ;  load A1,B1 -> LDS buf1
loop k:    issue MFMA on buf[k%2]           # compute current
           prefetch A[k+2],B[k+2] -> buf[(k)%2]   # load future (async/direct-to-LDS)
epilogue:  drain remaining MFMA
```

**When it helps:** compute-bound GEMM/attention where HBM/LDS latency would otherwise stall the MFMA units. CDNA3 supports **direct global→LDS loads** (skip the register round-trip) which is the key MI300X enabler.

**MI300X tradeoff — `num_stages`:** more stages = more LDS + registers = lower occupancy. The CDNA guidance is the opposite of CUDA's "more stages always better":
- single GEMM → `num_stages = 0`
- two fused GEMMs (FlashAttention) → `num_stages = 1`
- GEMM fused with a non-GEMM op → `num_stages = 0`
- no-GEMM kernel → `num_stages = 1`

Over-staging blows the VGPR/AGPR budget and triggers spills (each spill is an HBM round-trip → catastrophic). Validate occupancy with `rocprof-compute` after changing stages.

---

## 4. Online softmax

**What:** compute softmax in a single streaming pass with running max `m` and running denominator `l`, rescaling the accumulator as new blocks arrive — no full row of scores in memory.

```text
m = -inf ; l = 0 ; acc = 0
for block j of K/V:
    s   = q @ k_j^T * scale
    m_new = max(m, rowmax(s))
    p   = exp(s - m_new)
    l   = l * exp(m - m_new) + rowsum(p)        # rescale running denom
    acc = acc * exp(m - m_new) + p @ v_j        # rescale running output
    m   = m_new
out = acc / l
```

**When it helps:** prerequisite for FlashAttention; turns O(N²) score memory into O(block). Essential for long context where the full attention matrix won't fit in LDS/HBM budget.

**MI300X tradeoff:** the rescale (`exp(m - m_new)`) runs on the **VALU**, competing with the two MFMA GEMMs (QKᵀ and PV). Keep `m`/`l` in registers, fuse the rescale into the PV epilogue. Use the right exp intrinsic; avoid fp32 transcendental overuse.

---

## 5. FlashAttention

**What:** tile Q over rows and K/V over columns, run online softmax per tile, fuse QKᵀ → softmax → PV into one kernel so the N×N scores never touch HBM.

**When it helps:** prefill and long-context attention (memory-bound otherwise). On MI300X this is the canonical "two fused GEMMs" kernel → `num_stages = 1`.

**MI300X tradeoffs / state of the art:**
- Use AMD-tuned FA: the **AITER** attention kernels (CK-based and assembly), Dao-AILab FlashAttention ROCm/CK backend, and **HipKittens** (Nov 2025) which beats hand-optimized AITER assembly by **2.3×** on the backward pass and matches/exceeds all baselines for d=64 fwd, GQA bwd, and memory-bound cases.
- **Do not port NVIDIA warp-specialized FA verbatim** — producer-only waves waste CDNA's statically allocated registers. Use **8-wave ping-pong** (interleaved wave roles, explicit LDS management, barriers per stage) for d=128, or **4-wave interleave**.
- GQA/MQA: exploit shared KV heads to cut KV HBM traffic — decode attention is HBM-bound (KV cache reads dominate).
- FP8 attention (FA-3 style asynchrony/low-precision) is available; validate accuracy (see quantization file).

---

## 6. Chunked prefill

**What:** split a long prompt's prefill into chunks and **interleave** them with ongoing decode steps in the same batch, rather than running one giant prefill that stalls all decode.

**When it helps:** mixed serving (some requests prefilling, others decoding). Bounds **TTFT** for new requests and keeps decode latency (**ITL**) smooth. Raises GPU utilization because prefill (compute-bound) and decode (memory-bound) fill complementary resources.

**MI300X tradeoff:** chunk size is a knob — too small loses prefill GEMM efficiency (under-fills CUs), too large reintroduces decode stalls. Tune jointly with the GEMM split-K config so each chunk's GEMM still hits ≥1024 workgroups. vLLM/SGLang on ROCm expose chunked-prefill flags.

---

## 7. Speculative decoding — kernel implications

**What:** a small draft model proposes K tokens; the target model **verifies** them in one batched forward; accepted tokens commit.

**When it helps:** low-batch, latency-bound decode where the target model is HBM-bound at M=1 — verifying K candidates raises effective M to K with little extra HBM, converting wasted bandwidth into throughput.

**MI300X kernel implications:**
- Verification turns M=1 decode GEMMs into M=K GEMMs → **different tuned shapes**. Re-tune hipBLASLt/Triton for the verification batch dims (see GEMM tuning file).
- Tree/medusa attention needs custom masked attention kernels; ensure the mask path is fused (no separate mask kernel).
- AMD published a speculative-decoding enablement guide for MI300X (ROCm Blogs) covering EAGLE/draft-model paths.

---

## 8. GEMM + epilogue fusion

**What:** fold bias add, activation (GELU/SiLU), scaling, and quantization into the GEMM's epilogue so the result is transformed *in registers* before the single global store.

**When it helps:** every memory-bound pointwise op that follows a GEMM. Eliminates a full HBM write+read round-trip of the GEMM output (often >2× the GEMM's own HBM cost for memory-bound layers). See `fusion_patterns.md` for the catalog.

**MI300X tradeoff:** epilogue ops consume VGPRs and may force lower occupancy; `OPTIMIZE_EPILOGUE=1` keeps the MFMA-layout accumulator to skip a reblock. Heavy epilogues (per-token quant + dynamic scale reduction) can spill — measure.

---

## 9. Communication–computation overlap

**What:** in tensor-parallel / expert-parallel inference, overlap RCCL all-reduce / all-to-all with the GEMM that produces (or consumes) the data, by chunking the GEMM and launching collectives on a separate stream as each chunk completes.

**When it helps:** multi-GPU MI300X (8-GPU node via Infinity Fabric). All-reduce after each TP layer is otherwise pure stall.

**MI300X tradeoffs:**
- Set `NCCL_MIN_NCHANNELS=112` to use more channels on MI300X.
- Use separate HIP streams so the collective and GEMM truly overlap; ensure the GEMM grid leaves CU/queue room for the collective's kernels.
- MoE all-to-all overlap pairs with fused-MoE grouped GEMM (see fusion file). SGLang's MoE align&sort gives 7× on MI300X over the Triton baseline, freeing time to overlap.

---

## 10. Wave specialization vs ping-pong / interleave (CDNA3-specific)

**What (and the warning):** On NVIDIA, **warp specialization** (dedicated producer warps doing TMA loads while consumer warps do WGMMA) dominates. On CDNA3/CDNA4 this **underperforms**: AMD's static register allocation means producer-only waves hold registers without computing, shrinking output tiles and arithmetic intensity. On MI355X wave-spec reaches only ~80% of peak BF16 GEMM.

**Use instead (HipKittens, Nov 2025):**
- **8-wave ping-pong** — waves alternate compute/memory roles in lockstep with explicit LDS buffers and a barrier per stage; every wave both loads and computes, so no registers are wasted.
- **4-wave interleave** — interleave memory and MFMA at instruction granularity within each wave.

**When:** hand-written peak GEMM / attention. These patterns achieved peak AMD perf where wave-spec stalled; HK-attention backward (4-wave interleave) hit 2.3× over AITER assembly.

**Tradeoff:** much harder to write and tune than autotuned Triton — reserve for the hottest kernels. Use the **CDNA3 branch of HipKittens** for MI300X/MI325X.

---

## Putting it together: a decode vs prefill cheat sheet

| Phase | Bottleneck | Apply |
|---|---|---|
| **Prefill** (large M) | compute-bound MFMA | software pipelining, FlashAttention (`num_stages=1`), persistent+Stream-K, epilogue fusion, comm overlap |
| **Decode** (M=1..few) | HBM-bound (weights + KV) | split-K to fill CUs, KV-cache quant + GQA, speculative decoding (raise effective M), fused dequant epilogue |
| **MoE** | routing + grouped GEMM | fused MoE (align&sort + grouped GEMM), all-to-all overlap, per-expert tuned configs |

---

## Sources

- Stream-K++ adaptive GPU GEMM (arXiv 2408.11417): <https://arxiv.org/pdf/2408.11417>
- CUTLASS persistent kernels & Stream-K (Colfax, concepts): <https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/>
- HipKittens: Fast and Furious AMD Kernels (ping-pong / interleave, CDNA3/4): <https://arxiv.org/html/2511.08083v1> · <https://github.com/HazyResearch/HipKittens>
- FlashAttention-3 (asynchrony, low precision): <https://arxiv.org/pdf/2407.08608> · FlashAttention ROCm/CK backend: <https://github.com/Dao-AILab/flash-attention>
- AMD MI300X workload optimization (num_stages, split-K, grid ≥1024): <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
- Enabling Speculative Decoding on MI300X (ROCm Blogs): <https://rocm.blogs.amd.com/artificial-intelligence/ssd_mi300x/README.html>
- AMD Instinct MI300 CDNA3 ISA reference (register file, MFMA): <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>
- Hands-on with CK-Tile: optimized GEMM on AMD GPUs: <https://rocm.blogs.amd.com/software-tools-optimization/building-efficient-gemm-kernels-with-ck-tile-vendo/README.html>
