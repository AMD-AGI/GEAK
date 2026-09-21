# AMD RDNA (Radeon / Ryzen AI APU) Hardware Reference — DETECT THE BOX FIRST

This file covers **RDNA** parts, validated on **RDNA 3.5** (`gfx1151`, Strix Halo / Ryzen AI
MAX+ 395 w/ Radeon 8060S). RDNA is **not** a smaller CDNA: different matrix instruction, no AGPRs,
a different occupancy model.

**Two kinds of claim live here — do not promote one to the other:**

* **RDNA family invariants** — true across `gfx10`/`gfx11`/`gfx12`: WMMA rather than MFMA (and
  `gfx10` has neither), no AGPRs, 2 SIMDs/CU, the VGPR/occupancy model in section 2.
* **`gfx1151` APU profile** — `[measured]` on this part and **not** portable to other RDNA parts:
  40 CU, a 32 MB last-level cache, LPDDR5X **shared with the CPU** (UMA, so the "VRAM" is a
  carve-out of system RAM), 229-233 GB/s streaming DRAM. A discrete `gfx1100` has its own GDDR6 and
  a different cache hierarchy; **sections 3, 4 and 7 are about this APU**, not about RDNA.

**Routing — decide by architecture family, not by wave size:**

1. **gfx prefix decides the file.** `gfx9xx` → `amd_instinct.md`. `gfx10/11/12` → this file.
2. **Capability decides the strategy.** Ask what the matrix path *is* — `mfma`, `wmma`, or `none`
   (`gfx10` is RDNA with no matrix instruction, so "RDNA implies WMMA" is also wrong).
3. **Wave mode is a check, not the decision.** RDNA supports wave32 **and** wave64; a wave64 kernel
   on `gfx11` is not a CDNA kernel. Use the compiled wave size to size tiles and to catch a
   contradiction — if the family and the wave mode disagree with each other, stop and re-detect
   rather than guessing.

Every number tagged `[measured]` was taken on `gfx1151` with the scripts named beside it. Numbers
tagged `[vendor]` are datasheet figures and are **known to overstate** what you can reach.

## 0. Detect THIS box first (source of truth > this table)

```bash
# rocminfo lists the CPU agent FIRST; scope every field to the gfx agent or you get CPU numbers.
rocminfo 2>/dev/null | awk '/^ *Name: *gfx/{print $2; exit}'                        # gfx target
rocminfo 2>/dev/null | awk '/Name:.*gfx/{f=1} f&&/Compute Unit:/{print $3; exit}'   # CU count
rocminfo 2>/dev/null | awk '/Name:.*gfx/{f=1} f&&/Wavefront Size:/{print $3; exit}' # 32 on RDNA
```

Branch on the wavefront size, not on a name string: **RDNA is wave32, CDNA is wave64**, and the
kernels differ accordingly.

## 1. Card reference (verify on-box)

| Part | Arch / gfx | CU | Wave | LLC | Memory | Bandwidth |
|---|---|---:|---:|---:|---|---|
| Radeon 8060S (Ryzen AI MAX+ 395) | RDNA 3.5 / `gfx1151` | 40 | 32 | **32 MB Infinity Cache** | LPDDR5X, **shared with the CPU**, 96 GB carve-out of 128 GB | 256 GB/s `[vendor]` / **229-233 GB/s `[measured]`** |

`gfx1150` / `gfx1152` are the same family; `gfx1100`/`gfx120x` are RDNA 3 / RDNA 4 — detect, do not assume.

## 2. RDNA fundamentals — how it differs from CDNA

| | CDNA (`gfx942`/`gfx950`) | **RDNA (`gfx11xx`)** |
|---|---|---|
| Wavefront | 64 | **32** |
| SIMDs per CU | 4 | **2** |
| Matrix instruction | MFMA | **WMMA** (`__builtin_amdgcn_wmma_*`) |
| AGPRs | yes | **none** — all spills go to VGPR/scratch |
| Mixed-sign dot4 | `sdot4` only | `sudot4` available |
| LLC | small relative to HBM | **32 MB, and it matters enormously — see §3** |
| Memory | dedicated HBM, 5.3-8 TB/s | **LPDDR5X shared with the CPU**, ~0.23 TB/s |

Consequences: any strategy phrased in terms of MFMA shapes, AGPR budgeting, or wave64 lane
arithmetic **does not apply**. Translate to WMMA and wave32 or drop it.

### Occupancy vs VGPRs/thread (RDNA: 2 SIMDs/CU, wave32, 1536 VGPR/SIMD, granule 24, cap 16 waves)

| VGPRs/thread | ≤96 | 120 | 144 | 168 | 192 | 216 |
|---|---:|---:|---:|---:|---:|---:|
| waves/SIMD | 16 | 12 | 10 | 9 | 8 | 7 |

(Source: `perf_knowledge/expert_skills/skills/gluon_authoring/scripts/amd_occupancy.py`, verified
against ROCm 7.2.1 / LLVM 22.)

## 3. The Infinity Cache is the headline RDNA lever

`[measured]` — `bw.py`, streaming read-read-write, warmed, 20 iterations:

| Working set | Bandwidth |
|---|---|
| ≤ 32 MB (fits LLC) | **~790 GB/s** |
| 64 MB | 230 GB/s |
| 256 MB | 233 GB/s |
| 1024 MB | 229 GB/s |

**A working set that fits in the 32 MB Infinity Cache runs at 3.4x the bandwidth of one that does
not.** No CDNA part has this shape — MI350X pairs 8 TB/s HBM with a proportionally far smaller
cache, so *no CDNA optimization strategy contains this lever*. On RDNA, **"tile/block/split the
problem so the hot working set drops under 32 MB" is frequently the single largest available win**,
and it should be considered before any instruction-level tuning.

Corollary for roofline analysis: there are **two** rooflines on this part, not one. Decide which
regime the kernel is in before computing headroom.

> **This is a BANDWIDTH lever, and it does nothing for compute-bound work.** `[measured]`
> `rdna_roofline.py`, fp16 square GEMM: 1024^3 (6.3 MB, deep in LLC) reaches 35.16 TFLOP/s, while
> 3072^3 (56.6 MB, well into the DRAM regime) still reaches **34.13** — a 3% difference, not 3.4x.
> At those shapes the arithmetic intensity is ~683 FLOP/byte, so even the DRAM roof
> (683 x 230 GB/s = 157 TFLOP/s) sits far above the compute roof and bandwidth never binds.
> **Check which roof binds before spending effort on cache residency**: for high-AI kernels the
> 32 MB boundary is worth nothing, and the lever is real only for bandwidth-bound work — weight
> streaming in decode, elementwise, norms, low-reuse attention.
>
> The LLC roof is also **not a step function at 32 MB**: 6.3 MB gives 35.16 TFLOP/s but 25.2 MB —
> still nominally LLC-resident — gives 28.53. It degrades as the working set approaches capacity,
> so "under 32 MB" is not the same as "at the 790 GB/s roof".

**The 790 GB/s figure is READ-READ-WRITE. A pure-read stream goes higher.** `[measured by
ablation]` A same-grid / same-byte-stream kernel with the math deleted reached **782 / 913 / 945
GB/s** for LLC-resident pure-read working sets. So do **not** declare a read-dominated kernel
finished at 790 GB/s -- one GQA decode kernel that was called "saturated" against 790 was actually
at 83-86% of its real ceiling, with ~17% left. **Measure the ceiling for YOUR access mix by
ablation** (keep the grid and the byte stream, delete the math, time that) rather than taking a
number from this table.

## 3b. The compute roofline — and how far real kernels sit from it

`[measured]` `repro/rdna_roofline.py`. Section 3 gives `Peak_BW`; this gives the other half of
`attainable = min(Peak_Compute[dtype], AI x Peak_BW)`, without which no kernel can be scored in
absolute terms.

| | fp16 / bf16 WMMA |
|---|---|
| datasheet peak | **59.4 TFLOP/s** (40 CU x 64 lanes x 2 x 2.9 GHz x 4) |
| empirical peak | **35.16 TFLOP/s** — best any kernel reached here (`torch.mm` @ 1024^3) |
| empirical / datasheet | **59.2%** |

**Report both.** The empirical column is a *floor* on the true peak: nothing measured here saturated
the WMMA units, so a kernel scored against it looks better than it is, and a kernel scored against
the datasheet may be chasing a number this part cannot reach. Quoting one without the other is how
a tuning effort ends up aimed at the wrong ceiling.

Per-kernel efficiency at 2048^3 fp16 (working set 25.2 MB, LLC-resident, AI 683 FLOP/byte):

| kernel | achieved | vs empirical | vs datasheet |
|---|---|---|---|
| naive Triton `tl.dot` | 24.72 TFLOP/s | 70.3% | 41.6% |
| `torch.mm` (rocBLAS) | 28.53 TFLOP/s | 81.1% | 48.0% |

**This shape is compute-bound, by 9x** — the memory roof is 539 TFLOP/s against a 59.4 TFLOP/s
compute roof. So the whole section 5 argument about which library wins is an argument about
*compute* efficiency, and neither contender is above half the datasheet peak. That is where the
headroom is, not in cache residency.

**An efficiency above 100% of the empirical peak is not a fast kernel — it is proof the denominator
is wrong.** Treat it as a failed measurement and re-derive the peak. (Discipline borrowed from
`perf_knowledge/profiling/kernel_roofline.md`, which is CDNA-scoped and refuses RDNA outright
because it drives `rocprof-compute --roof-only`; that tool's roofline mode does not support
gfx10/11/12, so the terms here are measured a different way.)

**Pick the compute peak by the matrix instruction the kernel issues, not by its tensor dtype.** On
RDNA that is WMMA; `rdna_roofline.py` disassembles first and refuses to score anything if it sees
`v_mfma` or no `v_wmma`, because a CDNA peak table applied here would be silently wrong.

## 4. Bandwidth and FLOPS reality check

| Figure | Value | Note |
|---|---|---|
| DRAM, paper | 256 GB/s | `[vendor]` — **~12% optimistic** |
| DRAM, achievable streaming | **229-233 GB/s** | `[measured]` `bw.py` |
| DRAM, real KV-streaming kernel | 222 GB/s | `[measured]` llama.cpp hip-ep PR #675, RGP |
| LLC (≤32 MB working set) | ~790 GB/s | `[measured]` `bw.py` |
| fp16 GEMM 2048³, naive Triton | **23.59 TFLOP/s** | `[measured]` `wmma_check.py`, median of 40 interleaved rounds |
| fp16 GEMM 2048³, `torch.mm` | **27.52 TFLOP/s** | `[measured]` same run — the vendor path wins here; see the retraction in §5 |

**Build roofline on the measured number, never on 256 GB/s.** A kernel already at ~90% of the
practical 222-233 GB/s wall looks like it still has ~20% headroom against the paper figure. Chasing
that is wasted budget.

## 5. Vendor-library coverage: uneven, and not predictable from the architecture

`[measured]` This is **shape-dependent in both directions**. Do not carry a blanket "the vendor
library is weak on RDNA" prior; check the regime — and check *which* vendor library you reached,
because on this box `torch` defaults to rocBLAS and never touches hipBLASLt.

| shape / regime | naive Triton vs the vendor path | after tuning |
|---|---|---|
| large square, 2048^3 fp16 | `torch.mm` **1.16x faster** (27.52 vs 23.59 TFLOP/s) -- see the retraction note below | -- |
| small square, 128/256/512^3 | vendor path **1.6-2.1x faster** than naive Triton | a tuned kernel beats it **1.51x / 1.83x** at 128^3 / 256^3 but still **loses 13%** (0.87x) at 512^3 |
| skinny / decode, M<=32, N=K=4096 | Triton **~1.8x faster** than `torch.mm` | -- |
| short-K (`shortk_512`) | Triton **0.32x** -- the vendor path wins outright | -- |

> These three rows say "vendor path", not a library name, on purpose: only the 2048^3 row was
> re-measured with the backend checked. Which library the others actually reached is unverified,
> and on this box the default is rocBLAS rather than hipBLASLt.

> **Retraction.** An earlier version of this table claimed the opposite at 2048^3 -- "naive Triton
> 1.98x faster (24.99 vs 12.60 TFLOP/s)". That came from an unfair benchmark in `repro/wmma_check.py`:
> the vendor side allocated a fresh output tensor every iteration and got no warm-up, while Triton
> wrote into a pre-allocated buffer after five warm-up launches, and the two ran in sequence rather
> than interleaved. With both sides pre-allocated and warmed, interleaved, timed with GPU events and
> taken as a median of 40 rounds, the vendor path is **faster**. The script now does it that way.
> Caveat on any number from this box: round-to-round spread was 16.7% (Triton) and 34.0% (torch),
> consistent with the cpufreq governor sitting at `powersave`.
>
> **And the vendor library was misnamed.** `torch.backends.cuda.preferred_blas_library()` reports
> `_BlasBackend.Cublas`, which on ROCm is **rocBLAS** — hipBLASLt would report `Cublaslt`. So the
> comparison never ran against hipBLASLt at all, even though ROCm 7.2 ships `gfx1151`-tuned
> hipBLASLt kernels. The withdrawn claim was therefore wrong twice: unfair harness, wrong opponent.
> **Check which BLAS you are actually calling before attributing a result to a library**, and note
> that a `torch.mm` result on this part says nothing about hipBLASLt.

Read that as: **the vendor path wins in three of the four regimes here**, and is genuinely hard to
beat at square shapes -- a *tuned* Triton kernel still lost 13% at 512^3. The one regime where a
generated kernel clearly wins is **skinny / decode-shaped** GEMM, which is the shape that matters
most for token generation, so this is not a small exception. But note the asymmetry in evidence:
that row is one of the ones with **no repro script** (see `repro/README.md`), while the row that
did get a careful re-measurement is the one that flipped against Triton. Re-take the skinny number
before building a plan on it.

The honest rule is therefore **not** "the library is weak on RDNA" -- an earlier version of this
file said that, and the one claim of it that was rigorously re-measured collapsed. It is: **coverage
here is uneven and the direction is not predictable from the architecture, so measure the specific
shape, dtype and call path, in both directions, before assuming either that the library is a floor
or that it is beatable.**

## 6. Triton on RDNA — verified working

`[measured]` `triton_caps.py`, Triton 3.6.0 / torch 2.11.0 / ROCm 7.2.3:
elementwise, reduction/softmax, **`tl.dot` fp16**, **`tl.dot` bf16**, `triton.autotune`, atomics —
**all pass**. `tl.dot` lowers to genuine matrix instructions: generated AMDGCN for a 2048³ fp16 GEMM
contains **128 × `v_wmma_f32_16x16x16_f16`, zero `v_mfma`, zero scalar `v_fma`**.

When inspecting generated assembly on RDNA, grep for `v_wmma`, **not** `v_mfma`. Seeing no `v_mfma`
is expected and is not a fallback.

## 7. Tooling defects on gfx1151 — do not trust these

1. **`amd-smi static` reports `VRAM TYPE: GDDR7`.** It is LPDDR5X. `[measured]` Never derive
   bandwidth from amd-smi VRAM fields on an APU.
2. **Memory clock is not reported.** `rocprof-compute` leaves `$max_mclk` unpopulated on gfx1151, so
   its bandwidth and roofline metrics are wrong unless you pass `--specs-correction`.
3. **`rocprof-compute` output dir renamed** `strix_halo` → `rdna35_halo`; scripts globbing the old
   name silently find nothing.
4. `rocprofv3` itself is fine — `[measured]` it captures dispatches with correct grid/workgroup/
   timestamps on this part.
5. **PMC: the limit is a hardware-counter BUDGET, not a metric count.** `[measured]` For the one
   cumulative set that was swept -- `SQ_WAVES`, then `+ GRBM_GUI_ACTIVE`, then `+ FETCH_SIZE` -- all
   three combinations succeeded; adding `WRITE_SIZE` aborted with `error code 38: Request exceeds the
   capabilities of the hardware to collect`. **That result is about that set, not about the number
   three.** A metric named in a `pmc:` line is not necessarily one hardware counter: some expand into
   several, so a different trio can exceed the same budget while a larger set of cheaper counters
   fits. **Probe the specific set you want** rather than budgeting by list length. CDNA parts take many more, so a CDNA-derived counter list WILL fail here.
   Note what is *not* missing: the raw `TCP_`/`TCC_`/`TD_` **names** are unavailable, but the
   RDNA-native **`GL2C_HIT` / `GL2C_MISS` are collectable** and give L2 hit rate directly -- a real
   run measured **GL2 hit 50.37%** on this part. **Split into groups that fit and run one pass per
   group** -- that works and is cheap
   (`[measured]` 5 counters in 5 single-counter passes = 27 s total **in a warm harness
   process**; a cold process start costs ~85 s *per pass* -- see item 6). Do not spend agent turns
   rediscovering this: it cost 31 min of one 50 min run.
6. **The profiling cost is per PROCESS START, not per rocprofv3 pass.** The two measured numbers
   here look contradictory until the conditions are stated -- both are real:
   * `[measured]` **~5.4 s per pass** when passes run inside an **already-warm harness process**:
     the five single-counter `pmc_*` output directories of one run are stamped 02:21:30 through
     02:21:57, i.e. 27 s for five passes.
   * `[measured]` **~85 s per pass** when each pass **forks a fresh `python3`** that imports torch
     and initialises HIP from scratch (the `pmc_slots.sh` probe did exactly that): python sits at
     ~200% CPU with `gpu_busy_percent=0` for the whole startup, while the GPU work itself is
     under 200 ms.

   Consequences: (a) **structure profiling so the passes share one warm process** -- that is a 16x
   difference and the single biggest lever on profiling wall-clock here; (b) a profiling step showing
   high CPU, idle GPU and no new output is usually **starting up**, not hung, so wait >=2 min before
   calling it stuck; (c) budget by NUMBER OF PROCESS STARTS, not by kernel size -- shrinking the
   kernel does not make startup any faster.

## 8. Measurement discipline on an APU

- **Noise floor is low but the machine must be quiet**: `[measured]` idle CV 0.18%, 2σ = **0.4%**;
  sustained 4-minute memory-bound drift **+0.03%**. Treat anything under 0.4% as noise.
- **CPU and GPU share a power budget and the same LPDDR5X.** A busy CPU steals GPU bandwidth. The
  above characterises a memory-bound load; **a WMMA-heavy compute-bound load may throttle
  differently — characterise it separately.**
- `gpu_lock.sh` locks inside the container only. On a shared host it cannot see other users'
  containers. **Run an out-of-band idle check before measuring.**

## Critical Rules

1. **Route on gfx family, not wave size.** `gfx9xx` → `amd_instinct.md`; `gfx10/11/12` → this
   file. RDNA can run wave64, so wave size is a *check* on the decision, never the decision.
2. **WMMA, not MFMA. No AGPRs.** Translate or discard any MFMA/AGPR-phrased strategy. (`gfx10` is
   RDNA with no matrix instruction at all — check the capability, do not infer it from the family.)
3. **Check the 32 MB LLC boundary before anything else** (`gfx1151` figure — re-measure on any
   other RDNA part). Getting the working set under it is worth
   3.4x and is usually the biggest single lever on this part.
4. **Roofline on 229-233 GB/s measured, never the 256 GB/s paper figure** — and note this is the `gfx1151` DRAM number, not an RDNA one.
5. **Vendor coverage here is uneven — measure both directions, fairly.** Do not assume the vendor
   path is a hard floor *or* that it is beatable: for the exact shape, dtype and call path, measure
   both. The first claim in this file that a naive Triton GEMM beat it 2x at 2048^3 **was a
   measurement artifact** and reversed to 0.86x once both sides were pre-allocated, warmed and
   interleaved. Pre-allocate outputs on both sides, warm both, interleave, and use GPU events.
6. **Re-measure, never port CDNA numbers.** This part has ~1/25th the memory bandwidth and a
   completely different cache hierarchy; CDNA intuitions about what is memory-bound are wrong.
