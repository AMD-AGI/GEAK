# AMD RDNA (Radeon / Ryzen AI APU) Hardware Reference — DETECT THE BOX FIRST

This file covers **RDNA** parts — currently validated on **RDNA 3.5** (`gfx1151`, Strix Halo /
Ryzen AI MAX+ 395 w/ Radeon 8060S). RDNA is **not** a smaller CDNA: different wavefront size,
different matrix instruction, no AGPRs, a large last-level cache, and an order of magnitude less
memory bandwidth. **If the box is `gfx9xx`, use `amd_instinct.md` instead — do not mix the two.**

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

**The 790 GB/s figure is READ-READ-WRITE. A pure-read stream goes higher.** `[measured by
ablation]` A same-grid / same-byte-stream kernel with the math deleted reached **782 / 913 / 945
GB/s** for LLC-resident pure-read working sets. So do **not** declare a read-dominated kernel
finished at 790 GB/s -- one GQA decode kernel that was called "saturated" against 790 was actually
at 83-86% of its real ceiling, with ~17% left. **Measure the ceiling for YOUR access mix by
ablation** (keep the grid and the byte stream, delete the math, time that) rather than taking a
number from this table.

## 4. Bandwidth and FLOPS reality check

| Figure | Value | Note |
|---|---|---|
| DRAM, paper | 256 GB/s | `[vendor]` — **~12% optimistic** |
| DRAM, achievable streaming | **229-233 GB/s** | `[measured]` `bw.py` |
| DRAM, real KV-streaming kernel | 222 GB/s | `[measured]` llama.cpp hip-ep PR #675, RGP |
| LLC (≤32 MB working set) | ~790 GB/s | `[measured]` `bw.py` |
| fp16 GEMM 2048³, naive Triton | **24.99 TFLOP/s** | `[measured]` `wmma_check.py` |
| fp16 GEMM 2048³, hipBLASLt | 12.60 TFLOP/s | `[measured]` — see §5 |

**Build roofline on the measured number, never on 256 GB/s.** A kernel already at ~90% of the
practical 222-233 GB/s wall looks like it still has ~20% headroom against the paper figure. Chasing
that is wasted budget.

## 5. Where the vendor library is weak -- and where it is not

`[measured]` This is **shape-dependent in both directions**. Do not carry a blanket "hipBLASLt is
weak on RDNA" prior; check the regime.

| shape / regime | naive Triton vs hipBLASLt | after tuning |
|---|---|---|
| large square, 2048^3 fp16 | Triton **1.98x faster** (24.99 vs 12.60 TFLOP/s), untuned | -- |
| small square, 128/256/512^3 | hipBLASLt **1.6-2.1x faster** than naive Triton | a tuned kernel beats it **1.51x / 1.83x** at 128^3 / 256^3 but still **loses 13%** (0.87x) at 512^3 |
| skinny / decode, M<=32, N=K=4096 | Triton **~1.8x faster** than `torch.mm` | -- |
| short-K (`shortk_512`) | Triton **0.32x** -- the library wins outright | -- |

Read that as: hipBLASLt on `gfx1151` is tuned for **mid-size square** shapes and is genuinely hard
to beat there -- a tuned kernel still lost at 512^3 -- while it leaves large-square and
skinny/decode shapes on the table. The part is a second-class math-library target overall, so a
generated kernel *can* win outright, but **measure the specific shape first, in either direction**.
The honest rule is not "the library is weak"; it is "the coverage here is uneven, so the claim that
the library already does this must be verified per shape".

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
5. **PMC: the limit is a hardware-counter BUDGET, not a metric count.** `[measured]` With *simple
   raw* counters (`SQ_WAVES`, `GRBM_GUI_ACTIVE`, `FETCH_SIZE`, `WRITE_SIZE`) 1, 2 and 3 succeed and a
   4-counter `pmc:` line aborts with `error code 38: Request exceeds the capabilities of the hardware
   to collect`. Do not read that as "three metrics is always safe": a **derived** metric can expand
   into several hardware counters, so three derived metrics can exceed the same budget. **Probe the
   specific set you want.** CDNA parts take many more, so a CDNA-derived counter list WILL fail here.
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

1. **Detect wave size first.** wave32 → this file. wave64 → `amd_instinct.md`. Never mix.
2. **WMMA, not MFMA. No AGPRs.** Translate or discard any MFMA/AGPR-phrased strategy.
3. **Check the 32 MB LLC boundary before anything else.** Getting the working set under it is worth
   3.4x and is usually the biggest single lever on this part.
4. **Roofline on 229-233 GB/s measured, never 256 GB/s paper.**
5. **Do not assume the vendor library is a hard floor** — a naive Triton GEMM already beats
   hipBLASLt 2x here.
6. **Re-measure, never port CDNA numbers.** This part has ~1/25th the memory bandwidth and a
   completely different cache hierarchy; CDNA intuitions about what is memory-bound are wrong.
