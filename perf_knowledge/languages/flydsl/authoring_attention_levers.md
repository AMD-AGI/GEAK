---
title: "FlyDSL — attention authoring levers (fused multi-GEMM kernels)"
kind: language
gens: [gfx942]
dtypes: [bf16, fp16]
regimes: [prefill, training, both]
updated: 2026-08-12
sources:
  - AMD-AGI/GEAK@c0a1f937:src/minisweagent/skills/flydsl/docs/flydsl_optimization.md
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - https://arxiv.org/abs/2511.08083
---

> **Reference (how-to), not a verdict.** Attention-specific follow-on to
> [`authoring_optimization.md`](authoring_optimization.md), and the sibling of
> [`authoring_gemm_levers.md`](authoring_gemm_levers.md). For *which backend to use* on a given
> attention operator (library vs author) see the operator cards under
> [`../../operators/`](../../operators/) — authoring is usually the wrong answer when a tuned
> asm/CK path already dispatches for your shape.

# FlyDSL attention authoring levers

## Overview
Use this document when the kernel is an attention **fused multi-GEMM** — several matmuls chained
through register- or LDS-resident intermediates, with a softmax (or its derivative) in between — rather
than a single GEMM. The canonical cases are flash-attention forward (2 GEMMs) and attention backward
(up to 5: score, dP, dV, dK, dQ).

These kernels behave differently from GEMM in one decisive way: **the per-wave resident operand set,
not the tile size, is what binds.** A GEMM's registers are dominated by one accumulator you can shrink
by tiling; a fused attention kernel carries several long-lived operand sets *plus* accumulators across
the whole loop, so the 256 arch-VGPR cap is reached by the kernel's *structure* and most classical
"reduce instructions" optimizations lose to spill. Read
[`../../optimization/occupancy_and_registers.md`](../../optimization/occupancy_and_registers.md)
before pulling any lever here.

## Three decisions to make before writing code
Each of these is a rewrite if changed later, and none of them is a tuning knob.

### 1. The fusion boundary
Splitting a GEMM out of the chain forces its inputs to be **recomputed**, because the intermediates
(scores, probabilities) are too large to spill to HBM. Cost the split symbolically before choosing:
for a backward pass, per score element, the fused chain costs
`2·(HD_QK + HD_V + HD_V + HD_QK + HD_QK)` FLOPs while splitting the dQ GEMM into its own kernel adds a
second score + dP recomputation. Evaluate that ratio for *your* head dims — at symmetric
`HD_QK = HD_V = 128` it is ~+33%, and it rises as the head dims diverge. A split that adds >20% of
total work rarely wins back the register pressure it relieves.

### 2. The MFMA fragment orientation
For lane `l`, with `grp = l/16`, `col = l%16`, `g4 = 4·grp`, the two useful conventions differ in which
index lands on `col`:

```
standard:        A[m=g4+v, k=col]   B[k=col, n=g4+v]   C[m=col, n=g4+v]
operand-swapped: A[m=col,  k=g4+v]  B[k=g4+v, n=col]   C[m=g4+v, n=col]
```

Pick the one that makes an intermediate come out of its producing MFMA **already in the layout its
consumer needs**. In a backward pass the operand-swapped form yields `S[qrow, kpos]`, which is exactly
the A-operand layout that both the dV and dK GEMMs want (both contract over q rows), so the
probabilities and their gradient **never leave registers and never transit LDS**. The transpose you
avoid this way is free; the one you retrofit costs LDS traffic and a barrier.

### 3. Which axis is outer (and therefore what goes on atomics)
In a fused backward exactly one output contracts over the axis you parallelize, so exactly one output
must be accumulated across blocks. Choose the orientation that puts the **smaller** tensor on atomics:
a KV-outer loop makes KV rows exclusive to a block (dK/dV stored directly, no atomics, no workspace)
and accumulates dQ; a q-outer loop does the reverse. Compare the workspace volumes explicitly —
`fp32` accumulator bytes are `tokens · nheads · head_dim · 4`, so with typical `nheads ≫ 1` on the q
side this is usually a several-fold difference, not a marginal one. Decide once, then stop revisiting.

## Geometry derivation
Derive every count from the tile shape rather than hard-coding it, and **assert the divisibility
constraints at build time** so an invalid configuration fails loudly instead of computing plausible
garbage. With `WARP=64`, MFMA `16×16×16`, `FRAG=4`, `VEC=8` (one 128-bit bf16 load):

| symbol | derivation | meaning |
|---|---|---|
| `KPW` | `BN / NW` | KV rows per wave — the split axis |
| `NT_Q` | `BM / 16` | q-row tiles; also the dV/dK contraction steps |
| `KS_QK`, `KS_V` | `HD_QK / 16`, `HD_V / 16` | score and dP contraction steps |
| `NT_V`, `NT_HD` | `HD_V / 16`, `HD_QK / 16` | dV and dK output tiles |
| `TPW` | `NT_Q · NT_HD / NW` | per-wave share of the re-partitioned output |
| staging counts | `ceil((BM · HD_x / VEC) / BLOCK_THREADS)` | per-thread global loads per tile |

Required: `BN % (NW·16) == 0`, and `NT_Q·NT_HD` divisible by `NW` with a wave's share inside one q-row
tile. A hard-coded staging count silently drops part of a tile at any other geometry — this is the
most common cause of a kernel that is correct at one `BM` and wrong at another.

## The levers, in the order they pay
1. **Wave count — decide it against the register cap, not by occupancy intuition.** More waves means a
   smaller per-wave slice of the split axis, hence a *smaller* resident operand set; fewer waves gives
   each wave more registers but doubles what it must hold. Since the arch-VGPR cap is 256 **regardless
   of occupancy** (only the accumulator file grows as you drop to 1 wave/SIMD, and an MFMA cannot take
   an AGPR as an A or B operand), the fewer-waves direction often *loses* on these kernels even though
   it wins on GEMM. Measure both, and judge by the ISA — see
   [`../../optimization/mfma_scheduling.md`](../../optimization/mfma_scheduling.md).
2. **Read an operand in its awkward orientation instead of materialising the transpose.** Q and dO are
   each needed in two orientations (the score GEMM contracts over the head dim, dV/dK over q rows).
   Assembling a fragment from four 2-byte LDS reads down a *column* is the same operand and is
   conflict-free by construction — much cheaper than a second global read plus a second LDS copy. See
   [`../../optimization/lds_and_bank_conflicts.md`](../../optimization/lds_and_bank_conflicts.md).
3. **Keep the hot loop one scheduling region.** `s_barrier` ends an instruction scheduling region, so a
   global load and the MFMA chain you want it hidden behind must sit in the *same* region with no
   barrier between them. Double-buffer the staged tiles and issue tile `i+1`'s loads inside tile `i`'s
   MFMA chain. Keep the staging **branch-free** — clamp the tail index rather than guarding it, because
   a guard issues the same instructions under exec mask but ends the basic block and splits the region.
   The same reasoning explains why removing bounds-check VALU behind a block-uniform `scf.If` can
   measure *slower*. See [`../../optimization/memory_pipelining.md`](../../optimization/memory_pipelining.md).
4. **Re-partition the waves for the output that contracts over the split axis** instead of reducing
   partials. Publish the intermediate to LDS once, then divide the output tiles among the waves so each
   wave contracts over the whole axis alone. This converts a cross-wave reduction into a single barrier
   and collapses that accumulator's register cost by roughly `NW`.
5. **Order the grid axes by cost, not by convention.** Under causal masking a block's work falls
   monotonically with its KV-tile index, so putting that index on the *slowest* axis dispatches the
   expensive blocks first (longest-job-first). This is usually a few SALU and a permuted launch grid,
   and it is frequently the largest single win available. See
   [`../../optimization/xcd_l2_locality.md`](../../optimization/xcd_l2_locality.md).
6. **Emit the output dtype the caller wants.** Leaving gradients in fp32 for a bf16 pipeline buys a
   separate cast kernel per tensor and doubles the store traffic. Casting in the epilogue is
   register-neutral.

## Verification gates
Cheap, ordered, and each catching a distinct failure class:

1. **Parity** against an fp32 reference on *every* output (`cos ≥ 0.999` with a relative-error bound is
   a reasonable bf16 gate). One output passing while another fails localizes the bug: a good dV with a
   bad dQ points at the transpose or the re-partition indices, not at the masking.
2. **Echo the resolved geometry** (LDS bytes, tile shape, wave count, knob states) and treat it as part
   of the gate — a build that silently fell back to different parameters invalidates every timing.
3. **Register allocation**: arch VGPR, AGPR, **and `v_accvgpr` move count**. See the register-allocation
   gate in [`authoring_optimization.md`](authoring_optimization.md).
4. **MFMA count identity.** Compute the expected MFMA issues from the decomposition and compare against
   the profiled counter — it should match exactly. One counter pins the whole work decomposition and
   catches entire classes of masking and range bugs.
5. **Keep the ablation paths alive** behind knobs, permanently. Structural claims are only re-checkable
   if you can build both sides in one interleaved process (see
   [`../../profiling/benchmarking_methodology.md`](../../profiling/benchmarking_methodology.md)
   §single-kernel variant sweeps), and it lets a regression be bisected against a mechanism rather than
   a commit.

## Anti-patterns
| attempt | why it loses |
|---|---|
| Growing the KV/N tile to cut atomic traffic | the resident operand set scales with it and overruns the arch-VGPR cap |
| Hoisting invariant address math out of the hot loop | lengthens live ranges; the spill costs more than the arithmetic saved |
| Merging phases to remove a barrier | same cause — the merged region's live set exceeds the cap |
| Fanning atomics across more slots to spread contention | device-scope atomics are L2-uncached by construction, so this only multiplies DRAM traffic |
| Splitting the longest blocks to raise a low occupancy figure | if the block count is pinned by the register cap, total time is unchanged |
| Trusting `scratch`/`private_seg_size` as the spill measure | AGPR shuttling never appears there |
| Materialising a second orientation of a staged tile | see lever 2 |

**Before any change that keeps more values live, predict its effect on the live set and read the ISA.**
It is a two-minute check, and on these kernels it is the difference between a lever and a regression.

## Worked instance
An end-to-end application of every lever above, with per-step measured deltas, counter expectations and
a dead-end ledger, is recorded for a fused five-GEMM MLA backward with asymmetric head dims on gfx942:
[`../../operators/mla_attention/backends/flydsl.md`](../../operators/mla_attention/backends/flydsl.md)
and the skill
[`../../expert_skills/skills/flydsl_fused_attention_backward`](../../expert_skills/skills/flydsl_fused_attention_backward/skill.md).
Treat its constants as one instance of the derivations above, not as defaults to copy.

## Sources
- Generic authoring workflow this specializes: [`authoring_optimization.md`](authoring_optimization.md)
  (origin `AMD-AGI/GEAK@c0a1f937:src/minisweagent/skills/flydsl/docs/flydsl_optimization.md`).
- MFMA fragment layouts, AGPR/arch-VGPR split, `s_barrier` scheduling semantics: AMD CDNA3 ISA reference.
- Wave-scheduling patterns for AMD attention kernels (8-wave ping-pong / 4-wave interleave, and the
  register-allocation reason wave specialization loses on CDNA): HipKittens, arXiv 2511.08083 —
  see [`../hipkittens/primitives.md`](../hipkittens/primitives.md).
- Levers 1–6 and the anti-pattern table are each measured on-box; the instance, deltas and mechanisms
  are cited in the operator card above.
