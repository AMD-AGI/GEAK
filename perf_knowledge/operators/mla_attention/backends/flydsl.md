---
title: mla_attention backward on FlyDSL — SOTA card
kind: sota_card
operator: mla_attention
backend: flydsl
gens: [gfx942]
dtypes: [bf16]
regimes: [training]
status: sota
updated: 2026-08-12
sources:
  - https://arxiv.org/abs/2511.08083
---

# mla_attention (backward) × FlyDSL

> **This card is the measured instance.** The dimension-agnostic authoring levers it applies — fusion
> boundary, MFMA fragment orientation, which output goes on atomics, wave count vs the register cap,
> geometry derivation, verification gates — are in
> [`languages/flydsl/authoring_attention_levers.md`](../../../languages/flydsl/authoring_attention_levers.md).
> Read that for the *rules*; read this for the numbers that back them. The constants here are one
> resolution of those derivations, not defaults to copy.

## TL;DR
For the **unabsorbed MLA training backward** — `qk_head_dim=192`, `v_head_dim=128`, bf16, causal, varlen —
a hand-authored FlyDSL three-kernel chain is the **measured fastest option on gfx942**: **715–745 µs**
(~230 TFLOP/s of useful work) against **984 µs** for aiter's ASM backward with V padded 128→192, at
`cos ≥ 0.999999` on all three gradients. The structural point is that **no aiter configuration on gfx942
gets both the forward and the backward fast**: the ASM bwd dispatch is gated on `hdim_q == hdim_v` (so V
must be padded to 192) while the ASM *forward* requires `hdim_v == 128`. End to end that makes this
**1.49×** faster than the padded/ASM-bwd config and **1.79×** faster than the unpadded/CK-tile-bwd config.
This is a first-party on-box measurement, not a vendor number — and it is the source of GEAK's gfx942
counter-measurement to HipKittens §3.3.2 on wave counts (see
[[optimization/mfma_scheduling]]).

## SOTA implementation
Three device-side steps per backward call: zero the fp32 dQ accumulator, a preprocess kernel computing
`D = rowsum(dO ⊙ O)` into a **head-major** `[nheads, seqlen]` fp32 tensor, then one main kernel running
**all five backward GEMMs fused**. Fusing is not an optimisation to retrofit — splitting dQ into its own
kernel forces S and dP to be recomputed there, raising work from 1664 to 2304 FLOPs per score element
(+38%), which loses to the ASM baseline on its own.

Four structural decisions carry the performance — the general form of each is a lever in
[[languages/flydsl/authoring_attention_levers]]; here is how they resolved at these head dims. Two are
layout choices that are painful to retrofit:

- **Operand-swapped MFMA fragments** (`v_mfma_f32_16x16x16_bf16_1k`) make the score land as
  `S[qrow, kpos]`, which is already the A-operand layout both dV and dK want. **P and dS never leave
  registers and never transit LDS** — free, if chosen first.
- **KV-outer schedule**: KV rows are exclusive to a block, so dK/dV are stored directly, no atomics. Only
  dQ accumulates, via global atomics into a 606 MB fp32 workspace. The q-outer alternative puts dK/dV on
  atomics at 4.04 GB.
- **`kv_tile` on the slowest grid axis** (worth **20%**) — longest-job-first, because a block's cost falls
  monotonically with `kv_tile` under causal masking. See [[optimization/xcd_l2_locality]].
- **Read the second operand orientation from LDS instead of materialising it** (worth **19%**) — see
  [[optimization/lds_and_bank_conflicts]].

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| FlyDSL fused 5-GEMM bwd chain (native 192/128) | 13 varlen causal seqs, 32768 tokens, 2 heads | gfx942 (MI300X/MI325X, **SPX/NPS1**); bf16 | **715–745 µs, ~230 TFLOP/s useful** — vs padded-ASM 984 µs (−27.3% wall, −15.4% executed work); e2e **1008.6 µs/iter** vs 1497.9 (B) / 1806.9 (A) | asymmetric MLA head dims on gfx942 training |
| FlyDSL preprocess (`D = Σ dO·O`) | same | gfx942; bf16→fp32 | **3.9 µs** vs ASM `odo` 22.2 µs (**5.7×**) | always — smallest, easiest win |
| (ref) aiter ASM bwd, V padded to 192 | aiter v3 gfx942 dispatch | gfx942; bf16 | 984 µs, 196.9 TFLOP/s executed / 170.7 useful | symmetric head dims, where no padding is needed |
| (ref) CK-tile bwd, V unpadded | aiter fallback | gfx942; bf16 | 1560.8 µs | keeps the fast ASM forward, loses the backward |

## Config space / knobs
Defaults are the measured optimum; the last three exist only to reproduce the ablation ladder and should
be kept anyway, so the comparisons stay checkable in one interleaved process on new hardware.

| param | range / source | effect | default |
|---|---|---|---|
| `block_m` | 16 / 32 / 64 | q rows per tile; 64 is over LDS budget once double-buffered, 16 is rejected at 8 waves (12 dQ tiles don't divide by 8) | **32** |
| `block_n` | 64 / 128 / 192 / 256 | KV rows per block; ≥192 is 3–6× slower (K/V/Kᵀ operand set overruns the 256 arch-VGPR cap) | **128** |
| `num_waves` | 4 / 8 | 8 → 512 threads, 2 waves/SIMD. **4 is 15% slower on this kernel** — see Pitfalls | **8** |
| `dq_groups` | 1 / 2 / 4 / 8 | dQ atomic fan-out slots; monotonically worse (1187/1216/1259/1339 µs) — atomics are L2-uncached regardless | **1** |
| `kv_major` | bool | `kv_tile` on the slow axis (longest-job-first) | **True** (−20%) |
| `pipeline` | bool | double-buffered staging, 2 barriers instead of 3 | **True** (−5.4%) |
| `transposed_tiles` | bool | materialise the second orientation instead of reading it | **False** (True costs +19%) |
| `schedule` / `sched_steps` | `sched_group_barrier` hints | neutral once the region is branch-free and single | off |

Resolved geometry at the default: `LDS 50948 B BLOCK_M=32 BLOCK_N=128 waves=8 dq_groups=1 pipeline=True
kv_major=True`. Print it and treat it as part of the gate — anything else means no timing is comparable.

## Numerics / parity
fp32 accumulate throughout; gradients come back fp32 and the caller casts to bf16. `lse` is
`[nheads, total_q]` fp32 in natural log with the softmax scale already folded in, exactly as a
Flash-Attention forward emits it, because the kernel computes `p = exp2(s·scale·log2e − lse·log2e)`.
Causal convention is **bottom-right aligned** (`delta = seqlen_k − seqlen_q`, keep `j <= i + delta`);
top-left alignment is a different problem that silently produces high-cosine garbage. Masking is
branch-free and a zeroed P also zeroes dS, so no separate dS mask is needed.

Measured `cos = 0.999999` on dq/dk/dv with `max_err/scale ≈ 1.6–2.0e-03`; gate at `cos_tol=0.999`,
`rel_tol=3e-2`. Accuracy is also *better* than both aiter configs (dq max abs err 1.193e-02 vs 2.546e-02
unpadded-CK and 3.627e-02 padded-ASM).

## Counter expectations (rocprofiler-compute)
These confirm the design is behaving as specified rather than accidentally working:

| claim | counter |
|---|---|
| 8 waves/CU, 2 waves/SIMD, 1 block/CU | 128 arch VGPR + 128 AGPR, 112 SGPR, LDS 51200 B |
| no spill despite a full register file | scratch 16 B, **0** spill/stack instructions |
| the column-wise transpose read is conflict-free | 0.03 bank conflicts/access, 0 address conflicts |
| branch-free staging keeps the exec mask full | 63.97 of 64 VALU active threads |
| reading the transpose beats materialising it | LDS instructions 12.7M → 42.4M (3.3×), but vL1D read requests 322.6M → 66.0M (−80%) and VMEM instructions −54% |
| the decomposition is right | MFMA count exactly 21,582,080 (104/wave-iteration × 207,520), 100% BF16 |

**Nothing is saturated**: MFMA 16.1% of peak, VALU 19.7% of issue slots, LDS 22.8%, vL1D 26.1%, HBM 33%.
Per resident wave-cycle: 36.1% active, **40.4% waiting on a dependency**. The kernel is latency-bound at
2 waves/SIMD, and 2 waves/SIMD is fixed by the register cap — do not read the 51% occupancy figure as
free headroom (splitting the longest blocks to collect it was implemented and produced nothing).

## Pitfalls & anti-patterns
- **`num_waves=4` is 15% slower, and scratch will lie to you about why.** The best 4-wave build reports
  `private_seg_size 0 / agpr_count 123 / vgpr_count 379` with **75 `v_accvgpr` moves** — *lower* scratch
  than the 8-wave build, which has 256 VGPR / 0 AGPR / 0 moves. The spill relocated into the accumulator
  file. **Count `v_accvgpr` moves, not scratch** ([[optimization/occupancy_and_registers]]).
- **SPX/NPS1 is part of the specification.** The 20% grid-mapping win is built on static XCD round-robin;
  CPX changes both dispatch and atomic behaviour. If the `kv_major` ablation ratio comes out near 1.0
  instead of 0.79, check the partition mode before suspecting the kernel.
- **Optimisations that keep more values live lose to the register cap**: hoisting invariant staging
  address math out of the q-loop is **−8.6%** (triples spill), and deferring the dQ phase to fold to a
  single barrier is **−36%**. Predict the effect on the live set and check the ISA first — a two-minute
  check that pre-empts both.
- **Splitting the region on a block-uniform condition measures slower** even when it removes ~200 VALU of
  bounds checks, because the `scf.If` boundary stops the scheduler hoisting global loads into the previous
  iteration's MFMA chain. Overlap is worth more than instruction count here.
- Do not design around L2 residency for the dQ atomics — every one is uncached by construction.
- Budget ~49 µs of `bfloat16_copy_kernel` as this path's own overhead (it leaves gradients in fp32 where
  the ASM path folds the dQ cast into its convert pass). Emitting bf16 from the epilogue is
  register-neutral and the most accessible remaining win.

## How to verify
```bash
rocm-smi --showproductname                       # confirm gfx942 + expected CU count; check SPX/NPS1
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/isa python isa_probe.py "num_waves=8"
grep -E 'vgpr_count|agpr_count|private_seg_size' /tmp/isa/fmha_bwd_main/21_final_isa.s
grep -c v_accvgpr /tmp/isa/fmha_bwd_main/21_final_isa.s     # must be 0
# then: build BOTH sides in ONE interleaved sweep (min over replays), never across processes
```

## Alternatives / cross-links
[[operators/mla_attention/backends/aiter]] (sota decode/prefill; the ASM bwd baseline here) ·
[[operators/mla_attention/backends/ck]] (CK-tile bwd fallback) ·
[[operators/gqa_mqa_attention/backends/hipkittens]] (the other attention-backward card; note its 4-wave
recommendation does **not** transfer to this shape) ·
[`expert_skills/skills/flydsl_fused_attention_backward`](../../../expert_skills/skills/flydsl_fused_attention_backward/skill.md)
(the regulated procedure) ·
[`languages/flydsl/authoring_attention_levers.md`](../../../languages/flydsl/authoring_attention_levers.md)
(the general levers) ·
[[languages/flydsl/authoring_optimization]] · [[languages/flydsl/debugging]].

## Sources
- gfx942 MI300X/MI325X, SPX/NPS1, 13 varlen causal sequences of
  472–3638 tokens (32768 total), 2 heads, bf16, softmax scale 0.08838834764831843, bottom-right causal.
- aiter gfx942 v3 bwd `hdim_q == hdim_v` dispatch gate and ASM-forward `hdim_v == 128` requirement: same
  document §8.3–§8.4 (measured against an aiter source checkout).
- HipKittens §3.3.2 wave-schedule patterns that this kernel's 4-wave result contradicts on gfx942:
  https://arxiv.org/abs/2511.08083
