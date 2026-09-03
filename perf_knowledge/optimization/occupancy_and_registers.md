---
title: occupancy and registers (VGPR/AGPR, waves/EU)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
  - https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/
---

# occupancy and registers

## TL;DR
On CDNA3/CDNA4 each SIMD (Execution Unit, EU) has **512 × 32-bit registers**, allocated in
**16-register granules**, split between **architected VGPRs (≤256)** and **accumulation AGPRs (≤256)**.
Occupancy in waves/EU = `floor(512 / round_up(VGPR_used, 16))` and is capped at **8 waves/EU** (32/CU)
by the instruction-buffer slots. The tuning game is: keep enough waves resident to hide MFMA + memory
latency, but **do not push so hard that the compiler spills** (latency cliff). For dense GEMM the right
answer is usually *fewer waves, more registers* (large MFMA tiles); for memory-bound elementwise/norm
it is *more waves*. See `[[hardware/shared/wavefront_simd_vgpr_agpr.md]]` and
`[[hardware/cdna3_mi300/occupancy.md]]`.

## Concepts (the hardware)
- **Register file**: 512 VGPRs/EU, 32-bit each. A single wave can use up to 512 total =
  256 architected VGPRs + 256 AGPRs. When a wave uses <512 total, the VGPR/AGPR split is flexible.
- **Allocation granularity**: VGPRs are reserved per wave in units of **16** (the tuning guide's
  occupancy unit; the ISA states groups of 8 Dwords). So 170 used ⇒ 176 reserved.
- **AGPRs**: accumulation registers, the *only* destinations/sources for MFMA accumulators
  (`v_mfma_*`). They extend usable register space beyond the 256 architected VGPRs and can also be
  loaded directly from memory; the compiler also uses `v_accvgpr_{read,write}` for cheap spill/fill.
  See `[[optimization/mfma_scheduling.md]]`.
- **Wave slots**: 8 wavefront slots per SIMD ⇒ max **8 waves/EU, 32 waves/CU**. Occupancy never
  exceeds this even with tiny register footprints.

## The occupancy formula (worked)
`waves_per_eu = min( 8 , floor(512 / round_up(VGPR_per_thread, 16)) )`

| VGPR/thread (reserved) | waves/EU |
|---|---|
| ≤ 64 | 8 (slot-capped) |
| 96 | 5 |
| 128 | 4 |
| 176 (e.g. 170 used) | **2** (176×3 > 512) |
| 256 | 2 |
| 512 (256 VGPR + 256 AGPR) | 1 |

AGPRs come out of the *same* 512 budget, so a GEMM with a large MFMA accumulator tile in AGPRs is
inherently low-occupancy — and that is fine, because MFMA latency is hidden by the deep pipeline,
not by many waves (see `[[optimization/mfma_scheduling.md]]`).

## The levers
- **`waves_per_eu=N` (triton / `__attribute__((amdgpu_waves_per_eu(N)))`)**: a *hint*; the LLVM
  backend tries to cut VGPR usage so N waves fit. Raise it to force more parallelism on
  latency-bound kernels; it can backfire by inducing spills.
- **`num_warps` (triton)**: warps = wave64 wavefronts per workgroup. More warps = bigger workgroup,
  more LDS/registers consumed per block, fewer blocks/CU. Typical GEMM: 4–8 warps. See
  `[[optimization/wave_and_grid_sizing.md]]`.
- **`__launch_bounds__(maxThreads, minWavesPerEU)` (HIP)**: hard-caps VGPRs the compiler may use so
  the requested occupancy is guaranteed; under-setting it forces spills.
- **MFMA tile size**: `32x32` instructions hold a larger accumulator in AGPRs than `16x16`, raising
  register pressure and dropping occupancy — a key reason `16x16` often wins on MI300X
  (`[[operators/dense_gemm/tuning.md]]`).
- **Reduce live state**: recompute cheap values instead of holding them; shrink `BLOCK_K` accumulation
  scope; move loop-invariants to scalar (SGPR) regs.

## Occupancy vs spilling — the cliff
Spills convert register accesses into **scratch (global) memory** traffic; on a compute-bound GEMM a
single spilled inner-loop value can cost more than the occupancy it buys. Diagnose with the assembler
report and the profiler:
- Look for `scratch` usage / `buffer_store`/`buffer_load` to scratch in the ISA dump
  (`[[languages/triton_amd/isa_verify.md]]`).
- `rocprof` / Omniperf counters: `VALUBusy`, wavefront occupancy, `SQ_WAIT_INST_LDS`, scratch traffic
  (`[[profiling/]]`, `[[hardware/cdna3_mi300/occupancy.md]]`).
- Rule of thumb: prefer **2 waves/EU with no spills** over 3 waves/EU that spill, for GEMM-class kernels.

### Count `v_accvgpr` moves, not scratch — scratch under-reports the spill
Scratch is not the whole spill surface. When arch VGPRs run out, the compiler's *first* move is to park
values in the **accumulator file** and shuttle them with `v_accvgpr_read/write`, which costs issue slots
and serialises against MFMA **but never appears as `private_seg_size`, `scratch_` or `buffer_store`**.
A build can therefore report *less* scratch than its rival and be materially slower.

Measured instance (gfx942 MI300X, fused attention backward, same kernel, one interleaved process):

| waves | `private_seg_size` | vgpr | agpr | `v_accvgpr` moves | µs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 16 B | 256 | 0 | **0** | **902** |
| 4 | **0 B** | 379 | 123 | 75 | 1055 |

The 4-wave build has *zero* scratch and is 15% slower. Read the arch-VGPR count, the AGPR count **and**
the `v_accvgpr` move count together:

```bash
grep -E 'vgpr_count|agpr_count|private_seg_size' <final_isa>.s
grep -c v_accvgpr <final_isa>.s        # the number that actually discriminates
```

A small non-zero `private_seg_size` (e.g. 16 B) is often a fixed prologue slot rather than spill — confirm
by checking there is no `scratch_`/`buffer_store` targeting it.

**Corollary for occupancy hunting:** the arch-VGPR cap is **256 per wave regardless of occupancy** — only
the accumulator file effectively grows as you drop to 1 wave/SIMD, and an MFMA cannot take an AGPR as an A
or B operand. So dropping the wave count does *not* relieve pressure on operands that must be arch VGPRs;
in a kernel whose per-wave resident operand set scales with its work slice, halving the waves *doubles*
that set against a fixed cap. `[[optimization/mfma_scheduling.md]]` has the accumulator-bound vs
operand-bound split and which way each resolves.

## Pitfalls
- Treating "more occupancy = faster" as universal. MFMA-bound kernels run great at 1–2 waves/EU.
- Forgetting AGPRs count against the 512 budget — a fat accumulator silently caps occupancy.
- Setting `waves_per_eu` high without checking the ISA dump for spills.
- **Judging spill by scratch alone** — AGPR shuttling is invisible there (see the cliff section above).
- Reading a low total-wave-cycle "occupancy %" as free headroom. If the block count is pinned by the
  register cap, the usual way to collect it (split the work into more, smaller blocks) changes nothing:
  in the measured case a 2-way split of the longest blocks landed at 747 vs 743 µs.
- Assuming CUDA "blocks/SM" math; CDNA granularity is **16 VGPR**, slots are **8/SIMD**, wave is **64**.

## Verify
- ISA/asm: confirm VGPR/AGPR counts, zero scratch **and zero `v_accvgpr` moves** (`amdgpu-arch` dump;
  triton `TRITON_CACHE`/`AMDGCN`; FlyDSL `FLYDSL_DUMP_IR=1` — see `[[languages/flydsl/debugging.md]]`).
- Profiler: occupancy and `VALUBusy` from Omniperf; compare across `waves_per_eu` settings.
- A/B: sweep `waves_per_eu ∈ {1,2,3,4}` and `num_warps ∈ {4,8}`, keep the lowest latency with no spill.

## Sources
- 512 VGPR/EU, 16-granule, worked 170→176→2-waves example, `waves_per_eu` hint: ROCm MI300X workload guide.
- 256 architected + 256 AGPR pools, allocation granularity, `v_accvgpr_*`: AMD CDNA3 (MI300) ISA reference.
- Register-pressure / occupancy reasoning (CDNA lab notes): AMD GPUOpen register-pressure note.
- `v_accvgpr`-vs-scratch table, the 256-arch-VGPR-cap-regardless-of-occupancy corollary, and the
  occupancy-%-is-not-headroom result: first-party on-box gfx942 MI300X, ROCm 7.1.0 —
  `Attention-Kernels/geak_trans_py2flydsl/fmha_backward/FMHA_BWD_FlyDSL_Skills.md` §9–§12, mirrored in
  `[[operators/mla_attention/backends/flydsl]]`.
