---
name: asm-kernel-rewrite
description: >
  Use when rewriting a hot GPU kernel into hand-tuned MFMA assembly on AMD
  Instinct MI300X (gfx942 / CDNA3) — the highest-performance tier, below CK /
  ck_tile / Triton / TileLang. Covers the three sub-levels (MFMA intrinsics →
  inline `asm volatile` → raw `.s`), the `__builtin_amdgcn_mfma_*` family,
  VGPR/AGPR occupancy and register-spill avoidance, `s_waitcnt` overlap, and the
  shipped aiter hand-asm ops to call before writing anything by hand.
---

# Hand-tuned MFMA Assembly (AMD MI300X / gfx942 / CDNA3)

This skill is the **lowest, highest-performance tier** of the Instinct kernel
stack. Reach for it only when: you need the last 10–20% over a library/DSL, a
fused op no template expresses, or you must diagnose why a higher-level kernel
underperforms. Everything here is verified for **gfx942 / CDNA3 (MI300X)**;
gfx950 / CDNA4 differences are flagged inline.

## Workflow

1. **Call the shipped aiter asm op first.** A from-scratch rewrite competes
   directly against aiter's hand-written kernels — if one already covers the op,
   calling it wins. Verified on-box ops and signatures: `docs/overview.md`
   ("Try the shipped aiter hand-asm op FIRST").
2. **If you must write it, pick the right sub-level** (the 3-tier recipe):
   - **MFMA intrinsics** (`__builtin_amdgcn_mfma_*`) — the default;
     scheduler-friendly, the only form the SW pipeliner / `SchedGroupMask`
     recognizes. Start here. → `docs/mfma_intrinsics.md`
   - **inline `asm volatile`** — a tight hand-scheduled micro-loop for the
     loads/`ds_read` interleave; **never** put MFMA itself here. → `docs/raw_asm.md`
   - **raw `.s`** — a peak micro-kernel only when disassembly proves LLVM's
     schedule is suboptimal *and* the kernel is hot enough to amortize it.
     → `docs/raw_asm.md`
3. **Choose the MFMA shape.** Default **16×16×16** (higher max-achievable FLOPs
   than 32×32×8 on power-limited MI300X; fewer C accumulator regs/lane).
   bf16 builtins need the **`_1k`** suffix on gfx942; f16 do not. Verify every
   intrinsic with the matrix calculator + a device-only compile. → `docs/mfma_intrinsics.md`
4. **Budget registers for occupancy.** 512 VGPR/lane shared VGPR+AGPR pool, 16-reg
   granules, 64 KB LDS/CU. Aim for ≥2 waves/SIMD. → `docs/register_alloc.md`
5. **Overlap memory with `s_waitcnt`** (`lgkmcnt(1)` before MFMA is the canonical
   prefetch pattern); pin the interleave with `sched_group_barrier`. → `docs/raw_asm.md`
6. **Check the disassembly, not the config.** `grep -E 'v_mfma|s_waitcnt|accvgpr|scratch_'`.
   `scratch_`/spurious `accvgpr` = the #1 failure: register spill collapsing
   occupancy. → `docs/pitfalls.md`

## Verify (always)

```bash
amdclang++ -x hip --offload-device-only --offload-arch=gfx942 -O3 -S kern.cpp -o kern.s
grep -E 'v_mfma|v_smfmac|s_waitcnt|s_setprio|accvgpr|ds_read|buffer_load|scratch_' kern.s
hipcc --offload-arch=gfx942 -Rpass-analysis=kernel-resource-usage ...   # VGPR/AGPR/LDS report
```

## Reference Documentation

The `docs/` subdirectory contains the detailed guides:

- `docs/overview.md` — CDNA3 execution model, the 3 sub-levels & when to drop to
  each, and the shipped aiter hand-asm ops to call first.
- `docs/mfma_intrinsics.md` — `__builtin_amdgcn_mfma_*` family, the
  compile-verified gfx942 dense MFMA table (note the bf16 `_1k` suffix), fp8
  K-density, and the CDNA4 block-scaled `mfma_scale` family.
- `docs/raw_asm.md` — `s_waitcnt` overlap semantics, the relaxed-count
  software-pipelining pattern, `s_setprio` / scheduling barriers, SMFMAC sparse,
  and the global→LDS→MFMA dataflow.
- `docs/register_alloc.md` — VGPR/AGPR split, occupancy math (worked example),
  MFMA fragment placement, and avoiding the AGPR move tax / spills.
- `docs/pitfalls.md` — the ranked anti-patterns: spills, MFMA-in-asm defeating
  the pipeliner, inline-asm clobber bugs, LDS bank conflicts, fp8 fnuz vs OCP,
  `s_waitcnt` off-by-ones.
