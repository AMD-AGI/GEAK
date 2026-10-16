---
title: FlyDSL authored-kernel optimization
kind: language
gens: [gfx942, gfx950]
updated: 2026-10-16
source_commit: AMD-AGI/GEAK@c0a1f937
---

# Optimize an authored FlyDSL kernel

Read the complete `@flyc.kernel` and `@flyc.jit` call, including measured helper
stages and the numerical/state contract. Choose a change that addresses a measured
cost or tests a specific hypothesis. Kernel names, pass counts and fusion labels
do not establish a structural optimum; local tuning and reformulation can both help.
For existing aiter wrappers, use [knobs](knobs.md) and [patterns](patterns.md).

## Changes to consider

| Question | Candidate change and tradeoff |
|---|---|
| Can work or materialization be removed? | Fusion, reuse or equivalent decomposition; account for added registers, synchronization and recomputation |
| Are accesses inefficient? | Layout, vector width or coalescing; preserve alignment and masks |
| Is data reused across threads? | LDS staging; compare reuse benefit with extra movement and barriers |
| Is memory latency exposed? | Earlier loads, more independent requests or pipelining; check carried state and resource cost |
| Does the generated loop differ from the intended schedule? | Loop structure, MFMA shape or scheduler hints; verify the resulting lowering |

Prefetch can increase memory-level parallelism even in a load-heavy loop; its
benefit needs measurement. Stall locations suggest hypotheses, not unique causes.
LDS allocation pressure, bank conflicts and dependency latency require different
changes. Keep producer/consumer swizzles consistent and use target-specific limits.

## FlyDSL rewrite details

- `range_constexpr()` unrolls at tracing time. Runtime loops use the supported
  `range(..., init=...)`/`scf.for` path. Keep carried state aligned across init,
  yield and results; lower-level scf APIs require index bounds and raw `ir.Value`s.
- Cached `SmemPtr` views must obey SSA dominance. Recreate views in the consuming
  region; the imported source's `_view_cache` workaround is version-specific.
- `buffer_ops.buffer_load`/`buffer_store` offsets use elements of the selected
  dtype. Reinterpreting FP8 byte addresses as `i32` requires dividing by four.
- Allocate shared memory through `SmemAllocator`; call `allocator.finalize()` in
  the GPU module body. Include all fixed/dynamic allocations in the resource check.
- A pipelined loop needs prologue, carried payload/metadata and a valid tail.
  Guard invalid addresses and preserve masking; clamping a pointer alone does not
  make an invalid lane's contribution correct.
- When changing `Constexpr` arguments across recompiles, pass raw `torch.Tensor`
  inputs or rebuild the `flyc.from_dlpack()` wrappers. Reusing a cached wrapper
  can retain stale specialization/layout state in the imported runtime path.
- Preserve the task's public signature and edit boundaries. Wrapper changes may be
  needed for an equivalent implementation; this guide imposes no kernel-only rule.

Preserve the declared numerical behavior. FNUZ byte `0x80` represents NaN, but
ANDing every FP8 byte with `0x7F` destroys the sign of valid negative numbers.
Do not add NaN replacement, saturation or a different rounding rule unless the
contract permits it. Tile divisibility depends on the chosen instruction/layout,
not one universal multiple.

## Inspect and measure

Use the task's accepted correctness and unprofiled full-call timing. For a change
whose premise depends on lowering, a targeted dump can help:

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/flydsl-inspect python isa_probe.py
rg 'vgpr_count|agpr_count|private_seg_size|v_accvgpr|scratch_' /tmp/flydsl-inspect
```

The probe and dump filenames depend on the installed version. Bind the dump to the
measured source/configuration. Register counts and scratch allocation are not
achieved occupancy; `v_accvgpr` moves are register movement, not automatically
memory spills. A lower scratch allocation can coexist with more register shuttling
and slower timing. A small private segment may be a fixed/prologue allocation;
inspect its accesses before calling it a spill. See the [retained MLA example](../../operators/mla_attention/backends/flydsl.md)
for the source-specific observations. Compare changed work, movement and latency
before assigning cause.
Use [GEMM](authoring_gemm_levers.md), [attention](authoring_attention_levers.md) or
[debugging](debugging.md) details when relevant; no full diagnostic suite is required
for every edit.

## Sources

- Imported source: `AMD-AGI/GEAK@c0a1f937:src/minisweagent/skills/flydsl/docs/flydsl_optimization.md`
- [Tile programming](authoring_tile_programming.md)
- [AMD matrix-core programming](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [Hardware evidence](../../skills/kernel-hardware-evidence/SKILL.md)
