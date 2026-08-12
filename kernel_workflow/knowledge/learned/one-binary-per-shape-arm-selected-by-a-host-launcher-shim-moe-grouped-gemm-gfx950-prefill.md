---
key: int4 W4A16 packed-weight fused-MoE grouped GEMM on gfx950/MI355, Triton + immutable host, several n-width shape arms in one launch
type: lever
confidence: ★★
effect: the shared-binary control (one kernel, runtime branch) is 4.9% slower on the small-M arm (M=2048) than two specialized entries; per-arm launch constants that only the shim can set add +3.0-3.4% on M=32768/65536, all non-overlapping vs frozen baseline
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 4
toolchain: unknown
last_seen: 2026-08-12
name: one-binary-per-shape-arm-selected-by-a-host-launcher-shim-moe-grouped-gemm-gfx950-prefill
description: Split an int4 W4A16 MoE grouped GEMM into per-n-width Triton entries picked by a host launcher shim; the shim then owns each arm's launch constants.
keywords: ['moe-grouped-gemm', 'w4a16', 'int4-dequant', 'split-entry', 'launch-config', 'waves-per-eu', 'num-warps', 'triton', 'gfx950']
kernels: ['fused_moe_kernel_gptq_awq']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: prefill
layer: learned
lifecycle: active
cost: L2
verified_on: 2026-08-12
---
# One binary per shape arm, selected by a host launcher shim
- lever: give each n-width shape arm its own @triton.jit entry chosen on the host, rather than one kernel with a runtime branch: VGPR allocation and I-cache footprint are whole-binary properties, so a shared binary taxes whichever arm takes the narrow path.
- apply: a thin launcher shim in front of the (often immutable) host call selects the entry and owns the launch kwargs dict per arm - num_warps, waves_per_eu, loop_unroll_factor - so each arm is tuned against its own codegen.
- stack: total 2.80x geomean isolated (director-verified, accepted), many directions over 24 rounds, incremental in landing order
  - 1. per-arm split entries + shim-owned launch constants - the enabling mechanism; shared-binary control 4.9% slower on the small-M arm, launch-constant pair +3.0-3.4% on top
  - 2. dequant-tile fusion width (see the fusion-width card) - +24.4% then a further +23.8%
  - 3. K-loop schedule: single-stage tl.range + per-arm unroll factor - ~18-20% on the large-M cases
  - note: attribution is incremental in landing order, not independent; directions interact.
- verify: A/B each arm interleaved against a byte-identical control binary, and confirm the arm predicate engaged by diffing the generated AMDGCN - an unrecognised launch option is silently swallowed, so absence of an error proves nothing.
- pitfall: a constant that applied cleanly across arms read as compatible -> a tuning constant belongs to the codegen it was tuned against -> re-tune per arm, and sweep the pair jointly (num_warps and waves_per_eu both had to move DOWN together; no one-axis sweep finds that point).
a widened arm printed 1.6-1.8x while skipping work -> the perf path printed a ratio with no correctness binding -> gate any ratio above ~1.2 behind an md5-bound correctness token at print time.
- caution: also verify the shared-binary control is measured on the same session, since the split only pays once the narrow arm stops carrying the wide arm's register budget.
- source: run kernel_20_geak_0808_16h, proposal fused_moe_int4_w4a16-own16h, 2026-08-12; director-validated geomean 2.8019, correctness 8/8
