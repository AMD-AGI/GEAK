# Scheduling-model choice (layer 1.5) — where the scheduling intelligence lives

Who overlaps memory issue with MFMA issue — you (ring), the compiler, or the wave structure — is a choice
that precedes the pipeline layer, because it sets `warpsPerCTA`, the occupancy target and register use. Rung 3 of
`pipeline.md ### The overlap options and the signal for each`.

## Four paradigms, and why this target has three

| model | realize with | gfx950 status |
| --- | --- | --- |
| `authored_stage` | `commit_group` / `wait_group` ring (`pipeline.md ## Authoring the overlap yourself (the climb default on the Gluon path)`) | stock; dominant in production; works at 1 wave/SIMD |
| `compiler_interleave` | `llvm_fn_attrs` strategy + empty-asm fences (`instruction-scheduling.md`) | per-compile half stock on 3.8.0; `coexec` gfx1250-only |
| `inter_wave` | `gl.amd.warp_pipeline_stage` (`warp-pipeline.md`) | stock, Gluon-only, `num_warps >= 8`, 2 waves/SIMD; symbol 3.7.0+, phase offset GPU-verified race-free on 3.8.0 / 3.9.0 only (3.7.1: wrong results) |
| warp specialization | — | **absent** on 3.8.0: lowering hard-errors `Warp specialization is only supported on gfx1250, got <arch>` (compile-checked on gfx950) |

`authored_stage` creates the overlap; the other two schedule it and layer on top. (Older notes call
`compiler_interleave` `intra_wave`; `pipeline.md`'s "intra-wave" is a layer, not this model.) Ask per region:
does it need two instruction classes issued from the same wave (matrix + VALU, e.g. softmax)? Then only
instruction ordering can supply it.

## How to choose

Signals and preconditions per model (not a fixed order):

1. K-loop with staged operands → `authored_stage` fits (`## authored_stage — realize at the pipeline layer`).
   No loop → nothing to decide.
2. `inter_wave` on top requires `num_warps >= 8` (`warp-pipeline.md ## Gate 0: does the launch have two wave
   groups at all?`), 2 waves/SIMD, and ring waits that can sit between stages (3.8.0 rejects them inside a stage).
   Measured on a gfx950 / Triton 3.8.0 bf16 GEMM: the ring paid alone and the 8-warp markers added on top of
   it on warm (L2/MALL-resident) operands, but the marker gain vanished on one shape when every launch
   started from flushed caches — A/B under the deployment's cache state (a markers-without-ring build was
   not tested).
3. Accumulator read cadence: a read-modify-every-iteration accumulator (online softmax, gating) pays an AGPR
   round-trip per iteration if pinned there, so the AGPR-pinning hints are aimed at write-until-epilogue GEMM
   accumulators.
4. Toolchain knobs layer on top of whichever model is built. Recording the model you built (`authored_stage`
   when that is what it is) keeps later comparisons honest.

### When steps 1 and 2 both say no — the neither-model cell

1 wave/SIMD (large per-lane accumulator, e.g. large-head-dim attention) **and** a read-modify accumulator:
no second wave, and GEMM-only interleave passes can assert on VALU-between-matmul. Options: shed accumulator
liveness to buy the second wave back; author the AC/LR/DOT interleave by hand; sweep an `llvm_fn_attrs` ILP
strategy (this ILP-starved case is the signal for it). Record the model as **neither**; `authored_stage`
still applies.

## authored_stage — realize at the pipeline layer

`groups_per_stage` sets the `wait_group` arithmetic; `wait_group(N)` counts commit groups. On 3.8.0 the compiler emits an `s_barrier` after
every `wait_group` and before every refill, so ring barriers are not a source-level decision on 3.8.0
(`pipeline.md ### What the compiler inserts after wait_group: barriers and load_shared_relaxed`).

## inter_wave — realize with warp_pipeline_stage

Mechanism, gates, priority, boundary emission and authoring rules: `warp-pipeline.md`.

### inter_wave hazard: scaled-MFMA vs the shared-read bus (arch fact, low-precision path)

**Scaled MFMA (fp8/fp4) adds a hidden scale-load window, and paired SIMDs share LDS read bandwidth.**
ATT-measured on gfx950 (Triton 3.8.0 / rocprofiler-sdk 1.3.5, inline-asm microbenchmarks): SIMDs {0,1} and
{2,3} share LDS read throughput, so two waves reading in phase on a pair can turn a scaled `16x16x128` loop
LDS-read-bound (the MFMA interval stretches past its 32 cycles) where one wave, or two waves on unpaired
SIMDs, are not — with zero bank conflicts. Signal: shared-read stall with the bank-conflict counter at
zero. **A wider MFMA shape is a candidate** (fewer LDS bytes per FLOP; a scaled `32x32x64` kept its
interval with all four SIMDs reading in the same ATT trace), but in the one fp8 GEMM loop timed (gfx950 /
3.8.0) `32x32x64` fed from the same LDS layout introduced bank conflicts and was slower than `16x16x128`; it
needs the LDS layout re-derived for the new shape. The compiler re-tiles the same K work, so operand
layouts, bank conflicts and register pressure are the things to check (window is per shape — `../hardware/isa-mechanisms.md`).
Per-step scale reads from LDS add their own shared-read cost; keep scales register-resident when they are
reused. **Measure one- vs two-wave rather than assuming a loss:** in an LDS-fed fp8 `mfma_scaled` GEMM loop
(gfx950 / Triton 3.8.0) two waves/SIMD raised the LDS wait share and still ran faster than one. The
single-wave candidate is the alternative. AGPR forcing on an 8-warp ping-pong does not lower residency: the
CTA needs 2 waves/SIMD, so `N >= 256` fails at launch and a smaller N only re-splits the 256 budget
(`llvm-fn-attrs.md ### AGPR budget: amdgpu-agpr-alloc=N (intermediate values)`).
