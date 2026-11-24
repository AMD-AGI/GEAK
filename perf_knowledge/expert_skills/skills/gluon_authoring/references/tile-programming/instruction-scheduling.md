# Instruction scheduling — pacing what the overlap already created

Intra-wave half of rung 3 (`pipeline.md ### The overlap options and the signal for each`): it decides where in the
stream already-overlappable instructions land, so it has the most to act on once a ring/prefetch exists —
on a loop with nothing to interleave it measured as noise in the cases recorded. `llvm_fn_attrs` is applied in `make_llir`, shared by plain and
Gluon, so nothing here needs a shim or rebuild. Inline-asm API: `../gluon/inline-asm-reference.md`;
co-issue rates: `../hardware/isa-mechanisms.md ## Instruction-rate facts that flip a lever's sign (CDNA3/4)`.

## What production actually reaches for

| mechanism | gfx950 production use | controls |
| --- | --- | --- |
| `warp_pipeline_stage` markers | the dominant inter-wave construct | stage clusters, borders, `s_setprio` (`warp-pipeline.md`) |
| empty-asm scheduling fence | the intra-wave construct found most (direct sites and a `sched_barrier` fallback helper) | a data dependency the scheduler must respect; no instruction |
| placed `s_nop` | dependency ladders (DPP chains), after lane-broadcast reads | explicit delay |
| `llvm_fn_attrs` strategy | a few kernels / wrappers, chosen per kernel | LLVM machine-scheduler strategy per compile |
| hand-rolled `s_setprio` | HIP / FlyDSL ring kernels; none found in current aiter Gluon kernels | wave arbitration priority |
| stock `coexec` | gfx1250 only (automatic at `num_warps <= 4`) | not reachable on gfx950 / gfx942 |
| `sched_group_barrier` / `iglp_opt` / `schedule_hint` | **none** in the kernels surveyed (no user-facing surface on 3.8.0 Gluon) | see `## The declarative hints that are not there` |

Survey basis: the local aiter gfx950 Gluon kernels plus a code search; counts move as the corpus does, so recount
on the corpus you use.

## llvm_fn_attrs — the portable, per-compile scheduler strategy

Owned by `llvm-fn-attrs.md` (accepted names, failure modes, assembly-diff acceptance, the gfx950 recipe).
Here only: it is 3.8.0-only (earlier versions raise an unrecognized compile option), it is decided per kernel,
and the ILP strategies paid where the kernel was ILP-starved (1 wave, independent work left unscheduled) while
`iterative-minreg` paid through occupancy — the bound class says which to sweep.

## The empty-asm scheduling fence — the most-used and least-documented lever here

Empty asm with a tied operand is a compiler motion boundary, not a hardware fence (it emits no instruction, so it does not replace
`gl.barrier()`, `wait_group` or `s_waitcnt`; adding `~{memory}` only widens what the compiler may not
move across):
`acc = gl.inline_asm_elementwise("", "=v,0", [acc], dtype=acc.dtype, is_pure=False, pack=1)` —
`is_pure=False` is required (it registers read+write memory effects with the compiler, which stops passes
from moving or deleting the block; a pure block with an unread result is deleted — compile-checked 3.8.0 gfx950); one tie covers the whole
tensor (one empty `asm sideeffect` per register element). Production example, if aiter is available: `aiter/ops/triton/_gluon_kernels/gfx950/attention/sparse_mla_fp8.py`.
Its effect shows as placement in the disassembly; the response was non-monotone where swept, and a win is
tied to the compiler version it was measured on.

## `s_nop` placement — a pacing lever, not a hazard fix

A placed `s_nop` is one you can find in your own asm strings; an `s_nop` that appears only in the asm census
is a compiler-inserted hazard wait; removing the dependency addresses it, adding delay does not.

## Wave priority — the hand-rolled form, and its four roles

Bare `s_setprio` via inline asm, used where the marker path cannot go on 3.8.0 (waits inside a stage fail
conversion; one wave group at `num_warps <= 4`). The memory-high polarity is a marker-path convention;
single-group kernels ship both polarities. Hand-rolled `s_setprio` is rare in public Gluon/Triton kernels:
across aiter, sglang, vllm and the gfx950 Gluon tutorials only aiter's paged MQA-logits kernels use it
(single-group, compute-high: raised over the MFMA + interleaved-load window, dropped before the
reduction/store) — and their `s_set_prio` / `sched_group_barrier` import does not exist in stock Triton
3.6-3.9, so it falls back to a no-op there: check the ISA for `s_setprio` before crediting it. Every explicit
marker-path priority surveyed is memory-high. Roles reported by the upstream pack's survey (its corpus is not
identified here; per-shape windows also appear in FlyDSL kernels):
memory-high (raised across wait/LDS-read/barrier/copy, dropped before the MFMA), compute-high (inverse),
blanket (whole loop raised), whole-CTA arbitration (raised once, never lowered). Polarity, window width and
transitions are per-shape parameters (one prefill GEMM uses two narrow windows below an M threshold and one
wide window above it).

**Rule (v3.8.0 source): for the lowering to move the raise past the first MFMA, it must be the op immediately
preceding the MFMA.** MFMA lowering looks for a `SetPrioOp` with
`getPrevNode()` (`v3.8.0:third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp`, `convertDot` and
`convertScaledDot`); one statement in between silences it. The marker path places it itself. No surveyed
compilation unit combines a stage marker with hand-rolled `s_setprio`.

## How long an LDS burst may be before placement starts to matter — a countable precheck

The 8-entry-per-SIMD-pair LDS queue hypothesis was not supported on gfx950 (ATT, rocprofiler-sdk 1.3.5): one
wave queued about 16 `ds_read_b128` before issue stalled, fewer when a paired SIMD or all four SIMDs read at
once (`../hardware/isa-mechanisms.md ## LDS instruction behavior (gfx950)`). So a burst length alone does not
mark a cluster; what limits a burst is shared read bandwidth (SIMD pairs, all-SIMD streaming) and LDS-DMA data
landing. Count `ds_read` + `ds_write` runs per cluster (`kernel_workflow/scripts/kernel_tools/asm_loop_audit.py --opcodes 12`) as an inventory,
and A/B interleaving non-LDS work into long bursts (rather than adding loads). It is not an effect size.

**Scalar-operand placement** (a `descale` / `softmax_scale` load before/after the first async wait, operand
prefetch or the first matmul) costs no register or LDS; one production family exposes it as a four-valued
`constexpr`. Registered, not priced — a sweep is what prices it.

## The declarative hints that are not there

**`sched_barrier`, `sched_group_barrier`, `iglp_opt` have no user-facing surface on 3.8.0** (found only in
Proton instrumentation; absent from `gl.amd.cdna3` / `.cdna4`, introspection-checked) and appear in no
surveyed production kernel. A shipped
kernel imports them in `try/except` with **no-op stubs** — its hints are dead code.
`sched.barrier` mask 0 is reachable only as a side effect of a `warp_pipeline_stage` border (you do not choose
the mask); a marker written only for that wall is legitimate
(`warp-pipeline.md ### Before reading a "no" as a defect: are you here for the wall, not the offset?`).
Everything finer is inline asm. The `sched_group_barrier` masks (`0x008` MFMA, `0x020` buffer load, `0x100` DS
read, `0x200` DS write) describe the LLVM intrinsic, not a Triton API.

## Acceptance

A scheduling regression can be a layout question, which is worth ruling out before closing the direction.
The static read is what attributes a change: `asm_loop_audit.py` for the hot-loop mix, `mfma_efficiency.py`
for cadence, then a same-window A/B. A timing win with no change in the hot loop's op-class
stream and wait placement (empty asm diff, or register renaming only) is not attributed to scheduling. A pure
reorder keeps the instruction histogram, so compare order and waits (`asm_loop_audit.py` emits both), not
counts.
