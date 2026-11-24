# Warp-pipelining — the wave-level schedule, and what the compiler does with it

Read once `scheduling-model.md` picked **inter_wave** (rung 3 of `pipeline.md ### The overlap options and the signal
for each`). `gl.amd.warp_pipeline_stage` splits a CTA's waves into two groups held one stage out of phase,
so one group's compute covers the other's memory. Written for gfx950 / Triton 3.8.0.

**Availability:** 3.7.0+ (`gl.amd.warp_pipeline_stage` and `add_warp_pipeline` are absent on 3.6.0 — one
`hasattr`). **Gluon-only by construction (3.8.0 `compiler.py`):** `add_warp_pipeline` is called once, inside
`gluon_to_ttgir`; `make_ttgir` does not call it, and a `with gl.amd.warp_pipeline_stage(...)` block inside `@triton.jit`
does not compile at all (`ir.builder has no attribute create_warp_pipeline_border`; 3.8.0, gfx950). The pass bails on a
loop without markers. Lowering (`cond_barrier`, `s_setprio`, walls) happens later in `make_llir`, so "the
marker survived" and "the schedule was built" are separate checks. The marker path is GPU-verified race-free
here on 3.8.0 and 3.9.0 only. On 3.7.1 the conversion emitted no in-loop stage-border barriers or scheduling
walls (and no `s_setprio` for `priority=`), and even the corrected snippet below returned wrong results — do not
rely on the phase offset on 3.7 builds.

## Gate 0: does the launch have two wave groups at all?

**Rule (3.8.0, compile-checked on gfx950; the emitted group test is `tid < 256` at 4 and 8 warps alike): the
phase offset needs `num_warps >= 8`; at `num_warps=4` it silently does not exist.** The pass
**never reads `num_warps`**: group size = `warpSize * 4` threads, `warpIDX = tid.x / (warpSize * 4)`, so
groups = `num_warps / 4` (warpSize cancels — the threshold is 8 warps on wave64 and wave32 alike). At ≤ 4
warps `warpIDX` ≡ 0, `cond_barrier(warpHigh)` never fires, and nothing diagnoses it; markers still compile and
emit their cluster barriers, group test and `s_setprio` — a setprio count proves the markers lowered, not that
two groups exist.

- The upstream gfx9 test artifact, `test_warp_pipeline_gfx9.py` (3.8.0), pins `NUM_WARPS = 4` with
  `arch="gfx950"` — it grades the wall, not the offset. Its marker spelling and its
  `tl.range(..., loop_unroll_factor=)` use transfer; its launch does not exercise the offset. Upstream examples in
  `third_party/amd/python/examples/gluon/` are gfx1250-specific (WMMA + TDM + `mbarrier`).
- **2 waves/SIMD ≠ 8 warps/CTA:** two 4-warp CTAs on one SIMD satisfy occupancy and still give one group.
- The stock `coexec` hook fires only at `num_warps <= 4` and only for gfx1250 (3.8.0 `compiler.py`); the offset
  only at ≥ 8. On gfx950 the scheduler-strategy knob available is the `llvm_fn_attrs` strategies, at any
  `num_warps` (`llvm-fn-attrs.md`).

## Gate 1: is this kernel a candidate?

| ask | answer that predicts little gain |
| --- | --- |
| matrix-issue bound? | memory-bound — the barriers and register pressure are what remains |
| LDS staging in the loop? | none — nothing for the other group to overlap (markers still compile: silent) |
| 2 waves/SIMD reachable? | register/LDS caps it at 1 — one stream, nothing to interleave (silent) |
| scaled MFMA? | not a predictor of little gain: paired SIMDs ({0,1}, {2,3}) share LDS read bandwidth and two in-phase groups raised the LDS wait (ATT, gfx950), yet two waves/SIMD ran faster than one in the LDS-fed fp8 loop measured (gfx950 / 3.8.0) — time one wave vs two (`scheduling-model.md ### inter_wave hazard: scaled-MFMA vs the shared-read bus (arch fact, low-precision path)`) |

GEMM: two stages. Attention: four stages (softmax is a third instruction category). Elementwise /
reduction: no matrix stream to overlap. Decode / large per-lane accumulator: usually 1 wave/SIMD
(`scheduling-model.md ## When steps 1 and 2 both say no — the neither-model cell`).

### The fifth question: is this better than spending the same resources on the ring?

Two groups halve the register file per group at equal occupancy — the comparison is deep ring at 1 wave/SIMD
vs shallower ring at 2. **GPU-measured on one gfx950 fp16 GEMM family (upstream measurement): the 4-wave build
(deeper ring + instruction scheduling) beat the 8-wave two-group build at the K values tested**; the 8-wave build
led on in-loop MFMA efficiency only at the shortest K and ran at far fewer VGPRs. **The verdict reversed across
toolchain pins** of the same two kernels, and on other kernels the 8-wave build has paid — it is a per-build,
per-kernel A/B with the build identity recorded, run against the ring and instruction-schedule alternatives.
On stock Triton 3.8.0 / gfx950, on one upstream ring-kernel family (bf16 256x256 tiles, 4-wave builds
at 256x256 / 256x128, without the tutorial's scheduler / assembler tooling) the opposite held: the plain
8-warp ring beat those 4-wave builds at every K tried, and the warp-pipelined 8-warp build led once K was
long (it lost at short K, and from flushed caches its lead started at a longer K). On this page's own
snippet kernel (fp16, same build) the order differed again: a 4-warp ring at 128x128 with 2 CTAs/CU beat the
marker-less 8-warp ring, and the fixed warp-pipelined build led the best 4-warp ring narrowly, with
overlapping ranges on one shape. Per-kernel A/B.

### Before reading a "no" as a defect: are you here for the wall, not the offset?

`sched.barrier` has no Gluon spelling on 3.8.0; a stage border emits one, so a marker can be written purely as a
**scheduling wall** (often `num_warps` 1 or 4, `static_range`, no LDS, even no MFMA). At `num_warps=1` it is a pure
wall; at 4+ warps each border also emits an `s_barrier`, so it is a wall plus a CTA barrier (compile-checked 3.8.0,
gfx950). Grade such a kernel as a
fence — Gate 1 does not apply.

- **The wall is laid at parse time:** the border marker is itself `ROCDL::SchedBarrier(0)` when the frontend
  builds it, so it survives a declined loop, a one-group launch or a closed gate (except under a constexpr
  condition that traces false).
- **Clustering is not refunded by an inactive offset:** each boundary emits its barrier triple and the prelude an
  unconditional local `ttg.barrier`.
- **`s_setprio` is emitted only if some stage passes `priority=`** — the mnemonic tracks `priority=`, not
  intent. Settle offset-vs-wall on `num_warps`.
- **The barrier exemption does not recognise inline asm:** a boundary already carrying a `MemWaitOpTrait` op
  (`ttg.async_wait`) is wrapped with two `sched_barrier(0)` and no new barrier; a hand-issued `s_waitcnt`
  lacks the trait, so the full barrier set is added (`pipeline.md ### Draining below the group: bare
  s_waitcnt instead of wait_group`).
- **A census of markers is not a census of clusters:** three authored stage sites can become twelve clusters.

## When the gates say no: partition space instead of offsetting time

Without ping-pong the waves can be split **spatially**. gfx950 prefill attention (aiter `mha.py`, AMD
`fmha_v3`) splits along query rows (`warps_per_cta=[num_warps, 1]`) until each wave's M slice fills the MFMA M
extent (`BQ / warps_per_cta[0]`), so the softmax reductions stay inside each wave. When M is too small for
that (decode, small GQA head groups) and the grid does not fill the machine, the remaining waves can
take key slices, each with a private `(m, l, acc)` and one rescaled merge after the loop (GPU-measured on
gfx950 / Triton 3.8.0: key slices paid on an under-filled decode grid and not on a full one; for prefill
the M split was clearly ahead; gating/quantizing epilogues then belong after the merge);
that loop needs no barrier or wait. Gluon has no `warp_id` / `lane_id` and every tile→scalar route is a whole-tile
reduction, so a per-wave predicate (producer/consumer roles) is not expressible; warp specialization
hard-errors on gfx950 (3.8.0: "only supported on gfx1250", compile-checked) (`scheduling-model.md ## Four paradigms, and why this target has three`).

## The phase offset, and how it is built

`amdg.cond_barrier` barriers only the selected waves, diverges on purpose (needs reconvergence) and sets **no
memory fence**. Pre-loop: local barrier, then `cond_barrier` holds group `warpIDX != 0`; every stage boundary
re-locks the phase; post-loop: the complementary `cond_barrier` reconverges. An `s_barrier` under an
`exec`-masked branch before the loop is the phase shift, not a bug. Priority toggling alone (legacy
ping-pong) leaves synchronization untouched; priority is a stabilizer on top of the rendezvous.

## Stage priority: the shipped memory-high convention

```python
with gl.amd.warp_pipeline_stage("mem", priority=1):     # LDS reads, refills, address VALU
    ...
with gl.amd.warp_pipeline_stage("mfma", priority=0):    # the matrix ops
    ...
```

Both clusters contain VALU; the stated rationale for memory on top is that with compute on top the memory
wave's address updates can starve and the overlap shrinks. This is the convention AMD's gfx950 tutorial GEMMs
and aiter ship on the **marker path** (two groups contend). On a gfx950 / Triton 3.8.0 two-stage
warp-pipelined GEMM, memory-high, compute-high and equal priorities measured the same across the shapes and
tiles tried — the inverse did not collapse the overlap; the stage markers themselves were the lever. Memory-high
is the shipped convention; on a two-stage GEMM polarity measured as a low-impact knob. Hand-rolled `s_setprio` usually runs
single-group (`num_warps` 1 or 4) where both polarities ship (`instruction-scheduling.md`). The gfx950
tutorial attention kernels pass **no** priority, so no `s_setprio` is emitted (see the `priority=` rule above).

- Range 0–3; unprioritized stages reset to 0 once any stage sets one, none emitted if none does
  (`triton/experimental/gluon/language/amd/warp_pipeline.py` docstring) — dropping `priority=` from one
  stage silently pins it at 0.
- **On CDNA, MFMA lowering moves a `s_setprio` that immediately precedes a matrix op to after the first
  MFMA.** A toolchain re-pin once moved `setprio` relative to the loop's wrap-around barrier and cost several
  points of in-loop MFMA efficiency on attention with no kernel change (`../pitfalls/platform-known-issues.md`)
  — record the build with any priority measurement.

**Measured exception: the four-stage softmax-attention loop.** There the compute stages carry the
softmax VALU and the memory stages little address math; on gfx950 / Triton 3.8 matrix stages at 1, memory
at 0 (with P's `exp2` pinned in the PV stage) beat the memory-first polarity; without the pin neither
polarity changed time (gfx950 / 3.8.0). Treat the polarity as a
measured parameter on attention, not this section's default
(`../workloads/attention.md ### 8-warp four-stage ping-pong: when it pays`).

## The launch configuration

The model is 4 SIMDs × 2 waves: `num_warps = 8`, typically `waves_per_eu = 2`. Doubling the waves changes
`warpsPerCTA` `[2,2]` → `[2,4]`; with layouts built parametrically from the wave-grid constants (as AMD's
tutorial GEMM does) only the global-load layouts changed there (shared, dot-operand, MFMA layouts reused). Every layout's `warpsPerCTA` must
equal `num_warps` (*"Layout has 4 warps per CTA, but the context requires 8"*).

**Side operands that inherit the warp tiling fail quietly.** The scale tensors of a scaled-MFMA kernel are
derived per warp, so doubling warps along a dimension that splits the scale (e.g. warps along M for A scales)
**halves the bytes per thread** (compile-checked 3.8.0 with `get_mfma_scale_layout`; adding the warps along the
other dimension left it unchanged); in one recorded port, below the minimum width of the `ds_read_b64_tr` the
4-wave kernel used, the read decomposed into narrow reads plus lane shuffles — no error, just low MFMA
efficiency (that decomposition was not reproduced in a minimal scale path). Check the scale reads in the `.s`
after changing `warps_per_cta`. One fix: load the full un-split scale block and recover halves with a
register-only slice (free when it keeps the source layout). Bytes/thread of every side operand at the new
wave grid, and achieved occupancy of 2 waves/SIMD, are the two things to check when porting.

## Writing one: the six steps, and a skeleton to copy

1. Start from a working non-pipelined kernel's hot loop. 2. Two stages unless there are several dependent
compute phases (`## Partitioning stages`). 3. Every body op inside a stage. 4. Each async wait on the border
*before the compute stage* — after the memory stage that issued the copies — so every group has waited for its
own copies of a tile one border before any group's next memory stage reads it (`## LDS hazards close a stage
early`). 5. Multi-buffer with prologue and epilogue. 6. `num_warps=8`, `waves_per_eu=2`; confirm occupancy.

Points to check when writing it (3.8.0): the loop is short by the tiles in flight and the epilogue (outside any stage) owes them;
the wait sits outside both `with` blocks (on the `mem` -> `mfma` border); address arithmetic goes in the memory stage; a real `range`
(a `static_range` unrolls into a straight-line ping-pong, `## Authoring rules`). gfx950 async surface: `buffer_load_to_shared` + `commit_group` / `wait_group(N)`, `N` in
commit groups. Runnable: `scripts/pipeline_examples_cdna4.py` C5 (`num_warps=8`, `warpsPerCTA=[8,1]`, `[64,
32]` tile); gfx942 downgrade: `pipeline_examples_cdna3.py` B1 (sync staging, ordering from the border
`s_barrier`s). Neither has a `gl.barrier()` in the pipelined loop (see `## Authoring rules`).

### A GPU-verified gfx950 snippet (Triton 3.8.0 and 3.9.0, `num_warps=8`)

fp16 GEMM mainloop, 2-deep `buffer_load_to_shared` ring, two stages, memory at the higher priority; one 128-bit
direct-to-LDS access per lane. Assumes `K % BK == 0` and one output tile — a structure to copy. Race-tested on
gfx950 with random inputs and a NaN-filled output, bit-identical to the same body without markers at 128x128 and
256x256 over several K (including an odd tile count).

```python
import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.experimental.gluon.language.amd.cdna4 import async_copy as ac

@gluon.jit
def wp_gemm(a_ptr, b_ptr, c_ptr, K,                 # A [BM, K], B [K, BN], C [BM, BN]; fp16, row-major
            BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    NW: gl.constexpr = 8                              # two 4-warp groups -- the offset needs >= 8
    # one 128-bit (8 x fp16) direct-to-LDS access per lane per instruction; repeats only along rows
    LA: gl.constexpr = gl.BlockedLayout([1, 8], [64 // (BK // 8), BK // 8], [NW, 1], [1, 0])
    LB: gl.constexpr = gl.BlockedLayout([1, 8], [64 // (BN // 8), BN // 8], [NW, 1], [1, 0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    MF: gl.constexpr = gl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 16], transposed=True,
                                            warps_per_cta=[NW // 2, 2])
    a_s = gl.allocate_shared_memory(gl.float16, [2, BM, BK], SH)
    b_s = gl.allocate_shared_memory(gl.float16, [2, BK, BN], SH)
    oa = (gl.arange(0, BM, layout=gl.SliceLayout(1, LA))[:, None] * K
          + gl.arange(0, BK, layout=gl.SliceLayout(0, LA))[None, :])
    ob = (gl.arange(0, BK, layout=gl.SliceLayout(1, LB))[:, None] * BN
          + gl.arange(0, BN, layout=gl.SliceLayout(0, LB))[None, :])
    acc = gl.zeros([BM, BN], gl.float32, MF)
    ac.buffer_load_to_shared(a_s.index(0), a_ptr, oa)           # prologue: tile 0
    ac.buffer_load_to_shared(b_s.index(0), b_ptr, ob)
    ac.commit_group()
    ac.wait_group(0)                                             # tile 0 landed before the loop
    n = K // BK
    for k in range(n - 1):                                       # a real (scf.for) loop
        with gl.amd.warp_pipeline_stage("mem", priority=1):     # memory stage outranks compute
            a = a_s.index(k % 2).load(gl.DotOperandLayout(0, MF, 8))
            b = b_s.index(k % 2).load(gl.DotOperandLayout(1, MF, 8))
            ac.buffer_load_to_shared(a_s.index((k + 1) % 2), a_ptr, oa + (k + 1) * BK)
            ac.buffer_load_to_shared(b_s.index((k + 1) % 2), b_ptr, ob + (k + 1) * BK * BN)
            ac.commit_group()
        ac.wait_group(0)                                         # mem -> mfma border: tile k+1 landed in
                                                                 # this group's copies before the border that
        with gl.amd.warp_pipeline_stage("mfma", priority=0):     # precedes the other group's next mem stage
            acc = gl.amd.cdna4.mfma(a, b, acc)
    ac.wait_group(0)                                             # epilogue: last tile, outside any stage
    acc = gl.amd.cdna4.mfma(a_s.index((n - 1) % 2).load(gl.DotOperandLayout(0, MF, 8)),
                            b_s.index((n - 1) % 2).load(gl.DotOperandLayout(1, MF, 8)), acc)
    oc = (gl.arange(0, BM, layout=gl.SliceLayout(1, MF))[:, None] * BN
          + gl.arange(0, BN, layout=gl.SliceLayout(0, MF))[None, :])
    gl.amd.cdna4.buffer_store(acc.to(gl.float16), c_ptr, oc)

# launch: wp_gemm[grid](a, b, c, K, BM=256, BN=256, BK=64, num_warps=8,
#                      llvm_fn_attrs=[["amdgpu-agpr-alloc", "0"], ["amdgpu-sched-strategy", "iterative-ilp"]])
```

**Where the wait goes is a correctness decision, not a scheduling one.** `wait_group` covers only the calling
wave's copies, and the second group runs one border behind. With the wait at the top of the loop (the border
right before `mem`), the leading group reads tile k+1 before the trailing group has waited for its share of
it — wrong results on every K tried (gfx950, 3.8.0 and 3.9.0), though it compiles and the register numbers below
are identical. The wait on the `mem` -> `mfma` border is ordered before the barrier the leading group crosses into
its next `mem`. Two other shapes ship and are argued race-free from their ordering: a deeper wait count whose tile was
published one stage earlier (aiter `mha.py`), and copy ownership — each group reads only LDS it copied itself, or the leading group copied
and waited for a stage earlier (one GEAK a16w16 kernel). The C5 example in `pipeline_examples_cdna4.py` is safe
for a different reason (each thread reads exactly the elements it copied); a GEMM's dot-operand reads never are.
A deeper count at the loop top that relies on the copy's age alone (wait count 1 before `mem`, ring depth 3)
is latent, not safe: it passed the race tests on gfx950 / 3.8.0 only because that copy was issued two
stages earlier, and the ordering argument does not cover it.

**Observed pitfall: a masked K tail into LDS.** In the race tests (gfx950, 3.8.0) a masked
`buffer_load_to_shared(..., mask, other=0.0)` K tail gave wrong results even without markers — masked-off lanes
appeared to leave stale LDS data rather than `other` — and was correct once the tail was zeroed after the LDS
read. Not root-caused; zero the tail on read (or check the masked copy by copy-back) rather than relying on
`other=` for an LDS destination. A K that is not a multiple of 16 (fp16) does not lower to the direct-to-LDS
copy at all (32 or 128 bits per lane; `unrealized_conversion_cast`).

CPU AOT compile (the register numbers are the same for the shipped and corrected loop) for
`GPUTarget("hip", "gfx950", 64)` — pass `tt.divisibility=16` for the pointers **and `K`**
or the async copies fail with `unrealized_conversion_cast` (the JIT supplies this):

| launch | `.llir` | ISA |
| --- | --- | --- |
| `BM=BN=256`, defaults | 3 `llvm.amdgcn.s.setprio`, `icmp ult %tid, 256` | `.agpr_count: 128`, `.vgpr_spill_count: 12`, 177 `v_accvgpr_*` |
| `+ [["amdgpu-agpr-alloc", "0"]]` | `"amdgpu-agpr-alloc"="0"` on `attributes #0` | `.agpr_count: 0`, **spill 4**, no `v_accvgpr_*` |
| `+ ["amdgpu-sched-strategy", "iterative-ilp"]` too | both attributes | `.agpr_count: 0`, **spill 0** |
| `BM=BN=128`, defaults | same markers | no AGPRs, no spill |

`amdgpu-agpr-alloc=0` alone still spills here; the pair with `iterative-ilp` reaches spill 0. `"0"` sets only
the AGPR minimum (read `.agpr_count`); `"0,0"` is a hard cap — semantics in `llvm-fn-attrs.md`. Timing decides.

### How long the drain is, and why the skeleton's formula does not generalize

Drain and prologue = pipeline depth − 1 (tile indices live per iteration), not `NUM_BUFFERS − 1`: the
four-stage attention loop is depth 4 on two LDS buffers (`pipeline.md ### Hand-built buffering rules
(correctness + scheduling footguns)`). Prologue and drain grow with depth, which weighs more on short loops.

## Authoring rules

Each is a conversion failure or a silent loss (3.8.0; the error strings below were reproduced by compile-only
checks on gfx950):

- **Two stages minimum:** `warp_pipeline_stage borders did not produce at least two stages`.
- **Prefer a dynamic `range`:** a `gl.static_range` body is unrolled — the loop loses the
  `pipelined_for` attribute, but the conversion still emits the phase offset, a barrier at each border and
  `s_setprio`, so it behaves as a straight-line ping-pong with code-size growth (compile-checked 3.8.0,
  gfx950). Do not assume the static form is unpipelined; prefer the dynamic form for code size.
- **Waits go on a border:** a `wait_group` between two stages and one as the **first** op of a stage
  compile to the same ISA (the border takes it; example, if aiter is available:
  `aiter/ops/triton/_gluon_kernels/gfx950/attention/mha.py`); after any other op inside a stage: `barrier or wait op
  cannot appear inside a warp_pipeline_stage region`. Compiling to the same ISA is not the same as being
  correct: *which* border matters. The publish-wait for a tile must sit at least one border before the first
  stage, in either group, that reads it.
- No `gl.barrier()` in the pipelined loop body (*"unexpected op inside pipelined_for body"*); the border emits
  the `s_barrier`. Prologue/epilogue barriers are fine.
- **Every body op inside a stage** (except inter-stage waits and the iterator): a stray op between stages is
  **not** rejected — it silently changes the cluster structure (an LDS load there changed the `s_setprio`
  count).
- No `scf.for` / `scf.while` inside a stage. The epilogue for the tiles in flight is a correctness obligation.
- Two borders back to back make a dummy cluster tagged `triton.warp_pipeline.empty_cluster` — an intentional
  bubble.

### The no-wait-inside-a-stage rule is what makes layering this over an authored ring hard

On 3.8.0 a ring's `wait_group`s must land on stage boundaries (between stages or as a stage's first op); a
ring that cannot be cut that way cannot use the marker (it fails conversion).
On gfx950 most marker kernels carry no `commit_group` (small-M, no ring); hand-rolled inline-asm `s_setprio`
usually sits on a ring because inline asm is not checked
(`../gluon/inline-asm-reference.md ## Class 2 — scheduling control`). **Measured negative:** an attention
campaign's first cut failed with the diagnostic above; the repaired cut compiled and produced **wrong output**
(nonfinite values, non-zero empty rows, softmax denominator still correct) from a PV/shared-layout sync the
re-cut stages no longer ordered.

## What the compiler emits at each boundary

| boundary | emitted |
| --- | --- |
| stages share an LDS allocation | `sched_barrier(0)` + local-fencing barrier + `sched_barrier(0)` |
| no LDS overlap | `sched_barrier(0)` + bare `s_barrier` + `sched_barrier(0)` |
| pre-loop / post-loop | local barrier + `cond_barrier` / complementary `cond_barrier` |

An unexpected local-fencing barrier means the analysis sees two stages on one LDS interval — the structural
remedy is separate allocations or indexing. A fence or wait inside/at the head of a compute cluster drains the
MFMA stream; waits at the end of the memory cluster avoid that. Order in the compiler: `add_warp_pipeline` before
warp-group allocation (TTGIR); `add_warp_pipeline_conversion` after async-wait counts are final and before
`scf` → CF.

## LDS hazards close a stage early

One group runs a full stage behind, and `wait_group` covers only the calling wave's copies, so a tile is
readable by all waves only after *every* group has waited for it and then crossed a common barrier. Hence the
wait that publishes a tile sits on the border before the compute stage that precedes its reader (one stage of
slack), not on the border directly before the reading stage. Safe alternatives: a wait count that completes the
tile a stage early, or copy ownership (each group reads only its own copies). Stages form a ring across
iterations (iteration `i+1`'s first stage can depend on `i`'s last) — prime it with a prologue that waits before
the loop. Gate every new cut with a GPU race test (random inputs, NaN-filled output, many runs, bitwise against
the marker-less build); compile checks cannot see this race.

## Barrier taxonomy

| primitive | memory semantics |
| --- | --- |
| `s_barrier` | none — execution rendezvous only |
| `cond_barrier` | none; needs reconvergence |
| local-scope `ttg.barrier` / `gpu.barrier memfence [workgroup]` | rendezvous + LDS fence |
| `rocdl.barrier` | like `__syncthreads()`, includes a threadfence |
| `sched_barrier` (mask 0) | none at run time — a compiler scheduling wall |

## Membar is a standing risk, not a one-time bug

Membar may put a wait before the barrier of a memory-cluster opening, and **the AMD membar filter
early-returns when both hazard ops are async** (v3.8.0 `MembarUtility.cpp`), so a barrier between two async LDS
ops is never filtered on 3.8.0.
Escapes for a redundant compute-stage barrier (gate each with the determinism race-test,
`../method/benchmark-hygiene.md`):

- **`load_shared_relaxed`** — drops a barrier only where readiness is not a `wait_group` (a `wait_group`
  always brings its own; `pipeline.md ### What the compiler inserts after wait_group: barriers and
  load_shared_relaxed`); without a real wait it is a race.
- **Separate LDS allocations per region** — membar did not disambiguate sub-buffers of one allocation in the
  cases compiled; four quadrant tiles in four allocations got no fence. The structural option.
- **`_keep_alive()`** where the barrier exists because the allocator reused bytes
  (`../gluon/smem-lds-reference.md ## _keep_alive() — the name is the effect, the mechanism is its opposite`).

## Partitioning stages

The partitioning considerations: compute separated from memory; stage durations balanced (the short stage
idles at the barrier); few values live across a boundary; address arithmetic in the memory stage; each extra
stage costs a barrier + two walls. **The shipped kernels use stage count = 2 × dependent compute phases**: GEMM
one phase → `mem` / `mfma`; attention two (QK, then PV on softmaxed scores) → compute, memory, compute, memory. Then read off depth (tiles live per
iteration) for prologue/drain/buffers. Measurement balances stages by moving work; it does not add stages.

### The four-stage shape, and a difference worth knowing about

gfx950 tutorial kernels: compute clusters QK and PV, softmax split in halves across them and balanced by
slicing the score tile (`../workloads/attention.md ### Where the softmax goes, and why it rides with the matrix
op`). gfx1250 example: softmax halves ride the memory stages (`stage0` QK p0, `stage1` softmax-1 + V read + K
copy p1, `stage2` PV p0, `stage3` softmax-0 + K read + V copy p1). The gfx1250
example sets compute at priority 0, memory at 1; the gfx950 tutorial kernels pass none, so they emit no
`s_setprio`. When a priority is passed, the placement does not transfer across arches as is: CDNA MFMA lowering
relocates `s_setprio` after the first MFMA (v3.8.0 `MFMA.cpp`), which the gfx1250 WMMA path is not known to do,
so it is re-derived per arch.

## Architecture differences worth checking before porting

gfx942 downgrade: same marker path and `warpSize * 4` grouping; sync staging (no wide async ring — the refill
is `.store(gl.load(...))` ordered by the border `s_barrier`s); 64 KiB / 32-bank LDS, so a depth sized on
gfx950 can cap occupancy; `ds_read_b64_tr` absent. gfx1250 / RDNA: out of scope here (BlockPingpong on 32-lane
waves is not covered; the 3.8.0 Gluon `rdna3` / `rdna4` modules expose no async copy).

## Diagnosing

A first check: was the marker written for an offset (a one-wave launch makes the first rows fire together)?

| symptom | likely cause |
| --- | --- |
| markers present, no `ds_read`/`ds_write` | nothing staged — a kernel-structure problem |
| runs, no overlap | `num_warps <= 4`, or residency < 2 waves/SIMD |
| overlap but fragmented MFMA issue | fence/wait inside a compute stage |
| slower than non-pipelined | memory-bound, or timed from flushed caches (the marker gain on a gfx950 GEMM was measured on warm operands); polarity measured neutral on a two-stage GEMM (`## Stage priority`) |
| works unscaled, not on fp8/fp4 | paired-SIMD LDS read bandwidth on scaled MFMA (two in-phase groups on SIMDs {0,1} or {2,3}) |
| conversion fails | `gl.barrier()` in the pipelined loop, a wait after another op in a stage, or one stage only |
| wrong results, marker-less build correct | publish-wait on the border directly before the reading stage (other group's copies not landed), or epilogue drain missing / mis-counted |

Measure per-SIMD MFMA efficiency and loop fraction (in-loop vs prologue/drain), not only wall time.
