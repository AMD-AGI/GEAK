# Tile Programming Mental Model

Read this before authoring any Gluon layer. Tile programming is **block-level
scheduling under hardware budgets**: choose a plan `x` = (tile shape, layout
chain, memory path, pipeline, slicing, epilogue, dispatch) that minimizes
`T_kernel(x)` subject to `Feasible(x)` (LDS / VGPR / occupancy / in-flight caps
from `../hardware/roofline-models.md`).

The optimization loop is: **budget -> theory -> orchestrate -> prove -> reclassify**,
repeated per layer.

**Arch convention for this and every `tile-programming/` page:** written for **gfx950**
(CDNA4, MI350X/MI355X) first; **gfx942** (CDNA3, MI300X/MI325X) follows as a "gfx942
downgrade" note saying what is unavailable, what changes, and by how much. API spellings are
upstream Triton **3.8.0**; 3.6/3.7 differences appear only as short downgrade notes.

## Reduction & parallelization structure (decide before tiles)

Before any tile shape, decide *which axis the grid parallelizes* and *where each
reduction lands*. For ops with more than one reduction this decision usually
dominates every later tile-level choice, and revisiting it is far more expensive
than changing a tile or a pipeline.

- **P1 -- structure first.** Enumerate the op's reductions and their axes. The
  parallel axis makes a reduction on that axis **clean** (accumulate in
  registers, write once); a reduction on a *different* axis cannot be clean and
  must pay one of three costs (P2).
- **P2 -- recompute vs atomic vs materialize vs defuse.** A reduction not aligned
  with the parallel axis costs one of: **recompute** its upstream inputs,
  **atomic-add** across partitions, **materialize** partials + a second reduce
  pass, or **defuse** — split the fused kernel by output/parallel axis into separate
  dispatches so each output's reduction becomes clean and its atomic combine
  disappears (reverse of fusion; e.g. attention-bwd fused → dkdv + dq kernels).
  Choose by comparing these costs on the target HW; do not default to atomics. The
  fused↔split call is symmetric — floor-probe both sides.
- **P3 -- atomics are bandwidth, not latency.** Cross-partition atomic write
  *volume* scales with the number of contributing partitions; scheduling /
  overlap hides *latency*, never write *volume*. An atomic-heavy structure on a
  memory-bound op is a structural cost, not a scheduling bug. Reduce the volume by
  fewer contributors **or** a narrower atomic dtype (packed bf16/fp16 atomics --
  `memory-path.md ## Output / reduction path (atomics)`).

When two reductions sit on transposed axes you cannot make both clean in one
grid -- one side pays P2. Worked example (attention backward): `dk/dv` reduce
over keys while `dq` reduces over queries (a transpose conflict), so either
parallelize by key and recompute/atomic `dq`, or split into passes. The family
treatment lives in `../workloads/attention.md`.

Record the chosen structure in the Minimal Contract before tiling.

## Tile hierarchy (global -> LDS -> register)

```text
global problem
  -> grid of output tiles
  -> workgroup owns tile(s)
  -> waves/lanes move data
  -> LDS / registers hold fragments
  -> MFMA / reduction
  -> epilogue
  -> HBM
```

Logical tiles:

- GEMM: `A_tile [BLOCK_M, BLOCK_K]`, `B_tile [BLOCK_K, BLOCK_N]`,
  `C_tile [BLOCK_M, BLOCK_N]`; K-loop accumulates partial sums.
- Attention: `Q [Br, D]`, `K, V [Bc, D]`, `S [Br, Bc]`, `O [Br, D]`.

## Dataflow primitives (Gluon naming)

| Symbol | Meaning | Hardware op |
| --- | --- | --- |
| **AC** | async global -> LDS | `gl.amd.cdna4.async_copy.buffer_load_to_shared` + `gl.amd.cdna4.async_copy.commit_group` / `wait_group` |
| **GR** | global read to registers | `gl.amd.cdna4.buffer_load` / `gl.load` |
| **LW** | registers -> LDS | `smem.store` / `ds_write` |
| **LR** | LDS -> registers | `smem.load` / `ds_read` / `ds_read_b64_tr` (transpose-on-read) |
| **DOT** | matrix compute | `gl.amd.cdna4.mfma` / `gl.amd.cdna4.mfma_scaled` |

**gfx942 downgrade.** AC: the same `cdna4.async_copy` lowers on gfx942 only at **32 bit per
thread** with a shared destination `order=[1, 0]`, and is measured slower than synchronous
staging there, so the gfx942 default is GR -> LW -> LR (sync staging) —
`../gluon/pipeline/authored-overlap.md` row A1. GR/DOT spell `gl.amd.cdna3.buffer_load` /
`gl.amd.cdna3.mfma`; there is no `ds_read_tr` and no scaled MFMA (no `mfma_scaled`, no fp4/fp6;
fp8 is FNUZ). LDS is 64 KiB / 32 banks instead of 160 KiB / 64 banks.

The standard GEMM layout chain:

```text
BlockedLayout / DistributedLinearLayout (global offsets)
  -> PaddedSharedLayout / SwizzledSharedLayout (LDS)
  -> DotOperandLayout (k_width, parent = AMDMFMALayout)
  -> AMDMFMALayout (accumulator; version=4 on gfx950, version=3 on gfx942)
  -> BlockedLayout (store)
```

A layout is a **mapping from hardware indices (lane / warp / register) to tensor
coordinates**, not a drawing of the matrix. Details in `layout-recipes.md`.

## Latency vs throughput (the key distinction)

- **Latency** is hidden by **prefetch**: issue a load early so its result is
  ready when needed.
- **Throughput** is hidden by **interleave**: keep the issue rate high enough to
  saturate the unit (e.g., one `ds_read_b128` every 16 cycles).

Confusing the two is the most common pipeline mistake. A deeper pipeline only
helps when `stall_reduction > extra_cost`.

**Occupancy gates latency-removing levers (the key corollary).** Whether a lever
that removes EXPOSED serial-chain latency pays is decided by occupancy, not by the
lever itself. Such levers — dropping a redundant load mask/branch, conditionally
skipping a per-iteration accumulator rescale, folding a scalar off the dependency
chain — only help when the chain latency is actually exposed: low occupancy / few
resident waves (the same signal that classifies the kernel dependency/latency-bound:
the busy/throughput counter ≪ 100% while the duty-cycle counter ~100%). At higher
occupancy (2+ resident waves) the latency was already hidden by another wave's work,
so the identical source change is neutral-to-negative once its added control
flow/branch cost is counted. Practical consequence: the same change can be a real win
on an explicit, low-occupancy Gluon kernel and simultaneously neutral on an autotuned,
multi-wave plain kernel of the same algorithm — do not generalize a low-occupancy
latency win to an occupancy-rich build, and check waves/CU before attributing the gain.
This is the occupancy axis of the "reduce-op-count is neutral on dependency-stall"
rule in `../method/climb.md ## 2. Reversed-intuition traps — read this once` (trap 6).

Independence rule for a k-stage pipeline: `DOT(k)` must not depend on the
same-slot `LR(k+1)` or `AC(k+2)`; `wait_group` proves the LDS buffer is retired
before it is overwritten (gfx942 downgrade, sync staging: the barrier after the `smem.store`
plays that role). Stated in full at `pipeline.md ### Independence rule (correctness of pipelining)`.

## Author + compiler co-design

- The **author** engineers independent AC / LR / DOT, the register budget
  (slicing), and the layout chain.
- The **compiler** schedules within that structure, allocates registers, and can be
  pushed on both: the accumulator's AGPR/VGPR placement is reachable per compile on
  3.8.0 (`llvm_fn_attrs`), the stock co-execution (`coexec`) scheduler strategy is automatic only
  on gfx1250 at `num_warps <= 4` and opt-in on gfx950 / gfx942 (`TRITON_HIP_USE_COEXEC_SCHEDULER=1`
  or `llvm_fn_attrs`), a declarative interleave is reachable only from a pass you
  author, and there is no post-assembly stage at all —
  `compiler-contract.md ## What upstream 3.8.0 actually gives you`. Which model owns a region's
  scheduling is the layer-1.5 choice in `scheduling-model.md`.

Neither half alone reaches peak. An optimization is "landed"
only when budget + profile + IR all agree (`compiler-contract.md`).

## Porting a technique across architectures

A published technique encodes assumptions about its origin architecture's
**execution model**; porting it to a different model can make it an
**anti-pattern**, not just suboptimal. Two models to keep distinct:

- **asynchrony-first** (dedicated async matmul + async DMA + dynamic per-warpgroup
  register reallocation + cheap named-barrier warp groups): hides latency with
  **few warps + explicit pipelines**, so spending registers on pipeline state is
  nearly free and overlap is the dominant win.
- **occupancy-first** (many resident waves/CU, **static** per-wave register
  allocation, no dynamic register handoff): hides latency with **many waves**, so
  spending registers on pipeline state **directly removes the waves that were doing
  the hiding** (the tri-lemma, `slicing.md ## Occupancy budget (P8)`).

Before porting a named technique: identify (a) which execution-model feature it
relies on and (b) whether the target has it. **If the technique trades the exact
resource the target uses for latency hiding, predict a regression and demand the
floor/occupancy check first** (`../method/profile.md ## Rule: floor probe`). For the
asynchrony->occupancy direction specifically: producer/consumer warp-specialization,
dynamic register reallocation, and register-buffered multi-stage pipelines are
suspect; the parts that *do* port are those mapping to a primitive the target has
natively (direct-global-to-shared DMA, the scale fold, operand prefetch).

**gfx1250 (CDNA5 / MI450) — the asynchrony-first AMD data-center target.** gfx950/942 are
occupancy-first; gfx1250 gains the asynchrony-first primitives (named-barrier warp
groups -> producer/consumer warp-specialization), so techniques the occupancy-first
targets reject (a warp-specialized correction warpgroup, register-buffered staging)
begin to port there. **gfx1250 is CDNA5 (MI450 data-center), not RDNA4** — RDNA4
client (gfx1201, R9700 / RX9070 XT) is a separate WMMA fork that does **not** have
TDM or named-barrier warpgroups; do not transfer gfx1250 asynchrony conclusions to
RDNA4. Map the origin feature to its target analog before porting:
NVIDIA **TMEM** (MMA accumulators in a near-tensor-core store, off the register
file, freeing registers for larger tiles + a decoupled correction warpgroup)
corresponds to AMD's **AGPR** accumulator file — so the "store the accumulator off
the register file to enable larger tiles + overlap" lesson transfers (subject to the
AGPR read-cadence rule, `compiler-contract.md` RA-hint risk). gfx1250 microarch
constants are not yet tabulated in this skill — probe before
relying on it (`../hardware/capability-matrix.md`).

## The hand-asm ceiling (set the true target; know the abstraction floor)

A tuned-plain or even a beat-plain Gluon kernel is **not** evidence you are at the
hardware ceiling. When a **production hand-asm kernel** exists for the op (e.g.
aiter / CK), benchmark **it** directly on the same boundary — it is the true ceiling,
and the compiler/Gluon optimum can sit measurably below it. Do **not** declare "at
the practical optimum" from a profile alone; only the hand-asm comparison proves
whether headroom remains (a clean profile at optimal occupancy can still be far from
what raw asm reaches).

When such a gap exists, decide whether it is **compiler-expressible** before
chasing it. A class of hand-asm wins is *fundamentally* out of reach of a
correctness-preserving compiler:

- Hand-asm can keep two warp-groups in a **free-running offset** (`s_setprio`-only,
  no per-iteration barrier) because the author **cycle-counts** that the staggered
  groups never touch the same shared buffer at once.
- A provably-correct compiler **cannot** make that timing guarantee when buffer
  indices are **runtime** values (e.g. `k % nBuffers`): it must insert a
  cross-warp/LDS **correctness fence** between writer and reader, which **re-syncs the
  groups every iteration** and destroys the free-run overlap. The barrier is
  load-bearing (dropping it races), not removable scheduling overhead.

So a tight cooperative-load ping-pong is a hand-asm capability, not a knob. Closing
that last gap needs raw asm or a major compiler rewrite (emit timed staggering +
prove the buffer disjointness) — a project beyond kernel tuning. Record it as a
**scoped abstraction ceiling** (the Gluon-expressible optimum + the hand-asm target +
why the gap is not compiler-expressible), not as an open layer
(`compiler-contract.md ## Scenario B`).

## Why Gluon is full-explicit

Once `@gluon.jit` is open, layout selection, `convert_layout` placement, shared
layout, and `num_stages` auto-pipelining leave plain Triton's compiler
inference; the author owns them. On upstream 3.8.0 **`num_stages` is a dead knob on the Gluon
path** — no pass in `gluon_to_ttgir` consumes it — so it survives only as a budget parameter and a
field of the champion record, never as a value to carry over from plain and tune. That is why the
first Stage-Gluon step is a faithful transcription (an equivalence anchor), not optimization — see
`../method/transcribe.md`.

**The overlap is therefore authored, in the one order `pipeline.md ## Where the overlap comes from,
and it is not the same question per tier` defines:** (1) register-level prefetch; (2) an authored
LDS ring (gfx950: `async_copy` + `commit_group` / `wait_group`; gfx942 downgrade: sync staging, or
32-bit async with a destination `order=[1, 0]`); (3) `warp_pipeline_stage` plus the scheduling-model
choice (layer 1.5, gated on `num_warps >= 8`). Instruction scheduling and an authored pass sit on
top of whichever structure exists, and neither creates one; do not record a scheduling ceiling until
the overlap itself exists.

**Re-injecting plain's pipeliner is the lowest rung — a diagnostic and a last resort, not the
route.** The MFMA/VALU overlap the `num_stages` auto-pipeliner produces comes from
`add_schedule_loops` + `add_pipeline` in `make_ttgir`, which `gluon_to_ttgir` skips; splicing those
two passes over the module `gluon_to_ttgir` returns reproduces plain's multi-buffer loop, with **no
`libtriton.so` rebuild and no edit to an installed file** (upstream's `add_stages_inspection_hook`,
or the pack's `gluon_swp.py`, wrap the function rather than the pass manager it built). It has
exactly two uses: **below the parity gate**, to measure how much of the gap is `lost_pipeline` debt;
and **as a last resort** when the hand-written ring cannot reach parity. Its numbers are labelled
`injected` and are never a climb win, and it is **never applied to an incumbent (already-Gluon)
kernel**. Its ceiling is plain parity — it repays a debt and does not open a climb
(`pipeline.md ## Reproduce plain's software pipeline on the Gluon path (the parity-recovery route)`;
full recipe in `../method/recover.md`).

Two measured findings bound it, and they are separate. It reached **near-parity with plain** on a
gfx942 attention kernel (gfx942 downgrade measurement); the earlier "forcing it **regresses**" was a
retracted misfire — a hand-async loop with **no in-body `tt.load`** for the pipeliner to anchor on,
plus the `disableSched` occupancy cliff, not the passes themselves. A later case still **stands**: on
a sparse-paged attention kernel the injection was demonstrably firing and was a **large
regression**, because the pipeliner emitted an operand staging in a shared layout Gluon has no
constructor for. The live finding is why the acceptance bar for an injected variant is the layout
read, not the timing.
