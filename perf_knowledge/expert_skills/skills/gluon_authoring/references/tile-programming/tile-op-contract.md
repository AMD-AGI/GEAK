# Tile-op Contract Card (four-axis readout)

The **unit of work** that flows through the Stage-Gluon phases. It does not add a new
subsystem; it gives the four axes that already exist in this skill — scattered across
`mental-model.md` (Scope, Dispatch, the layout chain), `layout-recipes.md` (Layout),
and `pipeline.md` / `compiler-contract.md` (Handoff) — **one per-op spine**, so the
agent can reason and act layer by layer instead of cross-referencing four files for
one tile-op.

Lifecycle:

- **Build** the card at transcription, one per transcribed tile-op, in `anchor` state
  (`../method/transcribe.md`).
- **Mutate ONE cell per backbone layer** in the layer loop; the layer *is* the axis it
  makes explicit (`../method/climb.md`, the map below).
- **Close** a cell with three-evidence at evaluate (`../method/close.md`).

<!-- BEGIN portable core — byte-identical across the 4 tile skills (gluon / cutedsl / flydsl / tilelang). Keep in sync; everything below the END marker is this skill's per-DSL fill. -->

## Card schema (canonical)

```text
tile-op: <name> @ <program point>
- Scope:    issuing threads / lanes / wave / warp / workgroup / elect
- Layout:   <src layout> -> <dst layout>            (anchor | optimized)
- Dispatch: <dataflow primitive> + <concrete builtin / memory path>
- Handoff:  <signal / drain that releases the next consumer or frees the buffer>
            | "none = compiler auto"                 (compiler still owns the overlap)
state:      anchor | optimizing(layer N) | closed
evidence:   budget / profile / IR                    (three-evidence, per cell change)
budget_ref: -> the budget / per-layer-round records  (resource cost lives there, not on the card)
```

The card is a **pure four-axis readout**. Resource cost (register total, shared-memory
bytes, occupancy/waves) is NOT a card field — it stays in the budget record and the card
only points at it via `budget_ref`.

## The four axes (the one question each asks)

| Axis | The one question |
| --- | --- |
| **Scope** | *Which* threads issue / participate in this op? |
| **Layout** | *Where* does the tile live and in what layout, src -> dst? |
| **Dispatch** | *Which* hardware primitive / memory path issues it? |
| **Handoff** | *Who* may run next, on *what* signal, and *when* is the buffer free? |

A blank or `none = compiler auto` Handoff cell is **information, not an omission**: it
records that the compiler currently owns this op's overlap (the pipeline layer is where
that cell gets filled, on DSLs that have a compiler-owned pipeline).

## Axis <-> backbone-layer map (the unifying claim)

Each backbone layer is *defined* as making one card axis explicit. The card is the
persistent state the layer loop mutates, one cell per round.

| Backbone layer | Card cell it fills / optimizes |
| --- | --- |
| transcription / authoring anchor | **builds the whole card** at faithful values |
| memory path | **Dispatch** (the global -> shared staging path) |
| LDS / shared layout | **Layout** (shared): padding / swizzle for conflict-free reads |
| pipeline | **Handoff** (fill any `none = auto` gap: explicit produce/consume + interleave) |
| slicing / register / occupancy | **Scope** (lane/warp partition + register budget; cost via `budget_ref`) |
| beyond hot loop | **Scope** at grid granularity (tile scheduling / locality) |
| low-precision side | **Dispatch** (scaled matmul) + a separate scale side-path card |

## Closing a cell (three-evidence)

A cell change is **landed** only on the three-evidence bar the layer loop already uses —
budget consistent + profile delta positive + IR/asm signal confirmed. The card localizes
*which* cell the evidence is about; it adds no new gate. The enabling-step exception
applies unchanged: a cell may be provisionally changed on a flat/negative single-step
delta if it is a declared prerequisite for a coupled combination.

<!-- END portable core -->

## Where each axis is defined (this skill)

| Axis | Source of truth in this Triton-family pack |
| --- | --- |
| **Scope** | `mental-model.md ## Tile hierarchy`; occupancy in `slicing.md ## Occupancy budget (P8)` |
| **Layout** | the layout chain in `mental-model.md ## Dataflow primitives`; `layout-recipes.md`; the recovery map for the anchor (`../method/transcribe.md`) |
| **Dispatch** | the `AC / GR / LW / LR / DOT` table in `mental-model.md ## Dataflow primitives` |
| **Handoff** | the independence rule in `mental-model.md ## Latency vs throughput`; the overlap order in `pipeline.md ## Where the overlap comes from, and it is not the same question per tier`; the scheduling-model choice in `scheduling-model.md`; `compiler-contract.md` |

## gfx950 / Gluon fill table (this skill's arch fill)

Concrete values each axis cell takes on CDNA4 / gfx950 with Triton -> Gluon:

| Axis | gfx950 / Gluon fill |
| --- | --- |
| **Scope** | wave64; one **elected lane** issues an async copy / MFMA; workgroup owns the output tile(s); occupancy = `f(LDS/wg, VGPR+AGPR/wg)` |
| **Layout** | `BlockedLayout` / `DistributedLinearLayout` (global) -> `PaddedSharedLayout` / `SwizzledSharedLayout` (LDS) -> `DotOperandLayout` (k_width, parent `AMDMFMALayout`) -> `AMDMFMALayout` (v4 on gfx950) -> `BlockedLayout` (store) |
| **Dispatch** | `AC` = `gl.amd.cdna4.async_copy.buffer_load_to_shared` + `commit_group`/`wait_group`; `GR` = `gl.amd.cdna4.buffer_load` / `gl.load`; `LW` = `ds_write`; `LR` = `ds_read` / `ds_read_b64_tr`; `DOT` = `gl.amd.cdna4.mfma` / `mfma_scaled` |
| **Handoff** | **buffer_sync (authored at pipeline layer):** `commit_group` + `wait_group` + independence rule (`DOT(k)` must not depend on same-slot `LR(k+1)` / `AC(k+2)`). **interleave (compiler auto):** `none = LLVM` — the backend's own scheduler owns hot-loop MFMA/mem interleave (on 3.8.0 the stock co-execution strategy is automatic only on gfx1250 at `num_warps <= 4`; on gfx950 / gfx942 it is an opt-in A/B via `TRITON_HIP_USE_COEXEC_SCHEDULER=1` or `llvm_fn_attrs`), steerable per compile via `llvm_fn_attrs` / `amdgpu-sched-strategy` (`llvm-fn-attrs.md`, `compiler-contract.md`); the inter-wave alternative is `warp_pipeline_stage` (`num_warps >= 8`, `warp-pipeline.md`), chosen per region at layer 1.5 (`scheduling-model.md`). `sched_barrier` / `sched_group_barrier` have no user surface on 3.8.0 (`instruction-scheduling.md ## The declarative hints that are not there`). Not TileLang-style full auto. |

**gfx942 downgrade of the fill:** Dispatch `AC` is normally absent — `cdna4.async_copy` lowers only at 32 bit/thread into an `order=[1, 0]` destination and measured slower than sync staging, so the default is `GR` (`gl.amd.cdna3.buffer_load`) -> `LW` (`smem.store`) -> `LR`; no `ds_read_tr`; `DOT` = `gl.amd.cdna3.mfma` (`AMDMFMALayout` v3, no `mfma_scaled`). Handoff **buffer_sync** is then the barrier between the `smem.store` and the consumer read rather than `wait_group`. Scope: LDS/wg is budgeted against 64 KiB per CU, not 160 KiB.

**Build step = transcription.** The card is built by recovering the plain TTGIR layouts
(`../method/transcribe.md`); the Layout cell starts `recovered`. At the
faithful anchor, **buffer_sync** Handoff is `pending` (plain's `num_stages`
auto-pipeliner does not carry into Gluon, and `num_stages` is a dead knob there on 3.8.0 —
`mental-model.md ## Why Gluon is full-explicit`); **interleave** may read
`none = LLVM` where LLVM sched still applies. The pipeline layer **must** fill
buffer_sync by hand, in the order `pipeline.md` defines: register prefetch, then the authored
`commit_group`/`wait_group` ring (`pipeline.md ### Vetted double-buffer skeleton (copy, then
specialize)`; gfx942: sync staging), then `warp_pipeline_stage`. Re-injecting plain's pipeliner
fills the cell only as a below-parity diagnostic or last resort, labelled `injected`, and never on
an incumbent Gluon kernel. Interleave stays LLVM-owned unless the region's scheduling model says
otherwise (`scheduling-model.md`) or a sanctioned LLVM pass is authored
(`compiler-contract.md ## Scenario B`).

## How to use it

- At transcription: emit one card per transcribed tile-op (load A/B, the MFMA, LDS
  read/write, epilogue store) at faithful `anchor` values; Handoff **buffer_sync** =
  `pending`, **interleave** = `none = LLVM` where applicable — not TileLang-style
  `none = auto` for the whole pipeline.
- In the layer loop: name the card + the **single cell** the picked layer changes
  (`card_cell_changed` in `../method/records.md ## 5. Per-Layer Round Ledger (core)`), apply it, then close it on
  three-evidence.
- At evaluate: a closed layer = its card cell(s) closed with the IR signal on file.
