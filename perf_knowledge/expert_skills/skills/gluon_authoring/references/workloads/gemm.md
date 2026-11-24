# Workload: GEMM (a16w16 / a8w8 / a4w4)

Layer roadmap follows the general backbone (`../../reference.md`); the tables below are per-layer
reference points, not a separate path.

## Logical tiles

Default layout chain (gfx950): `BlockedLayout (global) -> Swizzled/PaddedSharedLayout (LDS) ->
DotOperandLayout (k_width) -> AMDMFMALayout(version=4) -> BlockedLayout (store)`
(`../tile-programming/layout-recipes.md ## Standard GEMM chain (gfx950, FP16)`).
**gfx942 downgrade:** `AMDMFMALayout(version=3)` via `cdna3.mfma`; re-derive LDS padding/swizzle
(LDS size/banks, `ds_read_b64_tr`: `perf_knowledge/hardware/cdna4_mi350/memory.md`).

## Usual bound class

Large-K GEMM is compute-bound; small-M/N or small-K can be latency/wrapper-bound — classify per shape.

- Ridge: `kernel_workflow/scripts/kernel_tools/hw_budget.py` derives it from the declared dtype and SKU (`perf_knowledge/hardware/data/sku.json`; table: `../hardware/amd-cdna4-skus.md`).
- **The HBM floor can mislead while the footprint fits the memory-side cache (MALL)** and the data is
  warm there: a floor built from probed HBM bandwidth is then not a lower bound, and the kernel can
  measure faster than it (GPU-measured on gfx950 with a warm read kernel). Tiny decode footprints (tens of
  KiB) sit at the launch / dispatch floor instead, so neither "% of HBM peak" nor a compute floor describes
  them — compare against a warm-cache probe of the same footprint, or against the launch floor.
  `--footprint-mb` decides which case applies (`../hardware/roofline-models.md ### Zeroth question: does a roofline apply to this kernel at all?`).

## Small-M (decode / GEMV) regime — memory/occupancy-bound

Metric is bytes, not MFMA efficiency; check whether the decode working set sits in the memory-side
cache before quoting "% of HBM peak".

- **Grid fill counted in CTAs vs CUs, not occupancy slots** (`cus` from
  `perf_knowledge/hardware/data/sku.json`): on the gfx950 decode GEMMs measured a second co-resident
  decode CTA added little, so the cliff sat just *above* one whole wave of CTAs. Levers and thresholds:
  `gemm-small-m.md ## Split-K decision rule (small-M GEMM)`;
  `gemm-small-m.md ## Decode GEMM time = launch floor + busiest CU`.
- Fusing a per-output re-quant into the GEMM repeats it per N-tile; under a graph price it against the
  cheap-but-not-free second dispatch (`../hardware/roofline-models.md ## Kernel time decomposition`).

### The mirror case: large M, tiny N — the same regime with the traffic reversed

- Activation streams, the weight is reused: parallelism comes from M (or split-K), and the streamed vs
  reused operand — hence the natural cache-hint choice — is the reverse of decode (reasoning, not measured).
- N below the instruction's N is under-filled at the instruction — geometry, not warp layout
  (`../method/profile.md ## Rule: low MfmaUtil has two distinct tile-size causes`).

**Folding K into the idle extent and keeping the diagonal** (inferred trick, not measured here):

```text
acc  -> reshape (M, FOLD, N, FOLD)        # FOLD K-shards on each free axis
keep  where fold_i == fold_j              # shard k against shard k
sum   over both fold axes
```

`FOLD^2` products for `FOLD` useful ones (accumulator registers vs an idle matrix pipe). Mask with a
select, not multiply-by-zero (`0*inf`); read the emitted loop — the select may lower to a cross-lane exchange.

## Mid-M ridge (regime transition) — needs its own tile

Decode and prefill tiles can both regress here; a separate bucket + tile is one remedy, located by
sweeping the served M-range
(`../method/benchmark-hygiene.md ## Served-range sweep`).

## Layer roadmap (a16w16 FP16)

The body layers (async loads -> conflict-free LDS -> authored ring -> slicing) are listed for reference,
not as a dependency order: measured on gfx950 / Triton 3.8.0 (bf16 GEMM, 256x256
8-warp and 128x128 4-warp tiles), a conflict-free LDS layout and an async ring each paid on their own and
composed in either order — treat each layer as separately measurable and A/B it where you add it. The
XCD-aware PID remap + `GROUP_SIZE_M` is body-independent and can be applied at any point: it measured
neutral on compute-bound prefill tiles whatever the body, paid standalone on a memory-bound decode
split-K grid, and regressed on one decode tile without split-K (gfx950 / 3.8.0) — A/B it per grid. Third column = what proves the layer closed, not a number to match.

| Layer | Mechanism | Closed when — and what it unlocks |
| --- | --- | --- |
| anchor | explicit layouts recovered from TTGIR, `cdna3/4.mfma` | numerics match with the comparator's own layouts; attribution baseline |
| memory | `buffer_load` -> `buffer_load_to_shared` async (gfx950; gfx942: 32-bit async only on Triton 3.8.0, usually register-staged) | no register-staged `ds_write` round trip; branch count collapses to loop-carried ones |
| LDS | padding/swizzle conflict-free `ds_read` | static audit shows the conflict-free `ds_read` interval for the access width |
| pipeline | register prefetch (decode / few-MFMA tiles) or an authored LDS ring (`async_copy` + `commit_group`/`wait_group` on gfx950, sync staging on gfx942; compute-bound LDS-staged tiles start here, `../tile-programming/pipeline.md ### The overlap options and the signal for each`) -> `warp_pipeline_stage`, whose cross-wave phase offset exists only at `num_warps >= 8` (it compiles at 4 with no offset; `../tile-programming/warp-pipeline.md ## Gate 0: does the launch have two wave groups at all?`, `../tile-programming/pipeline.md`) | MFMA no longer waits on same-iteration `ds_read`; matrix-issue efficiency rises at unchanged occupancy |
| slicing | slice-N -> slice-MN | spills gone, loads spread across regions; pays for an over-unrolled pipeline |
| beyond | XCD-aware PID remap + `GROUP_SIZE_M` | DRAM read requests fall at unchanged structure; composes with everything above |

Another kernel's per-layer gains are not expectations for this one
(`../tile-programming/compiler-contract.md ## Scope: reference for authoring, not a portable recipe`).

Side levers (not on the ladder): per-compile scheduler strategy via `llvm_fn_attrs`
(`../tile-programming/instruction-scheduling.md ## llvm_fn_attrs — the portable, per-compile scheduler strategy`;
on Triton 3.8.0 / gfx950 an unknown strategy value compiles silently to the default schedule, compile-checked — diff the asm); draining below the group
(`../tile-programming/pipeline.md ### Draining below the group: bare s_waitcnt instead of wait_group`);
inline asm — keyed on the tile (`pack`/constraints), so a lifted site is not portable; a hand-written
atomic skips the `buffer_wbl2` / `buffer_inv` that `release` / `acq_rel` gpu- or system-scope language
atomics emit (relaxed and cta-scope ones emit neither; Triton 3.8.0 / gfx950, compile-checked).

## Beyond hot loop (L2 / XCD locality)

XCD / L2 geometry and `pid % 8` dispatch: `perf_knowledge/optimization/xcd_l2_locality.md`. The lever:
remap program IDs so workgroups sharing `A`/`B` rows land on one XCD, with `GROUP_SIZE_M` on top:

```text
minimize  GM + ceil(P / GM)        # P = 32 programs per XCD dimension
P=32 -> GM = 4,5,6,7,8 all tie at 12   # the model does not discriminate; sweep {4, 8}; GM=1 scores far worse in the model
```

Validate with `TCC_EA0_RDREQ_DRAM_sum` (L2->DRAM read requests) before/after.

**Gate the remap.** Its measured gain came on an L2/HBM-bound grid (a decode split-K GEMM whose
M-tiles share weight blocks); on latency-bound grids (`MemUnitStalled` ~0, low HBM share) remaps and
`GROUP_SIZE_M` changes measured neutral or slower in the GEMM and MoE cases tried, consistent with
breaking the default linear streaming order; with a single M-tile it was flat. A cut in DRAM traffic alone
is not a time gain — the memory-side cache can absorb the re-reads, and one decode case got slower with
less DRAM traffic. A/B before closing the direction. Sharers already spaced
a multiple of 8 apart share an XCD — a remap cannot improve that and can break it. **Split-K:** placing
the M-tiles that share a weight block on one XCD fetches it once (`gemm-small-m.md ## XCD placement for split-K grids`).

### Worktile scheduling for load imbalance (LPT/SPT makespan)

For load-imbalanced grids (causal, varlen, ragged split-K), longest-first (LPT) worktile order reduces
makespan; it is a `pid -> worktile` remap only, no body change.

- A global longest-first sort measured slower on a causal varlen attention grid (it broke L2 reuse);
  the form that paid kept the batch axis outermost, heads in L2-capacity-sized sections, heaviest-first
  within a section. Not a rule: on other prefill kernels a global sort was neutral or won despite a lower
  L2 hit rate — A/B the global and the sectioned order.
- Varlen: a tiny preprocessing kernel writes a sorted virtual->actual batch map.
- Attention specifics: `attention.md ## Scheduling (load-imbalanced grids)`.
- **Small grids:** measured on gfx950 / Triton 3.8.0 (compute-bound variable-K GEMM tiles), LPT was
  neutral when every CTA fit at ≤ 1 CTA/CU and paid from about 1.5 CTAs/CU up, including a single
  residency round at 2 CTAs/CU. The only LPT loss seen there, as in the attention case above, came when
  sorting by length also regrouped which operand blocks concurrent CTAs read (L2 / XCD locality), not
  from grid size. Sort by length within locality groups, and A/B it. MoE measured LPT inert: `moe.md ## Load balance is a schedule, not a tile`.

## Low-precision

a8w8 (BF8): tile 256x256x128, `mfma_scaled` "e5m2", M+N slice. a4w4 (MXFP4): tile 256x256x256,
`mfma_scaled` "e2m1", M+N slice; its LDS round-trip scale path is the earlier form, not the mature
one (`../tile-programming/low-precision.md`).

**gfx942 downgrade** (`perf_knowledge/operators/scaled_quant_gemm/tuning.md`): the VALU block scale
between matmuls is `../tile-programming/compiler-contract.md ## VALU-between-matmul: manual interleave, or author a pass`.

## Preshuffled / block-scaled signals

- **Preshuffled** (`DistributedLinearLayout`, `reshape/permute/trans` unshuffle, packed scale
  layouts): preserve the transformation sequence; in the cases surveyed, wrong answers came from
  simplifying the unshuffle, not from the MFMA call.
- Block-scale routes: `moe.md ### Axis 2 — how the quantization reaches the matrix core`.
- **Scale-feed moves** (each separately attributable):
  - group along N/M a multiple of the tile (`group % BLOCK == 0`) -> one scalar per K step instead of
    a per-lane gather; branch outside the K loop;
  - rank-1 `dot * a_s[:, None] * b_s[None, :]` -> combine scales first, one FMA into acc (check
    tolerance on boundary shapes);
  - prefetch next step's scales like an operand;
  - per-row scale vectors are replicated across the warps that split N — read the replication from
    the layout / per-workgroup load count; let one warp load and share via LDS.
- One accumulator via running rescale (`U_k = P_k + (s_{k-1}/s_k) * U_{k-1}`) frees a register tile
  but puts the rescale on the MFMA chain — needs independent work to hide it.
- **Scaled-MFMA counters:** `MfmaUtil` / busy cycles count scaled MFMA on gfx950 (rocprofv3 1.3.5, measured); the op counters
  split by format (fp8 x fp4 -> `MOPS_F8`, fp4 x fp4 -> `MOPS_F6F4`) — state which path was measured
  (`../hardware/bound-class-signals.md ## Rules (necessary-not-sufficient)`).
- Register pressure before unrolling: `acc_bytes = BM*BN*4`; if one dot tile is a large share of the
  VGPR budget, body duplication spills.

## Stop conditions (matrix route)

Signals that the Gluon matrix route is not paying: the capability matrix marks the dtype/op
`wrong-result` or blocked (a correctness stop); the Gluon path adds unhoistable hot-loop
`convert_layout`; a plain/config path wins at the same boundary; or tuned Gluon has only local winners
not worth a dispatch entry.
