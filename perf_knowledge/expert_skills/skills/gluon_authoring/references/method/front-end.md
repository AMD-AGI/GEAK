# Front end: what the plain tier already covered

**What this page decides.** Nothing in a deep-dig run — and that is the point of reading it. It is the
plain-Triton playbook: the 0–15 direction ladder the broad-search front end works through, and the
plain (`tl.*`) form of each lever that both tiers can write. It tells you **what was already disposed**
before the champion bundle reached you, and what each plain lever *cannot pin* — which is the honest
content of any "plain is the ceiling here" or "this needs Gluon" claim.

**When you are here.**
- **From the Gluon deep dig (this skill):** to see what the front end settled, so you do not re-run it,
  and to read a lever's "what plain cannot pin" before you claim the explicit tier is needed.
- **As the stay-plain destination:** when the escalation seam (`entry.md`, §3) says stay plain — gap
  small, or owned by wrapper / dispatch / launch / algorithm — apply the applicable directions here.
  **Stay-plain is NOT wrapper-only**; wrapper-only is just the truly-near-peak sub-case.
- **In GEAK** the front end's role is played by GEAK's own plain-Triton rounds (tech_lead's specialist
  directions before the `deep_explore` port); upstream it is the `tile-programming-triton` pack. Either
  way the configuration sweep and the structural fan-out are **not this skill's to run**.

Paths: `$SKILL` = `perf_knowledge/expert_skills/skills/gluon_authoring`; `$KT` =
`kernel_workflow/scripts/kernel_tools`. Commands that name a triton-pack tool (`plain_autotune.py`,
`knob_expose.py`, `sweep_audit.py`, `champion_emit.py`, `phases/tune.md`, `phases/structure-census.md`,
`search-routing.md`) refer to `tile-programming-triton`; they do not ship here.

---

## 1. Two roles, and only the second is yours from a deep-dig pack

1. **The broad-search front end** — the direction search that decides whether there is anything to hand
   off at all. It sets `champion_ms`, the target line every later claim is quoted against, and its
   structural BRANCH fan-out over diverse plain hypotheses is the legitimate breadth search in this
   family (`orchestration.md`; the analysis method is `profile.md`, the sweep is the triton pack's
   `phases/tune.md`).
2. **Stay-plain destination** — apply the applicable directions when the seam says stay plain.

> **If you are reading this from a deep-dig pack, role 1 is not yours.** The configuration sweep and the
> structural fan-out are the front end's, and they arrive **already disposed** in the champion bundle:
> its configuration is pinned, `champion_ms` is the target line, and the round shape is one coupled layer
> at a time with no SWEEP and no BRANCH. That is why `phases/tune.md` and `phases/structure-census.md`
> are not shipped here — not an omission, and not something to reconstruct locally. Every direction
> below was settled by the front end before the champion was emitted: use this page to see what has been
> tried, and `entry.md` (§3.10, "What does NOT map plain -> Gluon") for which of these levers has no
> counterpart in the explicit tier. **Do not re-run them.** A gap that looks like "the configuration was
> never explored" is a message to send back to the front end (a `resweep_request` in the deep_engineer's result, which tech_lead turns into GEAK's next plain round), not
> a sweep to run. Role 2 below **is** legitimately yours to read; the per-direction material is
> reference, not a new work menu.

If a direction instead needs explicit layout / memory-path / matrix-pipeline control, it is a handoff
trigger — route back to the escalation seam (`entry.md` §3), not a co-equal "Gluon variant" here.

---

## 2. Direction priority (0-15; lower = higher priority)

| Priority | Direction |
| --- | --- |
| P0 | algorithm / decomposition (tiling, split-K, reduction tree, program granularity) |
| **P0.5** | **host / launcher ownership** — who owns the grid, the workspace, the config, the dispatch. **Not the same as P15's wrapper micro-work**: this is the layer that decides what the kernel is even *allowed* to be, and it gates several P0 moves |
| P1 | worktile / grid **scheduling** (makespan) — **when the grid is load-imbalanced** (causal / varlen / triangular / ragged split-K / uneven per-program work) |
| P2 | fusion / epilogue |
| P5 | memory / compute reorder (coalescing, LDS, register, accumulator, store) |
| P6 | shape-adaptive / visible dispatch |
| P8 | autotune / config / launch parameters |
| P15 | wrapper / launcher / dispatch-only |

Record the selected value as `direction_priority`, then pick the first direction that explains the
measured bottleneck; do not default to kernel-body work. P0, P1, P5, and P8 remain distinct priorities,
not lifecycle phases. Prove the direction before sweeping parameters. A kept win in one direction does
not close the others — return to the bottleneck census. The search method is selected separately: SWEEP
ranks configuration, BRANCH compares independent structures, and CLIMB advances one locked body
(`search-routing.md`, triton pack). How the ladder maps onto the explicit-tile layer backbone:
`entry.md` §3.10.

### Why host/launcher is P0.5, and what it is not

Host/launcher ownership is P0.5 because it determines which P0 decompositions are callable: grid,
workspace, config resolution, and dispatch selection. It is a gate on an algorithmic option, not an
algorithm itself. Establish the production boundary and call contract before classifying an option as
unavailable; a launcher may supply setup outside the timed region.

**P0.5 is not P15.** P15 is wrapper micro-work — shaving argument handling, avoiding a copy — and it stays
at the bottom. P0.5 is ownership: grid, workspace, config resolution, dispatch. P0's `launcher-object
escape` (below) is the mechanism, and its contract-guard template is not optional there.

## 3. Hot-path selection + stale fallback guards (do first)

Before editing: identify the measured hot subpath (not just the public wrapper); group equivalent
variants; verify disabled fast paths before assuming the fallback is the only hot path. Treat
guarded-off paths as hypotheses when they could change the optimization layer:

```text
guard source / current target + Triton/ROCm
disabled path / compile / correctness / performance vs safe path
fallback policy if unsupported
```

If guard removal switches execution path (grouped attention, native FP8, one-shot reduce elimination),
treat the switch as a standalone direction and quick-gate it immediately. Keep conclusions scoped to
target/dtype/Triton version.

## 4. P0 — algorithm / decomposition

Tiling, program granularity, split-K / split-stage, reduction-tree, loop order, no-mask fast paths, phase
decomposition. For fused/multi-phase kernels write a phase table first:

```text
phase | program count | work/program | memory traffic | can split?
```

If one phase has many tiny programs, try coarser program ownership or phase decomposition before pointer
cleanup. Grid utilization / saturation:

```text
grid_tiles = ceil(M / BLOCK_M) * ceil(N / BLOCK_N)
saturation_ratio = grid_tiles / target_CU_count   # gfx950 (MI350X/MI355X): 256 CU; gfx942 downgrade (MI300X): 304 CU
```

`< 1.0` critical under-utilization (smaller tiles / more programs); `1.0-2.0` likely under-utilized;
`2.0-4.0` standard; `> 4.0` well saturated. A 1024x1024 GEMM with 256x256 tiles is only 16 tiles on 256
CUs; 64x64 tiles give 256.

### Three named patterns

These are P0 structural patterns to evaluate through `CENSUS-0` (`phases/structure-census.md`, triton
pack — the census is the front end's, so in a deep-dig pack these patterns arrive already disposed rather
than open), not default recommendations. Each requires the current boundary, oracle, and comparator.

**`redundant-recompute split` — buy parallelism with duplicated work, and stay bit-identical.** When
device fill is low, split an *output* dimension across cooperating programs and let every sibling
**recompute the shared prologue identically** rather than reducing across programs. The candidate must
prove its result remains within the declared oracle. The gating question is not only "does this add
parallelism" but **"does it introduce a cross-program reduction?"**; record that semantic and numerical
obligation before benchmarking.

**`launcher-object escape` — export an object, not a bare `@triton.jit`.** A source file may export an
object implementing `__getitem__(grid) -> callable(**kwargs)`. It can own grid selection, workspace
setup, config resolution, and per-shape dispatch. It is a high-risk call-contract change, so the
following guard template is mandatory:

```text
1. forward **kwargs COMPLETELY -- start from the full signature, not from the args you use
2. _ok(): one explicit check per assumption you hardcoded (tile, split factor, group size,
          stride, dtype, flag). Each check names the assumption it guards.
3. not _ok() -> fall back to the UNMODIFIED path. A fallback is a correct answer; a
                silently-wrong fast path is not.
4. cache key covers EVERY independent variable of the thing you memoised. A plan keyed on
   M alone that carries a grid derived from N and an EVEN_K derived from K produced
   4.24M wrong elements of 4.33M, and a memory access fault, on the second shape.
```

**`in-kernel granularity fusion` — raise the effective tile without touching the grid.** When the grid is
owned by a caller you may not edit, a program can still cover several of the caller's blocks: do the work
of `G` consecutive blocks in one program at a `[G*BM, BN]` tile, and retire the surplus CTAs in the
prologue. Measured 1.69× → 2.39× on one MoE kernel. This matters because it dissolves a class of obstacle
rather than solving one: the kernel that deferred its top lead on the grounds that *"decoupling
`host.py`'s two uses of `BLOCK_SIZE_M` is the whole problem"* was reachable from this side without
touching `host.py` at all. **When a structural lead is blocked by who owns the grid, ask whether the same
effect can be had inside the kernel before you record the blocker.**

### Split-K / split / partition

Use when a monolithic kernel has poor parallelism, excessive reduction cost, or a large splittable
dimension with a controlled combine. Risks: combine/atomic cost, numerical-order/precision changes,
temporary buffers, per-shape-different factors.

```text
split_or_partition_knob:
program_count_before_after / inner_loop_work_before_after
main_kernel_latency / reduce_or_combine_latency / temporary_buffer_cost
correctness oracle + tolerance / precision-sensitive buckets
winner_buckets / decision
```

Decode/attention diagnosis: `total_workgroups = batch * head_groups * splits`, `work_per_group = seq_len /
splits`, `stage2_fraction = reduce/main`. Increase splits when workgroups underuse CUs and reduce cost is
small; decrease or test one-shot when per-group work is tiny or reduce cost is material (a useful first
signal for small decode grids: `batch * num_kv_heads <= 64`, verify locally). Compare the end-to-end
boundary, not only the main kernel.

## 5. P1 — worktile / grid scheduling (makespan)

**Trigger: the grid is load-imbalanced** — per-program work is uneven, so a naive `pid -> worktile` order
leaves the heaviest tiles on the tail while other CUs idle. Canonical cases: **causal** masking (work
grows with the query block), **varlen**, triangular/banded, **ragged split-K** tails. If every program does
equal work, skip this lever.

This is a **host-side launch-order remap** — choose the `pid -> worktile` linearization to minimize
**makespan** (longest-processing-time-first, heaviest tiles first). It is:

- **tier-agnostic** — pure grid/PID math, applies whether the escalation gate stays plain OR escalates to
  Gluon; it is **not** a Gluon-only / Layer-6-only lever. On a load-imbalanced grid, check it **right
  after the anchor**, before deeper kernel-body work.
- **cheap and body-preserving** — zero kernel-body / correctness change; re-sweep the shape dispatch after.
- **conditioned on grid fill** — at very small grids (≲2 waves/CU) a longest-first reversal can *worsen*
  balance; gate by worktile/wave count and A/B it.

Full makespan model (LPT/SPT, L2-section balancing, varlen preprocessing-kernel sort) is the single
authoritative section: `../workloads/gemm.md ### Worktile scheduling for load imbalance (LPT/SPT makespan)`. Attention
causal/varlen specifics: `../workloads/attention.md ## Scheduling (load-imbalanced grids)`.

## 6. P2 — fusion / epilogue

Fuse only when it removes traffic, temporaries, launches, or dtype conversions (GEMM/reduction epilogue,
norm+quant, bias/activation, small producer-consumer, mask cleanup). Risks: register/LDS pressure,
precision/order changes, crossing a wrapper/dispatch boundary. Producer-rewrite-to-remove-a-branch can
increase memory traffic enough to erase the win — measure extra producer writes vs branch savings vs
temp-buffer bandwidth vs full-boundary latency. When fusion is a candidate for escalation rather than a
plain lever, the rebuttable-stay-plain rule applies (`entry.md` §3.7).

## 7. P5 — memory / compute reorder

Use when profiling shows bandwidth pressure, uncoalesced access, repeated loads, scattered stores, masks,
spills, or store-path cost.

First moves: reorder pointers, vectorize/widen loads, simplify masks, split no-mask paths, hoist
invariants, reduce register pressure, use LDS only with reuse evidence. Change one invariant/mask/memory
path at a time; pre-issue loads only when addresses/masks are ready and register pressure is OK; preserve
dtype narrowing points unless a reference check proves the move safe.

Bandwidth-ceiling diagnostic:

```text
bytes_moved / access_pattern (contiguous|strided|scattered|page_indirect)
achievable_bandwidth_estimate / theoretical_min_us / measured_us / measured_over_theoretical
```

If measured/theoretical is below ~`2x`, body-level compute/layout tuning is unlikely to help unless it
reduces bytes moved, changes the access pattern, or removes hidden spill traffic. Register spill is hidden
memory traffic — a `waves_per_eu` change can reduce total HBM traffic with no visible load/store change.

Mask/pointer cleanup priority: (1) full-tile/no-mask paths; (2) invariant-specialized kernels; (3) coarse
coalescing; (4) cosmetic rewrites. If the compiler likely already simplifies a small expression, require
benchmark evidence before more rounds (`../tile-programming/compiler-contract.md ## Body micro-edits the compiler may already handle`).

> P5 "memory/compute reorder" is the **main escalation trigger** — when the reorder needs explicit LDS
> layout / async pipeline / MFMA layout, route to the escalation seam (`entry.md` §3).

## 8. P6 — shape-adaptive / visible dispatch

Use visible shape/feature dispatch when winners differ by stable bucket. Options: `@triton.autotune`, 2-3
explicit variants, no-mask/masked variants, invariant-specialized kernels, host-side dispatch.
Escalation: generic cleanup -> structural fast path -> invariant-specialized variant -> visible dispatch.
Keep correctness + performance on the same ordered stream; per-shape no-regression and the
bucket-dispatch recipe live in `close.md` (multi-shape no-regression + dispatch).

### Before building a per-bucket track: four admissions, all on fields you already have

**The default is not to.** A bucketed track costs a body, a comparator and a climb budget per bucket, and
the questions below are ordered so the cheap disqualifiers come first. Stop at the first "no" and record
*which* one stopped you — "this kernel is one bucket" is a result, and an unrecorded one gets
re-litigated. None of the four needs a measurement that has not already been taken.

1. **Is there a consumer, and is it inside the measured boundary?** What a bucketed track produces is *a
   set of implementations plus a selection rule*. With nowhere to put the rule, the set is dead code. The
   two questions that answer this — who owns the launcher, and what configuration surface is exposed —
   are already asked by the structure census. **Note the symmetry with unreachability, because it is easy
   to miss:** "this candidate is unreachable" is policed as an assertion that must be priced, and "this
   kernel has no dispatcher" is the same claim about the same thing. Price it the same way, or a bucketed
   track gets closed off permanently by an unexamined assumption. The cheap falsifier is the same one:
   export a same-named callable from the source and check whether the harness dispatches through it.
2. **Can the served distribution be ranked?** This needs a declared weight kind and a weight on every
   served row. `$SKILL/scripts/served_envelope.py` already refuses to produce numbers when either is
   missing, on the stated grounds that assuming a uniform served mix is a claim about the deployment
   nobody measured. **Keep that refusal rather than routing around it.** Without weights you do not have
   the question "which bucket is worth specializing", you have "which bucket is slow", which is a
   different and less useful one.
3. **Has a split actually been observed?** Read `per_case.split` from the published pin (`phases/tune.md
   ## After the sweep`, triton pack). `false` stops here and is a positive finding; `unlabelled` means go
   back and label the cases, not that there is no split.
4. **Is the split large enough to be worth a structural budget?** This converts question 3's boolean into
   a price. **The ceiling on what specializing a bucket can return is its `baseline_share`** — zero out
   that bucket's time entirely and that is all there is. So require the `baseline_share` of the split
   cases to exceed the share of the round budget you would spend on the split. Two rules go with it:
   - **Never compare `speedup` or `vs_default` across buckets.** Each bucket's default is a different
     kernel at a different shape, so ranking buckets by `vs_default` ranks them by whose baseline is
     weakest — which sends the structural budget to the worst-tuned bucket rather than the most-served
     one. Across buckets, compare `baseline_share`.
   - **A single served-weighted figure must not stand alone in a per-bucket report.** A low-weight bucket
     that was *never touched* barely moves it, because its current time defaults to its baseline — so the
     number stays respectable and the structural hole appears in none of its digits. Quote it only
     alongside the per-row speedups and an explicit list of which buckets have an owner and which were
     declined.

There is a fifth admission — whether the difference between buckets is *structural* or merely a
configuration difference — and it is the one that costs something: re-run dispatch with per-bucket launch
configs and read `per_case` again. If the split disappears, it was a configuration split and the buckets
can share a body. That one is a measurement; the four above are not.

Two things not to carry over from serving-stack practice. **Do not size the bucket count from the number
of shapes** — size it from the round budget, because every published body owes a climb floor or a priced
declination, and N shallow buckets fail that floor together. And **do not reach for a signed per-shape
registry with per-file hashes**: that is a shipping mechanism for a deployment with a runtime dispatcher
and a release process, and it answers a question about rollback, not about optimization.

### Dispatch verification (do not infer the path — verify it)

For every gated shape define `expected_path / allowed_fallbacks / forbidden_paths /
expected_config_source / expected_split_or_partition / verification_method / failure_behavior`. Methods:
explicit counter/hook, path tags in benchmark JSON, assert-on-fallback for must-not-fallback shapes,
per-shape path maps. Rules: if a fast path must execute, **assert** it (don't only log); record fallback
when it occurs; fail the benchmark early on a forbidden fallback for a contract shape. (The same rule
protects a Gluon number: a dispatcher that silently falls back to the plain kernel yields a "Gluon"
measurement that is the plain kernel's, and correctness checks endorse it — check the launched kernel
name, or make the fallback fail loudly.)

Prefer host-available dispatch variables (shapes, Python args, dtype/causal/window flags, block size).
Avoid `tensor.item()` / `max().item()` GPU->CPU sync in sub-ms paths (ROCm ~15-30us, enough to erase a
bucket win). Feature flags are valid buckets only with stable performance meaning. Threshold rules: test
the exact boundary and both sides; label `threshold-incomplete` when adjacent cases are missing.

## 9. P8 — autotune / config / launch parameters

Autotune refines a credible direction; it does not replace hot-path reasoning.

**Executable runner: `scripts/plain_autotune.py` — triton pack.** The sweep is a SWEEP activity and the
runner ships only in `tile-programming-triton`; from the gluon pack this section describes what the
champion you inherited was already tuned with, not a command you can run here (in GEAK the
equivalent is GEAK's plain rounds). The runner (1) **derives** the space rather than accepting a typed one
(`tuning_laws.py`: each axis an interval ∩ a legal set from this SKU's LDS/VGPR/CU facts and the served
shape, intersected with the knobs the kernel is measured to read) and excludes infeasible points on paper
under a **fitted** LDS model — an uncalibrated model excludes nothing rather than excluding wrongly; (2)
**folds** the ①② autotune-absorbable divergent cards (`decision.autotune_absorbed_candidates` —
dispatch/wave-quant knobs like `block_m_shrink` / `group_size_m` / `grid_remap_split_k`, plus `split_k`)
into extra grid dimensions rather than T0 fan-out arms; a ② card (`dependent_data_rebuild:true`, e.g.
split-K reduce buffers) rebuilds its dependent data per config (the cross-boundary-coupling rule below —
fixed dependent data makes the sweep INVALID); (3) fixes the **meter** once for every point (a wall-time
budget rather than an iteration count, and a declared hot/cold protocol — for a memory-bound kernel the
two protocols can rank two configs in the opposite order) and gates the **oracle** per point before its
time may rank; (4) **searches** the space rather than enumerating it, **one subprocess per config** (a bad
config crashes the LLVM backend); (5) emits `plain_best_config.json` with a **certificate** — order-2
local optimality at the measured noise band, pairwise coverage, interiority, and what was never covered —
plus `trust_level`, of which only `pinned` is a comparator. This is the pinned tuned-plain winner the Gluon
anchor must be dumped from (the champion bundle's `local_optimum` / `trust_level`, read by
`champion_gate.py`'s `[SAMPLING]` / `[GATED]` — `entry.md`). It does NOT sweep ③ (layer-0 reduction/defuse
— those are the plain-stage T0 fan-out) nor ④ (`requires_anchor` layer-4 pipeline/schedule — Gluon-loop-only).

**A kernel with no knobs is a P8 case, not an exempt one.** Expect a large share of a production op tree to
hardcode every tile and launch number — there is then nothing for the sweep to rank, and its envelope is
indistinguishable from a kernel already at its optimum (`sweep_audit.py` counts the rate for your tree;
`phases/tune.md` (triton pack) is the method). Exposing an axis the kernel pins is therefore part of P8, on
one condition: the new knob's default is the value in use today and the artifact compiled at that default
is **byte-identical** to the pre-edit one. Under that condition the promotion is a config change wearing a
diff — it cannot be booked as a source-level win, and the pre-edit baseline stays a valid comparator.
`scripts/knob_expose.py` (triton pack) names the axes a kernel of this type is known to be sensitive to,
quotes the line pinning each, and runs the identity gate (knob promotion is a front-end act, so a deep-dig
pack neither exposes new axes nor sweeps them).

Classify each knob before timing:

- **Algorithmic constexpr** (`BLOCK_M/N/K`): changes decomposition / loop trips / accumulation / store
  ownership / masks — check correctness before timing.
- **Launch-only** (`num_warps`, `num_stages`, `num_ctas`, `waves_per_eu`): changes scheduling/codegen, not
  ownership — unless the body hardcodes matching layout.
- **Hybrid** (preload flags, instr-shape selectors, split factors): source-specific legality.
- **Coupled** (`(num_warps, waves_per_eu)`, `(BLOCK_K, num_stages)`, tile triples): move together. If a
  compile option is also referenced by a Gluon `warps_per_cta` or a wrapper grid formula, it is not
  independently tunable.

**Cross-boundary coupling**: if a knob changes the shape/meaning of data produced outside the kernel (quant
block shape, scale layout, reduce buffers, dispatch), the harness must rebuild that dependent data per
candidate; fixed dependent data makes the sweep **invalid, not incorrect**.

Pre-sweep correctness gate (2-3 representative changes on a small + a path-sensitive shape) before timing.
Sweep order: cache/load policy -> `BLOCK_SIZE_{M,N,K}` -> `GROUP_SIZE_M`/swizzle -> `waves_per_eu` (incl.
`0`) -> `num_warps`/`num_stages` -> split factors -> combined bucketed configs. Change one knob family per
round unless source couples them; keep a same-knob baseline when legal; record the effective config, not
just the requested knob.

**gfx950 notes (plain tier).** Treat `num_stages` as a high-value pipeline-depth knob (test `num_stages=3`
when `BLOCK_K >= 64` and K-loop trips are high; `=2` safer for small/register-constrained tiles; `=4`
needs explicit evidence). Treat `BLOCK_SIZE_K` as a coupled perf/resource knob. Compiler hints are not
always improvements — test hint removal (`waves_per_eu`, `matrix_instr_nonkdim`, `cache_modifier`) as its
own hypothesis (updated compilers may choose better). `kpack` is deprecated on gfx950 as checked on Triton
3.7.0 — **not re-checked on 3.8.0**, so read its status off the build you are on
(`../hardware/capability-matrix.md`). **On the Gluon path none of this carries over as a knob:** in 3.8.0
no pass on the Gluon path consumes `num_stages` (budget parameter / champion record only), and
`waves_per_eu` copied from the champion caps the anchor's occupancy (`transcribe.md` §3).

For `@triton.autotune` as the final selector, keep ~8-12 high-confidence configs; use a separate offline
sweep for 20+. Config-frozen exit: stop the config route when all meaningful constexpr changes fail
correctness, tile dims define store ownership, the body hardcodes violated layout assumptions, or the knob
needs a body rewrite.

## 10. P15 — wrapper / launcher / dispatch-only

First-class for small/latency-bound kernels; low priority for throughput kernels. Start here when the
boundary is wrapper+kernel / full operator, the latency is small enough that Python overhead matters, or
the public API chooses configs / fallback / variants.

Audit before device work: `public wrapper / called kernel body / allocation path / metadata reads / config
selection / baseline effective config + source / dispatch or fallback policy / measured boundary`.

Common moves: avoid repeated `shape`/`stride`/dtype metadata reads in hot paths; simpler allocation when
shape/dtype/device match; avoid per-expert `nonzero()` loops (O(E) launches); cache host-side routing
decisions; avoid host dispatch that reads device tensors (`item()`) unless the sync is in-budget.
Launch-floor operators: remove hot-path asserts, cache temp/output tensors (only when the producer
overwrites every consumed element), inline shape arithmetic, cache launch configs/grid tuples, consider
fusion / graph replay / caller preallocation / boundary change. Expected wins are single-digit microseconds
— use the full boundary + repeat policy (`benchmark-hygiene.md`).

If the hot path is pure data rearrangement, compare PyTorch-native ops (`torch.take`,
`as_strided`+reshape, `view`/`permute`, `index_select`, `gather`, `scatter_`) under the same boundary before
writing a kernel. Keep the public ABI stable; report wrapper wins as integration wins.

Prefer **interface-preserving fusion of host preprocessing into the kernel** over a host copy: e.g. a
non-contiguous index/operand does not need a host `.contiguous()` — pass its row STRIDE and index it
strided in-kernel; a per-element dequant/scale can be pulled into the kernel. This removes the op from both
the host path and the timed window (the `.contiguous()` copy and its launch otherwise sit inside the
measured region) without changing the signature. (This is the only host-side lever once kernel-body work is
timed kernel-only under a CUDA graph — `benchmark-hygiene.md`, measurement basis.)

## 11. Search scaffolding -> final patch

Search stage may use helper functions / env knobs (record every knob + selected path). Finalization:
inline the kept heuristic into the smallest same-ABI patch, remove search-only layers, rerun correctness +
full benchmark on the minimal candidate. If the search version wins but the minimal patch does not, keep it
as evidence and do not ship the scaffolding.

## 12. Fixed-config CLIMB and stop conditions

After a sweep pins a comparator, CLIMB evaluates source-level changes at that fixed configuration. If a
kept body change can make configuration competitiveness stale, record config debt and run the declared
`body_refresh` SWEEP before making a refreshed comparator or handoff claim. Do not absorb that debt as
manual knob edits in the CLIMB ledger. A structural merge instead uses `post_merge`; each arm uses
`arm_local`; first and final publication use `initial` and `handoff`, respectively.

Stop a plain direction when the sweep only finds noise-band movement; winners do not form stable buckets;
threshold-adjacent shapes invalidate the rule; candidate overrides break tuned specialized configs; or
further search would be an unbounded parameter sweep rather than a named hypothesis. When a kernel is
deployed across Triton/ROCm builds, re-baseline and re-search version-sensitive knobs rather than assuming
a tuned config or a prior negative transfers. The champion is published only after the plain line was
climbed (`champion_gate.py [CLIMB]`, floor 10 rounds, or a priced `climb_declined` — `entry.md`).

---

## 13. Plain lever recipes (the `tl.*` form of each two-tier lever card)

The plain-Triton half of the lever cards. Every lever here is marked `expressible_in: ["plain", "gluon"]`
in `../hardware/lever-cards.json`, meaning **both tiers can write it** — but they write it differently, and
the card's main `authoring_ref` shows the Gluon form. This section is the `tl.*` form (the upstream
composer swaps it in as `plain_authoring_ref` when it filters the card set for the plain pack). Where the
recipe is already written out in a `../gluon/*` or `../tile-programming/*` page, it is a pointer here.

Read the paired **"what plain cannot pin"** note in each recipe before deciding to escalate: it is the
difference between the two tiers for that specific lever, and it is the honest content of a "plain is the
ceiling here" claim. A lever that plain expresses *indirectly* (you set an input, the compiler picks the
result) is a legitimate escalation trigger when the compiler picks wrong and you have IR showing it
(`entry.md` §3). Nothing here is a substitute for measuring. These are shapes, not predictions.

### Coalesce the access (plain)

Plain Triton has no layout to set, so coalescing is decided by the **index arithmetic**: the axis that
varies fastest across the lane dimension must be the contiguous axis in memory. Build offsets so the last
(fastest) dimension strides by 1.

```python
# BAD: the fastest-varying axis strides by N -> each lane touches a different cache line
offs = (offs_n[:, None] * stride_n) + (offs_k[None, :] * stride_k)   # k contiguous? not here

# GOOD: put the unit-stride axis last so consecutive lanes read consecutive addresses
offs_k = tl.arange(0, BLOCK_K)                  # contiguous in memory (stride_k == 1)
offs_n = tl.arange(0, BLOCK_N)
ptrs = base + offs_n[:, None] * stride_n + offs_k[None, :] * 1
x = tl.load(ptrs, mask=mask, other=0.0)
```

If the tensor's own layout puts the strided axis last, transpose the *access* (swap which axis indexes the
block) rather than the data, or pre-transpose outside the hot loop.

**What plain cannot pin:** the per-lane vector width and the thread->element mapping. Plain gets
"contiguous enough that the compiler *can* widen"; whether it emits `dwordx4` is the compiler's choice.
Verify in the ISA (`$KT/asm_loop_audit.py`) — if the addresses are contiguous and it still emits narrow
loads, that is the Gluon `wider_buffer_load` case (`../tile-programming/memory-path.md`).

### Cut HBM bytes (plain)

When the HBM pipe is saturated, more in-flight requests are inert — only fewer bytes help. All three plain
routes are source-level:

```python
# 1. narrow the dtype AT the load boundary (cast after loading the narrow value, not before)
a = tl.load(a_ptr + offs, mask=m, other=0.0)          # a_ptr is fp8/bf16 -> fewer bytes moved
acc = tl.dot(a.to(tl.float16), b.to(tl.float16), acc)  # widen in-register, for free

# 2. bias L2 residency on a re-read operand
w = tl.load(w_ptr + offs, mask=m, other=0.0, cache_modifier=".cg")

# 3. do not re-load an invariant: hoist it out of the K loop
bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)   # loop-invariant -> load ONCE
for k in range(0, K, BLOCK_K):
    ...
```

Fusing a separate streaming read->write pass into this kernel removes a whole round trip; that is a Layer 0
structure change (P2, §6), not a knob.

**What plain cannot pin:** nothing important. This lever is fully plain-expressible; the Gluon form differs
only in spelling. Do NOT escalate for it.

### Hide memory latency (plain)

**Pointer:** the mechanism, the pass list, how to confirm the pipeliner fired, the arch-gated ping-pong, and
the handoff criterion are written once in `../tile-programming/pipeline.md ## Plain Triton: the pipeliner runs for you`. The plain spelling, for reference:

```python
# the knob: more stages = deeper prefetch, but each stage costs a full LDS buffer
kernel[grid](..., BLOCK_K=64, num_stages=3, num_warps=4)
```

```bash
# confirm the pipeliner FIRED (it silently declines when LDS or registers do not fit)
bash "$KT/dump_ir.sh" python bench.py --variant plain --out ir/ --arch gfx950
grep -c 'ttg.memdesc_index' ir/plain/plain.ttgir      # multi-buffer local_alloc + memdesc_index = fired
grep -c 'async_copy' ir/plain/plain.ttgir             # gfx950 default async path; gfx942: expect 0
```

Sweep `num_stages` together with `BLOCK_K` (they trade against the same LDS budget) in the one batched SWEEP
(P8, §9). Read the depth off the **loop** (`tl.range(..., num_stages=N)`) as well as the launch
(`recover.md` §2.1).

**What plain cannot pin:** *where* the prefetch lands and how many buffers exist. The pipeliner declines
silently under register or LDS pressure, and when it declines there is no plain knob that forces it.
TTGIR showing no multi-buffer (`memdesc_index`) at `num_stages>1` — and, on gfx950 where async copy is the
default, no `async_copy` — is the canonical `manual_pipeline_prefetch` escalation evidence. (On gfx942 no
`async_copy` is expected from plain at all, so its absence there is not evidence.)

### Reuse a loaded operand (plain)

When the same operand region is read more than once per program (attention K/V across query rows, a GEMM
operand across N tiles), load it **once** outside the consumer loop and reuse the value.

```python
# BAD: re-reads the same K tile on every query block
for q in range(0, Q, BLOCK_Q):
    k = tl.load(k_ptr + k_offs, mask=k_mask, other=0.0)
    acc = tl.dot(tl.load(q_ptr + q_offs), k, acc)

# GOOD: hoist the invariant load; the value is reused from wherever the compiler placed it
k = tl.load(k_ptr + k_offs, mask=k_mask, other=0.0)
for q in range(0, Q, BLOCK_Q):
    acc = tl.dot(tl.load(q_ptr + q_offs), k, acc)
```

**What plain cannot pin:** whether the hoisted value lives in registers or LDS. Plain hands the compiler a
reuse *opportunity*; the Gluon version stages it into an explicit shared buffer and so controls both the
placement and the register cost. If hoisting spills (VGPR up, occupancy down), that is the
`reuse_shared_operand_load` escalation case, not a plain failure.

### Raise occupancy (plain)

Occupancy in plain Triton is a *consequence* of three inputs you do control: the tile size (register and LDS
footprint), the warp count, and the stage count.

```python
# each of these lowers the per-workgroup resource footprint -> more resident waves
kernel[grid](..., BLOCK_M=64,      # was 128: fewer accumulator registers per lane
             BLOCK_K=32,           # was 64:  smaller LDS buffer per stage
             num_stages=2,         # was 3:   one fewer buffer
             num_warps=4)
```

Occupancy is only worth buying when the kernel is **latency-bound with few resident waves**. Under the
`occupancy_gates_latency` gating law (`../hardware/bound-class-signals.md ## Lever gating laws (single source of truth)`), raising occupancy on a kernel that already has 2+ resident waves is neutral-to-negative — it
shrinks the tile for nothing. Read both limiters with `$KT/probe.py measure`.

**What plain cannot pin:** the register allocation itself. You cannot move the accumulator to AGPR, cap the
resident set, or dedup an LDS buffer; you can only shrink the inputs and hope. That indirection is the whole
`tile_slicing` / `reduce_accumulator_traffic` escalation argument.

### Shorten the critical path (plain)

Pure source arithmetic: use the cheap intrinsic, and hoist loop-invariant rescaling out of the dependency
chain.

```python
# 1. exp2 is a single hardware instruction; exp is a polynomial expansion
p = tl.math.exp2(qk * (scale * 1.44269504))     # fold log2(e) INTO the existing scale constant

# 2. hoist the rescale out of the inner chain: rescale the ACC once per block, not per element
acc = acc * alpha_correction        # once, after the loop
# not: acc += (a * alpha) @ b       # inside the loop, lengthening every iteration's chain
```

**What plain cannot pin:** nothing for the arithmetic itself. The Gluon form of this card additionally
reorders the *instruction schedule* around the shortened chain, which plain cannot express — but the
arithmetic win here is available in full. (When the exp unit itself is the bound, partial FMA emulation of
`exp2` is a different lever with a reversed sign elsewhere: `../gluon/technique-transfer.md ## Cross-layer: exp and exp2`.)

### Roll the loop (plain)

**Pointer:** the knob's semantics — the factor goes on the loop, never on the launch; `=1` pins the loop
rolled and `>1` asks for duplication; a host-side choice travels as a `tl.constexpr` the body names in the
annotation; and `loop_unroll_factor` is live on the Gluon path from 3.8.0 while `num_stages` is not — are
written once in `../gluon/pipeline/loop-knobs-and-targets.md` (and `../gluon/pipeline-reference.md ## Roll the loop (cut i-cache pressure)`). The plain spelling:

```python
# force the compiler to KEEP the loop rolled (do not unroll a large body into i-cache pressure)
for k in tl.range(0, tl.cdiv(K, BLOCK_K), loop_unroll_factor=1):
    ...
```

A fully unrolled hot loop can exceed the instruction cache, at which point every iteration pays an i-cache
miss. The signal is a large static instruction count in the loop body (`$KT/asm_loop_audit.py`) together
with instruction-fetch stall counters, not a guess. Suppression is the direction to reach for when the body
is a hand-scheduled sequence you do not want duplicated; amplification is for a body whose trip count is
only known at runtime, where no compile-time unrolling construct can be written at all.

**What plain cannot pin:** the resulting schedule inside the rolled body. Rolling removes the i-cache
pressure but also removes the cross-iteration overlap the unrolled form had; recovering that overlap
without unrolling is the explicit-pipeline (Gluon) case.

### Grow the tile to fill the matrix op (plain)

An under-filled matrix operation wastes the engine: if a tile dimension is smaller than the MFMA atom's,
the hardware still executes the full atom. Plain Triton controls this directly — the block sizes are
`tl.constexpr`.

```python
# a 16-row tile on an atom that is 32 rows wide runs at half the matrix throughput
kernel[grid](..., BLOCK_M=32, BLOCK_N=128, BLOCK_K=64)   # was BLOCK_M=16
```

Growing the tile costs registers and LDS, so it trades directly against occupancy — sweep it as one point
in the batched config SWEEP, not as an isolated CLIMB step.

**What plain cannot pin:** which MFMA atom the compiler selects. You choose the tile; it chooses the
instruction (and may pick a smaller atom than the tile would allow). Confirm the selected atom in
TTGIR/ISA — a tile that should fill a 32x32x16 atom but lowers onto 16x16x16 is the `mfma_tile_shape`
escalation case (`../gluon/atoms-reference.md`).

### Amortize the epilogue (plain)

When the epilogue (bias, activation, store, reduction) costs a fixed amount per program, growing the
parallel axis spreads that cost over more useful work.

```python
# each program now produces 4x the output rows for ONE epilogue execution
kernel[grid_smaller](..., BLOCK_M=256)      # was 64, with 4x the programs
```

Only pays when the epilogue is a measurable share of kernel time and the grid stays large enough to fill
the device — growing `BLOCK_M` shrinks the grid, so check `tail_efficiency` did not collapse (and the
saturation ratio, §4).

**What plain cannot pin:** the accumulator layout that a very large tile now needs. Past a point the grown
tile spills, and plain has no way to keep one accumulator fragment resident while converting once — that
is `reduce_accumulator_traffic`.

### Fold scalars off the vector chain (plain)

Work that is uniform across the block belongs on the scalar unit, computed once, not recomputed per lane
inside the vector chain.

```python
# BAD: a block-uniform quantity computed in the vector domain, inside the loop
for k in range(...):
    scale = tl.load(scale_ptr + pid)          # same value for every lane, re-loaded every iteration
    acc += tl.dot(a, b) * scale

# GOOD: hoist it; a block-uniform scalar stays on the scalar unit
scale = tl.load(scale_ptr + pid)              # once, outside; uniform -> SALU/SGPR
for k in range(...):
    acc += tl.dot(a, b)
acc = acc * scale                             # fold the uniform factor once, at the end
```

**What plain cannot pin:** nothing material — uniformity analysis is the compiler's job and it does it well
when the value is genuinely loop-invariant and block-uniform. Confirm in the ISA that the value landed in an
SGPR (`s_load` / `s_mul`) rather than a VGPR.

### Predicate, do not branch (plain)

Divergent control flow serializes the wave. Plain Triton's native idiom is already predication: masks on
loads/stores and `tl.where` for selects.

```python
# BAD: data-dependent branch -> both sides execute, serialized
if x > 0:
    y = f(x)
else:
    y = g(x)

# GOOD: predicate -- one pass, no divergence
y = tl.where(x > 0, f_val, g_val)

# masked memory ops are predication too: no branch around the boundary tile
x = tl.load(ptr + offs, mask=offs < N, other=0.0)
tl.store(out + offs, y, mask=offs < N)
```

Uniform (block-invariant) branches are fine — they do not diverge. Only *lane-varying* conditions cost.

**What plain cannot pin:** nothing. This lever is fully plain-expressible and plain is arguably the better
tier for it. Do NOT escalate for it.

### Downcast rounding (plain)

**Pointer:** the rounding-mode mechanism, its upstream source (`cast(..., fp_downcast_rounding=)`,
truncating-only guard), the verification recipe (convert-class share and VALU:MFMA down, oracle re-run,
round-half-up unchanged to ~1 ULP, RTZ shown under tol) and the split-precision use that *requires* RTZ
are written once in `../tile-programming/low-precision.md` (and the accumulator/epilogue form in
`../gluon/matrix-reference.md`). The plain spelling:

```python
out = acc.to(tl.float16)                      # default: round-to-nearest-even
# RTZ where the tolerance allows it (verify against the reference, do not assume):
out = acc.to(tl.float16, fp_downcast_rounding="rtz")
tl.store(out_ptr + offs, out, mask=m)
```

Round-toward-zero is cheaper than round-to-nearest-even on some paths, and the difference is measurable in
an epilogue-bound kernel — but it changes numerics. Treat any rounding-mode change as a **numerics** change:
re-run the correctness comparison at the tolerance the caller actually requires, and record the tolerance in
the result. A speedup bought by silently loosening numerics is not a speedup.

**What plain cannot pin:** nothing for the rounding mode itself. The Gluon form additionally controls the
packing of the narrowed values into the output layout.
