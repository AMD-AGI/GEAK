# Recover: pay the transcription debt by hand, then clear the parity gate

Advisory. After a faithful anchor passes `transcribe.md`, this page attributes the anchor→plain gap
to `lost_pipeline` / `lost_layout` / `lost_RA` from compiled artifacts and lists how each is repaid.
`$SKILL` = `perf_knowledge/expert_skills/skills/gluon_authoring`; `$KT` =
`kernel_workflow/scripts/kernel_tools`. gfx950 main line; gfx942 notes are downgrades. Port entry on gfx950:
[`### The async-ring loop form that reaches plain parity (gfx950)`](#the-async-ring-loop-form-that-reaches-plain-parity-gfx950)
and the `plain@ns=1` control (§2.1); the steps before it are `transcribe.md ## Port checklist`.

**Why a debt exists: `gluon_to_ttgir` runs no software pipeliner (`add_schedule_loops` +
`add_pipeline`) and no automatic block ping-pong** — checked on 3.6.0 / 3.7.0 / 3.7.1 / 3.8.0 — so
plain's `num_stages` overlap does not survive transcription, and on the Gluon path no 3.8.0 pass
consumes `num_stages`. The debt is often **zero** (champion compiled at `num_stages=1`, or its shipped
depth was a pessimisation); size it with `plain@ns=1` (§2.1) before building anything. Compare against
the plain baseline tuned to its own best config, never a default-config strawman.

## 2. Attribute: which of the three suspects owes the gap

Dump champion and anchor under their own variant names (`$KT/dump_ir.sh <cmd> --variant <name> --out ir/
--arch gfx950` → `ir/<name>/<name>.{ttgir,amdgcn}` + `meta_*.json`) and compare:

- **LDS bytes from the Triton cache metadata `shared` (`meta_*.json`); the `.amdgcn` `LDSByteSize` is a
  structural 0 on Triton kernels.** Without it the LDS half is untested, not cleared.
- **Occupancy = min(register term, LDS term) per arm; name the binding term.** `; Occupancy: N` is the
  register term only, and an anchor that lost a wave to a materialized staging buffer moves no register
  count.
- **Register cliff on `.amdhsa_next_free_vgpr` (ArchVGPR + AGPR share one 512-register file per
  SIMD)**, not `num_vgpr`. `$KT/probe.py measure --dir ir/anchor/` gives both limiters compile-only.

| suspect | artifact signal | repaid by |
| --- | --- | --- |
| **`lost_pipeline`** | champion TTGIR has `ttg.memdesc_index` / `ttg.local_store` / `tt.num_stages > 1` / a peeled prologue, anchor does not; anchor ≈ `plain@ns=1` < `plain`. **`iter_args >= 2` is not evidence** (every accumulator / online-softmax loop has it; `ttgir_bridge.py` reads the depth from the indexed LDS ring or the compile metadata, else reports INCONCLUSIVE) | authored pipeline (§3.1) |
| **`lost_layout`** | load/LDS width narrowed (`dwordx4`→`ushort`, `ds_read_b128`→`ds_read_u16`), or `shared` crossed the LDS/CU divisor (160 KiB gfx950; 64 KiB gfx942), occupancy LDS-bound | re-recover (`transcribe.md`) |
| **`lost_RA`** | same instruction multiset, but VGPR rose across a wave threshold, spill appeared, or the **distinct address registers feeding the `ds_read` burst collapsed: an address rematerialized just above each `ds_read` puts every read behind a WAR hazard on one register** — invisible to layout checks; read `ds_read` operands, not counts. Reproduced on gfx950 / 3.8.0: the cost is largest when the burst is latency-exposed (one wave per SIMD) and mostly hidden when a second wave per SIMD can issue | slicing / register budget |

Sub-cases that fold into the three:

| signature | owner | fix |
| --- | --- | --- |
| **lost vectorization**: anchor far below plain, load histogram `dwordx4` → `ushort` (a `convert_layout` folded backwards into the load) | `lost_layout` | **give that `local_alloc` its own `allocate_shared_memory` with the recovered `*_SMEM` layout** — restored `dwordx4` and beat a plain `gl.convert_layout` in a reproduction on gfx950 / Triton 3.8.0. There an explicit `gl.convert_layout` was not folded into the load; the narrow load came from a transcribed `tt.load` that already carried the dot-operand layout — confirm the load width in the `.s` |
| **LDS budget**: every layout verifies, `shared` crosses the divisor, one fewer WG/CU | `lost_layout` | pass-through vs staged `local_alloc` choice (`transcribe.md ### Not every`); `verify` is blind to size |
| **lost schedule**: identical multiset, different waits (one kernel: **21 relaxed `lgkmcnt` waits vs plain's 10**) | `lost_RA` family | reorder the body toward plain's program order |
| MFMA clumped behind memory | `lost_pipeline` symptom | pipeline |

If all three clear and the gap remains, decompose wall time (wrapper / host / kernel); time an empty
kernel at the same grid and at `(1,1,1)` for a dispatch claim.

### 2.1 The `plain@ns=1` control — sizing `lost_pipeline` before paying it

Re-run plain at its own config with the pipeline off. How: call the plain launcher
with its own config dict and `num_stages` overridden to 1 (for aiter GEMMs: `cfg, _ = _get_config(M, N, K);
cfg["num_stages"] = 1; gemm(..., config=cfg)`), time it with the task's timer, and dump it once with
`dump_ir.sh` to confirm the loop lost its `memdesc<2x…>` / `buffer_load_to_local`. A faithful anchor
(`transcribe.md ## Port checklist` step 4) landed on `plain@ns=1` within noise on every bucket of an fp8 GEMM.

| reading | meaning |
| --- | --- |
| `plain@ns=1` ≈ `plain` | never pipelined; no debt. A faithful anchor should land ≈1.00; below that is `lost_layout`/`lost_RA` |
| `plain@ns=1` **faster** | shipped `num_stages` is a pessimisation — recovering it would recover a slowdown. **A wrapper that passes no `num_stages` inherits the AMD default `num_stages=2`** |
| anchor ≈ `plain@ns=1` < `plain` | the whole gap is the pipeline |

**Read which knob sets the depth — off the loop, not the launch:**

| loop | depth set by | flipping the launch arg |
| --- | --- | --- |
| `tl.range(..., num_stages=N)` | **the annotation sets the loop's depth regardless of the launch argument** | the TTGIR is identical at every launch depth; the launch value can still change later lowering (register allocation in the `.amdgcn`) |
| bare `range` with a `tt.dot` | launch argument | works |
| bare `range`, **dot-free** | nothing — no anchor op to pipeline | inert |

**`tl.range(num_stages=None)` inherits the launch value** (tuned kernels write `num_stages = None if
ENABLE_PIPELINING else 1`). A hard-coded `tl.range(num_stages=1)` makes all three arms byte-identical
and fakes "no debt". Settle it on the dump: re-dump at launch `num_stages` 1/2/3 and diff the `.ttgir`
(load count and `memdesc_index` scale with the real depth). `$SKILL/scripts/pipeline_survey.py <tree>`
screens which pipeline forms a source tree can exercise; only a dump settles a dispatch.
Verified on Triton 3.8.0 / gfx950.

**A pessimising depth is usually a register wall: read `spill=`.** Measured, tile-only variation: the
wide tile at `ns=3` was **clearly slower than its own `ns=1`** (pinned at 256 VGPRs, spilling inside the
loop) while the narrow tile gained; `WGs/CU by LDS` was 1 for both. The debt sign can flip per shape
and per arch (narrower MFMA keeps 2x the dot-operand registers live). Size it at the champion's own
tile, per shape (`../gluon/pipeline/reinjection.md ### What to expect, and what to measure yourself`).

## 3. Repay by hand, per suspect

Fix a structural success signal (buffer count, barriers, `lgkmcnt` shape, `ds_read` width, spill)
before reading a clock.

### 3.1 `lost_pipeline` → an authored pipeline (hand-written before re-injection)

Forms (`../tile-programming/pipeline.md ## Authoring the overlap yourself (the climb default on the
Gluon path)`), chosen by signal: register prefetch fits a register-resident tile or few MFMAs per
iteration; an authored LDS ring fits a compute-bound LDS-staged tile (the champion's own staging is the
hint); `warp_pipeline_stage` + scheduling model sit on top (the warp-group split needs `num_warps >= 8`).

- **gfx950 async ring: `buffer_load_to_shared` / `global_load_to_shared` + `commit_group` /
  `wait_group`, 128- or 32-bit per lane. Falsifiable signature: `ds_write == 0` in the loop with a
  matching count of direct-to-LDS loads.** Runnable check: `scripts/pipeline_examples_cdna4.py`.
- **gfx942: sync staging (`.store()` / `.load()`); barriers are inserted by the compiler, so an
  extra `gl.barrier()` "to be safe" adds a sync without adding safety.** Async only at 32 bit/lane with `order=[1, 0]`
  (`scripts/pipeline_examples_cdna3.py`).
- Each extra buffer is one more copy of the staged tiles; check the LDS divisor before deepening.
- **A hand register-prefetch and a re-injected pipeliner are not expected to compose** (the pipeliner is the
  prefetcher; a manual one consumes its slot), and a hand-staged loop and a re-injected one are
  different bodies.

### The async-ring loop form that reaches plain parity (gfx950)

Verified on Triton 3.8 / gfx950 on an fp8 GEMM whose tuned plain dump stages through
`amdg.buffer_load_to_local` (plain's async pipeliner). A ring that copies plain's *structure* but not its
*compiled loop form* stayed measurably below plain on the large-tile buckets; copying the loop form
closed it to a tie on every bucket. Steps, each with its check:

1. **Layouts from the tuned (pipelined) dump**, not the `num_stages=1` one: the async-copy source layouts
   (`TO_SMEM` / `TO_SMEM_2`, linear, 16 B per lane along K) and the `padded_shared` `A_SMEM` / `B_SMEM`.
   Allocate `[NUM_STAGES, BM, BK]` with the 2-D recovered shared layout. *Check:* `probe.py measure --dir`
   reports the same `lds/WG` as plain.
2. **Depth: on the tiles measured, plain's tuned `num_stages` was fastest.** One more stage on a 128x128x128 fp8 tile moves
   LDS/WG across the 160 KiB/CU divisor (2 → 1 WG/CU) and is slower; fewer stages on the small-M
   buckets is slower too. *Check:* `WGs/CU by LDS` in `probe.py` before timing a depth change.
3. **Prologue**: issue tiles `0..S-2` with a K mask (the tile may not exist), one `commit_group` each.
4. **Steady state, bare**: loop `num_k - (S-1)` times doing `wait_group(S-2)` → issue the copies for tile
   `k+S-1` into slot `(k+S-1) % S` (no mask, no `other`) → `commit_group` → LDS reads of slot `k % S`
   → `mfma`. Issuing the next copies *after* the LDS reads, with a mask on every in-loop copy, was the
   slower form (re-measured on gfx950 / Triton 3.8.0 at 4- and 8-warp prefill tiles and decode tiles: the
   issue order carries it; a mask alone tied). *Check:* `asm_loop_audit.py` hot loop has `buffer_load_dwordx4` with no `v_cndmask`
   and the same `s_barrier` count as plain's loop.
5. **Peeled drain**: `S-1` iterations of `wait_group(S-2-d)` → LDS read → `mfma`, no copies.
6. **`other=` on `buffer_load_to_shared` measured slower** — plain's sites carry none (OOB/masked lanes read
   0). With `other=0.0` the TTGIR shows `buffer_load_to_local ... other = %cst` and the kernel was
   markedly slower on a mid-M bucket. A mask alone on in-loop copies measured a tie, so the bare form is
   for parity hygiene, not a large lever.
7. **Shapes whose rows are not 16-byte aligned** (fp8 `K % 16 != 0`) cannot use 16-byte async copies:
   compile fails with `LLVM Translation failed for operation: builtin.unrealized_conversion_cast`.
   Plain's pipeliner falls back to register staging there; route those shapes to a register-prefetch
   kernel (step 1 of §3.1) or to plain, and keep them in the correctness set. (The only other legal
   per-lane width on Triton 3.8.0 is 32 bit, gfx950 and gfx942: `../gluon/memory-reference.md ### Minimum per-thread granularity (applicability by dtype)`.)
8. **Not levers here**: `waves_per_eu` carried from plain and `amdgpu-sched-strategy=iterative-ilp`
   were ties on this ring. Past parity, candidate levers are listed in the playbook
   (`playbook.md ## Skill levers on gfx950 (signal-gated)`: 8-warp ping-pong, coalesced epilogue).

A register-prefetch body (§3.1 step 1: load tile `k+1` into registers before the LDS store / MFMA of
tile `k`) did not reach plain on the measured gfx950 GEMM (a tie at best on the small-M buckets, slower on
the large-K mid-M bucket); it served the step 7 (unaligned) shapes, and the async LDS ring was the parity
route.

### 3.2 `lost_layout` → re-recover

Back to `transcribe.md`; a hand-derived basis cannot be verified there. Choose between numerically identical shared layouts
with `ttgir_bridge.py view` (their `ds_read`/`ds_write` mix) before timing — one anchor went from ~half
of plain to parity in one round that way.

### 3.3 `lost_RA` → slicing / register budget

`../tile-programming/slicing.md`; not visible in TTGIR. The champion's `waves_per_eu` pins the
anchor's occupancy to exactly that value when carried over (`amdgpu-waves-per-eu` "N, N"), which can
lower it; A/B it rather than inheriting it — forcing a higher value can also lose (spills). A correctly attributed residual can still not move if a second limiter
binds at the same point.

## 4. The parity gate

Suggested claim check: until `plain_baseline / current` meets the run's declared criterion, a round is a
recovery, not a win. Below it, check cheapest first: the ring did not land (no `ttg.memdesc_index`, no
relaxed `lgkmcnt`); a recovered layout not wired onto an operand (re-run `verify`, read `missing`); a
non-pipeline suspect. If `plain@ns=1` showed no debt, the residual is not a `lost_pipeline` debt. An
inexpressible residual (`UNRECOVERABLE` `amd_rotating_shared`, fp8's unscaled `tt.dot_scaled`) is a
hand-back signal, not more recovery rounds.

---

## Last resort: re-injecting plain's pipeliner

**Diagnostic / last resort only, below the parity gate, on a converted (mode A) anchor** — to size
`lost_pipeline` (injected − anchor bounds the debt) or when the hand-written repayment (§3.1) cannot reach
parity. Label every number `injected`, never a win; never on an incumbent (mode B); one armed/unarmed variant
per process. Mechanism, kernel conditions (dot anchor, `gl.load` not `buffer_load`, hand staging removed),
error strings, the cache trap, IR landing tells, depth and measured outcomes: one home,
`../gluon/pipeline/reinjection.md ## Re-injecting plain's pipeliner — the measured recipe`.
