# Gluon — pipeline (API router)

API surface only: *which spelling exists on which build* and *which chapter owns the mechanism*. How the
mechanisms compare: `../tile-programming/pipeline.md`.

**Gluon ships no auto-pipeliner and no mbarrier pipeline library:** `gluon_to_ttgir` does not call
`add_schedule_loops` / `add_pipeline` (3.6.0–3.8.0; 3.8.0 source-checked), so overlap is **authored**.
Re-injecting plain's passes is a diagnostic / fallback (`pipeline/reinjection.md`). Chapters are written for 3.8.0 on
gfx950 with gfx942 downgrade notes; where your build disagrees, re-probe (`../../scripts/probe_levers.py --all`).

| Read this when | Chapter |
| --- | --- |
| gfx950 async LDS ring (`async_copy` + `commit_group` / `wait_group`), drain depth, barrier placement | `pipeline/async-ordering.md` |
| hand-built overlap: register prefetch, mechanism × arch tables (A/B/S), gfx942 worked set | `pipeline/authored-overlap.md` |
| `warp_pipeline_stage`, fork-only `warp_predicate`, gfx942 async-copy width gate | `pipeline/marker-and-version-gates.md` |
| per-target path, `num_stages`, `loop_unroll_factor` | `pipeline/loop-knobs-and-targets.md` |
| diagnostic only: sizing `lost_pipeline` debt by re-injecting plain's pipeliner | `pipeline/reinjection.md` |

**`num_stages` is inert on the Gluon path (3.6.0–3.8.0; 3.8.0 source-checked)** — no pass reads it; it is only a buffer-count
budget parameter (`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`).

## Which spelling exists on which build — the four-version probe table

3.8.0 column confirmed against a full tree; older columns are container probes. Pass rows come from
`libtriton` / `third_party/amd`, not from a Python-only extraction.

| | 3.6.0 | 3.7.0 | 3.7.1 | 3.8.0 | owner |
| --- | --- | --- | --- | --- | --- |
| `gl.amd.cdna4.async_copy` (`global_load_to_shared`, `buffer_load_to_shared`, `commit_group`, `wait_group`, `load_shared_relaxed`) | ✓ | ✓ | ✓ | ✓ | `pipeline/async-ordering.md` |
| `cdna4.compute_efficient_padded_shared_layout` (gfx950-only: asserts MFMA `version == 4`) | **absent** | **absent** | **absent** | ✓ | `layout-reference.md ## Let 3.8 Compute The Padded Layout For You (CDNA4)` |
| `gl.barrier` (3.6.0: `gl.thread_barrier`, no alias) | `thread_barrier` | ✓ | ✓ | ✓ | `pipeline/authored-overlap.md` |
| `gl.amd.warp_pipeline_stage` / `add_warp_pipeline` (symbol) | **absent** | ✓ | ✓ | ✓ | `pipeline/marker-and-version-gates.md` |
| … phase offset race-free (GPU, gfx950; 3.9.0 also race-free) | — | not tested | **✗** wrong results | ✓ | `../tile-programming/warp-pipeline.md` |
| `add_schedule_loops` / `add_pipeline` in `libtriton` | ✓ | ✓ | ✓ | ✓ | `pipeline/reinjection.md` |
| … called from `gluon_to_ttgir` | ✗ | ✗ | ✗ | ✗ | same |
| `num_stages` read by any pass on the Gluon path | ✗ | ✗ | ✗ | ✗ | `pipeline/loop-knobs-and-targets.md` |
| `passes.ttir.add_loop_unroll` in `gluon_to_ttgir` (makes `loop_unroll_factor` live) | ✗ | ✗ | ✗ | **✓** | same |
| shared-descriptor `.gather` / `.scatter` | **absent** | ✓ | ✓ | ✓ | `smem-lds-reference.md` |
| shared-descriptor `atomic_scatter_*` | **absent** | **absent** | **absent** | ✓ | `smem-lds-reference.md` |
| `._reinterpret` with defaulted `dtype`/`shape`/`layout` (all explicit works everywhere) | **absent** | **absent** | **absent** | ✓ | `smem-lds-reference.md` |
| `gl.dot_fma` with rank-3 (batched) accumulator | **absent** | **absent** | **absent** | ✓ | `matrix-reference.md` |
| `llvm_fn_attrs` compile option | **absent** | **absent** | **absent** | ✓ | `../tile-programming/llvm-fn-attrs.md` |
| `sched_barrier` / `sched_group_barrier` / `set_prio` in `gl.amd.cdna3`/`.cdna4` | **absent** | **absent** | **absent** | **absent** | `pipeline/authored-overlap.md` (B2) |
| `gl.warp_predicate` (vendor fork only) | **absent** | **absent** | **absent** | **absent** | `pipeline/marker-and-version-gates.md` |

- A failed `hasattr(cdna4, "compute_efficient_padded_shared_layout")` on 3.7.x is a **version** gate, not an arch gate.
- On gfx942 the same `cdna4.async_copy` entry points lower only at 32 bits/thread
  (`pipeline/marker-and-version-gates.md ## The gfx942 async-copy width gate — available, and narrow`).
- **Co-exec scheduler strategy is set only on gfx1250 in 3.8.0** (`is_coexec_scheduler_supported`); on gfx950/gfx942
  `llvm_fn_attrs=[["amdgpu-sched-strategy", "coexec"]]` draws an LLVM "only supported for gfx1250"
  warning and a non-coexec schedule (`../tile-programming/non-upstream-reserve.md`). Fork-only env vars name nothing on a stock build.

## Getting `wait_group` right

Group semantics, the tail race, `load_shared_relaxed`, trace-time depth: `pipeline/async-ordering.md ## Getting wait_group right`.

## Authored overlap (no compiler patch)

Register prefetch, A/B/S availability per arch, gfx942 worked set: `pipeline/authored-overlap.md ## Authored overlap (no compiler patch)`.

## Roll the loop (cut i-cache pressure)

`loop_unroll_factor` is a loop keyword (read by a pass on 3.8.0 only), `num_stages` is inert:
`pipeline/loop-knobs-and-targets.md ## Roll the loop (cut i-cache pressure)`.

## Re-injecting plain's pipeliner — the measured recipe

Diagnostic / fallback, numbers labelled `injected` (they describe plain's overlap, not an incumbent Gluon kernel): `pipeline/reinjection.md`.

## And on attention — two dots chained through a softmax

Whether re-injection reaches a two-dot softmax body: `pipeline/reinjection.md`.
