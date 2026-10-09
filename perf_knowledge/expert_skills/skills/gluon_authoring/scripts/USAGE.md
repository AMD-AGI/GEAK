# scripts — usage, by stage

Run from the GEAK repo root with

```bash
SKILL=perf_knowledge/expert_skills/skills/gluon_authoring     # the skill: Gluon-specific tools + pack runtime
KT=kernel_workflow/scripts/kernel_tools                       # GEAK's shared kernel tools (ISA, profiling, roofline)
```

Three owners, and the skill defers to the first two:

- **GEAK infrastructure** — `kernel_workflow/scripts/gpu_lock.sh` (the only GPU lock; lock dir
  `/tmp/team_gpu_locks`, override `GEAK_GPU_LOCK_DIR`), `kernel_workflow/scripts/profile_kernel.sh` +
  `profile_policy.sh` (the profiler entry), `e2e_workflow/scripts/harness_lib.py` (acceptance timing and
  correctness), `scripts/gpu_identity.py` (`gfx`, `target`, `sku`).
- **`$KT`** — tools with no GEAK counterpart that GEAK now shares across workflows. Their data is
  `perf_knowledge/hardware/data/` (resolved by `$KT/_hwdata.py`; override `GEAK_HW_DATA_DIR`).
- **`$SKILL/scripts`** — Gluon-specific tools and the pack runtime. Every tool that moved to `$KT` keeps a
  **shim** here, so `$SKILL/scripts/<tool>` and `import <tool>` still work.

**Arch is never defaulted.** A tool that needs one refuses without `--arch` (or a dump that names its target);
examples use `--arch gfx950`, the main line, with gfx942 as the downgrade.

## Stage map

| stage | tools | decides |
| --- | --- | --- |
| entry | `$SKILL/scripts/champion_gate.py`, `env_gate.sh`, `probe_levers.py` | the bundle is assertable; the box can run; which version-sensitive knobs are live |
| budget | `$KT/hw_budget.py`, `calc_perf.py`, `mem_bw_probe.py`, `extract_sku.py` | the prize, the floor, the bound-class prior |
| transcribe | `$SKILL/scripts/ttgir_bridge.py`, `recover_gluon.py`, `ttgir_to_gluon.py`; `$KT/dump_ir.sh`, `probe.py`, `amd_occupancy.py` | a faithful, equivalence-checked anchor |
| recover | `$SKILL/scripts/parity_gate.py`, `pipeline_survey.py`; last resort: `gluon_swp.py`, `patch_reinject.py`, `patch_async_reinject.py` | who owes the gap; parity |
| evidence | `kernel_workflow/scripts/profile_kernel.sh` (GEAK); `$KT/capture.sh`, `rocprofv3_safe.sh`, `rocprof_compute_probe.sh`, `parse_pmc.py`, `parse_rc.py`, `kernel_breakdown.py`, `hotspot_analyzer.py`, `att_*.py`, `tile_trace.py`, `serve_traces.py`, `asm_loop_audit.py`, `asm_schedule_viz.py`, `mfma_efficiency.py`, `deep_mfma_analysis.py`, `layout_facts.py`, `gfx950_isa.py`, `hw_sources.sh` | the four dials |
| measure | `harness_lib.py` (GEAK, acceptance); `$KT/ab_bench.py` (screening), `create_harness.py`, `parse_correctness.py`, `harness_stub_env.py` | is the difference real |
| climb | `$SKILL/scripts/lever_index.py`, `pipeline_examples_cdna4.py` (gfx950), `pipeline_examples_cdna3.py` (gfx942 downgrade) | which AMD lever applies to the named bound |
| close / runtime | `$SKILL/scripts/toolctl.py`, `canonical_record.py`, `round_record.py`, `recordctl.py`, `close_audit.py`, `report_lint.py`, `served_envelope.py`, `run_state.py`, `stage_context.py`, `context_query.py`, `context_contracts.py`, `source_excerpt.py`, `profile_payload.py`, `debt.py`, `skill_index.py`, `wait_for.sh`, `locus.sh`, `runtime_env.sh` | the auditable record |
| pack hygiene | `$SKILL/scripts/selftest_all.sh`, `smoke_test_recover.sh`, `check_pack_refs.py`, `check_term_index.py` | the toolchain and the references are sound offline |

## Optional: the pack's record tools

GEAK's roles call the tools in this file directly (`skill.md ## Roles in GEAK`). The pack's staged runtime is
**optional bookkeeping** a deep_engineer may use to keep its records — not a separate run mode:

```bash
python3 scripts/toolctl.py stage --work "$WORK" --role deep --json      # stage card + its method section
python3 scripts/toolctl.py context query --work "$WORK" --role deep --artifact mref --markdown-heading "<heading>"
```

`$WORK` must be absolute (e.g. the engineer's `OUTPUT_DIR`).

## Entry

| command | gives you |
| --- | --- |
| **`champion_gate.py --champion <bundle> [--allow-provisional] [--allow-ungated] [--allow-default-anchor] [--allow-shallow-climb] [--json]`** | **G1, both entry modes.** Fourteen checks, each printed by name (`references/method/entry.md ## Stage-Entry: the champion assertion` lists them with their status and escape flag). The two that fail most silently: `SOURCE` (the recorded sha still describes the file on disk) and `LIVE` (the file the run loads is byte-identical to the measured one — a comparator overwritten by a later winner passes `SOURCE` and fails here). Also `CONFIG` (dump from the pinned config, cross-checked against `ttg.num-warps`), `COMPARATOR` (not an inverted strawman), `GATED` / `SAMPLING` (oracle-gated sweep, covered grid), `CLIMB` (the front end actually climbed), `LOCUS` / `TOOLCHAIN` (measured where it runs). **`SAMPLING`'s PASS is unfalsified, not verified** — spot-check the pin at ±1 grid step per swept axis. An escape flag declares a limitation into `caveats[]`; it does not remove it. On failure: edit nothing, report `blocked`. |
| `env_gate.sh` | can this box run the stages: exit **0** clear · **1** hard blocker · **2** nothing to test · **3** degraded (a profiler missing, rocprof-compute unsupported on RDNA) — degraded is not a failed run |
| `probe_levers.py --all [--arch gfx950]` | which version-sensitive knobs are `live` / `dead-declaration` / `absent` on **this** build. There is no positional probe name. Run before sweeping a knob: out-of-range knobs are accepted and change no IR |

## Budget

| command | gives you |
| --- | --- |
| **`$KT/hw_budget.py --sku MI355X --workload <archetype> --shapes k=v,… --dtype bf16`** | no GPU. Intensity vs the SKU ridge → bound-class prior with `bound_direction`; the MFMA-only floor and your multiple over it; the `references/workloads/index.md` row for the archetype (read it); `numerator_basis` / `denominator_basis` / `may_gate` on every ratio. Calibrate with `--measured-dram-mb`, `--measured-hbm-tb-s`, `--measured-tflops lo,hi`; declare what the kernel moves with `--tensors`, `--flops`, `--flops-engine`; C0 with `--grid`, `--footprint-mb`, `--dispatches` |
| `$KT/calc_perf.py --sku <SKU> …` | achieved TFLOP/s and bandwidth against `sku.json`; refuses a dtype the row does not list and refuses without `--sku` / `--arch` / manual peaks |
| `$KT/mem_bw_probe.py --sku <SKU> --runlen … --stride … --rw-mix …` | the **in-shape** memory ceiling (a range, with its program count) — the denominator a gate may use |
| `$KT/extract_sku.py --print <SKU> \| --check \| --render … \| --check-docs \| --sync-docs` | reads, validates and renders `perf_knowledge/hardware/data/sku.json` (the single per-SKU source); every SKU table in the docs is generated from it |

## Transcription

Start with `ttgir_bridge.py`. The two older tools below it are still the only option where `import triton`
is unavailable, but they carry a hand-written mapping and a text-level equivalence check.

| tool | what it does |
| --- | --- |
| **`ttgir_bridge.py recover\|verify\|view --arch gfx950`** | **The recovery path.** Hands the `.ttgir` to the compiler's own parser and upstream's `layoutToGluon()`, so there is no per-kind mapping to fall behind Triton: an unsupported kind surfaces as a named `UNRECOVERABLE` row. Every layout carries a **round-trip proof** and a `num_warps`-vs-`warps_per_cta` cross-check. `verify` compares **LinearLayout normal forms** (two spellings of one layout compare equal; unroll skew is informational `MULTIPLICITY`). Needs `import triton` (≥ 3.7 for full capability; 3.6 compares shared layouts as text, no `view`). No GPU and no ROCm needed — `--arch` works anywhere, and is required. |
| `$KT/dump_ir.sh <compile cmd …> --variant <name> --out <ir_dir> --arch gfx950 [--knobs …] [--emit-gluon layouts\|anchor\|pipeline] [--kernel module.path:object] [--kernel-name <substring>]` | runs a compile with IR dumping and collects `.ttir` / `.ttgir` / `.amdgcn` per variant; accepts a single `bash -c` string as the command. **`--kernel-name` pins which compiled kernel's artifacts are taken** (multi-kernel ops); `--kernel` is `module.path:object` for the translator and a bare name there is refused. Old caches are moved aside, never deleted. `--emit-gluon` refuses without an arch |
| `recover_gluon.py` | the older driver: dump → recover → emit an anchor → `--verify`. Use it for **anchor assembly** (`--with-skeleton`) and `--record`; take the layout gate from `ttgir_bridge.py verify`. Its `--verify` compares canonical text as a set — sound in one direction only. Peels `@triton.autotune` / `@triton.heuristics` before translating |
| `ttgir_to_gluon.py` | the pure-text parser/emitter under `recover_gluon.py`; no GPU and no `triton` import — the one reason to reach for it. Emits `tilesPerWarp` / `elementBitWidth` on `AMDMFMALayout` when the TTGIR prints them. Its output is a starting point, not a proof; it does not place `convert_layout` and cannot name `amd_rotating_shared` |
| **`$KT/probe.py measure --dir <ir_dir> [--arch gfx950]` · `plan --arch gfx950 …`** | **compile-only occupancy (G4) — run it the moment the anchor builds.** `measure` parses the artifact for `shared` bytes/WG, ArchVGPR+AGPR and waves/SIMD in seconds, joined per kernel, and prints **both** limiters in workgroups/CU and which binds; `plan` adds up resident tensors on paper. The only instrument that sees the LDS cost of transcribing a pass-through `ttg.local_alloc` as a user buffer |
| `$KT/amd_occupancy.py --vgpr N --arch <gfx>` | the occupancy model `probe.py` delegates to (CDNA: ArchVGPR+AGPR in one 512-entry file, granule 8, cap 8; LDS term per arch); `--compiler-sweep` derives an RDNA table from `llc` |
| `smoke_test_recover.sh [<gfx950-gluon-tutorials checkout>]` | offline end-to-end check of the port toolchain and every port tool's `--selftest`; run it before trusting these scripts on a new box |
| `smoke_recover_gpu.py` | the on-GPU version; needs `torch` + `triton` |

### Reading `ttgir_bridge.py recover`

Four things it prints that decide the port, in the order they matter:

- **`num_warps cross-check`** must be `PASS`. A `FAIL` means the dump contradicts itself and nothing in
  it is trustworthy — **exit 4**, deliberately distinct from the exit 1 that an `UNRECOVERABLE` layout
  gives, because that one means "part of this kernel is not expressible, the rest is sound".
- **`UNRECOVERABLE: N`** must be 0. The one seen in practice is `amd_rotating_shared`. Treat the row as a
  **prompt to probe your own build, not as proof of a language gap** — the Gluon surface moves, and
  `amd_wmma` sat behind identical wording while being constructible as `AMDWMMALayout` all along. It is
  also **not** a `num_stages` artefact — one kernel still shows it at `num_stages=1` — so re-dump at
  `ns=1` and re-recover before concluding the body is untranscribable. Where the constructor really is
  absent, the consequence is worse than a missing constructor and there is **no Python-side workaround**:
  `builder.to_linear_layout(attr, shape)` wants an `ir.attribute`, and on 3.7.1 / 3.8.0 no binding
  obtains an encoding attribute from a Value or a Type (`ir.value` exposes get_type/get_shape/get_loc,
  `ir.type` exposes is_fp16/is_integer, `ir.make_attr` builds dense integer arrays). So that layout's
  normal form is unreachable from Python and **a substitution for it cannot be verified against the
  original even in principle.** Closing it needs a C++ binding; the earlier claim that one Python binding
  (`to_linear_layout_from_memdesc`) would suffice was tried and is **retracted**.
- **`round-trip: EXACT=N`** must cover every layout. This is the proof the Python object kept every
  field the attribute had, and it is the check a hand-written mapping cannot offer.
- **the source names, not the role names.** Roles rank by op kind, so on attention three global loads
  that all feed a `local_alloc` are indistinguishable by rank and only one of them gets called
  `GLOBAL_LOAD`; `FROM_SMEM` can be the layout going *into* shared. Each provenance line therefore ends
  with the source variable and line taken from the compiler's own location info, and the emitted file
  carries a `# DOTS` block naming every `tt.dot`'s operands:

```
ttg.local_load result[0]  shape=[64, 128] f16 (reg)  <- q @ fwd_decode.py:507
tt.dot operand[1]         shape=[128, 64] f16 (reg)  <- kT @ fwd_decode.py:652
#   dot #2: A=dv[128, 16]  B=do[16, 128]  -> dv[128, 128]
```

Three more of its outputs are worth acting on directly:

- **`COMPILED FORM of the N buffer_load site(s)`** — transcribe the **dump**, not the source, and
  transcribe the *bucket*: each site is reported as bare / mask-only / mask+`other`, **detected rather
  than inferred**. Both directions of getting this wrong cost the same thing. `tl.load(..., other=0.0)`
  frequently compiles to a load carrying **neither** operand (buffer OOB returns zero on CDNA), so
  passing `other=` in Gluon adds a `v_cndmask` per register that plain never paid; and adding a *mask* to
  a site whose compiled form is bare costs the same. Two operands are not reachable from
  `gl.amd.cdna4.buffer_load` (gfx942: `gl.amd.cdna3.buffer_load`) at all: `contiguity`, and `stride` — the latter appears only on the
  pipeliner's peeled prologue loads, which a non-pipelined anchor does not have and the injection puts
  back itself.
- **`LDS: N allocation(s)`** — compare against the anchor's, because layout equivalence is blind to
  allocation size by construction, in this tool and in `recover_gluon.py`. A shared total that crosses
  the LDS/CU divisor halves workgroups per CU while every layout still verifies. The divisor is
  **arch-specific — 64 KiB on CDNA3/gfx942, 160 KiB on CDNA4/gfx950** — so `recover` derives it from the
  `--arch` you passed and **declines to name one** for an arch this skill has no figure for, rather than
  applying gfx942's number to a generation with 2.5× the budget.
- **`constants-digest`** (with `--out`) — a digest over the sorted constructor expressions with role
  names dropped. Use it, not a file hash, to compare two recoveries of the same body: the emitted header
  carries the dump path *and* the recovering Triton's version, and role names legitimately drift with the
  dump — the blocked global-load layout is `A_LOAD`/`B_LOAD` on a `num_stages=2` dump and `FROM_SMEM` on
  the `ns=1` dump of the same body, values identical. Two correct recoveries therefore do not compare
  byte-for-byte, and hand-rolled normalisations are not comparable to each other either.
- **the unsupported-op table** — `recover` audits *layouts*, not *ops*, so 100% layout recovery does not
  mean transcribable. `amdg.in_thread_transpose` has no Gluon builtin and used to appear as a
  *successful* row; it is now named.

`verify` has four states and three exit codes. `RECONCILED` (exit 0) means there are differences but
**every one** has a named structural cause — a disclosed substitution at a shape where plain's layout
was `UNRECOVERABLE`; a `MISSING` layout the anchor has at another shape (pipelined plain vs single-dot
anchor); or a `MISSING` layout produced by an op Gluon cannot express, where no correct transcription of
that body could ever produce the row. `FAIL` (exit 1) is reserved for a difference with no such cause.
In none of these cases can `verify` tell you the substitution was **free** — only the ISA can.

> **The multi-kernel hazard.** `dump_ir.sh` takes the freshest artifact in the cache, so an op that
> compiles two kernels — an attention body plus a split-K reduce, or an MLA op whose reduce kernel
> compiles *last* — hands you whichever finished last, and every layout recovered from it is confidently
> wrong for the body you meant. Silently: the dump looks fine. It now warns and lists the candidates
> when more than one exists; pass `--kernel-name <substring>` to pin it.

## Recover

| command | gives you |
| --- | --- |
| **`parity_gate.py --champion-ms C --anchor-ms A --champion-asm F --anchor-asm F [--champion-ttgir F --anchor-ttgir F] [--champion-lds N --anchor-lds N] [--threshold 0.95] --arch gfx950 [--json]`** | **G3 — performance parity vs the champion, mode A only.** Exits **2** while `champion_ms/anchor_ms` is under the run's declared threshold: the round is `recovery` against the suspect it closed, never a win, and climbing is not permitted. Attributes the gap from the artifacts across `lost_pipeline` (champion TTGIR has `memdesc_index` / `local_store` / `num_stages>1`; `iter_args >= 2` is **not** evidence), `lost_layout` (load-width or LDS-op histogram narrowed, or `shared`/WG grew) and `lost_RA` (same multiset, an address rematerialized right above the `ds_read` that consumes it). Pass `--*-lds` from the Triton cache metadata's `shared` (the asm's `LDSByteSize` is a structural 0). An anchor faster than the champion clears and is told to attribute it. **Mode B: do not run it as a gate** (equal sides → vacuous CLEARED); as a diagnostic, say so. Its "parity" is performance parity — GEAK's numerical parity is `harness_lib`'s correctness |
| `pipeline_survey.py <root> [...]` | inventories a plain source tree by which pipeline form each kernel can exercise (A cross-iteration pipeline, B block ping-pong, C async/direct-to-LDS) — a screen for what to measure, not a verdict; only a dump settles it. Roots from arguments or `TILE_PIPELINE_SURVEY_ROOTS`; a missing root is refused |

### Last resort: re-injecting plain's pipeliner

**Lowest priority.** The pipeline is written by hand first (`references/tile-programming/pipeline.md`).
Re-injection is for **sizing** a `lost_pipeline` debt below the parity gate, or for repaying it when the
hand-written ring cannot reach parity. Its numbers are labelled `injected`, never count as a win, need **one
process per armed/unarmed variant**, and it is never applied to an incumbent (already-Gluon) kernel. No upstream
`gluon_to_ttgir` calls `add_schedule_loops` / `add_pipeline` (3.6.0 – 3.8.0) while both ship in `libtriton`,
so this is a reachability problem, not a rebuild. Full recipe: `references/method/recover.md ## Last resort:
re-injecting plain's pipeliner`.

| tool | what it does |
| --- | --- |
| **`gluon_swp.py`** | Wraps `HIPBackend.gluon_to_ttgir` in-process and runs the two passes as a second pass manager over the module the stock function returns, so **nothing on disk changes**: a read-only or shared site-packages, a later `pip install --force-reinstall`, and a crash mid-experiment all stop being hazards, and the effect ends with the process. Produces **byte-identical TTGIR to the on-disk splice on all four versions**, armed and unarmed, so nothing is given up for that. `capabilities()` (also the bare CLI) reports what this build has, **probed rather than inferred from the version**, and inspects the *original* rather than whatever is currently installed — without which a second `enable()` at a new depth is refused as "this tree already splices the passes", i.e. every depth sweep breaks. `enable()` refuses that genuine fork case, and refuses `num_stages < 2` where the pipeliner is a no-op. `cache_tag()` keys a cache dir on the arming (see the two-sided cache trap below). `buffer_ops=True` restores plain's ORDER — pipeline first, buffer conversion after — which is what lets an anchor be written with `gl.load` and still end on buffer ops. |
| `patch_reinject.py apply\|revert\|status` | The on-disk form, kept for when you want the pass list itself visible in `compiler.py` while reading. Env-armed (`TRITON_GLUON_SWP=N`, plus `TRITON_GLUON_SWP_BUF=1` for the buffer half) so splice-ON and splice-OFF are the **same binary**, which is the only way an IR diff between them means anything. The splice point is version-dependent and measured, not assumed: before `add_warp_pipeline` on 3.7/3.8; after the last `add_*` call on 3.6, which has no warp pipeline at all. Writes a `.orig_swp` backup; `revert` restores it and clears the `__pycache__`. |
| `pipeline_survey.py <root> [...]` | Inventories a plain-Triton source tree by which pipeline **form** each kernel can exercise: A = cross-iteration software pipeline (the one re-injectable here), B = block ping-pong, C = async copy / direct-to-LDS. Classification is from source text, which is a **screen and not a verdict** — a source saying `num_stages=2` can dispatch a branch compiled at 1, and only a dump settles it. Use it to rank what to measure. |
| `patch_async_reinject.py apply\|revert\|status` | Splices `add_coalesce_async_copy` for the case where an async pattern is **not** on a native per-lane width, since the pass is what makes such an access legal by adding a bounce. It is **not** what enables async copy — on a native width both entry points lower from the stock pass list — so reach for it only after `pipeline_examples_cdna4.py` has shown the width is the problem, and price the bounce it adds — prefer fixing the layout to match a native width. Same shape as `patch_reinject.py`: env-armed (`TRITON_GLUON_ASYNC=1`, so unarmed is byte-identical to stock) with a `.orig_async` backup that `revert` restores. |

**Two conditions the anchor must meet, or injection changes nothing at all**: the loop must be a
pipelining candidate (a loop **containing a dot** is one on a bare `range`; a **dot-free** loop needs
`tl.range(..., num_stages=N)`, where `None` inherits the launch value), and the loads must still be
`tt.load` when the pipeliner runs, i.e. `gl.load` rather than `gl.amd.cdna4.buffer_load` (gfx942: `cdna3`). On a dot kernel
the hand-written LDS staging has to come out as well — and **un-staging without arming the injection is a
regression, not a neutral intermediate**, so the two halves go together. The 2×2 and the per-shape
behaviour are in `references/method/recover.md`.

**Read the landing tell that matches your shape.** `ttg.memdesc_index` is the multi-buffered-*LDS*
signature, so it is the right tell **only where the loop has a dot**; a dot-free loop prefetches into
registers, never touches LDS, and reads `memdesc_index == 0` on an arm that demonstrably pipelined.
There, read `iter_args` 0→1, the load count scaling with depth, `tt.num_stages` on the `scf.for`, and a
visible peeled prologue.

> **The cache trap has two sides and each gives a false negative on its own.** *In process*, Triton's JIT
> cache is keyed on `(function, signature, constexprs)` and knows nothing about the arming, so two arms
> differing only by the wrapper hit the same artifact; `TRITON_ALWAYS_COMPILE=1` does not fix it. *On
> disk*, a per-arm `TRITON_CACHE_DIR` does not encode the **depth**, so an `ns=3` probe pointed at the
> `ns=2` directory is served the `ns=2` binary and reads as "depth does nothing". Give each arm its own
> kernel object, key the directory with `gluon_swp.cache_tag()`, and verify the tell in each arm's own
> `.ttgir`.

> **Availability is what COMPILES, not what imports.** Three claims in this package were wrong until
> they were compiled rather than imported: `cdna4.async_copy` lowers on gfx950 at 32/64/128 bits per lane
> but on gfx942 **only at 32 bits** with a clean tiling and `order=[1, 0]` (a wider or repeated access fails
> with wording that reads like a missing op); `sched_barrier` / `sched_group_barrier` / `set_prio` are absent
> from `gl.amd.cdna3` and `.cdna4` on all four versions (aiter's `pa_decode_gluon` imports them in a
> `try/except` and runs **no-op stubs**); and `gl.warp_specialize` is in core `gl` everywhere and still
> aborts the pass manager on CDNA. `gl.amd.warp_pipeline_stage` works on both (at `num_warps >= 8`) — and it
> is a scheduling hint, not data movement. `pipeline_examples_cdna4.py` (gfx950) and
> `pipeline_examples_cdna3.py` (gfx942) re-check this on your box.

> **The Gluon source surface drifts across the versions this package spans, and it will bite the examples
> before it bites your kernel.** Two renames matter: `gl.thread_barrier` became `gl.barrier`, and
> `gl.zeros(..., layout=)` is **unusable on 3.6.0** — it is a `GluonJITFunction`, so the layout has to
> survive `_flatten_ir` and no layout class implements that; use the `gl.full` builtin instead.
> Consequence: **`pipeline_examples_cdna3.py` does not run on 3.6.0 at all**, failing on both of those
> before it reaches any arch question, so a "CDNA3 examples all fail" report on a 3.6 box is a version
> result rather than an arch one. `pipeline_examples_cdna4.py` is written against 3.6 and runs there.
> Version-gated additions to be aware of when reading a lever as absent: `gl.amd.warp_pipeline_stage` /
> `warp_pipeline` arrive in 3.7, as do `compute_efficient_padded_shared_layout` and `scaled_upcast`.

> **`buffer_ops=True` is opt-in because it fails three different ways.** On an anchor whose **loads** are
> already `buffer_load` it aborts loudly (`PassManager::run failed`). On one whose **stores** are buffer
> ops it does not raise at all — `LLVM ERROR: Fatal pipeliner error` kills the interpreter. Arm it only
> on an anchor written throughout with `gl.load` / `gl.store`.
>
> **"Throughout" means the whole function, not the loop** — that is the third way, and it looks like
> neither of the first two. A single `gl.amd.cdna4.buffer_load` left *outside* the loop (a prologue tile,
> an epilogue bias, a scalar guard) is enough: the rejection comes from
> `TritonAMDGPUCanonicalizePointers`, which runs over the function rather than the pipelined region, so a
> loop body that is clean on its own still fails. Grep the anchor source for `buffer_` before arming,
> rather than reading the loop.

> **`recover`'s `LDS:` line is in ELEMENTS, and it sums *declared* allocations without modelling
> liveness.** Two separate reasons not to compare it directly against a byte figure: the unit differs from
> `probe.py`'s `lds/WG` (multiply by the element size), and it over-reports on a non-pipelined dump where
> the backend allocator would have reused one buffer across disjoint live ranges. Read it as an **upper
> bound** in elements and as a comparator against plain's own line — not as the `shared` bytes/WG the
> kernel will be charged. `probe.py measure` reads the compiled artifact and is the figure to quote once an
> anchor exists.

> **If you are on a copy of `gluon_swp.py` older than this one, check `disable()` first.** It used to
> capture `gluon_to_ttgir` as a *resolved* attribute, which loses the `staticmethod` descriptor, so
> restoring it left an instance method and every subsequent Gluon compile in that process died with
> `gluon_to_ttgir() takes 3 positional arguments but 4 were given`. That broke every in-process A/B of
> differently-armed builds — which this skill no longer runs: one process per patched variant. The version here captures from
> `__dict__` and its selftest asserts the descriptor **kind** survives a round trip, not just the
> resolved function.

> **`TRITON_GLUON_SWP_PIPELINE` is not the knob**, and neither are `TRITON_GLUON_COOP_LDS` /
> `TRITON_GLUON_PINGPONG`. All three are additions to a **vendor fork's** `GetEnv.h`; no upstream version
> reads any of them. Measured on clean 3.7.1 and 3.8.0 they are *tolerated and inert* — so is a knob
> invented on the spot — which is the worst of the three possible outcomes: nothing errors, nothing
> changes, and the null result reads as "this technique does not work here".

## Evidence

**Profiler entry is GEAK's**, under its GPU lock — never pin `HIP_VISIBLE_DEVICES` inline:

```bash
bash kernel_workflow/scripts/profile_kernel.sh <gpu_id> "<harness cmd> --profile" <out_dir> \
     [--pmc | --derived] [--att] [--spi] [--kernel <regex>]
```

The default run is unchanged (arch-aware profiler order from `profile_policy.sh`, raw output in
`profile_report.txt` ending with `Profiler used:`). The optional layers write `pmc/`, `att/`, `spi/` beside it,
append a section each, record their state in `profile_layers.json`, and **degrade** (never fail the run) when
the tool, arch or ATT decoder is missing. Interpretation: `kernel_workflow/knowledge/profiling_guide.md` and
`references/method/profile.md`.

| `$KT` tool | gives you |
| --- | --- |
| `capture.sh` | one command → the anchor's evidence bundle (kernel trace, PMC, ATT, static asm audit); runs under `gpu_lock.sh`, moves old outputs aside |
| `rocprofv3_safe.sh --dev <gpu> --kernel <re> [--att] -- <cmd>` | rocprofv3 with a timeout, a mandatory kernel filter and the HIP/ROCR guard (touches `HIP_VISIBLE_DEVICES` only when ROCR is also set); `--no-timeout` for e2e serving, whose finalize is legitimately slow; exits 3 = degraded |
| `rocprof_compute_probe.sh --dev <gpu> …` | rocprof-compute SOL (memory block, `warp_state`, coalescing) into `rc_metrics.json` via `parse_rc.py`. `--roof-only` on gfx95x needs rocprof-compute ≥ 3.6.0 |
| `parse_pmc.py` | PMC groups and their parse: busy counters (`VALUBusy`, `MfmaUtil` — not duty-cycle `VALUUtilization`), C1 achieved DRAM bandwidth by independent routes (as a range). gfx950 `FETCH_SIZE`/`TCC_BUBBLE` under-count reads — compare routes |
| `parse_rc.py`, `kernel_breakdown.py`, `hotspot_analyzer.py` | rocprof-compute parse; per-kernel time/bound breakdown; hot-PC ranking from ATT |
| `att_to_perfetto.py`, `att_merge_perfetto.py`, `att_timeline.py`, `att_opclass.py`, `tile_trace.py`, `serve_traces.py` | ATT warp-trace views and op-class rollups; decoder via `ROCPROF_ATT_LIBRARY_PATH` (not vendored) |
| `asm_loop_audit.py [--opcodes] [--meta <ir_dir>] [--kernel <substr>]` | static hot-loop audit (A1, A3, B3): `next_free_vgpr`/AGPR/spill, LDS bytes/WG from the cache metadata (not `group_segment_fixed_size`, a structural 0), per-opcode histogram + diff |
| `mfma_efficiency.py`, `deep_mfma_analysis.py`, `asm_schedule_viz.py` | B1/B2: ranked inter-MFMA bubble ownership, LDS-vs-global feed split, schedule views; routes buckets to lever cards |
| `layout_facts.py`, `gfx950_isa.py`, `hw_sources.sh` | MFMA operand-layout facts (B4); the offline gfx950 ISA (other targets `--arch cdna3\|rdna3\|rdna4`); fetching primary sources |

## Measure

**Acceptance is GEAK's** — `harness_lib`: CUDA-event device time with a sync per sample, a read-evict cache
flush before every sample, the median, a fresh process per leg (`measure_legs`), against the same-window
baseline and the `MIN_IMPROVE` commit gate.

| command | gives you |
| --- | --- |
| **`$KT/ab_bench.py --module <adapter>.py [--permute] [--control auto] [--cache cold\|hot] [--min-improve-pct 2] [--json F]`** | **screening — a pre-filter in the deep_engineer's loop before GEAK verify).** Same-window interleaved A/B on `harness_lib.time_op` — read-evict **cold by default** (`hot` is search-only), **median** headline with min as an extra field, oracle before timing. A screen pass needs a delta beyond the measured noise band **and** the 2% gate; a measured band can only make the verdict stricter. Refuses: a non-finite metric (`NaN > tol` is False — add an `outputs(name)` hook and it scans tensors itself); duplicate `fingerprint(name)` values (two variants differing only by a `gl.constexpr` share a cache entry); arms reporting different `toolchain(name)` identities (one process per patched variant). A flat set across 3+ arms is a **collision suspect**; `--permute` reverses the order and reports whether each number followed the code or the position |
| `$KT/create_harness.py --kernel-name <k> --output <harness.py> [--kernel-type triton\|gluon]` | a harness on `harness_lib`: `h.time_op` timing, `h.check_correct_multi` correctness, one `GEAK_RESULT_LATENCY_MS=<ms> case_id=<id>` line per case (+ `GEAK_RESULT_GEOMEAN_MS`), rc 2 `HARNESS_UNFILLED` until filled |
| `$KT/parse_correctness.py [--parse geak --exit-code N] [--tol T]` | correctness verdicts from a log: GEAK unittest exit codes 0/1/2/3, `UT_HARNESS_INCOMPLETE`, per-case `max_rel_err` JSON; `--tol` can only make a pass stricter |
| `gpu_lock.sh <gpu_id\|pool> <cmd>` | shim → `kernel_workflow/scripts/gpu_lock.sh` (one lock namespace for every tree; `--help`, `--selftest`). The broker in `../scheduler/` is opt-in: `GEAK_GPU_BROKER=1` and a live socket |
| `locus.sh`, `runtime_env.sh` | only when the comparator was measured in a container: record/verify the execution locus (host vs container) the comparator was measured at; per-run runtime dirs under `$WORK/.tile-runtime/` |

## Climb

| command | gives you |
| --- | --- |
| `lever_index.py --bound <class> --arch gfx950` | the AMD lever cards (`references/hardware/lever-cards.json`) expressible in Gluon for the bound your profile named — chip-absent levers excluded, gating-law-forbidden flagged. Experience, not verdict; a ranked bucket with no card is a catalogue gap |
| `pipeline_examples_cdna4.py` | **the gfx950 reference** for authored overlap: sync-staging control plus the async forms (and a warp-pipeline case), numerics-checked; each case reports per-lane access width, whether it compiled, `ds_write` count and a numeric verdict, so an unsupported width separates from a layout that does not cover the tile. Needs a GPU |
| `pipeline_examples_cdna3.py` | **the gfx942 downgrade** set: sync staging and the `warp_pipeline_stage` hint (async only at 32 bits with `order=[1, 0]`). Needs a GPU; does not run on Triton 3.6.0 |

## Close and records

`canonical_record.py` (the record, outcome enum, `request-structure-suspect`), `round_record.py` (per-round,
refuses `kept` inside the noise band), `recordctl.py`, `close_audit.py` (finds the close's unbacked claims),
`report_lint.py`, `served_envelope.py` (weigh a gap by the served shape mix), `run_state.py`, `debt.py`,
`profile_payload.py`, `wait_for.sh --launch | --await` (record a long run's outcome rather than sleeping).
Schemas: `../runtime/*.schema.json`. Method: `references/method/close.md`, `references/method/records.md`.

## Selftests — offline, no GPU

```bash
bash "$SKILL/scripts/selftest_all.sh"          # everything: pack scripts, $KT tools, core runtime, scheduler, refs
bash "$SKILL/scripts/smoke_test_recover.sh"    # the port toolchain only
python3 "$SKILL/scripts/check_pack_refs.py"    # every link / heading / runtime reference resolves
```

`ttgir_bridge.py --selftest` has two layers. The pure layers (type splitting, role ranking, operand
attribution, the rank guard, the config precheck) run with no `triton` at all and report the live layer
as skipped. Where `triton` *is* importable it additionally parses a synthetic TTGIR and asserts that
every layout round-trips EXACT, so a regression in upstream's converter is caught here rather than on a
kernel. Two of its guards are not defensive style: handing a memdesc to `get_gluon_layout_from_tensor`
**segfaults**, and calling `to_linear_layout` at a mismatched rank trips an LLVM assert. Both are
process death with no traceback, so both are checked before the call rather than caught after.

## Triton-version notes

Checked against upstream `triton-lang/triton` at `v3.6.0`, `v3.7.1`, `release/3.8.x` and `main`. The
transcription path (`--emit-gluon layouts` → `ttgir_to_gluon.py` → `--verify`) works on all of them: every
TTGIR attribute the parser reads and every Gluon constructor it emits is spelled identically across the
four.

**Recovery is version-invariant; performance is not.** `ttgir_bridge.py` was run over 16 kernels
(8 aiter, 8 from a separate tuned-Triton set) × clean upstream `3.6.0` / `3.7.0` / `3.7.1` / `3.8.0` in
per-version containers: identical recovered counts and byte-identical layout constants on all four,
32/32. That is the cross-check saying a recovered constant is the compiler's rather than one build's, so
a layout preamble may be carried across versions. Timings may **not** — the same 8 anchors measured on
`3.8.0` moved in *both* directions against `3.7.1`, and the direction differs per kernel: one attention
forward's plain regressed 1.83× while its Gluon anchor lost only 11% (so the anchor's ratio jumped from
1.005 to 1.655 without the anchor improving at all), while a GEMM's anchor regressed 1.20× as its
`plain@ns=1` improved 1.14×, collapsing a 1.36× win to parity. **Re-measure after a Triton bump; never
carry a ratio across one.**

One more portability note worth having before you write LDS staging: `gluon_to_ttgir` runs no membar
pass, which invites the conclusion that a hand-authored LDS loop needs explicit `gl.barrier()`. Membar
insertion happens *lower*, inside the shared `TritonGPUToLLVM` conversion, so it applies to Gluon too —
stripping all four `gl.barrier()` calls out of a working anchor left it numerically correct and still
emitted 6 `s_barrier` (vs 7 with them). Use `gl.barrier()` only to **suppress or reposition**; one
anchor paid a redundant barrier for the opposite belief.

Two things are not portable — plus one failure below that reads like a version problem and is not:

- **`--with-skeleton` (and `--emit-gluon anchor|pipeline`) needs the modern translator — 3.8+ upstream.**
  It imports `translate_paths` and `TranslatorTarget` from `triton.tools.triton_to_gluon_translator`; at
  `v3.6.0`, `v3.7.0` and `v3.7.1` the package is spelled `triton_to_gluon_translater` and exposes only
  `convert_triton_to_gluon(src)` with no `target` argument. The import failure is caught and the run
  degrades to layouts-only with a note on stderr, so the anchor is still produced — just without the
  algorithm skeleton. **Decide this from the import, never from `triton.__version__`:** the rewrite
  landed on `main` and never on the `release/3.7.x` line, so a main or vendor-fork checkout can still
  report `3.7.0` and carry the 3.8-era `translator` package with its `target.py`. The import is what the
  script actually tests; the version string is not evidence either way. In practice, on an **official**
  pip-installed build the translator package is simply not there — it has been found only in the
  gfx950 tutorial *fork* — so plan for the loop being **recover, then hand-author the anchor**, and treat
  any claim about the automatic translator as a fork claim until the import succeeds on your box.
- **`dump_ir.sh --knobs LLIR_SCHED|AMDGCN_AS|RA_HINTS` is fork-only.** `TRITON_ENABLE_LLIR_SCHED`,
  `TRITON_ENABLE_AMDGCN_AS` and `TRITON_ENABLE_AMDGPU_RA_HINTS` appear in no upstream version, and neither
  does `triton.tools.amdgcnas` (which `probe_levers.py`'s `gemm_compiler_stack` calls "decoupled / stock
  Triton"). On a stock build these export env vars nobody reads: a silent no-op, not an error. Do not
  attribute a delta to them without `probe_levers.py --all` first.
- **A wrapped kernel translates to nothing, and it used to look like a missing translator.** Upstream
  resolves `module:object` with a bare `getattr`, so under `@triton.heuristics` / `@triton.autotune` the
  AST rewriter receives the wrapper instead of the kernel and returns an *empty* translation without
  raising. `recover_gluon.py` peels to the `JITFunction` first — byte-identical to the stock path when
  there is nothing to peel — and names the wrapper on stderr. An empty result is now reported as such
  rather than as "translator unavailable", because the two call for opposite responses: point `--kernel`
  at the kernel, versus your Triton is too old.

## Notes on vendored output

- `recover_gluon.py --record` / `--verify` prints `perf_delta_vs_plain: <fill> # regression expected, NOT a
  reject`. That is the transcription stage's expectation; the run's parity criterion is declared and enforced
  by `parity_gate.py` (`references/method/recover.md`), and a port that clears it from above is normal.
