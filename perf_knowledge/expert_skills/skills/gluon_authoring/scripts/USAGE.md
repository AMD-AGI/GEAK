# scripts — usage

Ten tools for two jobs: **transcribing a tuned plain-Triton kernel's layouts into Gluon**, and
**recovering the software pipeline that a faithful transcription drops** — where the champion had one
worth recovering, which is a question this package makes you measure rather than assume. Run from the
GEAK repo root with `SKILL=perf_knowledge/expert_skills/skills/gluon_authoring`.

## Transcription

Start with `ttgir_bridge.py`. The two older tools below it still work and are still the only option
where `import triton` is unavailable, but they carry a hand-written layout mapping and a text-level
equivalence check, and both of those have already cost this branch a patch.

| tool | what it does |
| --- | --- |
| **`ttgir_bridge.py recover\|verify\|view`** | **The recovery path — prefer this.** Hands the `.ttgir` to the COMPILER's own parser and then to upstream's `layoutToGluon()`, so there is no per-kind mapping in this package to fall behind Triton: an unsupported kind surfaces as a named `UNRECOVERABLE` row, never as a plausible wrong constructor. Every layout carries a **round-trip proof** (the recovered object is re-printed as MLIR and compared against the source text), plus a `num_warps`-vs-`warps_per_cta` cross-check. `verify` compares **LinearLayout normal forms**, so two spellings of one layout compare equal and unroll skew is reported as informational `MULTIPLICITY` rather than mistaken for a difference. Needs `import triton` (**≥ 3.7** for full capability; 3.6 runs with shared layouts compared as canonical text and no `view`). No GPU and no ROCm: an upstream wheel in a plain `python:3.10-slim` reports `backends installed: ['amd','nvidia']`, so `--arch gfx942` works anywhere. |
| `dump_ir.sh` | Runs a compile command with IR dumping on and collects `.ttir` / `.ttgir` / `.amdgcn` per variant. `--emit-gluon layouts\|anchor\|pipeline` additionally emits a Gluon skeleton from the dumped TTGIR. Grammar: `bash dump_ir.sh <compile_cmd ...> --variant <name> --out <ir_dir> [--knobs ...] [--emit-gluon ...] [--kernel module.path:object] [--kernel-name <substring>] [--arch gfx950]` — every token that is not one of those flags belongs to `<compile_cmd>`, so it may be split around them. `--help` prints the same. **`--kernel-name` pins WHICH kernel's artifacts are taken** and is not the same flag as `--kernel` (that one is `module.path:object`, for the translator) — see the multi-kernel hazard below. |
| `recover_gluon.py` | The older driver: dump → recover layouts → emit an anchor → `--verify`. Worth running for the **anchor assembly** and for the algorithm skeleton (`--with-skeleton`); prefer `ttgir_bridge.py verify` for the gate. Its `--verify` compares canonical attribute **text** as a set — sound in one direction only (equal text means equal layout, but two spellings of one layout read as different), and the set predicate is itself a patch over an earlier multiset version that could not pass on an auto-pipelined plain kernel. Normal forms do not need that patch. `--selftest` runs the equivalence checks offline. |
| `ttgir_to_gluon.py` | The pure-text parser/emitter underneath `recover_gluon.py`. No GPU and no `triton` import needed, which is the one reason to reach for it. Covers `#blocked`, `#amd_mfma`, `#swizzled_shared`, `#padded_shared`, `#linear`, `ttg.dot_op`, `ttg.slice`. Its output is a starting point, **not a proof**: a hand-written mapping silently emits a layout missing a field whenever upstream adds or renames one, which is exactly what `835a3c1` had to fix (`tilesPerWarp` / `elementBitWidth` were being dropped, and `--verify` caught it only as an unexplained text mismatch). Does not place `convert_layout` (manual, see `references/tile-programming/layout-recipes.md`) and cannot name `amd_rotating_shared`. |
| `smoke_test_recover.sh` | Offline end-to-end check of the recovery toolchain. |
| `smoke_recover_gpu.py` | On-GPU version; needs `torch` + `triton`. |

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
  `gl.amd.cdna3.buffer_load` at all: `contiguity`, and `stride` — the latter appears only on the
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

## Pipeline re-injection

No upstream `gluon_to_ttgir` calls `add_schedule_loops` / `add_pipeline` — checked on 3.6.0, 3.7.0,
3.7.1 and 3.8.0 — while both passes ship in `libtriton` on all four. Only the Python pass list omits
them, so this is a reachability problem, not a rebuild.

| tool | what it does |
| --- | --- |
| **`gluon_swp.py`** | **Prefer this.** Wraps `HIPBackend.gluon_to_ttgir` in-process and runs the two passes as a second pass manager over the module the stock function returns, so **nothing on disk changes**: a read-only or shared site-packages, a later `pip install --force-reinstall`, and a crash mid-experiment all stop being hazards, and the effect ends with the process. Produces **byte-identical TTGIR to the on-disk splice on all four versions**, armed and unarmed, so nothing is given up for that. `capabilities()` (also the bare CLI) reports what this build has, **probed rather than inferred from the version**, and inspects the *original* rather than whatever is currently installed — without which a second `enable()` at a new depth is refused as "this tree already splices the passes", i.e. every depth sweep breaks. `enable()` refuses that genuine fork case, and refuses `num_stages < 2` where the pipeliner is a no-op. `cache_tag()` keys a cache dir on the arming (see the two-sided cache trap below). `buffer_ops=True` restores plain's ORDER — pipeline first, buffer conversion after — which is what lets an anchor be written with `gl.load` and still end on buffer ops. |
| `patch_reinject.py apply\|revert\|status` | The on-disk form, kept for when you want the pass list itself visible in `compiler.py` while reading. Env-armed (`TRITON_GLUON_SWP=N`, plus `TRITON_GLUON_SWP_BUF=1` for the buffer half) so splice-ON and splice-OFF are the **same binary**, which is the only way an IR diff between them means anything. The splice point is version-dependent and measured, not assumed: before `add_warp_pipeline` on 3.7/3.8; after the last `add_*` call on 3.6, which has no warp pipeline at all. Writes a `.orig_swp` backup; `revert` restores it and clears the `__pycache__`. |
| `pipeline_survey.py <root> [...]` | Inventories a plain-Triton source tree by which pipeline **form** each kernel can exercise: A = cross-iteration software pipeline (the one re-injectable here), B = block ping-pong, C = async copy / direct-to-LDS. Classification is from source text, which is a **screen and not a verdict** — a source saying `num_stages=2` can dispatch a branch compiled at 1, and only a dump settles it. Use it to rank what to measure. |
| `pipeline_examples_cdna3.py` | The four authored-overlap examples from `references/gluon/pipeline-reference.md`, kept **runnable and numerics-checked** so the op counts quoted there can be re-derived on your own box instead of trusted. Needs a GPU. The async multi-buffer path is deliberately absent: `cdna4.async_copy` **imports on gfx942 and does not lower there**, so on CDNA3 the authored surface is sync staging plus the `warp_pipeline_stage` hint |
| `probe_levers.py --all [--arch <gfx>]` | Per-build capability probe — there is **no** positional probe-name argument. Its `reinject_ttgir_pipeliner` entry answers whether the passes are present in *this* `libtriton.so`. Read `available: true` for exactly that and nothing more: the symbols existing is not the pass biting on your IR, and those two hypotheses come apart in practice. `gluon_swp.capabilities()` answers the same question plus whether the tree already pipelines; `skill.md` step 2d has the read-the-IR-back check that answers the second. |

**Two conditions the anchor must meet, or injection changes nothing at all**: the loop must be a
pipelining candidate (a loop **containing a dot** is one on a bare `range`; a **dot-free** loop needs
`tl.range(..., num_stages=N)`, where `None` inherits the launch value), and the loads must still be
`tt.load` when the pipeliner runs, i.e. `gl.load` rather than `gl.amd.cdna3.buffer_load`. On a dot kernel
the hand-written LDS staging has to come out as well — and **un-staging without arming the injection is a
regression, not a neutral intermediate**, so the two halves go together. The 2×2 and the per-shape
behaviour are in `references/gluon/pipeline-reference.md`.

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
> they were compiled rather than imported: `cdna4.async_copy` imports on gfx942 but does not lower
> there; `sched_barrier` / `sched_group_barrier` / `set_prio` are absent from `gl.amd.cdna3` and
> `.cdna4` on all four versions (aiter's `pa_decode_gluon` imports them in a `try/except` and runs
> **no-op stubs**); and `gl.warp_specialize` is in core `gl` everywhere and still aborts the pass
> manager on CDNA3. `gl.amd.warp_pipeline_stage` is the one that *does* work on gfx942 — and it is a
> scheduling hint, not data movement. `pipeline_examples_cdna3.py` is what re-checks this on your box.

> **`buffer_ops=True` is opt-in because it fails two different ways.** On an anchor whose **loads** are
> already `buffer_load` it aborts loudly (`PassManager::run failed`). On one whose **stores** are buffer
> ops it does not raise at all — `LLVM ERROR: Fatal pipeliner error` kills the interpreter. Arm it only
> on an anchor written throughout with `gl.load` / `gl.store`.

> **If you are on a copy of `gluon_swp.py` older than this one, check `disable()` first.** It used to
> capture `gluon_to_ttgir` as a *resolved* attribute, which loses the `staticmethod` descriptor, so
> restoring it left an instance method and every subsequent Gluon compile in that process died with
> `gluon_to_ttgir() takes 3 positional arguments but 4 were given`. That breaks precisely the
> one-process interleaved-arms protocol this module exists to serve. The version here captures from
> `__dict__` and its selftest asserts the descriptor **kind** survives a round trip, not just the
> resolved function.

> **`TRITON_GLUON_SWP_PIPELINE` is not the knob**, and neither are `TRITON_GLUON_COOP_LDS` /
> `TRITON_GLUON_PINGPONG`. All three are additions to a **vendor fork's** `GetEnv.h`; no upstream version
> reads any of them. Measured on clean 3.7.1 and 3.8.0 they are *tolerated and inert* — so is a knob
> invented on the spot — which is the worst of the three possible outcomes: nothing errors, nothing
> changes, and the null result reads as "this technique does not work here".

All seven `--selftest` entry points run with no GPU and no ROCm:

```bash
python3 "$SKILL/scripts/ttgir_bridge.py"     --selftest   # recovery + equivalence (see note)
python3 "$SKILL/scripts/ttgir_to_gluon.py"   --selftest   # parser / emitter
python3 "$SKILL/scripts/recover_gluon.py"    --selftest   # layout equivalence
python3 "$SKILL/scripts/probe_levers.py"     --selftest   # probe plumbing
python3 "$SKILL/scripts/gluon_swp.py"        --selftest   # wrapper install/restore; skips with no backend
python3 "$SKILL/scripts/pipeline_survey.py"  --selftest   # the dot-candidacy rule, both directions
python3 "$SKILL/scripts/patch_reinject.py"   --selftest   # the version-dependent splice point
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
  script actually tests; the version string is not evidence either way.
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

## Where the vendored text contradicts this skill

The scripts and `references/` are an upstream snapshot with three corrections (see
`skill.md ## Sources`). One thing they teach disagrees with this skill — this skill wins — and a second
points at files the package does not carry:

- `recover_gluon.py --record` / `--verify` prints `perf_delta_vs_plain: <fill> # regression expected,
  NOT a reject`. That is upstream's transcribe-only step, where the pipeline layer came later. **Here the
  port is transcribe plus a conditional pipeline recovery and closes at ≥95% of tuned plain**, as the
  exit condition of step 2 rather than on any round boundary. Take the gate from
  `skill.md ## Procedure` step 3, not from this line — and note that "regression expected" is upstream's
  expectation for half the procedure, so do not carry it over as this skill's: a port that clears the bar
  rather than approaching it from below is a normal outcome here, and step 4 is written on the assumption
  that the track continues past it.
- Several docstrings and `cmd` fields cite files this package does not carry (`compiler-contract.md`,
  `transcribe.md`, `experiment-records.md`, `lever-cards.json`, `opt_swp_test.py`, `bench.py`,
  `prof_driver.py`, `asm_loop_audit.py`, `mfma_efficiency.py`, `champion_gate.py`). Dead by design — do
  not go looking for them.

**Two previously-flagged conflicts are resolved upstream and no longer apply.**
`references/tile-programming/pipeline.md` used to send you to
`scripts/probe_levers.py reinject_ttgir_pipeliner`, a form the CLI exits 2 on; it now names that form as
wrong and says `--all` is the whole CLI. Its vetted double-buffer skeleton still writes `cur = i % 2`,
but it now explains that this is the readable *correctness* template and that the literal-index unroll is
the scheduling-optimal one — which is what `skill.md ## Knobs & pitfalls` asks for, so the two agree.
