# scripts — usage

Seven tools for one job: **transcribing a tuned plain-Triton kernel into Gluon, and re-injecting the
pipeline afterwards**. Run from the GEAK repo root with
`SKILL=perf_knowledge/expert_skills/skills/gluon_authoring`.

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
- **`UNRECOVERABLE: N`** must be 0. The one seen in practice is `amd_rotating_shared`, a genuine
  `gluon.language` gap and not a tool gap: `AMDRotatingSharedEncodingAttr` has the C++ traits for a
  `toLinearLayout`, but Python can only obtain a layout through a `builder.get_*_layout` constructor and
  `get_gluon_layout_from_memdesc` is precisely the one that throws. It is **not** a `num_stages`
  artefact — one kernel still shows it at `num_stages=1` — so re-dump at `ns=1` and re-recover before
  concluding the body is untranscribable.
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

- **`COMPILED FORM of the N buffer_load site(s)`** — transcribe the **dump**, not the source.
  `tl.load(..., other=0.0)` compiles to a buffer_load with a mask and *no* `other` operand (buffer OOB
  returns zero on CDNA), but passing `other=` in Gluon emits it and costs a `v_cndmask` per register.
  Measured at 1.2–2% on two kernels. `contiguity` is not reachable from `gl.amd.cdna3.buffer_load` at all.
- **`LDS: N allocation(s)`** — compare against the anchor's. One kernel's entire 1.69× residual was its
  shared total crossing the 64 KiB/CU divisor (2 workgroups per CU became 1) while every layout verified.
  Layout equivalence is blind to allocation size by construction, in this tool and in `recover_gluon.py`.
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

| tool | what it does |
| --- | --- |
| `probe_levers.py` | Per-build capability probe. Run it as `--all [--arch <gfx>]` — there is **no** positional probe-name argument — and read the `reinject_ttgir_pipeliner` entry: it answers whether plain's `add_schedule_loops` / `add_pipeline` are present in *this* `libtriton.so` before you edit `compiler.py`. Read `available: true` for exactly that and nothing more — it says the symbols exist, not that the pass will transform *your* IR. Those are two different hypotheses and they come apart in practice; `skill.md` step 2 has the read-the-IR-back check that answers the second. That is the probe this skill uses; the other five it exposes belong to lever cards that are not part of this package. |

All four `--selftest` entry points run with no GPU and no ROCm:

```bash
python3 "$SKILL/scripts/ttgir_bridge.py"   --selftest   # recovery + equivalence (see note)
python3 "$SKILL/scripts/ttgir_to_gluon.py" --selftest   # parser / emitter
python3 "$SKILL/scripts/recover_gluon.py"  --selftest   # layout equivalence
python3 "$SKILL/scripts/probe_levers.py"   --selftest   # probe plumbing
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

The scripts and `references/` are an upstream snapshot with one correction (see `skill.md ## Sources`),
so three things they print or teach disagree with this skill — this skill wins — and a fourth points at
files the package does not carry:

- `recover_gluon.py --record` / `--verify` prints `perf_delta_vs_plain: <fill> # regression expected,
  NOT a reject`. That is upstream's transcribe-only step, where the pipeline layer came later. **Here the
  port is transcribe *plus* re-injection and closes at ≥95% of tuned plain**, as the exit condition of
  step 2 rather than on any round boundary. Take the gate from `skill.md ## Procedure` step 3, not from
  this line — and note that "regression expected" is upstream's expectation for half the procedure, so do
  not carry it over as this skill's: a port that clears the bar rather than approaching it from below is
  a normal outcome here, and step 4 is written on the assumption that the track continues past it.
- `references/tile-programming/pipeline.md` tells you to run
  `scripts/probe_levers.py reinject_ttgir_pipeliner`. The CLI takes no positional probe name and exits 2;
  use `--all` as above. Its **vetted double-buffer skeleton also indexes with `s.index(i % 2)`**, which
  is the runtime buffer index `skill.md ## Knobs & pitfalls` bans outright — unroll by 2 so each index is
  a literal.
- Several docstrings and `cmd` fields cite files this package does not carry (`compiler-contract.md`,
  `transcribe.md`, `experiment-records.md`, `lever-cards.json`, `opt_swp_test.py`, `bench.py`,
  `prof_driver.py`). Dead by design — do not go looking for them.
