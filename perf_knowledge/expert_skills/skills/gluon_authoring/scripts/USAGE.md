# scripts — usage

Six tools for one job: **transcribing a tuned plain-Triton kernel into Gluon, and re-injecting the
pipeline afterwards**. Run from the GEAK repo root with
`SKILL=perf_knowledge/expert_skills/skills/gluon_authoring`.

## Transcription

| tool | what it does |
| --- | --- |
| `dump_ir.sh` | Runs a compile command with IR dumping on and collects `.ttir` / `.ttgir` / `.amdgcn` per variant. `--emit-gluon layouts\|anchor\|pipeline` additionally emits a Gluon skeleton from the dumped TTGIR. Grammar: `bash dump_ir.sh <compile_cmd ...> --variant <name> --out <ir_dir> [--knobs ...] [--emit-gluon ...] [--kernel module.path:object] [--arch gfx950]` — every token that is not one of those flags belongs to `<compile_cmd>`, so it may be split around them. `--help` prints the same. |
| `recover_gluon.py` | The driver: dump → recover layouts → emit an anchor → **`--verify`** the recompiled anchor's layout attributes against plain. Always finish with `--verify`; it is the only gate that catches a layout recovered wrong (wrong `order`/`kWidth`) behind a passing numeric oracle. It compares the attribute **set**: an auto-pipelined plain kernel mentions each layout once per unrolled dot, so the counts legitimately differ from the anchor's — that skew is printed but does not fail. `--selftest` runs the equivalence checks offline. |
| `ttgir_to_gluon.py` | The parser/emitter underneath. Pure text, no GPU and no `triton` import needed. `--selftest` checks it offline against three built-in TTGIR samples; `--pipeline` also emits a pipeline skeleton. Covers `#blocked`, `#amd_mfma`, `#swizzled_shared`, `#padded_shared`, `#linear`, `ttg.dot_op`, `ttg.slice`. Does **not** place `convert_layout` (manual, see `references/tile-programming/layout-recipes.md`) and cannot name `amd_rotating_shared` (language-surface gap). |
| `smoke_test_recover.sh` | Offline end-to-end check of the recovery toolchain. |
| `smoke_recover_gpu.py` | On-GPU version; needs `torch` + `triton`. |

## Pipeline re-injection

| tool | what it does |
| --- | --- |
| `probe_levers.py` | Per-build capability probe. Run it as `--all [--arch <gfx>]` — there is **no** positional probe-name argument — and read the `reinject_ttgir_pipeliner` entry: it answers whether plain's `add_schedule_loops` / `add_pipeline` are present in *this* `libtriton.so` before you edit `compiler.py`. Read `available: true` for exactly that and nothing more — it says the symbols exist, not that the pass will transform *your* IR. Those are two different hypotheses and they come apart in practice; `skill.md` step 2 has the read-the-IR-back check that answers the second. That is the probe this skill uses; the other five it exposes belong to lever cards that are not part of this package. |

All three `--selftest` entry points run with no GPU and no ROCm:

```bash
python3 "$SKILL/scripts/ttgir_to_gluon.py" --selftest   # parser / emitter
python3 "$SKILL/scripts/recover_gluon.py"  --selftest   # layout equivalence
python3 "$SKILL/scripts/probe_levers.py"   --selftest   # probe plumbing
```

## Triton-version notes

Checked against upstream `triton-lang/triton` at `v3.6.0`, `v3.7.1`, `release/3.8.x` and `main`. The
transcription path (`--emit-gluon layouts` → `ttgir_to_gluon.py` → `--verify`) works on all of them: every
TTGIR attribute the parser reads and every Gluon constructor it emits is spelled identically across the
four. Two things are not portable — plus one failure below that reads like a version problem and is not:

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
  port is transcribe *plus* re-injection and closes at ≥95% of tuned plain** — due in round 1, or round 2
  when plain is auto-pipelined. Take the gate from `skill.md ## Procedure` step 3, not from this line.
- `references/tile-programming/pipeline.md` tells you to run
  `scripts/probe_levers.py reinject_ttgir_pipeliner`. The CLI takes no positional probe name and exits 2;
  use `--all` as above. Its **vetted double-buffer skeleton also indexes with `s.index(i % 2)`**, which
  is the runtime buffer index `skill.md ## Knobs & pitfalls` bans outright — unroll by 2 so each index is
  a literal.
- Several docstrings and `cmd` fields cite files this package does not carry (`compiler-contract.md`,
  `transcribe.md`, `experiment-records.md`, `lever-cards.json`, `opt_swp_test.py`, `bench.py`,
  `prof_driver.py`). Dead by design — do not go looking for them.
