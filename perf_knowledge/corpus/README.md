# corpus/ — source evidence plus actionable development decisions

A FlyDSL author writing a GEMM has to answer, by hand, roughly twenty questions that Triton answers
for them: which MFMA instruction, what tile, how many waves, how LDS is laid out and swizzled, where
to pin instruction order. AITER already contains the same operator family answered six ways — in
FlyDSL, Triton, Gluon, CK, HIP and hand-written assembly. Source presence alone says nothing about
whether an answer was measured or won.
This directory turns them into two different products so the first move on a new kernel is a lookup
rather than a guess:

1. **Source evidence** — what an implementation literally contains, with a content-bound ID and
   reproducible `file:line`.
2. **Development decision cards** — when a pattern applies, what to try, why, alternatives, evidence
   strength and limits.

The cards also publish a small machine-readable `type` + `match` block.  `semantic` cards explain how
to express an invariant, `constraint` cards identify impossible/illegal candidates, and
`performance_candidate` cards add something worth measuring.  A static match is never a performance
ranking.

## The one rule

**Never make the reader infer a recommendation from a source-match count.**

Source evidence can establish: *this file, at this line, selects
`mfma_f32_16x16x32_bf16`.* It cannot establish: *that instruction is right for this shape.* The old
page stopped at the first sentence and called each regex category a "decision axis", leaving the
author to reverse-engineer the actual advice from a raw hit dump. That is an evidence index, not knowledge.

`gemm_decisions.md` supplies the missing semantic layer. Every card has conditions, actions,
alternatives, rationale, evidence strength and limitations:

- `source_observed` means "include this implementation precedent as a candidate", never "prefer it";
- `shipped_config` means "AITER selected this parameter seed", not "the corpus has its benchmark";
- measured guidance stays in learned cards/expert skills behind their existing feature switches.

Measured claims remain in learned cards or `perf_knowledge/expert_skills/`, attached to the environment
and run that earned them. The always-on corpus does not copy them and bypass those systems' switches.

## Layout

| path | what it is |
|---|---|
| `catalog.yaml` | discovery surface for operator/pattern families; its `patterns` cover every taxonomy GEMM operator id, so a TechLead's `kk_operator` resolves. |
| `benchmarks/<family>.yaml` | upstream UT/benchmark/tuner entrypoints plus the accounting a normalized runner must enforce; not timing evidence by itself. |
| `_extract_impl_facts.py` | reads an AITER checkout and writes raw source/tuning evidence. Needs `--aiter`. |
| `evidence/gemm_source.yaml` | machine-readable source observations: stable ID, question category, language, `file:line` (plus `end_line` where a Python statement spans lines), match, arch scope, verbatim excerpt. Besides kernel source it holds one record per row of AITER's per-arch ASM kernel inventories (`hsa/<gfx>/<kind>/*.csv`) and the routing statements of AITER's runtime dispatch (`tuned_gemm.py`, `ops/gemm_op_*.py`), each filed under the backend it routes to. |
| `evidence/gemm_tuned_configs.yaml` | AITER's shipped selected Triton configs, each with stable `cfg_…` attribution ID, grouped by `(gfx, variant, M bucket)` with exact source JSON paths. |
| `evidence/gemm_csv_tuned_configs.yaml` | AITER's tuned GEMM databases (`aiter/configs/**/*tuned*gemm*.csv`) without their timing columns: FlyDSL selections decoded into knobs and grouped by `(gfx, CUs, database, family, dtype, M bucket)` (`cfg_…`), and per-bucket counts of which backend AITER's tuner selected (`sel_…`). |
| `performance_axes/gemm.yaml` | backend-neutral performance questions plus loss-aware aliases for FlyDSL, Triton, CK and ASM knobs, and the `development_order` that places those questions in the order a kernel is written. |
| `_normalize_performance_decisions.py` | projects backend spelling onto those axes while retaining source fields, completeness and comparison limits. |
| `gemm_source_evidence.md` | generated evidence index for tracing a card back to source; not the first page an author should read. |
| `decisions/gemm.yaml` | curated development cards with axes, conditions, actions, alternatives, cross-backend solutions, background references and evidence strength, plus the `traits` a card may require. |
| `_render_decisions.py` | validates every citation (source, shipped selection, background doc, axis, trait) and the development order, and renders both pages below. |
| `gemm_decisions.md` | generated, actionable page the Workflow reads first: cards in development order (steps 1-10), then FlyDSL shipped seeds and the per-bucket backend-selection table. |
| `gemm_triton_seeds.md` | generated page with the Triton shipped seeds, split out because they are another backend's knob names. |
| `_select_candidates.py` | deterministically matches card conditions to an operator/language/gfx/dtype/regime context — or to every case of a task `meta.json` with `--workload`; it does not rank performance. |
| `_aggregate_decision_outcomes.py` | folds run-local validation manifests into per-decision and bundle outcome summaries; a single referenced card is planner attribution, not causal proof. |
| `_validate_products.py` | checks the corpus against implementations the workflow actually produced: per-case selection, whether the cards a run cited apply to it, which techniques the product wrote that no applicable card cites, and whether the run can feed the measured layer. |
| `benchmarks/ingest_workflow_products.py` | turns validated workflow products into implementation-registry records, one per case and Director replicate; `select_implementations.py --task-dir` queries every case of a task. |
| `test_corpus.py` | source reproducibility, card grounding and generated-page contracts. |

The extraction/rendering split matters: extraction needs a source tree that is not part of this repo,
while both renderers must not. That is what lets `--check` run in CI on a box with no aiter and no GPU, and a
generated doc that nothing checks is how `kernel_families.md` came to state a default tile that no
longer existed.

## How to refresh

```bash
python3 perf_knowledge/corpus/_extract_impl_facts.py --aiter /path/to/aiter --emit
python3 perf_knowledge/corpus/_render_facts.py --emit
python3 perf_knowledge/corpus/_render_decisions.py --emit
```

The extractor writes all three evidence files (`gemm_source.yaml`, `gemm_tuned_configs.yaml`,
`gemm_csv_tuned_configs.yaml`); `_render_decisions.py` writes `gemm_decisions.md` and
`gemm_triton_seeds.md`. Commit the evidence, cards and every generated page together. CI rejects a page
when it does not match its inputs, and rejects a decision card whose content-bound evidence ID is
absent or stale.

**Extraction refuses a dirty source.** If any file that produces evidence has uncommitted changes, the
emit aborts, because its `file:line` would resolve for you and for nobody else — and a citation index
whose citations do not resolve is not a weaker index, it is a different document that looks like one.
This is not hypothetical: the first version of this corpus shipped 56 records pointing into a locally
modified kernel, under a commit that did not contain those edits, with the whole suite green. The test
asserted a commit was *present*; it never asked whether the commit *identified* what was read.

The fix when your working copy is dirty is a throwaway clean checkout, which leaves your edits alone:

```bash
git -C /path/to/aiter worktree add /tmp/aiter-clean <commit>
python3 perf_knowledge/corpus/_extract_impl_facts.py --aiter /tmp/aiter-clean --emit
```

`--allow-dirty` extracts anyway and marks every affected record `unreproducible: true`, which is
useful for a local look and is rejected by `test_corpus.py` for the committed artifact. The refusal is
scoped to files that actually *contributed* evidence — an unrelated edit elsewhere in aiter cannot make a
citation unresolvable, and refusing on it would be superstition rather than a check.

Provenance records `aiter_origin` (the remote) rather than a local path, since the pair that
identifies a source is repository plus commit. A path like `/tmp/aiter-clean` names a directory that
no longer exists and never meant anything to anyone else.

## Conditioned lookup and measured feedback

Static applicability is machine-queryable without claiming a winner:

```bash
python3 perf_knowledge/corpus/_select_candidates.py \
  --operator-family scaled_quant_gemm --target-language flydsl \
  --gfx gfx942 --flydsl-version 0.3.0 \
  --dtype fp8_e4m3fnuz_blockscale --regime prefill \
  --m 32768 --n 4096 --k 1024 --trait lds_staging \
  --outcomes /path/to/decision_outcomes.json
```

A task that spans shapes — decode plus prefill, or a batch of buckets — is classified case by case,
because one static answer is too narrow for one case or too wide for another. `--workload` reads the
task's `meta.json` (its `cases[]` and `dtype`), `--shape MxNxK` adds cases, and the result lists each
case's eligible/deferred/rejected cards plus which cards hold for every case and which for some.
Both meta layouts in use are read: kernel tasks give every case `sig`, `m`, `n`, `k`; seams extracted
from a served model give each case `name`, `m` and `regime`, keep `N`/`K` once in `model_dims`, mark
correctness-only edges `timed: false` (classified, but not part of "every case"), and may state the
dtype as a whole contract (`bf16 in / bf16 weight / bf16 out, fp32 accumulate` reads as `bf16`). With
`--outcomes`, each case also lists the measured priors that apply to it:

```bash
python3 perf_knowledge/corpus/_select_candidates.py \
  --operator-family scaled_quant_gemm --target-language flydsl --gfx gfx942 \
  --workload meta.json --shape 256x4096x1024
```

Card dtypes are exact, canonical spellings. The selector canonicalises the spellings in use for one
format (`float8_e4m3fnuz`, `fp8_e4m3_fnuz` → `fp8_e4m3fnuz`) and then matches exactly; the scale
contract is part of the name (`fp8_e4m3fnuz_blockscale` is 128x128 block scale), so a per-token card
is not handed to a block-scale task. `regime=both` keeps decode- or prefill-only cards, since a
`both` workload contains cases of each.

A card whose match `requires` a trait — a property of the design, such as `lds_staging` — is
*deferred* until the planner passes `--trait`: the selector cannot see a design, so deferred means
"applies once your kernel does this". Traits are defined, with the sentence a planner reads, in
`decisions/gemm.yaml` and on the page; a trait passed that no card file defines is returned as
`undefined_traits` instead of silently deferring everything that needed its correct spelling.

Run-local measurements stay outside the source corpus. Aggregate their decision attribution with:

```bash
python3 perf_knowledge/corpus/_aggregate_decision_outcomes.py \
  --runs /path/to/exp --output /path/to/decision_outcomes.json
```

`speedup_vs_frozen_baseline` is the candidate's absolute score. A single-ref direction is still
planner attribution, not proof that the card caused the change; multi-ref directions remain bundles.
Only finite, correctness-passing results from reliable runs with a `ranking_basis` enter candidate
ordering. The basis is `accepted` when the Director accepted the run, or `graph_timed_lower_bound` when
its only flags are the timing-receipt ones (`timing_receipt_absent`, `timer_unprimed`) and every timed
case was a CUDA-graph replay: both legs of a replay carry the same submission bubble, which pulls the
ratio toward 1.0, so the measured speedup is a lower bound. Eager-timed cases carry each
implementation's own host dispatch and get no such basis. The flags are read from the manifest's
`validation.flag_reasons` / `timing_basis` / `graph_timed`, or, for manifests older than those fields,
from the run's `director_validation.json` and task `meta.json`. Regressions remain valid negative
evidence; failed/flagged measurements remain visible but do not supply a numeric prior. A
quality-passing multi-ref result may rank that exact bundle in `measured_bundles`, but it never
creates a prior for any card inside the bundle.

A run's direction scores are in the metric the lane scored candidates on, `measurement.primary_metric`
(time-weighted when the caller passed a workload or the frozen oracle is itself weighted). A measured
run matches a query only through its observed language; an older optimize run over a seam that
carries two backends recorded none, and `--assume-language RUN=LANGUAGE` supplies it, recorded as
`observed_language_basis: assumed_by_operator`. A run that measured several shapes matches a case
whose shape it contains.

## Validating the corpus against workflow products

The workflow's own products are the test the corpus has to pass: an implementation that won on a
real task, built by agents that read these cards. `_validate_products.py` takes run directories and
plain task directories (`meta.json` plus `kernel_src/`) and reports, per product:

- **selection** — every case classified, as a planner starts and with the design traits the product's
  source shows (LDS staging, a scheduled split-K hot loop), and the cards those traits unlock;
- **citations** — every card the run cited, where the selector places it for the product's own
  context, and what the citing directions reached; citing a card the selector rejects there, or an ID
  that does not exist, is a contradiction between planner and corpus;
- **coverage** — the categories the product's own code writes (the lines its final patch adds),
  scanned with the extractor's rules, against the evidence categories and axes of the applicable
  cards; a category no applicable card cites is a gap for this kind of kernel;
- **measured readiness** — the run's observed language, ranking basis, flags and metric, as the
  aggregator reads them.

```bash
python3 perf_knowledge/corpus/_validate_products.py --runs /path/to/exp \
  --task /path/to/merged_task --gfx gfx950 --json report.json --markdown report.md
```

It measures nothing and writes nothing into the corpus. A gap it finds is closed the usual way — with
a card that cites AITER source evidence — never by copying the product's numbers into this layer.
`gemm-streamed-operand-cache-policy` and `flydsl-lds-pipeline-depth` came from such a check: FlyDSL
GEMM products wrote a per-operand cache policy and an LDS ring depth that no applicable card spoke to.
`waves` and `tile_shape` can still be reported where a card states the same knobs as an option
tuple (`config_space`); the axis-level coverage shows whether the question itself is covered.

Validated products can also seed later runs as whole implementations, behind the same switch as
every other registry. `benchmarks/ingest_workflow_products.py` writes one normalized result per case
and Director replicate — suite = the task's `short_name`, baseline = a hash of its frozen
`baseline_src/`, the case's shape, the oracle's timer — and merges them; only cases with a ranking
basis are `measured`, the rest are kept as `not_rankable`. `select_implementations.py --task-dir`
queries every timed case of a task with the same derivation:

```bash
python3 perf_knowledge/corpus/benchmarks/ingest_workflow_products.py --runs /path/to/exp \
  --output measured_products.json --torch-version "$TORCH" --hip-version "$HIP"
python3 perf_knowledge/corpus/benchmarks/select_implementations.py --registry measured_products.json \
  --task-dir TASK_DIR --language flydsl --gfx gfx950 --flydsl-version 0.3.2 \
  --aiter-commit "$AITER" --torch-version "$TORCH" --hip-version "$HIP"
```

## Cards follow the development order, and carry other backends' answers

The questions a GEMM author answers are the axes of `performance_axes/gemm.yaml`; its
`development_order` puts them in the order a kernel is written — 1 interface and math semantics,
2 architecture and implementation path, 3 grid/tile/waves/split-K, 4 layout and addressing, 5 data
movement, 6 MFMA compute, 7 pipeline and synchronization, 8 reduction/epilogue/write-back, 9 host,
ABI and launch, 10 validation and tuning. The order is a reading order, not a second vocabulary:
every axis sits in exactly one step. Step 1 carries a note instead of axes (the math contract is
fixed by the task), and step 10 a note beside its axes: the task's oracle and timing belong to the
workflow's verify step, while `validation_contract` records the gate AITER's own test applies.

A card's first axis is the question it answers, and `gemm_decisions.md` places the card under that
axis's step, constraint cards first. Every other step its axes touch lists it again by link, so a
decision's consequences are visible where they land — the split-K combine card sits in step 8 and is
listed in step 3, where split-K is chosen, and step 9, where its host state is written. A step no
card answers says so on the page; that is the coverage gap, stated rather than hidden.

A card's optional `solutions` block is the same question as other AITER implementations answer it —
Triton's `remap_xcd`/`pid_grid` next to FlyDSL's `xcd_swizzle`, Gluon's `SwizzledSharedLayout` next
to the FlyDSL XOR16 helper, Triton's reduce-kernel split-K next to the FlyDSL flag-then-atomic combine. Each
solution cites its own source evidence, and the renderer checks that the evidence's language is the
stated backend. Transfer the intent, not the spelling; a CK template name or Triton compiler op is not
a FlyDSL instruction.

`references` point at always-on background docs under `perf_knowledge/` (hardware facts such as LDS
bank counts or FNUZ/OCP formats, how-to), as `path.md[:line[-line]]`; the renderer checks that the
file and the lines exist. They are never measured guidance, which stays behind the learned-card and
expert-skill switches.

## Shipped tuned databases: FlyDSL seeds and backend selection

AITER's tuned GEMM databases record, per shipped `(gfx, CU count, M, N, K)`, which backend its tuner
selected among asm, ck, cktile, flydsl and triton, and the selected configuration. The extractor keeps
the selection and drops the timing columns (`us`, `tflops`, `bw`, `errRatio`) on read: they belong to a
box this corpus cannot name. Two products follow:

- **FlyDSL seeds** (`cfg_…`): FlyDSL kernel names decoded with AITER's own naming schemes (split-K
  HGEMM and A8 preshuffle) and grouped by `(gfx, CUs, database, family, dtype, M bucket)`, rendered as
  *seed candidate / vary next* like the Triton seeds.
- **Backend selection** (`sel_…`): per-bucket row counts of the selected backend. They show where
  FlyDSL is AITER's production choice and where it has never been selected (for example gfx942 fp8
  and every block-scale database). A count is a selection, never a speedup; a shape shipped in two
  model databases counts twice.

Cards cite these IDs in `shipped_evidence`. Databases without a `libtype` column are inventoried and
reported as gaps rather than attributed to a backend by kernel-name guessing.

## Unified performance decisions

`performance_axes/gemm.yaml` supplies the common vocabulary that raw API mapping does not:

```text
compute_instruction  workgroup_tile  work_partition  execution_geometry
pipeline_schedule    operand_layout  memory_access   epilogue
architecture_capability  configuration_constraints  tunable_surface
implementation_variant  runtime_contract  validation_contract
```

Each axis keeps its comparison contract. For example, a direct FlyDSL `rocdl.mfma_*`, Triton's
`tl.dot`, a CK XDL template and an ASM object name all answer the `compute_instruction` question,
but they remain respectively instruction/compiler-op, template-family and binary-identity evidence.
The normalizer never calls those values equivalent.

The model also publishes:

- `comparison_context`: math contract, dtype, architecture, shape, baseline, accounting and
  toolchain/provenance fields that must match before measured bundles are comparable;
- `dependencies`: cross-axis rules such as architecture selecting the MFMA/DMA/async-copy bundle,
  tile geometry setting resource legality, and preshuffle changing both host and kernel layout;
- axis `kind`: separates performance choices from hard constraints, search spaces, structural
  variants and runtime integration contracts.

Every category currently emitted by `gemm_source.yaml` maps to at least one axis. A source record may
answer more than one question—for example an ASM launch formula is both execution geometry and a
runtime contract—but it remains one observation and gains no extra performance weight.

Inspect current source coverage and normalize shipped configs with:

```bash
python3 perf_knowledge/corpus/_normalize_performance_decisions.py \
  --source-evidence perf_knowledge/corpus/evidence/gemm_source.yaml \
  --tuned-evidence perf_knowledge/corpus/evidence/gemm_tuned_configs.yaml \
  --config-id cfg_386a49c8c892741c
```

`benchmarks/merge_results.py` applies the same normalizer to every measured implementation. Registry
rows therefore carry a `performance_decisions` vector alongside the untouched backend config:

- exact components such as `workgroup_tile.{m,n,k}` and `work_partition.split_k` can be compared;
- partial source/config observations stay `partial`;
- opaque runtime-selected CK/ASM configs remain explicitly unresolved;
- whole-bundle speedup remains on the implementation and is never copied onto an individual axis.

## Normalized upstream benchmark

`benchmarks/gemm.yaml` inventories reusable AITER UT/benchmark/tuner entrypoints. The first normalized
adapter compares fp8 block-scale implementations with one input/oracle/accounting protocol:

```bash
python3 perf_knowledge/corpus/benchmarks/run_fp8_a8w8_blockscale.py \
  --m 8 --n 4096 --k 1024 \
  --candidate triton --candidate ck --candidate cktile --candidate asm \
  --baseline triton --output /path/to/measured_fp8_gemm.json
```

It correctness-gates each implementation, interleaves CUDA/HIP-event samples, and records build or
unsupported failures as data. Input generation, JIT/build, final-output allocation and external
preshuffle are outside its declared steady-state timing; backend-internal workspace allocation and
layout preparation remain inside. The resulting whole-implementation speedup ranks bundles;
it cannot assign causal credit to an individual API or decision card.
The AITER FlyDSL preshuffle entry point is intentionally absent here: it uses per-token scaling, not
this suite's 128×128 block-scale contract. Generated block-scale FlyDSL remains measured by
`kernel_workflow`; correctness must not be traded for a superficially complete backend list.

Merge cached runs without losing their environment/config identity:

```bash
python3 perf_knowledge/corpus/benchmarks/merge_results.py \
  --input /path/to/run_a.json --input /path/to/run_b.json \
  --output /path/to/measured_implementations.json
```

Physical repeats retain distinct measurement IDs and share a comparable-context fingerprint. Top-K
selection requires that fingerprint, or the baseline plus matching AITER/Torch/HIP identity; it
deduplicates implementations and ranks their repeat median. Dirty-source or unresolved-runtime
identities remain in the registry for audit but cannot become an automatic prior.

## Reading the source-evidence table without drawing the wrong conclusion

The table counts **where an implementation question is answered in source**, per question per
language. The zeros mislead if read as absences of the choice itself:

- `scheduling` is 118 in FlyDSL, a handful of barrier/wait builtins in HIP's `opus.hpp`, and
  empty elsewhere. Triton kernels *are* scheduled — by a compiler pass. CK picks a
  `BlockGemmPipelineScheduler` enum. Neither writes a per-instruction ordering statement, so neither
  appears. The real content of that row is "FlyDSL is the language where this becomes your
  problem", which is worth knowing and is not the same sentence.
- `tile_shape` is nearly empty for Triton. Its tiles are `tl.constexpr` parameters, so the source
  states *that there is a knob* (`tunable_param`, 448 of them) while the values live in the shipped
  JSON. Two different source observations, in two different places, both recorded.
- An empty cell can also just mean no rule matches that idiom yet. The rules are a list at the top of
  `_extract_impl_facts.py`. A missing idiom is a fixable gap, not a finding about the language.

## What AITER states outside kernel source

Four kinds of AITER knowledge are not kernel code and are still read:

- **ASM kernel inventories.** The `.co` binaries publish no source, but `hsa/<gfx>/<kind>/*.csv`
  lists every kernel an architecture has: tile, B-preshuffle and split-K capability. Read by header,
  because the kinds disagree on column spelling. They answer "which tiles does the vendor ship for
  this dtype on this arch" — the fp8 block-scale kernels are the same six on gfx942 and gfx950
  (tile_n = 128, tile_m 32 to 128, all split-K capable); bf16 is 22 kernels on gfx942 and 24 on
  gfx950, which adds 256x256; fp4 exists only on gfx950 (35 kernels).
- **Grid-fill heuristics in the launchers.** The ASM launchers choose tile and split by the number
  of rounds `ceil(workgroups / num_cu)`, the idle CUs in the last round and `tile_m*tile_n /
  (tile_m + tile_n)`; the HIP skinny launcher sizes waves per group with `mindiv(rows, CuCount *
  YTILE, …)`. They are recorded as `grid_fill`, and they hold on every architecture because the CU
  count is read at run time.
- **Runtime dispatch.** `tuned_gemm.py` and `ops/gemm_op_*.py` decide the default backend when no
  tuned row exists — for example the bf16/fp16 skinny route for M <= 16, and the gfx950-only ASM
  route for B-preshuffled block scale. A routing statement is filed under the backend it routes to,
  with an excerpt that reaches up to its condition; no architecture is inferred, because a route in
  the `else` of a gfx942 test belongs to the other architectures.
- **Test gates.** `test_flydsl_splitk_hgemm.py` states the reference and tolerance AITER holds its
  FlyDSL HGEMM to, and the cases where it relaxes them — two M=1 B-in-LDS split-K cases. Recorded
  as `parity_tolerance` under the `validation_contract` axis: where the implementation says its
  numerics drift, never a replacement for the task's own oracle.

## Why regex, and where that runs out

Six languages is six grammars, and a Python AST pass cannot read a `.cu`. For a citation index what
matters is that a hit points at the right *line* — the reader opens the file. Where structure is
genuinely required there is an AST pass, applied only to the `.py` files where it is valid.

One such place is extent. A record is one line plus a two-line excerpt, which is enough for a
constant and not for a mechanism: the FlyDSL split-K combine sits in a 76-line
`if const_expr(IS_SPLIT_K):` block. For `.py`
files the extractor records `end_line` for any hit that starts a multi-line statement, and pages
cite `file:line-end`. `end_line` is location, not identity — it is left out of the content-bound ID,
so an edit deep inside a long function does not invalidate every card that cites its first line.

Comments are stripped before matching. This is not hypothetical: `detect_language.py` learned it by
classifying five plainly-HIP files as CK on the strength of `ck_tile::` mentions that appeared only
inside commented-out code.

Two things the regex approach deliberately does not attempt:

- **CK's positional template arguments.** Forty-odd unnamed positions are exactly what a regex gets
  subtly wrong. What is recorded is the instance template chosen and the two arguments CK names —
  loop scheduler and pipeline version — because those have an analogue a FlyDSL author must decide.
- **Instruction-level detail inside the `.co` binaries.** The launchers (`asm_gemm_*.cu`) publish the
  assembly's contract: object name, workgroup size, grid formula, argument block layout. The
  instructions themselves need a disassembly pass, which is separate work.

## Invariants `test_corpus.py` enforces

1. Every source record carries a language from the same vocabulary `kernel_workflow/scripts/kb.py` uses, so a
   corpus record and a learned card can be filtered by the same term.
2. Every source record carries `file` and a positive integer `line`, and the excerpt is verbatim — no
   paraphrase, because a paraphrase is a new artifact that can be wrong, while a quote can only be
   stale, and staleness is detectable from the commit.
3. No source-evidence record contains a performance number. Timings there would be measurements without
   a machine, which is the shape of an unfalsifiable claim.
4. Missing subtrees are reported in `missing`, never silently dropped. A section that is empty because
   nothing was found and one that is empty because the search never ran look identical otherwise, and
   the difference decides whether the next person repeats the search.
5. `gemm_source_evidence.md` matches the evidence, and `gemm_decisions.md` matches cards plus evidence.
6. The committed corpus resolves at the commit it names: `aiter_dirty_sources` is empty and no record
   is flagged `unreproducible`. This is the promise in the first paragraph, checked rather than
   assumed — "a commit is recorded" and "the commit identifies what was read" are different
   properties, and only the second one is worth anything here.
7. Every decision card states conditions, actions, rationale and limitations; source IDs must resolve
   to unchanged evidence, and measured guidance is rejected from this always-on layer.
8. Every card names known `axes`; `development_order` places every axis in exactly one step, so
   every card has a step; each solution's evidence is in the stated backend's language; references
   name existing files and lines; shipped IDs resolve in the tuned evidence.
9. The tuned-database evidence carries no timing column, and its `cfg_…`/`sel_…` IDs are unique.
10. `missing` is reported per declared pattern, not only when a whole language matched nothing.
11. New extraction rules are additive: they must not change the identity of any existing record, so
    no card citation moves when a rule is added. Deliberate exceptions so far: `padded_m` moved from
    `lds_pad` to `dispatch_padding`, where no card cited it; and ten `scale_operand` records were
    dropped because they were Triton kernel-name serialisers (`*_repr =`), not scales — the one card
    that cited three of them now cites the lines that actually apply the scale.
12. Every `match.requires` names a defined trait, and every card dtype is already canonical, so no
    card can be deferred or rejected forever by a spelling nobody passes.
13. `end_line` appears only on `.py` records and only after `line`; it never enters an evidence ID.
14. Both target architectures are covered from source: gfx942 and gfx950 each have ASM inventory
    records, and the page's *Coverage by architecture* table counts, per gfx, the cards that can
    apply, arch-specific source, ASM kernels by kind, FlyDSL seeds, backend selections and Triton
    seeds. An inventory CSV's `splitK` column is a capability flag (`split_k_capable`), not a split
    factor: the launchers accept a split above 1 only for a kernel whose flag is 1.
15. A dispatch routing record carries a kernel language (the backend it routes to) and no inferred
    architecture; routes to libraries (`hipblaslt`, `torch`) are not recorded, since they are not
    languages a kernel is written in.
