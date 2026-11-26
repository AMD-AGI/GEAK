# Fragment: expert_skills (kernel layer) — ADVISORY, injected only when use_expert_skills is ON

> Appended to a role's prompt by `kernel_workflow.js` **only when `use_expert_skills` is true
> (opt-in; default OFF)**. When OFF, this fragment is not injected; the base role prompt remains active,
> including general target-backend and language guidance from `perf_knowledge`.
> Consumed by the
> tech_lead (planning) and author/engineer roles. **Advisory**: a matched recipe skill is a high-prior
> candidate to reproduce, never a mandate, and never overrides your isolated A/B vs the oracle. A
> reference fact store (see the last section) is queried, not reproduced.

## What expert skills are
Human-authored, validated kernel recipes under `EXPERT_SKILLS_DIR` — especially **migration skills**
(port an op from one backend/DSL to another, e.g. TileLang→Triton, →FlyDSL) and authored-kernel
playbooks. They are *recipes with regulated steps*, not facts; they let you reproduce a known win
faster but can never reduce a result below your measured baseline.

One kind is not a recipe: a skill whose `skill.md` front matter says `role: reference_fact_store`
(today `gluon_authoring`) is a store of conditional facts to query. Step 3 below and the high-prior /
`Procedure` framing do not apply to it; use it as described in "Reference fact stores". Its index entry
is `draft` / `auto_apply: false` (it claims no validated win), so the `validated` condition in step 2
does not apply to it: consult it when its `match` gens and backends fit.

## How to use them

1. Read `EXPERT_SKILLS_DIR/index.yaml`.
2. A skill matches the current op when ALL hold:
   - `scope: kernel`. The index also carries `scope: tuning` entries — the vendored tuning skillset,
     owned by the e2e tuning phase. Those match on `operator: '*'`, so they would otherwise match
     everything here; they are about tuning an op the stack already dispatches, not authoring one.
   - `match.operator` is either a scalar equal to this op's operator
     (`KK_OPERATOR` / `op_spec.op_kind`) or a list containing that operator
   - box `gen` ∈ `match.gens`; `op_spec.dtype` ∈ `match.dtypes`; `op_spec.regime` ∈ `match.regimes`
   - migration skills: `from_backend`→`to_backend` fits this run's `mode`/`target_language`
     (e.g. authoring Triton from a TileLang source → a `tilelang→triton` skill applies)
   - `validation_status == validated` (ignore draft/failed; `stale` = plain reference only)
   - `'*'` in a `match` field matches any value. A skill matched only through `'*'` (generic over
     operators or dtypes) is a reference: each of its levers holds for the arch / dtype / kernel class
     its evidence names; on another dtype or kernel class treat that lever as a hypothesis to A/B, not
     as a high-prior recipe.
   - The match depends only on the op, dtype, regime and box: once your plan or memory records the
     matched skill ids for this op, reuse that record instead of re-reading `index.yaml`.
3. For each matched recipe or migration skill, Read the skill file and treat its `Procedure` as a
   **high-prior author/optimize candidate**: follow its kernel structure and the named lever, honor `Knobs & pitfalls` and
   `Do-no-harm notes`, then measure against the immutable oracle as usual. The skill's
   `expects.isolated_speedup_min` is a sanity reference, not an acceptance shortcut.
4. Always write your own measured baseline first; the skill seeds the optimization direction, it does
   not replace the COMMANDMENT / oracle. The isolated A/B picks the winner.

If no skill matches, proceed exactly as without this fragment.

## Reference fact stores (`role: reference_fact_store`, e.g. `gluon_authoring`)
- Read its short `skill.md` (scope, evidence convention, how to query), then query instead of reading
  files: `python3 EXPERT_SKILLS_DIR/skills/<id>/scripts/q.py "<words or a pasted error>"`
  (`--arch`, `--dtype`, `--kind fact|error|api|isa|hw|tool|trap`, `--id <id> --full`, `--toc`).
  `INDEX.md` beside it lists every card id, its keys and the error-string table.
- Each card is a conditional claim with a `scope` (arch, Triton version, dtype, regime) and an
  `evidence` label. Judge each card by its evidence and scope: a card whose scope does not cover
  your case is not evidence for it, and `evidence: unverified` cards rank last — treat them as
  hypotheses and run the card's `check` before relying on one.
- A card with `overrides[]` supersedes the named `perf_knowledge` file(s) for the card's scope only;
  outside that scope the file stands.
- Cards carry no plan or step order and the skill's `expects` is not a target: your plan, the frozen
  COMMANDMENT and the isolated A/B vs the oracle decide, as without this fragment.
- `q.py --kind tool` lists the shared kernel tools one line each; `q.py --kind tool <words>` gives purpose,
  command, outputs and refusal codes; each tool's `--help` is its detailed doc. `INDEX.md` is large: grep it.
  Paste an error line verbatim into `q.py` (exact strings rank first).
