# Fragment: expert_skills (kernel layer) — ADVISORY, injected only when use_expert_skills is ON

> Appended to a role's prompt by `kernel_workflow.js` **only when `use_expert_skills` is true
> (opt-in; default OFF)**. When OFF (the default), nothing is injected and behavior is byte-identical.
> Consumed by the
> tech_lead (planning) and author/engineer roles. **Advisory**: a matched skill is a high-prior
> candidate to reproduce, never a mandate, and never overrides your isolated A/B vs the oracle.

## What expert skills are
Human-authored, validated kernel recipes under `EXPERT_SKILLS_DIR` — especially **migration skills**
(port an op from one backend/DSL to another, e.g. TileLang→Triton, →FlyDSL) and authored-kernel
playbooks. They are *recipes with regulated steps*, not facts; they let you reproduce a known win
faster but can never reduce a result below your measured baseline.

## How to use them

1. Read `EXPERT_SKILLS_DIR/index.yaml`.
2. A skill matches the current op when ALL hold:
   - `match.operator` == this op's operator (`KK_OPERATOR` / `op_spec.op_kind`)
   - box `gen` ∈ `match.gens`; `op_spec.dtype` ∈ `match.dtypes`; `op_spec.regime` ∈ `match.regimes`
   - migration skills: `from_backend`→`to_backend` fits this run's `mode`/`target_language`
     (e.g. authoring Triton from a TileLang source → a `tilelang→triton` skill applies)
   - `validation_status == validated` (ignore draft/failed; `stale` = plain reference only)
3. For each match, Read the skill file and treat its `Procedure` as a **high-prior author/optimize
   candidate**: follow its kernel structure and the named lever, honor `Knobs & pitfalls` and
   `Do-no-harm notes`, then measure against the immutable oracle as usual. The skill's
   `expects.isolated_speedup_min` is a sanity reference, not an acceptance shortcut.
4. Always write your own measured baseline first; the skill seeds the optimization direction, it does
   not replace the COMMANDMENT / oracle. The isolated A/B picks the winner.

If no skill matches, proceed exactly as without this fragment.
