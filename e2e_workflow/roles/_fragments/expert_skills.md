# Fragment: expert_skills (e2e layer) — ADVISORY, injected only when use_expert_skills is ON

> This fragment is appended to a role's prompt by `e2e_workflow.js` **only when `use_expert_skills`
> is true (opt-in; default OFF)**. When OFF (the default), nothing is injected and behavior is
> byte-identical to a run without this feature. It is consumed by routing/integration roles (System
> Architect, Op Benchmarker, e2e
> Integrator). It is **advisory**: a matched skill is a high-prior candidate to reproduce, never a
> mandate, and never overrides your on-box A/B gate.

## What expert skills are
Human-authored, validated optimization recipes under `EXPERT_SKILLS_DIR` (one file per skill). Unlike
`perf_knowledge/` (facts) these are end-to-end *recipes with regulated steps* that already passed an
e2e/isolated validation gate. They can only help you find and reproduce a known win faster — they can
**never reduce a result below your measured baseline**, and if a skill conflicts with your measurement,
the measurement wins (note it so the skill is later marked `stale`).

## How to use them (per phase)

1. **Read the selector.** Open `EXPERT_SKILLS_DIR/index.yaml`.
2. **Match against the live bottleneck** you are routing/optimizing. A skill matches when ALL hold:
   - `match.operator` == the bottleneck operator (same names as `capability_index.yaml`)
   - the box `gen` ∈ `match.gens`
   - `env_report.model_arch_class` ∈ `match.arch_class` (or `match.arch_class` contains `'*'`)
   - if the skill is a migration skill (`from_backend`/`to_backend` set), the live path / your author
     plan fits that source→target
   - `match.profile_signature` (if present): the Top-N op name matches `op_name_regex` and its
     `pct_gpu_time ≥ min_pct_gpu`
   - `validation_status == validated` (ignore `draft`/`failed`; treat `stale` as a plain reference only)
3. **For each matched skill**, Read its file and treat its `Procedure` as a **high-prior candidate**:
   - In routing (System Architect): list it in the head/kernel `author_plan` BEFORE generic backends,
     annotated `source: expert_skill:<id> (advisory)`.
   - In bake-off / integration (Op Benchmarker / e2e Integrator): reproduce its Procedure as one
     candidate, honor its `Knobs & pitfalls` and `Do-no-harm notes` (e.g. keep decode generic), and
     still run the normal e2e A/B gate. The skill's `expects` is a sanity reference for the delta, not
     an acceptance shortcut.
4. **Never skip measurement.** Multiple matched skills all enter the candidate set (no ranking); the
   on-box A/B picks the winner. Do not re-route away from what the profile says just because a skill exists.
5. **Close the loop.** When you curate `knowledge/learned/` (update_experience phase), record on the
   relevant card the skill id you used and its MEASURED result, so the skill's validation can be refreshed.

If no skill matches, proceed exactly as you would without this fragment.
