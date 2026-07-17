# Fragment: expert_skills (e2e layer) — ADVISORY, injected only when use_expert_skills is ON

> This fragment is appended to a role's prompt by `e2e_workflow.js` **only when `use_expert_skills`
> is true (opt-in; default OFF)**. When OFF (the default), nothing is injected and behavior is
> byte-identical to a run without this feature. It is consumed by routing/integration roles (System
> Architect, Op Benchmarker, e2e
> Integrator). It is **advisory** for ordinary skills: a matched skill is a high-prior candidate to
> reproduce, and never overrides your on-box A/B gate. **Exception:** a matched skill with
> `enforcement.mode: strict` is a MANDATE — see "Strict-enforcement skills" below.

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

## Strict-enforcement skills (MANDATE, not advisory)

A matched skill whose index entry (or frontmatter) has `enforcement.mode: strict` is **not** an advisory
prior — it is a **MANDATE**. The only such skill today is **`flash_mla_tilelang_to_triton`**: it fires
whenever the live bottleneck is an **MLA-attention decode** op on a **DeepSeek-V4 / `deepseek_mla`** model
served on **gfx950 (MI355X)** with an **FP8** KV cache and a **TileLang** source you are porting to
**Triton**. When such a skill matches, you MUST:

- **Follow its `Procedure` verbatim** — you may NOT re-route to a generic backend or a different recipe.
  It is the required candidate, not one option among many.
- **Honor its `enforcement` block** from the skill file: use its unit test as-is
  (`unittest.cases_immutable` — do not re-derive the case set or loosen tolerances) and implement every
  item in `optimization.mandatory_specs`. Skipping a mandatory spec is allowed ONLY with an explicit
  on-box benchmark that justifies the skip (`skip_requires_benchmark`).
- **Still gate the final decision on your on-box A/B.** Strict means the *recipe is mandatory*, NOT that
  the result ships unmeasured: the skill can never force a result below your measured baseline. If, after
  faithfully reproducing it, measurement contradicts the skill, the measurement wins — record it so the
  skill can be marked `stale`.

In short: for DeepSeek-V4 flash-MLA you cannot answer "no skill needed" or substitute your own plan —
reproduce this skill, then let the numbers decide whether it ships.
