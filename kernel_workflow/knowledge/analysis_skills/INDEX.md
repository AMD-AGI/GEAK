# Analysis skills — index (kernel_workflow)

Pluggable **profile-analysis** skills for Kernel Workflow. A dedicated `analysis_engineer` runs at
most ONE of these after `profile_engineer` completes its own bottleneck classification
(compute/memory/latency/lds/balanced/overhead), to
enrich that classification with operator-specific structure the generic labels cannot express — e.g.
a distributed MoE kernel's route/expert imbalance and Stage1/Stage2/combine time-share.

Selected by the `analysis_skill` arg (`kernel_mega.js`). `analysis_skill=none` (the default) or an
unknown/unreadable skill dir disables the step entirely and the run behaves exactly as it did before
this feature existed — see "Degradation" in each skill.

This mirrors `e2e_workflow/knowledge/analysis_skills/` (the `roofline` skill there is the model this
was built from) but is scoped to Kernel Workflow's single-op profiling loop rather than an e2e serving
run. The two directories are independent; a skill added here does not appear in e2e's list and
vice versa.

| skill | dir | what it adds | when to use |
|---|---|---|---|
| `moe_bottleneck` | `moe_bottleneck/` | **READY contract** with typed evidence catalog, corrected timing/ATT math, and conservative findings; workload status may remain `awaiting_measurement` | a distributed MoE kernel with multi-rank profiling and controlled-experiment data |
| `none` | — | nothing (pre-feature behavior) | any non-MoE kernel; reproducing an old run byte-for-byte; skill is misbehaving |

## Contract every skill must honour

1. **Analysis only.** A Skill emits findings, bounded hypotheses, constraints, bounds, unknowns, and
   unranked reference patterns. It never emits or ranks implementation `directions[]`, never prunes a
   candidate, and never overwrites `bottleneck` or `top_kernels`. Step-3 TechLead owns candidate
   generation/ranking; isolated A/B against the frozen oracle remains the judge.
2. **Deterministic arithmetic first.** Each skill provides a checked-in runner for parsing and unit
   math, built on `kernel_workflow/scripts/multi_rank_analysis/`. `SKILL.md` defines doctrine,
   required evidence, and degradation. A runner failure degrades to the generic profile result; an
   agent must not improvise a confident numerical verdict by hand.
3. **Degrade, never fail.** Every skill defines an explicit degradation ladder ending in "emit nothing
   and let the caller fall back to the pre-skill behavior".
4. **Declare confidence.** Every emitted number carries a confidence level, and the consumer
   (`tech_lead.md`'s `plan_round`) is told what it is allowed to do at each level.

## Adding a skill

Drop a new directory here containing a `SKILL.md` that follows the contract above, add one row to the
table, and pass `analysis_skill=<dirname>`. No orchestration or role change is required — the
`ANALYSIS_SKILL`/`ANALYSIS_SKILL_ON`/`ANALYSIS_SKILL_INPUTS` gating block in `kernel_mega.js`
resolves any `<dirname>` under this directory generically.
