# Analysis skills — index

Pluggable **profile-analysis** skills. The Profiler runs at most ONE of these after `parse_profile.py`
has produced the standardized Top-N, to enrich it with a headroom estimate the Architect can route on.

Selected by the `analysis_skill` arg (`e2e_workflow.js`). `analysis_skill=none` (or an unknown /
unreadable skill dir) disables the step entirely and the run behaves exactly as it did before this
feature existed — see "Degradation" in each skill.

| skill | dir | what it adds | when to use |
|---|---|---|---|
| `roofline` | `roofline/` | per-kernel % of the hardware roofline, bound type, attainable speedup, expected e2e gain | default; any GPU-bound serving run |
| `none` | — | nothing (pre-feature behavior) | reproducing an old run byte-for-byte; skill is misbehaving |

## Contract every skill must honour

1. **Advisory only.** A skill may ADD fields, ADD annotations and SUGGEST an ordering. It may never
   prune a candidate, never overwrite the measured `pct_gpu_time`, and never be the sole reason a
   kernel is or isn't optimized. The on-box measurement is always the judge.
2. **Markdown-first.** The skill's logic lives in its `SKILL.md` so an agent can execute it by reading.
   Helper scripts are OPTIONAL mechanical primitives (parsing, unit math). If a helper is missing or
   raises, the agent completes the analysis by hand from `SKILL.md` — a broken script must not disable
   the skill, and a broken skill must not fail the run.
3. **Degrade, never fail.** Every skill defines an explicit degradation ladder ending in "emit nothing
   and let the caller fall back to the pre-skill behavior".
4. **Declare confidence.** Every emitted number carries a confidence level, and the consumer is told
   what it is allowed to do at each level.

## Adding a skill

Drop a new directory here containing a `SKILL.md` that follows the contract above, add one row to the
table, and pass `analysis_skill=<dirname>`. No orchestration or role change is required.
