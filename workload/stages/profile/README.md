# Stage `profile` — decompose an AgentX reproduction into independently runnable kernel unit tests

**Input** `$IN` = baseline's out (`baseline.json`: which serving run is being attributed,
at what throughput, with what noise band).
**Output** `$OUT`: per-kernel share of GPU time, plus a frozen unit-test task directory
for each selected kernel.

## Implementations

| Directory | Notes |
| --- | --- |
| `kernel-extract/` | default. GEAK main's `kernel-extract` skill (`e2e_workflow/knowledge/kernel_extraction_skills/`) with the shared tools it copies into each task bundled under `shared/`; `PROVENANCE.json` records the origin revision, the origin hash of every file and which ones differ |

This skill is **agent-shaped** (contract §2B): it has no `run.sh`, because choosing a
boundary, identifying the callable and deciding which dispatch branch constitutes a whole
operator are judgement calls, not a procedure. The driver only prepares the environment;
an agent reads the skill's `SKILL.md` and follows it.

The skill's own convention has `SKILL_ROOT` pointing at itself and `EVAL_DIR` as its
working directory. Wired into this pipeline (the skill's
`references/pipeline-handoff.md` gives the full mapping):

```
SKILL_ROOT   = $SKILL_ROOT     (already exported by the driver; the names coincide)
SHARED_TOOLS = $SKILL_ROOT/shared
EVAL_DIR     = $OUT
```

## Artifacts → `result.json`

The skill knows nothing about `result.json`, and should not need to: it stays usable
outside this pipeline. That mapping is performed by `finalize.py` in the **slot
directory**. Because it belongs to the slot rather than to any implementation, it keeps
working when the profile skill is replaced, as long as the new skill emits files of the
same names.

```bash
python3 "$WORKLOAD_ROOT/stages/profile/finalize.py"
```

It checks for and records:

| File | Produced by | Consumed by |
| --- | --- | --- |
| `kernel_workload.json` | §2, workload census | reporting; the denominator for all shares |
| `kernel_ranking.json` | §2 | input to `select_targets.py` |
| `selection_targets.json` | §2, `select_targets.py` | **the roofline stage's ranking base** (`targets[].overall_pct` is `pct_gpu_time`) |
| `task_plan.json` | §3, boundary declarations | audit |
| `tasks/<name>/` | §4, frozen harness | **one per GPU in the optimize stage**; must contain `unittest.py`, `meta.json`, `kernel_src/`, `cases.py` |
| `REPORT.md` / `results.json` | §5 | reporting |

`metrics` records `kernels_censused`, `targets_selected`, `tasks_built` and
`selected_pct_gpu_time` (the summed share of the selected kernels).

## Acceptance rules

- `targets_selected` ≠ `tasks_built` → `status: "blocked"`. Every selected identity
  either has a task or a stated blocker; a missing task must not be papered over by
  renormalizing the remainder to 100%.
- A task directory without `unittest.py` cannot be run by the optimize stage and is
  rejected outright.
- When GPUs are unavailable it is acceptable to complete the CPU-only portion and report
  `blocked`, but a CPU-only check must never be presented as a complete extraction.

## Pitfalls

- `overall_pct` in `selection_targets.json` is the share of the **whole workload**, not
  the share within the trace window. Window share is a different number: the capture
  window for this scenario once held a prefill:decode ratio of 32:7 against a true ratio
  of 1:1119, and ranking on window share would have promoted prefill kernels while
  end-to-end time was bound in decode.
- The default selection rule is **per-phase Top 5 AND overall share strictly > 3%** —
  rank first, then filter, with no backfilling of the sixth entry. To narrow it, use
  `params.max_targets`; do not edit the skill.
