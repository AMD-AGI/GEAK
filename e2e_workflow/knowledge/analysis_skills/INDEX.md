# Analysis skills — index

The Profiler runs at most one selected analysis skill after `parse_profile.py` produces the
standardized Top-N. Its output adds advisory model context and coverage for the Architect.

Select with `analysis_skill` in `e2e_workflow.js`. `none`, `false`, an empty string or an
unreadable skill directory skips analysis. The default is `kernel-roofline`; the legacy
`roofline` name is an alias for it.

| skill | directory | what it adds | when to use |
|---|---|---|---|
| `kernel-roofline` | [kernel-roofline/](kernel-roofline/SKILL.md) | native model work/bytes, latency, roofline, operation/kernel call stacks and coverage | default `tl-only`; explicit `rocprof` for hardware counters |
| `none` | — | nothing | disable profile analysis |

## Contract

1. **Advisory only.** Never prune candidates, overwrite measured `pct_gpu_time`, or make an analysis
   metric the sole reason to optimize or stop. Each skill must explain which interpretations its
   metrics support. A roofline reference distance is not a predicted attainable speedup or e2e gain.
2. **Follow the selected engine.** Read `SKILL.md` and the relevant references. Install required
   dependencies when available. If a native model or report is unavailable, annotate the reason;
   never replace it with guessed metadata, handwritten formulas or a different engine.
3. **Analysis failures do not block profiling.** Preserve the Top-N and supported results. Missing
   models, metadata, references or tooling remain unavailable; if no artifact can be produced,
   return an empty path and continue ordinary routing.
4. **Declare evidence and scope.** Preserve native fields, units, provenance, requested-case
   coverage and uncertainty. Distinguish captured metadata from UT/source-reviewed enrichment.

The default skill's [GEAK integration contract](kernel-roofline/references/geak-integration.md)
defines the `profile_roofline_json` handoff and its linked native artifacts.

## Adding a skill

Add a directory with `SKILL.md`, document its output and degradation behavior, add it above,
and select `analysis_skill=<dirname>`. A new artifact schema also needs consumer instructions.
