# Cleanup on the evo branch

This branch organizes GEAK and the existing EVO engine for continued development.
The engine now connects to supplied execution services. An admitted native task
can run through search, final measurements and checked result export. The full
EVO dispatcher adapter is implemented; installation and live validation remain.

## Changes

- Related EVO code lives in planner, engineer, archive, selection and evaluation
  modules. Saved programs, ideas, measurements and parent links keep their formats.
- The shared host, admitted command and result exporter use the existing owner,
  budget and evaluator. The kernel lane can invoke an optimizer command directly.
- Serving result checks and reports have separate modules. Shared parsing and
  packaging code has one implementation. The public GEAK entry points remain.
- JavaScript workflows build from readable source files. The generated copies
  stay for the existing single-file runner; CI checks that they match.
- Roles and prompts use shorter instructions. Correctness, required cases,
  source identity, timing and result contracts stay explicit. Backend choice and
  GPU use follow the actual task rather than a historical experiment's settings.
- Hardware guides use product-specific limits and state model assumptions.
  Missing evidence stays unknown. Useful compiler and debugging recipes remain
  available through concise guides and references.

## Intentional behavior changes

The old B4/B5/B6 study switches are retired. New calls containing `ablation_*`
fail before dispatch. This does not remove archive diversity, crossover or
continued engineer development from EVO. Old study records retain their original
versions; do not rewrite their configuration for the new branch.

The original serving flags, environment and overlay now survive tuning and resume.
They are separate from the current timing reference. A legacy state that did not
retain them reports unknown values. Correctness roles require recorded baseline
evidence before using a recovered seed; they must report incomplete or flagged
validation when it cannot be established.

MI350X and MI355X use separate published dense capacities. Bare `gfx950` has no
product-specific compute roof. `expected_target=unknown` means no product
constraint at Setup/Freeze; explicit supported targets are checked. An FP8 counter
conversion now requires a verified factor instead of silently using 512.

The vendored tuning tree keeps its pinned upstream bytes. Its benchmark uses
legacy plausibility thresholds (`gfx950`: 2500 BF16 TFLOP/s), which are not
product-specific roofline denominators. Use the canonical MI350X/MI355X peak
table for model percentages; keep GEAK-specific interpretation outside that tree.

Candidate-saving instructions use the declared primary metric and a private Git
baseline, including new source files. Retry instructions preserve required
measurement coverage and allow an honest incomplete result at the deadline.
All instruction changes apply to new runs; shorter prompts do not establish
equal model behavior or better discovery.

## Records and continued work

The archive now indexes each exact trial source supplied at engineer settlement,
including failed and slower trials. Host-assigned authorship is recorded. A
reviewed intermediate source can start a fresh direction through normal parent
checks; it is not automatically promoted. The [archive guide](archive.md) covers
the deterministic folder view and its restoration limits.

An idea card keeps the assigned question; the linked engineer record keeps the
strategy and outcome. Idea status remains unassessed until reviewed. Old frozen
records stay intact.

Unused imported CLI planner lifecycle, old settlement/observation helpers and
fixture-admission wrappers are retired. The active native planner, evaluation,
roofline analysis and saved-data formats remain. Compact regression digests
preserve all six GEAK workflow scenarios; generated bundles still ship.

## Validation and remaining work

CPU checks cover the search components, deterministic archive, shared host,
admitted command, checked result export and generic dispatcher. Serving result
checks, GPU identity checks and existing workflow regressions were also exercised.
Exact versions, commands and receipts belong in the dated integration handoff.
Claiming compatibility with the retained versions requires a run with
`GEAK_EVO_ORIGINAL_SRC`, `GEAK_EVO_OCT15_SOURCE` and
`GEAK_EVO_CODEX_ROUTE_SOURCE` supplied and zero skipped tests. Public CI can pass
without those private source references; that does not establish compatibility.

The native adapter connects the generic dispatcher to the admitted EVO command.
The next step qualifies its installed model/GPU path and full-workload result.
Generated Hyperloom tasks still need support for their paired timing evidence.
Weighted scoring requires the caller's complete workload weights; the exporter
checks those weights and preserves the lane's objective. A prompt cannot supply
missing workload evidence.

New features belong on separate `feat/evo-*` branches. Keep versioned experiments
and their source/instruction pins unchanged. The installed personal hardware
skill and frozen experiment sources were not replaced by this source cleanup.

New model calls require an explicit backend, requested model, reasoning effort
and pinned route. Recovery preserves that profile; historical labels do not
choose a model for new work. CPU checks reject a changed profile or route.
The launcher remains responsible for provider configuration and verification of
the served model. See the [shared host guide](shared-host.md) for the runtime
contract and the [architecture](architecture.md) for module responsibilities.
