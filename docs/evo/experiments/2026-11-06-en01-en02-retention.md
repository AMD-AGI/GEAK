# EN01 and EN02: retained development and completed callable results

This is a research record. It is not optimizer input or a runtime policy change.
The tracked EVO revision `9f925de2` contains later prospective realization-return
guidance; EN01 did not run that later revision. These observations therefore do
not validate that commit or isolate the benefit of its guidance.

## Completed measurements

Each final uses all 21 captured prefill cases and 1,590 call weights. Its primary
is the median of three complete O-X-O ratios of weighted times. Each candidate
has its own ORIGINAL brackets; repeated timings are not independent searches.

| Experiment and candidate | Final speedup over its ORIGINAL | Three-round range |
| --- | ---: | ---: |
| EN01 EVO a1 | 1.127721078893x | 0.002038346528 |
| EN01 EVO a2 | 1.061034812467x | 0.006127324941 |
| EN01 selected GEAK b0 | 1.044524018117x | 0.003548637082 |
| EN02 control c2 | 1.144030876688x | 0.008883584876 |
| EN02 fixed-reference guidance t2 | 1.069920062628x | 0.011036729329 |

EN01's selected normalized EVO/GEAK ratio is 1.079650691925, computed as the
a1 primary divided by the b0 primary. It is not direct same-final EVO/GEAK
timing and supports neither general EVO superiority nor attribution to one
component. EN02 is a guidance comparison with EVO in both conditions. Its
control has the higher selected normalized primary. The retained c1 presearch
failure and t1 empty-ORIGINAL ineligibility remain unscored; neither receives
an invented primary or range. The range is descriptive and is not a confidence
interval or cross-condition test.

The reported c2 and t2 measurements are complete evaluations of selected
post-hoc archive sources. Their native delivery outcomes remain separately
recorded, and the three timing rounds are not independent search replicates.

These are historical eager callable measurements, including any per-call
packing. The 512 MiB read-evict condition is not serving-calibrated. Current
AgentX E2E gain, official quality, substantial-region megakernel benefit and
workload roofline attainment are not established by these results.

## What the actual source development shows

- **a1 retained development:** native verified scores progressed from
  1.0058787756x through 1.0909103684x to 1.1380761275x, followed by the independent
  final above. The first source was below the promotion threshold but was
  available as a parent. Source inspection confirms continued development.
  This is an exercised retention mechanism, not a no-retention counterfactual.
- **a2 donor survival:** a correct local alternative scored 0.9748719296x.
  The agent's integration and later development retain its selective transposed
  computation together with other case-specific paths. Surviving source is
  stronger evidence than a donor label. It does not isolate the donor's benefit,
  establish within-case epistasis, or show cross-run transfer.
- **c2 incumbent development:** the archive records ORIGINAL to r1_d0 to r2_d0,
  with no recorded donors. The final patch matches r2_d0. This is continued development
  of a winner; parent labels alone do not establish survival of every earlier
  computation.

In each candidate's median-primary round, a1 saves 64.326070 weighted call-ms,
a2 32.751496 and c2 71.441651. a1 concentrates savings in query/output/engram;
c2 has positive totals for all seven families, while four individual cases
regress in every round. All regressions remain in the primary metric. These
case/family differences motivate a composition test; they are not workload
GPU-time shares or additive E2E predictions.

## Decisions and falsifiable next tests

Keep promotion and retention separate, preserve actual implementation bytes and
remaining questions, and let engineers develop or combine sources when the
experiment permits it. No structural-headroom bonus or invented roofline score
is justified. The fixed reference does not change the latency ranking; any
benefit must arise through better engineering decisions.

R1 prospectively compares retained-parent and incumbent-only EVO policies with
GEAK controls. Preserve its complete search and independent finals. Do not
infer policy efficacy from a partially observed archive or missing result.

M15-PORTFOLIO-FINALS-02 tests a fixed researcher-composed a1/c2 dispatcher against
both constituents, with complete ORIGINAL/candidate correctness and three O-X-O
rounds each, using each run's own ORIGINAL brackets. Its frozen descriptive rule requires the composed median to exceed each
constituent median by more than the sum of their observed ranges. It tests
available complementary value; even a win would not establish autonomous
cross-run recombination or E2E benefit. No completed M15 outcome is asserted here.

The decisive workload study remains the latest applicable AgentX recipe:
native trace/telemetry evidence, an agent-selected substantial complete region,
qualified ORIGINAL state, sustained source-backed engineering, exact delivery,
full serving and natural quality, followed by reprofile. Researcher-written
solutions from this record must not be inserted into independent discovery arms.

## Evidence identities

The [EN01 complete raw recomputation](/run/user/13060/en01-independent-final-readout-20261105-01/ALL_RAW_FINAL_RECOMPUTATION_03.json)
is the direct source for a1, a2, b0, their three-round ranges and the selected
normalized ratio (SHA256
`b4672a54c7f6e0d3755f69d7b3550e0333ba38b3fc719389e8d3bd2c641bc7da`).
b0's exact primary is `1.0445240181169255`, its range is
`0.0035486370822002122`, and the exact a1/b0 ratio is `1.0796506919254463`.

The [c2 raw recomputation](/run/user/13060/en02-c2-independent-final-readout-20261106-01/RAW_RECOMPUTATION_01.json)
has SHA256 `0926b70660f7fc8c6a755da5148b19c6c9906f3588d505e1a9ff49cc0710fc6b`.
The [t2 raw recomputation](/run/user/13060/en02-t2-independent-final-readout-20261106-01/RAW_RECOMPUTATION_01.json)
has SHA256 `7fa543284f106a8a877276d7868d7fe87c9d6e91fdaa49bc3a50983f771cdf94`.
They retain explicit default failure/ineligibility records and identify the
post-hoc source and completed-evaluation scope.

The [design readout](/run/user/13060/en01-en02-local-design-readout-20261106-01/DESIGN_READOUT_01.md)
(SHA256 `26f2d69a506d2fca5de9fc4ff0888d1dc8d08ca2019312fede5d7deb3fc97839`)
links the exact sources, native development measurements and M15's frozen
composition protocol. Its [additive family accounting](/run/user/13060/en01-en02-local-design-readout-20261106-01/ADDITIVE_FAMILY_ACCOUNTING_01.json)
has SHA256 `4e388208bc76771670b966a34dc7f6868ef70adc55cd25513dcbfca9b152f25b`.
Family savings use each candidate's own median-primary round, not a sum of
per-case medians or a common-session measurement.

The later guidance is commit `9f925de2902815d3125f0a51fd4d0dc2d7116f01`,
`kernel_workflow/roles/evolution.md`, inspected read-only in the retained
worktree `/run/user/13060/evo-workspace-local-exclusions-20261106-01/worktree`.
It requests returning the best correct requested realization even when slower
than the selected parent, while preserving the incumbent and leaving promotion
to independent verification. The historical EN01 observations do not validate
that later guidance.

No new benchmark was run for this note. The local fact-check manifest records
source hashes and exact field bindings; this note changes no optimizer input.
