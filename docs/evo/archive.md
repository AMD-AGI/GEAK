# Inspect retained candidates

GEAK-evo keeps its population under `<eval_dir>/evolution/`:

```text
evolution/
  archive.json           candidate metadata and selected evidence
  base/                  fixed source at the start of the optimize loop
  patches/<id>.diff      cumulative candidate patches against that base
```

Open the JSON directly with a text editor or a standard JSON viewer:

```sh
python3 -m json.tool /absolute/path/evaluation/evolution/archive.json
```

The archive schema is `geak.evolution/1`. Its top-level `base` and `base_commit`
identify the fixed starting source. `baseline` describes the optimize-loop
starting point, including any prior warm start; the ID `ORIGINAL` refers to that
run base. `best_id` identifies the selected validated candidate, or `ORIGINAL`
when no retained candidate has been selected.

## Entry fields

| Field | Meaning |
| --- | --- |
| `id` | Workflow-assigned direction ID, such as `r2_d0`. |
| `parent_id`, `donor_ids` | Selected starting candidate and declared donor references. |
| `title`, `approach`, `specialty` | Planner descriptions; `approach` is an agent-supplied label. |
| `base_commit`, `patch_ref` | Base identity and retained patch path, SHA-256, and base commit. |
| `verified`, `speedup`, `status` | Verification state and primary metric when available. |
| `engineer`, `verifier` | Selected structured evidence from the existing role outputs. |
| `engineer.observed_evidence`, `verifier.observed_evidence` | Native reader facts only, when an external result was checked; absent for ordinary reported evidence. |
| `evolution_context` | Parent resolution, reference errors, and starting-workspace provenance. |
| `retention_error`, `last_record_error` | Source retention or later recording failure, when present. |

A `correct` or `slower` entry has passed verification with a positive, finite
speedup. `slower` means its speedup is at most 1 against the task comparator;
`correct` does not imply that it improved the current best enough for promotion.
`incorrect` records failed correctness; `unverified` lacks accepted verification.
An unavailable measurement is represented by a null speedup.

Follow `patch_ref` for the retained source and the original round outputs for
full reports and raw measurements. Patches include inherited parent changes
against the fixed base. Check retention errors before treating a patch as an
available parent. Keep the evaluation directory and referenced evidence together.

## How agents use the archive

The planner receives the latest twelve entries plus the best entry, with a path
to the complete archive. It can inspect older entries and their evidence before
choosing a parent. The workflow materializes the parent's patch into a private
workspace; donor patches are references for the engineer to inspect.

With `evolution_evidence_handoff: true`, the engineer prompt also includes a
compact view of the selected parent and donors: their patch references, status,
verification state, recording errors, and stored engineer/verifier evidence.
Prose is summarized if its JSON exceeds 16,384 UTF-8 bytes. A separate
`selected_case_evidence` view carries complete case reports when they fit that
budget. Otherwise it supplies no partial case table: it reports the omission,
row counts and exact archive entries to read. This option defaults to false and
does not change stored archive rows or promotion rules.

The archive is a per-run record. Configured population lineages can publish
committed checkpoints and import peer source/evidence into it; see
[checkpoint migration](configuration.md#checkpoint-migration). This exchange is
separate from resuming a terminated archive, for which there is no protocol.
An ancestry link records a selection; useful contribution survival and later
improvement require inspection and measurement.

With `evolution_case_evidence_view: true`, each exposed planning entry and current
round-memory entry also carries `case_evidence`: its archive/patch references,
entry verification state, task metric, and separate engineer/verifier reports.
The view preserves every supplied per-case row, including extra fields such as
repeat counts or measurement spread, and all supplied raw evidence references.
It does not filter winners, infer families, calculate scores or certify reports.
`case_rows` counts supplied rows only; it does not establish complete task coverage.
Missing case evidence is `null`, distinct from an explicitly empty array. Reference
paths may be unavailable and still need checking. Legacy verifier reports retain
their existing authority; native measurement checks remain separate.

This option defaults to false and leaves the current planning/memory input shape
unchanged when omitted or false. It is independent of engineer evidence handoff.
It also supplies `CANDIDATE_CASE_EVIDENCE` to same-round EVO integration, with the
same complete-or-reference-only size policy as selected engineer case evidence.
Both settings retain the same full archive, including supplied engineer weighted/
arithmetic metrics and verifier variance/graph-safety notes. The recent-twelve-plus-
best entry selection, full-archive access, eligibility and delivery rules are unchanged.
Both options are supported by the direct lane, optimize/author pass-through, and
the E2E handoff's existing nested-lane argument mapping. Values are forwarded
without coercion; only boolean `true` enables a view. Bakeoff also forwards these
evidence options and the EVO selector into its language lanes.
Forwarding establishes configuration support, not actual evidence reads or gains.

## Question-linked development

Optional `evolution_question_state: true` links the existing memory ledger to
retained source. A ledger row can add `question_id`, `next_question`, `source_ids`,
`evidence_refs`, `next_experiment`, `reopen_if`, and `development_tip_id`. The ID
stays stable when revisiting the same question. Each update supplies its complete
current interpretation; omitted fields remain unknown.

`revision` is an append sequence across the archive's questions, not a separate
counter for each question.

The archive retains these rows unchanged in append-only `question_revisions`,
alongside source references and their status at the time of linking. The planner
receives a compact latest-question index and may set `DIRECTION.question_id` to
carry that exact revision to the engineer. Full records remain in the archive
when prompt context is abbreviated. Unknown question IDs are reported explicitly;
the workflow does not substitute another question.

A development tip is an agent-nominated source for further work, including a
slower or unverified attempt. It does not change `best_id`, choose the next parent,
or qualify a measurement. Unknown source references and absent patches remain
visible. Evidence paths and applicability are author claims until checked; the
linker does not open them or infer whether a hypothesis is true. The option is
off by default and adds no model phase or evaluation.
At first-stall activation, existing question rows are copied with
`source_scope: pre_archive_history`. Their source IDs remain unresolved, even
when an imported source has the same ID: older planner IDs and `ORIGINAL` can
refer to another source namespace. The agent must inspect and relink them in a
new revision. Question context follows the existing retained-source visibility
rule; an empty archive does not add evolutionary guidance to the initial plan.

When this option is enabled for a publishing population lineage, each immutable
checkpoint context includes the `question_revisions` available at publication.
`question_state_scope` records the origin lineage, checkpoint round, archive and
last linked round. Publication precedes the current round's memory update: a
checkpoint does not include questions written afterwards and is never refreshed
to add them. Question identity includes the origin lineage, checkpoint,
`question_id` and revision; a peer's local IDs must not be treated as local sources.
The context can be read through its existing `reported_context_ref`. Questions
remain author interpretations, and their linked sources and evidence are
unavailable to peers unless separately shared. Qualification of the checkpoint
does not qualify those other sources.

## Native observations and reported rows

For checked external results, `observed_evidence` is built only from the trusted
reader response. It retains available source/evaluation identity, patch/receipt
references, timing basis, objective, phase metrics, correctness and case rows.
Unavailable fields are null. In particular, missing reader case rows remain null;
an empty returned array remains empty. `case_rows` describes supplied reader rows,
not expected task coverage or per-case correctness that the reader did not supply.
The original model's case rows are retained separately as `reported_per_case`.
Neither a source hash nor the aggregate validity flag independently validates
those reported rows. Invalid external reads have null `observed_evidence`.
The reader also owns native acceptance fields such as status, correctness and
reproduction flags; missing values cannot be inherited from a model report.
Ordinary role results cannot supply the reserved host-provenance field.

This preserves existing reader facts; it does not add a manifest join, infer
shape/regime/counts, or compute roofline utilization. Ordinary-lane reports keep
their existing authority. Gate formulas and task thresholds are unchanged, but
native qualification/export now requires the reader to supply the facts they
consume.
