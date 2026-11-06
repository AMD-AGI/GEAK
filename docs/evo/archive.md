# Inspect retained candidates

GEAK-EVO keeps its population under `<eval_dir>/evolution/`:

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

The archive is a per-run record, without a separate exporter, semantic diversity
index, or cross-run resume protocol. An ancestry link records a selection; useful
contribution survival and later improvement require inspection and measurement.

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
without coercion; only boolean `true` enables a view. Bakeoff remains unsupported
for these options: its explicit mapping also does not forward the EVO selector.
Forwarding establishes configuration support, not actual evidence reads or gains.

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
consume. Use this authority repair in both arms of a future presentation ablation.
