# Save programs, ideas and results

Use the existing `ResearchStore`. It keeps exact source files in `objects/` and
appends checked history in `records/`. The index is a cache and can be rebuilt.
Do not maintain a second best-program log by hand.

## What to save

The native host snapshots source before evaluation. At engineer settlement, the
collector supplies its retained trials, including slower, failed and unfinished
implementations. Missing snapshots stay explicit errors. An unfinished episode
may not yet have a complete trial index.

| Record | Required meaning |
| --- | --- |
| Source | SHA-256 identity of exact files, declared absences, layout and executable bits |
| Candidate | Source plus task, measurement context, assigned parent and donors |
| Attempt | Host-assigned engineer and operation; do not infer authorship from prose |
| Idea | Stable idea ID, intended change and question; separate from implementation success |
| Evaluation | Exact source, cases, raw evidence, correctness and measured results or failure |
| Hardware claim | Existing `ClaimLog` reference with quantities, units, assumptions and evidence |
| Next test | Smallest useful change or observation needed to continue the idea |

Engineer notes should answer five short questions: what changes, why it might
help, what hardware observation would support it, what the test actually showed,
and what to try next. Link existing evidence instead of copying logs. Record
unknowns explicitly. A slow implementation does not by itself refute an idea.
Keep the original question separate from the implementation report. Parent and
donor links describe source ancestry; they do not prove a useful contribution.

## Browse a saved store

From the repository root, export to a new directory:

```sh
python3 -m geak_evo.archive.export --store /path/to/store --out /path/to/view
```

If idea cards reference compiler or diagnostic records, also pass
`--frozen-cases cases.json`: the host's `{job_id: [case_id, ...]}` registry.
Missing required case bindings fail clearly; they are not inferred from a claim.

```text
view/
  index.csv, index.json       programs and their recorded status
  programs/<order>-<id>/      program/source manifests and exact source/
  ideas/                     idea cards and revisions
  attempts/                  engineer, parent and saved implemented-strategy fields
  directions/                current saved review and next test for each direction
  claims/, claims.json       exact committed claims and corrections, with their chain head
  measurements.csv           one row per evaluation/trial; per-case times inside
  observations/              exact saved measurements and their task/source inputs
  trials.json                all supplied trials, including retention errors
  trial-history.json          complete shared trial view, grouped by task/epoch/source
  timeline.json              changes to the host's selected program
  lineage.json               parent and donor links
  RESUME.json                saved directions and restoration limits
  EXPORT.json                journal identity and exported file hashes/modes
```

Start with `index.csv`, then open a program's `program.json`. Its `research_files`
links lead to the investigation, current direction review and program-specific
claims. Use `measurements.csv` and `observations/` for the measured results.
`trial-history.json` is the same derived view used by the planner and parent
handoff. It lists every supplied trial, including repeated measurements of the
same source, and links exact inventory, outcome and feedback records. Query its
`groups` by task, measurement epoch and source, then follow the `trial_ids`.
These are view row IDs: newly discovered links can change the canonical group.
Keep `inventory_refs` for stable references to exact immutable input rows; source,
candidate, operation and evaluation identities retain their original meanings.
Only explicitly linked identical inventories are deduplicated. Unlinked episodes
remain separate, even if their bytes match.
The view counts retained trial rows in any state and reports coverage; work absent from a collected
inventory remains unknown. A missing row does not mean an idea was never tried.

Request identity, reported admission, correctness and benchmark stages stay
separate. Unknown stages stay null; error prose does not fill them. Timing
conflicts suppress displayed values while retaining the conflicting references.
Recorded timing is a development observation: no new speedup, winner, or verdict
on an idea is calculated. Stage fields come only from structured saved feedback
with matching source, operation and turn. Generic status words remain separate.

This view and its per-parent counts are visible to the planner and engineer.
They change the evidence presentation and therefore belong only in prospectively
pinned runs. They must not be inserted into a frozen study. The projection reads
needed metadata and linked feedback rather than eagerly loading every raw build
or trace; existing source and ancestry validation still applies.

Check the program/source IDs: a direction review describes its current tip, and
an investigation's strategy need not describe every intermediate trial.

Missing explanations stay null. Long strategies have a marked 4,096-character
preview plus the full exact UTF-8 text in `attempts/<hash>.strategy.txt`, its
SHA-256, and the original artifact/field reference. Claims and corrections come
from the saved campaign head. Strategies, direction reviews, claims and
corrections are attributed statements, not validation; blinded reviewers must
exclude all of them as required by their study's review policy.
Conversations and bulk reports are not copied.

Export one campaign at a time to a new directory for each journal snapshot.
The same snapshot produces identical files and modes; stale or changed output
is rejected. Current direction revisions must retain `archive_card_ref` for the
current idea link; older cards remain in the store.

Journal timestamps record saving, not discovery or measurement. `timeline.json`
shows host choices, not a newly calculated best-speedup curve. Mixed test outcomes
remain separate; failed or incomparable measurements do not become improvements.

## Continue an idea

Restore the candidate's exact source and read its linked research and trials.
A reviewed intermediate trial can start a fresh direction through the host's
parent checks. Retention alone does not promote it to a verified best result.

The export is portable research material, not a complete live-run backup. Keep
the original store and pinned task, harness, runtime and evidence files. Full
resume needs those dependencies and compatible host services; never rewrite
historical pins to make a changed runtime appear equivalent.
