# Save programs, ideas and results

Use the existing `ResearchStore`. It keeps exact source files in `objects/` and
appends checked history in `records/`. The index is a cache and can be rebuilt.
Do not maintain a second best-program log by hand.

## What to save

The native host captures source before evaluation and retains the evaluation output.
At engineer settlement, the native collector supplies every trial it captured.
The engine registers each exact source, including slower, failed and unfinished
implementations. A missing snapshot stays an explicit retention error.

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

The native evaluation path already snapshots before testing. The outer trial
index is populated after the engineer settles. It cannot recover a source that
an external collector never saved, and it does not guarantee a complete view of
an unfinished episode. Ordinary GEAK's old best-patch memory is not this archive.

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
  timeline.json              changes to the host's selected program
  lineage.json               parent and donor links
  RESUME.json                saved directions and restoration limits
  EXPORT.json                journal identity and exported file hashes/modes
```

Export one campaign store at a time. Current direction revisions must carry
forward their `archive_card_ref` to retain the current idea link in this view;
older cards and raw evidence remain in the store.

This view is derived from committed records. The same store snapshot produces
the same files and modes; an altered or stale destination is rejected. Use a new output
directory for a newer journal. A journal timestamp is when a record was saved,
not when code was discovered or measured. The timeline copies host choices;
it does not rank incomparable measurements or invent a speedup for a failed test.
Several outcomes saved in one checkpoint remain separate; a mixed result cannot
hide a failed test behind a passing one.

Each program and idea links to its investigation and current direction review.
An investigation's strategy names the source reported by the collector; it does
not describe every intermediate trial. Missing explanations stay null. Long
strategy fields use an explicitly marked 4,096-character prefix and retain the
exact artifact/field reference. The full exact UTF-8 text is in the linked
`attempts/<hash>.strategy.txt` file, with its SHA-256 recorded beside the preview.
No conversation or bulk report is copied.
Direction notes remain attributed interpretations. Claim files preserve exact
records and corrections from the saved campaign head; the export does not
reassess their truth or measurement authority.
The export includes attributed agent statements; blinded reviewers must exclude
those contents as required by their study's review policy.

## Continue an idea

Restore the candidate's exact source and read its idea, attempt and trial records.
A reviewed intermediate trial can start a fresh direction through the host's
normal parent checks. Retaining
a source does not promote it to a verified best result.

The exported source and included observation files are portable; the export is
not a complete live-run backup.
Keep the original store and pinned task, harness, runtime and evidence files.
Full resume still needs those dependencies and compatible host services. Never
rewrite historical pins to make a changed runtime appear equivalent.
