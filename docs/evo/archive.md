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
The view is omitted if its JSON exceeds 16,384 UTF-8 bytes. This option defaults
to false and does not change stored archive rows or promotion rules.

The archive is a per-run record, without a separate exporter, semantic diversity
index, or cross-run resume protocol. An ancestry link records a selection; useful
contribution survival and later improvement require inspection and measurement.

## Optional peer sources

EVO accepts `evolution_peers: [{id, root, native_epoch}]`; the default is `[]`,
as required for GEAK and exchange OFF. The existing owner mounts each peer's
ordinary evolution directory read-only at the declared absolute `root` and
supplies its native measurement epoch. Composed recipients also declare
`policy.imports: "selected"`. Declared peer sources are independent of the
composition's finalist members, so a one-member composition can use an external
EVO archive. The owner enforces read-only mounts and private execution contexts;
path checks do not establish that isolation. The owner may preallocate an empty
evolution directory for the mount. The workflow initializes it in place and
refuses prior contents. This adds no publisher or GPU service.

Each planning round saves exact peer `archive.json` bytes and their hashes in
that round's artifacts. Replans reuse this snapshot. If parsing a peer fails, it
contributes no offers; other peers remain available. Eligibility uses the archive's existing
`verified` verdict, a retained patch and a native evaluation reference, without
reinterpreting copied verdict fields. Peer import currently requires an external
native objective; other runs supply no eligible native source offers. Only selected parent/donor
patches are reconstructed and copied. Both actual run bases must match the same
task's protected ORIGINAL, including executable modes; Git commit labels may
differ. Imported rows keep publisher identity, ancestry, snapshot and foreign
measurement references under `origin`, with `verified: false` and `speedup: null`.
Recipient native verification is required before a descendant can be promoted.
Missing snapshots leave local search available with visible errors; a bad
selected source fails its direction. Archive snapshots include existing notes,
even with inline evidence handoff off: this is source reuse with archive-context
access, not an experiment isolating source bytes from context.
