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
| `design_genome` | Optional compact author claims associated with the retained source; see below. |
| `design_genome_omitted` | Informational reason an optional author record was omitted; not a source failure. |
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

## Optional Design Genome

With `design_genome_mode: "record"` or `"handoff"`, an engineer may add this
optional field to its existing result; an integrator may add it under `best`:

```json
{
  "design_genome": {
    "patch_sha256": "<SHA-256 of the submitted cumulative best_patch.diff>",
    "text": "<compact author prose about this implementation>"
  }
}
```

The text must be nonempty and at most **4,096 UTF-8 bytes**. There is no required
narrative form or realization state machine. The
[writer guide](../../kernel_workflow/roles/design_genome.md) asks for intended
versus realized computation, observations versus beliefs, implementation gaps,
rivals, failed alternatives, open questions and next experiments. Correctness
and performance remain in the existing evidence fields. A poor or unfaithful
implementation does not by itself refute the intended idea; a faithful but slow
implementation can motivate reformulation without protecting it from normal
budget decisions.

The host preserves valid text verbatim in `design_genome` with
`kind: "author_claims"`, the retained `base_commit` and `patch_sha256`. When the
existing native verifier establishes the saved patch's identity, it also adds
`source_id` and `external_evaluation`. An engineer measurement against another
workspace source prevents that narrative from acquiring the verifier's source
association. A matching hash establishes association with bytes, not the truth
of the claims. Report Markdown is useful context but is not the authoritative
carrier and cannot revise a published record.

Missing, malformed, oversized or mismatched optional metadata is omitted after
normal source recovery. It never sets a retention failure or changes donor
eligibility, fitness or promotion, and the host does not request a repair call.
A new child or integrated candidate needs its own matching author record;
parent and donor prose is never copied into its result automatically.

Committed population checkpoints can carry the selected record. If publication
changes the diff representation, the host carries it only when an already
checked native source identity proves that the retained and published sources
are the same. Otherwise it omits the prose. Query output exposes only sanitized
`record.design_genome`, separately from mandatory source qualification, including
when the source is eligible but its public delivery patch is unavailable. Peer
imports in `handoff` keep the donor's association and checkpoint provenance.
Uncommitted archive entries do not become peer donors.

`record` and `handoff` share writing guidance and retention of their own records.
Both automatically include own-lineage claims in the planner's archive
summaries, the engineer's selected parent/donor
context, and the integrator's selected candidate inputs. `handoff` additionally
copies imported peer claims into local archive rows and exposes them there;
`record` keeps peer rows and prompts at the ordinary projection. Published peer
checkpoints remain available for deliberate retrieval through existing tools in
both modes, so `record` does not guarantee absence of all peer knowledge. The
engineer context omits the genome packet if adding it to any requested evidence packet exceeds
16,384 UTF-8 bytes. Its archive path remains available. This opt-in mode does
not activate `evolution_evidence_handoff`, which controls the older raw evidence
projection separately.

The archive is a per-run record, without a separate exporter, semantic diversity
index, or cross-run resume protocol. An ancestry link records a selection; useful
contribution survival and later improvement require inspection and measurement.
