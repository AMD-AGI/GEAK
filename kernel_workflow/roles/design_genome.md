# Design Genome

Use this optional guidance when EVO enables `design_genome_mode: record` or `handoff`.
Authoring guidance is identical in both modes. The task's full numerical/state contract
and native correctness/performance evaluation remain authoritative.

Keep a compact account of the intended computational organization: how required values
are produced, owned, reused or materialized, and the layout, schedule or backend choices
that matter. Explain what changed from the parent and what was adapted from donors in
the actual computation. Choose details useful to this design; no fixed narrative form
or category list is required.

Separate **What we know** — observations from source, correctness, native timing and
relevant compiler/runtime evidence — from **What we believe** — explanations, open
questions and plausible rivals. Link evidence and keep claims within its source, cases
and measurement scope. A memory-bound explanation needs applicable evidence and a
competing explanation; counters alone do not establish causation. Reuse applicable
evidence and collect diagnostics when they would change the next experiment.

Assess whether the intended idea was realized, partially realized, not realized, or
remains unknown, explaining the support and exact gap. Realization is a revisable author
claim, separate from correctness and speed. Do not discard an idea because an attempt
implemented it poorly or unfaithfully. Preserve the intended idea, attempted source and what
the result actually tested. An unrealized intervention leaves its performance effect
untested; one faithful slow implementation establishes only its scoped result, not
exhaustion of the whole approach.

Identify the best validated source and any useful developing alternative. Retain relevant
failed efforts, the unresolved question, and the next repair or experiment with an outcome
that would change the explanation. Continued development may repair, revise, combine or
abandon the idea as evidence develops; retaining a question does not guarantee more budget.

## Returning a record

Engineers, including deep engineers, may add this optional field to their existing result:

```json
"design_genome": {"patch_sha256": "<SHA-256 of submitted patch bytes>", "text": "<narrative>"}
```

Keep `text` within 4096 UTF-8 bytes and put the same narrative in the existing `report.md`.
Use that space for the decision-relevant idea, gap and next experiment; cite full evidence
artifacts instead of copying raw output. Refer to parent/donor history by its existing
direction or checkpoint IDs and source/evaluation references in the prose.
Hash the exact submitted cumulative `best_patch.diff` against the supplied base, not just
the latest edit or a donor patch. The narrative must describe that submitted source;
distinguish other attempts by their own source/evidence references.

An integrator places its own record under `best.design_genome`, hashing the exact file
named by `best.patch_file`. A new child or integrated source needs a freshly authored,
matching account; do not copy a parent's or donor's record onto it. The host supplies
base/source/evidence associations separately. Hashes establish association, not truth.

The carrier is optional. Missing, invalid or unavailable narrative is missing metadata,
not grounds to reject a source or its measurements, change fitness, or make a repair or
retry call for the genome. Preserve useful notes in the ordinary report; that report is
not the authoritative carrier or a substitute for the existing result submission.

## Using a record

Planners, engineers, integrators and memory/report writers should treat available records
as attributed references to inspect alongside their source and evidence. An author calling
a claim independently checked does not make it verified. When a plan uses or rejects a
record, cite its existing direction/checkpoint ID and say whether realization evidence is
author inspection or an independent check, with its artifact reference. Preserve uncertainty and
implementation gaps when summarizing; do not promote an inherited explanation into an
observation. Revise the hypothesis when evidence warrants it and choose the next useful
experiment within the full contract. A missing record means its narrative is unavailable,
not that the source lacks a useful idea or valid evidence.
