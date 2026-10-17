# Save an engineer's report beside each trial

`patches/report-context.patch` adds optional pre-evaluation report capture to the
existing external provider. It patches the producer and harvester together; it
does not copy the provider into GEAK. `SOURCES.json` pins the exact originals,
patch and resulting files. This branch builds on `evo` descendant `a6a48fc8`.

Apply the patch to a fresh copy of the listed original provider sources:

```sh
git apply --check -p1 /path/to/report-context.patch
git apply -p1 /path/to/report-context.patch
```

Verify the before/after hashes. The new `workspace_helpers/report_context.py`,
both changed provider files and the GEAK commit must enter a new authenticated
source graph and installation closure. Keep every unchanged provider file from
the pinned original source map. No frozen source, existing installation or run
clock is changed by this source package. No installer or launcher is supplied.

Enable capture in the protected native `probe_config` template:

```json
{"report_context": {"capture": "report-v1", "delivery": "history-v1"}}
```

Omit the field, or set both values to `off`, to disable it. This version supports
delivered history only. It rejects private capture-only mode: the existing
feedback, archive and read-grant paths have not been qualified to hide captures
from every search model. Delivery changes available model inputs and capture
adds time to the existing budget. Neither is claimed to be behavior-neutral.

Report-writing guidance is independent. In the protected engineer entry template,
set `report_update_instruction` to `before-trial-v1` to ask for a concise update
before submission. Omit it or use `off` for the existing prompt. This is a separate
model-input treatment; enabling storage does not enable the instruction or ensure
useful notes. The instruction uses the existing `report.md`, with no new form.

The host derives `.idea-output/<turn>/report.md` from its admitted workspace and
turn. After saving the source and before evaluating it, the host saves up to
256 KiB of complete, stable UTF-8. Missing, empty, oversized, unsafe or changing
files have explicit availability. Oversized files are not read in full or hashed;
their full-content digest remains unknown. Identical report hashes mean identical
bytes, not that an idea is stale or relevant to a changed source.
Retained context exposes `report_sha256`, `capture_monotonic_ns` and `boot_id`
for this comparison. Compare captures within the same turn and boot using their
recorded time; file-list or request-name order is not a capture order. Equal
hashes are repeated text, not additional ideas. Missing or ambiguous ordering
must remain unknown; this package does not invent a previous-capture link.

Source, candidate, operation, job, epoch and owning turn come from the host.
The `probe_request_sha256` field identifies the probe request, not the different
Arena request. Notes are `narrator_text`: stated context observed at this trial
boundary, not proof of why an edit was made or of a hardware or timing claim.
The source and native result remain separate evidence.

The original capture pointer is immutable in host request storage. Same-ID
reconciliation uses it, or records unavailable capture; it never rereads a later
report and labels it pre-evaluation. Feedback carries a separate optional
`context_refs` envelope. Its exact feedback ref is already bound by the native
receipt and the outer collection intent, so a collector retry inherits the same
capture without searching the store or starting another evaluation.
If publication stops after writing context files but before feedback, a retry
checks and reuses the original files and writes only missing ones. Different
existing bytes remain an explicit optional-context error; they are not replaced.
For missingness reports, inspect `context_refs.error` as well as `availability`:
`capture_unavailable` can mean a publication conflict, not only a missing capture.

Optional files stay outside `artifact_index` and native validity checks. The
harvester catches note failures separately and adds only retained links to the
existing archive navigation after native checks. Normal archive/read grants can
then deliver the original note to another engineer. Full note bodies are not
inserted into the default prompt. The compact export does not gain a new report
viewer or bulk-report export through this change.

Independent researcher release remains a separate study decision: prescribe
population, cutoff, narrator-text inclusion, masking, purpose and exact refs
before a reader receives these records. Search-model delivery does not authorize
revealing narration to a masked rater.

The CPU fixtures replay the exact external patch, test the producer/read helper
and follow the normal public archive path after collector interruption. Process
execution and native measurement are synthetic. A fresh installed-source
qualification is still required; no performance benefit or live recovery is
established by this package.
