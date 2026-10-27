# Warm start — how to use a prior run's record

The knowledge base holds results from earlier runs. Inspect each record's deployment match
(model, gfx, serving framework and version, precision, TP, and workload point); some matches
relax TP or workload dimensions. The orchestrator may have evaluated some offers before your
phase started. Reference-only mode and a zero replay budget can leave them all unmeasured.
Receiving a record does not mean this run measured or adopted it.

## The one rule

**A stored number is a hypothesis. Only the measured column is evidence.**

Every file in `KB_REFERENCE_DIR/` named `e2e_reference_*.md` records what another box reported —
possibly on a different day, a different ROCm build, and against a different baseline. The file
`measured_on_this_box.md`, when present, records per-entry replay verdicts, including unmeasured
reference or incomplete outcomes. Read each verdict; the file's existence is not measurement
evidence. **Where current-run measurements disagree with stored claims, the current measurements
win.** They do not correct the stored record; the disagreement is itself useful evidence.

If no current-run measurement exists for an offer, treat it as a historical lead. Use
`CURRENT_FLAGS`, `CURRENT_ENV`, the active overlay and the current profile as the starting point;
any proposed change must earn acceptance through the normal measurement gate.

Never quote a stored throughput as this deployment's number. The only throughput this run may claim
is one it measured.

## What each outcome means for you

**`adopted` (a configuration).** It is already in your `CURRENT_FLAGS` / `CURRENT_ENV`. It is part of
the starting point, not a proposal. Re-proposing it measures the current state against itself and
burns a server launch to learn nothing. Propose only things that **compound on top of it**.

**`adopted` (a kernel).** It is already in the active overlay, and the profile you are routing from
was captured *with* it applied. That op is done. Its share of GPU time in your Top-N already
reflects the improvement — do not read the reduced percentage as a fresh opportunity.

**`rejected`.** The entry failed the normal acceptance gate. Read the measured values and reason:
rejection may reflect performance or correctness, and does not by itself establish a throughput
loss. Three things this does **not** mean:

- it does not mean each knob inside it is worthless — a stored config is applied as one unit, so a
  single regressing axis can sink an otherwise-good set;
- it does not mean the *direction* is wrong — the idea may simply need to be reached differently
  here;
- it does not mean the record was false — it may well have been true on the box that wrote it.

So: **do not re-propose a rejected entry verbatim.** Do feel free to propose one axis out of it, or
the same underlying idea approached another way, as an ordinary candidate that stands on its own
rationale.

**`reference`.** No separate replay measurement is recorded for this entry. It may be reference-only,
outside the replay budget, or already covered by another replay; inspect its reason and the current
stack. Do not infer adoption solely from reference status.

**`incomplete`.** No completed usable A/B verdict is available. Partial or failed measurements may
exist; read the reason and artifacts before proposing further work. It is not an accepted result.

## A trap specific to recovered configurations

A flag that a newer framework version renamed or removed may be rejected or silently ignored.
When ignored, the measurement can look identical to "this config made no difference." If
you propose anything recovered from the store, **verify from the server log that the flag was
actually honoured** before you believe a null result.
