# Warm start — how to use a prior run's record

The knowledge base holds results from earlier runs on this same deployment (model, gfx, serving
framework and version, precision, TP, and workload point). Before your phase started, the
orchestrator read that page and **benched what it found on this box**. You are being handed both
halves: what the store claimed, and what this run measured.

## The one rule

**A stored number is a hypothesis. Only the measured column is evidence.**

Every file in `KB_REFERENCE_DIR/` named `e2e_reference_*.md` records what another box reported —
possibly on a different day, a different ROCm build, and against a different baseline. The file
`measured_on_this_box.md` records what happened when this run applied the same thing here, through
the same gate a fresh idea faces. **Where the two disagree, the measured file wins.** It is not a
correction of the stored record; the disagreement is itself the useful datum, which is why both
files are kept.

Never quote a stored throughput as this deployment's number. The only throughput this run may claim
is one it measured.

## What each outcome means for you

**`adopted` (a configuration).** It is already in your `CURRENT_FLAGS` / `CURRENT_ENV`. It is part of
the starting point, not a proposal. Re-proposing it measures the current state against itself and
burns a server launch to learn nothing. Propose only things that **compound on top of it**.

**`adopted` (a kernel).** It is already in the active overlay, and the profile you are routing from
was captured *with* it applied. That op is done. Its share of GPU time in your Top-N already
reflects the improvement — do not read the reduced percentage as a fresh opportunity.

**`rejected`.** The compounded whole did not beat this baseline on this box. Three things this does
**not** mean:

- it does not mean each knob inside it is worthless — a stored config is applied as one unit, so a
  single regressing axis can sink an otherwise-good set;
- it does not mean the *direction* is wrong — the idea may simply need to be reached differently
  here;
- it does not mean the record was false — it may well have been true on the box that wrote it.

So: **do not re-propose a rejected entry verbatim.** Do feel free to propose one axis out of it, or
the same underlying idea approached another way, as an ordinary candidate that stands on its own
rationale.

**`reference` / `incomplete`.** It was never measured here — because it could not be replayed from a
record, or because the run's replay budget was spent, or because the read was in reference-only
mode. Treat it exactly as a lead from a colleague: worth a look, worth nothing until measured.

## A trap specific to recovered configurations

A flag that a newer framework version renamed or removed is accepted silently on the command line
and then ignored. On the measurement that looks identical to "this config made no difference." If
you propose anything recovered from the store, **verify from the server log that the flag was
actually honoured** before you believe a null result.
