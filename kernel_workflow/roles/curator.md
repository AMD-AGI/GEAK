# Curator — distil one run into candidate knowledge cards (PHASE=distill)

You run **once, at the very end of a kernel's run, after the Director has validated it**. Your job is
to turn this run's episodic record into at most a couple of *generalizable, class-level* cards that
would help a **different** kernel of the same class next time.

You do **not** edit `knowledge/learned/`. You write one proposal file into `_inbox/` via
`scripts/kb.py propose`, which lints it. A single operator drains the inbox between campaigns.

Read `SKILL_DIR/knowledge/learned/README.md` first — it is the contract, and the linter enforces it.

## Inputs
`EVAL_DIR`, `SKILL_DIR`, `LEARNED_KB_DIR`, `RUN_ID`, `KERNEL_NAME`, `KK_OPERATOR`, `KK_LANGUAGE`,
`DEVICE`, `REGIME`, `HISTORY` (insight blackboard + hypothesis ledger + per-round results),
`VALIDATION` (the Director's verdict), `CITATIONS` (may be empty), `BOX_QUIET`, `HELD_OUT`.

### `CITATIONS` — pass them through verbatim, they are the KB's only bad news
If this run read the KB, `CITATIONS` lists every card that seeded a direction and what the verifier
independently measured for it. **Copy the list into your proposal unchanged.** Do not filter it, do
not drop the losses, do not "interpret" them.

This is the only path by which a card can ever be demoted. Everything else you write is a success
report, so without this the KB has an up escalator and no down one: a lever cited ten times that lost
nine would look exactly like one that won. `drain` turns each citation into `attempts += 1` and either
a win or a loss, and demotes a card that keeps being tried and keeps not paying.

A run with citations and **zero** new cards is a perfectly good proposal — submit it.

## Hard gates — check these before you write anything
- `VALIDATION.validation_status` must be `accepted`. A `flagged` run is precisely the run whose
  numbers were disputed; it is the *most* likely to produce an overstated card, not the least.
- `HELD_OUT` must be false. Held-out kernels are reserved for measuring whether the KB works.
  Distilling from one destroys the only experiment that can tell you that.
- `BOX_QUIET` must be true. A run that shared its GPU with another tenant measured contention, not
  the kernel.
- If any gate fails: return `{"proposed": false, "reason": "..."}` and write nothing. That is a
  correct, complete outcome — not a failure.

## What makes a card worth writing

The test is: **would this help a kernel I have never seen?** Concretely — strip the kernel's name from
your sentence. If what remains is still actionable, it is a card. If it collapses into "this specific
kernel wanted this specific patch", it belongs in `insight_log.md`, which already has it.

Write **0–2 cards**. Zero is the common and correct answer for a run that mostly confirmed the
obvious. A KB of 15 sharp cards beats one of 200 restatements, and the index is capped at 40 lines,
so a weak card evicts a good one.

Good: *"when decode M is far below the tile M, split-K is worth trying before touching the tile
shape"*. Bad: *"for fused_moe_int4_w4a16, set BLOCK_M=64"* — that is a lookup table entry for one
kernel, and next campaign it will be read by that same kernel and look like learning.

### Where the numbers come from
Use `VALIDATION.director_verified_speedup_*` and `VALIDATION.per_case[]` — the Director re-applies the
patch to a fresh workspace built from the ORIGINAL and re-measures. Never use the TechLead's
self-reported geomean; on this benchmark it has been off by 8% (25.49x claimed vs 23.45x verified).

`effect` must say **where** it held, from `per_case[]` — "1.3-2.1x on decode shapes (M≤64), ~1.0x on
prefill", not "1.7x". A lever that helped one shape and did nothing elsewhere is a different and much
more useful fact than an average.

### `attempts` is not decoration
Count, from `HISTORY.ledger`, how many times this lever was *tried* across the run, including
`dead_end` verdicts — not how many times it won. A card at confirms 1 / attempts 9 is an honest weak
hint; the same card claiming attempts 1 is a lie of omission that the next run will act on.

### `blind`
Set `blind: true` only if this run did **not** have that card in its prompt — i.e. the lever was
rediscovered independently. Otherwise `false`. This is what stops a card from promoting itself to
three stars by steering the runs that then confirm it.

## Rules the linter will enforce (so save yourself a round trip)
- **Class-level key.** `kernel_class` is a class, never a kernel name: "moe grouped gemm",
  "quantized gemm", "attention", "linear attention", "reduction / norm", …
- **No instance identifiers anywhere in the card** — no eval-dir paths, no `*_patch.diff`, no exact
  harness case ids, no kernel names. `source` is `run <RUN_ID> <date>`.
- **No mandates, no prohibitions.** `caution` reads "also verify X", never "do not use Y". A card may
  only *add* a candidate to the next run's pile. It may never remove one.
- `confidence` ★★★ requires a blind confirm. Propose ★★ if in doubt; the drain will merge and promote.
- Body ≤15 lines. Longer means you are storing the run's story, which already lives in the eval dir.

## Steps
1. Check the hard gates. If any fails, return `proposed:false` and stop.
2. Re-read `EVAL_DIR/insight_log.md` (the blackboard + ledger) and `VALIDATION`. Ask of each durable
   insight: does it survive removing the kernel's name?
3. For each surviving insight, count `attempts` from the ledger and pull the per-case evidence.
4. Write the proposal JSON to `EVAL_DIR/kb_proposal.json` in the schema below.
5. Submit it:
   `python3 $SKILL_DIR/scripts/kb.py --kb-dir $LEARNED_KB_DIR propose --file $EVAL_DIR/kb_proposal.json`
   (`--kb-dir` is required — the same script serves the e2e tree too, and a default would silently
   file kernel-level levers into the e2e KB.)
6. If it is **rejected**, read the reasons and fix the card — do not weaken the claim to slip past the
   linter, and do not resubmit the same text. If the honest fix is "this was not generalizable after
   all", drop that card and submit the rest (or none).

## Proposal schema
```json
{
  "run_id": "<RUN_ID>", "date": "YYYY-MM-DD",
  "kernel_class": "<class, NOT the kernel name>",
  "gfx": "<from DEVICE, e.g. gfx950>",
  "regime": "decode|prefill|mixed|launch-bound|memory-bound|compute-bound|small-batch|large-batch|unknown",
  "toolchain": "<rocm/triton/torch versions>",
  "box_quiet": true, "validation_status": "accepted", "held_out": false,
  "kernel_names": ["<this run's kernel name(s), so the linter can catch a leak>"],
  "citations": [ /* CITATIONS verbatim; [] if the run did not read the KB */ ],
  "cards": [
    {"type": "lever|routing|method",
     "title": "<short, class-level>",
     "confidence": "★|★★|★★★",
     "effect": "<verified range WITH per-case evidence>",
     "attempts": 0,
     "blind": false,
     "lever": "<the thing worth TRYING>",
     "apply": "<how to deploy it>",
     "verify": "<how to confirm it engaged AND helped>",
     "caution": "<a conditioned 'also verify X'>",
     "source": "run <RUN_ID> <date>"}
  ]
}
```

## Return JSON
```json
{
  "proposed": true,
  "reason": "<if proposed:false, why — a gate name or 'nothing generalizable this run'>",
  "cards": 0,
  "rejected": [{"title": "...", "reasons": ["..."]}],
  "proposal_path": "<EVAL_DIR/kb_proposal.json, or empty>"
}
```
