# Update Experience

After final independent validation, propose at most one reusable lesson from
WINNER.speedup and the final report. Read LEARNED_DIR/README.md. SCOPE=lane teaches
a measured method; SCOPE=bakeoff may teach backend selection from CANDIDATES on the
same frozen baseline. A one-off number or already-covered idea warrants skipped.

LEARNED_DIR is the only knowledge sink. Do not edit cards, counters or INDEX.md
there directly, and never write e2e_workflow/knowledge/learned/. Create proposal
files under EVAL_DIR; kb.py propose writes the inbox and the operator's drain is
the only card/index writer. Do not claim an end-to-end gain from isolated timing.

Read the index and allowed current cards to check overlap by meaning. Merge by
reusing the exact existing key; do not create near-duplicates. With
CURATION_ISOLATE=true or empty PERF_KNOWLEDGE_DIR, do not mine CDNA/Instinct or
operator SOTA cards; restrict index/card reads to the allowed platforms. When
REQUIRED_PLATFORMS is nonempty, copy those exact platforms into the proposal.

Preserve useful ideas: cards add options, cautions say what to verify, and one
failure is not a general prohibition. Distill one principle with exact source/run
provenance, applicability, per-case relative evidence and observed pitfalls
(symptom, cause, fix). Keep absolute timings, throughput, clock and power numbers
in EVAL_DIR. Cards use ratios/percent deltas; a roof fraction requires a supported
denominator, otherwise leave it unknown. Keep shapes, dtypes and relevant knobs.

For stacked wins report the final verified total, then each observed incremental
contribution in landing order, with its round and verified/claimed status. Do not
invent standalone attribution. HISTORY and CITATIONS are context; the citation
ledger and drain own attempts, confirms_cited, confirms_blind and losses.

Use the README card schema and kb.py lint. Include title, name, description
(<=160 chars), keywords, kernels, platforms, kernel_class, regime, key, type,
confidence, effect, source, last_seen and attempts>=1. Choose existing vocabulary
where it fits. The plain-English key distinguishes operation, architecture,
framework/dtype/regime rather than repeating a bare class/gfx/regime triple.
Use optional levers/cost/roofline/verified_on only with evidence; verified_on is
this run's date only if its A/B confirmed the effect. Set confirms_cited=0 and
confirms_blind=0 in every new proposal. Do not claim three-star standing from
self-confirmation; respect the README's evidence tiers and lint's cap.

Write EVAL_DIR/kb_proposal.json using the real run identity/date/kernel_names,
accepted validation status, box_quiet and held_out status, citations:[], and
cards:[<one card>]. Skip if validation is not accepted, the box was contended,
the run was held out, or the evidence cannot support a reusable claim. Do not
invent admission fields to make lint pass.
```bash
python3 "$SKILL_DIR/scripts/kb.py" --kb-dir "$LEARNED_DIR" lint --file "$EVAL_DIR/kb_proposal.json"
python3 "$SKILL_DIR/scripts/kb.py" --kb-dir "$LEARNED_DIR" propose --file "$EVAL_DIR/kb_proposal.json"
```
Fix only the proposal when rejected, or skip honestly. Do not drain, regenerate
the index, archive existing cards or change their counters. A successful propose
queues a request; it does not create or merge a published card. Return JSON with
action:"created" or "merged" to identify the requested operation, card_path:""
until a published path exists, key, and a note explicitly saying the proposal was
queued with its actual receipt/path. For no proposal use action:"skipped".

```json
{
  "action": "created | merged | skipped",
  "card_path": "path under knowledge/learned/, or \"\" if nothing distilled",
  "key": "<the card's plain-English key line, e.g. 'MXFP8 E8M0 dense linear, decode-bound · gfx950'>",
  "note": "one line: what was distilled or why nothing was"
}
```
