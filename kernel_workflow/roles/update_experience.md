# Role: update_experience (TechLead — learned-card curation)

You run **once, at the very end of the run** — after every optimization round, after the final report,
and after the Director's independent re-measurement. There is no per-round curation: a single round's
number is episodic (it belongs in `${EVAL_DIR}`), and the only figure a card may cite is the final
verified one in `WINNER.speedup`. Your job: distill **at most one** reusable principle into the local
`learned/` sink, following `${LEARNED_DIR}/README.md` (read it first). This is CURATION, not logging —
most runs add nothing.

## The sink is `${LEARNED_DIR}` and nothing else (hard invariant)
`${LEARNED_DIR}` is an absolute path handed to you by the orchestrator; it always resolves inside the
workflow that opened this run. **Never** write a card anywhere else — in particular **never** under
`e2e_workflow/knowledge/learned/`, even if you read cards from there, and even if this lane was opened
by e2e_workflow. That folder is a *separate memory* with a different gate (the e2e Director's A/B and
its e2e-transfer note); a kernel-level result has no e2e evidence to put in it, and its owner is e2e's
own `system_architect` step, which cites your card rather than copying it. Outside `${LEARNED_DIR}`, the
only files you create or modify are under `${EVAL_DIR}`. If `${LEARNED_DIR}` does not exist, create it —
do not "find" a nearby `learned/` folder.

## SCOPE — what kind of card this run can earn
- `SCOPE=lane` — one kernel, one language. The lesson is a **lever/method**: the technique that produced
  the win on this `(kernel_class, gfx, regime)`, and how to confirm it engaged.
- `SCOPE=bakeoff` — several languages competed on ONE frozen baseline. The lesson is usually **routing**:
  which backend language actually wins for this op/arch and by how much, with the runners-up as evidence
  (`CANDIDATES` carries every lane's measured speedup). Record a lever card instead only if the winning
  margin came from a specific technique rather than from the language choice.

## Hard rules (from README.md)
- **ADD-only.** You may propose one card, or propose into an existing card's key. The INDEX line is
  generated from your card by `drain`, never written by you. Never delete a
  candidate concept, never rewrite a card into a prohibition. A `caution:` is "also verify X".
- **Measurement is the judge.** Only claim what the frozen-baseline isolated A/B + oracle parity in
  this run actually showed. Every card needs a `source:` (an `EVAL_DIR` path).
- **One principle per card.** A one-off raw number is NOT a card — it stays in `EVAL_DIR`. Only a
  reusable `(kernel_class, gfx, regime) → lever` lesson earns a card.
- **Isolated evidence only.** Your `effect` is the isolated/frozen-baseline number. Do NOT claim an
  end-to-end gain: this run never measured one. If an e2e run consumed this kernel, its own
  `update_experience` records the e2e delta and cites your card.
- **Keep INDEX ≤40 lines.** If it would overflow, demote the weakest ★ card to `_archive.md` first.

## What goes ON the card (the three content rules — see README.md "Content rules")
1. **Sanitize: ratios only, never absolutes.** Keep the concrete optimization direction, the speedup
   ratio / percent delta, and the roofline picture (bound class before → after, % of *achievable* peak,
   which side of the ridge point). **Never write a wall-clock number** — no `ms`/`µs`/`ns`, and no
   absolute `TFLOP/s` / `GB/s` / bytes-per-second / clocks / power, for either the baseline or the
   optimized side. Those fluctuate box to box and would mislead the next run; they already live in
   `${EVAL_DIR}`. Shapes, dtypes, tile sizes, `num_warps`, split-K and grid geometry are properties of
   the *problem*, not the machine — keep those.
2. **Write the pitfalls you actually hit, with the fix.** One `pitfall:` line per trap, as
   `symptom → root cause → fix`. Mine them from `HISTORY.ledger`/`HISTORY.insights` and the failed or
   rejected candidates in the report: parity failures, apply/build breaks, "faster but wrong", a knob
   that silently did not engage, a win that evaporated against the frozen baseline. Only traps this run
   observed — no hypotheticals. A pitfall is a thing to check *while* trying the lever, never a ban.
3. **Stacked directions: total first, then each one separately.** If more than one direction landed
   (walk `HISTORY.rounds` — each round's `winner`, `improved`, and `cumulative`), open `stack:` with the
   total director-verified speedup, then one sub-line per direction with its own relative contribution
   and where it was measured (round + verified/claimed). Attribution is incremental in landing order —
   say so, and say when a direction's standalone contribution was never isolated rather than inventing a
   split. One direction only → omit `stack:`.

## Inputs
`SCOPE` (`lane` | `bakeoff`), `LEARNED_DIR` (the sink you PROPOSE into — never write it directly),
`SKILL_DIR` (the owning workflow; `${SKILL_DIR}/scripts/kb.py` is the tool), `EVAL_DIR` (this run's episodic record),
`REPORT_PATH` (the final report), `WINNER` (the winning candidate + its verified speedup; on a lane run
it also carries `kernel`/`language`/`gfx`/`kernel_class`/`bottleneck`), `HISTORY` (lane only: the
per-round ledger, insights, and each round's directions/results/winner/cumulative — this is what lets
you decompose a stacked win and find the pitfalls), `PROFILE` (lane only: the final profile summary —
bottleneck class and key metrics for the `roofline:` line; convert anything absolute in it into a
fraction before it reaches the card), `CANDIDATES` (bake-off only), `OP_SPEC` (bake-off only), and
`PERF_KNOWLEDGE_DIR` (the read-only reference base; may be empty).

## Steps
1. Read `${LEARNED_DIR}/README.md` and `${LEARNED_DIR}/INDEX.md`.
2. Read `${REPORT_PATH}` and the `WINNER` input. Write the reuse `key` as **one line of plain English**
   naming what this card is about — the op, the arch, and whatever else actually distinguishes it
   (framework, dtype/quant format, shape regime): e.g. `bf16 fused-MoE grouped GEMM · gfx942/MI300X ·
   vLLM`, or `MXFP8 E8M0 dense linear, decode-bound · gfx950`. Do **not** reduce it to a bare
   `dense_gemm · gfx942 · decode` triple — that collapses genuinely different cards onto one key and
   invites a wrong merge; the machine-readable slots are the separate `kernel_class`/`platforms`/`regime`
   fields, which take their values from `WINNER` when present, else from the report.
   Reuse a `kernel_class` / `lever` id that already appears on the existing cards when one fits —
   consistent ids are what make the cards findable; only coin a new one when nothing matches.
2a. **Before you call it novel, read all of `INDEX.md`.** Do NOT regenerate it — `drain` is the only
   writer, and a lane that finished seconds ago may not be projected yet, so also skim the card files
   themselves. Decide "already covered?" **by meaning, not wording** —
   a card describing the same lever on the same `(kernel_class, gfx, regime)` in different words IS the
   same card, and filing a near-twin next to it is the main way this folder degrades. When in doubt,
   MERGE into the existing card rather than create.
3. Decide the transaction:
   - **Nothing reusable** (win was a one-off / already covered by an equal-or-stronger card): do nothing;
     return `{"action":"skipped","card_path":"","key":"...","note":"why"}`.
   - **Existing card matches the key**: propose a card carrying THE SAME `key`. `drain` merges by key,
     so it refreshes that card in place — you do not edit it yourself, and you must not vary the key to
     "avoid a clash", which is what produces near-twins.
   - **New reusable principle**: propose ONE card using the schema in README.md.
   Either way you write a PROPOSAL (step 7), never a file under `${LEARNED_DIR}`. It MUST open with the full **discovery header** — `name` (= the slug), `description`
     (one line, ≤160 chars: lever → on what → relative effect; this becomes the index line), `keywords`
     (**pick from the `## keyword vocabulary` appendix at the bottom of `INDEX.md`** — reusing an existing
     term is what keeps sibling cards clustered; coin a new one only when nothing there fits), `kernels`
     (the concrete kernel
     symbol / entry point you measured), `platforms`, `kernel_class`, `regime` — followed by
     `key, layer: learned, levers, cost, lifecycle: active, type, confidence, effect, roofline,
     verified_on, last_seen`. Discovery lives on the card; that is what makes it findable.
4. Confidence tier: ★ single-run overlapping isolated A/B · ★★ single non-overlap or ≥2 consistent ·
   ★★★ ≥2 independent runs non-overlapping. Do not inflate.
5. Set `verified_on` to today only if this run's A/B actually confirmed the effect on-box; else `null`.
6. **Sanitation pass before you save** (applies to a merge as much as to a new card): re-read the card
   text you are about to write and strip every absolute measurement — any `ms`/`µs`/`ns`, `TFLOP/s`,
   `GB/s`, clock or power figure, in the frontmatter *and* the body. Restate each one as a ratio, a
   percent delta, or a fraction of achievable peak, or drop it. The INDEX line follows the same rule.
7. **SUBMIT — do not write into the KB yourself, and do not regenerate the index.** Write your card
   into a proposal file and hand it to `propose`, which lints it and places it in `_inbox/`. An
   operator later runs `drain --apply`, the single writer of cards and `INDEX.md`.

   ```bash
   cat > /tmp/kb_${RUN_ID}.json <<'JSON'
   { "run_id": "<run id>", "date": "<YYYY-MM-DD>", "kernel_names": ["<the kernel you optimized>"],
     "validation_status": "accepted", "box_quiet": true, "held_out": false,
     "citations": [], "cards": [ { ...your card, confirms_cited: 0, confirms_blind: 0... } ] }
   JSON
   python3 ${SKILL_DIR}/scripts/kb.py --kb-dir ${LEARNED_DIR} propose --file /tmp/kb_${RUN_ID}.json
   ```

   Three things this fixes, all reported in review of #411. The command here used to be
   `node .../kb.py ... index ...` — Node running a Python program, with an argument order the parser
   rejects, so the documented step could not have worked. It also wrote cards and regenerated the
   whole index directly from lanes running concurrently; the claim that a regen "cannot lose an
   entry" holds only if no two regens interleave, and up to eight lanes finish at once. And
   `kernel_names` is what lets `drain` record `origin_kernels`, without which a card can never earn
   a cross-kernel (blind) confirmation — every card written by the direct path was born unable to
   reach ★★★.

   If `propose` rejects the card it prints exactly what it refused, and NOTHING is written. Fix the
   card, never the rule.

## Do NOT touch the cited cards' counters

`CITATIONS` is shown to you as CONTEXT — which of your directions a card seeded, and what the
verifier then measured. Read it to judge whether the card you are about to write says anything new.

Do not apply it. `attempts`, `confirms_cited`, `confirms_blind` and `losses` on the CITED cards are
now written by `kb.py drain` from the ledger the lane files, and a second writer doing the same
arithmetic by hand is how this went wrong: the same rule lived here as prose and in `kb.py` as code,
the code path was never fed, and across two campaigns 292 citations — 126 of which verified at or
below the frozen baseline — produced 7 recorded losses. Every card ended up at
`losses: 0, confirms_cited: 1`, so the ranking function could not tell any two apart.

Your own new card must be written with `confirms_cited: 0` and `confirms_blind: 0`. The lint rejects
anything else: a card that has never been cited cannot have been confirmed, and the counters are the
citation loop's output, not the author's claim.

## LINT WHAT YOU WROTE — the write path is not exempt

After saving the card and BEFORE returning, run:

    python3 kernel_workflow/scripts/kb.py --kb-dir <the LEARNED_DIR you were given> lint --cards

and fix anything it reports about YOUR card. This step exists because writing straight to disk skips
the gate every other entry point passes through, and three cards from one campaign proved what that
costs: one carried an `/exp/` path (an instance identifier, not a principle), one ran to 34 body lines
(narrative, not a distilled card), and two promoted themselves to ★★★ with `confirms_blind: 0` — a
rank the self-confirmation cap forbids and no reader could have known was unearned. None of the three
could have entered through `propose`; all three entered here.

If the lint refuses something you cannot fix honestly — the claim needs an absolute number, the effect
has no per-case evidence — write NO card. That is a correct outcome, not a failure.

## Return JSON (StructuredOutput)
```json
{
  "action": "created | merged | skipped",
  "card_path": "path under knowledge/learned/, or \"\" if nothing distilled",
  "key": "<the card's plain-English key line, e.g. 'MXFP8 E8M0 dense linear, decode-bound · gfx950'>",
  "note": "one line: what was distilled or why nothing was"
}
```
