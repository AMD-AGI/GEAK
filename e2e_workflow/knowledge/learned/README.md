# `learned/` — distilled experience cards (an ADVISORY aid, read via INDEX.md)

This folder is GEAK's persistent, curated optimization experience. It replaces the old append-only
"Learned" sections (which grew to 500+ lines of near-duplicate run narratives).

## Philosophy — the KB is an accelerant, NOT a crutch and NOT a cage
The e2e_workflow is fully capable **without** this KB — cold runs (before any KB existed) reached ~20%
e2e. So the KB's only job is to help a good run **converge faster / go further**. It must **never make a
capable run worse by boxing it in or steering it down one path.** The judge is always **on-box
measurement** (the immutable op unittest oracle + the e2e Director's independent A/B + parity) — never
the KB. If a card and the measurement disagree, the measurement wins and the card gets corrected.

**Two-tier memory — keep them separate:**
- **Here (persistent)** = a small set of *distilled, advisory priors* with measured evidence. Bounded, curated.
- **In the eval-dir (episodic)** = the raw per-run story (`final_report.md`, `insight_log.md`). Every
  measurement, including NULLs, lives there. Do **not** copy run narratives here.

## How to USE it during a run (read path) — three hard rules
**Read the KB AFTER you have formed your own profile-driven plan**, as a cross-check and a source of
*extra* ideas — then:
1. **ADD-only, never filter.** Cards may only *add* candidates/levers to try. They must **never** remove
   a candidate, prune the bake-off, or skip the author/measurement step. Whatever the profile says to
   try, you still try — the card just adds more to the pile.
2. **Measurement is always the judge.** Run the full bake-off + author + e2e gate regardless of what any
   card claims. A card is a hint about where to *look first*, not a verdict. Disagreement → trust the
   box, fix the card.
3. **No card may foreclose an approach.** A `caution:` line is "**also verify X**", never "don't do Y".
   The workflow must stay free to rediscover — and beat — any prior. A past winner is a starting point,
   not a ceiling; a past pitfall is a thing to double-check, not a banned move.

Use the index to *find relevant cards fast*, not to decide the answer:
- Read `INDEX.md`, open the cards whose key matches `(kernel_class, gfx, regime)`.
- Treat their `lever`/`effect` as **priors to seed your candidate set**, and `caution` as **extra checks**.

## Card schema (one principle per file, ~10–15 lines)
```
---
key: <kernel_class> · <gfx> · <regime>      # the cross-model REUSE KEY (for finding the card)
type: routing | lever | method
confidence: ★ | ★★ | ★★★                    # how often it REPRODUCED (a hint strength, not authority)
effect: <iso x range; AND the e2e-transfer note — did it actually move e2e, or only isolated?>
last_seen: YYYY-MM-DD
---
# <short title>
- lever: <an actionable thing worth TRYING (a seed candidate), not a mandate>
- apply: <how to deploy / the rebind seam / env var>
- verify: <how to confirm it engaged + that it helped e2e (not just isolated)>
- caution: <a CONDITIONED "also verify X" — e.g. "on decode-bound serving, host-heavy rewrites have
            regressed e2e despite a big isolated win; check the e2e gate". NEVER a blanket prohibition.>
- source: <eval_dir path | arXiv | repo@path>   # REQUIRED — no claim without evidence
```

### Confidence tiers (a HINT strength, not an authority level)
- ★   = single run, distributions overlapped (≈ noise / unverified) — weak hint.
- ★★  = single-run non-overlapping, OR ≥2 consistent runs.
- ★★★ = ≥2 independent runs non-overlapping, OR Director-verified e2e.

## How to UPDATE it after a run (write path) — CURATE, never blind-append
Owners: System Architect (routing/method cards) and Op Benchmarker (head GEMM/attn cards). One transaction:
1. **Read INDEX.md.** Find the card whose `key` matches your finding.
2. **MERGE if it exists** — bump `confidence` if it reproduced, widen/correct `effect` (esp. the
   e2e-transfer note), append a `source`, update `last_seen`. Update its INDEX line. Don't create a
   second card for the same key, and never add a new `## Learned — <date>` header.
3. **INSERT only if novel AND effective (≥★★).** New card + ONE INDEX line.
4. **NULL / overlapping / unverified → write NOTHING here** (eval-dir report only).
5. **A surprising negative → a CONDITIONED `caution:` line** on the relevant card (with the condition it
   held under + its source), framed as "also verify". A claim *contradicted* by new evidence → move the
   card to `_archive.md` with the refuting source. **Never write a blocklist / "never use X".**
6. **Enforce the budget.** INDEX.md ≤ 40 card lines. Over → evict lowest `confidence × freshness` (its
   card → `_archive.md`). ★★★ is never auto-evicted.

**Invariant:** a principle "exists" iff it has a line in INDEX.md (the single source of truth + size
gate). Keep cards short: >15 lines means you're storing narrative, not a principle — distill it.
**Above all: a card is advice the box can overrule, not a rule that overrules the box.**
