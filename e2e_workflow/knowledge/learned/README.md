# `learned/` — distilled experience cards (read via INDEX.md)

This folder is GEAK's **persistent, curated** optimization experience. It replaces the old
append-only "Learned" sections (which grew to 500+ lines of near-duplicate run narratives).

**Two-tier memory — keep them separate:**
- **Here (persistent)** = a small set of *distilled principles*, each one card. Bounded, curated,
  evidence-cited. This is what drives future runs.
- **In the eval-dir (episodic)** = the raw per-run story (`final_report.md`, `insight_log.md`).
  Every measurement, including NULLs, lives there. Do **not** copy run narratives here.

## How to USE it during a run (read path — cheap, index-first)
1. Read `INDEX.md` only. It is grouped by reuse key `kernel_class · gfx`.
2. For each Top-N kernel, find the index lines whose key matches `(kernel_class, gfx, regime)`.
3. Open ONLY those 1–3 cards. Rank bets by `EV = Amdahl_ceiling × confidence`.
4. Honor each card's `dead-end:` lines — don't re-try a scoped regression.

## Card schema (one principle per file, ~10–15 lines)
```
---
key: <kernel_class> · <gfx> · <regime>      # the cross-model REUSE KEY
type: routing | lever | method               # routing=where to optimize; lever=a concrete win; method=reusable technique
confidence: ★ | ★★ | ★★★
effect: <iso x range; e2e %/status>          # measured, not guessed
confirms: <n>                                # independent runs that agree
last_seen: YYYY-MM-DD
---
# <short title>
- lever: <the one-line actionable method>
- apply: <how to deploy / the rebind seam / env var>
- verify: <how to confirm it engaged live>
- dead-end: <scoped negative + the condition under which it bites>   # optional, repeatable
- source: <eval_dir path | arXiv | repo@path>                        # REQUIRED — no claim without evidence
```

### Confidence tiers (this is also the retention/eviction weight)
- ★   = single run, distributions overlapped (≈ noise / unverified). Lowest value.
- ★★  = single-run non-overlapping, OR ≥2 consistent runs.
- ★★★ = ≥2 independent runs non-overlapping, OR Director-verified e2e.

## How to UPDATE it after a run (write path — CURATE, never blind-append)
Owners: System Architect (routing/method cards) and Op Benchmarker (head GEMM/attn lever cards).
Do this as one transaction:
1. **Read INDEX.md.** Find the card whose `key` matches your finding.
2. **MERGE if it exists** — bump `confirms`, raise `confidence` if the new run strengthens it, widen
   or correct `effect`, append a `source`, update `last_seen`. Update its INDEX line. **Do NOT create
   a second card for the same key, and do NOT add a new `## Learned — <date>` header anywhere.**
3. **INSERT only if novel AND effective (≥★★).** Create a new card + ONE INDEX line.
4. **NULL / overlapping / unverified → write NOTHING here.** It goes in the eval-dir report only.
5. **Contradicted by new evidence → move the card to `_archive.md`** (with the refuting source) and
   delete its INDEX line. If the negative is *scoped & reproducible*, instead add a `dead-end:` line
   to the relevant card (e.g. "tile X regresses decode" — a precise, conditioned negative, never a
   blanket "backend Y doesn't work").
6. **Enforce the budget.** INDEX.md ≤ 40 lines. If over, evict the lowest
   `value = confidence × freshness` line (its card → `_archive.md`). ★★★ is never auto-evicted.

**Invariant:** a principle "exists" iff it has a line in INDEX.md. The index is the single source of
truth and the size gate — that is the discipline, no linter required. Keep cards short: if a card
exceeds ~15 lines you are storing narrative, not a principle — distill it.
