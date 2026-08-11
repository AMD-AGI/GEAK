# `learned/` — distilled experience cards (an ADVISORY aid, read via INDEX.md)

This folder is GEAK's persistent, curated optimization experience. It replaces the old append-only
"Learned" sections (which grew to 500+ lines of near-duplicate run narratives).

## Philosophy — the KB is an accelerant, NOT a crutch and NOT a cage
The e2e_workflow is fully capable **without** this KB — cold runs (before any KB existed) reached ~20%
e2e. So the KB's only job is to help a good run **converge faster / go further**. It must **never make a
capable run worse by boxing it in or steering it down one path.** The judge is always **on-box
measurement** (the immutable op unittest oracle + the e2e Director's independent A/B + parity) — never
the KB. If a card and the measurement disagree, the measurement wins and the card gets corrected.

## Which sink? (`e2e_workflow/` vs `kernel_workflow/` — two memories, one direction of reference)
This folder holds **e2e-gated** lessons: an exploration that actually moved end-to-end throughput/latency
through the Director's A/B. Kernel-level results — "this lever/backend makes the op itself faster,
measured on the frozen-baseline isolated A/B" — belong in **`kernel_workflow/knowledge/learned/`**, and
they are written there by that workflow's own `update_experience` step, *including* when e2e was the one
that opened the kernel lane. The split is by the gate that produced the evidence, not by who launched
the run.

**Reference, don't duplicate.** When an e2e gain came from a kernel this pipeline optimized, your card
records the **e2e delta and which exploration paid off** (which routing/direction was worth the budget)
and **cites the kernel card** — `kernel_workflow/knowledge/learned/<slug>.md` — for the technique itself.
One technique, one home. Never copy a kernel card into this folder, and never write a card into
`kernel_workflow/knowledge/learned/` from here.

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

## Discovery — READ `INDEX.md`, then open the cards that look relevant
Retrieval here is **semantic, done by you** — not a string match. The index is small by construction
(≤40 cards) and every line carries the card's own `description`, the kernel symbols it was measured on,
and its keywords. So:

1. **Read `INDEX.md`.**
2. **Judge relevance by meaning**, not by exact wording. A card written for `split-k on skinny-M GEMM` is
   worth opening for a tall-K GEMM; a `cudagraph-safe integration` card applies to any rebind seam. You
   are better at this than a keyword query — that is why the matching happens in the reader.
3. **Open 0–3 cards.** Nothing relevant is a legitimate outcome: plan cold, as the pipeline did before
   any KB existed. Treat a card's `lever`/`effect` as **priors that seed your candidate set**, and
   `caution` as **extra checks**.

`grep` for an exact kernel symbol is a fine shortcut when you already know the name, but it is **not** the
lookup mechanism — never conclude "there is no card for this" from a failed string search.

Cards are **self-describing**: each opens with a discovery header, so it can be read directly and
`INDEX.md` can be **regenerated** from the cards rather than hand-maintained:

```bash
node kernel_workflow/scripts/build_learned_index.js e2e_workflow/knowledge/learned
```

(One generator serves both `learned/` sinks — referenced in place, not copied, the same convention the
kernel bake-off uses for this workflow's `op_benchmarker` + `harness_lib.py`. `--check` = fail if stale.)

> **Migration in progress.** The cards in this folder predate the discovery header, so `INDEX.md` here is
> still the hand-written one and is **known to drift** — cards have existed on disk that no index line
> mentioned, i.e. invisible to every reader. Until the headers are backfilled: when you write a card, give
> it a full discovery header (schema below); and when you read, `ls` this folder before concluding a card
> does not exist. Once every card has a header, the index becomes generated and that failure mode is gone.

### Keeping the vocabulary from drifting
`split-k` / `split_k` / `splitk` are one concept and three index entries. Three defences: the reader is
semantic, so a synonym costs ranking and not retrieval; the generator normalizes spelling mechanically
(lowercase, `_`/space → `-`, dedupe); and the generated index publishes a `## keyword vocabulary`
appendix — every term in use with its card count — that curators pick from before coining a new one,
with surviving near-duplicates flagged for a human call rather than silently merged.

## Card schema (one principle per file, ~12–20 lines)
```
---
# --- discovery header: how this card is FOUND (drives the generated INDEX.md) ---
name: <slug>                                # == the filename without .md
description: <ONE line, ≤160 chars: lever → on what → e2e effect. This becomes the INDEX.md line.>
keywords: [<lowercase-hyphenated terms>]    # PICK FROM the index's keyword vocabulary before inventing
kernels: [<kernel symbol / entry point>]    # e.g. fused_moe_kernel — a name a future run would recognise
platforms: [<gfx>]
kernel_class: <dense_gemm | moe_grouped_gemm | attention_decode | method | ...>
regime: decode | prefill | both | n/a
# --- classification + evidence ---
key: <one line of plain English identifying WHAT this card is about — the op, the arch, and whatever
     else distinguishes it: framework, dtype/quant format, shape regime. e.g.
     "bf16 fused-MoE grouped GEMM · gfx942/MI300X · vLLM". NOT a bare "dense_gemm · gfx942 · decode"
     triple: that collapses different cards onto one key and invites a wrong merge. The machine-readable
     slots are the header fields above; `key` is the human identity + merge target.>
type: routing | lever | method
confidence: ★ | ★★ | ★★★                    # how often it REPRODUCED (a hint strength, not authority)
effect: <iso x range; AND the e2e-transfer note — did it actually move e2e, or only isolated?>
lifecycle: active                           # `archived` leaves the index but keeps the file
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
1. **Read INDEX.md — and `ls` the folder** (the index can be missing a card; see the migration note).
   Find the card whose `key` matches your finding, judging **by meaning, not wording**: a
   differently-worded card for the same lever on the same op/arch IS a match.
2. **MERGE if it exists** — bump `confidence` if it reproduced, widen/correct `effect` (esp. the
   e2e-transfer note), append a `source`, update `last_seen`, extend `keywords`/`kernels` with any new
   search terms. Don't create a second card for the same key, and never add a new
   `## Learned — <date>` header.
3. **INSERT only if novel AND effective (≥★★).** ONE new card, opening with the full **discovery header**
   (`name`, `description`, `keywords`, `kernels`, `platforms`, `kernel_class`, `regime`) — that header is
   what makes it findable and what the generated index is built from.
3a. **Publish it in the index.** Once this folder's cards all carry headers, that is
   `node kernel_workflow/scripts/build_learned_index.js e2e_workflow/knowledge/learned` and nothing is
   hand-edited. Until then, add the ONE index line by hand — and check no other card is missing one.
4. **NULL / overlapping / unverified → write NOTHING here** (eval-dir report only).
5. **A surprising negative → a CONDITIONED `caution:` line** on the relevant card (with the condition it
   held under + its source), framed as "also verify". A claim *contradicted* by new evidence → move the
   card to `_archive.md` with the refuting source. **Never write a blocklist / "never use X".**
6. **Enforce the budget.** INDEX.md ≤ 40 card lines. Over → evict lowest `confidence × freshness` (its
   card → `_archive.md`). ★★★ is never auto-evicted.

**Invariant:** a principle "exists" iff a card file carries it with `lifecycle: active`. The card is the
source of truth; `INDEX.md` is its projection (today hand-kept, soon generated) and the size gate. Keep
cards short: >20 lines means you're storing narrative, not a principle — distill it.
**Above all: a card is advice the box can overrule, not a rule that overrules the box.**
