# `learned/` — distilled kernel_workflow experience cards (ADVISORY, read via INDEX.md)

This folder is the kernel_workflow's persistent, curated optimization experience — the symmetric twin
of `e2e_workflow/knowledge/learned/`, and it follows the same contract. It holds a small set of
*distilled principle cards*, one idea per file, written by the TechLead's `update_experience` step and
read by the planning/authoring roles as **advisory priors**.

It is **not** a run log. The raw per-run story stays in `EVAL_DIR`.

## Which sink? (`kernel_workflow/` vs `e2e_workflow/` — two memories, one direction of reference)
The two `learned/` folders are separated by **the gate that produced the evidence**, not by who launched
the run. A lane opened *by* e2e_workflow still writes here, because what it measured is a kernel-level
number.

| | `kernel_workflow/knowledge/learned/` (here) | `e2e_workflow/knowledge/learned/` |
|---|---|---|
| Evidence | frozen-baseline isolated A/B + oracle parity | e2e Director's A/B (throughput/latency) + parity |
| Card says | *this lever/backend makes the kernel faster* | *this exploration moved e2e — and by how much* |
| Written by | TechLead `update_experience` (every lane run; once centrally per bake-off) | System Architect / Op Benchmarker after a milestone |
| Claims e2e? | **never** — an isolated win is not an e2e win | yes, that is the whole point |

**Reference, don't duplicate.** When an e2e run gains from a kernel this workflow produced, the e2e card
records the *e2e delta and which exploration paid off* and **cites the kernel card** (`kernel_workflow/
knowledge/learned/<slug>.md`) for the technique itself. The technique lives here in exactly one place;
the e2e-transfer evidence lives there. Never copy a card across, and never write a card into the other
folder — the sink is always the `LEARNED_DIR` your orchestrator handed you.

## Philosophy — the KB is an accelerant, NOT a crutch and NOT a cage
kernel_workflow is fully capable **without** this KB; cold runs work exactly as they always did. The
KB's only job is to help a run converge faster / go further. It must **never make a capable run worse
by boxing it in.** The judge is always **on-box measurement** — here that means the **frozen-baseline
isolated A/B** (the immutable oracle baseline pinned at Benchmark, every candidate re-timed against
that same frozen dividend) plus **oracle parity** (the correctness gate in `verify_engineer`). If a
card and the measurement disagree, the measurement wins and the card gets corrected. (Same rule as
e2e's "bake-off + e2e gate"; only the gate identity differs.)

**Two-tier memory — keep them separate:**
- **Here (persistent)** = distilled, advisory priors with measured evidence. Bounded, curated.
- **In `EVAL_DIR` (episodic)** = the raw per-run story (`tech_lead_report.md`, per-round metrics, the
  per-candidate verify JSON). Every measurement, including NULL / negative results, lives there.
  Do **not** copy run narratives here.

## Discovery — READ `INDEX.md`, then open the cards that look relevant
Retrieval here is **semantic, done by the reader** — not a string match. `INDEX.md` is small by
construction (≤40 cards) and every line already carries the card's `description`, the kernel symbols it
was measured on, and its keywords. So the read path is simply:

1. **Read `INDEX.md`** (one file, ≤~60 lines).
2. **Judge relevance by meaning**, not by exact wording. A card written for `split-k on skinny-M GEMM` is
   worth opening for a tall-K GEMM; a `launch-overhead` card is worth opening for any dispatch-bound op,
   whatever the class. You are better at this than any keyword query — that is the point of doing it in
   the reader instead of in a matcher.
3. **Open 0–3 cards.** Nothing relevant is a legitimate outcome: plan cold, exactly as this workflow does
   with no KB at all.

`grep` for an exact kernel symbol is a fine shortcut when you already know the name, but it is **not** the
lookup mechanism: it matches strings, and what you are looking for is a concept. Never conclude "there is
no card for this" from a failed grep.

Each card also opens with the same **discovery header** (`name`, `description`, `keywords`, `kernels`,
`platforms`, `kernel_class`, `regime`, `confidence`), so it stays self-describing when opened directly or
when the index is missing.

`INDEX.md` is **generated** from those headers by `kernel_workflow/scripts/build_learned_index.js`
(grouped by `kernel_class`, ordered by confidence, plus the keyword vocabulary appendix). The generator is
sink-agnostic — it takes the folder as an argument, so the same one serves `e2e_workflow`'s `learned/`
(`node kernel_workflow/scripts/build_learned_index.js e2e_workflow/knowledge/learned`); one mechanism,
referenced in place, never copied.
**Never hand-edit it** — edit the card's `description`/`keywords`/`confidence` and regenerate. Two
consequences worth knowing: the index can never drift from the cards, and parallel lanes cannot lose each
other's entries (a regen republishes whatever is on disk, instead of each lane appending its own line).

### Keeping the vocabulary from drifting
`split-k` / `split_k` / `splitk` / `Split K` are one concept and four index entries — that fragmentation
is the main way a keyword scheme rots. Three defences, in order of how much they carry:

1. **The reader is semantic** (above), so a synonym costs relevance ranking, not retrieval. This is why
   drift is a hygiene problem here and not a correctness one.
2. **The generator normalizes** mechanically: lowercase, `_`/space → `-`, collapse repeats, dedupe. The
   curator's spelling discipline is not load-bearing.
3. **The vocabulary is published and reused.** `INDEX.md` ends with a generated
   `## keyword vocabulary` line — every term currently in use, with its card count. A curator picks from
   that list and only coins a new term when nothing fits. Synonyms that survive normalization (`split-k`
   vs `splitk`) are **flagged** in the index with a ⚠ block; the fix is to edit the offending card and
   regenerate. The generator never auto-merges them — collapsing `mfma`/`mfmas` behind the curator's back
   would be a worse failure than a visible warning.

The same "reuse before coining" rule applies to `kernel_class` and `lever` ids, and matters more there:
those are what group the index.

## How to USE it during a run (read path) — three hard rules
Read `INDEX.md` (or grep the headers directly) **after** you have formed your own profile-driven plan, as
a cross-check and a source of *extra* ideas — then:
1. **ADD-only, never filter.** Cards may only *add* candidate levers/directions to try. They must never
   remove a candidate, prune the direction set, or skip the author/measurement step.
2. **Measurement is always the judge.** Run the full author + isolated A/B + oracle parity regardless of
   what any card claims. A card is a hint about where to *look first*, not a verdict.
3. **No card may foreclose an approach.** A `caution:` line is "**also verify X**", never "don't do Y".
   A past winner is a starting point, not a ceiling; a past pitfall is a thing to double-check.

Open the cards whose `key` matches this run's `(kernel_class, gfx, regime)`; treat their `lever`/`effect`
as **priors that seed your candidate set**, and `caution` as **extra checks**.

## Card schema (one principle per file, ~12–20 lines)
```
---
# --- discovery header: how this card is FOUND (drives the generated INDEX.md; keep greppable) ---
name: <slug>                                # == the filename without .md
description: <ONE line, ≤160 chars: lever → on what → relative effect. This is the INDEX.md line.>
keywords: [<lowercase-hyphenated terms>]    # PICK FROM the "keyword vocabulary" appendix at the bottom
                                            # of INDEX.md before inventing one — reusing a term is what
                                            # keeps sibling cards clustered. e.g. split-k, lds-tiling,
                                            # launch-overhead, dot-scaled, aiter, decode, skinny-m
kernels: [<kernel symbol / entry point measured>] # e.g. _gqa_sparse_fwd_kernel, fused_moe_kernel. The
                                            # concrete name matters: greps hit it long before a class does.
platforms: [<gfx>]                          # e.g. [gfx942] — the arch the evidence was measured on
kernel_class: <kernel_class>                # e.g. dense_gemm | moe_grouped_gemm | attention_decode | method
regime: decode | prefill | both | n/a       # the shape regime the evidence covers
# --- classification + evidence ---
key: <one line of plain English identifying WHAT this card is about>
                                            # The human-readable identity + dedupe/merge target. Write it
                                            # as a sentence fragment, not a rigid triple: name the op, the
                                            # arch, and whatever else actually distinguishes this card —
                                            # framework, dtype/quant format, shape regime.
                                            #   good: "bf16 fused-MoE grouped GEMM · gfx942/MI300X · vLLM"
                                            #   good: "MXFP8 E8M0 dense linear, decode-bound · gfx950"
                                            #   bad:  "dense_gemm · gfx942 · decode"  <- collapses a vLLM
                                            #         MXFP8 card and an sglang bf16 card into one key and
                                            #         invites a wrong merge.
                                            # The MACHINE-readable slots are the discovery-header fields
                                            # above (kernel_class/platforms/regime); `key` does not need
                                            # to repeat their job, so let it say what a person would say.
layer: learned
levers: [<lever id>]                        # e.g. host.launch-overhead, mem.lds-tiling — free-form but reused
cost: L0|L1|L2|L3                           # L0 env/flag · L1 config/knob · L2 wrapper/host rewrite · L3 new kernel
lifecycle: active
type: routing | lever | method
confidence: ★ | ★★ | ★★★                    # how often it REPRODUCED (a hint strength, not authority)
effect: <RELATIVE only — e.g. "1.34x isolated (weighted), non-overlapping vs frozen baseline". No ms.>
roofline: <bound class before → after, + % of achievable peak, e.g. "HBM-bound 41%→ compute-bound 78% of
           achievable BW"; relative positions only, never absolute GB/s or TFLOP/s>
verified_on: YYYY-MM-DD | null              # the date an on-box A/B actually confirmed it
last_seen: YYYY-MM-DD
---
# <short title>
- lever: <an actionable thing worth TRYING (a seed candidate), not a mandate>
- apply: <how to deploy / the rebind seam / env var / the shape of the patch>
- stack: <ONLY when >1 direction landed. Total first, then per-direction — see "Stacked wins" below.>
- verify: <how to confirm it engaged + beat the frozen baseline on the isolated A/B>
- pitfall: <symptom observed> → <root cause> → <the fix that worked>   # repeatable; one line each
- caution: <a CONDITIONED "also verify X". NEVER a blanket prohibition.>
- source: <EVAL_DIR path | arXiv | repo@path>   # REQUIRED — no claim without evidence
```

### Content rules — what a card may and may not record

**1. Sanitize the numbers: RATIOS, never absolutes.** Wall-clock varies by box, clock/power state,
driver, and neighbour load, so an absolute figure copied into a card is stale on arrival and misleads
the next run into treating it as a target.
- **Record:** speedup ratios (`1.34x`), percent deltas (`+18%`), *fractions of achievable peak*
  (`62% of achievable HBM BW`), occupancy %, cache hit %, arithmetic intensity, the roofline **bound
  class** and which side of the ridge point the op sits on, and workload constants that are properties
  of the problem, not the machine (shapes, dtypes, tile sizes, `num_warps`, split-K, grid geometry).
- **Do NOT record:** `ms`/`µs`/`ns` wall-clock (baseline or optimized), absolute `TFLOP/s`, `GB/s`,
  bytes/s, achieved-vs-spec bandwidth in absolute units, kernel duration, power, or clocks. The raw
  timings already live in `EVAL_DIR` — that is the right home for them.
- If a lesson genuinely needs a magnitude, express it against something on the same box: "≈2× the
  launch overhead of the fused path", "≈0.4 of the roofline ceiling before, ≈0.8 after".

**2. Record the pitfalls, not just the win.** The traps cost the next run more time than the lever
saves it. One `pitfall:` line per trap actually hit during this run, in `symptom → root cause → fix`
form, and only for traps *observed here* (a hypothetical is not evidence). Typical sources: a candidate
that failed oracle parity, an apply/build failure, a "faster but wrong" result, a config that silently
did not engage, a win that vanished when the baseline was frozen properly. A pitfall is not a
prohibition — it is the thing to check *while* trying the lever.

**3. Stacked wins: total first, then each direction separately.** When several optimization directions
compounded into the final number, a single blended figure is unusable — the next run cannot tell which
lever to try first, or which one carried the win. Give the total, then attribute per direction, and say
plainly if the attribution is approximate (directions interact; a merged patch's parts are rarely
additive).
```
- stack: total 1.62x isolated (weighted, director-verified) = three directions compounded
  - 1. mem.lds-tiling — 1.31x standalone (round 2, verified) — the bulk of the win
  - 2. host.launch-overhead — +12% on top of (1) (round 3, verified) — only pays once (1) removed the stall
  - 3. compute.mfma-nonkdim16 — +9% on top of (1,2) (round 4, verified)
  - note: attribution is incremental in landing order, not independent; (2) measured ~+3% alone.
```
Each entry carries its own relative effect and where it was measured (round + verified/claimed). If a
direction's individual contribution was never isolated, say so rather than inventing a split. When only
one direction landed, omit `stack:` entirely — `effect:` already says it.

### Confidence tiers (a HINT strength, not an authority level)
- ★   = single run, isolated distributions overlapped (≈ noise / unverified) — weak hint.
- ★★  = single-run non-overlapping isolated A/B, OR ≥2 consistent runs.
- ★★★ = ≥2 independent runs non-overlapping on the frozen-baseline A/B.

## How to UPDATE it after a run (write path) — CURATE, never blind-append
Owner: **TechLead** (holds the global routing view; runs the `update_experience` step after Report).
One transaction:
1. **Read the whole index before you write** — including its keyword vocabulary appendix. Find the card
   whose `key` matches your finding, judging by meaning (a differently-worded card for the same lever on
   the same class/arch IS a match — merge into it rather than filing a near-twin). If the index looks
   thinner than the folder, regenerate it first: a lane that finished seconds ago may not be projected yet.
2. **MERGE if it exists** — raise/lower `confidence` by what reproduced, widen/correct `effect`, append a
   `source`, refresh `last_seen`, add any new `keywords`/`kernels` the run surfaced. Never a second card
   for the same key.
3. **INSERT only if novel AND effective (≥★★).** ONE new card, with a complete discovery header, obeying
   the three **Content rules** above (ratios only — no ms; the pitfalls you actually hit;
   total-then-per-direction for a stacked win). This applies to a MERGE too: never let an absolute timing
   in through the back door of an updated `effect:`.
3a. **Regenerate the index — never hand-edit it.**
   `node kernel_workflow/scripts/build_learned_index.js` (add `--check` in CI to catch a stale file).
4. **NULL / overlapping / unverified → write NOTHING here** (the `EVAL_DIR` report is enough). A one-off
   raw number is not a card; only a reusable `(kernel_class, gfx, regime) → lever` lesson earns one.
5. **A surprising negative → a CONDITIONED `caution:`** on the relevant card (with the condition it held
   under + its source), framed as "also verify". A claim *contradicted* by new evidence → move the card
   to `_archive.md` with the refuting source. **Never write a blocklist / "never use X".**
6. **Enforce the budget.** ≤ 40 active cards; the generator prints the count and flags the overflow in
   the file itself. Over → set the weakest card's `lifecycle:` to `archived` and move it to
   `_archive.md` (lowest `confidence × freshness`; ★★★ is never auto-evicted), then regenerate.

**Invariant:** a principle "exists" iff a card file carries a discovery header with `lifecycle: active`.
The card is the source of truth; `INDEX.md` is its projection. Keep cards short: >20 lines means you're
storing narrative, not a principle — distill it.
**Above all: a card is advice the box can overrule, not a rule that overrules the box.**
