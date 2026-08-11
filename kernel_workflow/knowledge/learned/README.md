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

`INDEX.md` is **generated** from those headers by `kernel_workflow/scripts/kb.py ... index`
(grouped by `kernel_class`, ordered by confidence, plus the keyword vocabulary appendix). The generator is
sink-agnostic — it takes the folder as an argument, so the same one serves `e2e_workflow`'s `learned/`
(`python3 kernel_workflow/scripts/kb.py --kb-dir e2e_workflow/knowledge/learned index`); one mechanism,
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
lifecycle: active
# --- OPTIONAL below: write them when you have them; the lint validates FORMAT, not presence. They
#     were documented as required for a while and no card ever carried one, which teaches a curator
#     that the schema is approximate — so they are marked here for what they are.
layer: learned                              # optional
levers: [<lever id>]                        # optional. e.g. host.launch-overhead, mem.lds-tiling
cost: L0|L1|L2|L3                           # optional. L0 env/flag · L1 config/knob · L2 wrapper/host
                                            # rewrite · L3 new kernel. Checked against that set.
type: routing | lever | method
confidence: ★ | ★★ | ★★★                    # how often it REPRODUCED (a hint strength, not authority)
effect: <RELATIVE only — e.g. "1.34x isolated (weighted), non-overlapping vs frozen baseline". No ms.>
roofline: <optional. bound class before → after, + % of achievable peak, e.g. "HBM-bound 41%→ compute-bound 78% of
           achievable BW"; relative positions only, never absolute GB/s or TFLOP/s>
verified_on: YYYY-MM-DD | null              # optional, but say it if you know it: the date an on-box
                                            # A/B actually confirmed this. Checked for format.
                                            # `source:` (REQUIRED) already carries run id + date.
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

## What is ENFORCED and what is advice

Everything above describes intent. This section says which parts a machine checks, because a rule
that lives only in prose decays — this repo's own e2e index grew a "MANDATED LEVER" and a "do NOT use
it" one edit after its README banned both, and 47 of the first 78 real cards carried an absolute
wall-clock number under a Content rule that forbids them.

`python3 kernel_workflow/scripts/kb.py --kb-dir <this dir> lint --cards` REJECTS a card that:

| check | why it is mechanical |
|---|---|
| cites ms / µs / ns / FLOP/s / B/s / GHz / watts | Content rule 1. A box's absolute number reads to the next run as a target. |
| contains a mandate or prohibition | the ADD-only contract. A `caution:` must read "also verify X". |
| carries an eval-dir path, patch filename or harness case id | that is memorising a run, not distilling a principle. |
| names a specific kernel in `key` or the body | cards are class-level; campaigns re-run the same kernels. |
| has a bare `class · gfx · regime` `key` | the header already holds those slots; `key` is the human merge target. |
| has an empty/missing discovery-header field, or a `description` over 160 chars | the index line is built from them, and a card the reader cannot find is indistinguishable from a KB that learned nothing. |
| gives a bare geomean with no per-case evidence | a lever that helped one shape and did nothing elsewhere reads identically otherwise. |
| claims ★★★ without `confirms_blind >= 1` | self-confirmation cannot buy authority. |
| has an unknown `lifecycle`, a `cost` outside L0-L3, or an unparseable `verified_on` | a malformed field is worse than an absent one. |

The audit reads ARCHIVED cards too. Every other caller sees active only, so a card whose `lifecycle`
is a typo would otherwise be invisible to the one check that would have caught it.

Not enforced, and deliberately so: whether a card is TRUE, whether its lever generalises, and whether
it was worth writing. Those are judgement, and the box overrules all three.

## How a card LOSES standing (the only downward pressure)

Confidence that can only rise is not a signal. A card read fifty times that carried nothing looks
exactly like one that carried every round it touched, so the write path alone cannot tell them apart.

The loop: the planner names, in a direction's `learned_refs`, the card that seeded it; the verifier
re-measures that direction without knowing what suggested it; `update_experience` joins the two into
the cited card's counters. Declared rather than inferred, because the read path is semantic — nothing
downstream can reconstruct which card the planner acted on.

- the cited direction won its round -> `confirms_cited` += 1. **Only this is a confirmation.**
- it did not beat the frozen baseline -> `losses` += 1.
- anything between -> `attempts` += 1 and nothing else. `verified_geomean` is measured against the
  FROZEN baseline, so at 2.5x cumulative every non-regressing direction clears 1.0; counting those
  would let a card bank credit for advancing nothing. Seen for real: cited twice at 2.548x and
  2.555x, winner neither time.
- `losses >= 3` and above `confirms_cited` -> drop one star and add a `caution:` naming the base rate.

Two runs may not produce a card at all, and neither case is a failure: a **contended box** measured
its neighbours, and a **held-out kernel** is the instrument for measuring whether the KB works — a
card distilled from it means the next A/B over that kernel reads back its own answer.

Each run also returns `direction_entropy` (distinct/issued directions). A KB that helps raises the
verified speedup; a KB that CAGES lowers this without raising that.

## Bulk import — seeding this KB from campaigns that already ran

One run produces at most one card, so a KB starting empty needs as many runs as it wants cards. It
does not have to: `kb.py drain` takes proposals from `_inbox/` (one JSON per run, written by
`propose`, name carries the run id so parallel lanes never collide), merges by `key`, applies every
gate above, enforces the per-class budget, and regenerates the index — one operator, between
campaigns, so that "merge if the key exists" is actually implementable.

That is how this tree was seeded: 20 kernels' worth of finished 16h campaigns distilled in one pass
instead of 20 runs. Anything the gates refuse is reported with its reason rather than dropped.

**Budget is per class, not global.** A flat total is set by whichever class was optimized most — one
ingest put 15 cards in `moe_grouped_gemm` and 2 in `quantize_cast`, and a global cap would have had
the prolific class evict the sparse ones wholesale, narrowing the KB toward the last campaign. Over
cap, the lowest `confidence × freshness × earned-standing` card in THAT class is archived. Standing
is in that product on purpose: ranking on stars and date alone let a card written this morning that
nothing had ever tested outrank one cited and confirmed three times.

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
   `python3 kernel_workflow/scripts/kb.py --kb-dir kernel_workflow/knowledge/learned index` (add `--check` in CI to catch a stale file).
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
