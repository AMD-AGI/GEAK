# `learned/` — distilled kernel-optimization experience (advisory, measured, curated)

The kernel workflow has cross-*round* memory (the insight blackboard + hypothesis ledger in
`insight_log.md`) but it all dies when the run ends. This folder is the persistent tier: a small,
curated set of **advisory priors** carrying the evidence they were derived from.

Sibling, deliberately separate: `e2e_workflow/knowledge/learned/` holds *e2e-level* routing and
config knowledge. This folder holds *kernel-level* levers. Same schema, different concerns; do not
cross-write.

## Philosophy — the KB is an accelerant, never a cage

The workflow is fully capable **without** this KB — every result in `exp/` to date was produced cold.
So the KB's only job is to help a good run converge *faster* or go *further*. It must never make a
capable run worse by boxing it in.

**The judge is always on-box measurement** — the `verify_engineer` re-measurement and the Director's
independent A/B against the frozen baseline. If a card and the measurement disagree, the measurement
wins and the card gets corrected.

**Two tiers, kept apart:**
- **Here (persistent)** — distilled, class-level, bounded, curated.
- **In the eval dir (episodic)** — the raw per-run story (`insight_log.md`, `tech_lead_report.md`,
  `director_validation.json`), including every NULL. Do not copy run narratives here.

## Read path — three hard rules

Read the KB **after** you have formed your own profile-driven plan, as a cross-check and a source of
*extra* ideas.

1. **ADD-only, never filter.** A card may only add a candidate. It may never remove one, prune the
   round, or skip a measurement. Whatever the profile says to try, you still try.
2. **Measurement is always the judge.** A card says where to look first, not what is true.
3. **No card may foreclose an approach.** A `caution:` is "*also verify X*", never "don't do Y". A
   past winner is a starting point, not a ceiling.

Two of these are also enforced mechanically, because prose contracts decay — the sibling e2e INDEX
contains "MANDATED LEVER" and "do NOT use it" one edit after its README banned both:
- `kernel_workflow.js` caps how many directions per round may be KB-seeded (`kb_dir_cap`, default 1)
  and strips the excess. A wrong prior can cost at most one budget unit.
- `scripts/kb.py drain` **rejects** cards whose body contains mandate/blocklist language.

Round 1 is always cold: the read path is gated on `ROUND >= 2`, so the first plan of every run is
derived from the profile alone.

## Card schema (one principle per file, ≤15 lines)

```
---
key: <kernel_class> · <gfx> · <regime>   # CLASS level. Never a kernel name — see "Leakage".
type: routing | lever | method
confidence: ★ | ★★ | ★★★
effect: <verified range WITH per-case evidence, e.g. "1.3-2.1x on decode shapes (S<=1024),
         ~1.0x on prefill" — a bare geomean is not enough>
confirms_cited: <int>    # card was in the prompt and the run won. Cannot promote past ★★.
confirms_blind: <int>    # a KB-off run, or a DIFFERENT kernel under this key, found it independently.
losses: <int>            # cited, then the verifier measured no improvement. The down escalator.
attempts: <int>          # how many times this lever was TRIED, incl. dead ends. The base rate.
toolchain: <rocm/triton/torch fingerprint>
last_seen: YYYY-MM-DD
---
# <short title>
- lever: <an actionable thing worth TRYING>
- apply: <how to deploy it>
- verify: <how to confirm it engaged AND helped>
- caution: <a CONDITIONED "also verify X". Never a prohibition.>
- source: <run-id + date>   # REQUIRED. NOT an eval-dir path — see "Leakage".
```

### Confidence — a hint strength, not an authority level
- ★ single run, or distributions overlapped (≈ noise).
- ★★ single-run non-overlapping, or ≥2 consistent runs.
- ★★★ **requires `confirms_blind >= 1`.** A card that has only ever confirmed itself is capped at ★★
  forever, no matter how many times it reproduces.

That last rule is the one that breaks the self-confirmation loop: a card steers run N+1, whose success
would otherwise promote the card, which steers run N+2 harder. Citation-confirms are still recorded —
they just cannot buy authority.

### `attempts` is not optional
A KB that only records wins hides the base rate. "Split-K gave 1.4×" reads very differently at
`confirms 2 / attempts 3` than at `confirms 2 / attempts 14`. The hypothesis ledger already records
`dead_end` verdicts, so this number is free — omitting it would be a choice to look better.

## Leakage — the failure mode that would make this look brilliant and be worthless

Campaigns re-run the **same kernels**. A card distilled from `fused_moe_kernel` and then read by the
next `fused_moe_kernel` run is not learning, it is memorisation — and an A/B would show a huge,
entirely fake win. Three rules, all enforced at drain time, not by good intentions:

1. **Keys are class level.** `fused-MoE grouped GEMM · gfx942 · decode`, never `fused_moe_int4_w4a16`.
2. **No instance identifiers in a card** — no eval-dir paths, no `*_patch.diff`, no exact shapes
   lifted from a `test_cases.json`. `source:` carries a run-id and a date; the run's artifacts stay
   in the eval dir where they belong.
3. **Evaluate on held-out kernels.** The campaign's `ab_split.json` pre-registers which kernels the KB
   may be distilled from and which are reserved for measuring it. Never distil from a held-out kernel.

## Write path — CURATE, never blind-append

Per-run curators do **not** edit this folder. They write one proposal to `_inbox/<run_id>.json`
(unique filename, create-once, no lock). A single operator later runs `scripts/kb.py drain`, which is
the only writer.

That split is deliberate. 8 drivers × 2 hosts writing one `INDEX.md` over NFSv3 would need
cross-host locking that has never been exercised here, and would race `.git/index.lock` besides — but
more importantly, 20 concurrent curators *cannot see each other*, so "MERGE if the key exists" is not
even implementable concurrently: they would each insert a near-duplicate. One writer with the whole
inbox in view dedupes correctly.

`kb.py` serves **every** `learned/` tree in this repo — this one and `e2e_workflow/`'s — which is why
`--kb-dir` is required and has no default. One implementation, several data dirs: a default is how the
second copy of a rule starts drifting from the first.

### The down escalator — citations
Everything a curator writes is a success report, so on its own this KB could only inflate. The
counterweight is `citations`: every KB-seeded direction is joined against what the **verifier**
independently measured, and the whole list is passed through to the proposal unfiltered. `drain` turns
each one into `attempts += 1` and either a win or a `losses += 1`; a card with ≥3 losses that is losing
more than it wins is **demoted a star** and gets a conditioned caution recording how often it has been
tried without paying.

This is the only way a card can lose standing. A lever cited ten times that lost nine must not read
like one that won — and the run that cited it is the only thing that knows.

`drain` performs, per proposal:
1. **MERGE if the key exists** — bump the right confirms counter, add to `attempts`, widen `effect`,
   append `source`, update `last_seen`.
2. **INSERT only if novel AND ≥★★.**
3. **NULL / overlapping / unverified → write nothing.**
4. **A surprising negative → a conditioned `caution:`** on the relevant card. A card *contradicted* by
   new evidence → `_archive.md` with the refuting source. Never a blocklist.
5. **Budget:** `INDEX.md` ≤ 40 card lines; over → evict lowest `confidence × freshness`. ★★★ is never
   auto-evicted.
6. **Report coverage** — "N of M validated runs produced a proposal". Curators degrade to null
   silently on API faults; without this number you would ship a KB built from 6 of 20 runs and never
   know it.

Refuse to distil from: a `flagged` validation, a run with no director verdict, or a run flagged as
having shared its GPU with another tenant. All three produce numbers that describe the box, not the
kernel.

**Invariant:** a principle exists iff it has a line in `INDEX.md`. Cards over 15 lines are storing
narrative, not a principle — distil them.

**Above all: a card is advice the box can overrule, not a rule that overrules the box.**
