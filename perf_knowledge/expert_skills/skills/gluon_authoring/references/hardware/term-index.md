# Term index — the word you would search for → the heading that answers it

A lookup table from the vocabulary a reader *brings* to the heading that already answers them.
Nothing here is new knowledge. Every row points at text that already exists; if a row and its
target ever disagree, the target wins.

**Why this file exists.** Two lanes independently paid full measurement cost to re-derive facts
these references already held, and neither was a content gap:

* one re-derived `tail_efficiency` by hand while the formula sat in a file it did not open;
* one violated a rule this pack states correctly, with the remedy attached, because the heading
  that carries it is `## Check the instrument before you read the scale off it` — which contains
  none of the words a reader with a fitted curve would type.

That is the same failure the pack already diagnosed for `lever-cards.json` — *the cards stayed,
nothing indexed them* — and fixed by restoring an index rather than by rewriting the cards. This
is that fix, applied to prose. It is a **pilot scoped to the AMD references**; if it works the
pattern spreads.

**Why an index and not better headings.** Headings here are addresses: other files cite them
verbatim as a filename-plus-heading pair, and the link checker validates only the *file* half — it
splits on the heading separator and never confirms the heading text. So a renamed heading breaks
every citation of it **silently**, with nothing in the gate to catch it. An index is additive: it
introduces no new way for anything to break.

**How a row is meant to be found.** By `grep`, not by reading this file top to bottom — the same
way the misses above would have been avoided. That also bounds what this file can fix: it helps a
reader who searches, and it does nothing for one who scans a heading list. A pointer to it from
the pack router is the obvious complement and is not in this pilot's scope.

<!-- Machine-readable declaration of the rows that are deliberately pack-conditional.
     `scripts/check_term_index.py` reads these and fails on any conditional row NOT declared here,
     so the caveat prose below and the check cannot drift apart. One per line.

     None at present. The instrument rows were declared conditional here while
     `## Check the instrument before you read the scale off it` shipped in only two packs; that
     gap has since been filled and the declaration was retired. `stale_declarations()` is what
     reported it: an exemption whose reason has gone away keeps passing forever, because nothing
     ever exercises it. If you add a row back, add the declaration with it. -->

## Seeded from a documented miss

These rows exist because someone actually searched for the term and did not find the section.

| you searched for | it is answered at | which tells you |
| --- | --- | --- |
| `fit`, `fitted`, `fitted curve` | `../method/benchmark-hygiene.md ## Check the instrument before you read the scale off it` — **see the note below** | when a fitted reference may be read from at all |
| `regression`, `intercept`, `slope` | same | *(`regression` is absent from most copies of that file — grep reaches it in some packs and not others)* |
| `residual`, `residuals`, `structured residuals` | same | a fitted reference cannot be extrapolated from while its own residuals are structured |
| `extrapolate`, `extrapolation`, `out of range` | same | the remedy, stated with the rule |
| `leverage`, `influential point` | same | *(`influential point` appears nowhere; `leverage` does, so grep reaches this one)* |
| `wave quantization`, `quantization tail` | `roofline-models.md ## Saturation / wave quantization` | the mechanism and the formula |
| `tail efficiency`, `tail_efficiency` | `roofline-models.md ## Saturation / wave quantization` | `grid_tiles / (waves * CUs)` |
| `tail efficiency` **as an A/B explanation** | `planning-constants.md ## Tail efficiency is one-sided too` | it can veto a geometry; it **cannot rank two configurations** |
| `grid_tiles`, `saturation` | `roofline-models.md ## Saturation / wave quantization` | **not** the same quantity as the `saturation` defined in the SKU files — see those files' own note |
| `waves`, `last wave`, `partial wave` | `roofline-models.md ## Saturation / wave quantization` | how much of the last wave is wasted |

**How much of this a grep would have replaced — and why that number moves.** Measured across the
copies of `../method/benchmark-hygiene.md` shipping today: `influential point` appears in none of them, and
`regression` in a minority. Those are the rows no full-text search would have found. The rest —
`fit`, `residual`, `extrapolate`, `leverage`, `intercept` — *are* in that section's body, so a
grepping reader could have reached them; they are listed anyway because the miss they came from
was real, which says that reader was scanning headings rather than grepping, and a heading-level
entry is what that reader needs.

Read the split above as measured-at-a-time, not as a property of the terms — and note that its
earlier version failed in **two different ways**, which need two different repairs.

- **Overtaken by events.** `leverage` was genuinely unreachable and is now present in every copy,
  because a shared regime file carrying the fitted-reference derivation was promoted into the
  packs that lacked it. Nothing was wrong when written; the tree moved. The repair is a
  re-measurement, which is why this paragraph is dated by its contents rather than asserted.
- **Wrong when written.** `slope` was never absent — it had statistical-sense occurrences in the
  target file at the time the row claimed none did, and `regression` had a non-statistical one.
  No amount of re-measuring later would have caught that; the claim never held. The repair is to
  examine the method that produced it. It was produced by grepping one copy of a per-pack file and
  generalizing to all of them — which is the same error this index's own conditional-row machinery
  exists to prevent, committed in the prose instead of in the pointers.

Nothing in the table depends on the split — the pointers are checked on both halves and hold
either way — but if you quote the number, re-measure it, across every copy, rather than copying it
from here.

**Note on the instrument rows — `../method/benchmark-hygiene.md` is per-pack, not one shared document.** The
copies differ by roughly an order of magnitude in length. `## Check the instrument before you read
the scale off it` now resolves in every pack that ships this index, so the rows above are
unconditional; they were not always, and the history is worth keeping because of what it shows.
While the section existed in only two packs, the filename resolved everywhere and the heading did
not — the exact shape of breakage a filename-only link check cannot see, and the reason this file
is checked on both halves. The rule those rows point at — *do not extrapolate from a fitted
reference while its own residuals are structured* — is now derived in each pack rather than
asserted here.

## Plausible, not yet observed

Same form, but nobody is recorded as having missed these — they are judgement, and a row here is
weaker evidence than a row above.

| you searched for | it is answered at |
| --- | --- |
| `occupancy gain`, `free up VGPR`, `free up LDS`, `will more waves help` | `planning-constants.md ## Occupancy is a one-sided criterion` |
| `waves/SIMD`, `wg/CU`, `workgroups per CU` | `planning-constants.md ## waves/SIMD is not workgroups/CU — convert before spending a round on it` |
| `; Occupancy:`, `compiler occupancy comment`, `occupancy from the asm` | `planning-constants.md ## The emitted ; Occupancy: N is a register-term answer` |
| `OutOfResources`, `Hardware limit`, `shared memory limit`, `LDS budget` | `planning-constants.md ## Make an occupancy table falsifiable before you quote it` |
| `unrealized_conversion_cast`, `compiler bug or capacity` | `planning-constants.md ## Make an occupancy table falsifiable before you quote it` |
| `lds_per_cu_kib`, `lds_per_wgp_kib`, `LDS scope`, `per-WGP` | `planning-constants.md ## Terminology` |
| `ridge`, `ops/byte`, `arithmetic intensity` | `roofline-models.md` §Roofline + ridge; SKU peaks in the `amd-*-skus.md` file for your chip |
| `which lever`, `what should I try` | `bound-class-signals.md`; `capability-matrix.md` gates availability per arch |

## Adding a row

Add one when you searched for something and did not find it — that is the evidence. Put it in the
first table and say what you typed. Do not add a row for a term you merely imagine someone might
type; that is what the second table is for, and it should stay the smaller of the two.

**Record the measurement that established the miss, not just the term.** If you claim a word is
unreachable by grep, say which files you grepped and when — and if the target is a per-pack file,
grep every copy, because one copy is not the pack set. A row without that is unfalsifiable after
the fact: when it later turns out to be false, there is no way to tell whether it went stale or
was never right, and those need opposite repairs. The paragraph above the second table records one
of each, which is why both are described there instead of being quietly corrected.
