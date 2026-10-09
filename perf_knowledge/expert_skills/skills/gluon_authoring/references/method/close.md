# Close — accept, stop, and report against `champion_ms`

**What this stage decides.** Whether the run's best candidate is a **win**, a **partial**, a negative
(`negative_revert_plain` / `negative_keep_baseline`) or a **timeout**; whether it ships for every served
shape bucket or behind visible dispatch; and whether the record that carries it will survive the audit
(provenance receipts, closure self-review, report lint). It decides nothing about *what to try next* — that
is [`climb.md`](climb.md).

**When you are here.** When a stop condition below holds (the round budget is spent, or a global
non-host-solvable blocker halts every remaining direction), or a hard stop fires. A finished layer, a
measured lever, a passed diagnostic, or a degraded evidence layer is **not** a reason to be here.

**Who closes, in GEAK.** The deep_engineer running the `deep_explore` direction closes its own track:
it states the verdict against `champion_ms`, writes the closure self-review
(`## Closure challenge (self-review before a ceiling / keep-baseline / negative close)` below) and returns
GEAK's `worker_result` (ENG_SCHEMA) with `best_patch.diff`. Acceptance is not the engineer's:
`verify_engineer` re-benchmarks the patch in a clean workspace (timing from
`e2e_workflow/scripts/harness_lib.py`, fresh process per leg, same-window baseline), the commit gate
(`MIN_IMPROVE` = 2% over the cumulative best; a measured noise band only makes it stricter) decides the
merge, and Director validates against the true baseline and arbitrates. This page supplies the verdict
vocabulary, the acceptance bars and the caveats the engineer's result must carry (`parity_unreached`,
`injected` numbers, served-range validation, the closure self-review's verdict). The pack's
`canonical_record.json` / `run_state.json` / `close_audit.json` / `final_report.json` are optional
bookkeeping the engineer may keep through `scripts/canonical_record.py`; they are not GEAK's deliverable.

Evaluate every candidate from a clean isolated root using the immutable `COMMANDMENT.md`
([`benchmark-hygiene.md`](benchmark-hygiene.md)). The two baselines — `plain_baseline` (= `champion_ms`,
the target line) and `gluon_anchor` (the working baseline) — and their identity rules are defined in
[`recover.md`](recover.md) (`## Two baselines`); this page applies them.

---

## Hard stops

- Missing required entry artifacts, an unverified comparator, or a failed correctness oracle is a hard
  stop; record the exact missing/failed item.
- A global non-host-solvable environment blocker, or an exhausted declared budget. A finished probe, one
  failed direction, a degraded optional evidence source, or a scoped tool failure is not by itself a global
  stop: record it and use the documented fallback ([`triage.md`](triage.md)).
- Budget left is not a stop. Closing with rounds or wall clock unspent and a climb below the floor is a
  finding against the close, whatever it called itself. Spend it or price what is left.
- Never report paper timings, silently replace the comparator, infer completion from elapsed time, or
  close with an unrecorded measurement.

## Stop conditions

Stop when the round budget is spent, or a global non-host-solvable blocker halts every remaining direction.
A finished layer, a measured lever, a passed diagnostic, or a degraded evidence layer is **not** a stop
(the `## Hard stops` list above reads the same here). Scoped reroutes (record, fall back, keep going) vs a
global halt: [`triage.md`](triage.md) (`## Retryable vs scoped-ceiling vs global (continue / defer / halt)`).

**The `outcome` enum.** `outcome` takes one of `win | partial | negative_revert_plain |
negative_keep_baseline | timeout`, with a `deferred[]` list on `partial`. That is the whole enum
`canonical_record.py` accepts (`OUTCOMES`), and it is the one to write. The run's `status` is a separate
field: `closed | deferred_needs_user | blocked | partial_time_limit` (`RUN_STATUSES`).

| outcome | when |
| --- | --- |
| `win` | the best Gluon result reaches **and exceeds** `champion_ms` by a real, repeat-confirmed margin (acceptance bar 3), correct, validated across the served range |
| `partial` | a scoped ceiling needs the user, or blocked layers left part of the work undone; the best in-scope result is delivered with `deferred[]` |
| `negative_revert_plain` | the Gluon result is slower than `champion_ms` after the budget — the plain champion is the delivered result (Gluon-specific binding #5) |
| `negative_keep_baseline` | no candidate beat the working baseline the run started from; the baseline is kept |
| `timeout` | the budget ran out; must be consistent with `budget.remaining == 0` |

**A Gluon result slower than the champion is a negative, never the winning stage.** Report
`negative_revert_plain` with the measured gap; Director records the cross-skill verdict when it arbitrates. A Gluon wall
does **not** prove plain was the ceiling — that is a separate claim needing its own evidence.

**`structure_suspect` is an artifact, not an outcome.** A structural wall is reported as the honest outcome
for the measurement you have — usually `partial`, or `negative_revert_plain` when Gluon finished under
`champion_ms` — **plus** a `structure_suspect.json` emitted through `canonical_record.py
request-structure-suspect`. That file carries the evidence that the *structure*, not a lever inside it, is
the ceiling, and it is what hands the decision back to tech_lead ([`entry.md`](entry.md)). Writing
`structure_suspect` into `outcome` is rejected by both `canonical_record.py` and `report_lint.py`, so a run
that does it cannot close. The shape to copy is `resweep_request` (`canonical_record.py request-resweep`),
which works the same way: an outcome plus a request file, never an outcome value of its own. Both requests
travel in the deep_engineer's result (notes + artifact paths); tech_lead turns them into the next round's
direction (a resweep goes to GEAK's own plain tuning).

**`parity_unreached`.** A run that never cleared the parity gate reports `parity_unreached` in
`caveats[]` with the attributed residual split — the number is still reportable, but it is a recovery
number, and saying so is what keeps it honest. On an **incumbent** entry the parity caveat does not apply;
say that, with the entry mode, rather than leaving it unaddressed. Any number produced by re-injecting
plain's pipeliner is labelled `injected` in the report and never counted as the win ([`recover.md`](recover.md)).

**Two vocabularies both spelled `stage`, and they do not overlap.** The stage card's `stage` is a spine id
(`entry` / `transcribe` / `recover` / `evidence` / `climb` / `close`). `worker_result.json`'s `stage` is
validated against `canonical_record.PHASES` (`anchor`, `search`, `branch`, `converge`, `resweep`,
`finalization`) and takes none of the spine names. Map it:

| you are in spine stage | write `worker_result.stage` |
| --- | --- |
| `entry`, `transcribe` | `anchor` |
| `recover`, `evidence`, `climb` | `converge` |
| `close` | `finalization` |

(`search` / `branch` / `resweep` are front-end phases; a CLIMB-only owner never writes them.)

## Three acceptance bars

1. **Anchor stage**: regression vs `plain_baseline` is **accepted, not rejected** (transcription almost
   always starts slower). Gate = `correctness == plain` plus the four equivalence checks
   ([`transcribe.md`](transcribe.md)). GEAK's `candidate_floor` must be set below 1.0 for the
   anchor to be tracked at all ([`index.md`](index.md) G0).
2. **Per layer**: a candidate is judged vs `gluon_anchor` / running-best — must not regress, plus the
   three-evidence closure (budget + profile + IR) of the **single tile-op card cell** the layer changed
   (`../tile-programming/tile-op-contract.md`). **Enabling-step exception:** a layer may regress vs
   running-best *iff* it is a **declared enabling step** toward a coupled combination — it is then
   *provisionally accepted* (not closed), and the **combination must net-beat the checkpoint within the
   transient-regression budget**, else it rolls back to the checkpoint ([`climb.md`](climb.md)
   `## Self-monitoring`, `## A direction matures over rounds — do not revert it on round one`). This
   removes the per-layer double-block; it does not relax bar 3. Below the parity gate a kept round is
   recorded as `recovery` against the suspect it closed, never as a win.
3. **Final**: the best Gluon result must **reach and exceed `plain_baseline`** (`champion_ms`) to count as a
   win. If, after the round budget, it cannot, record `negative_revert_plain` and revert to plain (the
   hand-off decision over-estimated headroom). "Exceed" additionally means clearing the 2% commit gate
   in GEAK's acceptance timing (`verify_engineer`).

A run can also end **deferred** (a scoped ceiling needing the user, [`triage.md`](triage.md)
`## Retryable vs scoped-ceiling vs global (continue / defer / halt)`): keep the best in-scope result as the
fallback, and report it under `outcome: partial` with `status: deferred_needs_user`.

**Win type (a speedup ratio is not the only win).** A candidate closes as one of: `win_type = speedup`
(beats the target line by a real, repeat-confirmed margin), `win_type = unblock` (the production baseline
**does not run** in the contracted environment — a compile/backend assert, resource exhaustion, or an
unsupported path — so any correct, running candidate that passes the oracle is a first-class `closed` win
even though no ratio exists), or `win_type = tie_kept` (within the noise band but kept for a non-latency
reason, e.g. correctness, robustness, or removing a hazard). For `unblock`, `vs_baseline` is legitimately
null; the win is gated on the oracle (or the swept-cell correctness proxy,
[`benchmark-hygiene.md`](benchmark-hygiene.md) `## Swept-cell correctness proxy`) passing, not on a
speedup. Always record `baseline_status` (`ran | crashed | oom | unsupported | missing`) so an `unblock` is
auditable rather than implied by a null number. When a fair like-for-like plain target line is not
constructible, the substitution rule (production config + external ASM ceiling, stated in the contract)
is in [`recover.md`](recover.md) `## Two baselines`; it closes as `tie_kept` / `unblock`, never as a
fabricated speedup.

Know the oracle's gate TYPE, not just its tolerance: a cosine/angular-similarity gate can be stricter than
the abs/rel gate and can reject a low-precision (fp8/fp4) matmul that passes abs/rel. Check it before
adopting a low-precision matrix path (`../tile-programming/low-precision.md`), and when a correctness check
fails, read the failure MAGNITUDE before calling it borderline numerics: a near-1.0 cosine-diff (or large
abs error) is a real bug, not reduction-order noise — only a value hovering at the tolerance is genuinely
borderline ([`triage.md`](triage.md) `## Correctness-failure magnitude (before blaming numerics)`).

## Mechanics

```text
1. restore/copy baseline into a clean candidate root
2. apply the candidate
3. clear only candidate-local build/cache artifacts   (move aside; no `rm` in scripts GEAK roles run)
4. run COMMANDMENT correctness
5. run COMMANDMENT full benchmark (per-shape)          (acceptance timing: harness_lib, fresh process per leg)
6. record per-shape results + aggregate (geomean) + IR acceptance
7. reject any candidate that changed the boundary, oracle, shapes, or markers
```

In GEAK steps 1–5 are the Verify phase (`verify_engineer` / `measure_legs`); the deep_engineer makes sure
step 6's IR acceptance and step 7's invariants are on file in its OUTPUT_DIR.

Primary metric: geometric mean speedup vs `plain_baseline`; report arithmetic mean and **per-shape
regressions** too. A higher aggregate with a large important-shape regression is rejected or moved behind
visible dispatch.

## Multi-shape no-regression + dispatch

**Validate across the served shape range before claiming a win** (Gluon-specific binding #6). Optimizing to
the champion's single anchor shape can produce a clean-looking win that loses over most of the range's
GPU-seconds; bucket the dispatch if the crossover does not close. The bundle's `served_range` is the shape
list; weigh the served mix with `scripts/served_envelope.py` (weighted harmonic mean).

A single kernel spanning multiple shapes may route different shape buckets to different tiers (plain |
gluon | gluon+llvm) via visible host-side dispatch. The **per-bucket fallback guard** is mandatory: keep a
candidate for a bucket only if it beats the production baseline (correct identity + boundary) for that
bucket; for any regressing bucket keep the production baseline via visible dispatch. Never ship a
per-bucket regression to win an aggregate.

Pick the contract first (rank if more than one applies): `all_shape_no_regression` |
`aggregate_improvement` | `production_weighted_improvement` | `shape_specific_experiment` |
`no_fallback_for_selected_shapes`. Common ranked default: hard = correctness + required path policy;
primary = aggregate latency on the gated stream; secondary = per-shape map; fallback = visible dispatch for
repeat-confirmed losers. Always compare the same ordered shape stream, boundary, aggregate type, and oracle
on both sides.

Decision recipe:

- all shapes improve -> keep (correctness + boundary pass);
- aggregate improves but some shapes regress -> do NOT keep by default; if a stable bucket rule separates
  winners/losers, prefer visible host-side dispatch;
- some improve / some regress / aggregate neutral -> bucket dispatch, plain fallback + fast path, or the
  safer all-shape version;
- hard no-regression + any repeat-confirmed regression -> add visible bucketed dispatch, keep the baseline
  for the losing bucket, or reject.

Bucket-dispatch recipe (the dispatch itself is a Stage-Plain / P6 outcome — [`front-end.md`](front-end.md)):
find per-shape winners; group by interpretable fields (M/N/K, seq len, dtype, flags); simulate a visible
rule on the full stream; add threshold-adjacent sanity shapes (just below / at / above each cutoff); keep
the safe baseline for systematic loser regions; rerun the final dispatch code, not the simulator. Dispatch
worthiness = `aggregate_incremental_gain / best_bucket_gain / loser_bucket_regression /
dispatch_overhead_in_boundary / added_complexity / production_weighting`. Keep the simpler
global/baseline path when dispatch only adds noise-band movement; separate `local-winner` /
`aggregate-not-worth-it` / `production-weight-needed` rather than "the implementation is bad".

Material regression: use the task contract if given, else weigh effect size + repeat stability (tiny
unstable deltas = noise; stable positive deltas on contract shapes = real regressions). Run a full repeat
by default when the aggregate win is below `2%` — the same line as GEAK's `MIN_IMPROVE`, which a win below
it does not clear anyway.

## Attributing a change that only creates headroom

Some changes do not make the loop faster — they make a later change *possible*. Measured alone they read as
neutral or negative, and the default acceptance bars then revert exactly the change the next round needed.
Two families:

- **A resource freed but unspent.** Removing work from a region that shares a fixed budget (a conditional
  rescale, a fold that shortens a dependency chain) frees capacity; only something downstream that
  *spends* that capacity converts it into cycles. The tell is a large move in the **resource** metric
  (window fill, register headroom, occupancy) with a flat boundary time.
- **A precondition satisfied.** A structural change that a compiler-realized lever requires but cannot
  itself create (`../tile-programming/compiler-contract.md`), where the lever is not on yet.

**How to attribute instead of reverting.** Keep the pair as **one** attributable step and report both legs:
the enabling change's own measurement (expected: resource moves, time flat) and the combined measurement
(expected: time moves). If the combination does not pay, the negative belongs to the **pair**, not to the
enabler — record it that way so the next round does not re-derive the enabler and re-measure it alone. A
change that is neutral at the boundary and positive on the resource is `retain-as-precondition`, not `win`
and not `revert`; say which lever is supposed to spend it, or it is just an unfalsifiable claim.

**A corollary for schedule work: cycles and wall time can disagree for a real reason.** A denser
matrix-issue stream draws more power, and against a power cap that buys back frequency
(`../hardware/planning-constants.md ## Extended planning (attention / fused kernels)`). So **judge a
scheduling change by cycles and the kernel by both**; a change that improves cycle metrics while flat on
wall time is usually still the right change, and the reverse (time improves, cycles worse) is a signal to
look for a frequency or measurement artifact before believing it.

## IR acceptance (final)

Each closed layer must have its IR/asm signal on file (`../tile-programming/compiler-contract.md`): async
path has no load-path `ds_write`; padded/swizzled LDS shows 16-cyc `ds_read`; llirSched shows interleaved
MFMA; slicing shows no hot-loop spill and reduced `v_accvgpr_mov`.

A closed layer = its tile-op card cell(s) closed: the IR signal above is that cell's `evidence: IR`
(Dispatch for memory path, Layout for LDS, Handoff for pipeline, Scope for slicing), so this is the existing
bar phrased per cell, not a new gate (`../tile-programming/tile-op-contract.md`).

## Closure challenge (self-review before a ceiling / keep-baseline / negative close)

Before the deep_engineer returns an at-ceiling / keep-baseline (`negative_keep_baseline`) / negative
(`negative_revert_plain`) / stay-plain close, it challenges its own claim in writing: **is this close
credible, or did the search stop short of the roofline?** This is a step of the deep_engineer's own loop,
not a role: no other engineer and no extra agent is involved, and it is done entirely from what is on disk.
It is **advisory** — it changes no outcome, no gate exit status and no verify number.

Do it at the close only, never mid-run: it is closure-shaped, and starting one reads as a decision to wrap
up. A `win` does not need one.

**What it reads — on-disk evidence only, never a re-measurement.** A number you cannot reproduce from the
artifacts is a finding to record, not a measurement to re-run.

- the round log — the exploration narrative in `OUTPUT_DIR/report.md` and the per-round trajectory
  ([`records.md`](records.md) `## 1c. Per-round log entry — written as you go, not reconstructed`:
  evidence cited, edit, before/after, keep/revert). This is the primary record of a tool-first pack; there
  are no `round_*/*.json` files and their absence is expected.
- `best_patch.diff` and `checkpoint/` — the kept wins, each with its diff and metrics.
- `worker_result.json` — the claim being reviewed.
- the champion it was handed (`plain_champion.json`) plus its `champion_gate.py` result — a close quoted
  against an unasserted champion is `contradicted` on its face.
- the profiling summary — `profile_engineer`'s baseline `profiling_summary` and the engineer's own
  re-profiles (`OUTPUT_DIR/profile_rN/`), the gap decomposition and the `asm_loop_audit.py` loop audit
  against the comparator.
- the IR dumps (`dump_ir.sh` TTGIR / LLIR / ISA) of the anchor and of the best candidate.
- [`../hardware/optimization-gotchas.md`](../hardware/optimization-gotchas.md) and
  [`climb.md`](climb.md) `## 2. Reversed-intuition traps — read this once`.

The deep track never fans out, so no arm directories is correct, not a gap; a missing artifact this regime
*does* produce (a round with no log entry, a kept win with no checkpoint) is a finding.

**What it writes — `OUTPUT_DIR/closure_review.md`, fixed headings, nothing else.** A fixed shape keeps the
review deterministic and stops it becoming a new ratchet: no numeric score, no minimum objection count,
and it MAY conclude the close is credible.

```text
# Closure review

## Claim reviewed
<the close type + the headline number + the comparator it rests on (champion_ms, its gate result)>

## Strongest supporting evidence
<the 1-3 on-disk facts that BEST support the claim, each with a path>

## Strongest counter-evidence
<the 1-3 on-disk facts that MOST undermine it: a bound the search never attacked, a structural direction
that could raise the ceiling, an unproven wall, a comparator that was not tuned. Cite paths. If there are
none, say "none found" and why the search looks complete.>

## Untried or ambiguous alternatives
<specific directions the trajectory did NOT try that could plausibly beat the result: a structural
reformulation (split-K / stream-K / defuse / fusion split) -- which comes back as `structure_suspect`, not
a climb -- a #2-bound lever, an off-card idea. Reason from THIS kernel's bound class and roofline gap. If
the roofline gap is already small, say so.>

## Gotcha checks
<walk optimization-gotchas.md and climb.md `## 2. Reversed-intuition traps`. For each RELEVANT row, state
whether the trajectory fell into it or correctly avoided it, with a path; skip the rest with a reason.>

## Bundle attribution
<did any round glue UNRELATED levers into one step so a loser could hide inside a winner? Were the coupled
levers genuinely coupled (async + layout, enabler + payoff, a shared-budget tri-lemma) -- attribution
intact -- or batched to save rounds?>

## Verdict: credible | weak | contradicted
<one word + one sentence. credible = well-earned, no material hole found. weak = plausible, but a named
alternative should be tried first. contradicted = an on-disk fact directly undercuts the claim.>

## Recommended next experiment
<if weak / contradicted: the single most valuable experiment before the close stands. If credible:
"none -- record the close".>
```

**Discipline.** The author of the claim is also its reviewer, so argue against your own result: write the
counter-evidence as if someone else had made the claim. Every statement cites a path. The headings are
scaffolding; the value is the specific untried direction a fixed rule list would miss. Do not manufacture
objections to look thorough — a search that genuinely reached the roofline should read `credible`.

**What it returns and who reads it.** The deep_engineer puts the review into its GEAK `worker_result`
notes — `closure_review` path, `verdict`, the one-line `strongest_counter`, the one-line
`recommended_next` — and owes a disposition for each counter-evidence item: `accepted` (it ran the
recommended experiment, budget permitting, before returning) or `rebutted` *with the on-disk reason it does
not apply*. **tech_lead** reads it before planning the next round: a `weak` / `contradicted` verdict with
budget left is a candidate for the next direction. **Director** quotes the strongest counter-evidence in its
`arbitration_note` when it arbitrates the result. If the engineer keeps the pack's optional
`final_report.json`, the same content goes into `skeptic_verdict` (`credible | weak | contradicted |
not_run`) and `skeptic_items[]` (upstream schema field names, kept for `report_lint.py`), with
`verified_by: self` — a self-check and an independent check leave the same artifacts behind, and a review
whose verdict is recorded but whose challenges are not itemized reads exactly like agreement, which is the
one reading it must never be able to produce.

## Process provenance: where each producer receipt goes

*Optional bookkeeping.* This section and the next apply only when the deep_engineer keeps the pack's
canonical record (`canonical_record.py`, `close_audit.py`, `report_lint.py`) beside GEAK's own record; in
GEAK the acceptance evidence is `verify_engineer`'s re-benchmark and Director's validation, not these
receipts. When you do keep the record, keep it honestly:

Four artifact roles owe a process-time receipt at a front-end close (`sweep`, `champion`, `audit`,
`final_report`); a deep consumer owes the last three and a direct owner the last two. A missing one is
`producer_receipt_missing` at **error** severity, which drops the result below conditional acceptance no
matter how good the measurement is. This section exists because that is the single most common way a
measured win fails to close — provenance, not performance.

**Write each receipt when its artifact is finished, not at the close.** A receipt rebuilt at finalization is
honest only if it says so, and saying so is the `late_reconstruction` finding — which is unwaivable. The
receipt is what proves the artifact existed at process time; a reconstruction proves only that the file
exists now.

```bash
# as each artifact lands, naming the tool that produced it
python3 scripts/canonical_record.py write-producer-receipt \
  --artifact <work>/<comparator artifact> --artifact-role sweep    --executable <the sweep tool>
python3 scripts/canonical_record.py write-producer-receipt \
  --artifact <work>/<champion bundle>     --artifact-role champion --executable <the champion tool>
python3 scripts/canonical_record.py write-producer-receipt \
  --artifact <work>/close_audit.json      --artifact-role audit    --executable scripts/close_audit.py
```

Each writes a **sidecar** next to its artifact (`<artifact>.receipt.json`) holding that artifact's sha256.
Then carry them into the report — and only them:

```bash
python3 scripts/canonical_record.py write-final-report --out <work>/final_report.json ... \
  --producer-receipt <work>/<comparator artifact>.receipt.json \
  --producer-receipt <work>/<champion bundle>.receipt.json
# close_audit.json.receipt.json is picked up automatically from --close-audit-ref
```

**Do not pass a receipt for `final_report.json` itself.** `write-final-report` writes that one, as a
sidecar, after the report bytes are on disk — which is the only order that can work. A receipt embedded
inside `provenance.receipts[]` that covers `final_report.json` records a hash of bytes the embedding then
changes, so it is stale the instant it lands and every later verification reads it as
`canonical_entry_invalid` forever. No write order fixes it and the command now refuses it. This is worth
stating plainly because the embedded route has twice been reported upstream as proof that
"self-referential receipts can never validate": the *embedded* one cannot, the sidecar always can, and the
runs that used the sidecar closed with zero high findings.

Verify before you close, rather than discovering it in the audit:

```bash
python3 scripts/canonical_record.py project --report <work>/final_report.json \
  --work-root <work> --out /tmp/projection.json --strict
```

`integrity.findings == []` and `integrity.eligible_clean == true` mean the provenance chain is complete.
Anything else names the missing role.

## final_report.json (schema)

**The authoritative shape is `runtime/final-report-v3.schema.json`, and the authoritative *check* is
`scripts/report_lint.py`.** They are not the same thing and the lint is the stricter one. The schema
(`schema: kernel_opt.final_report/3`) requires `schema`, `run_id`, `generation`, `lifecycle_profile`
(`front-end | deep-consumer | direct-owner | blocked` — this pack is normally `deep-consumer`), `stage`,
`outcome`, `arbitration`, `status`, `refs`, `audit`, `provenance.receipts`, `integrity`
(`source_schema`, `normalization_status`, `legacy_unverified`, `eligible_clean`), `caveats`, `deferred`
and `known_wrong`; `measurement`, `headline` and `served_range` are optional there. The lint additionally
enforces fields the schema leaves optional (`measurement`, `scope`, `refs.canonical` / `.champion` /
`.close_audit`, `provenance.receipts`, `skeptic_verdict` + `skeptic_items`, `close_audit` + `verified_by`,
and a `stage` / `status` / `outcome` drawn from `canonical_record.PHASES` / `RUN_STATUSES` / `OUTCOMES`).
Write the report with `canonical_record.py write-final-report` as shown above, then run `report_lint.py`
and fix what it names — do not hand-build the object from a schema listing, and do not treat a
schema-valid document as a closeable one.

An earlier revision of the evaluate page inlined a full field-by-field JSON example. It drifted out of
agreement with both of those, so it is deliberately not reproduced here: one copy of a contract is
auditable, two are a guess about which one is current. The human-readable twin is the Final Delivery
Record in [`records.md`](records.md) (`## 8. Final Delivery Record (1:1 with final_report.json)`).

**Required result, whatever the role writes.** The terminal result names status, comparator and boundary,
best measurement, correctness, stage and round budget spent, checkpoint/evidence refs, caveats, deferred
work and every audit disposition. An unmeasured or unverified candidate is a blocker/unknown, not a win.

**Budget burn-down is part of the delivered report, not just an internal ledger.** `budget.{N, spent,
remaining}` make `timeout` vs `partial` auditable from the artifact alone. They are the same counters the
self-monitoring thresholds run on (warn / wrap-up / timeout at 70 / 85 / 95 % of N —
[`records.md`](records.md) `## 6. Self-Monitoring / Insight Buffer`, [`climb.md`](climb.md)
`## Self-monitoring`): `outcome: timeout` must be consistent with `budget.remaining == 0`.

## Output

```text
optimized/<best gluon source> + best_patch.diff   (only if outcome == win)
final_report.json + summary.md
```

On `partial`, deliver the best in-scope result plus the `deferred[]` list (what the user must resolve + the
resume entry, [`records.md`](records.md) — the Residual / Deferred-Task & Resume ledger). **Resume**
re-optimizes after the user resolves the blocker and **merges/updates** the existing `final_report.json` in
place; when nothing remains deferred, `outcome` graduates from `partial` to `win`.

On `negative_revert_plain`, keep the plain baseline as the delivered result and record why Gluon did not pay
off (which layers were blocked, the residual gap, `parity_unreached` if it applies).

In GEAK the deliverable is GEAK's — the committed patch (Merge) and the Report / Validate phases' records.
Carry this page's verdict, `caveats[]` (`parity_unreached`, `injected`, served-range status) and the closure
self-review's verdict and dispositions into the deep_engineer's `worker_result` notes and `report.md`, where
tech_lead and Director read them.

## 6. Cost discipline — last, because it decides nothing

This section is genuinely last, and nothing binding sits below it: the hard stops, the acceptance bars and
the `outcome` / `worker_result.stage` enums are **above**, because a run that stops reading before them
discovers at close that its record will not lint. Everything above is the method; this section only makes
it cheaper. **None of it changes the round budget, the search, or a single verdict** — the denominator is
still the round count, and "I was saving context" is never a reason to skip a dial, a recovery attribution,
or a gate.

Keep the runtime context to the current decision and its evidence. Three rules prevent stale or unnecessary
material from distorting the method:

1. **Use the completion protocol for long experiments.** `scripts/wait_for.sh --launch` / `--await` records
   the command outcome. Do not infer completion from a blind sleep or a launch receipt.
2. **Compact a round the moment it is disposed.** A kept / reverted / negative direction already has its full
   evidence in `decision_log.md` / `checkpoint/` (declared sufficient to cold-resume). Keep one line in live
   context; do not let a reverted round's profile dump ride every later turn.
3. **Read only routed evidence.** Keep the resident contract and current record; load targeted excerpts from
   large artifacts only when the active claim requires them.

Session-level context management (lease, compaction, resume capsule) is [`orchestration.md`](orchestration.md).

## Sources

Merged from: `references/phases/evaluate.md` (all sections except `## Two baselines`, which went to
`recover.md`); `references/method-reference.md` `## Stop conditions` and `## 6. Cost discipline — last,
because it decides nothing`, plus the close-audit bullet of `## Non-Negotiable Rules` ("A close records who
checked it…", also kept in `index.md`); `tile-programming-gluon.md` "**Hard stops.**" and "**Required
result.**" paragraphs and its compressed `## Stop conditions` / `## 6.`; the upstream
`amd-closure-skeptic` agent contract (inputs, artifact set, output headings, verdict enum, discipline, return
fields), now `## Closure challenge` — the agent file itself is not shipped; `runtime/final-report-v3.schema.json`; the enums in
`scripts/canonical_record.py` / `scripts/report_lint.py`.

Conflicts resolved:
- `result: partial | timeout | win` (evaluate.md) → `outcome:` (v3 schema / `canonical_record.OUTCOMES`);
  "record a negative result and revert to plain" → `negative_revert_plain`; deferred → `outcome: partial`
  + `status: deferred_needs_user`.
- Acceptance timing and commit gate: acceptance is GEAK's (`verify_engineer`, harness_lib, fresh process per
  leg, `MIN_IMPROVE` 2%, noise band only stricter; Director validates and arbitrates).
- Re-injected numbers labelled `injected`, never a win (pipeline priority rule).
- The closure skeptic is no longer a separate agent launched by a captain: it is the deep_engineer's own
  closure self-review (`closure_review.md` in OUTPUT_DIR), read by tech_lead and quoted by Director.
  Dropped from the agent contract: the gated-pack artifact regime (`context.json`, `round_*/*.json`,
  `t0/**/arm_result.json`, `run_round.sh` notices), the triton-only branch-arm and `plain_champion.json`
  bundle-field checks, `mode: mid_run`, the `dsl` / `skill_root` brief fields, and the separate return
  JSON (its fields now ride in the worker_result notes).
- The pack's canonical record / receipts / `final_report.json` demoted to optional bookkeeping; the run's
  run-mode split (pack stage spine vs GEAK) removed — GEAK's phases are the stage machine.
- Mechanics step 3 "clear" → move aside (no `rm` in scripts GEAK roles run).

Dropped: nothing. Pointers updated: `../experiment-records.md` → `records.md`; `../failure-triage.md` →
`triage.md`; `../benchmark-hygiene.md`, `../phases/harness.md` → `benchmark-hygiene.md`;
`../plain-strategies.md` → `front-end.md`; `layer-loop.md` → `climb.md`; `transcribe.md` (parity gate) →
`recover.md`.
