# R03 candidate-to-finalist funnel

This audit joins already-exported completed R03 records. It identifies where evidence stops for each observed direction; it does not treat every unsuccessful direction as a runtime failure. All eight lineages are represented in the [per-direction CSV](r03-candidate-funnel-directions-20261028.csv) and [evidence JSON](r03-candidate-funnel-20261028.json): **81 observed direction identifiers plus four integration attempts**. This is a timeline/archive union, not a census of all proposed ideas or distinct source programs.

## What the funnel establishes

- Four hybrid directions failed during parent initialization before engineer dispatch: B r2_d0/d1/d2 and D r2_d1. Their synthetic failed statuses are not executed engineer failures. All four selected parents have the recorded SHA256 of a zero-byte patch; this is a stored digest comparison, not a fresh file stat.
- Ordinary GEAK has six engineering calls without a truthy outer return; the hybrid has eleven, in addition to the four pre-dispatch failures. The source files may still exist. E-c r2_d2 and E-d r1_d0 had no retained engineer result but recovered valid, gain-eligible verifier results.
- Seven direction rows carry retained correct/gain-eligible verifier results below the illustrative source-default promotion gate. These are arithmetic comparisons, not exported per-decision rejection reasons. One, E-c r2_d1, repeats the already selected candidate ID under another evaluation.
- Six other distinct retained verifier candidate IDs with small recorded gains are absent from the common-final candidate set. They have not received separate common qualification. The audit therefore establishes no common-qualified source discarded by the selector.
- E-d r1_d1 explicitly records invalid_measurement / correctness=not_checked. Other missing verifier returns/evidence remain unresolved; they are not rewritten as numerical-correctness failures. All four integrator calls returned truthy; only E-c r1 has a retained integrated candidate in these exports. The other three outcomes cannot be classified as integration failures from return flags or absent rows.

## Eight-lineage coverage

Each call column is attempts / truthy returns. A truthy return does not establish usability or correctness. A missing return does not establish no candidate artifacts.

| Population / lane | Observed direction IDs | Engineer | Verify | Integrate | Init failure before dispatch | Commits |
|---|---:|---:|---:|---:|---:|---:|
| I / a | 12 | 12 / 11 | 7 / 6 | 0 / 0 | 0 | 1 |
| I / b | 9 | 9 / 7 | 5 / 4 | 0 / 0 | 0 | 1 |
| I / c | 9 | 9 / 8 | 8 / 8 | 1 / 1 | 0 | 1 |
| I / d | 10 | 10 / 8 | 7 / 7 | 0 / 0 | 0 | 0 |
| E / a | 12 | 12 / 8 | 12 / 10 | 0 / 0 | 0 | 0 |
| E / b | 11 | 8 / 7 | 8 / 7 | 1 / 1 | 3 | 0 |
| E / c | 9 | 9 / 5 | 9 / 5 | 2 / 2 | 0 | 1 |
| E / d | 9 | 8 / 6 | 8 / 5 | 0 / 0 | 1 | 0 |

I is the external four-ordinary population; E is the hybrid ordinary A plus EVO B/C/D population. All eight finals report stopping at their deadline. The ordinary per-direction source, verification and promotion records are not present in these exports; their table cells remain unknown. E-a also uses the ordinary policy and has no EVO archive.

## EVO retained directions

`T`/`F` are outer returned-result flags, not pass/fail kernel verdicts. `—` means no corresponding timeline event in the complete exported lane timeline. A score in this table is retained verifier/archive evidence, not the engineer’s claim or a new common measurement. `ready child` is a later recorded parent initialization, not proof that the child succeeded.

| Lane / row | Parent → init | Eng / verify return | Retained result | Promotion / final relationship | Later use recorded | Candidate ID |
|---|---|---|---|---|---|---|
| b r1_d0 | ORIGINAL → ready | T / T | unverified; recorded empty patch digest | not promoted; admission unknown | init failed: r2_d0 | not exported |
| b r1_d1 | ORIGINAL → ready | F / T | unverified; recorded empty patch digest | not promoted; admission unknown | init failed: r2_d1 | not exported |
| b r1_d2 | ORIGINAL → ready | T / T | unverified; recorded empty patch digest | not promoted; admission unknown | init failed: r2_d2 | not exported |
| b r2_d0 | r1_d0 → pre-engineer failure | — / — | unverified | not promoted; admission unknown | no later ready-parent edge | not exported |
| b r2_d1 | r1_d1 → pre-engineer failure | — / — | unverified | not promoted; admission unknown | no later ready-parent edge | not exported |
| b r2_d2 | r1_d2 → pre-engineer failure | — / — | unverified | not promoted; admission unknown | no later ready-parent edge | not exported |
| b r3_d0 | ORIGINAL → ready | T / T | slower 0.949089× | not promoted; admission unknown | ready: r4_d1 | not exported |
| b r3_d1 | ORIGINAL → ready | T / T | slower 0.994713× | not promoted; admission unknown | ready: r4_d0 | not exported |
| b r3_d2 | ORIGINAL → ready | T / T | slower 0.946959× | not promoted; admission unknown | no later ready-parent edge | not exported |
| b r4_d0 | r3_d1 → ready | T / T | unverified | not promoted; admission unknown | no later ready-parent edge | not exported |
| b r4_d1 | r3_d0 → ready | T / F | unverified | not promoted; admission unknown | no later ready-parent edge | not exported |
| c r1_d0 | ORIGINAL → ready | T / T | slower 0.996346× | not promoted | donor ref: r1_integrated,r2_d0 | e3886377d043 |
| c r1_d1 | ORIGINAL → ready | T / T | correct 1.015382× | below gate arithmetic; unconfirmed | donor ref: r1_integrated,r2_d1 | 5fb69e5a4edf |
| c r1_d2 | ORIGINAL → ready | T / F | unverified | not promoted; admission unknown | ready: r2_d2 | not exported |
| c r1_integrated | ORIGINAL → integration N/A | — / — | correct 1.023728× | promoted; common-qualified | ready: r2_d0,r2_d1,r3_d1; donor ref: r2_d2 | 0ccf17175927 |
| c r2_d0 | r1_integrated → ready | T / T | correct 1.026041× | below gate arithmetic; unconfirmed | ready: r3_d0 | 50924071060b |
| c r2_d1 | r1_integrated → ready | T / T | correct 1.018585× | same candidate as finalist | donor ref: r3_d1 | 0ccf17175927 |
| c r2_d2 | r1_d2 → ready | F / T | correct 1.009846× | below gate arithmetic; unconfirmed | ready: r3_d2 | 2100c1eb0883 |
| c r3_d0 | r2_d0 → ready | F / F | unverified | not promoted; admission unknown | no later ready-parent edge | not exported |
| c r3_d1 | r1_integrated → ready | F / F | unverified | not promoted; admission unknown | no later ready-parent edge | not exported |
| c r3_d2 | r2_d2 → ready | F / F | unverified | not promoted; admission unknown | no later ready-parent edge | not exported |
| d r1_d0 | ORIGINAL → ready | F / T | correct 1.003278× | below gate arithmetic; unconfirmed | ready: r3_d0; donor ref: r2_d0 | 0b9c082442bc |
| d r1_d1 | ORIGINAL → ready | T / T | unverified; invalid measurement; recorded empty patch digest | not promoted; admission unknown | init failed: r2_d1; donor ref: r3_d1 | not exported |
| d r1_d2 | ORIGINAL → ready | T / F | unverified | not promoted; admission unknown | ready: r2_d2 | not exported |
| d r2_d0 | ORIGINAL → ready | T / F | unverified | not promoted; admission unknown | donor ref: r3_d0 | not exported |
| d r2_d1 | r1_d1 → pre-engineer failure | — / — | unverified | not promoted; admission unknown | no later ready-parent edge | not exported |
| d r2_d2 | r1_d2 → ready | T / T | correct 1.009423× | below gate arithmetic; unconfirmed | ready: r3_d2 | c55eddcb0c0d |
| d r3_d0 | r1_d0 → ready | T / T | slower 0.995973× | not promoted | no later ready-parent edge | 1f936de3fe29 |
| d r3_d1 | ORIGINAL → ready | F / F | unverified | not promoted; admission unknown | no later ready-parent edge | not exported |
| d r3_d2 | r2_d2 → ready | T / T | correct 1.009483× | below gate arithmetic; unconfirmed | no later ready-parent edge | 1d03ea657f4c |

“No later ready-parent edge” only describes the retained archive. It does not exclude free-form reading, donor use, failed initialization or unrecorded development. All 30 rows were retained at the endpoint; 29 are direction records and C r1_integrated is an integration record. Ready initialization includes exact initial tree/commit and parent-patch references in the JSON.

## Recorded gains absent from final admission

| Row | Retained verifier score | Later ready child | Common-qualified? |
|---|---:|---|---|
| E-c r1_d1 | 1.015382× | none recorded | unavailable |
| E-c r2_d0 | 1.026041× | r3_d0 | unavailable |
| E-c r2_d2 | 1.009846× | r3_d2 | unavailable |
| E-d r1_d0 | 1.003278× | r3_d0 | unavailable |
| E-d r2_d2 | 1.009423× | r3_d2 | unavailable |
| E-d r3_d2 | 1.009483× | none recorded | unavailable |

Four of these six gain candidates have later ready children with recorded outcomes:

| Retained gain → ready child | Child evidence |
|---|---|
| C r2_d0 → r3_d0 | No truthy engineer or verifier return; retained unverified |
| C r2_d2 → r3_d2 | No truthy engineer or verifier return; retained unverified |
| D r1_d0 → r3_d0 | Retained verified slower result, 0.995973× |
| D r2_d2 → r3_d2 | Retained verified small gain, 1.009483×, below the default promotion gate |

No retained verified descendant above the gate is observed along these four paths. This is not a claim of no actual result: two paths have measured descendants, and missing returned results on the other two do not establish absence of source work.

C r2_d1 is excluded from this six-candidate list because its candidate ID equals the selected r1_integrated candidate. Different evaluation IDs and scores are not distinct source programs. The final selector enumerates published checkpoints and lineage finals; retention alone does not admit every archive program. These six search-time gain records therefore expose a retention-to-delivery boundary, without establishing that they would beat the selected source under common final measurement.

The strict default promotion rule is candidate score > cached incumbent × 1.02. C r2_d0 at 1.026041 is below 1.023728 × 1.02 = 1.044202. D’s three small gains are below a fresh default cumulative-1 gate of >1.02. The measured r0 score is a separate observation and is not substituted for initial cumulative. Exact historical gate state/rejection messages were not exported.

## Verification and integration limits

- No retained EVO row is labelled incorrect. Unverified does not mean numerically incorrect or a disproved idea.
- E-d r1_d1 has a truthy verifier return but an explicit invalid measurement with correctness not checked. This is a concrete unusable verification result.
- Ten hybrid and two ordinary verification calls have no truthy outer return. This identifies unavailable returned verification, not its runtime cause or whether a GPU test ran.
- I-c r2, E-b r3 and E-c r2 each have a truthy integration call without a joined retained integrated candidate. The source requires an improved conclusion and a checked score above the best individual before recording integration. Their actual conclusions/dispositions are not exported.
- D r1_d1’s recorded engineer candidate ID equals the ORIGINAL candidate ID (`9befb93fc09593295428cfe74d6e39a0eedab0350b1e7f38e627400ccc322241`). The B parent candidate identities remain unavailable in these exports. Neither the empty digest nor the D identity match establishes that no source was ever produced.
- Engineer-reported verified status must not replace verifier evidence: B r1_d0/r1_d2 and C/D r1_d2 illustrate successful/claimed engineering evidence that remained unverified at the archive boundary.

No utilization, token cost, provider usage, wait time or effect on final performance is inferred. Source/readout scope, all full identities, JSON pointers, overlapping category states and unavailable fields are in the companion JSON. Existing historical artifacts and source were not changed. Actual Claude reviewed the Markdown and CSV and requested the narrow digest, identity and descendant-outcome additions above. Those additions are applied; full-JSON review is not implied.
