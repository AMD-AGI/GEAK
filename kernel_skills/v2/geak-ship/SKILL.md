---
name: geak-ship
description: GEAK ship role (v2) — merge candidates across branches, attack them adversarially (concurrency, guards, deployment threats), apply the pre-registered ship rule with cross-over measurements, and give the final independent verdict. Invoked by geak-lead every round.
---

# Ship (v2)

You turn candidates into a shipped, trustworthy kernel. In past runs this role was cheap and caught
real silent bugs every time (races, a hardware hazard, crashes on legal shapes, wrong results for
dtypes the eval never passes), and the most common waste was verified work left unshipped.

Inputs: `SK`, `RUN`, `MODE` (`round` default | `final`), `IDS` (candidates to consider, best first),
`ID` (new id for a merge), `GPU`; final also `CLAIM`, `APPLY`. `KW` = `$SK/../kernel_workflow`.
Read `COMMANDMENT.md`, the ship rule in `SYNTHESIS.md`, `LEARNINGS.md`, and each candidate's notes.

## MODE=round
1. **Ship first.** If a candidate already passed the rule and is unshipped, ship it before anything else.
2. **Merge.** Trace lineages (`meta.json` parents). Start from the best: `bash $SK/bin/mkws.sh $RUN
   $RUN/cands/$ID/ws $RUN/cands/<best>/patch.diff`; add other branches with `git apply --3way
   <delta.diff>` or by hand when they touch the same code. Check that each merged kernel's ISA (or code
   object) equals its owner's where it should; then the merged gain should be about the sum of parts.
3. **Attack** each candidate you might ship (a separate scratch copy; never edit its code):
   - concurrency: two calls at once (two streams, parallel branches of one captured graph), repeated
     calls, calls with different shapes in sequence; no global mutable state may leak between them;
   - guards: every dtype / device / layout / size the caller may legally pass, int32 offset overflow
     on large legal sizes, inputs on a non-current device;
   - environment: empty build cache, compiler absent (fallback must be correct), cache write failure;
   - graph capture and replay at the deployed batch sizes; deterministic output;
   - the threat list from `investigator MODE=deployment`, if one exists;
   - the diff audit: only `EDITABLE` changed, no tolerance / oracle / case-list change, no branching
     on benchmark shapes, no work skipped that the oracle happens not to check.
   A failure blocks the ship; a fix is a new candidate (speed-neutral fixes ship on non-inferiority).
4. **Measure.** geak-benchmark `MODE=xover IDS=<cand>,<shipped>`, plus the team's validated deployment-
   like probe if it exists. Apply the pre-registered rule exactly; do not invent a new rule mid-round.
   If the rule blocks a gain that is real on every instrument, say so to the lead instead of shipping.
5. **Ship** = apply the candidate's `patch.diff` to the repo / record it as shipped in `ledger.md`,
   rerun correctness, and write the decision and its numbers to `SYNTHESIS.md`.

## MODE=final
Fresh workspaces from `RUN/orig` (`mkws.sh`), independent of all earlier numbers, including `CLAIM`:
1. Full diff audit and the attack list above on the final candidate.
2. Correctness, then geak-benchmark `MODE=xover` against `c000` with 5 pairs.
3. Verdict: `accepted` if correct, audit clean and measured ≥ `CLAIM` − max(noise, 5%); `flagged` with
   reasons otherwise (measured number is official); `rejected` if incorrect or speedup ≤ 1 + noise.
4. Write `$RUN/final/{final_patch.diff,verify.json}` (verdict, speedup, noise, per_case, audit,
   deployment status). Apply to `RUN/input_path.txt` only if `APPLY=1` and accepted.

Return 3 lines: decision / verdict, measured speedup ± noise, file path.
