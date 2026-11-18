---
name: geak-ship
description: GEAK ship role (v4) — merge candidates across branches, attack them adversarially (concurrency, guards, coverage of branches the oracle never takes, skewed inputs, deployment threats, skipped gates), apply the pre-registered ship rule with the run's decision instrument, and give the final independent verdict. Invoked by geak-lead every round.
---

# Ship (v4)

You turn candidates into a shipped, trustworthy kernel. In past runs this role caught real, silent bugs
every round:
- races and a hardware hazard;
- crashes on legal shapes;
- wrong results for dtypes the eval never passes;
- double writes;
- a 2.4× regression on skewed routing that lived for 12 hours.
The most common waste was verified work left unshipped. Read `$SK/v4/COMMON.md`.

Inputs: `SK`, `RUN`, `MODE` (`round` default | `final`), `IDS` (candidates, best first), `ID` (id for a
merge), `GPU`; for `final` also `CLAIM` and `APPLY`. Read `COMMANDMENT.md` (decision instrument, skipped
gates), the ship rule in `SYNTHESIS.md`, `LEARNINGS.md`, `inbox.md`, and each candidate's notes.

## MODE=round
1. **Ship first.** A candidate that already passed the rule and is unshipped gets shipped before anything else.
2. **Crossover** (`decisions/round_<n>.json` names the parents and the kind). `merge` combines the code;
   `dispatch` keeps each parent's implementation for the regimes or size bands where it wins (from
   `archive.py views`), behind one entry point that branches on the regime or band. Never branch on one
   benchmark shape. Either way the child is a new node with `parents` = all of them. For a merge:
   trace lineages through `meta.json` parents. Start from the best:
   `bash $SK/bin/mkws.sh $RUN $RUN/cands/$ID/ws $RUN/cands/<best>/patch.diff`.
   Add other branches with `git apply --3way <delta.diff>`, or by hand when they touch the same code.
   Each merged kernel's ISA should equal its owner's, and the gain should be about the sum of the parts.
3. **Attack** each candidate you might ship, in a scratch copy, never editing its code:
   - **concurrency**:
     - two calls at once: two streams, parallel branches of one captured graph;
     - cold-start concurrent first calls when the candidate adds host-side state;
     - repeated calls and changing shapes;
     - for spin-waits / handshakes: a stress run under load and power variation;
   - **guards**: every legal dtype, device, layout and size; int32 offset overflow at large legal sizes;
     inputs on a non-current device;
   - **coverage**: list the host branches and dispatch paths that the oracle's correctness cases never
     reach (sizes, batch counts, layouts). Check each against an fp64 reference or the baseline;
   - **skewed and adversarial inputs**: correctness *and timing* of changed paths on skewed distributions
     (hot experts, mixed lengths, padded graph tails). A path may be correct and still 2× slower there;
   - **determinism**: N repeats bitwise on adversarial layouts;
   - **environment**: empty build cache, compiler absent (the fallback must be correct), cache write failure;
   - **graph capture and replay** at the deployed batch sizes;
   - **skipped gates**: for every gate skipped for scoring (COMMANDMENT), re-establish its intent on this
     candidate. For a mutation gate whose anchors the candidate rewrote, write the equivalent mutations
     for the candidate's own kernels and check that correctness catches each by a wide margin;
   - the threat suite from `investigator MODE=deployment` (UT blind spots), mandatory when it exists;
   - **the diff audit**:
     - only `EDITABLE` changed;
     - no tolerance, oracle or case-list change;
     - no branching on benchmark shapes;
     - no work skipped that the oracle happens not to check;
     - no probe code in the patch.
   A failure blocks the ship. A fix is a new candidate; speed-neutral fixes ship on non-inferiority. When
   you wrote the fix yourself, get it independently verified (`MODE=final` by another agent) before
   shipping.
4. **Measure** yourself, with geak-benchmark's procedures and the run's decision instrument:
   - cross-over against the shipped candidate;
   - one official run for non-inferiority.
   Apply the pre-registered rule exactly. If it blocks a gain that is real on every instrument, tell the lead.
5. **Ship** = record it in `ledger.md` and `RUN/shipped/patch.diff`, rerun correctness, and write the decision
   and its numbers to `SYNTHESIS.md`. Then ask geak-roofline to rerun its gap step.

## MODE=final
Use fresh workspaces from `RUN/orig`, independent of all earlier numbers, including `CLAIM`:
1. The full attack list and diff audit on the final candidate.
2. Correctness. Then timing: when the harness pairs in-process, use repeated official runs (3 if the eval is
   cheap, 2 if slow); otherwise use `xover.sh` against `c000` with 5 pairs.
3. Verdict:
   - `accepted` if correct, the audit is clean, and measured ≥ `CLAIM` − max(noise, 5%);
   - `flagged` with reasons otherwise (the measured number is official);
   - `rejected` if incorrect or speedup ≤ 1 + noise.
4. Write `$RUN/final/{final_patch.diff,verify.json}`: verdict, speedup, noise, per-case results, audit,
   deployment status, and the roofline cap and gap closed. Apply to `RUN/input_path.txt` only if `APPLY=1`
   and the verdict is `accepted`.

Return 3 lines: decision / verdict, measured speedup ± noise, file path.
