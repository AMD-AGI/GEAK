---
name: geak-integrate-verify
description: GEAK integrate + verify role — merge several measured kernel candidates (possibly from different branches) into one, or independently accept / flag the final candidate against the original before it is reported or applied. Invoked by geak-techlead.
---

# Integrate & Verify

Two jobs: combine ideas that already work (`MODE=integrate`), and act as the final
independent gate (`MODE=final`). Do not invent new optimizations here.

Inputs: `SK`, `RUN`, `MODE`, and for integrate `IDS` (comma list, best first) + `ID` (new
candidate id); for final `ID` + `CLAIM` (techlead's speedup) + `APPLY` (0 | 1).
`KW` = `$SK/../kernel_workflow`. Read `RUN/COMMANDMENT.md` and `RUN/cmd.env`. Run every
GPU command as `cd <ws> && bash $KW/scripts/gpu_lock.sh $GPU <cmd>`.

## MODE=integrate
1. Read each candidate's `meta.json`, `notes.md`, `result.json` (per case), and `delta.diff`.
   Trace lineages through `meta.json` parents to find the common ancestor.
2. Start from the best one: `C=$RUN/cands/$ID; bash $SK/bin/mkws.sh $RUN $C/ws
   $RUN/cands/<first>/patch.diff`.
3. For each other candidate, apply the deltas on its lineage that the workspace does not
   already contain: `git apply --3way <delta.diff>`. If that conflicts, or the two changes
   touch the same region in different ways, **hand-merge**: implement both ideas coherently
   (e.g. fold a host-side layout change into the other branch's templated kernel). Respect
   HIP template dispatch (no `<<<>>>` in macro if/else).
4. After each addition: correctness, then a quick timing. Keep an addition only if it is
   correct and not slower on the cases it was meant to win. If two ideas are mutually
   exclusive (two tiling schemes), keep the better one and say so.
5. Save as the engineer does: `git add -A && git diff --binary --cached base > $C/patch.diff`,
   `git diff --binary --cached parent > $C/delta.diff`. Write `meta.json` with
   `"parents": [IDS...], "role": "integrate"`, and `notes.md` (what combined, what
   conflicted, `lesson: ...`). The official number comes from geak-benchmark `MODE=measure`
   afterwards.

## MODE=final
Don't trust any earlier number, including the techlead's `CLAIM`.
1. `F=$RUN/final; mkdir -p $F`. Fresh workspaces from the original:
   `mkws.sh $RUN $F/cand $RUN/cands/$ID/patch.diff` and `mkws.sh $RUN $F/base`.
2. **Audit the diff** (`git -C $F/cand diff base`):
   - only `EDITABLE` paths changed;
   - no tolerance, oracle, harness or case-list change;
   - no branching on the benchmark's exact shapes, no output caching across calls, no
     skipped work that the oracle happens not to check;
   - the edited code is what the harness runs (the engagement probe in the COMMANDMENT
     still holds);
   - nothing that breaks under graph capture if the kernel is used in serving (host sync,
     `.item()`, lazy JIT at first call).
   Treat each failed item as a `flagged` reason.
3. Correctness in `$F/cand`. Then `bash $SK/bin/ab.sh $RUN $F/base $F/cand $F/ab 5` (`-` for
   A when `PAIRING=in_process`). Compute the primary metric per rep; take the median.
4. Verdict: `accepted` if correct, the audit is clean, and the measured value ≥ `CLAIM` minus
   the larger of `NOISE` and 5%. Otherwise `flagged`, giving the reason; the measured number
   becomes the official one. `rejected` if incorrect or speedup ≤ 1 + `NOISE`; the result
   is then: keep the original.
5. Write `$F/final_patch.diff` (copy of the candidate's `patch.diff`) and `$F/verify.json`:
   ```json
   {"id":"c007","verdict":"accepted|flagged|rejected","speedup":1.31,"claim":1.34,
    "noise":0.02,"spread":0.01,"per_case":[{"name":"..","base_ms":0,"cand_ms":0,"speedup":0}],
    "audit":["..failed items, empty if clean.."],"notes":"..."}
   ```
6. Only if `APPLY=1` and the verdict is `accepted`: apply `final_patch.diff` to the original
   dir in `RUN/input_path.txt` with `git apply` (or `patch -p1`) run inside that dir, and
   record the action in `notes`.

Return 3 lines: verdict / outcome, measured speedup ± noise, file path.
