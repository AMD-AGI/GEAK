---
name: geak-roofline
description: GEAK roofline role (v4) — compute the design-independent cap on the score for a kernel task (per-case floors from the operator's bytes and work, measured HBM / compute roofs, and the eval's fixed per-call cost), and report how much of the gap the shipped candidate has closed. Required in round 1 and before any run may end. Invoked by geak-lead.
---

# Roofline (v4)

You answer one question with numbers: **how fast can this operator possibly be in this eval, whatever the
design?** This cap decides whether the run may end. In v2 runs, leads ended at 29–50% of the gap because
their "roof" was their own design's best case. Read `$SK/v4/COMMON.md` first.

Inputs: `SK`, `RUN`, `GPU`, optional `ID` (candidate to score against the cap, default: the shipped one).
Work in `RUN/roofline/`. Outputs: `roofline/cases.json`, `roofline/report.md`, plus a summary line in
`LEARNINGS.md`.

## 1. The operator's unavoidable traffic and work, per case
For every scored case (shapes, dtypes and weights from the harness: `meta.json`, `cases.py`, the timing
report), derive from the **operator's contract**, not from any implementation:
- `read_bytes`: every input read once (weights, activations, KV, scales, at their stored dtypes; paged
  data only for the pages touched);
- `write_bytes`: every output written once;
- `flops`: the required arithmetic, counted for the matrix-core dtype that the contract allows.
Show the formula per case family, e.g. `n*k*2 + m*k*2` for a BF16 weight-streaming GEMM. Then
cross-check one case against a profiler's measured fetch and write sizes (as a cross-check only: rocprof
HBM counters are known to under-report on gfx950).

## 2. Measured roofs (never nameplate, never one weak probe)
- HBM: `bash $KW/scripts/gpu_lock.sh <GPU> python3 $SK/bin/hbm_stream.py --json roofline/hbm.json`.
  It takes the best of several streams. On MI355X, a non-temporal HIP stream reaches about 7.1 TB/s, while
  `torch.sum` reaches only 3.7. If any role's kernel streams faster, that number is the roof.
- Compute (only if some case may be compute-bound): an MFMA throughput probe at the contract's dtype.
  Take the fastest of a vendor GEMM (hipBLASLt / aiter) at a large shape and a hand probe.
- Small working sets (≤ a few MB) that stay in L2 / MALL: use the measured cache bandwidth for that
  size, and say so.

## 3. The fixed per-call cost
Time an **empty kernel** with the same launch path in the eval's exact regime: graph replay or eager, the
same timing events and flush, the same harness. That is `fixed_us`. If the eval's window holds more than
one launch, each extra node costs at least one dispatch; count only what the operator's contract
requires (normally one).

## 4. Cap and gap
Write `roofline/cases.json` (format in `$SK/bin/roofline.py`) with `base_us` and `cand_us` from the
latest official receipt, then run:
`python3 $SK/bin/roofline.py roofline/cases.json --md roofline/report.md --json roofline/cap.json`.
Report:
- the weighted **cap**, the shipped **score**, and **gap closed** = (score − 1) / (cap − 1);
- per case: floor, bound (memory / compute / fixed-cost dominated), the distance to it, and the cases that
  hold most of the remaining weighted gap (the tool sorts them);
- a sanity check: no floor may exceed its baseline time, and the shipped candidate may not beat a floor.
  If one does, the bytes, roofs or fixed cost are wrong; fix them before reporting.

## 5. When it is used
- Round 1: produce the cap with the baseline. The lead plans by the cases holding the largest gap.
- After each ship: rerun step 4 only (bytes and roofs do not change).
- Before the run may end (`geak-lead` end rule): a fresh rerun of steps 2–4, by you, not by the team.

Return 3 lines: weighted cap, shipped score and gap closed, and the top case by remaining weighted gap.
