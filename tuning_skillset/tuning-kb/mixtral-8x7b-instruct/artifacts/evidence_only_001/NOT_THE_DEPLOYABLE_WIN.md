# These two files are a record, not a deployment

`mixtral8x7b_tuned_fmoe.csv` and `001_aiter_tuned_fmoe_mixtral.patch` were the deployable win of
**round 1**. They are no longer what this entry ships. `../deploy/` is.

## Why they are kept

They still carry evidence, and it is evidence nothing else in this directory carries:

- Round 1 measured them at **+18.59 tok/s = +0.269%** on `crsuse2-m2m-110` (3 instances per arm,
  complete separation) and **+30.08 tok/s = +0.436%** on `crsuse2-m2m-261` (2 vs 6, complete
  separation), 14 independently booted instances across two hosts, combined Fisher p ≈ 0.013.
  That result was never refuted.
- They passed a gsm8k gate: strict-match 0.6664 ± 0.0130 against a same-node baseline of
  0.6505 ± 0.0131.
- They are the worked example behind three of the four withdrawn claims in the parent `README.md`,
  and behind the finding that the CK 2-stage heuristic under-sizes the K-tile on this geometry.

## Why they are not deployed

Round 2, on a third node, measured `002` against pristine aiter and then measured the two together,
interleaved boot by boot:

| arm | instances | mean tok/s | vs that node's base (6918.43) |
| --- | --: | --: | --: |
| 002 only | 4 | 6950.41 (restart sd 4.57) | +31.98 |
| 001 + 002 | 2 | 6939.56 (restart sd 9.50) | +21.13 |

Adding 001 on top of 002 does not add throughput. The point estimate is −10.9 tok/s, which is
inside the stack arm's own restart spread at n=2, so the defensible statement is **no measurable
benefit** — not that 001 is harmful. Patch 001 did engage on that node (8 named `fused_moe` kernel
selections per rank, 0 defaults, at every token bucket including decode token=64), so this was a
real comparison. Source: `FINDINGS.md` §3.5, `patches/002_aiter_tuned_gemm_mixtral/RESULT.md`
"Interaction with patch 001".

**001 alone was never measured on round 2's node.** Its +0.269% stands on round 1's two hosts and
has not been re-tested against pristine aiter on a third.

One more thing round 2 saw and could not explain: the stack arm's *within-instance* sd was 11.36
and 12.79 tok/s, against 2.56–7.67 for every other instance in the bundle. Adding the fmoe table
made run-to-run behaviour noticeably noisier.
