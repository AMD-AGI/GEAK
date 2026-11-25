# Whole-layer fusion and memory

Origin: `muse-spark`. Case: `muse-spark/large-fusion-memory`. Evidence includes public
source, serving logs and same-node GPU comparisons.

**When it transfers.** A short decode or verify can spend time crossing operator boundaries.
Check the batch widths actually served, the expert layout and memory left for KV.
TokenSpeed TP8 loaded 60/62 target and 5/5 draft layers at both C8 and C28. Its path
requires one or two equal-width requests. All 7,976 C8 decode samples met the request-count
condition, compared with 853/5,371 at C28. [C8](https://inferencex.semianalysis.com/api/v1/reproduce?id=448536) · [C28](https://inferencex.semianalysis.com/api/v1/reproduce?id=448532).

**What changed the plan at C8.** In SGLang 1027 TP8/EP8, the fused route duplicates
expert weights (about 47.8 GiB/rank). At fixed admission two, enabling the target alone
increased output throughput 15.27% and full-response P90 18.46%. Adding the fused draft
changed them by −1.32% and −2.43% in single 20-minute windows. The combined route still
beat ordinary execution by 13.75% and 15.59%, with full KV capacity falling from 17.47M
to 8.42M tokens. The ordinary arm had one stream error during a frontend restart.
This evidence puts the target first for a compatible port. [Native comparisons](../../raw/agentx-muse-gap-20261125.json).

At the same external setting, TokenSpeed TP8/EP1 reads each rank's shard of every
expert directly. Enabling its whole fused path raised output 13.66% and P90 11.59%,
and lowered P90 time to first token from 4.58 to 3.21 seconds. Even with fusion off,
it exceeded SGLang's ordinary route by 10.26% output and 37.65% P90. The remaining
contrast includes the two engines' topology, scheduling and implementation differences.
[Within-engine and cross-engine receipts](../../raw/agentx-muse-gap-20261125.json).

**Why check scheduling independently.** On the earlier six-kernel deployment, lowering
admission from 128 to two improved P90 37.08%, reduced output 13.27%, and raised P90
time to first token from 1.07 to 6.98 seconds. On the fused TokenSpeed route, halving
the prefill/decode interval reduced P90 time to first token by 26.50% and raised output
1.40%, while full-response P90 fell 1.25%. The original interval remained the reference
for simultaneous gains. [Raw observations](../../raw/agentx-muse-gap-20261125.json).

[Release history](../../routes/muse-spark.md) · [SGLang 1027 source analysis](../../raw/collections/agentx-mi355x-20261113/reports/muse-spark/1027.md) · [Earlier reversed-order GPU pair](gpu-muse-1027-mono-single-pair-20261113.md).
