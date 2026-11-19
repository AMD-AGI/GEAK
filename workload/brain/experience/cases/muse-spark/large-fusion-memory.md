# Large fusion memory

Origin: `muse-spark`. Case: `muse-spark/large-fusion-memory`. Kind: retrospective public/source case.

**Signal:** Small batches expose cross-operator costs, with a larger fused implementation available.

**Historical action:** Evaluate mono coverage together with admission, graph widths and KV budget.

**Observed result and limit:** The expert copies consume about 47.8 GiB per rank; same-version policy and kernel gains are not additive independent effects.

**Next discriminator:** At fixed scheduling, how much does fusion save, and when does its capacity cost reverse the benefit?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/muse-spark.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `muse-spark/large-fusion-memory`;
original chapter `reports/muse-spark/1027.md`.
