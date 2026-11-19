# Cache regime

Origin: `muse-spark`. Case: `muse-spark/cache-regime`. Kind: retrospective public/source case.

**Signal:** Long contexts at higher concurrency exceed effective GPU KV capacity.

**Historical action:** Add a host KV tier where capacity pressure is present, retaining GPU-only controls.

**Observed result and limit:** Published TP4/C16 output improved 16.82% and P90 30.63%; TP8 at the same concurrency did not show that gain.

**Next discriminator:** Does reduced recomputation outweigh host-transfer cost under the current reuse pattern?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/muse-spark.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `muse-spark/cache-regime`;
original chapter `reports/muse-spark/ROUTE.md`.
