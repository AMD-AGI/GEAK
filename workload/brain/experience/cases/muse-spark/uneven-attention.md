# Uneven attention

Origin: `muse-spark`. Case: `muse-spark/uneven-attention`. Kind: retrospective public/source case.

**Signal:** Requests in one batch have different actual KV lengths.

**Historical action:** Partition verify-attention work by actual length rather than equal request count.

**Observed result and limit:** Some larger-concurrency output results improve while C40 P90 declines; isolated attribution is absent.

**Next discriminator:** Does balancing shorten the slow tail enough to pay for its scheduling cost?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/muse-spark.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `muse-spark/uneven-attention`;
original chapter `reports/muse-spark/1106.md`.
