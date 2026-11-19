# Sharded fc

Origin: `kimi-k31-pro`. Case: `kimi-k31-pro/sharded-fc`. Kind: retrospective public/source case.

**Signal:** Draft FC repeats the same long-prefill work across TP ranks.

**Historical action:** Shard rows for the large FC/norm and gather the required output, keeping short-input fallback.

**Observed result and limit:** The author cites about 7.6 ms replicated FC versus 0.6 ms gather at 16K; complete serving attribution is absent.

**Next discriminator:** What is the net saving under the actual prefill fraction and exchange cost?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/kimi-k31-pro.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `kimi-k31-pro/sharded-fc`;
original chapter `reports/kimi-k31-pro/SGLANG_ROUTE.md`.
