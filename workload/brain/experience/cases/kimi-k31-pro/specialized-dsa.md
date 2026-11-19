# Specialized dsa

Origin: `kimi-k31-pro`. Case: `kimi-k31-pro/specialized-dsa`. Kind: retrospective public/source case.

**Signal:** Low-concurrency target DSA still hands off between separate steps and FFN.

**Historical action:** Fold split-K and top-k transformation into K2 while retaining the preceding indexer chain.

**Observed result and limit:** The C1 release improves output 4.52% and P90 9.05%, with other changes present.

**Next discriminator:** Can a same-image toggle isolate the DSA/fold contribution?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/kimi-k31-pro.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `kimi-k31-pro/specialized-dsa`;
original chapter `reports/kimi-k31-pro/VLLM_ROUTE.md`.
