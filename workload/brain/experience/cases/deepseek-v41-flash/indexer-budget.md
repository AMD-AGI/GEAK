# Indexer budget

Origin: `deepseek-v41-flash`. Case: `deepseek-v41-flash/indexer-budget`. Kind: retrospective public/source case.

**Signal:** Dense logits and scratch grow with worst-case geometry.

**Historical action:** Combine paged low-bit scoring with request-aware workspace sizing.

**Observed result and limit:** The author short-window improvement lacks complete raw support; compute, traffic and capacity contributions are not isolated.

**Next discriminator:** Which benefits come from computation, smaller logits or usable KV capacity?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/deepseek-v41-flash.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `deepseek-v41-flash/indexer-budget`;
original chapter `reports/deepseek-v41-flash/VLLM_ROUTE.md`.
