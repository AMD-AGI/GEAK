# Padding

Origin: `deepseek-v4-pro`. Case: `deepseek-v4-pro/padding`. Kind: retrospective public/source case.

**Signal:** A native TP shard of width 384 is padded to 512.

**Historical action:** Preserve the backend legal 128-aligned intermediate width.

**Observed result and limit:** This removes FFN padding, not experts; the isolated serving gain is unknown.

**Next discriminator:** How do tile efficiency, weight traffic and memory headroom change after removing padding?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/deepseek-v4-pro.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `deepseek-v4-pro/padding`;
original chapter `reports/deepseek-v4-pro/VLLM_ROUTE.md`.
