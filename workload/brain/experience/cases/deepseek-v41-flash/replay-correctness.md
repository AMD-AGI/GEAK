# Replay correctness

Origin: `deepseek-v41-flash`. Case: `deepseek-v41-flash/replay-correctness`. Kind: retrospective public/source case.

**Signal:** A new bounded replay path does not cover a sparse SWA boundary on ROCm.

**Historical action:** Disable the faulty path, retain prefix caching, then re-enable after repair.

**Observed result and limit:** Failure, withdrawal and recovery are all part of the route.

**Next discriminator:** How much prefill does the repaired replay eliminate on representative repeated prefixes?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/deepseek-v41-flash.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `deepseek-v41-flash/replay-correctness`;
original chapter `reports/deepseek-v41-flash/VLLM_ROUTE.md`.
