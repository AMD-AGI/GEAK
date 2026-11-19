# Offload sweep reversal

Origin: `deepseek-v4-pro`. Case: `deepseek-v4-pro/offload-sweep-reversal`. Kind: retrospective public/source case.

**Signal:** More host cache or overlap does not necessarily improve serving performance.

**Historical action:** Keep capacity, TBO, delayer and fusion alternatives, including reversals.

**Observed result and limit:** The final configuration selects only some switches and also corrects checkpoint-specific acceptance calibration.

**Next discriminator:** How do capacity, transfers, scheduling and extra memory interact on this workload?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/deepseek-v4-pro.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `deepseek-v4-pro/offload-sweep-reversal`;
original chapter `reports/deepseek-v4-pro/PR2823_DEVELOPMENT.md`.
