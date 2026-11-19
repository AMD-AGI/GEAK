# Role placement

Origin: `deepseek-v4-pro`. Case: `deepseek-v4-pro/role-placement`. Kind: retrospective public/source case.

**Signal:** Prefill and decode have different bottlenecks.

**Historical action:** Use MegaMoE only in suitable roles and budget its heap separately.

**Observed result and limit:** Prefill work does not establish decode gain; heap allocation is outside the memory-fraction budget.

**Next discriminator:** Does the current role mix change the best resource split and host tier?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/deepseek-v4-pro.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `deepseek-v4-pro/role-placement`;
original chapter `reports/deepseek-v4-pro/MORI_ROUTE.md`.
