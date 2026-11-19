# Admission before kernel

Origin: `deepseek-v41-flash`. Case: `deepseek-v41-flash/admission-before-kernel`. Kind: retrospective public/source case.

**Signal:** One external session produces multiple internal requests while admission is too small.

**Historical action:** Remove the observed queue restriction and separately evaluate kernel changes.

**Observed result and limit:** The public C1 release gain greatly exceeds the author local MoE step improvement; it cannot all be assigned to kernels.

**Next discriminator:** What are the separate effects of admission and MoE backend at fixed workload conditions?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/deepseek-v41-flash.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `deepseek-v41-flash/admission-before-kernel`;
original chapter `reports/deepseek-v41-flash/VLLM_ROUTE.md`.
