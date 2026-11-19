# Fusion eligibility

Origin: `deepseek-v4-pro`. Case: `deepseek-v4-pro/fusion-eligibility`. Kind: retrospective public/source case.

**Signal:** A shared-expert fusion switch exists but can miss the active path.

**Historical action:** Check quantization, topology, library capability and actual registration before estimating coverage.

**Observed result and limit:** Some TP8 paths register heterogeneous MoE while DPA rejects it; residual warnings do not prove universal failure.

**Next discriminator:** Which steady-state calls actually use the candidate, and what share do they represent?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/deepseek-v4-pro.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `deepseek-v4-pro/fusion-eligibility`;
original chapter `reports/deepseek-v4-pro/VLLM_ROUTE.md`.
