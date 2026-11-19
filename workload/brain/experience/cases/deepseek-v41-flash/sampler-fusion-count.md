# Sampler fusion count

Origin: `deepseek-v41-flash`. Case: `deepseek-v41-flash/sampler-fusion-count`. Kind: retrospective public/source case.

**Signal:** Draft Markov sampling launches several kernels at each position.

**Historical action:** Fuse the chain for a position while retaining its true invocation count.

**Observed result and limit:** This is not one launch for the entire Markov chain and has a limited small-request domain.

**Next discriminator:** What fallback should serve inputs outside the measured domain?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/deepseek-v41-flash.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `deepseek-v41-flash/sampler-fusion-count`;
original chapter `reports/deepseek-v41-flash/VLLM_ROUTE.md`.
