# Producer consumer layout

Origin: `deepseek-v41-flash`. Case: `deepseek-v41-flash/producer-consumer-layout`. Kind: retrospective public/source case.

**Signal:** Activation quantization, scales and GEMM create repeated representation work.

**Historical action:** Match producer output to the consumer layout and real scale grouping.

**Observed result and limit:** TP/K alignment and true block32 scales constrain compression; bundled gains remain unseparated.

**Next discriminator:** Do fewer launches and scale reads improve the actual served shape mix?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/deepseek-v41-flash.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `deepseek-v41-flash/producer-consumer-layout`;
original chapter `reports/deepseek-v41-flash/VLLM_ROUTE.md`.
