# Cpu prefix reuse

Origin: `minimax-m31-flash`. Case: `minimax-m31-flash/cpu-prefix-reuse`. Kind: retrospective public/source case.

**Signal:** Long multi-turn prompts repeat tokenization.

**Historical action:** Cache encoded prefixes only at safe tokenizer/special-token boundaries.

**Observed result and limit:** The author encode P50 changes from 152 to 9 ms; an isolated serving gain is not established.

**Next discriminator:** How much of the current critical path is encoding and queueing at each prompt regime?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/minimax-m31-flash.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `minimax-m31-flash/cpu-prefix-reuse`;
original chapter `reports/minimax-m31-flash/VLLM_75BC.md`.
