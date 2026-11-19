# Prefix chain lru

Origin: `minimax-m31-flash`. Case: `minimax-m31-flash/prefix-chain-lru`. Kind: retrospective public/source case.

**Signal:** Later CPU KV blocks survive while the usable chain head is evicted.

**Historical action:** Protect continuous prefix availability in the eviction order.

**Observed result and limit:** A new CPU tier is also added at C32; the full effect is not an isolated LRU gain.

**Next discriminator:** Does a short test reach representative capacity pressure and reuse?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/minimax-m31-flash.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `minimax-m31-flash/prefix-chain-lru`;
original chapter `reports/minimax-m31-flash/VLLM_75BC.md`.
