# Host allocation recovery

Origin: `minimax-m31-flash`. Case: `minimax-m31-flash/host-allocation-recovery`. Kind: retrospective public/source case.

**Signal:** High-concurrency GPU-only performance degrades and the first HiCache write faults.

**Historical action:** Use legal host allocation and retest the cache path.

**Observed result and limit:** C64 output 536.21557 to 2622.68665 and P90 5.59285 to 59.79453 describe the recovery bundle.

**Next discriminator:** How can capacity and policy be separated without reinstating the faulty allocator?

Isolated GPU contribution and complete delivery cost are not established by this record.
The summary is not a verified predecision log. Check current versions and conditions.

[Route evidence](../../routes/minimax-m31-flash.md). [Raw collection](../../raw/README.md):
`agentx-mi355x-20261113:data/reusable-decisions.jsonl`, record `minimax-m31-flash/host-allocation-recovery`;
original chapter `reports/minimax-m31-flash/EARLY_AND_SGLANG.md`.
