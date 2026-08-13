---
key: split-KV paged decode attention on gfx950 that ships as two dispatches (partial attention + reduce), HIP/C++ template source
type: lever
confidence: ★★
effect: +9.5% over the host-optimized incumbent for the fusion itself (all three cases single-dispatch, profiler-confirmed one dispatch per call with zero reduce rows), then +3.0% more from epilogue protocol tuning, largest on the 2-sequence case; part of an accepted 3.98x geomean
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 1
toolchain: unknown
last_seen: 2026-08-12
name: one-dispatch-for-split-kv-decode-and-the-protocol-that-pays--attention-decode-gfx950-decode
description: Fuse the split-KV reduce into the attention kernel behind a padded arrival counter with arch-cheap release bits: +9.5%, then +3.0% of epilogue protocol tuning.
keywords: ['decode', 'paged-attention', 'split-kv', 'dispatch-collapse', 'cross-workgroup', 'arrival-counter', 'coherence']
kernels: ['paged_attention_ll4mi_QKV_mfma16_kernel', 'paged_attention_ll4mi_reduce_kernel']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
---
# One dispatch for split-KV decode, and the protocol that pays for it
- lever: - lever: collapse the two-kernel split-KV decode into one dispatch — decouple tile size from partition size so one workgroup owns several partitions, then let the last arriver run the combine behind a global arrival counter.
- apply: - apply: per-store coherence bits (sc0/sc1) plus a bare vmcnt(0) as the cross-workgroup release; relaxed agent-scope atomic loads on the counter; an agent ACQUIRE fence ahead of plain wide combine loads, with the wide form gated on partitions%4==0 since it is pure cost at 2 partitions.
- verify: - verify: profiler dispatch count == call count with no reduce rows, and hand-check parity on EVERY case — the shipped correctness gate here covered only the largest case, so a corrupted small-batch path scored ~9.9 dB while the gate printed pass.
- pitfall: - pitfall: a device-scope release fence for the same handshake lowers to an L2 writeback and made the largest-batch case several times slower than the unfused incumbent -> per-store coherence bits + wave-scoped waits instead.
- pitfall: arrival counters packed one int per (seq, kv-head) false-share a line -> pad them to a full-line stride.
- pitfall: two handshake barriers where only wave 0 issues the producer stores -> a wave-0-only release drain removes one of them as pure latency.
- caution: - caution: also verify how much of the combine ceiling is harvestable before budgeting rounds at it — a probe that deleted the whole protocol showed ~20% on the small-batch case, but decomposition priced the reachable part at ~3% because the incumbent already sat on the publish/signal/consume round-trip floor.
- source: - source: kernel_workflow 16h campaign, run kernel_20_geak_0808_16h, 2026-08-12; director-validated geomean 3.98x, correctness pass
