---
key: paged decode attention on gfx950/CDNA4 — the Python/HIP wrapper host path around an already-tuned aiter-style device kernel
type: lever
confidence: ★★
effect: 2.08x isolated vs the frozen baseline from the host path alone; per-case it is the short-context cases that move (they are host/dispatch-bound and their device code stays byte-identical), while the long-context case gains least. Campaign cumulative reached 3.32x with device levers stacked on top.
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-11
name: host-wrapper-allocation-cache-scale-hoist-on-decode-attentio-attention-decode-gfx950-decode
description: Caching wrapper scratch allocations and hoisting scale prep out of the per-call path on paged decode attention: 2.08x alone, the campaign's largest single lever
keywords: ['decode', 'attention', 'launch-overhead', 'host-runtime', 'allocation-cache', 'dispatch-bound']
kernels: ['paged_attention_decode']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
---
# Host wrapper allocation cache + scale hoist on decode attention
- lever: On a decode attention op whose device kernel is already tuned, attack the wrapper first: cache per-shape scratch/output buffers across calls and hoist per-call scale and metadata preparation out of the hot path.
- apply: Reuse the allocation instead of re-allocating per call; compute scale tensors once per weight/layout rather than per invocation. A 'frozen' baseline label may cover only the device kernel — check whether the wrapper is actually in scope before assuming it is off-limits.
- verify: Isolated A/B on the smallest-context case first: it should move most while the device ISA is unchanged (diff the disassembly to prove nothing on the device side shifted).
- pitfall: The automated verdict reported no improvement on a round that had genuinely improved -> run-to-run drift on the smallest case dominated the scripted geomean -> re-verify by median of the largest case plus an ISA diff.
- caution: A compile cache keyed on a parameter hash can silently serve the previous binary — also verify the rebuild actually happened before trusting a flat A/B.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 2026-08-11
