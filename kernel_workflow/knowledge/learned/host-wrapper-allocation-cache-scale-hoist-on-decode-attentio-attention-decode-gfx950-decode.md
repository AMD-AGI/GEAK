---
key: paged decode attention on gfx950/CDNA4 — the Python/HIP wrapper host path around an already-tuned aiter-style device kernel
type: lever
confidence: ★★
effect: 2.08x isolated vs the frozen baseline from the host path alone; per-case it is the short-context / small-batch cases that move (they are host/dispatch-bound and their device code stays byte-identical), while the long-context case gains least. Campaign cumulative reached 3.32x with device levers stacked on top. Reproduced in a second, independent run on the same op class as a memoized foreign-call submit path in the Python wrapper: 1.52x geomean director-verified, per-case 1.65x / 1.51x / 1.40x from smallest to largest batch (same monotone decay, since the saving is a fixed per-call host cost), bit-stable and order-independent, with the device kernels, dispatch count and VGPR/LDS/scratch all unchanged. In that run the host lever was the ONLY direction that landed out of seven tried over three rounds.
confirms_cited: 2
confirms_blind: 0
losses: 0
attempts: 3
toolchain: unknown
last_seen: 2026-08-12
name: host-wrapper-allocation-cache-scale-hoist-on-decode-attentio-attention-decode-gfx950-decode
description: Cache wrapper allocations, hoist scale prep, memoize the submit path on paged decode attention: 2.08x and 1.52x in two runs, each run's largest lever
keywords: ['decode', 'attention', 'launch-overhead', 'host-runtime', 'allocation-cache', 'dispatch-bound', 'caching', 'ctypes', 'dispatch-floor', 'hip-graph']
kernels: ['paged_attention_decode']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
---
# Host wrapper allocation cache + scale hoist on decode attention
- lever: On a decode attention op whose device kernel is already tuned, attack the wrapper first: cache per-shape scratch/output buffers across calls, hoist per-call scale and metadata preparation out of the hot path, and memoize the compiled-callable lookup plus foreign-call argument setup so a repeat call is a table hit plus the submits.
- apply: Reuse the allocation instead of re-allocating per call; compute scale tensors once per weight/layout rather than per invocation; key the memo on the compile-time/template tuple only (pointers, grid and stream refill per call). A 'frozen' baseline label may cover only the device kernel — check whether the wrapper is actually in scope before assuming it is off-limits.
- verify: Isolated A/B on the smallest-context case first: it should move most while the device ISA is unchanged (diff the disassembly to prove nothing on the device side shifted), and the ratio should decay monotonically as the case grows — that decay is the signature of a fixed per-call saving.
- pitfall: The automated verdict reported no improvement on a round that had genuinely improved -> run-to-run drift on the smallest case dominated the scripted geomean -> re-verify by median of the largest case plus an ISA diff.
- pitfall: A second host round returned below 1.0 (wrapper-level graph capture) -> most of the residual host gap was per-call tensor creation inside a caller outside the editable surface, and the per-call signature hash plus the on-path gate cost about what replay saves -> measure what FRACTION of the residual host gap is editable before funding another host round.
- pitfall: A device edit that cut isolated body time by ~17% on the small-batch case scored only +0.6% on the wall, inside drift -> at this point the wall is floor-dominated (the per-case spread was only ~1.6x over a 32x problem-size range) -> grade device work on the scored wall, and only on the case where device time is the majority of it.
- pitfall: Rewriting the split-K combine dispatch to be LDS-free and barrier-free bought ~+2.4% on the largest case and nothing elsewhere -> that dispatch's cost is dominated by its existence (launch plus unavoidable partial-output traffic), not its body -> price DELETING it (partition size >= context length, so one partition) rather than leaning it out.
- pitfall: A partition-size template parameter was plumbed through in two consecutive rounds and never actually measured — the generated source still carried the default constant -> a direction that only ENABLES a measurement was graded as if it had made one -> grep the generated code for the new value, and grade on the measured number.
- caution: A compile cache keyed on a parameter hash can silently serve the previous binary — also verify the rebuild actually happened before trusting a flat A/B (move the build cache aside before each measurement of a codegen'd op).
- caution: After this lever lands the profile label can stay 'overhead' while the fixed cost has moved to the DEVICE side of the launch (a scalar-scale host-to-device copy emitted by a frozen caller, plus the split-K reduce dispatch) — also re-profile before assuming the remaining overhead is still host work.
- source: 16h per-kernel time-budget campaign, run chuschen16h, 2026-08-11; reproduced run kernel_20_geak_0811_2h, 2026-08-12
