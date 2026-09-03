---
key: adding KV-split parallelism to bf16 decode attention judged against a golden output captured at a fixed reduction order, on gfx950
type: anti-pattern
confidence: ★★
effect: closed axis across three split factors at batch B=2: gate max_rel_err 1.24 / 1.01 / 1.43 against a 1e-2 threshold at 2 / 4 / 8 partitions, while cosine stayed above 0.999997; the same path at 1 partition is bit-exact
confirms_cited: 1
confirms_blind: 0
losses: 0
attempts: 2
toolchain: unknown
last_seen: 2026-08-17
name: a-single-partition-control-separates-a-rejected-kv-split-fro-attention-decode-gfx950-decode
description: KV-split parallelism on decode attention fails the elementwise oracle for reduction reorder, not kernel error; a single-partition control shows which it is
keywords: ['split-kv', 'flash-decoding', 'oracle-parity', 'reduction-order', 'decode', 'anti-pattern', 'paged-attention']
kernels: ['kernel_unified_attention_2d']
platforms: ['gfx950']
kernel_class: attention_decode
regime: decode
layer: learned
lifecycle: active
cost: L3
verified_on: 2026-08-12
levers: ['algo.split-kv']
origin_kernels: ['kernel_unified_attention_2d']
---
# A single-partition control separates a rejected KV split from a wrong KV split
- lever: When the structural lever left is more parallelism over KV, run the split-and-combine path at one partition first as a control; bit-exact there says the kernels are correct and the parity gate is rejecting the reduction reorder, which is a cheap way to price the rest of the lane.
- apply: The control reuses the identical split and combine code with the partition count set to one, compared elementwise-equal against the golden output; then step the partition count and watch which metric moves.
- verify: Check which metric the gate actually binds on — here cosine similarity stayed at parity across every split factor while the elementwise relative-error gate failed, so the metric identity, not the numerics, decided the verdict.
- pitfall: At the worst element the split answer was closer to an fp64 reference than the golden was, and lost by one bf16 ulp → the gate scores agreement with a reference reduction order rather than accuracy → a more accurate kernel can still be refused.
Reclaiming the apparently dead row-half of the block to fund the split → MFMA 16x16 padding makes those rows free → the narrower block measured 21% slower.
- caution: Also verify whether your harness threshold is elementwise or aggregate, and whether the golden can be regenerated at the candidate's reduction order, before treating the lane as closed for your setup.
- source: run kernel_20_geak_0808_16h, rounds 1 and 5 parity evidence, 2026-08-12
