---
name: re-test-an-inherited-cross-variant-dead-precision-label-agai-attention-gfx950-memory-bound
description: Re-test an inherited cross-variant DEAD precision label on this variant's own dispatch: narrowing KV storage took the run from 1.0615x to 1.2993x
keywords: [fp8, dtype-dialect, memory-bound, attention, decode, measurement-method]
kernels: [paged_attention_decode]
platforms: [gfx950]
kernel_class: attention
regime: memory-bound
key: narrowing streamed-once KV storage bytes (fp8 OCP e4m3 vs fnuz) while keeping dequant and accumulate wide, in a memory-bound gfx950 attention variant that already decouples storage from accumulate
lifecycle: active
type: lever
confidence: ★★
effect: A round had already STOPped at 1.0615x on an inherited 'narrow-KV storage is numerically DEAD' verdict measured on a sibling kernel and not on this one. Re-reading the dispatch showed this variant already ships a decoupled STORAGE/ACCUMULATE path (narrow KV bytes, wide in-register dequant and accumulate), and the sibling's 17dB failure had come from casting the softmax probabilities, not the KV storage. Building it moved the run 1.0615x -> 1.2993x cumulative in one direction, worst-element allclose 5e-2 PASS, best validated pass 1.3212x. Per-case it is a pure large-shape win: 1.6086x-2.0705x on all eight heavy cases (1.8015x / 2.0705x / 2.0439x / 2.0547x / 1.7700x / 2.0309x among them) and ~0 on the eight cases already sitting at the launch floor, which end at 0.80-0.9536x. Roofline-emp 0.440 -> 0.540, memory-bound both before and after.
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 4
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 16.43h / 6 passes, 2026-08-11
last_seen: 2026-08-11
---
# Re-test an inherited cross-variant DEAD precision label against this variant's own dispatch path before letting a STOP gate stand on it
- lever: When the campaign inherits a DEAD label on an operand-precision axis from a related kernel, treat it as untested until it is re-measured against THIS kernel's dispatch: variants of the same family often already separate storage dtype from accumulate dtype, so the failure that killed the sibling may live on a different tensor entirely (probabilities, scales, one of the two dots) than the one you were about to narrow. A STOP gate whose evidence is a measurement from another kernel is worth spending one round to disconfirm; the same reasoning applies to any 'known dead' entry with no local number behind it.
- apply: Narrow the streamed-once KV bytes while keeping the math wide — that is the form that pays, and it pays only where the HBM read dominates.
- verify: Re-read the variant's own dispatch for a decoupled storage/accumulate path before accepting the inherited verdict, then gate on worst-element allclose rather than a mean metric.
- pitfall: the narrowed operand produced silent garbage -> the cache's x-interleave factor scales as 1/sizeof(elem), so a plain cast of the narrowed operand mis-addresses, and the fp8 flavour (OCP e4m3 vs fnuz) shifts the decode by 2x -> re-derive the layout as part of the change, not as a detail.
- caution: Also verify the split by case size: this is a pure large-shape win and the cases already at the launch floor came back at or slightly below parity.
- source: chuschen 16h time-budget campaign run, 16.43h / 6 passes, 2026-08-11
