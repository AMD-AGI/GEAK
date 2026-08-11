---
name: a-high-percent-of-bandwidth-reading-can-be-a-compute-tail-ar-attention-gfx950-decode
description: Read the VALU-to-VMEM ratio before funding a streaming rewrite off a high percent-of-bandwidth reading: the rewrite returned 1.0x, as did two adjacent ones
keywords: [roofline, counters, measurement-method, decode, attention, occupancy, vgpr, dequant, operand-reuse]
kernels: [paged_attention_decode]
platforms: [gfx950]
kernel_class: attention
regime: decode
key: pricing a load-streaming / double-buffering rewrite on a dequantizing attention decode kernel on gfx950 whose roofline plot reads as bandwidth-bound
lifecycle: active
type: anti-pattern
confidence: ★★
effect: a deep_explore round budgeted a ground-up double-buffered streaming rewrite off a '~57% of the bandwidth roof' reading on the largest case, expected 1.2x, returned 1.0x and no patch (cumulative unchanged at 3.181x); counters overturned the premise - VALU:VMEM was 34:1 and LDS-wait exceeded busy cycles, so HBM was idle during the dequant, the LDS transpose (~44% of VALU) and the softmax tail, and both streamed operands were already 32 VGPRs each inside an 86-VGPR/occupancy-5 footprint, leaving no exposed load latency to hide; even a dream 90%-of-roof scenario on that case bounded the geomean at ~1.09x, and two adjacent bandwidth-headroom directions (L2 residency, narrowing the partial-output dtype) also returned 1.0x
confirms_cited: 0
confirms_blind: 1
losses: 0
attempts: 3
toolchain: rocm 7.x / triton 3.6.0 / torch 2.11.0
source: chuschen 16h time-budget campaign run, 15.73h / 17 passes, 2026-08-11
last_seen: 2026-08-11
---
# A high percent-of-bandwidth reading can be a compute-tail artifact
- lever: Before spending a deep or expensive direction on load streaming, prefetch or double-buffering, classify the kernel with a VALU:VMEM counter ratio and check whether the loads are actually EXPOSED on the critical path rather than already front-loaded into registers - a percent-of-roofline figure is an average over the whole dispatch and reads high when a compute tail leaves the memory system idle at the end.
- apply: Pull the counters on the dominant case and compare the register footprint of the streamed operands against the occupancy budget; then do the cheap arithmetic companion check - compute what the whole-benchmark geomean would be if this case hit ~90% of the roof, and if that number is small the direction is priced at zero before any code is written.
- verify: Confirm the ratio and the LDS-wait share are read on the same case the roofline number came from.
- caution: Also verify the reading is not dominated by a serial epilogue before concluding the whole dispatch is compute-bound.
- source: chuschen 16h time-budget campaign run, 15.73h / 17 passes, 2026-08-11
