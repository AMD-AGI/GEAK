# Kernel fusion analysis (default prompt)

Used only when no `kernel_fusion.prompt` is given. Pass your own prompt for real runs.

Goal: for this serving workload, choose the fusion patterns that remove the current dominant
bottleneck with the SMALLEST fusion range, and rank them by expected end-to-end gain.

1. **Model and deployment.** From `config.json`, the server args/env and the framework source, map
   each module of one decoder layer (and the per-step glue, and any draft/MTP path) to the kernels
   it launches in the trace. Note TP/EP communication points and which steps run under CUDA/HIP graph.
2. **Workload.** Per step kind: steps, device time, kernels/step, idle gap, short-kernel share, and
   the weight of each step kind in the measured workload. Separate per-layer from per-step work.
3. **Regions.** Group kernels into regions (norm, attention, KV cache, router, MoE, quant,
   communication, residual, draft). Adjacent in time is not fusable by itself: confirm tensor
   dependencies, multiple consumers, side effects (KV-cache writes), collectives, shapes and
   precision, and size-guarded dispatch branches.
4. **Candidates.** For each hot region consider L1 (element-wise chain), L2 (main operator +
   prologue/epilogue), L3 (whole stage) and L4 (whole layer). For each: kernels, occurrences per
   layer/step, launches before → after, Python seam, quantified gain per step and weighted e2e %
   (upper bound and expected, with the discount explained), risks (VGPR/LDS/occupancy/tile
   compromise), difficulty 1–5, bit-exactness, and conflicts with other candidates.
5. **Ranking.** Top-K by expected weighted gain, a recommended non-conflicting set with a total that
   does not double-count, fallbacks, regions not to fuse, and what information is still missing.

Write the report as `report.md` in the output directory.
