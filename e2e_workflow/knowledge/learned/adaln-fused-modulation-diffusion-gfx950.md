---
key: reduction_norm + elementwise_overhead (AdaLN modulation) · gfx950 · sglang-diffusion DiT
type: lever
confidence: ★★
effect: isolated **3.52–3.58× geomean** (fused HIP replacement of an unfused 4-kernel chain) on
  5.07% (trace) – 8.3% (cluster) of denoise GPU time → Amdahl ceiling +3.7% to +6.3% e2e.
  **Isolated only — the e2e A/B never ran (box VRAM blocked); treat as a pending win, not a banked one.**
confirms: 1
last_seen: 2026-07-31
---
# FLUX.1-dev AdaLN `norm→scale→shift` is UNFUSED on ROCm — fuse it

Math contract: `out = layer_norm(x, eps=1e-6, elementwise_affine=False) * (1 + scale) + shift`,
`x=[B,M,3072] bf16`, `scale/shift=[B,1,3072]` broadcast over M, fp32 accumulation. Live M buckets
512 (dual-stream) and 1024 (single-stream); **114 calls per DiT forward × 28 steps**.

- **routing fact:** sglang-diffusion HAS a `fused_norm_scale_shift` fast path and `D=3072` qualifies,
  but it is **NOT engaged** on this build — the live path launches FOUR HIP kernels
  (`vectorized_layer_norm_kernel` + three broadcast elementwise ops). Check engagement before
  assuming a framework fast path is live; a shipped fused kernel that never binds is free headroom.
- lever: author a single fused HIP kernel (LN reduce in fp32, then apply `(1+scale)`/`shift` in the
  same pass). Measured 3.52× geomean vs the stock chain, numerics within 1 bf16 ULP (max rel 8.7e-3).
- apply: PYTHONPATH-only overlay binding THREE seams — `diffusers.models.normalization` (AdaLayerNorm*)
  and `sglang.multimodal_gen.runtime.models.dits.flux` — which together cover 114/114 calls/forward.
  Never patch site-packages; a finder-based overlay keeps the run reversible and provenance clean.
- verify: count engaged calls with a one-shot banner (expect 114/DiT forward), and confirm the HIP
  extension actually loaded (`hip_kernel_true`) — a silent Python fallback looks identical in output.
- caution: **also verify the profiler's attribution before believing a low %GPU.** `attribute_weights
  --name-match` matched only 1 of the 4 kernels and reported 1.66%, then emitted its generic "seam is
  probably NOT the live kernel" warning — a FALSE POSITIVE for a multi-kernel op. Sum the constituent
  kernels by hand (and restrict elementwise entries to the right operand shapes) before dropping the op.
- source: exp/e2e_FLUX-1-dev_20260729_132053_288743_23791 · kernels/k1_adaln_norm_scale_shift_task,
  overlay/cand_k1_adaln (2026-07-31)
