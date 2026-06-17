---
id: mla_tilelang_to_triton
title: "MLA decode: port the TileLang core to Triton (gfx942)"
kind: expert_skill
authors: [zihao]
scope: kernel
match:
  operator: mla_attention
  arch_class: [deepseek_mla]
  gens: [gfx942]
  dtypes: [bf16, fp8_e4m3_fnuz]
  regimes: [decode]
  from_backend: tilelang
  to_backend: triton
expects:
  isolated_speedup_min: 1.15
  parity: required
validation:
  status: draft
  last_verified: ""
  gpu: ""
  model: ""
  measured: {isolated: "", e2e_pct: "", parity: ""}
  artifact: ""
role: advisory_prior
supersedes: []
---

## When to use
The MLA (multi-head latent attention) **decode** path is the bottleneck on a DeepSeek-class model on
MI300X (gfx942) and the live implementation is a TileLang kernel. Port it to Triton to (a) get an
editable, CUDA-graph-capturable core the workflow can further tune, and (b) close the gap to the
hand-tuned aiter/asm path while staying author-friendly.

## Mechanism
MLA decode is memory-bound on the KV-latent read with a low-rank up-projection fused into the attention.
TileLang expresses the tiling explicitly but is harder to autotune inside the kernel layer and can carry
a host-sync / graph-capture hazard. A Triton port lets you: keep the latent in registers/LDS across the
absorbed up-proj, pick MFMA-friendly `BLOCK_M/BLOCK_N` for the small decode M, and split the KV loop
(`split_k` / flash-style online softmax) — all as Triton autotune knobs. The win is removing the
per-step host sync and matching the MFMA tile to the latent head_dim, not raw FLOPs.

## Procedure
1. Extract the MLA decode op into the immutable unittest task dir (shapes spanning decode M∈{1,64} and
   the latent dims; bf16 and fp8_e4m3_fnuz operands; reference I/O oracle from the TileLang core).
2. Author a Triton kernel: flash-style online softmax over the KV-latent loop; absorb the up-projection
   (W_UK/W_UV) into the QK / OV matmuls so the latent never expands to full head_dim in HBM.
3. **Graph-capture safety is mandatory** (this kernel overlays the live decode path): steady-state hot
   path must be host-sync-free — no `.item()/.cpu()/.synchronize()`, no Python branch on a GPU scalar.
   Cache any weight prep (preshuffle/requant) once by `weight.data_ptr()`, never per call.
4. Tune: `BLOCK_M` small (decode), `BLOCK_N`/`split_k` over the KV loop, `num_warps`, `kpack`, fp32
   accumulate. Keep the fp8 path fused (fold block-scale into the operand scale → one fp8 MFMA).
5. Gate isolated vs the oracle; then let the e2e layer overlay it (bare core, capture-safe).

## Knobs & pitfalls
- Decode M is tiny — wide tiles waste lanes; start `BLOCK_M=16/32`.
- `split_k>1` reductions with small `BLOCK_M` can go numerically wrong on this build — verify parity per
  config, not just speed.
- Do NOT re-materialize bf16 weights for the absorbed proj across all layers (memory blow-up → forces
  mem-fraction down → KV-cache starves → net e2e regression even if the kernel is faster).

## Do-no-harm notes
- This is a decode-regime skill; do not apply the decode tiling to prefill MLA (different shape regime).
- If parity fails at any candidate config, drop that config — never trade correctness for speed.

## Sources
- Recipe distilled from the MLA / DeepSeek case studies in `perf_knowledge/case_studies/by_model/
  deepseek_mla_mi300x.md` and `deepseek_v3v4_attention.md`.
- STATUS: draft — needs on-box validation. Run `_contribute/validate_skill.py mla_tilelang_to_triton
  --emit-plan` then `--record` with the kernel_workflow isolated A/B eval dir.
