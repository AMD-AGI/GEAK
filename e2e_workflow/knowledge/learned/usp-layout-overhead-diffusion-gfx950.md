---
key: elementwise_overhead (Ulysses permute/pack + QKV cat) · gfx950 · sglang-diffusion DiT, SP=8
type: lever
confidence: ★★
effect: isolated **1.615×** over six pure layout ops covering 12.2% of denoise GPU time (17.5% if the
  collective's D2D staging + the single-stream MLP cat are folded in) → Amdahl ceiling only **+4.9% e2e**.
  **Isolated only — the e2e A/B never ran (box VRAM blocked); pending, not banked.**
confirms: 1
last_seen: 2026-07-31
---
# Ulysses SP leaves ~12–17% of a DiT denoise in pure layout/copy

Under SP=8 on FLUX.1-dev (B=1, S_local=512, H_global=24 → H_local=3, D=128), the *local* halves of the
all-to-all wrapper plus the replicated-prefix joint-sequence assembly are six **pure permutation/copy**
ops — `in_pack`, `in_unpack`, `out_unpack`, `prefix_head_slice_cat`, `out_merge`, `dual_qkv_join_cat` —
launched as `elementwise_kernel_manual_unroll(direct_copy)` 6.48% + `CatArrayBatchedCopy_contig` 5.72%.
No arithmetic, no reduction: every output is a bit-exact rearrangement.

- lever: fuse the permute/pack/cat chain into single-pass copies (one kernel per logical rearrangement
  instead of a torch `permute→contiguous→cat` sequence). Bit-exactness makes image parity free.
- apply: PYTHONPATH overlay over three modules — `multimodal_gen/runtime/layers/usp.py`,
  `layers/attention/layer.py`, `models/dits/flux.py`. Verify the seam binds in all three and that each
  of the six ops is bit-identical to the original before spending an A/B.
- verify: SP layout ops only appear when world_size>1, so a single-GPU bake-off must synthesize the
  live geometry (`h_start`, `num_rep`) rather than read it from a rank-0 trace.
- caution: **also verify what you are allowed to count.** Excluding the collective's own `Memcpy DtoD`
  staging (3.29%, 1401 calls) from the timed region is deliberate — otherwise a "candidate" can win by
  deleting communication. Scoring the honest 12.2% instead of the 17.5% cluster drops the ceiling from
  ~7% to 4.9%, which on a 1% noise band is plausible but not decisive: this op needs a real,
  many-repeat A/B, not an Amdahl argument.
- source: exp/e2e_FLUX-1-dev_20260729_132053_288743_23791 · kernels/k0_usp_permute_pack_cat_task,
  overlay/cand_k0_usp (2026-07-30)
