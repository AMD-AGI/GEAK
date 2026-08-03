---
key: aten library elementwise/copy/cat/index ops · any gfx · serving Top-N tail
type: routing
confidence: ★★★
effect: saves whole extraction attempts — these profile entries have NO editable body/seam; near-noise ceiling regardless
last_seen: 2026-07-30
confirms: 6
---
# Hot small copy/cast/cat/index ops in the Top-N are ATen library kernels — do NOT route to kernel extraction
- lever: profile names like `elementwise_kernel_manual_unroll` / `direct_copy_kernel_cuda` /
  `vectorized_elementwise_kernel (add)` / `CatArrayBatchedCopy_contig` / `index_elementwise_kernel
  (index_put)` are PyTorch/ATen native ops compiled into libtorch(_hip).so — grep of the sglang AND
  aiter source trees returns ZERO definitions. There is (1) no editable Python/Triton body to copy into
  kernel_src/, and (2) no single rebindable `module:attr` seam (they fire IMPLICITLY from
  `.contiguous()`/reshape/`.to(long)`/`torch.cat`/`x[mask]` scattered across hundreds of call sites).
- apply: mark editable=false, target_callable="", write a drop-record meta.json; do NOT synthesize a
  proxy oracle (that is a fake extraction). Route to the config/cuda-graph track (collapse launch +
  host roundtrips) or a producer-side source-fusion refactor (native layouts / preallocated buffers),
  which is an Integrator/source task, not a frozen-oracle kernel_workflow task.
- verify: `grep -rn <symbol> <sglang>/python/sglang <aiter>/aiter` == 0 hits ⇒ library op.
- caution: also check Amdahl first — every such op this run was 0.17-0.76% GPU time (amdahl_priority
  ≤0.27), so even a large isolated win cannot move e2e; usually just DROP.
- source: /wekafs/test_results/Qwen3_14B_20260730 (k5,k8,k9,k10,k11,k12 all editable=false, dropped)
