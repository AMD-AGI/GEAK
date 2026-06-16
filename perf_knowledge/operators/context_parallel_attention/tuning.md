---
title: context_parallel_attention — tuning
kind: technique
operator: context_parallel_attention
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill]
updated: 2026-06-08
sources:
  - https://github.com/sgl-project/sglang/issues/22223
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
---

# context_parallel_attention — tuning

## The two levers: comm/compute overlap and load balance
CP perf = (local FA tile time) overlapped with (KV rotation / all-to-all). Tune both layers.

### 1. Overlap the collective with the local tile
- **Ring**: issue the next K/V block's P2P send/recv **before** computing the current local FA tile, so
  XGMI transfer hides under compute. CP only scales if `seq/cp` is large enough that the local tile takes
  longer than the KV block transfer. At small `seq/cp` you expose XGMI latency → poor scaling.
- aiter's **Iris** GPU-initiated comm (`aiter/ops/triton/comms/{all_gather,reduce_scatter,iris}.py`) lets
  the Triton kernel issue the collective inline, tightening overlap vs a separate RCCL call.

### 2. Causal load balancing (zigzag)
Naive causal CP gives rank 0 almost no work and the last rank almost all of it. Use a **zigzag** token
assignment (each rank gets one early + one late chunk) so all ranks do equal work. SGLang's prefill-CP
proposal is explicitly zigzag ring attention. Without it, CP scaling collapses on causal masks.

### 3. The local FA tile (same as [[attention_prefill_fmha]])
- Triton: `matrix_instr_nonkdim=16`, `num_warps=4`, `num_stages=1`, `waves_per_eu=2–3`,
  `schedule_hint=attention`, `knobs.amd.use_buffer_ops=ON`.
- The local tile is a normal FA prefill at `seq/cp` — tune it as such.

## AMD topology rules (do this first)
- **Stay within one 8-GPU XGMI island using TP before reaching for CP.** MI300X Infinity Fabric is
  ~448 GB/s/dir — fast but below NVLink; cross-island/cross-node CP comm is the bottleneck.
- Collective backend: `VLLM_ALL2ALL_BACKEND="allgather_reducescatter"` and
  `--disable-nccl-for-dp-synchronization` on ROCm (NCCL/RCCL all-to-all is not the fast path here).
- `NCCL_MIN_NCHANNELS=112` (multi-GPU), `HSA_NO_SCRATCH_RECLAIM=1`, `GPU_MAX_HW_QUEUES=2`.

## The LSE-merge
The cross-rank `merge_attn_states`/`merge_state` (combine partial `(O, m, l)` with their log-sum-exp) is
cheap but must be fp32 and associative. It's the same primitive as split-KV / MLA decode merge. Don't let
it serialize the ring — merge incrementally per received block.

## Autotune sketch
Sweep CP degree `cp ∈ {2,4,8}` at the target context; measure TTFT scaling and the comm/compute overlap
ratio (profile the XGMI transfer vs FA tile). Pick the largest `cp` that still keeps the local tile >
transfer time. Re-tune per context length.

## Verify scaling is real
TTFT should drop roughly with `cp` at long context. If TTFT flattens or worsens, the collective is exposed
(no overlap) or load is imbalanced (no zigzag) — profile XGMI utilization with `rocprofv3`.

## Sources
- SGLang zigzag ring attention + split-KV transfer: https://github.com/sgl-project/sglang/issues/22223
- ROCm 8-GPU XGMI island / TP-first, allgather_reducescatter: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- aiter Iris GPU-initiated comm primitives: `ROCm/aiter@a6bb49937:aiter/ops/triton/comms/` (on-box) ; https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html
