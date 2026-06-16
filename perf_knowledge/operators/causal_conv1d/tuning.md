---
title: causal_conv1d — tuning
kind: operator_overview
operator: causal_conv1d
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [prefill, decode]
updated: 2026-06-05
sources:
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:aiter/ops/triton/causal_conv1d.py
  - ROCm/aiter@a6bb499375849eec45d68c5ccaebc8865fd422c0:csrc/kernels/causal_conv1d_update.cu
  - https://github.com/vllm-project/vllm/pull/17146
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# causal_conv1d — tuning

## The decision first
This op is **memory-/launch-bound**, not compute-bound — there is no MFMA, no K-loop pipeline, no LDS
GEMM staging. So the classic GEMM levers (`matrix_instr_nonkdim`, `num_stages`, split-K) **do not
apply**. The real levers are: (1) **channel-last layout** so loads coalesce, (2) keep the ≤4 taps and
the sliding window **in registers** (not LDS), (3) **grid sizing** so all 304 CUs are fed, and (4) at
serving time, **split prefill vs decode requests** into the right kernel (the biggest measured win — see
below). Do not hand-tune before confirming the model even has GDN/Mamba layers ([overview.md](overview.md)).

## Measured baseline (on-box, decode `causal_conv1d_update`, Triton)
| batch | dim | width | dtype | median | min |
|---|---|---|---|---|---|
| 128 | 4096 | 4 | bf16 | 69.1 µs | 65.2 µs |
| 64 | 5120 | 4 | bf16 | 68.6 µs | 66.7 µs |
| 256 | 8192 | 4 | bf16 | 86.2 µs | 82.3 µs |

@ MI300X gfx942, ROCm 7.2.0, aiter@a6bb49937, 20 warm reps, 2026-06-05. Note latency is **flat** across
a 4× `dim` range at fixed batch — confirming the cost is launch + state I/O, not arithmetic. The lever
that moves this number is **batching more tokens per launch**, not making the math faster.

```bash
# microbench (decode update):
python -c "
import torch,time
from aiter.ops.triton.causal_conv1d import causal_conv1d_update
b,d,w=128,4096,4
x=torch.randn(b,d,device='cuda',dtype=torch.bfloat16)
cs=torch.randn(b,d,w-1,device='cuda',dtype=torch.bfloat16)
W=torch.randn(d,w,device='cuda',dtype=torch.bfloat16); bias=torch.randn(d,device='cuda',dtype=torch.bfloat16)
for _ in range(5): causal_conv1d_update(x.clone(),cs.clone(),W,bias,activation='silu')
torch.cuda.synchronize(); ts=[]
for _ in range(20):
  xc,c=x.clone(),cs.clone(); torch.cuda.synchronize(); t=time.perf_counter()
  causal_conv1d_update(xc,c,W,bias,activation='silu'); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1e6)
ts.sort(); print('median',ts[len(ts)//2],'us')"
```

## Triton kernel config (aiter `causal_conv1d.py`)
The aiter Triton kernels are **not autotuned** — they ship fixed, hand-picked launch params:

| param | prefill (`_causal_conv1d_fwd_kernel`) | decode (`_causal_conv1d_update_kernel`) | why |
|---|---|---|---|
| `BLOCK_M` (tokens/block) | **8** | n/a (one block per batch) | small M — prefill seqs are short relative to channel width; large M wastes the register window |
| `BLOCK_N` (channels/block) | **256** | **256** | the coalesced channel tile; matches a wave64 × 4 vectorized load |
| `num_stages` | **2** | default | tiny prefetch; deeper staging gives nothing on a 4-tap conv |
| grid | `(batch, ceil(max_seqlen/BLOCK_M), ceil(dim/BLOCK_N))` | `(batch, ceil(dim/BLOCK_N))` | folds batch×channel-tiles to fill CUs |
| `KERNEL_WIDTH` | `constexpr` 2/3/4 | `constexpr` 2/3/4 | width is compiled-in → the tap loop fully unrolls |
| `NP2_STATELEN` | `next_pow2(width-1)` | `next_pow2(state_len)` | power-of-2 state for the masked state load |

These are reasonable defaults; if you re-author, the search space is small — `BLOCK_N ∈ {128,256,512}`,
`num_warps ∈ {2,4}`, `BLOCK_M ∈ {4,8,16}` (prefill). There is no MFMA to schedule, so a wider sweep is
wasted. Bake the winner per (width, dtype); do not autotune on the hot path (sourcing rule #2).

## HIP decode kernel config (aiter `causal_conv1d_update.cu`)
- `kNThreads = 64` — **one wavefront** (the comment: "Optimized for AMD wavefront size"). One block per
  batch, channels split across `blockIdx.y * 64 + tid`.
- Weights and the `width`-element sliding window are held in **per-lane registers** (`weight_vals[kWidth]`,
  `x_vals[kWidth]`), `#pragma unroll` over the ≤4 taps → no LDS, minimal VGPR.
- `kWidth` and circular-vs-linear buffer are **template params** → fully specialized, no runtime branch in
  the inner loop. Width is dispatched at the host (2/3/4 only).
- This is the right shape for decode: minimal launch footprint, state stays in cache, no LDS barrier.

## The biggest serving-time lever: split prefill/decode (not a kernel knob)
vLLM PR #17146 found the conv1d (and the Mamba2 SSD) kernels **collapse when chunked-prefill mixes
prefill+decode requests in one batch** — a single kernel sized for one regime is wrong for the other.
Splitting the batch so prefill tokens go to `causal_conv1d_fn` and decode tokens to
`causal_conv1d_update` gave a "big total throughput improvement". On AMD the same split applies; make
sure your serving stack routes the two regimes to the two kernels. This dwarfs any per-kernel tile tweak.

## CDNA3 vs CDNA4
No MFMA, so the LDS/AGPR budget changes that matter for GEMM/FMHA are irrelevant here. The op is HBM-
bandwidth + launch-overhead bound on both gfx942 and gfx950; gfx950's higher HBM BW helps the channel
loads marginally but the launch floor dominates at decode. Same kernels target both.

## How to verify a tune helped
Isolated bench at your exact `(batch, dim, width, seqlen, dtype)`, median ≥3 warm reps, vs the current
kernel; parity vs `causal_conv1d_ref` (see [numerics.md](numerics.md)). e2e: run the GDN/Mamba model and
confirm decode tok/s, gate on delta>band AND non-overlapping. Because the op is launch-bound, the e2e
signal is usually "fewer/faster launches in the GDN block", best read with a trace
(`rocprofv3`/omnitrace) showing the conv kernel duration and gaps, not just FLOPs.

## Sources
- aiter Triton launch params (BLOCK_M=8, BLOCK_N=256, num_stages=2, grid, constexpr width): `ROCm/aiter@a6bb49937:aiter/ops/triton/causal_conv1d.py`.
- aiter HIP decode kernel (64-thread/1-wave block, register sliding-window, template width/circular): `ROCm/aiter@a6bb49937:csrc/kernels/causal_conv1d_update.cu`.
- Split prefill/decode requests = big throughput win (chunked prefill): https://github.com/vllm-project/vllm/pull/17146
- ≥1024 workgroups to fill 304 CUs / launch-bound guidance: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Measured µs: perf_knowledge on-box microbench, MI300X gfx942, ROCm 7.2.0, 2026-06-05.
