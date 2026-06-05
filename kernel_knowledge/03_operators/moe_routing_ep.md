# MoE Routing & Expert/Data Parallelism on AMD MI300X (CDNA3 / gfx942)

> Scope: the **routing** half of MoE (top-k gating math, routing/permute kernels) and the **distributed**
> half (expert parallelism, all-to-all dispatch/combine, DeepEP, load balancing/capacity). AMD-only
> (gfx942). Pairs with `moe.md` (full fused pipeline) and `grouped_gemm.md` (the expert GEMM that
> consumes the routed layout). The routing kernels are small but on the critical path; the EP comm can
> dominate at scale.

---

## 1. Top-k gating math

Per token `x∈R^H`, gate weight `W_g∈R^{H×E}`:

```
logits  = x · W_g                       # [E]
topk_v, topk_i = topk(logits, k)        # k experts/token (k=2 Mixtral, 8 DeepSeek-V3)
# normalization variants:
#   softmax-then-topk:  p = softmax(logits); take top-k of p
#   topk-then-softmax:  g = softmax(topk_v)            (DeepSeek/most: renormalize the chosen k)
#   sigmoid + group:    DeepSeek-V3 grouped gate (node-limited routing)
g[token, j] = g_e   for j in 0..k-1     # combine weights, sum-to-1 over the k
```

DeepSeek-V3 specifics worth knowing: **sigmoid** gating (not softmax) with a learned per-expert bias for
load balancing, **group-limited routing** (experts partitioned into groups; a token may only pick experts
from a few groups → bounds the number of nodes a token's compute touches → cheaper EP comm), and
**no aux-loss** (bias-based balance). The routing kernel must implement the model's exact variant — get
the normalization wrong and you silently lose accuracy.

This is a **tiny GEMM + a top-k reduction**: cheap FLOPs, but a launch and a small reduction on the
critical path. Often fused with the subsequent argsort/cumsum.

---

## 2. Routing / permute kernels on MI300X

After top-k you have `topk_ids[T,k]` and `topk_weights[T,k]`. To feed grouped GEMM you must reorder tokens
so each expert's rows are contiguous and block-aligned. This is the **MoE align & sort** kernel
(detailed in `moe.md` §3) producing `sorted_token_ids`, `expert_ids` (per-block), `num_tokens_post_pad`.

The permute itself is a **gather/scatter**:

```python
# PERMUTE (gather): build expert-contiguous activations (often fused into the GEMM A-load, see moe.md)
@triton.jit
def moe_permute_kernel(X, X_sorted, sorted_token_ids, num_valid, H,
                       BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)                          # one program per sorted slot (row)
    src = tl.load(sorted_token_ids + pid)          # which original token row -> this slot
    if src >= num_valid:                            # padding slot: skip
        return
    offs_h = tl.arange(0, BLOCK_H)
    row = tl.load(X + (src // TOP_K) * H + offs_h)  # gather H-vector (k slots share a token)
    tl.store(X_sorted + pid * H + offs_h, row)      # contiguous write in expert order

# UNPERMUTE + COMBINE (scatter-add): sum the k expert outputs back per token, weighted by g_e
@triton.jit
def moe_unpermute_combine_kernel(Y_sorted, Y, sorted_token_ids, topk_weights,
                                 num_valid, H, TOP_K: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    src = tl.load(sorted_token_ids + pid)
    if src >= num_valid: return
    offs_h = tl.arange(0, BLOCK_H)
    o = tl.load(Y_sorted + pid * H + offs_h)
    w = tl.load(topk_weights + src)                # routing weight for this (token,expert) slot
    tl.atomic_add(Y + (src // TOP_K) * H + offs_h, o * w)   # accumulate k contributions per token
```

MI300X notes:
- The **gather/scatter is bandwidth-bound** (one HBM read + one write per element, irregular addresses).
  Coalesce on the H axis (contiguous), vectorize 128-bit (`dwordx4`), and keep the irregular index on the
  outer (row) axis.
- **Best practice: fuse the permute into the expert GEMM's A-load** (the `offs_token` gather in `moe.md`
  §2) so you never materialize `X_sorted` in HBM — saves a full `[T·k, H]` round-trip.
- The combine's `atomic_add` (k contributions per token) is the typical unpermute; alternatively the
  layout guarantees each token's k slots are known and you do a deterministic k-way add (avoids atomics,
  better on gfx942 where global atomics contend). `MUL_ROUTED_WEIGHT` in the GEMM epilogue can pre-apply
  `g_e` so combine is a plain add.
- **XCD awareness** (same as align/sort): size the grid to a multiple of 8, keep work XCD-local; cross-die
  L2 traffic hurts these memory-bound kernels (recall MI100 > MI300X on the pure sort).

---

## 3. Expert parallelism (EP): the distributed picture

When the model's experts don't fit on one GPU (or to scale throughput), experts are **sharded across GPUs**
(expert parallel). Each token must be **sent** to the GPU(s) holding its top-k experts (dispatch),
computed there, and the results **sent back** and combined (combine). This is an **all-to-all**.

| Parallelism | What is sharded | MoE use |
|---|---|---|
| **TP** (tensor) | each weight matrix split across GPUs | shared expert / dense parts; high comm per layer |
| **EP** (expert) | whole experts assigned to GPUs | the routed FFN; all-to-all dispatch/combine |
| **DP** (data) | replicate, split the batch | attention + router; pairs with EP for the FFN |
| **EP+DP** | DP for attention, EP for experts | **DeepSeek/vLLM standard**: scales throughput |
| **EP+TP** | TP within an EP group | low-latency / interactive, low concurrency |

AMD's MI300X crossover finding: **TP+EP** wins low-latency interactive (low concurrency); **DP+EP** wins
high-throughput batch. The 192 GB HBM per GPU means DeepSeek-R1 (671B) fits in a **single 8×MI300X node
without pipeline parallelism**, so intra-node **xGMI/Infinity Fabric** carries the all-to-all (no RDMA)
for single-node EP — a big latency advantage.

---

## 4. DeepEP — the dispatch/combine comm library

DeepEP (DeepSeek) is the reference high-perf EP comm library: high-throughput + low-latency all-to-all
GPU kernels for MoE **dispatch** and **combine**, with FP8 dispatch and near-zero SM occupation. AMD
maintains a port (**ROCm/DeepEP**) that swaps NVIDIA's NVSHMEM for **rocSHMEM**.

### Two kernel modes (V1 / the modes vLLM exposes)

| Mode | Optimized for | vLLM flag | Output to experts |
|---|---|---|---|
| **normal / high-throughput** | prefill, large batches; intranode (NVLink/xGMI) + internode (RDMA) forwarding | `--all2all-backend deepep_high_throughput` | standard (variable per-expert counts) |
| **low-latency** | decode, small batches; pure RDMA, minimal SM, hook-based overlap | `--all2all-backend deepep_low_latency` | **batched/masked** (fixed per-expert capacity) |

- **high-throughput** kernels forward tokens efficiently across the NVLink/xGMI (intranode) + RDMA
  (internode) domains; designed for max bandwidth, more SMs.
- **low-latency** kernels minimize latency and SM occupation for decode; produce a **masked/batched**
  layout (fixed capacity per expert) so the downstream GEMM is a static **batched** GEMM
  (`batched_gemm.md`) — no dynamic align/sort on the hot decode path. Async → enables **dual-batch
  overlap (DBO)** and shared-expert overlap.

> "high_throughput and low_latency kernels are optimized for disaggregated serving and may show poor
> performance for **mixed** workloads" — pick the mode per phase (prefill vs decode), don't share one.

### API shape (legacy/V1, what the ROCm port tracks)

```python
# 1. compute routing layout (per-rank/per-expert token counts)
num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, handle = \
    buffer.get_dispatch_layout(topk_idx, num_experts)
# 2. dispatch: send tokens to the GPUs owning their top-k experts (optionally fp8)
recv_x, recv_topk_idx, recv_topk_w, num_recv_per_expert, handle, event = \
    buffer.dispatch(x, topk_idx, topk_w, num_tokens_per_rank, num_tokens_per_expert,
                    use_fp8=True)          # fp8 input passed as (data, scales)
#    ... run grouped/expert GEMM on recv_x (num_recv_per_expert -> group sizes) ...
# 3. combine: reduce expert outputs back to the originating ranks
y, event = buffer.combine(expert_out, handle)
```

`handle` carries the routing metadata so combine is the exact inverse of dispatch.
`num_recv_per_expert` feeds straight into grouped-GEMM group sizes.

### Numbers (V3 config, 8K tok, 7168 hidden, top-8, fp8 dispatch / bf16 combine; NVIDIA ref)

| Topo | Dispatch BW | Combine BW | #SMs |
|---|---|---|---|
| intranode NVLink (EP8) | ~726 GB/s | ~740 GB/s | 64 (max) |
| intranode NVLink (EP8) | ~643 GB/s | ~675 GB/s | 24 (min SM) |
| internode RDMA (EP8×2, CX7) | ~90 GB/s | ~81–91 GB/s | 12 |

On MI300X substitute **xGMI/Infinity Fabric** for NVLink intranode and **ConnectX-7 / AMD Pensando**
RDMA internode. AMD candidly notes the ROCm port mirrors the upstream API but lags NVIDIA on advanced
tuning (NVIDIA-only PTX-load tricks not mirrored); correctness + baseline overlap first. **DeepEP V2**
(NCCL-Gin backend, unified `ElasticBuffer`, analytical SM count, up to 1.3× perf + 4× fewer SMs) is newer
and **not yet fully mirrored in the ROCm port**. AMD also ships **ROCm/mori**, a next-gen comm library
for Wide-EP / KV-cache transfer / collectives.

---

## 5. Load balancing & capacity

Routing is **skewed**: a few hot experts get most tokens, some get none — and skew is worst in **decode**
(small token count). Two regimes:

| Strategy | Mechanism | Trade-off |
|---|---|---|
| **Drop-free (variable)** | process exactly `M_e` tokens/expert via grouped GEMM + align/sort | no token dropped; dynamic shapes; tail latency from hot expert |
| **Capacity (fixed)** | cap each expert at `C = factor · T·k/E`; drop overflow, pad underflow | static batched GEMM; some tokens dropped (accuracy) or padding waste |

- **Training-time balance** (aux loss / DeepSeek bias-based) flattens the distribution so inference skew
  is mild — but never zero.
- **Inference**: drop-free grouped GEMM is the default (no accuracy loss). The **low-latency DeepEP**
  path uses fixed **capacity/masked** layout for static decode shapes (pad to capacity; the GEMM masks
  padding) — trading a little padding for a static, low-latency batched kernel.
- **Group-limited routing** (DeepSeek-V3) bounds how many *nodes* a token touches → caps EP all-to-all
  fan-out → directly cheaper dispatch/combine.
- **Hot-expert mitigation on MI300X**: split-K the hottest expert's GEMM, or raise the grouped-GEMM
  scheduler's tile-sharing (NUM_SM stride) so one expert's many tiles spread across all 304 CUs.

---

## 6. Optimizing routing kernels on MI300X — checklist

| Kernel | Bottleneck | gfx942 levers |
|---|---|---|
| gate GEMM + top-k | small GEMM + reduction launch | fuse gate+topk+argsort; cheap, keep on critical path short |
| align & sort | memory-bound, cross-die sync | grid = multiple of **8 XCDs**; ~5KB LDS, <7% bank conflict, 0 spills; 1-warp-load/1-warp-compute (see `moe.md` §3) |
| permute (gather) | HBM bandwidth, irregular | **fuse into GEMM A-load** (no `X_sorted` materialization); 128-bit coalesced on H |
| unpermute+combine | scatter-add, atomics | pre-apply `g_e` in GEMM epilogue; deterministic k-add over atomics; coalesce on H |
| EP dispatch/combine | xGMI/RDMA all-to-all | pick mode per phase (HT prefill / LL decode); fp8 dispatch halves bytes; overlap with shared-expert / DBO; single-node xGMI when model fits 192GB |

> Recurring MI300X lesson for *all* the bookkeeping kernels here: they are **memory-bound**, and the
> multi-die XCD interconnect penalizes cross-die traffic. Keep grids XCD-aligned (×8), keep data
> XCD-local, avoid `hipCooperativeLaunch` patterns that inflate die-die L2 traffic. The compute (GEMM)
> half loves the 304 CUs; the routing/comm half is gated by bandwidth and fabric.

---

## 7. Backend ladder (MI300X routing + EP)

| Tier | Mechanism | Edit? |
|---|---|---|
| A — select | aiter moe-sort vs SGLang align-sort vs Triton; DeepEP HT vs LL vs naive | no |
| B — tune | grid/block of sort+permute; EP SM count / mode per phase | no |
| C — rewrite | edit Triton permute/combine; fuse permute into GEMM | yes |
| D — quant | fp8 dispatch (halve comm bytes) | flag → accuracy gate |

| Component | First try on gfx942 |
|---|---|
| align & sort | **SGLang custom HIP** (7× vs Triton) or aiter moe-sorting kernel → Triton fallback |
| permute/combine | fuse into aiter/Triton fused_moe (no standalone buffer) |
| EP comm | **ROCm/DeepEP** (rocSHMEM): HT for prefill, LL for decode; fp8 dispatch; or ROCm/mori for Wide-EP |
| EP topology | single-node xGMI when model fits 192GB×8; else RDMA (CX-7 / Pensando) |

---

## Sources
- DeepEP (DeepSeek) — dispatch/combine kernels, normal vs low-latency, fp8, SM/overlap, BW numbers: https://github.com/deepseek-ai/DeepEP
- ROCm/DeepEP — AMD port (rocSHMEM, MI300X/MI308X, CX-7 / Pensando NICs): https://github.com/ROCm/DeepEP
- Efficient MoE Align & Sort in SGLang (routing/sort kernel, XCD awareness, 7× MI300X): https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang
- The vLLM MoE Playbook: TP/DP/PP/EP on AMD MI300X (TP+EP vs DP+EP crossover): https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html
- vLLM Expert Parallel Deployment (deepep backends, all2all flags): https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- vLLM Fused MoE Kernel Features (dispatch/combine backend × quant matrix): https://docs.vllm.ai/en/latest/design/moe_kernel_features/
- Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X (single-node EP, 671B in 192GB×8): https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html
- ROCm 7.2 release (DeepEP improvements, mori comm library): https://rocm.blogs.amd.com/software-tools-optimization/rocm7.2/README.html
