# RCCL & Communication for Distributed LLM Inference on MI300X (gfx942)

> Scope: collectives, xGMI/Infinity Fabric topology, RCCL/NCCL env tuning, comm/compute overlap, MoE
> all-to-all, and custom all-reduce for **TP / EP / DP** LLM inference on AMD Instinct MI300X. AMD-only.
> Verified against ROCm RCCL docs (2.22-2.28), AMD xGMI blog, sglang/vllm ROCm guidance (2024-2026).
>
> Key reality check (from the vLLM-on-MI300X team): LLM inference is compute- and memory-bound, so RCCL
> tuning gains are usually **modest** (single-digit %) — *until* comm becomes the bottleneck (small TP
> groups, EP all-to-all, multi-node, large all-reduce). Know **when** comm dominates; spend there.

---

## 1. Collectives in LLM inference — which op, which parallelism

| Collective | Where it appears | Parallelism |
|---|---|---|
| **all-reduce** | after each attn + MLP (sum partial activations) | **TP** — the dominant inference collective |
| **all-gather** | gather sharded weights/activations; sequence-parallel | TP / SP |
| **reduce-scatter** | TP with sequence parallelism (pairs with all-gather) | TP+SP |
| **all-to-all** | MoE expert **dispatch** (tokens→experts) + **combine** (experts→tokens) | **EP** |
| **broadcast / gather** | sampling, logits, DP coordination | DP |

Per-layer cost: TP all-reduce fires **2× per transformer layer** (attn out, MLP out). On a 70B model
TP=8 that is ~160 all-reduces per forward — small messages at decode (M=batch), large at prefill.

---

## 2. xGMI / Infinity Fabric topology (8× MI300X node)

| Property | Value |
|---|---|
| Interconnect | xGMI (External Global Memory Interconnect) over Infinity Fabric |
| Topology | **fully connected** — each GPU has a dedicated link to each of the other 7 |
| Per-link BW | 64 GB/s theoretical, **45-48 GB/s realized** |
| Per-GPU aggregate (unidir) | 7×64 = 448 GB/s theoretical, **315-336 GB/s realized** |
| Measured all-reduce busbw (8-GPU, 16G msg) | **~319 GB/s** standard, ~319-330 GB/s with graph mode (`-G 1`) |
| Bottleneck | the **slowest link** caps the collective (e.g. one 45.2 GB/s link → ~316 GB/s aggregate) |

Implications:
- **Use all 8 GPUs to get full bandwidth.** A 2- or 4-GPU collective uses only a fraction of the links →
  only a fraction of node bandwidth. RCCL compensates by pre-defining **more channels** for small groups
  (32 channels at TP=2, 24 at TP=4 on an 8×MI300X box).
- **Stay within one xGMI island (≤8 GPUs, TP only).** Beyond 8 / cross-node, use **PP** (pipeline) — do
  not stretch TP across nodes (you fall off xGMI onto slower fabric/NIC).
- There is **no NVLink/NVSwitch analog with a central switch** — it's a direct mesh, so collective algos
  (ring/tree) and channel counts behave differently than on H100.

---

## 3. RCCL / NCCL env tuning table

RCCL is NCCL-API-compatible, so most knobs use the `NCCL_*` prefix; AMD adds `RCCL_*` extensions.

| Env var | Default / range | Effect | MI300X guidance |
|---|---|---|---|
| `NCCL_MIN_NCHANNELS` | RCCL picks per-topo (32 @TP2, 24 @TP4) | min channels (parallelism of the collective) | **`112` for single-node E2E** is the AMD recommendation; biggest knob for sub-8-GPU TP. **Setting it bypasses RCCL's MI300X tuning model** — benchmark both ways |
| `NCCL_MAX_NCHANNELS` | topo | cap channels | rarely needed; also bypasses tuning model |
| `NCCL_THREAD_THRESHOLDS` | tuned | LL/LL128 thread thresholds | setting it (or MIN/MAX_NCHANNELS) **disables** the channel tuning model |
| `RCCL_MSCCLPP_THRESHOLD` | 1 MB | msg-size cutoff to use MSCCL++ fast kernels | small-msg TP collectives benefit; raise to push more sizes through MSCCL++ (where supported) |
| `NCCL_PROTO` | auto | `LL` / `LL128` / `Simple` | leave auto; LL/LL128 chosen by tuning model for small msgs |
| `NCCL_ALGO` | auto | `Ring` / `Tree` / `CollnetDirect` | leave auto on single node mesh |
| `NCCL_IGNORE_CPU_AFFINITY` | 0 | ignore job CPU affinity, use GPU affinity | `=1` helps multi-node scaling |
| `NCCL_P2P_LEVEL` | auto | P2P transport threshold | keep P2P on (xGMI is the point) |
| `NCCL_NET_GDR_LEVEL` / `RCCL_NET_GDR_LEVEL` | — | GPUDirect RDMA level | `=2` for **multi-node** (RDMA NIC) |
| `NCCL_IB_GID_INDEX` | — | RoCE GID index | set (e.g. `3`) for RoCE multi-node |
| `RCCL_ENABLE_CONTEXT_TRACKING` | 0 | per-GPU context tracking | `=1` can "significantly improve performance in certain scenarios" |
| `HSA_FORCE_FINE_GRAIN_PCIE` | 0 | P2P over PCIe (needs large BAR) | `=1` for PCIe-connected GPUs |
| `NCCL_DEBUG` / `RCCL_DEBUG` | — | `VERSION` / `INFO` | `INFO` + `RCCL_DEBUG_SUBSYS=INIT,GRAPH` to see chosen algo/channels |
| `NPKIT_DUMP_DIR` | — | NPKit event trace dir | profiling (one GPU per process) |

> Deprecation note: upstream NCCL deprecated `NCCL_MIN_NCHANNELS` in favor of `NCCL_MIN_CTAS`. **RCCL
> still documents and honors `NCCL_MIN_NCHANNELS`** — keep using it on ROCm.

MSCCL / MSCCL++: enabled by default on MI300X; gives efficient small-message all-reduce/all-gather
kernels. (Note: some recent RCCL builds turned legacy MSCCL API symbols into no-ops for link compat —
the MSCCL++ kernel path is what matters; controlled by `RCCL_MSCCLPP_THRESHOLD`.)

---

## 4. Framework custom all-reduce (often beats RCCL for small messages)

For TP all-reduce at decode the messages are **small** and latency-bound; framework custom AR usually
beats stock RCCL there.

| Framework | Mechanism | How to engage |
|---|---|---|
| **sglang** | AITER custom all-reduce / all-gather | `SGLANG_USE_AITER_AR=1`, `SGLANG_USE_AITER_AG=1` (needs `SGLANG_USE_AITER=1`) |
| **vLLM** | Quick Reduce (quantized custom AR, ROCm) | `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION={FP,INT8,INT6,INT4}`, `_CAST_BF16_TO_FP16=1`, `_MAX_SIZE_BYTES_MB` |
| both | xGMI peer-to-peer custom AR | on by default within an island |

Quick Reduce **quantizes the reduction** (INT8/FP/INT6/INT4) to cut bytes on the wire — a bandwidth play
for large AR; treat any non-`NONE` setting as an **accuracy gate** (it changes the reduced values).

Caveat: AITER custom AR has had stability bugs on MI300X (e.g. aiter issue #1542, AR kernel segfault) —
if you see crashes in the AR path, fall back to `SGLANG_USE_AITER_AR=0` (stock RCCL).

---

## 5. MoE all-to-all (EP dispatch / combine)

MoE expert parallelism turns the FFN into two all-to-alls per layer: **dispatch** (route tokens to the
GPU holding their expert) and **combine** (gather expert outputs back). At scale this is the comm
bottleneck, not all-reduce.

| Backend | Engage | Note |
|---|---|---|
| **DeepEP** | sglang `--enable-deepep-moe` | tuned all-to-all dispatch/combine kernels for EP |
| vLLM all2all backend | `VLLM_ALL2ALL_BACKEND="allgather_reducescatter"` | recommended for ROCm DP+EP; `--enable-expert-parallel`, `--data-parallel-size N`, `--disable-nccl-for-dp-synchronization` |
| MoRI / pplx | framework-specific | alt all-to-all libs |

For large MoE (DeepSeek/Kimi) the winning topology is usually **DP attention + EP MoE** (each GPU owns a
DP rank for attention, experts sharded across the island). Tune the all-to-all (`RCCL_MSCCLPP_THRESHOLD`,
DeepEP) before touching anything else — it can be 30-50% of decode time at high EP.

---

## 6. Overlapping comm with compute

| Technique | How |
|---|---|
| High-priority RCCL streams | `TORCH_NCCL_HIGH_PRIORITY=1` — RCCL streams don't always overlap compute; forcing high priority helps (esp. FSDP-style) |
| More HW queues | `GPU_MAX_HW_QUEUES=2` (HIP) — lets comm + compute kernels run on separate queues |
| Multi-stream | sglang `SGLANG_ROCM_USE_MULTI_STREAM=1` |
| Tensor-register hook | `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1` (helps registration/overlap) |
| Graph mode | RCCL `-G 1` (HIP-graph-captured collectives) lowers small-msg launch latency ~3-5% |
| CPU affinity | `--bind-to numa` (mpirun), `NCCL_IGNORE_CPU_AFFINITY=1`, `numa_balancing=0` |

Sequence/tensor-parallel comm can be hidden behind GEMM if streams are independent; verify with a
rocprofv3 trace that the all-reduce kernel overlaps the next layer's GEMM rather than serializing.

---

## 7. When is comm the bottleneck? (decision guide)

| Signal | Comm-bound? | Action |
|---|---|---|
| TP=2/4 (sub-island), small model | **likely** | `NCCL_MIN_NCHANNELS=112`, custom AR (`SGLANG_USE_AITER_AR=1` / Quick Reduce) |
| TP=8 single island, dense decode | usually **not** (compute/mem bound) | leave RCCL on tuning model; small gains only |
| EP MoE, high concurrency | **yes** (all-to-all) | DeepEP / `VLLM_ALL2ALL_BACKEND`, `RCCL_MSCCLPP_THRESHOLD` |
| Multi-node / cross-island | **yes** (off xGMI) | switch to PP across nodes; `NCCL_NET_GDR_LEVEL=2`, IB/RoCE tuning |
| rocprofv3 shows AR kernels not overlapping GEMM | **yes** | high-priority streams, `GPU_MAX_HW_QUEUES=2` |
| prefill (large M) | rarely | comp dominates; AR is large but amortized |

Diagnose with **rccl-tests** to get the ceiling, then compare to in-app comm time:
```bash
mpirun -np 8 --bind-to numa -env NCCL_DEBUG=VERSION \
  rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1 -G 1   # add -G 1 for graph mode
# expect ~316-330 GB/s busbw on a healthy 8×MI300X node; lower => a slow xGMI link
./TransferBench a2a 64M 8   # per-link all-to-all sanity (catches a degraded link)
```
If `all_reduce_perf` busbw is well below ~310 GB/s, you have a hardware/link problem — fix that before
any software tuning.

---

## 8. Optimizer checklist

1. **Baseline the fabric**: `rccl-tests all_reduce_perf` (+`-G 1`) and `TransferBench a2a` — confirm
   ~316-330 GB/s and no slow link.
2. **Single node TP≤8**: try `NCCL_MIN_NCHANNELS=112` + framework custom AR (sglang `*_AITER_AR`, vLLM
   Quick Reduce). A/B against RCCL defaults (the 112 setting bypasses the tuning model).
3. **MoE/EP**: enable DeepEP / `VLLM_ALL2ALL_BACKEND`, tune `RCCL_MSCCLPP_THRESHOLD`.
4. **Overlap**: `TORCH_NCCL_HIGH_PRIORITY=1`, `GPU_MAX_HW_QUEUES=2`, multi-stream; verify overlap in a
   trace.
5. **Multi-node**: `NCCL_NET_GDR_LEVEL=2`, `NCCL_IB_GID_INDEX`, prefer PP across nodes; keep TP in-island.
6. **Accuracy gate** any quantized custom AR (Quick Reduce INT/FP) — it changes reduced values.

---

## Sources
- RCCL usage tips (env vars, tuning model, channel defaults): https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html
- Understanding RCCL Bandwidth & xGMI on MI300X (topology, bandwidth, rccl-tests): https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html
- MI300X workload optimization (RCCL/NCCL_MIN_NCHANNELS, affinity): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- vLLM serving on AMD MI300X best practices (NCCL tuning, TP sizing): https://blog.vllm.ai/2024/10/23/vllm-serving-amd.html
- RCCL releases (MSCCL++, versions): https://github.com/ROCm/rccl/releases
- AITER all-reduce segfault on MI300X (issue #1542): https://github.com/ROCm/aiter/issues/1542
- vLLM V1 ROCm optimization (Quick Reduce, all2all backend): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html
