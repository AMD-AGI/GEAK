# Measured rounds: what paid, what did not, what is left

The only measured evidence behind this skill. Three whole-layer decode kernels for MiMo-V2.6-Flash on
MI300X with four tensor-parallel ranks: two HIP C++ kernels written independently (design A, design B)
and one FlyDSL kernel that a GEAK kernel lane then optimized for four rounds. All figures are ratios or
shares. Use them to **rank** candidates for a similar layer (MoE with FP4 experts, sliding-window and
global attention layers); they are not predictions for your kernel, and the oracle still decides.

## How the three kernels are organised

Common to all: one resident workgroup per CU (grid = CU count; launch bounds and LDS sized so one
workgroup fills a CU); every workgroup runs the stages in a fixed order and waits only on earlier
stages' outputs (all resident + the same path on every rank + an acyclic wait graph = no deadlock); no
device-wide barrier between stages; both all-reduces in the kernel — each workgroup stores its output
channels into IPC-mapped uncached peer buffers with system-scope 64-bit atomics and sums in rank order.

| | HIP design A | HIP design B | FlyDSL kernel |
|---|---|---|---|
| threads per workgroup | 512, two 4-wave sides | 256, one sequence of stages | 512 (8 waves), per-stage task tables with fixed bases |
| rows | workgroup groups per row | rows as MFMA columns | tasks keyed by request |
| hand-offs | tagged (value, tag) pairs | tagged pairs; one epoch per step from round 1 on | data + drain + flag := epoch per task; (value, epoch) pairs across GPUs; waits without a deadline |

The FlyDSL kernel has a timeline build with 16 stamps per workgroup; every stage share below comes
from it.

| stage | HIP designs | FlyDSL kernel |
|---|---|---|
| input norm | residual add + RMSNorm in the native library's reduction order; FP8 quant per 128 from the FP32 normed value; QKV weights and the first K tile already in flight | a workgroup per token: hidden + residual in fp32, RMSNorm, fp8 quant per 1×128 |
| QKV | FP8 block-scale GEMV on fp8 MFMA; 16 output channels per workgroup (B) or 4 rows per wave (A); rounded to BF16, then published | tiles of 16 rows, fp8 MFMA with K split over the 8 waves; RoPE and the fp8 KV write in the epilogue |
| attention | wave-local online softmax, K straight from the cache into MFMA; RoPE + KV write folded in; sliding window: one workgroup per KV head with sinks; global: split-KV over up to one chunk per CU, then a combine | sliding window: a workgroup per (request, KV head) over the window + sink; global: a workgroup per (request, key chunk), then a combine per (token, head) |
| o_proj + first all-reduce | BF16 MFMA GEMV, weights prefetched during attention, all-reduce in the kernel | tiles of 16 rows; bf16 partials pushed to every rank |
| post-attention norm | every workgroup computes it | a workgroup per token: rank-ordered fp32 sum of the partials + residual, RMSNorm |
| router | logits spread over workgroups; top-k with cross-lane max + ballot in every workgroup | router rows × K quarters as tasks; 8 reader workgroups per token (one per XCD) do the top-k |
| experts + second all-reduce | FP4 kept in the framework's layout, permute decode in registers, routing weight folded into down | union of the step's experts; gate/up by (expert, 16-row tile); down fused with the slot sum and pushed to the all-reduce |

## Where the time goes

Native path (profile of the reference model at about 30K context, one request): attention 31%, MoE
experts 25%, FP8 GEMMs 10%, all-reduce + RMSNorm 9%, RoPE / quant / other elementwise 10%, the rest
(routing logic, router GEMM, KV write, others) about 15%; about 16–17 device kernels per layer. In a
second run of the native path only about three quarters of a decode step was kernel time, the rest gaps
between small dependent kernels.

FlyDSL kernel after the lane, sliding-window MoE layer, one request × 3 token rows (68% of the weighted
metric):

| input norm | QKV | attention | o_proj | post-attention sum + norm | router GEMV | router top-k | expert union | expert gate/up | expert down + final sum | output | layer-to-layer gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5% | 9% | 7% | 4% | 6% | 2% | 4% | 2% | **38%** | **19%** | 2% | 2.5% |

Before the lane the same layer took 1.5× as long: gate/up 39%, down + final sum 21%, everything before
the MoE 36%. Waves were still latency-bound at stage boundaries: about 57% waiting on dependencies, 26%
issuing loads, under 20% active. The lane's floor estimate (bytes at 60% of peak bandwidth, about half
of the achieved time, plus hand-offs worth 15–22%) leaves the achieved layer at 1.35–1.5× its floor.

## What paid

FlyDSL kernel, lane rounds. Metric: weighted speedup over 21 cases weighted by how often each launch
shape occurs in serving. Every kept change was bit-identical to the starting kernel on 89 recorded
serving steps.

| round | what landed (effect on its own stage) | cumulative |
|---|---|---|
| 1 | gate/up: activation hoisted to LDS, next task's weights prefetched (−19%). Down + final sum fused into one stage over all experts, fp32 slot-order sum in LDS, pushed straight into the second all-reduce (−36%). o_proj moved off the late workgroups; load-first GEMV passes; scalar peer bases; (value, tag) pairs from router top-k to the expert union | 1.19× |
| 2 | sliding-window attention: sink and old K/V loaded before the QKV wait, wait per (request, KV head), one LDS exchange instead of an 8-round merge (−52%). Down: flags polled first, weights coalesced + LDS transpose (−15%, partly lost again: down + final sum ended the lane at −40%) | 1.28× |
| 3 | gate/up: XCD-aware tile map, cached block scales, lane redistribution by LDS permute, depth-2 loop, one-wave epilogue (−23%). Global attention: block table in LDS, K/V one step ahead, combine loads first (−16% at the longest context). Router top-k: 8 readers per token, cross-lane top-k (−45%) | 1.44× |
| 4 | down weights carried in gate/up's last prefetch slots: gate/up lost about 1.5× what down gained — not committed | 1.44× |

Layer time: −34% on the sliding-window layer, −24% on the global-attention layer at the longest context,
−31% weighted. Independent directions on different stages multiplied as predicted (within 0.4%).
Earlier tuning of the same kernel: five attempts, one kept (−5.7%).

HIP designs. Metric: serving P90 inter-token latency at concurrency 1, relative to the fastest native
configuration (= 1.00); "step" is against the previous round.

| design A round | change | step | of native |
|---|---|---|---|
| 0 | two wave sides, MFMA attention, cross-lane ops, branch-free prefetch; 1 row only | — | 0.70 |
| 1 | cooperative q FP8 quant into LDS; KV write after the attention loop | −5% | 0.67 |
| 2 | 2-row steps (two workgroup groups); early V issue | −20% | 0.54 |
| 3 | FP4 permute decode; QKV / o_proj rows shared by both sides; 1 / 2 / 4 rows | −14% | 0.46 |
| 4 | K straight from memory per wave; Vᵀ XOR swizzle | +0.6% | 0.46 |
| 5 | long attention tasks: scores to a memory scratch, K read once | −15% | 0.39 |
| 6 | first K tile issued at kernel start | −2.3% | 0.38 |
| 7, 8 | top-k from per-lane sorted candidates; all-reduce pairs polled by parallel lanes | +5%, +0.1% | 0.40 |
| 9 | multi-row global attention workgroups by context length | −6% (−1.5% against the best earlier round) | 0.38 |
| 10 | expert scales in read order; XCD-aware order; padded rows return; 8 rows | −12% | 0.33 |
| 11–13 | sliding-window V / K and global V issued at kernel start | −1.2%, 0, −0.3% | 0.33 |

| design B round | change | step | of native |
|---|---|---|---|
| 0 | MFMA GEMV / attention, cross-lane ops, batched polling, prefetch (layer time to 57% of the first working version); 1 row | — | 0.67 |
| 1 | one epoch per step (layer gap −68%); q / k / v polled into registers; first KV block into LDS while waiting | −0.3% | 0.66 |
| 2 | wave-local global attention, no start barrier, FP4 multi-accumulators | flat (median −10%) | 0.66 |
| 3 | wave-local sliding-window attention; 16-byte LDS alignment (repairing a loss of about a tenth of the layer); scales before weights | −1% | 0.66 |
| 4 | 2–8 rows in the kernel | −59% | 0.27 |
| 5 | pairs of rows share norm / down reductions | −1.5% | 0.27 |
| 6–7 | cacheable 1-row expert loads; down issued during gate/up | no gain, reverted | 0.27 |

End results: A 3.05×, B 3.73× on that P90 (an accuracy benchmark was not significantly different).
Median latency with one request against native: A 1.5× at about 1K
context, 1.7× at 30K, 2.5× at 128K; B 2.0× at 1K, 2.2× at 20K. The FlyDSL kernel reached 2.25× on P90
against a *different* baseline (the fastest native configuration with multi-token prediction on), so
do not compare it with A and B; its figures were never re-measured independently, and one accuracy run at concurrency 8
took about 1.4× as long as native — a concurrency-1 gain does not carry over to wider steps for free.

## On an already written kernel: the lane's moves by mechanism

Every figure is a change in that stage's own time on the FlyDSL kernel; most rows are bundles.

| mechanism | what was done | effect |
|---|---|---|
| input-independent work in front of the wait; wait only for consumed tiles | sliding-window attention: sink and old K/V loaded before the QKV wait, wait per (request, KV head); global attention: block table in LDS, K/V one step ahead, combine loads first; load-first GEMV passes | attention stage −52% (sliding window), −16% (longest context) |
| less work after the wait | one LDS exchange instead of an 8-round merge (inside the −52%); down fused with the final sum, pushed straight into the all-reduce; batched chunk combine with the query staged in LDS | down + final sum −36%; −5.7% of the layer (the one keeper of five earlier attempts) |
| small inputs staged once in LDS | activation hoisted to LDS, with the same stage's next task prefetched; (HIP design) 16-byte LDS alignment | gate/up −19%; misalignment had cost about a tenth of a layer |
| few readers; small stages off the late workgroups | 8 top-k readers per token, one per XCD, wave-level top-k; o_proj moved off the workgroups that finish attention last | top-k stage −45%; not isolated (inside the 1.19× round) |
| another stage's weights earlier | deeper gate/up prefetch; down weights in gate/up's last prefetch slots; (HIP design) down issued during gate/up | noise; gate/up lost 1.5× what down gained; reverted |

## Dead ends (measured, with the mechanism where known)

| tried | result | mechanism |
|---|---|---|
| top-k and expert union in every workgroup | that stage 2.2× slower | every workgroup read the same few KB of logits: a hot spot in the device-shared cache |
| input norm folded into the QKV workgroups | QKV finished about 20% later; +14% time in an earlier attempt | the norm was already hidden behind the QKV weight stream; every QKV workgroup re-read hidden state and residual |
| deeper prefetch in gate/up | noise | the per-task data wait was already negligible; the stage is bound by its in-CU dependency chain |
| placement variants of the down stage | tail unchanged | bound by its own chain plus the memory time of its weights |
| exact-fill split-KV (one chunk per CU) | no gain; output drifted (relative mean error about 0.4%) | — |
| attention combine split 4× over more workgroups | the following o_proj 2.3× slower | collided with o_proj's placement; four times the flags |
| down weights carried in gate/up's prefetch slots | gate/up lost 1.5× what down gained | memory was already saturated |
| idle workgroups helping the KV loop | slower | the KV loop was already memory-bound |
| register double buffer for K/V | 19–81 spilled registers | VGPR cap |
| touch-ahead by direct global → LDS loads | serialized loads | the compiler put a full memory wait before the next LDS read |
| touch-ahead with registers before a wait | slower | — |
| wide KV stores; gating the o_proj prefetch; wave priority changes | neutral | — |
| expert tasks in rounds of six; software-pipelined K/V loads; two dim tiles per merge round | slower | — |

Lesson the lane drew: average-bandwidth slack is not usable slack — probe by injecting the bytes before
predicting a gain from overlap. Note one conflict between designs: the HIP kernels select top-k in
every workgroup, the FlyDSL kernel measured that form 2.2× slower. Measure.

## Next directions the lane ranked (estimates, not done)

| rank | direction | estimated gain |
|---|---|---|
| 1 | pipeline gate/up and down per expert | about 13% of the layer |
| 2 | router GEMV inside the workgroups of the post-attention sum + norm, and top-k → union in one step | about 6% |
| 3 | a finer partial round in gate/up | about 2% |
| 4 | early KV fetch for global-attention workgroups that also run a QKV tile | about 2.5% at the longest context |
| 5 | split straggler QKV tiles along K | not estimated |

## Reading

- The largest step in both HIP designs was taking concurrent (multi-row) decode steps into the kernel;
  the 1-row path had plateaued at about two thirds of native latency.
- After that: reading K once at long context (−15%), memory-layout fixes for scales with placement
  (−12%). Early-issue tweaks were at most 2–3% each on P90, even when a standalone bench improved.
- The fixed-order 256-thread design with rows as columns reached the better result in fewer rounds
  (7 against 13) than the two-sided design.
- On an already tuned kernel the remaining time sits in the expert stages and in dependency waits;
  earlier tuning of that kernel kept one change in five.
- Plateau rule used: stop after three consecutive rounds each under 2–3% better than the best so far.
- Save every verified step at once: work that was finished but not yet integrated when the next round
  started was lost for that round.
