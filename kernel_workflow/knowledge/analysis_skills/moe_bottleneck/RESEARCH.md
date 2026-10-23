# Distributed MoE analysis and fusion research

This document records the reusable public and local evidence behind the measurement-first
`moe_bottleneck` Skill. Stored speedups are feasibility references, never local performance targets.

## Public implementation precedents

### DeepGEMM Mega MoE

- Repository: https://github.com/deepseek-ai/DeepGEMM
- Scheduler:
  https://github.com/deepseek-ai/DeepGEMM/blob/891d57b4/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh
- Implementation:
  https://github.com/deepseek-ai/DeepGEMM/blob/54e22612/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh

Transferable:

- symmetric workspace;
- expert arrival counts;
- expert-wave state machine;
- persistent block scheduling;
- one dependency graph covering dispatch, both linears, activation, and combine.

Not directly transferable:

- Blackwell TMEM/TMA/CTA-cluster assumptions;
- SM100 FP8/FP4 layouts and scheduler constants.

### DeepEP Single-Batch Overlap

- Design PR: https://github.com/deepseek-ai/DeepEP/pull/390
- Merged LL-SBO: https://github.com/deepseek-ai/DeepEP/pull/483

Transferable:

- shared expert overlaps Dispatch Recv;
- Down GEMM publishes one signal per expert and `block_m` row group;
- Combine Send polls signals and transmits ready groups;
- communication uses fewer SMs in overlap mode;
- double-buffered data/signals prevent next-epoch overwrite while a peer still reads.

This is the closest public protocol for Stage2-producer→combine-consumer overlap.

### Comet / Flux

- Paper: https://arxiv.org/abs/2502.19811
- MLSys 2025:
  https://proceedings.mlsys.org/paper_files/paper/2025/file/e27ea0cd50b798ff8942caf9203f0992-Paper-Conference.pdf
- Code: https://github.com/bytedance/flux

Transferable:

- shared-tensor dependency decomposition;
- local-token-first receive→GEMM scheduling;
- column-wise GEMM→reduction/communication scheduling;
- communication/computation thread-block specialization;
- adaptive resource assignment.

Key warning: token-level communication and tile-level GEMM have mismatched dependencies. A token is
not automatically the best publication unit.

### Tile-level producer/consumer scheduling

- Paper: https://arxiv.org/abs/2607.19539

Transferable:

- one rank-wide persistent GEMM producer covering all local experts;
- one small persistent communication consumer on disjoint SMs;
- GEMM-epilogue tile-ready signals;
- remote-owner-aligned row layout;
- remote-heavy tiles scheduled before local work;
- segment-granular one-sided transfers.

This is one candidate intermediate architecture, not a prescribed implementation.

### AMD MORI / SGLang overlap

- MORI guide: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md
- MI355X article:
  https://www.amd.com/en/developer/resources/technical-articles/2026/win-on-tco.html
- SGLang MORI TBO: https://github.com/sgl-project/sglang/pull/19216

Transferable:

- IntraNode XGMI/P2P EP8 path;
- dual HIP streams for throughput overlap;
- async/SDMA send/receive paths;
- shared/routed expert overlap;
- communication launch geometry tuned under overlap, not in isolation.

Verify feature availability against the local MORI/SGLang revisions.

## AMD observability

### AMD Thread Trace

The prior gfx950 GEMM analysis used `rocprofv3 --att` with one warmed steady-state dispatch,
target-CU/SIMD selection, kernel regex filtering, and `SQ_INST_LEVEL_VMEM` /
`SQ_INST_LEVEL_LDS` perfcounters. Its stats CSV exposes per-instruction Hitcount, Latency, Stall,
Idle, and Source fields; viewer output adds code, wave-state, selected wave, and occupancy data.

The checked-in collector and parser are:

```text
kernel_workflow/knowledge/collectors/rocprofv3_att.json
kernel_workflow/scripts/multi_rank_analysis/instruction_analysis.py
```

ATT diagnoses instruction-pipeline causes inside a kernel. It must be combined with E2E timeline,
XGMI, rank-wait, and software-counter evidence.

### XGMI

- rocprof-sys sampling:
  https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/xgmi-pcie-sampling.html
- communication runtime profiling:
  https://rocm.docs.amd.com/projects/rocprofiler-systems/en/develop/how-to/communication-runtime-profiling.html
- TransferBench: https://github.com/ROCm/TransferBench

rocprof-sys can sample accumulated XGMI read/write data through AMD SMI. Since one MoE replay is
shorter than the sampling period, repeat an identical graph for a long controlled interval and pair
the sampled delta with exact software byte counters.

TransferBench establishes pairwise, pcopy, and all-to-all ceilings for compute-kernel and SDMA
transfer engines.

### Device-side readiness

- put-with-signal: https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/api/sigops.html
- wait/test: https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/api/pt2pt_sync.html

Required protocol:

```text
payload and scale stores
→ drain outstanding stores
→ system release
→ remote signal/arrival
→ wait/test
→ system acquire
→ consume
```

## Local MegaMoEV2 mechanisms

Current Stage1 already provides:

- arrival-ticket planner/producer/consumer CTA roles;
- parity and cross-rank launch epochs;
- destination-owned route planning;
- exact `tile_expected`/`tile_ready` publication;
- agent-scope fan-in before system publication;
- sharded persistent GEMM1 work heads;
- ping-pong GEMM1 pipelines.

Current Stage2 provides:

- persistent M scheduling;
- pipelined A/B/scales;
- GEMM2, route weighting, optional blockwise FP8 transport, and P2P scatter in one launch.

Missing architectural edges:

- no GEMM1 row-tile completion signal for Stage2;
- no Stage2 output-tile completion signal for combine;
- combine waits for whole Stage2 completion and an all-rank barrier;
- Stage1 work heads pull static indices and can wait on an unready tile; no producer-pushed ready
  queue exists.

## Verified and high-confidence safety hazards

1. Fixed-slot `group_done` used EP8 destination indices while only one element was allocated. The
   candidate fix allocates `world_size`; native GPU regression remains required.
2. Destination `tile_expected` uses local `payload_chunk_rows`, while sources publish with their own
   selected value. Unequal rank token buckets can deadlock exact waits.
3. Quantization, Stage1, Stage2, and combine do not consistently preserve a caller-supplied stream.
4. Invalid routes can be skipped during dispatch while direct combine still expects every top-k slot.
5. Exact cross-rank epoch waits have no timeout.
6. `combine_no_stage1` is enabled through a class-global flag.
7. Future in-kernel Stage2→combine consumption needs explicit system publication; current kernel
   termination supplies the ordering.
8. Queue consumers and producers must be simultaneously resident; oversubscribed spinning grids can
   deadlock.

## Six required measurement tracks

1. Frozen scattered MegaMoEV2 EP8 decomposition.
2. Cross-card bytes plus full/no-payload/local-loopback causal controls.
3. Per-expert/tile readiness, wait cycles, useful/padded work, and rank-load matrix.
4. Token/owner-row-band/GEMM-tile/chunk publication sweep.
5. Dependency DAG and Amdahl ceiling for one complete persistent kernel per rank.
6. Common workgroup, LDS/VGPR union, resident-role liveness, and bounded queues.

No high-confidence root-cause or fusion verdict is allowed while any track is missing.

## Candidate pattern catalog (non-binding)

Measured evidence may justify one or more of:

- Stage2 producer / combine consumer signaling;
- rank-wide persistent expert work;
- dispatch receive→GEMM1 readiness;
- local-first or remote-critical scheduling;
- dynamic stealing or arithmetic tile streams;
- quantization ingress;
- static roles, role reuse, or another liveness-preserving scheduler.

A one-launch implementation must prove residency, producer/consumer progress, epoch-qualified
visibility, buffer safety, and deadlock/stale-read correctness. Bounded descriptor queues and static
roles are candidate mechanisms, not requirements. Appending blocking branches to the existing
oversubscribed Stage1 grid is unsafe.
