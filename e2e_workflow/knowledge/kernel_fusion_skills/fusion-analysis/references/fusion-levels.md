# Fusion levels

Kernel fusion is classified into four levels by scope.

## L1 — Element-wise fusion

Fuses several lightweight element-wise kernels.

Typical cases:

- add + mul;
- activation + element-wise transform;
- the lightweight chain after a normalization;
- several consecutive kernels of 1–5 µs each.

Consider L1 first when:

- each kernel does little computation;
- launch overhead is a large share of the time;
- the kernels have a simple, direct producer-consumer dependency;
- intermediate tensors are repeatedly written to and read from HBM;
- fusion does not noticeably increase resource pressure.

Goals: fewer kernel launches; less intermediate HBM traffic.

## L2 — Operator-wise fusion

Fuses one dominant operator with the lightweight operations before and after it.

Typical cases:

- GEMM + bias + activation;
- GEMM + quantization;
- attention + local pre/post-processing;
- normalization + projection.

Consider L2 first when:

- one operator takes most of the region's compute time;
- several lighter kernels surround it;
- the intermediate tensors are large;
- an epilogue or prologue fusion removes an HBM round trip;
- fusion does not noticeably degrade the dominant operator's optimal tile, VGPR/LDS use or occupancy.

Goals: keep the dominant operator at full performance while removing the surrounding small kernels and
their intermediate memory traffic.

## L3 — Stage-wise fusion (MegaKernel)

Fuses several operators of one functional stage.

Typical cases:

- dispatch + expert GEMM + combine;
- router + metadata preparation + dispatch;
- several consecutive stages inside attention;
- communication overlapped with computation.

Consider L3 first when:

- a stage contains several interdependent kernels;
- no single operator dominates the stage;
- there is a clear producer-consumer pipeline;
- several launches or synchronizations can be removed;
- communication and computation can overlap;
- there are enough tokens, batch or concurrency to keep the fused pipeline busy.

Observation: Mega and MegaMoE kernels usually gain more at higher concurrency — more tokens, a larger
effective batch per expert, higher expert-GEMM utilization and more room for communication/computation
overlap. High concurrency is a prior, not a rule.

## L4 — Layer-wise fusion (MonoKernel)

Fuses most or all stages of a decoder layer into one persistent or monolithic kernel.

Typical cases:

- the decoder layer is split into many short kernels;
- attention, router and MoE run as a persistent workgroup pipeline.

Consider L4 first when:

- concurrency or batch size is low;
- the workload is latency-sensitive (high interactivity);
- the layer contains many short kernels;
- launch and synchronization overhead is a large share of the time;
- the stages have strong serial dependencies;
- stages can overlap through tile-level or stage-level pipelining;
- intermediate HBM traffic can be removed;
- the VGPR/LDS/occupancy degradation after fusion is acceptable.

Observation: MonoKernel usually gains more at low concurrency and high interactivity. Low concurrency
alone does not justify it; also establish:

- whether there are many short kernels;
- whether launch and synchronization overhead is actually significant;
- whether GEMM and attention already take nearly all of the time;
- whether whole-layer fusion would cause a severe occupancy loss;
- whether the stages' tile and resource requirements conflict.
