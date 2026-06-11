---
title: profiling — trace analysis (rocprof-sys / Perfetto, gaps & host stalls)
kind: technique
gens: [gfx906, gfx90a, gfx942, gfx950]
updated: 2026-06-08
sources:
  - https://rocm.blogs.amd.com/software-tools-optimization/profilers/README.html
  - https://rocm.docs.amd.com/en/docs-6.3.0/about/release-notes.html
---

# Trace analysis: timelines, gaps, and host stalls

## TL;DR
When the problem is *between* kernels — not inside one — you need a timeline, not counters. Use
**`rocprof-sys`** (ex-Omnitrace) for a unified **host + GPU + comm (MPI/RCCL)** trace, or `rocprofv3`
for a lighter GPU-only trace; both export to **Perfetto** (`ui.perfetto.dev`). Read it for: kernel
**gaps** (GPU idle), **host overhead** (Python/launch on the CPU lane), and **CPU↔GPU sync stalls**
(`hipDeviceSynchronize`/`hipMemcpy` blocking). A GPU lane that is mostly white is a host/launch problem,
not a kernel problem — do not reach for rocprof-compute.

## Collect a trace
```bash
# Whole-system (CPU + GPU + comm), the rich view:
rocprof-sys-run -- python serve.py            # produces a Perfetto-loadable trace
# GPU-only, lightweight HIP + kernel trace via rocprofv3:
rocprofv3 --hip-trace --kernel-trace --output-format pftrace -- python serve.py
```
Open the resulting `*.pftrace` / `proto` in **ui.perfetto.dev**. (rocprof-sys binaries are
`rocprof-sys-*`; the old `omnitrace` name is dead as of ROCm 6.3.)

## What to look for, in order
1. **GPU-idle gaps between kernels.** White space on the GPU lane = the GPU is waiting. Causes:
   host can't launch fast enough (Python overhead, small kernels), or a blocking sync. Fix with
   **HIP/CUDA graphs** to replay a launch sequence with near-zero host cost
   ([`benchmarking_methodology.md`](benchmarking_methodology.md)), or kernel fusion to cut launch count.
2. **Host overhead on the CPU lane.** Long HIP API bars (`hipLaunchKernel`, `hipMalloc`) or framework
   bars dwarfing the kernel they launch → you are launch-bound. Many tiny kernels with big gaps is the
   signature; fuse or graph-capture them.
3. **CPU↔GPU sync stalls.** A `hipDeviceSynchronize` / `hipStreamSynchronize` / blocking `hipMemcpy`
   bar where the CPU sits idle waiting for the GPU (or vice versa). Remove unnecessary syncs; overlap
   copies on a separate stream; avoid per-step `.item()`/`.cpu()` in the hot loop.
4. **Comm stalls (multi-GPU).** RCCL/MPI bars where compute waits on a collective → overlap or fuse
   (e.g. [`../operators/fused_allreduce_rmsnorm/overview.md`](../operators/fused_allreduce_rmsnorm/overview.md)).

## HIP API trace specifics
The HIP API lane shows each runtime call with duration. Use it to attribute gaps: a 200 µs gap aligned
with a 200 µs `hipMemcpyDtoH` is a copy stall; a gap with *no* host bar underneath is the GPU genuinely
starved (launch latency). Correlate HIP-API timestamps with kernel-dispatch timestamps to prove which.

## When trace → which next tool
- Found the **longest kernel bar**? → profile *that* kernel with rocprof-compute
  ([`rocprof_compute_workflow.md`](rocprof_compute_workflow.md)) and read counters
  ([`reading_a_kernel_bottleneck.md`](reading_a_kernel_bottleneck.md)).
- Found **gaps / host overhead**? → graphs + fusion; this is not a kernel-internals problem.

## Pitfalls
- Tracing inflates host time; a gap that exists only under tracing may be the tracer. Confirm the
  launch-bound symptom also shows in untraced wall-time (kernel time « e2e time).
- Forgetting that profiling/tracing perturbs timing — never quote a *speed* number from a traced run;
  trace to *diagnose*, benchmark separately to *measure* ([`common_pitfalls.md`](common_pitfalls.md)).
- A "fork-storm" of host enumeration (e.g. repeated `rocm_agent_enumerator`) shows as CPU thrash on the
  host lane and starves launches — see [`common_pitfalls.md`](common_pitfalls.md).

## Verify
After a graph/fusion fix, the GPU lane should be denser (gaps shrink) and e2e wall time should drop by
roughly the removed idle, in a clean A/B.

## Sources
- rocprof-sys = unified host+device+MPI trace; rocprofv3 = lightweight GPU trace; Perfetto export; tool roles: ROCm Blogs profilers intro.
- rocprof-sys naming (ex-Omnitrace) and binary names: ROCm 6.3.0 release notes.
