---
title: MI300X occupancy estimates and tuning
kind: hardware
gens: [gfx942]
dtypes: []
regimes: [both]
updated: 2026-10-16
---

# MI300X occupancy estimates and tuning

Occupancy estimates depend on register allocation, LDS, workgroup placement and
hardware wave slots. Use the target and wave mode recorded in the executable;
compiler resource requests and rounded allocations are distinct. Do not assume
one 16-register rounding rule or add VGPR and AGPR fields without their decoder.

Collect the actual launch dimensions, allocated register metadata, fixed and
dynamic LDS, and private-memory accesses. Use the installed compiler/tool's
occupancy calculation with the matching gfx942 definitions. A hand-derived
register quotient alone omits other residency limits and allocation rules.

| Hypothesis | Useful comparison |
|---|---|
| Live values limit residency | Smaller tiles or changed lifetimes; inspect allocation and scratch accesses |
| Shared memory limits resident workgroups | Fewer buffers or different packing; include dynamic LDS |
| Loads expose latency | More independent requests, prefetch or a different wave count; account for added resources |
| Communication costs dominate | Different ownership, layout or synchronization; measure the whole operation |

`waves_per_eu` and launch-bound settings influence compilation; they do not force
achieved residency. More occupancy is not always faster. Measure correctness and
unprofiled latency, and use documented occupancy counters only with their time
interval, active/all-CU denominator and short-kernel limitations. See the
[resource-evidence guide](../../skills/kernel-hardware-evidence/references/ir-isa-dataflow.md).

## Sources

- [LLVM AMDGPU resource definitions](https://llvm.org/docs/AMDGPUUsage.html)
- [HIP hardware implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
- [ROCm MI300X workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
