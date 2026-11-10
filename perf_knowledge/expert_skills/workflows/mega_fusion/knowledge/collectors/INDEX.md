# Analysis collector registry

Collectors describe how to obtain evidence. They never prescribe an optimization. Every collector
records:

```text
tool and capability probe
exact command template
preconditions
output artifacts
field/unit semantics
what the data can diagnose
what it cannot diagnose
raw-artifact provenance
known profiler perturbation
```

## Available collectors

- `hardware_identity.json` — cross-checked AMD-SMI, rocminfo, and torch static context.
- `torch_chrome_trace.json` — per-rank launch/kernel intervals and physical overlap.
- `rocprofv3_att.json` — AMD Thread Trace instruction-level Hitcount/Latency/Stall/Idle, wave state,
  occupancy, VMEM/LDS perfcounters, and source/instruction mapping.
- `rocprof_sys_xgmi.json` — sampled XGMI link state and accumulated read/write data over repeated
  controlled intervals.
- `transferbench_ceiling.json` — pairwise, fan-out/fan-in, remote-write, and all-to-all ceilings.
- `memory_launch_ceiling.json` — validated HBM-stream and batched empty-kernel launch-floor
  measurement contract; explicitly not ready until microbenchmarks are checked in.
- `controlled_variants.json` — full/no-payload/local-loopback/compute-only/materialization
  difference contract with explicit non-additivity.
- `operator_software_counters.json` — exact logical bytes/readiness/wait/padding contract; explicitly
  marked `contract_only_not_implemented`.

Every trace, ATT, route, or software-counter artifact admitted to
`geak-analysis-bundle-v2` must carry `geak-collection-provenance-v1`.

## Required collector families

The framework should select collectors based on the question:

- hardware identity: `rocminfo`, `amd-smi --json`, `torch.cuda.get_device_properties`;
- launch/timeline/physical overlap: PyTorch profiler and rocprofv3 tracing;
- kernel hardware utilization: rocprof-compute / supported rocprofv3 counter sets;
- instruction pipeline: rocprofv3 ATT;
- XGMI bytes/topology: rocprof-sys AMD-SMI sampling;
- pairwise/all-to-all ceiling: TransferBench;
- communication calls: MORI/SHMEM/RCCL tracing where available;
- exact operator bytes/readiness/wait/padding: in-kernel software counters;
- causal exposed time: full/no-payload/local-loopback/compute-only controlled variants.

No single collector answers all questions. ATT cannot measure XGMI traffic; XGMI sampling cannot
attribute instruction stalls; summed trace durations do not establish critical-path contribution.

## Evidence confidence

- **high** — exact software counters or cross-checked static properties;
- **medium** — repeated hardware counters/traces with profiler perturbation measured;
- **low** — one sampled trace, external documentation, or an uncontrolled difference.

Analysis outputs must carry collector ID, tool version, command, timestamp, scope, repetitions,
raw-artifact path, units, confidence, and cross-checks for every derived field.
