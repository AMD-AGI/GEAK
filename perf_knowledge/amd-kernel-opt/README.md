# AMD GPU Kernel Optimization Knowledge Base

![format](https://img.shields.io/badge/format-OKF%20v0.1-blue)
![target](https://img.shields.io/badge/target-MI300X%20gfx942%20(CDNA3)-E1140A)
![kernels](https://img.shields.io/badge/kernels-91%20referenced%20·%2018%20case%20studies-success)
![patterns](https://img.shields.io/badge/patterns-12%20·%20anti--patterns%204-informational)

> Distilled, reusable success experience from real GPU kernel-optimization campaigns on
> **AMD Instinct MI300X (gfx942 / CDNA3)**. For any new kernel it answers: *what levers
> actually worked, why, on which bottleneck — and when to stop.*

This is a curated knowledge base, not a code library. Every entry is a plain markdown file
in **[Open Knowledge Format (OKF v0.1)](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)**
— readable on GitHub, greppable, and machine-parseable. It is organized along two axes:
**operator domain** (attention / MoE / GEMM / KV-cache / …) and **bottleneck class**
(launch / occupancy / memory / compute).

## The one-line takeaway

**Host-side levers — no kernel rebuild — deliver most of the ROI.** Graph replay, routing to a
faster prebuilt kernel, and `block_m` schedules repeatedly beat hand-tuning the kernel body.
Classify the bottleneck *before* you tune, and verify a win survives end-to-end (not just the
microbenchmark).

## Start here

1. **[methodology/bottleneck-first-classification.md](methodology/bottleneck-first-classification.md)** — classify the bottleneck before tuning.
2. **[index.md](index.md)** — full progressive-disclosure index.
3. **[catalog/kernel-registry.md](catalog/kernel-registry.md)** — every referenced kernel, its speedup, and a link to its case study.

## Headline results

| kernel | speedup | the lever that worked |
|---|---:|---|
| [fused_moe_int4_w4a16](cases/fused-moe-int4-w4a16.md) | **5.19×** | int4 load-once-unpack both nibbles + scale/zp group dedup (cut L2 read BW) |
| [paged_attention_decode](cases/paged-attention-decode.md) | **4.39×** | host-side route CK → prebuilt ASM decode kernel (gated fallback) |
| [_per_token_group_quant_fp8](cases/per-token-group-quant-fp8.md) | **2.90×** | memory-bound quant rewrite |
| [write_req_to_token_pool](cases/write-req-to-token-pool.md) | **2.05×** | host launcher: `do_not_specialize` + cached launch, skip `launch_metadata` |
| [_topk_forward](cases/topk-forward.md) | **1.90×** | HIP-graph replay on a launch-bound op |
| [gemm_a8w8_blockscale](cases/gemm-a8w8-blockscale.md) | **1.82×** | graph replay + per-shape CK-vs-ASM dispatch |
| [moe_stage2](cases/moe-stage2.md) | **1.31×** | CK pipeline V3→V1 (2 blocks/CU) + `block_m` to routing sparsity |

End-to-end serving (MiniMax-M2.5, 4×MI300X): **+13% to +37% output tok/s**, driven by decode
`block_m 64→16` + CK bpreshuffle GEMM. See [catalog/kernel-registry.md](catalog/kernel-registry.md).

## What's inside

```
amd-kernel-opt/
├── index.md          OKF root index (progressive disclosure)
├── log.md            change history
├── methodology/      how to approach an optimization (bottleneck-first)
├── patterns/         12 reusable success techniques
├── anti-patterns/    4 verified negative results
├── cases/            18 per-kernel case studies (baseline → win → reverted attempts)
└── catalog/          kernel registry (91 kernels) + raw speedup CSVs
```

**The flow:** [registry](catalog/kernel-registry.md) (every kernel + speedup) →
[cases](cases/index.md) (deep evidence for the notable ones) →
[patterns](patterns/) / [anti-patterns](anti-patterns/) (the distilled, reusable lessons).

### Patterns at a glance
| pattern | bottleneck | typical gain |
|---|---|---|
| [Host-side HIP-graph replay](patterns/host-graph-replay.md) | launch | 1.2–2.05× |
| [Backend dispatch swap](patterns/backend-dispatch-swap.md) | memory/compute | 1.19–4.39× |
| [Per-shape kernel dispatch](patterns/per-shape-kernel-dispatch.md) | shape-dependent | 1.19–1.37× |
| [block_m to routing sparsity](patterns/block-m-routing-sparsity.md) | occupancy | 1.08–1.31× (decisive e2e) |
| [CK pipeline V3→V1 for occupancy](patterns/ck-pipeline-v1-occupancy.md) | occupancy | 1.08–1.31× |
| [Triton do_not_specialize launcher](patterns/triton-launcher-do-not-specialize.md) | launch | up to 2.05× |
| [Hoist K-loop-invariant math](patterns/hoist-kloop-invariant-math.md) | compute | 1.05–1.36× |
| [int4 load-once-unpack](patterns/int4-load-once-unpack.md) | L2 BW | up to 5.19× |
| [L2-locality pid remap](patterns/l2-locality-pid-remap.md) | memory | 1.05–1.52× |
| [Launch-config autotune](patterns/launch-config-autotune.md) | compute/occupancy | 1.05–1.63× |
| [Single-pass attention](patterns/single-pass-attention.md) | launch | 1.18–1.58× |
| [Empty (not zeros) output buffer](patterns/output-empty-not-zeros.md) | memory | small |

And four hard-won **[anti-patterns](anti-patterns/)**: non-temporal-load regression,
benchmark over-fit (incl. a retracted 17.39× harness bug), launch-bound body opts being
invisible, and numerics-gate violations.

## Scope & provenance

- **Hardware: MI300X only (for now).** Every optimization here was measured and validated on
  **AMD Instinct MI300X (gfx942 / CDNA3)**. The levers, numbers, and "at-ceiling" verdicts are
  specific to that part — they are *not* yet validated on MI325X, MI350X/MI355X (gfx950 / CDNA4),
  or any other GPU. Treat cross-generation transfer as unverified, especially anything touching
  FP8 encoding (gfx942 FNUZ vs gfx950 OCP), MFMA atoms, or LDS sizing.
- Built from the **documented** corpus only (per-attempt optimization reports with authored
  "what changed" notes). Deep case studies are limited to **verified speedup < 10×**.
- High-speedup (>10×) automated runs are **excluded** — they carry no "what changed", and very
  high numbers are suspected harness artifacts (one 17.39× was traced to a grid bug and retracted).
- Raw aggregated speedups are vendored in [`catalog/`](catalog/): `kernel_speedups.md`,
  `kernel_speedups_task_results.csv` (107 kernels), `kernel_speedups_llm_inference.csv` (91 LLM kernels).

## Notes for readers

- **Citations are intentionally dangling.** Each entry's `# Citations` section points to the
  original campaign source tree (e.g. `KernelForge/results/...`), which is *not* shipped here.
  OKF consumers tolerate broken citation links — the knowledge in each entry stands on its own.
- **Numbers vary by harness/regime.** The same kernel may show different speedups across
  measurement contexts; each entry states which regime it measured.

## License

Licensed under the [Apache License 2.0](LICENSE).

---
*Format: [OKF v0.1](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md). Hardware: AMD Instinct MI300X, gfx942 / CDNA3.*
