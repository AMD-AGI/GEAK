---
title: Kernel ecosystem references
kind: landscape
updated: 2026-10-16
---

# Kernel ecosystem references

These guides locate implementation and tooling options. They do not rank current
performance, establish local availability or prescribe new GEAK/GEAK-EVO features.
Use the installed revision and actual task to decide what to try.

| Guide | Use it for |
|---|---|
| [Libraries](multi_backend_libraries.md) | Existing operator implementations and integration surfaces |
| [Authoring languages](authoring_dsls.md) | Programming model and toolchain choices |
| [Kernel agents](ai_kernel_agents.md) | Search and evaluation research |
| [Autotuning](autotuning_frameworks.md) | Candidate sweeps and reusable tuning records |
| [AMD examples](amd_sota_2026.md) | Published implementation examples by operator |
| [Serving dispatch](serving_stack_registries.md) | How a selected implementation reaches a real workload |

Backend names can hide several implementations: aiter, for example, includes
assembly, CK, Triton and FlyDSL paths. Identify the selected callable and variant,
then measure it. A dated benchmark, registry badge or narrower architecture tag
cannot select the winner for an unmeasured shape.

## Sources

The linked guides retain relevant official documentation, repositories and papers.
