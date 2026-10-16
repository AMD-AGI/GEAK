---
title: MI350X and MI355X clocks and power
kind: hardware
gens: [gfx950]
dtypes: []
regimes: [both]
updated: 2026-10-16
---

# MI350X and MI355X clocks and power

| Published product specification | MI350X | MI355X |
|---|---:|---:|
| Compute units | 256 | 256 |
| Peak engine clock | 2200 MHz | 2400 MHz |
| Typical board power | 1000 W | 1400 W |

Use the [canonical product YAML](../../../e2e_workflow/knowledge/analysis_skills/roofline/peaks.md)
and [peak table](peak_tables.md). Equal CU counts do not imply equal published
per-GPU throughput. These clocks and power ratings do not establish the clock or
power drawn during a particular kernel.

Record clock, power, temperature and throttling telemetry during the relevant
measurement, using the installed `amd-smi` interface. Preserve warmup, sampling,
partition mode and timing scope. Sparse device-wide samples do not establish each
XCD's clock during a short dispatch. Compare repeated unprofiled latency under
matching conditions before assigning a gain to clocks or cooling.

HBM pin bandwidth is a separate specification from engine clock. Dependencies,
address generation, cache behavior and scheduling can still make a memory-heavy
kernel sensitive to engine clocks. Neither higher power nor fewer modeled bytes
alone guarantees a speedup.

## Sources

- [AMD MI350X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)
- [AMD MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)
- [AMD SMI documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
