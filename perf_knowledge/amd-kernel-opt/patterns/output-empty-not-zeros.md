---
type: Optimization Pattern
title: Allocate output with empty, not zeros
description: Use torch.empty for an output buffer the kernel fully overwrites, skipping a needless device memset.
tags: [domain-any, bottleneck-memory, lever-host-side, no-rebuild, gfx942]
bottleneck: memory (spurious write)
lever_class: host-side
median_speedup: small (stacks into larger wins)
timestamp: 2026-06-22T00:00:00Z
---

# When to use
The kernel writes every element of its output tensor, but the host allocates it with
`torch.zeros` — paying a full-tensor memset the kernel immediately overwrites.

# Mechanism
Replace `torch.zeros(...)` with `torch.empty(...)` for fully-overwritten outputs. Trivial,
bit-exact when the kernel truly covers all elements.

# Evidence
- [MLA prefill](/cases/mla-prefill.md) — `torch.zeros → torch.empty` for the output buffer, stacked with launch-config tuning into **1.21×**.

# Caveats
- Only safe if the kernel writes **all** output elements (including any padded/masked
  tail) — otherwise you leak uninitialized memory. Verify against the correctness gate.

# Citations
1. spare_kernels/k04_fmha_prefill/reference_solution/OPT_NOTES.md
