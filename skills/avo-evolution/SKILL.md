---
name: avo-evolution
description: >
  Use during an AVO-style continuous kernel-evolution run (geak-avo): a single
  variation step that reads the committed lineage, runs an
  edit-evaluate-diagnose loop, commits only verified improvements via
  save_and_test, and self-pivots when a direction stalls. Load at the START of
  every variation step. NOT for one-shot GEAK --num-parallel round-loop runs.
---

# AVO Evolution — Variation Operator Playbook

Distilled from the NVIDIA AVO paper (*Agentic Variation Operators for
Autonomous Evolutionary Search*, arXiv:2603.24517). In GEAK, **one
`OptimizationAgent` run = one variation step**; an outer controller owns the
lineage, the budget, and the supervisor. Your job is the variation step.

## When to use

- A `geak-avo` continuous-evolution variation step.
- The task body contains an `## AVO Variation Step Contract` block.
- NOT for standard `--num-parallel` round-loop runs (use `strategy_manager` alone).

## Task → doc routing

| Situation | Read |
|-----------|------|
| First actions in a variation step | `docs/variation_step_contract.md` |
| Choosing WHICH optimization to apply (by kernel class) | `docs/optimization_playbook.md` |
| Reading / reasoning about prior commits | `docs/lineage_usage.md` |
| 3+ attempts with no improvement on one direction | `docs/stagnation_self_check.md` |
| Adapting a kernel to a new variant (e.g. MHA→GQA) | `docs/kernel_adaptation.md` |

## Non-negotiable rules (anti-cheat)

- Do NOT "fix" or "optimize" by simplifying away the kernel's structure (tiling,
  MMA, pipeline) or by falling back to PyTorch / cuBLAS / cuDNN — unless that was
  the reference's intent. A trivial ≈1.0x rewrite is rejected by the commit gate.
- Do NOT change precision or input/output shapes to inflate a number.
- Classify the kernel (GEMM / elementwise / reduction) and apply only matching
  optimizations — see `docs/optimization_playbook.md`.
- Honor the **Optimization stage** marker in the task body: structural-first,
  profiling-guided later (delayed profiling avoids premature local optima).

## Core workflow (every variation step)

1. Read the lineage summary in the task body → note `best` id and its speedup.
2. Read the active direction (assigned by the supervisor) in the contract block.
3. `strategy_manager show` → see what is already successful / failed / pending.
4. Classify the kernel and pick ONE matching optimization (`docs/optimization_playbook.md`).
5. Confirm the bottleneck only when the stage is PROFILING-GUIDED (`profile.json` / `profile_kernel`).
6. Implement **ONE** focused change aligned with the direction → `save_and_test`.
7. If it is a correct, verified improvement → submit (the controller commits it).
8. If 3+ attempts on this direction fail to improve → `strategy_manager mark`
   it `failed` with a reason, then `strategy_manager next`.
9. Never declare the overall optimization complete — only this step ends.

## Commit gate (what the controller enforces — design for it)

A version enters the lineage ONLY if:

- correctness passes, AND
- the independently-verified benchmark speedup is ≥ the running best
  (within a tiny epsilon).

Self-reported or unverified speedups do not count. Always go through
`save_and_test`; never hand-roll a benchmark.

## Integration with GEAK tools

| Tool | AVO usage |
|------|-----------|
| `save_and_test` | the scoring function `f`; call after every meaningful edit |
| `strategy_manager` | the direction queue; mark `failed`/`skipped`, request `next` |
| `query` / `optimize` (RAG) | the knowledge base `K`; consult before big changes |
| `profile_kernel` | bottleneck diagnosis between attempts |

## Paper-backed expectations (§4.4)

- Improvements arrive in **discrete jumps**, not smooth gradients.
- Early steps: coarse architectural changes. Late steps: micro-arch tuning
  (registers, fences, pipeline overlap).
- Many internal attempts per commit is normal — failed attempts are useful
  signal, not wasted effort. Keep working; do not stop or lower test rigor.
