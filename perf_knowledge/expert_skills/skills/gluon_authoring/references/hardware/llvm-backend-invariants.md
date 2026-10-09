# LLVM AMDGPU backend decision gates

This reference applies when a current profile attributes a remaining gap to LLVM lowering,
scheduling, or register allocation. It is a decision method, not evidence for any kernel.
Start with the active `stage-context`, then use the runtime tool's `--describe` and `--help` output
to obtain current probe interfaces.

This page holds only the two **gates** and the acceptance checklist. The co-design tier itself is
defined once in `../tile-programming/compiler-contract.md` (what upstream Triton 3.8.0 gives you,
Scenario B) and `../tile-programming/llvm-codesign-handbook.md` (the pass-plugin workflow). Check the
upstream per-compile routes before any co-design: `llvm_fn_attrs` (LLVM function attributes per
compile, 3.8.0 only — `../tile-programming/llvm-fn-attrs.md`) and the stock coexec scheduler
strategy (`TRITON_HIP_USE_COEXEC_SCHEDULER`) — off by default on gfx950/gfx942 (upstream 3.8.0 enables it automatically only on gfx1250); opt in with `TRITON_HIP_USE_COEXEC_SCHEDULER=1` (process-wide, applies at `num_warps <= 4`) or per compile with `llvm_fn_attrs=[["amdgpu-sched-strategy","coexec"]]` (any `num_warps`), accepted on an assembly diff. Env names
that exist only on a fork are listed, labelled, in `../tile-programming/non-upstream-reserve.md`.

## Ownership boundary

| Layer | Owner | Required proof |
| --- | --- | --- |
| DSL source and layout | kernel author | source/IR change and current timing |
| Front-end lowering | DSL/runtime | matching IR observation |
| LLVM scheduling / register allocation | compiler co-design owner | compiler/assembly change and current timing |
| Device code | backend/hardware | assembly and resource facts |

An upper-layer request is not evidence that a lower layer honored it. Verify the decision at the
layer where it becomes observable.

## Gates before co-design

1. **Allocation versus live set.** Probe the requested register placement and compare its spill,
   live-set, and occupancy consequences with the active architecture facts. If the change only
   moves pressure between register files or increases spilling, classify the proposal as
   live-set-limited and return to kernel structure/layout work.
2. **Schedule versus dependency chain.** Use a safe, explicitly marked floor probe to determine
   whether independent work can exist in the alleged idle region. If operands are not ready, the
   dependency chain is the constraint; a scheduler directive is not a remedy.

Both gates must record the target identity, changed condition, observed IR/assembly signal, timing
boundary, and falsifying result. A failed gate is a scoped conclusion, not permission to continue
building a compiler patch.

## Co-design verification

Before accepting a compiler-level change:

1. Prove the intended pass/toggle was reached.
2. Compare the emitted hot loop or resource record, not only a summary metric.
3. Run a positive control whose effect should be visible in the emitted code.
4. Re-measure the unchanged production boundary and verify correctness.

If an existing stage context exposes no supported compiler-control route, leave the proposal
unavailable and route it to the applicable source-level or DSL-level stage. Do not manufacture a
new runtime interface from this document.
