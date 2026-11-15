# Acceptance gates for a persistent layer kernel

The contract for optimizing and accepting a fused layer kernel: how to run the oracle, what its numbers
mean, the seven hard gates a candidate must pass, and how to check the gates that the unittest does not
cover. The stage-level counterpart is [stage_authoring.md](stage_authoring.md).

## Task shape

- The target is the candidate's layer kernel (conventionally `kernel_src/mk_<kind>.py`) plus a thin
  adapter (`kernel_src/mk_adapter.py`): **one persistent FlyDSL kernel per rank that computes a whole
  decoder layer for one decode step**, including the in-kernel TP all-reduce.
- The frozen baseline is the framework's own native layer. The task uses a two-leg overlay: the baseline
  leg runs the stock class, the candidate leg rebinds it to `Mk<LayerClass>`.
- The seed you start from already passed the launch and correctness checks (it is correct and launches
  once per layer); it may be slower than native.

## Running the oracle

- Run it from the task workspace, with every GPU of the TP group (the ids your run was allocated) leased
  in one lease: `cd $WORKSPACE && bash $SKILL_DIR/scripts/gpu_lock.sh group:<ids> -- python3 unittest.py
  [--correctness|--benchmark]` (`$SKILL_DIR` is the `kernel_workflow/` directory). The unittest starts
  one process per rank itself and reports rank-agreed results. Never set `HIP_VISIBLE_DEVICES` or
  `CUDA_VISIBLE_DEVICES` yourself (the one exception is a compile-only run that sets them to -1, see
  [FlyDSL idioms](flydsl_idioms_and_pitfalls.md#flydsl-idioms)), and never run a rank by hand on a GPU you
  have not leased.
- For a multi-rank probe of your own (rocprofv3, a micro-benchmark, a debugging script), use the same
  whole-group lease: `bash $SKILL_DIR/scripts/gpu_lock.sh group:<ids> -- <cmd>`. The command sees the
  leased GPUs in `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` and must start one process per rank
  itself, passing both variables on unchanged (on ROCm a process whose two values differ is aborted when
  HIP starts). A command running under a lease cannot request another lease.
- The oracle files (`unittest.py`, `cases.py`, `meta.json`, `reference_io/`, the vendored harness) are
  integrity-pinned. The unittest refuses to run once any of them changed.

## What the numbers mean

- Per-case time is device time under **CUDA-graph replay** inside the framework's `graph_capture()`, with
  an L2/MALL flush and a cross-rank barrier before each sample. It is the median over samples of the
  **slowest rank**. That is the serving number; there is no separate eager number to optimize.
- `GEAK_WEIGHTED_SPEEDUP` (when a workload is set) weights the decode widths by serving frequency. It is
  the primary target; the unweighted geomean speedup is secondary.

## Hard gates

1. **Correctness passes** (the layer unittest prints `CORRECTNESS: PASS`). That covers eager goldens,
   random parity against the native layer, graph replay across widths, sequences and output independence,
   on every rank. Every comparison is against the output the native leg produced for the same check. The
   tolerance in `meta.json` (`meta.tol`) is calibrated against measured native error and negative
   controls; never loosen it.
2. **One launch per layer per rank** on the eligible path, and none of the native layer's kernels. A
   candidate that splits back into several kernels, or falls back to the native path for eligible
   widths, is not a mono kernel.
3. **The outputs come from the kernel.** With every `@flyc.kernel` body of `mk_<kind>.py` emptied, the
   correctness check must fail. The adapter computes nothing itself: it calls no `self.<submodule>(...)`.
4. **Every cross-workgroup or cross-GPU wait goes through `mk_runtime` with a deadline.** No bare spin
   loops. `kernel_src/mk_runtime/` stays identical to the reference copy of the runtime that the task
   supplies (a runtime bug is reported, not patched in the candidate).
5. **Launch epoch from the device** (`launch_begin`/`launch_end`). Nothing that changes per step is
   passed from the host, so graph replay stays correct.
6. **The decision on the native path is identical on every rank**, so no rank waits on a peer that took
   the other path.
7. **Memory.** No per-call allocations other than the output tensors. Any weight repack is one-time,
   keyed by `data_ptr`, and its bytes are reported.

Stage boundaries are not a gate: merging stages (a norm into the GEMM before it, two GEMMs sharing an
activation read) is legal. The per-stage test module `kernel_src/mk_stages.py` may go away; while it
exists, the final check re-runs the per-stage tests through it and only records the result.

## Checking gates 2, 3 and 4 outside the unittest

The unittest does not cover gate 2 (launch census), gate 3 (hollowed kernel) or the runtime part of
gate 4 (runtime unmodified). Bare spin loops, the rest of gate 4, are caught later by a fault-injection
test: stall one rank, and the run must end in the deadline error, not a hang. This repository ships no
script for these checks: perform them by hand, or script them for the task at hand. After a candidate
passes the unittest, check these on a clean copy of the candidate before accepting it:

- **Launch census (gate 2).** Count the kernels one decode step launches per layer per rank: one
  mono-kernel launch, none of the native layer's kernels.
- **Hollowed kernel (gate 3).** Empty every `@flyc.kernel` body: the correctness check must fail.
- **Runtime identical (gate 4).** `kernel_src/mk_runtime/` matches the task's reference copy of the
  runtime, file by file.
- **Static pass.** CPU-only static checks (no GPU, seconds) can be run before handing a patch over.

A candidate that fails any of these is marked failed, with the problems listed in your notes, even when
the speedup improved.

After the optimization lane, the final state is re-checked from scratch: the full layer unittest re-run
in a fresh copy (the reported speedup is taken from this run), the hollowed kernel, the launch census and
the per-stage record, and the check walks back over the round commits. A state that fails is never
installed, and the speedup that counts is the one measured by that final check, not the lane's own
number. A round built on a violating candidate is wasted.

## Where to plan from

Plan from the symptom → lever table in [../skill.md](../skill.md#choosing-a-lever-by-symptom): measure first
([measuring.md](measuring.md)), then pick the lever whose symptom you see. The levers are directions; the
oracle and the gates decide.

## Sources

- Lease behaviour quoted above (`group:<ids>`, both variables exported, no nested lease):
  `kernel_workflow/scripts/gpu_lock.sh` and `kernel_workflow/scripts/gpu_lease.py` in this repository.
