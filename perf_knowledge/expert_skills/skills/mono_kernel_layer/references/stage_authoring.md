# Authoring one stage of a mono kernel

A **stage** is a slice of one decoder layer (e.g. QKV projection + RoPE, o_proj + TP all-reduce, router
top-k). A later fusion step inlines the stage's code into one persistent kernel together with the other
stages, so write it to be inlined. This page is the contract for the person or agent who writes, reviews
or times a stage; [acceptance_gates.md](acceptance_gates.md) is the matching contract for the fused layer.

The stage's own task statement (shipped with the task) names the files, the inputs and outputs, and the
bounds; this page states the same contract in general terms.

## Oracle (unchanged, never edited)

- The oracle files are integrity-pinned: `unittest.py`, `cases.py`, the stage case definitions,
  `meta.json`, the stage's task statement and the baseline overlay. So are the layer's reference
  implementation of the stages, its weights and its captured inputs, which the oracle reads by absolute
  path.
- Correctness is measured against a **float64 reference** at a bound derived from the arithmetic, not
  against the native op. Matching the native op's rounding is therefore not the goal; being within the
  bound is. Errors are reported as err / tol per output (pass means <= 1).
- The speed denominator is the stage's native framework op sequence.
- Run it from your workspace, under the GPU lock. Every stage runs on all ranks of the layer's TP
  group, so lease the whole group in one lease (see [acceptance_gates.md](acceptance_gates.md#running-the-oracle));
  the unittest starts the per-rank processes itself.
- Never run the unittest of the canonical task directory: the harness tests the `kernel_src/` that sits
  next to the `unittest.py` it runs, and the task directory's own `kernel_src/mk_stage.py` is a stub that
  raises.

## Shape of the code (hard gates)

These seven rules are the stage's hard gates. This repository ships no script for them: treat them as
review criteria and test what can be tested (the oracle, several workgroup counts, an emptied device
function).

1. **Device function + launcher.**
   - `kernel_src/stage_<id>.py` holds `def stage_<id>(ctx, *, tid, bid, n_blocks, ...)`: a plain Python
     function that emits IR inside a `@flyc.kernel` and takes raw i64 addresses of its inputs, outputs and
     weights, plus a thin `@flyc.kernel` + `@flyc.jit` launcher.
   - `kernel_src/mk_stage.py:run(layer, **inputs)` is the entry the unittest binds: it reads the weight
     addresses, allocates the outputs, launches, and returns the outputs in the order of the stage list.
   - When the stage runs inside a fixed layer layout (the last section of its task statement, "the
     skeleton this stage runs in", see [layer_structure.md](layer_structure.md#layouts)), that section binds the
     thread count, the block counts, the VGPR / LDS budget and the signature `stage_<id>(ctx, io, *, ...)`
     with its `io.wait` / `io.publish` calls. Hand-off data goes through `io.load` / `io.store`: these are
     sc1 accesses, because in the layer kernel a plain access of that data is not ordered by the waits.
     Everything this section binds (the thread count, the block counts, the VGPR / LDS budget, the
     signature with its `io.wait` / `io.publish` calls, and the `io.load` / `io.store` use for hand-off
     data) is checked from the ISA and the source.
2. **Grid-agnostic work split.** The device function must be correct for any `n_blocks` >= 1 and any
   subset of workgroup ids (`bid` in `[0, n_blocks)`), because fusion gives it a share of the CUs. Use
   grid-stride loops. Test it with several workgroup counts: the stage checks run it with 1 and 7.
3. **No host-side control flow per call.** No `.item()`, no host syncs, no allocations that depend on
   data. The launcher must be CUDA-graph capturable (the unittest replays it).
4. **Static widths.** Every width (decode row count) is a compile-time constant of its own build. The
   launcher's `func.__name__` must be unique per configuration, because FlyDSL keys its compile cache on
   the launcher source and closure scalars, not on helper modules.
5. **Cross-GPU or cross-workgroup data uses the runtime primitives only**
   ([runtime_primitives.md](runtime_primitives.md)). Author mode wipes `kernel_src/`, so copy the runtime
   that the task supplies into `kernel_src/mk_runtime/` unchanged (it is compared file by file against the
   original) and import it from there. Every wait carries a deadline. A hand-rolled spin loop fails
   review.
6. **Weights in place.** Read the framework's weight tensors as they are loaded. A one-time repacked copy
   is allowed only if it is created at first call, keyed by `data_ptr`, and its bytes are reported.
7. **No framework code.** On the candidate leg, calling the layer or any of its modules, or the native
   stage implementation the baseline leg uses, raises. Also rejected: framework imports and
   reference-implementation imports, torch used for arithmetic, and a device function whose emptied body
   still passes (the outputs must come from it).

## Read the native kernels first

When the task statement has a section on how the framework computes the stage (the list of native
sources), read those vLLM / SGLang / aiter / Triton sources before writing: how they lay the data out,
how they split the work over threads, waves and workgroups (a K split, a KV-sequence split and its
softmax combine, the reduction across ranks), the vector widths of their loads, where they use matrix
cores. Port that method to FlyDSL; a first kernel that keeps one thread per output row, or reads a byte
at a time, is far off the native time and each later round has to undo it. Read only: importing or
calling them is rejected.

## Performance view (benchmark role)

Time under CUDA-graph replay of the launcher, with a cache flush between samples (the unittest's
per-case lines already do this). Decode widths dominate, so report a secondary geomean over the decode
widths only. Gates for the verify role: graph-safe, the derived bound holds, and the shape rules 1–7
above.
