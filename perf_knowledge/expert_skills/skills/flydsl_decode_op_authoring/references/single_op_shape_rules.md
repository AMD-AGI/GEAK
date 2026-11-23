# Shape rules for a single-operator FlyDSL kernel

Rules for the *form* of the code, independent of which operator it computes. They were written for
operators that are later inlined into a persistent whole-layer kernel, and are restated here for the
general case: one operator, one launch, tested by an immutable unit test. Status of each statement:
*measured* (ratio given), *rule* (a gate enforced it on gfx942 with FlyDSL 0.4.x), or *derived* (my
generalisation, not separately tested).

**Which rules are universal.** Sections 1, 2, 4, 5, 6, 7 and 12 hold for any single-operator FlyDSL
kernel that a unit test replays under a CUDA graph. Sections 3, 8, 9 and 11 are *required* only when
the operator will later be inlined into a larger kernel, or launched on a grid it does not choose. For a
standalone launch that picks its own grid they are still the safest default, because they are what
makes a design pass at 1 and at 7 workgroups; but a deployed single-operator kernel that fixes its grid
is not wrong for doing so.

## 1. Read the native kernel's method before writing

*Measured.* First kernels written without reading the native implementation were far off the native
time, and every later round had to undo their structure:

| operator (first, naive kernel) | time versus the native op |
|---|---|
| input RMSNorm | about 8x slower |
| fused QKV projection | about 22x slower |
| output projection | about 18x slower |
| fused-add RMSNorm | 21x to 37x slower |
| decode attention at a 20k context | about 5400x slower |

The attention kernel used one thread per head and read the fp8 KV cache a byte at a time. Its own
roadmap already listed split-KV and matrix cores; there was simply no optimize round left to add them.

What to read in the native source (vLLM / SGLang / aiter / Triton / the FlyDSL kernel library), and
copy as *method*, never as an import:

- the data layout it assumes (row stride, padding, tile shuffles, where scales live);
- how it splits the work over threads, waves and workgroups: a K split, a KV-sequence split and its
  softmax combine, a per-expert or per-head grid;
- the vector width of its loads and whether a byte is ever read twice;
- where it uses matrix cores and where it does not;
- its rounding points (these belong to `native_decode_numerics`; note them, do not re-derive them).

A kernel that keeps one thread per output row, or reads a byte at a time, is the signature of having
skipped this step.

## 2. Device function plus thin launcher

*Rule.* Split the kernel into:

- a **device function**: a plain Python function that emits IR inside a `@flyc.kernel`, taking the thread
  id, the workgroup id, the workgroup count and the raw 64-bit addresses of its inputs, outputs and
  weights;
- a **launcher**: a thin `@flyc.kernel` + `@flyc.jit` pair that calls it;
- a **host entry** with the signature the unit test binds: it reads the weight addresses, allocates the
  outputs (allocation only), launches, and returns the outputs in the order the test expects. An
  in-place output (a fused add that updates the residual) is written back into the input tensor.

Keeping the device function free of launch details is what lets the same code run under any grid and
later be inlined next to other operators.

## 3. Grid-agnostic work split

*Rule.* The device function must be correct for any workgroup count `n_blocks >= 1`. Number the tasks
`0 .. T-1` and let workgroup `b` run tasks `b, b + n_blocks, ...` (a grid-stride loop). Consequences:

- a task **owns** the output tiles it writes; no two tasks write the same element (or they write
  identical bytes, as the repeated KV write in [`rope_kv_write.md`](rope_kv_write.md));
- a task may wait only for tasks of a **lower index**, or not wait at all and use a last-arrival
  protocol (see [`flydsl_idioms_and_resources.md`](flydsl_idioms_and_resources.md#cross-workgroup-patterns));
- test it: give the launcher a workgroup-count override, and run the correctness command at 1, at 7 and
  at the full grid. With fewer workgroups than tasks the tasks of one row run one after another, which
  exercises every arrival order.

A kernel that is right only at the full grid has an ownership or ordering bug that will show up later
as a rare wrong answer.

## 4. No host control flow per call

*Rule.* The launcher must be CUDA-graph capturable, because the unit test replays it:

- no `.item()`, `.tolist()`, host synchronisation or data-dependent allocation on the call path;
- no arithmetic or conversion on the host (`.to()`, `.float()`, `.clone()`, `.contiguous()`, tensor
  maths). If a tensor does not have the dtype, shape and strides the kernel was built for, raise; do not
  convert;
- lengths and other metadata that the host needs are mirrored at construction time, not read back from
  the device per call;
- a workspace that must start zeroed is allocated and zeroed on the **first** call, which therefore has
  to run eagerly (outside graph capture). Raise if the first call happens inside a capture.

## 5. Static widths, unique launcher name

*Rule.* Each decode width (row count) is a compile-time constant of its own build. FlyDSL keys its
compile cache on the launcher's source and closure scalars, not on helper modules or module globals, so:

- give the launcher a name that encodes the configuration (width, workgroup count, every variant
  switch);
- a profiling or timeline variant gets its own name;
- after editing a helper module, change the name or bypass the cache, otherwise the old kernel runs and
  the edit looks ineffective.

Details and the trace-time-global trap: [`flydsl_idioms_and_resources.md`](flydsl_idioms_and_resources.md#jit-cache-and-trace-time-pitfalls).

## 6. Weights in place

*Rule.* Read the framework's weight tensors as they are loaded: element format as loaded (on gfx942 an
fp8 checkpoint may have been normalised to the FNUZ format at load time), strides from the tensor itself
(rows may be padded; a scale tensor may be column-major; a weight may be tile-shuffled in place while
still reporting dense strides). A one-time repacked copy is acceptable only if it is created at first
call, cached by the weight's data pointer, and its size is reported in the notes, since it is extra
device memory that a serving deployment pays for.

## 7. The outputs must come from the device function

*Rule.* Do not call the layer, its modules or the native op on the candidate path; do not compute with
torch on the host. A useful self-check (*derived* from the gate that enforced this): replace the device
function's body with nothing and confirm that the unit test now fails. If it still passes, the outputs
are coming from somewhere else.

## 8. Cross-workgroup data

*Rule.* Data that one workgroup writes and another reads inside the same launch goes through
device-scope coherent (SC1) stores and loads, is drained before the flag or counter that announces it,
and every wait has a deadline that traps. A hand-rolled unbounded spin loop is a hang waiting to
happen: a flag that never arrives is a bug, not a slow peer.

## 9. Decide the output unit before the task split

*Rule.* Fix the unit in which outputs are produced first: 16 columns of a `[rows, hidden]` output, one
head of a `[rows, heads, dim]` output, one row of a per-row result. Then make each task own whole units.
When the task map and the output unit disagree, a unit ends up written twice or not at all at some
workgroup count.

## 10. Known failure modes of operator authoring

Observed in an internal mono-kernel generation workflow (not public); each cost at least one round.

| failure | how it showed | what to do |
|---|---|---|
| the test did not test the edited kernel | another copy of the candidate (the native op, or a stub) was picked up, so the result was always PASS or always an error | run the oracle with the correctness command of your role file (author) or exactly as the lane's COMMANDMENT file says once it exists (optimize rounds), and confirm once that breaking your kernel makes it fail |
| the candidate called the framework op | PASS with a speedup of about 1 | rule 7 |
| a refinement round handed back byte-identical code | a full round spent re-measuring the same kernel | diff against the previous attempt before submitting |
| resources not checked before handing in | a kernel rejected twice for a small spill (non-zero scratch in the ISA) | read the ISA (registers per wave, scratch, LDS) before every hand-in |
| a busy or contended GPU read as a wrong kernel | timeouts on a lock or an IPC handle counted as a failed attempt | treat "no verdict" as no verdict; hold every GPU of a multi-rank task in one lease for the whole command |
| a profiler run that never returned | a hung profile, in one case with an unbounded log | bound the profile's run time and output size |
| a correct seed with no optimize round | the lane only measured the slow seed | ask for at least one optimize round on the operator that dominates |

## 11. Why a library kernel cannot always be dropped in

A FlyDSL library kernel is a standalone launch with its own grid, thread count and LDS use. As the seed
of a single-operator task it is fine if the oracle passes with it. It stops being usable as-is when the
operator has to live inside a larger kernel, because then:

- thread count and workgroup count are given from outside, so the work split must be grid-agnostic;
- registers and LDS have to fit a budget with zero scratch, known before the build rather than
  discovered as a spill;
- waits and releases follow the host kernel's protocol;
- a weight preshuffle may run only once, keyed by data pointer, with its bytes accounted for;
- the rounding points have to be switches read from the native source, not whatever the library
  kernel happens to do.

Writing every operator of a layer from scratch took more than half of the wall-clock of one full
generation run, with correctness as the only goal. That is the cost a per-operator design note removes.

## 12. How the op is timed

Time the launcher under CUDA-graph replay with a cache flush between samples, at every decode width the
task lists; decode widths dominate serving time, so report a geometric mean over them. The denominator
is the native op sequence as the frozen unit test runs it, never an implementation of your own.
