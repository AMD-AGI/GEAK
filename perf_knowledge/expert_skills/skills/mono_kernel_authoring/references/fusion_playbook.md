# Fusion playbook: direct route, stages route, failure categories

Open this for steps 5 and 6 of the procedure in [`../skill.md`](../skill.md).

**Status.** Both routes were implemented and exercised in an internal mono-kernel generation workflow (not
public): the direct route with vLLM on two models, the stages route with vLLM and SGLang on one MoE layer kind,
the per-stage fallback once. The scripts that prepared workspaces, ran the probe and judged acceptance are not in
this repository; what they did is described here so it can be done by hand.

## 1. Evidence (measured; not controlled comparisons)

| route | case | contract done → accepted seed | seed against the native layer |
|---|---|---|---|
| direct | three layer kinds of a model with mixed residual streams and sparse latent attention; four ranks, width 1; attention written from scratch | about 1.5 to 3.3 hours of authoring per kind, no fix round | 3.1×, 1.9× and 2.3× faster; one launch where native ran 52 to 91 kernels |
| direct | a fourth layer kind of that model | not accepted | all values passed; 2 of 767 sequence checks differed by one byte in exact KV-write comparisons (the summation order of a native fp32 GEMM was not reproduced) |
| direct | an MoE layer of MiMo-V2.6-Flash; four ranks, width 1; all eight stages started from tested op code, and the layer kernel was written on a ready-made layout mechanism (hand-off and scheduling code) | under two hours; first correct kernel within ten minutes | 1.08× faster |
| stages | the same MoE layer, older tooling version | about six hours | about 80× slower (stages were only made correct) |
| stages | first attempt on that layer kind | about seven hours, not accepted | fusion failed, see 4.3 |
| direct, then stages for failing stages | the same layer with one stage deliberately broken | about three hours | the probe named the stage; six stages reused, two re-authored; accepted |

The rows differ in tooling version, stage count (seven or eight), authoring model and stage budget.

## 2. Rules shared by both routes

- Work in a copy of the task under version control. Edit only `kernel_src/`. Everything else stays byte-identical
  to the task; scratch scripts and logs live outside the copy. Commit every milestone; a fix continues from the
  last commit. If the harness looks wrong, stop and report it. Do not work around it.
- Products in `kernel_src/`: the adapter; the layer kernel (one kernel function and a builder that returns one
  launcher per decode width, with a launcher name unique per configuration); one file per stage with its
  device function and a thin standalone launcher; the synchronization runtime, unmodified.
- **Adapter.** A subclass of the layer class that overrides only `forward`, with the native signature. The
  inherited constructor means the weights are the framework's own tensors, read in place.
  - Take the kernel path only when this call is eligible: width in the feasible set, decode only, no refusal
    condition. Otherwise return the native `forward`.
  - Decide from what every rank shares (shape, dtype, configuration), never from data.
  - Create lazy state (peer buffers, weight views, compiled launchers) on the first eligible call, outside any
    graph capture.
  - Call nothing on `self` but the adapter's own methods and tensor metadata. A call such as `self.mlp(...)` is
    native compute hidden in the adapter.
  - Read the attention metadata from the framework's forward context and pass the KV cache, slot mapping, block
    table and sequence lengths to the kernel on every call. The KV write happens in the kernel, at the
    framework's slot and in its format.
  - The rebound class covers every instance of the layer class. For instances of another layer kind that shares
    the class (kinds that differ only by layer index) the adapter must return the native `forward`.

## 3. Direct route: one author writes the whole layer

### 3.1 Stage-boundary convention (from the first line)

- **One device function per stage of the contract**, defined exactly once, in its own file, with its own thin
  launcher. Its parameters are the stage's ports plus what it needs (weights, scratch, ids).
- The layer kernel only schedules, hands off and synchronizes. Inline a stage if you like, but **never merge two
  stages into one function**: the probe, the review and a later per-stage fallback all need the boundary.
- **State a stage writes is one of its outputs.** The KV write of the current token (and any recurrent state)
  belongs to the stage the contract names, every step.
- A stage that starts from tested op code keeps that body; change its hand-offs and scheduling, and write down
  every change to the body itself.

### 3.2 Milestones

| milestone | content | self-check before the commit |
|---|---|---|
| **correct** | one kernel runs the whole layer; a simple schedule is fine (every block does each stage in turn, a grid barrier between stages); the all-reduce is in the kernel; the adapter is wired | the layer oracle's correctness passes on every rank; the probe passes single-step and multi-step |
| **resources** | ISA scratch (register spill) 0, VGPR ≤ 512, LDS ≤ 64 KiB, every wait bounded through the runtime. **A failure on its own**: a kernel with scratch above 0 is not a seed, even when the oracle passes and it is faster than native | read spill, VGPR and LDS from the ISA of the correctness run ([how](../../../../languages/flydsl/authoring_optimization.md)); the static review criteria |
| **seed** | per-stage times inside the kernel; the largest stage first (typically MoE and long-context attention); re-run the "correct" self-check after every change | the oracle's benchmark; stop at the time budget with the last commit that passed both earlier milestones |

- Resources come before speed: a seed that fails resources or the static rules is not a seed.
- An intermediate with one launch per stage is allowed while getting the numbers right. Only the one-launch
  kernel is accepted.
- Speed is not a pass criterion here. A correct seed slower than native is a valid result.

### 3.3 The probe build

A separate build of the layer kernel, with its own kernel and launcher name, never the installed launcher.

- After the call it exports **every input and output port of every stage** (hand-off buffers; layer inputs and
  outputs under each stage that reads or writes them).
- It runs with a requested block count (1, 7, all) and reports the count it actually used.
- It reports the error word with the site table and, after a timeout, the tiles that were never published.
- KV cache: before the call, overwrite the current token's rows with NaN (captured pages already hold the native
  write, so otherwise a missing write is invisible). After the call decode the rows and compare them with the
  reference write. Bytes outside the current token's page must not change.

### 3.4 Two references: the local one judges, the chain one reports

| reference | computed from | use |
|---|---|---|
| **local** (verdict) | the stage's float64 reference on the inputs **the kernel actually gave that stage** | the first stage above its own bound is the faulty one; a routing selection is recomputed from the kernel's own logits |
| **chain** (report) | the float64 chain from the layer input, each output rounded to its port dtype before the next stage | shows how the error grows; whether the layer is within tolerance stays the layer oracle's call |

Why: an upstream difference inside its bound can be amplified by a softmax or flip a top-k downstream. Judged on
the chain, the downstream stage would be blamed. A stage input that no port carries means the boundary disagrees
with the contract: fix the contract, do not guess.

### 3.5 Steps, block counts, repeats

- **Multi-step**: about four consecutive decode steps with the kernel's own state carried over, once from an
  empty cache and once from a captured one. Local within bounds at every step while the chain grows: error
  accumulation, not a stage bug. It is a failure class only when the layer oracle fails; while the oracle still
  passes, report the growing chain as a warning. Local above its bound from some step on: that stage at that
  step (for example it read state that was never written).
- **Block counts 1, 7, all**: wrong at every count is numerics or indexing; right with 1 and wrong with more is a
  hand-off or synchronization error.
- **Repeats** (about three per configuration): results that differ and are wrong mean a race. Results that
  differ within the stage's bound are a warning. *(Measured: a stage accumulating with atomic adds differed by
  one ulp in 1 of 4096 outputs between repeats.)*
- The probe build changes the timing, so a race can vanish under it. The probe only localizes. If the oracle
  fails while the probe passes, report "only in the production build, suspected race".

### 3.6 How the probe was verified

*(Exercised: five faults injected into accepted kernels; all five reported as expected.)*

| injected | expected report |
|---|---|
| a stage output perturbed beyond its bound | that stage, stage numerics |
| an upstream perturbation inside its bound that flips a downstream top-k | no failure; the chain report shows the difference |
| the KV write removed | the attention stage, state write (multi-step only) |
| one port's wait removed | that stage, hand-off / race |
| one port's publish removed | that stage, hang, with the site translated correctly |

## 4. Stages route: author every stage alone, then fuse

### 4.1 Rules for a stage written alone

1. A device function plus a thin launcher, tested against the float64 reference at the stage's derived bound.
   Being within the bound is the test for a stage's values, not matching the native op's rounding. Exception:
   what is written to persistent state (the KV write) is compared byte for byte, so the stages that produce it
   must reproduce the deployed path's rounding points and summation order.
2. **Grid-agnostic**: correct for any block count ≥ 1 and any subset of block ids; use grid-stride loops; test at
   1 and 7 blocks.
3. No host-side control flow per call, no host sync, no data-dependent allocation: the launcher must be
   capturable in a graph.
4. Every width is a compile-time constant of its own build.
5. Data that crosses blocks or GPUs goes through the synchronization runtime only; every wait has a deadline.
6. Weights are read in place. A one-time repacked copy is allowed only if it is created at the first call,
   keyed by the weight's address, and its bytes are reported.
7. No framework code on the candidate side, and the emptied device function must fail the stage test.

Read the native kernels of the stage first: data layout, the split over threads, waves and blocks, the vector
width of the loads, where matrix cores are used. A first kernel that keeps one thread per output row or reads a
byte at a time is far from the native time, and every later round has to undo it.

### 4.2 Two fusion milestones

1. **Chained.** The adapter calls each stage's launcher in order: one launch per stage, hand-offs through device
   tensors. This settles everything unrelated to persistence: routing and arguments, attention metadata, the KV
   write, stage interfaces and dtypes, each rank's shard.
2. **Persistent.** One kernel replaces the launch sequence; the adapter changes only at the launch.

Both are committed, so when they disagree you compare their intermediate tensors. The layer kernel imports each
stage's device function and never defines another of the same name: the stage tests then cover the code the
fused kernel runs. Only the persistent kernel is accepted; a chained result has as many launches as native.

### 4.3 What failed in the exercised run

1. The workspace held a stale copy of the runtime; no stage could be imported.
2. The attention stage read the KV cache and never wrote the current token. Its single-step stage test passed;
   the layer failed on multi-step sequences at about 14× the layer tolerance.
3. With the write added, sequences from an empty cache reached about 1.2× the layer tolerance. The QKV stage
   agreed with its float64 reference but not bit for bit with the deployed GEMM path, and the difference
   accumulated through the KV cache. This is error accumulation: it could only be reported.

### 4.4 Direct first, stages for what fails

Run the direct route. If acceptance still fails after the fix rounds, hand only the stages the probe names to
separate authoring and reuse the rest from the layer kernel's per-stage files. Each reused stage must first pass
its own stage test; one that fails it, or is not in its own file with its own launcher, is re-authored too. A
failure that names no stage (launch count, hollow) has no fallback. Error accumulation and a hollow failure are
not fixed on either route.

## 5. Failure categories

| category | recognize it by | first thing to try |
|---|---|---|
| oracle / build | a pinned file changed; a new Python file outside `kernel_src/`; a build or import error; a runtime copy that differs from the shared one | fix this first: restore the pinned files, re-vendor the runtime |
| resources | ISA scratch above 0, VGPR above 512 or LDS above 64 KiB. **Fails on its own**, whatever the oracle and the speed say | the stage with the largest use: the resource notes of the op code it started from, or smaller tiles |
| hang | a deadline expired (rank, stage, port) or the outer wall clock ran out | all waits share one deadline, so the error word may name a downstream waiter. **Look first at the producer stage of the most upstream port whose tiles do not carry this launch's tag**, not at the stage that reads it: does it publish under every block count and for every tile? |
| missing probe port | the probe build did not export a port or did not report its block count | export it |
| stage numerics | a stage is wrong on its own inputs at every block count | only that stage: its numerics rules against the native source (skill `native_decode_numerics`) |
| hand-off / race | right with 1 block and wrong with more; repeats differ and are wrong; the oracle fails while the probe passes | the stage's input waits and the upstream publish order: read early, published early, two writers of one tile |
| state write | the KV write is missing, misplaced or wrong; a stage is right at step 0 and wrong later | where and when the state is written: slot, format, scale, every step |
| error accumulation | every stage within its bound on its own inputs **and** the chained error grows over the steps. **A failure class only when the layer oracle fails; while the oracle still passes it is a warning to report** | the stage's numerics path differs from the deployed one: move it to the deployed path's rounding points (skill `native_decode_numerics`). A fix round alone cannot close it: report it; loosen nothing |
| launch count | not exactly one kernel per replay, or a native kernel in the trace | remove the extra launch or the native call |
| hollow | the oracle still passes with every kernel body emptied | the tests miss a part or the adapter falls back: report it; never change a test |
| contract | a stage reference cannot run on probe data | the boundary disagrees with the contract: report it and fix the contract |

Give the category first, then the first failing stage, then the evidence. Fix the cause, never the check. If
several apply, report the first of: oracle or build, the probe's class (hang, missing port, stage numerics,
hand-off / race, state write, contract), error accumulation or race with a failing oracle, resources, launch
count, hollow.

## 6. Symptom heuristics *(examples from an internal planning note; not distilled from real repairs, not validated)*

| symptom | likely cause |
|---|---|
| some cases slightly over tolerance | a race, or a stale tile was read |
| every case wrong | a layout or index error |
| wrong only from the second replay on | a flag that is not reset, or a tag collision |
| wrong at one width only | slot capacity or grid size mismatch |

- To bisect, fuse only the first `k` stages, dump at the hand-off and compare with the reference intermediate.
- Register spill after fusion: the options are splitting the stage, less unrolling, or another wave division.
