# Review criteria: what must hold before the kernel is accepted

Open this for steps 7 and 8 of the procedure in [`../skill.md`](../skill.md).

**Status.**

- **Acceptance criteria** (section 2): implemented in an internal mono-kernel generation workflow (not public)
  and exercised on accepted kernels of two models and on rejected kernels of one. Criteria 1, 2, 4–8 and the
  import, host-arithmetic and adapter-call part of 3 were run by a script; the wait and per-step-value rules in 3
  and criteria 9 and 10 were written requirements, checked by reading. The script is not in this repository;
  every criterion is written as something to do by hand.
- **Robustness checklist** (section 3): a first version is implemented, in which an agent writes its own drivers
  and reports its own results; no run of it on a real kernel is recorded. The redesigned suite (script-run
  verdicts, a multi-instance check, host-only fault injection, re-running the checklist after a repair, a speed
  record before and after) was **designed only**. Repair in a copy, with the acceptance check before the repaired
  kernel replaces the accepted one, is implemented and was tested with stubs only.
- **Framework delivery checks**: first version only; see skill `apply_mono_kernel_to_framework`.

## 1. Why nothing here may rest on the author's word

- The stub adapter of a fresh task returns the native `forward`. It passes the layer oracle on every rank at a
  speedup of about 1.0. "The oracle passes" therefore does not show that a kernel computes anything.
- Falling back to native on eligible calls is the easiest way to be correct, and timing noise then looks like
  progress.
- A self-reported launch count or stage regression is a claim, not a measurement.

So: run again in a fresh copy, empty the kernel, count the launches.

## 2. Acceptance criteria

Check in this order. Stop at the first failure among 1–4; on the direct route, if 4 fails, still run the probe
once to name the stage. Read the resources (8) from the run of 4. Criteria 1–3 need no GPU.

| # | criterion | how to check by hand | what it catches |
|---|---|---|---|
| 1 | oracle untouched | everything outside `kernel_src/` is byte-identical to the task; no new Python file outside `kernel_src/`; the task's integrity hashes verify | an edited test, case file or tolerance |
| 2 | shape | the adapter class is where the task binds it; a layer kernel function exists; each stage's device function is defined once, in its own file, and the layer kernel imports and uses it; the vendored runtime equals the shared one | two diverging copies of a stage; a privately patched runtime |
| 3 | static rules | kernel sources import no framework or contract code and do no arithmetic on host tensors; the adapter calls nothing on `self` but its own methods and tensor metadata; **every wait on memory written by another block or GPU goes through the runtime with a deadline**; **no value that changes per step is passed from the host** (the launch epoch comes from the device) | native compute hidden in the adapter; bare spin loops; a host-side step counter |
| 4 | layer oracle, fresh | in a fresh copy with a fresh compile cache, run the oracle exactly as the task says: it **passes on every rank** (goldens, random parity against native, graph replay across widths, sequences, output independence). The speedup is taken from this run | a self-reported pass; a stale binary |
| 5 | hollow | in another copy, replace the body of **every** kernel function with a no-op and run the oracle's correctness: it **must fail** | outputs that do not come from the kernel |
| 6 | launch census | trace kernels during graph replay: **exactly one launch per layer per rank** on the eligible path, and **none of the native layer's kernels** | several launches; native kernels mixed in |
| 7 | per-stage evidence | stages route: every stage test passes with the device function taken from the fused sources. Direct route: the probe's local reference passes single-step, multi-step from a captured and from an empty cache, at block counts 1, 7 and all, with repeats | a stage broken during fusion; a missing state write; a race |
| 8 | resources, **pass/fail on their own** | from the ISA of the oracle run: register spill (scratch) 0, VGPR ≤ 512 (≤ 256 in a 512-thread layout), LDS ≤ 64 KiB, thread count as designed. A kernel with scratch above 0 fails even when the oracle passes and it is faster than native | a seed that only works by spilling |
| 9 | same decision on every rank | read the adapter: the choice between kernel and native path uses shape, dtype and configuration only, never data | one rank on the native path and another in the kernel: both sides wait to the deadline |
| 10 | no per-call allocation | the adapter allocates nothing per call except the output tensors; a weight repack is one-time, keyed by the weight's address, and its bytes are reported | memory growth; a capture that breaks |

How to do the less obvious ones:

- **Census (6).** Count on graph replay, not on the unit test's eager path: there the harness emits its own
  flush and copy kernels and eager calls clone their arguments, while a graph holds only the layer. Capture one
  decode step, replay it several times between two sentinel kernels (the internal check used five replays), and
  divide the number of kernels in the window by the number of replays. Trace the native leg the same way to
  learn the native kernel names. **Count totals; do not group by correlation id**: one graph launch can map to
  several ids (a native dense layer ran 18 kernels per replay, recorded under two ids), and grouping can split a
  two-kernel candidate into two groups of one. The internal check used the ROCm kernel-trace profiler.
  *(Exercised: the stub adapter showed the native count on both legs and was rejected, as it must be.)*
- **Scope of 5 and 6.** Together they answer "this one kernel computes the whole layer". They apply to the
  eligible widths; other widths going to the native path is legitimate. Criterion 6 belongs to a kernel authored
  to the one-launch structure; a deployed kernel of a dual-launch structure legitimately shows two launches.
- **If hollow still passes**, either the adapter falls back or the tests miss a part. Report it. Never change a
  test.
- **Fresh (4).** A new copy and a new compile cache, because the compile cache does not see edits in helper
  modules and an old binary can pass for the new source.
- **Speed.** A seed must be correct; it may be slower than the native layer. Use the number from criterion 4.
  The original flow recorded the author's own figure next to it; after optimization it raised a warning when the
  two were more than 10% apart (timing noise was about 2%).
- **Between small changes** the subset of criteria 1–3, 5 and 6 (no fresh oracle run) took roughly half the
  time of a full pass. Only the full pass decides.
- **After optimization** per-stage evidence is recorded, not required: fusing across a stage boundary is
  legitimate then (skill `mono_kernel_optimize`).

## 3. Robustness checklist

Acceptance sees one layer instance, the eligible widths, a few sequence steps and a few replays. These checks
cover what it does not. Each is pass or fail, with evidence.

| check | do | pass when | status |
|---|---|---|---|
| long sequences | at least 32 consecutive decode steps with the KV cache growing; candidate and native start from the same state (redesign: 64 steps from an empty cache, across several cache-block boundaries) | outputs within tolerance and the KV cache byte-identical after every step | first version, no recorded run |
| switching | in one process: width `w` → `w+1` (not eligible, so native) → `w`; decode ↔ prefill; a kernel layer next to a native layer of another kind | outputs and KV cache match native after every switch; the launch counter advanced only on eligible calls | first version, no recorded run |
| variants | every eligible width × every build flag the kernel defines | each compiles (record the compile time) and passes correctness at its width | first version, no recorded run; with a single width it only checks the flags |
| fault injection | stall one rank so that it never contributes | every rank ends within the deadline plus a margin; the error word decodes to the expected cause and site on every rank; no rank returns an output; after a reset a clean launch succeeds. **A hang is a failure** | first version (method left to the tester), no recorded run; a prescribed host-side stall was designed only |
| repeated runs | the layer oracle's correctness 10 times in a row in fresh processes (and the stage tests while stage boundaries exist) | all pass | first version, no recorded run |
| bare waits | read `kernel_src/`; list every loop that waits on memory written by another block or GPU | each is a runtime wait with a deadline | first version, no recorded run |
| memory | per rank and width: extra weight copies, peer buffers, control region, scratch, measured around the adapter's lazy initialization | nothing but the outputs is allocated per call; every extra weight copy is reported in bytes with the place it is allocated (first eligible call in the test adapter; in a framework integration before the KV cache is sized) | first version, no recorded run |
| graph replay | capture one decode step containing the kernel layer inside the framework's graph capture on every rank; replay it 1000 times with fresh inputs copied into the static buffers | the last replay matches native; no error on the way | first version, no recorded run |
| multi-instance | three instances of the layer class, each with its own weights, called in sequence in every step for 16 steps, also under replay | each instance matches its own native; the launch counter grows by the number of instances per step; swapping the call order changes nothing | **designed only** |

- The "do" and "pass when" cells merge the first version with the redesign where the redesign is stricter.
- **Fault injection.** The first version leaves the method to the tester: a kernel variant in which one rank
  returns before the all-reduce (which means editing the kernel under test), or holding one rank on the host
  past the deadline. The redesign prescribes the host-side stall: the other ranks wait inside the kernel and must
  reach the deadline, with no cooperation from the kernel. The opposite fault (a rank's kernel leaves midway)
  needs a debug-only switch in the runtime; designed only. Bound every driver with an outer wall-clock timeout
  that kills only the processes it started.
- **Multi-instance** decides whether delivery can replace every instance of a layer kind (a model has tens of
  them). It looks for a control region shared between instances, epochs that collide across instances, and an
  instance reading another's weights.
- **Repair.** A robustness failure is repaired in a copy. The repaired kernel must pass the acceptance criteria
  again before it replaces the accepted one. Record the speed before and after: a drop of more than 10% is a
  warning, not a failure *(the speed record and the full re-check after a repair were designed only)*. Do not
  trade correctness for speed. One repair; if it still fails, report the layer kind as not hardened and keep the
  accepted kernel as it was.
- **Why the first version was redesigned.** Its drivers are written anew each time and by design nobody
  re-checks them: a driver that runs the candidate on both sides, or injects no fault, would pass vacuously. With
  one eligible width the variants check is nearly empty, and one layer instance says nothing about many.

## 4. Out of scope here

- Prefill, mixed batches and widths beyond the feasible set always take the native path.
- Serving-level checks (greedy output parity with the switch on and off, launch counters growing on every rank,
  the switch-off path identical to the original) belong to skill `apply_mono_kernel_to_framework`.
- Speed against native after optimization belongs to skill `mono_kernel_optimize`.
