# Acceptance: rules for every task, rules for a fused seed only, and what the oracle sees

A whole-layer kernel can pass its unit test and still be unacceptable, because the cheapest way to look
fast is to stop being a mono kernel. Each rule below exists because its violation is the cheapest path
to a better number. Decide the task shape first ([kernel_structures.md](kernel_structures.md)):
(a) a deployed mono kernel whose baseline is that kernel itself, or (b) a freshly fused seed whose
baseline is the framework's native layer.

## Running the oracle (every role, both shapes)

- Run it exactly as the lane's COMMANDMENT file and the docstring at the top of the task's
  `unittest.py` say. Do not invent flags, do not shorten the case list, do not run a rank by hand.
- A multi-rank unit test starts one process per rank itself. **Every GPU of the rank group is held in
  one lease for the whole command.** The COMMANDMENT's command takes that lease (some unit tests lease
  their own peers): run it as written and never wrap it in a second lease. A rank on a GPU you do not
  hold, or a second command sharing one of the GPUs, invalidates the timing and can stall the peers.
- Leave the visible-device list as the lease set it.
- The same holds for your own probes (timeline driver, microbenchmark, kernel trace): one lease over
  the whole group, one process per rank started by your command.
- An environment error because the test got fewer GPUs than ranks is not a candidate failure: fix the
  lease; do not touch the candidate or the test.
- Correctness before benchmark, always. Budget for it: a multi-rank run is slow.
- The oracle files (`unittest.py`, the case list, metadata, reference inputs and outputs, any vendored
  harness) are immutable; usually they are integrity-pinned and the test refuses to run once one has
  changed. A candidate that needs an oracle edit is rejected.

## Universal rules (shape (a) and shape (b))

The task statement's own constraints — memory footprint, graph safety, no host-side win — rank above
every lever and above every figure in this skill.

| # | rule | why it exists | how to check by hand |
|---|---|---|---|
| 1 | Correctness passes on every rank under the tolerance as given. Never loosen the tolerance | it is calibrated against the native path's own error and against negative controls; a looser one hides drift that later forks outputs | the oracle; repeated after any synchronization or LDS change |
| 2 | No more launches per layer per rank than the baseline, and no native-layer kernels beyond those the baseline's own path runs | splitting back into several kernels, or returning to the native path for eligible widths, gives a "speedup" of about 1 plus noise — and is no longer a mono kernel | kernel trace: kernels per decode step per rank ÷ layers, and their names |
| 3 | The outputs come from the kernel; the host side (adapter / wrapper) computes nothing new | work moved to the host disappears from the kernel's time and breaks graph replay | hollow test: with the kernel body emptied, correctness must fail |
| 4 | Never remove a bound (deadline, error report) from a wait; a new wait uses the kernel's existing wait helper, never a bare spin loop | an unbounded wait turns a lost publish or a dead peer into a hung GPU, and the unit test does not see it | read every wait you touched; a forced timeout ends in the deadline error inside the deadline — it never hangs and never continues on stale data |
| 5 | The launch tag comes from device memory; nothing that changes per step is passed from the host; every argument is a fixed pointer | graph replay re-issues the same arguments; a host step counter yields stale tags on replay | the oracle's graph-replay cases; the launcher's argument list |
| 6 | The choice between kernel and native path is identical on every rank, made from data all ranks share (shape, dtype, flags) | a rank on the other path leaves its peers waiting until the deadline | read the dispatch condition |
| 7 | The all-reduce stays an fp32 sum in rank order on every rank | it is what keeps the ranks bit-identical | rank agreement in the oracle |
| 8 | No per-call allocation other than the outputs. No additional or re-laid-out copy of weights or scales when the task states a memory limit, unless it replaces the original in place with no extra bytes or the task statement itself allows that kind of copy (then its bytes are stated against the limit); where memory is left open, a repack is made once, keyed by the weight's data pointer, with its bytes stated | per-call allocation breaks graph capture; every extra copy is memory taken from the KV cache | read the task statement and the wrapper; state extra bytes in your notes |
| 9 | No host synchronization on the steady decode path | a host sync inside the captured step stalls capture or serializes the step | read the wrapper for value reads, device syncs, branches on device scalars |
| 10 | The speedup that counts comes from an independent re-run (the lane's verify step): baseline and candidate timed in the same run, in a clean copy with a fresh build / JIT cache | an engineer's own number can come from a stale cache, a different GPU group or a lucky sample | the lane's verify step; a gap above about 10% between your number and the re-run is a warning (run-to-run noise was about 2% in one harness; measure the A/A band of your own oracle) |
| 11 | The timeline / profiling build is never the installed kernel | acceptance applies to what is installed | name and argument list of the installed launcher |
| 12 | ISA scratch 0 and VGPR / LDS inside the caps (for a deployed kernel: not above the baseline's) | a spill turns a win into a loss, and values parked in the accumulator file cost issue slots without showing as scratch | register counts, scratch size and the number of accumulator moves in the ISA, after every prefetch or state-holding change |

Notes. Rule 2 means "not more than the baseline": a deployed kernel with a front launch and a tail
launch stays legal with two, and a partial kernel keeps the framework's attention kernels. Rule 4 is one
rule for every kernel: keep each bound the baseline has; give a new wait the existing helper; if the
kernel's waits have no deadline at all, adding them is a separate, declared change, not part of a speed
patch. The lane itself records a round winner only above its minimum improvement (2% by default) and
its engineer role saves a patch as soon as it beats the incumbent: do both exactly as the role files
say. "Above twice the A/A band of this task's oracle" is what counts as evidence in your notes; it is
not a threshold for saving a patch — the verify re-run decides.

## Rules only for shape (b), a freshly fused seed

The seed arrives correct and with one launch per layer (its authoring flow checked both) and may be
slower than native. These rules apply only when the kernel vendors that flow's adapter and
synchronization runtime; a deployed kernel that happens to be timed against the native layer is shape (a).

| rule | why |
|---|---|
| Exactly one launch per layer per rank on the eligible path | the seed was fused to that design; only a deployed kernel may bring its own launch structure |
| The adapter calls no framework submodule | on the candidate leg the native layer is one call away: with it the speedup is about 1 and any noise looks like progress |
| Every wait goes through the vendored synchronization runtime, and that runtime is not edited | a runtime defect is reported, not patched inside one candidate; a bare spin loop passes the unit test and shows only when a peer stalls |
| Know the candidate floor before you start | the lane times against native, and its candidate floor is a launch argument (1.0 unless set), not something a role chooses. A seed below the floor has every step filtered out: no patch is kept, no winner recorded. Read the floor in your prompt; if the seed is below it, say so at once, report gains relative to the seed, and state that steps under the floor will not be recorded. Whoever launches such a lane should set the floor relative to the seed (the source flow used half of the seed's speedup) |
| Stage boundaries are not a gate | merging stages (a norm into the GEMM before it, two GEMMs sharing one activation read) is legal; per-stage tests kept from fusion become a record; whole-layer correctness is the gate |
| Work in the lane's workspace, never in the seed's directory | a failed final state must not overwrite the seed |
| A final speedup below 1.0 is not a failure in itself | one launch per layer still saves launches and host overhead, but it weighs on delivery — report it plainly |

If the task comes from a flow with an acceptance step after the lane, expect it to re-run correctness
and timing in a fresh copy, repeat the hollow test and the launch census, and go back through the
round states, newest first, keeping the first that passes; the speedup reported is that re-run's. So:
one verified step per saved state; a round built on a violating candidate is wasted; never fold a
violation into a later step.

## What the oracle checks, and what it does not

What a whole-layer unit test of this kind typically checks — read the docstring of yours, do not
assume:

- correctness on every rank: fixed goldens, random parity against the baseline leg, graph replay across
  widths, sequences over several steps, output independence;
- every comparison is against what the baseline leg produced for the same check;
- time per case: device time under graph replay, with a cache flush and a cross-rank alignment before
  each sample; the median over samples of the slowest rank; there is no separate eager timing to optimize;
- a weighted speedup when the cases carry serving weights (the primary target); the geometric mean is
  secondary.

What it does **not** check — yours to check by hand:

| not checked by the oracle | how to check | when |
|---|---|---|
| launch count; native kernels on the path | kernel-trace census | before banking a structural change |
| that outputs come from the kernel | hollow test | after touching the adapter / wrapper |
| unbounded waits | reading; a forced timeout | after any synchronization change |
| rare races (LDS reuse, a missing barrier) | repeat correctness three times on the state you save | after a synchronization or LDS change (one run per step while iterating) |
| tag wrap | read the tag arithmetic; run a wrap test if the task has one | when tags or epochs change |
| memory: per-call allocations, hidden copies | read the wrapper | when buffers are added |
| register spills (rule 12) | scratch and accumulator moves in the ISA | after every prefetch or state-holding change |
| serving-side behaviour: capture modes, a KV cache re-allocated after start-up, a host watchdog | not a lane matter. First delivery of a kernel behind its switch: `apply_mono_kernel_to_framework`. A deployed kernel changed by a lane: the e2e workflow's overlay re-integration and throughput gate (new kernel against old, switch on in both) | at delivery |

Serving-integration facts an optimization must keep (do not "simplify" them away):

- KV-cache pointers are re-read every step; the cache can be re-allocated after start-up profiling.
- Steps captured in a piecewise or small dummy capture mode take the native path, on every rank alike.
- A weight layout the kernel does not expect disables the kernel on every rank alike; it does not
  crash.
- Peer memory is created outside graph capture, once, collectively on all ranks.
- A buffer shared over IPC is opened at its allocation base; a tensor sub-allocated by a caching
  allocator sits at base + offset. Treating the tensor's own address as the base gave wrong results
  with no error (*seen in the source workflow*) — remember it when you add a peer buffer.

## Reporting

State the structure class and the task shape; which levers were already applied; what was measured and
what was assumed; how many correctness repeats passed; the speedup from the same-run comparison and,
when there is one, from the independent re-run. List every rule above you could not check as
**unchecked**, never as passed.
