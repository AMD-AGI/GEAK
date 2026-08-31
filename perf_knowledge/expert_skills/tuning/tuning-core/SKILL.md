---
name: tuning-core
description: The universal loop for tuning GEMM and other GPU ops on AMD Instinct — scope the search space, measure honestly, gate on correctness, and prove the tuned artifact actually engaged. Start here before any language- or framework-specific tuning skill.
---

# Tuning core

This is the discipline every other tuning skill in this set specializes. Read it first.
It is language-agnostic: the same six steps apply whether you are tuning a Triton kernel,
racing hipBLASLt solutions, or deploying an aiter config into a live server.

## The claim this skill exists to defend

> A tuning run that reports a speedup has proven nothing until you have shown that the
> artifact it produced is the code the machine actually runs.

The dominant failure in GPU op tuning is not picking a bad config. It is picking a **good
config that never gets used** — a lookup key that misses, a config file in a folder
nothing reads, an env var the live dispatch path ignores. Nothing errors. The tuner prints
a win. End-to-end performance does not move. If you take one thing from this skill, make
it step 6.

## The loop

```
1. SCOPE      what op, what shapes, what dtype, what actually runs today
2. BASELINE   measure the current path — before touching anything
3. SEARCH     enumerate a space, prune it, race it
4. GATE       reject on correctness before you believe any timing
5. DEPLOY     write the artifact where the live path reads it
6. VERIFY     prove engagement, then re-measure end to end
```

Steps 1, 2, 4 and 6 are where tuning efforts fail. Step 3 is the part everyone thinks is
the hard one; it is usually the most automated.

### 1. Scope

Answer these before running anything. Guessing here wastes the entire run.

- **Which op, and which implementation is live right now?** Not which one you expect —
  which one dispatches. A framework may route the same `nn.Linear` through torch, a
  library heuristic, or a fused kernel depending on env vars and shape.
- **Which shapes?** Capture real ones if the workload exists (see the per-framework
  skills). Otherwise generate a corpus spanning the regimes — see `../benchmark/`.
- **Which dtype, and which dialect of it?** On gfx942, FP8 is FNUZ. On gfx950 it is OCP.
  Same 8-bit layout, different exponent bias — a mismatch is silently wrong numerics,
  not a crash.
- **What is the arithmetic intensity?** A K-heavy 4096×4096×4096 GEMM and a decode-shaped
  M=1 GEMV are different problems with different winning strategies. Compute-bound work
  wants tiles and MFMA scheduling; memory-bound work wants split-K, vectorization, and
  occupancy.

### 2. Baseline

Measure the existing path first, on the same box, in the same process shape you will use
for the final comparison. A tuned number compared against a remembered or documented
baseline is not a measurement.

Pin your GPUs. On a shared box, other tenants will move your numbers:

```bash
rocm-smi --showuse --showmemuse     # find genuinely idle GPUs
export HIP_VISIBLE_DEVICES=4        # then pin. every run. no exceptions.
```

See `measurement.md` for how to time correctly. It is not as simple as wrapping a timer.

### 3. Search

Enumerate → prune → race. The details are per-tool, but the strategy is shared; see
`search_strategy.md`. Two rules that hold everywhere:

- **Prune before you race, not after.** Search spaces are combinatorial. Constraining to
  hardware-sane candidates first (tile shapes that match the MFMA instruction, workgroup
  counts that fill the CUs, configs that do not spill) cuts run time by an order of
  magnitude without costing you the winner.
- **Change one variable at a time** when hand-tuning. If you change three things and it
  gets faster, you have learned nothing transferable.

### 4. Gate on correctness

**Never accept a config on timing alone.** A config that races ahead because it computes
the wrong thing is the easiest way to ship a regression.

The gate must be a **relative** error measure. Absolute error is meaningless at scale — a
bf16 GEMM at K=4096 accumulates ~1.0 absolute max error against an fp32 reference purely
from dtype, with nothing wrong. See `correctness_gates.md`, which works that exact case.

The convention used across the AMD tuning tools is `err_ratio < 0.05` against a
higher-precision reference. Match it unless you have a reason not to.

### 5. Deploy

Write the artifact where the live path reads it. This is per-tool and per-framework, and
it is where lookup-key mismatches are introduced. The recurring trap: a tuned entry is
keyed on a tuple, and **every field must match the live call exactly**. A single wrong
field — most often a `bias` flag or a device-name string — means a 100% lookup miss.

Tuned artifacts do not transfer across GPU architectures. Some lookup keys literally
include the CU count (304 on MI300X, 256 on MI355), so a config tuned on one part is a
guaranteed miss on the other. Re-tune per arch; never copy. When moving a whole tuning
setup to a new part, `arch_migration.md` lists what has to be re-derived and, more
importantly, which of it fails silently rather than erroring.

### 6. Verify engagement — then re-measure

Two distinct checks, both required.

**(a) Did the artifact engage?** Find the positive signal that the tuned path was taken.
Every per-tool skill in this set ends with a concrete one. Examples of the *form* it takes:

```bash
# a log line the library emits on a successful tuned lookup -- note the env var: aiter's
# hit line is gated behind it, so without the flag this returns 0 on a WORKING deploy
AITER_LOG_TUNED_CONFIG=1 <workload> 2>&1 | grep -c "is tuned on cu_num"

# a library-provided predicate
python3 -c "from aiter.ops.triton.utils._triton.gemm_tune_check import gemm_tune_check; ..."

# strongest form: the kernel that actually executed, from a profile
rocprofv3 --kernel-trace -- <workload>        # then confirm the tuned kernel name is there
```

If you cannot find a positive engagement signal for a path, **that is the first thing to
build**, before tuning anything. Tuning without it is unfalsifiable.

Two traps that make a log-line check report failure on success, both covered in
`engagement_verification.md`: the hit line may only be emitted under a debug flag you cannot
always set, and hit *counts* are usually cached and so measure shape diversity rather than call
frequency. Prefer kernel identity from a profile for anything you intend to publish.

**(b) Did the number move where it matters?** An isolated kernel speedup is not an
end-to-end speedup. A kernel that is 3× faster but 2% of total GPU time buys you 1.3%.
Check the op's share of total time before investing in it, and re-measure end-to-end after.

## When to stop

Stopping rules matter as much as search:

- **Gate met** → stop. Define the target before you start.
- **Three consecutive iterations under ~2% improvement** → the config space has plateaued.
  Further tile tuning is not the lever; a structural change is.
- **Profiler shows the kernel near a hardware limit** (MFMA utilization saturated, or at
  achievable bandwidth for a memory-bound op) → stop, you are done.
- **Register pressure sitting at an occupancy boundary** → more config search will not
  help; the kernel needs restructuring.

## Anti-patterns

| Anti-pattern | Why it burns you |
| --- | --- |
| Comparing backends without checking operand layout | A row-major vs column-major B operand changes throughput substantially. Two "GEMM benchmarks" at the same M/N/K can be different problems. |
| Absolute error as the correctness gate | Scales with K and dtype; either passes everything or fails everything. |
| Tuning on a shared, unpinned GPU | Another tenant's load lands directly in your numbers. |
| Trusting a tuner's own reported speedup as the result | It measured its candidate in isolation, not your live path. |
| Copying tuned configs across architectures | Different CU count, LDS budget, and instruction set. Often a silent lookup miss. |
| Reporting a win without an engagement check | The most common way a "successful" tuning effort delivers zero. |

## Where to go next

| You are tuning | Read |
| --- | --- |
| A Triton or Gluon kernel you control | `../tuning-triton/` |
| A FlyDSL kernel | `../tuning-flydsl/` |
| A raw HIP kernel | `../tuning-hip/` |
| CK / ckProfiler instances | `../tuning-ck/` |
| hipBLASLt solution selection | `../tuning-hipblaslt/` |
| aiter's per-shape config DBs | `../tuning-aiter/` |
| A live vLLM server | `../tuning-in-vllm/` |
| A live sglang server | `../tuning-in-sglang/` |
| Missing tools in your container | `../env-setup/` |

Supporting detail in this skill: `measurement.md`, `correctness_gates.md`,
`search_strategy.md`, `engagement_verification.md`, `arch_migration.md`,
`clocks_and_power.md`, `graph_captured_benchmarking.md`.

**If you are tuning a serving decode path, read `graph_captured_benchmarking.md` first.** It
governs both how you must time a kernel and whether your change can take effect at all, and it
changes the noise floor every other page's advice is calibrated against.
