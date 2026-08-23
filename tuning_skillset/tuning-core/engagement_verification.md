# Engagement verification

> A tuning run that reports a speedup has proven nothing until you have shown that the
> artifact it produced is the code the machine actually runs.

This is the step that separates a tuning effort that delivers from one that produces a
folder of tuned configs and no measurable change. It is also the step most often skipped,
because **nothing fails when it goes wrong**. The tuner succeeds. The file is written. The
server starts. The performance is identical to before.

## Why this fails silently

Deployment is almost always a **lookup**: the live code builds a key from the call it is
about to make and looks for a tuned entry. If the key does not match, there is no error —
there is a fallback to the default path, which is exactly what you had before.

Every field in the key is a chance to miss:

- a `bias` flag that was `true` during capture and `false` at serving time
- a device-name string that resolves differently than you expect
- a dtype recorded as the storage format rather than the compute format
- a CU count baked into the key, because you copied a config from another machine
- a file in a directory nothing reads, or an env var the live path never consults
- a transposed shape — see below

## Establish the engagement signal *before* you tune

If you cannot state, in advance, the command that will prove the tuned path was taken,
**do not start tuning**. Build the check first. Tuning without it is unfalsifiable.

An engagement signal must be **positive** — evidence the tuned path ran. "Performance went
up" is not a signal (many things move performance, and noise is ~9% on this box). "No error
appeared" is not a signal.

## The four forms a signal takes

**Ranked by how much they actually prove, form 4 is first.** A profile names the kernel that
*executed*; a log line reports on a *lookup that preceded execution*, which is a weaker claim
and one that goes stale whenever the library changes its logging. Forms 1–3 are cheaper, so
use them as your fast loop — but the evidence you publish should be form 4, or form 1
corroborated by form 4.

### 1. A log line emitted on successful lookup

Common, but **read the two warnings below before you rely on it.** Libraries that do
per-shape lookup often log a hit — but frequently only when asked, and the log volume rarely
means what it appears to mean.

```bash
# aiter: the hit line does NOT exist unless you set the env var
AITER_LOG_TUNED_CONFIG=1 <your workload> 2>&1 | grep -c "is tuned on cu_num"
```

**Warning 1: the flag is not optional, and omitting it fails in the dangerous direction.**
On aiter `d9e5ef7c` the *miss* line (`not found tuned config ... will use default config!`)
prints unconditionally, while the *hit* line is emitted only under
`AITER_LOG_TUNED_CONFIG=1`. Without the flag the transition on a successful deploy is not
`not found tuned config` → `is tuned on cu_num`; it is **`not found tuned config` → silence**.
So `grep -c "is tuned on cu_num"` returns **0 against a fully working, verified deploy**, and
you will read that as *not engaged* when the deploy is fine. Verified on this image against a
deploy whose engagement was independently confirmed by kernel identity in a profile.

The useful corollary: because the miss line needs no flag, **the disappearance of the miss
line for your shape is itself a flag-free positive signal** — and a shape you deliberately
left untuned makes an excellent negative control, since it should keep missing. Use both
directions; a blanket disappearance of all miss lines would instead suggest logging was
suppressed wholesale.

**Warning 2: hit and miss counts do not measure call frequency.** In aiter the lookup is
wrapped in `functools.lru_cache(maxsize=1024)` keyed on **raw M**, so a shape logs once and
then goes quiet, with sporadic re-logging when eviction thrashes the cache. Line counts
therefore measure the **diversity of M values**, not how often a shape executes. A decode
shape that runs at a single M can log twice while executing ~110k times, whereas a spread-out
prefill band logs hundreds of times while contributing far less wall clock. Ranking tuning
targets by log-line count will point you at the wrong shape. Rank by measured execution time
instead.

Partial engagement is still the common middle case and looks like a disappointing speedup
rather than a bug — just establish it from execution time, not log volume.

### 2. A library-provided predicate

Some libraries expose a direct "is this shape tuned?" query. Use it if it exists — it tests
the same lookup the live path performs:

```python
from aiter.ops.triton.utils._triton.gemm_tune_check import gemm_tune_check
is_tuned = gemm_tune_check(some_gemm_fn, N=1280, K=4096, M=16, shuffle=True)
```

### 3. Validator/provenance metadata in the artifact

The best-designed tools refuse to apply an artifact from the wrong environment. torch's
TunableOp writes validators into its results file:

```
Validator,PT_VERSION,2.10.0
Validator,HIP_VERSION,702
Validator,HIPBLASLT_VERSION,100202-dabb6df2b9
Validator,GCN_ARCH_NAME,gfx942:sramecc+:xnack-
Validator,ROCBLAS_VERSION,5.2.0.dabb6df2b9
GemmTunableOp_BFloat16_NN,nn_1024_1024_1024_ld_1024_1024_1024,Gemm_Rocblas_-621294427,0.0105563
```

`GCN_ARCH_NAME` means a file tuned on gfx942 is rejected on gfx950 rather than silently
misapplied. Check your artifact carries such metadata; if it does not, you are responsible
for tracking arch/version provenance yourself.

Query the same information at runtime:

```python
torch.cuda.tunable.is_enabled()        # the mechanism is on
torch.cuda.tunable.get_results()       # what it actually selected, per op
```

`get_results()` returning entries for the ops you tuned is direct engagement proof.

### 4. Profiler evidence — the strongest form, and the one to publish

Not a fallback. **Prefer this.** Capture a trace and confirm the kernel that runs is the one
you tuned. Kernel names usually encode the tile configuration, so the trace tells you which
variant executed. Use `rocprofv3` (present in both target images) or `rocprof-compute` where
available.

It dominates the log-line form on every axis that matters:

- It reports **execution**, not a lookup. A config hit does not prove the tuned kernel ran;
  a kernel name in a trace does.
- It needs **no cooperation from the library** — no env var, no logging path that can be
  changed, removed, or gated in the next version. It cannot produce the
  false-negative-on-success failure described in form 1.
- It works when the configuration is **frozen** and you cannot set a logging flag (see below).
- It is the only form that works for hand-written kernels with no lookup layer at all.

Its cost is that it needs a profiled run, so keep forms 1–3 for the fast iteration loop and
reach for this one to confirm anything you intend to ship.

## When the configuration is the measurement contract

Most engagement signals ask you to change something — set a logging env var, raise a
verbosity level, enable a profiler. **On any A/B you would actually publish, you often
cannot**, because the configuration *is* the thing being held constant. If the frozen recipe
does not include `AITER_LOG_TUNED_CONFIG=1`, then adding it makes the run you verified a
different run from the run you measured.

Do not resolve this by verifying on the measured run. Resolve it by separating the two:

1. **Verify on a side run** that is identical except for the observability flag, and treat it
   as evidence about the *deploy*, not about the measurement. Confirm the tuned entry is
   found there.
2. **Use flag-free signals on the measured run itself.** For a lookup table, the
   disappearance of the unconditional *miss* line for your shape needs no flag at all. Keep
   an untuned shape as a negative control so you can tell "engaged" from "logging turned off".
3. **Confirm by kernel identity** (form 4) on a separate profiled run. Profiling perturbs
   timing, so never publish throughput from the profiled run — but kernel identity is not
   sensitive to that perturbation, which is what makes it usable here.

State in your writeup which run each piece of evidence came from. "Engagement verified on a
side run with logging enabled; throughput measured on the frozen recipe" is honest and
sufficient. Silently mixing the two is what produces published numbers nobody can reproduce.

## Verify the deploy handle, not just the tuning result

A tuner's own output identifier is not necessarily the identifier the deploy path wants.
Concrete case from `hipblaslt-bench`: `--algo_method all` prints a winner block headed
`[66]:`. That bracket number is the **position in the enumeration**, not the solution index.

```bash
# WRONG — the bracket number from the Winner block
$BENCH ... --algo_method index --solution_index 66
# error: NO solution found!

# RIGHT — ask for the real index, then pin it
$BENCH ... --algo_method all --print_kernel_info
#   --Solution index: 205113
$BENCH ... --algo_method index --solution_index 205113     # replays the same kernel
```

The bracket position also drifts between identical runs (`[99]` in one run, `[66]` in the
next, same solution). **Always round-trip your deploy handle**: take the identifier the
tuner gives you, feed it back in as a pinned selection, and confirm you get the same kernel
and the same performance. If the replay does not reproduce, your handle is wrong, and
whatever you deploy will silently fall back.

## Shapes are not always what you think

When capturing shapes from a live workload, the values a library logs may be the transposed
problem. Capturing a `(512,1024) × (1024,2048)` torch matmul yields:

```
hipblaslt-bench --api_method c -m 2048 -n 512 -k 1024 ... --transA N --transB N
```

M and N are swapped relative to the source-level call, because the library sees a
column-major view. **Replay the captured command rather than reconstructing it by hand** —
the dump is already correct, and hand-reconstruction is where mismatches enter.

Also de-duplicate: shape dumps contain one line per call, so a short run produces thousands
of identical lines.

## The procedure

1. **Before tuning:** identify the engagement signal and run it — confirm it currently
   reports *not engaged*. A check that cannot show the negative case proves nothing.
   Equally, confirm the check can show the *positive* case: point it at a shape the library
   already ships a tuned entry for. A check you have only ever seen return zero is
   indistinguishable from a check that is broken, which is exactly the trap in form 1.
2. Tune, gate on correctness, deploy the artifact.
3. **Run the engagement check.** Expect a positive. Do **not** expect the hit count to match
   the call count — cached logging makes counts a measure of shape diversity, not call
   frequency (form 1, warning 2). Judge coverage from where execution time goes.
4. **Re-measure end to end.** Engagement without an end-to-end improvement means you tuned
   something that was not the bottleneck — useful information, and a different problem.
5. **Record provenance** with the artifact: GPU arch, library versions, dtype, layout. An
   artifact you cannot attribute is one you cannot safely reuse.

## Failure triage

| Symptom | First thing to check |
| --- | --- |
| Zero hits | **First: is the hit line even emitted?** Many libraries gate it behind a flag (aiter: `AITER_LOG_TUNED_CONFIG=1`), so a working deploy logs nothing. Confirm the check can produce a positive at all before concluding anything. Only then suspect a key mismatch and diff every field of a captured call against a tuned entry. |
| Zero hits, and the flag is set | Key mismatch. Diff every field of a captured call against a tuned entry. Cross-check with the *miss* line: if it still names your shape, the key is wrong; if it has gone quiet, the lookup is succeeding and the logging is what is broken. |
| Some hits, most missing | Shape coverage — the live workload hits shapes you never tuned (often un-bucketed M). |
| Hits, but no speedup | You tuned something off the critical path, or the win is inside the noise floor. |
| Worked, then stopped | Version/arch drift, or the artifact path is no longer being read. Check validators. |
| Deploy handle rejected | You are using a display index rather than the real identifier. Round-trip it. |
