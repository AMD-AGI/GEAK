# Benchmarking a kernel that runs inside a captured graph

> If the decode path is graph-captured — and in SGLang it is, by default — then both *what you
> must measure* and *when your change can take effect at all* are different from the eager case.
> Getting this wrong does not produce an error. It produces a number.

Provenance: measured on MI355X (gfx950) / SGLang 0.5.17 / aiter `d9e5ef7c`, FP8 Qwen3-8B decode,
during a tuning run in which **three of four harness iterations were wrong for graph-related
reasons** before the fourth was trustworthy.

## The two consequences

**1. A change only takes effect at capture time.** The graph records the kernels that were
selected when it was captured, which happens once during server startup. Editing a config CSV,
setting an env var, or swapping a kernel implementation in a *running* process changes nothing —
the graph replays what it recorded. There is no warning, and the server keeps serving.

Therefore: **every candidate requires a process restart**, and every A/B is a
restart-to-restart comparison. That is not a detail about convenience, it sets your noise floor
— see Rule 3b in `measurement.md`, where the restart spread was 26× the within-process spread on
this workload.

**2. A kernel inside a graph is a different measurement from a kernel launched alone.** Launch
and dispatch cost is amortized across every kernel in the graph — ~790 of them in the decode
graph here — so a kernel timed by itself in eager mode carries a per-launch overhead it will
never pay in production. For small kernels that overhead can *be* the measurement, and you end
up ranking candidates by how cheap they are to launch.

The direction of the error is not fixed, which is what makes it dangerous. Eager timing
overstates the cost of small kernels, so it **overstates the win** from making them faster; but
it also hides regressions that only appear under the memory and occupancy conditions of a full
graph replay.

## Symptoms of a harness that is lying to you

| symptom | cause |
| --- | --- |
| Replay time is a few µs and barely moves with problem size | **Empty graph.** The kernel launched on a stream that was not being captured, so nothing was recorded. Extremely common with DSLs and libraries that manage their own stream. |
| Kernel win is large and reproducible; end-to-end delta is zero | You measured launch overhead, or the op is off the critical path (Rule 7), or the change never reached capture. Check engagement before concluding anything. |
| First timed iteration is a large outlier | JIT compile / autotune / workspace allocation happened inside capture or on the first replay instead of during warmup. |
| Numbers change when you reorder candidates in one process | Capture state or cache state leaking between candidates. Use one process per candidate. |

The empty-graph case deserves emphasis: it is fast, silent, and looks like a spectacular win.

## The harness contract

`../benchmark/graph_harness.py` implements this; read its module docstring before using it. The
rules it encodes, which apply to any harness you write yourself:

- **Allocate once, outside.** A replay reuses the same memory every time. Inputs and outputs are
  allocated before capture; the timed closure takes no arguments and allocates nothing that must
  differ between replays.
- **Warm up before capture, on a side stream.** Anything that compiles, autotunes, or allocates
  a workspace must have already happened. None of it is capturable.
- **Launch on the current stream.** `torch.cuda.graph` captures a private stream. A kernel
  launched on the default/NULL stream is not recorded.
- **Time replays with events, not wall clock.** Events bracket the GPU stream timeline, so the
  host-side replay launch cost is excluded — which is the quantity that matters.
- **Guard the capture.** Corrupt the output, replay once, and verify the result was recomputed.
  If it was not, the graph captured no work and the timing must be rejected rather than reported.
  This is the `dirty`/`verify` pair, and it is the difference between a silent
  mis-measurement and a loud one.

```python
from graph_harness import cuda_graph_bench

result = cuda_graph_bench(
    lambda: my_op(x, out),          # one invocation, pre-allocated tensors
    warmup=10, iters=30,
    dirty=lambda: out.zero_(),      # non-negotiable: proves the graph did work
    verify=lambda: torch.allclose(out, ref, atol=1e-2, rtol=1e-2),
)
assert result["mode"] == "cudagraph", result["mode"]   # never publish an eager fallback as graph
```

Check `mode` on every result. The harness falls back to eager timing when capture fails so that
a run always produces measurements, and a fallback silently accepted is exactly the error the
guard exists to prevent.

`../benchmark/run_case.py` does the generic disciplines correctly — synchronized, warmed,
median of independent `do_bench` calls, correctness and SNR alongside timing — but it is
hardwired to `torch.mm` on a single operand pair, and it neither captures a graph nor rotates
buffers. Treat it as a calibration reference for the timing methodology, not as something you
can point at a library op on a graph-captured path.

## Reading a profile of a graph-captured run

**The `Percentage (%)` column of a rocprof-style summary can measure launch wrappers rather than
kernel time, and will rank kernels wrongly.** In the trace from this workload it attributed time
to `hipGraphLaunch` — the host-side API call that replays an entire graph — so the ranking
reflected which graphs were replayed most, not which kernels consumed the GPU. Reading it as a
kernel ranking sent the first round of analysis at the wrong target.

Defences, in order:

1. **Confirm what the column is summing** before ranking anything by it. Separate host API
   traces from device kernel dispatches; they are different tables and only one of them answers
   "where did GPU time go".
2. **Rank by summed device duration per kernel name**, computed yourself from the dispatch
   records, rather than by any pre-computed percentage.
3. **Cross-check the top entry against arithmetic.** A kernel's time should be consistent with
   the bytes or FLOPs it must move; a roofline estimate that disagrees with the profile by an
   order of magnitude usually means you are reading a wrapper.
4. **Sanity-check that the ranking sums to the measured wall time.** A ranking that accounts for
   far more or less than the elapsed decode time is measuring the wrong thing.

Also note that a profiled run is not a measurement run: profiling perturbs timing, so publish
throughput from an unprofiled run. Kernel *identity* is not sensitive to that perturbation,
which is what makes a profiled run the right place to prove engagement
(`engagement_verification.md`, form 4).

## Checklist

- [ ] Confirmed whether the path under test is graph-captured at all
- [ ] Every candidate deployed **before** server start, and A/B done across restarts
- [ ] Noise floor measured across restarts, not repeats (`measurement.md` Rule 3b)
- [ ] Harness captures a graph and times replays, with events not wall clock
- [ ] `dirty`/`verify` capture guard in place, and `mode` asserted on every result
- [ ] Warmup (compile/autotune/workspace) completed before capture
- [ ] Kernel-level win re-checked end to end before it is called a win
- [ ] Profile ranking computed from device dispatch durations, not a percentage column
- [ ] Throughput published from an unprofiled run; engagement proven on a profiled one
