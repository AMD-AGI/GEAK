# Mode: rocprof

Use rocprof-compute counters for executed work and memory traffic. Keep them
separate from TL's modeled work/bytes, native timings, and reference selection.
Counter utilization does not measure whether the work was necessary.

## The recipe

Follow [runner.md](runner.md) for the shared per-case runner and batch interface.
The default driver requires a frozen peak file and per-call GPU markers:

```bash
export ROOFLINE_MIBENCH_CACHE=/path/to/verified/roofline.csv
python3 scripts/run_roofline.py --workdir "$PWD" \
  --cmd "python3 scripts/case_runner_template.py --adapter adapter.py --task op --case case1 --arm candidate --iters 3" \
  --name op_case1_candidate --output results/rocprof/leg --expected-calls 3 --peaks both
```

`profile --roof-only` collects the selected counter set; `analyze -b 4 -d ...`
analyzes only production dispatches. Setup/JIT/autotune/warmup, conditioning and
markers are excluded from this set. `--scope whole-process` is an explicit
historical diagnostic, not a production score. `--skip-noise` is retired because
symbol-based filtering can remove real operator work.

## 1. Version gate: rocprof-compute ≥ 3.6.0 on gfx95x

This gate concerns the microbenchmark that produces the reference, not whether
an older profiler may collect counters using independently verified references.
The gfx950 campaign found CDNA3 MFMA paths in old microbenchmarks: BF16 about
614,663 versus 2,453,780 GFLOP/s, and FP8 about 2,455,000 versus 4,949,468.
Upgrading to 3.6.0 alone did not resolve every precision's calibration problem.
Verify the actual instruction family, units, device/partition, and calibration
source. Do not install a permanently hardcoded historical ceiling.

`peak_audit.py` checks positive HBM values and flags materially inconsistent
equal-/scaled-rate MFMA references on gfx950, including F8 versus F6F4/2. The
driver rejects suspect calibration by default. `--allow-unverified-peaks` retains
the warning and labels output diagnostic; it does not replace the peak.
These ratio checks are warnings about calibration, not proof of a physical bound.

## 2. Freeze the denominator across every comparison

`ROOFLINE_MIBENCH_CACHE` must name an existing file. The driver records its hash,
contents, and selected tool version and copies it into each profile directory.
An environment variable pointing to a nonexistent file is an error, not a bypass.
Per-case automatic microbenchmarking is disabled. Preserve the independently
collected calibration evidence; verify that the tool actually reused the file.

Missing dtype peaks remain missing. There is no automatic `2 * FP8` substitution
for F6F4 and no automatic sibling substitution. An explicit compute reference
must be labeled with its provenance. `--hbm-peak-const` remains a diagnostic
bandwidth-utilization reference; it does not override the empirical roof's HBM
peak, which comes from the frozen CSV. TL/rocprof peak ratios can be computed
from the two actual files via `tl_only.py --rocprof-reference`.

## 3. HBM read bytes read 2× low on gfx950

Affected rocprof versions inferred 128-byte requests through `TCC_BUBBLE`, which
is zero on gfx950. The actual correction depends on the request-size histogram;
blindly doubling bytes is wrong for mixed/scattered loads. The bundled
`rocprof_hbm_patch.py` provides the historical backport. Verify the applicable
counter definitions and tool version before using it. Patching read counters does
not repair a compute calibration. Preserve raw counters and the patch identity.

## 4. `rocprofv3` is not `rocprof-compute`

Raw rocprofv3 FETCH_SIZE/WRITE_SIZE use KiB in the documented campaign convention;
rocprof-compute emits derived rates. Never mix their units or double-count two
read-byte estimators. See [measurement.md](measurement.md) for version-specific
raw counter formulas. Counter analysis is not required for TL-only.

## 5. At most 4 TCC counters per pass

The campaign's hardware rejected a fifth simultaneous TCC counter. Split
incompatible sets and align the case, input, cache policy and measured calls.
The current window adapter requires identical selected ID/name sequences across
CSV passes because `analyze -d` applies one list to all passes; it fails and keeps
evidence if the tool remaps IDs. Do not silently reuse the first pass's IDs.

## 6. One shape per profiled command

The process can contain arbitrary setup dispatches, including same-named install
self-checks and Triton autotune trials. Name filtering cannot separate those from
production. The runner primes the exact callable in the same process, then emits
one begin/end marker pair per measured call. `rocprof_window.py` validates all
windows, conditioning presence and repeated kernel multiplicity before analysis.
An explicit adapter `expected_dispatches` strengthens this check. It does not
force arbitrary autotuner configurations or assume they survive another process.

The batch interface takes `task case regime call_weight`, preserves a complete
request manifest, protects stdin with fd 3 and `/dev/null`, propagates failures,
and collects results against that manifest. It does not silently skip old output.
Use a fresh output directory for each leg.

## 7. Judge contention on `achieved`, not on AI

Record repeated timings, ordering, device/NUMA and load. Stable AI with changed
throughput can suggest changed conditions; it does not by itself prove contention.
For A/B work removal, `fold.py --candidate` reports time/byte/FLOP ratios when
counter counts are available. If bytes shrink more than runtime while runtime
improves, it emits `AI_SHIFT_NOT_REGRESSION`. Lower counter utilization is not a
reason to revert a faster, correct implementation.

## 8. Eff > 100% is a disproof of the denominator

Preserve the value and investigate scope, units, timing, reference and traffic
assumptions. Empirical calibration can be exceeded, so this historical heading
must not be read as proof of one particular cause. Never clamp or select a lower
byte model to make the percentage plausible. A disagreement with TL's modeled
axis is not an independent measurement of the hardware bottleneck.

## 9. VALU kernels have no datasheet peak

The bundled gfx950 datasheet mapping does not supply every vector/integer peak.
Keep those reference-specific efficiencies unavailable and report coverage.
Likewise, integer MFMA work must not be scored with a FLOP-only achieved rate.

For small nonzero AI, never use the two-decimal display AI as authoritative.
`counter_metrics.py` derives AI from same-report FLOP/s and GB/s when possible.
For the memory-bound branch it uses `achieved_BW / peak_BW` directly. The six
original top-k reports are regression fixtures; the ctx418k baseline now gives
237.86/6857 = 0.0346886393 even though displayed AI is 0.00. Source rates may
themselves be rounded; retain them and do not claim exact unrounded counters.
When work or peaks truly are unavailable, report a reason and available memory
utilization separately rather than inventing a complete roofline.

## 10. Changing the peak policy does not require touching the GPU

`rerender.py` and `repeak.py` can process saved text reports into a separate
directory. Their source-text precision remains a limitation. New driver outputs
include numeric JSON, full profile CSVs, marker audit, and provenance. Fixing an
old aggregate's scope requires original dispatch data or a new measurement;
re-rendering a polluted aggregate cannot remove its warmup/autotune contribution.

## 11. `--roof-only` and `-b 4` are not the same thing — do not delete only one

The former selects collected counters, the latter displayed analysis blocks.
`--mode full` collects all available counters and retains measured-window
selection. Request it only when additional diagnostics are needed.

## 12. A finished profile can still exit non-zero

Plot export or post-processing can fail after counters were collected. Preserve
both stdout and stderr, the return status, and the entire profile directory.
Do not claim a successful measurement merely because a CSV exists. The current
driver retains failure evidence instead of deleting its temporary profile.

## Verify

Check process success, selected windows, dispatch counts, cache protocol, source
precision, peak provenance, and coverage. `collect_results.py` emits missing rows
for the original requested cases; `fold.py` distinguishes case and kernel
coverage. Partial coverage never appears as a complete campaign score.
