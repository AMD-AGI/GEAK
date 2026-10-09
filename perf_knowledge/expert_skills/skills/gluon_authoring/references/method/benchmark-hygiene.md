# Benchmark hygiene — harness, instrument, timing, acceptance

**What this decides.** Whether a number may be believed: which harness produced it, which source tree
and device it measured, which protocol and statistic it was read with, and whether it is a search
reading or an acceptance reading. **When you are here:** before the first timing of a run (build the
harness, pin the contract), at every A/B (pick the protocol, check the instrument), and at every keep /
revert / close decision (acceptance, then the evidence-reasoning checks below). In this skill the two
baselines are `plain_baseline` (the target line) and `gluon_anchor` (the working baseline,
`recover.md`); the rules below apply to any A/B pair. Record schemas align with `records.md`,
`budget.md`, `profile.md` and the `final_report.json` in `close.md`.

## The binding rule: GEAK's harness owns acceptance timing

One owner. **Every acceptance number** — a keep, a commit, a champion, a close — is
measured the way `e2e_workflow/scripts/harness_lib.py` measures it:

| property | what `harness_lib` does | where |
| --- | --- | --- |
| timer | **CUDA/HIP events** around the launch(es): GPU-timeline device time, which excludes host launch/dispatch. Wall-clock (`perf_counter` + sync) is measured in the same loop as a **reference only**; a large wall ≫ device gap flags a host-bound op | `time_op(call, warmup=10, repeats=50, inner=1, graph=False, *, detail=False)` |
| per-sample sync | every sample is flush → sync → `start.record()` → `inner` launches → `end.record()` → `end.synchronize()`. Batching samples into one queue with no intervening sync was **tried and rejected on measurement**: fitting per-launch ms against `1/inner` on MI355X (torch 2.9 / ROCm 7.2) to separate kernel time K from per-window overhead E showed batching **raised E in 5 of 6 configurations**, and under flush let the two timers disagree on K itself by **up to 2.8x** — flush and kernel become two commands in a packed queue and the cold-HBM guarantee stops holding | `_time_events`, `_time_graph` |
| cache condition | **read-evict before every sample** (default; cannot be disabled or switched to writes): one float32-sum read over a buffer larger than the last-level cache (`HARNESS_CACHE_FLUSH_MB`, default 512 MiB > MI300's 256 MiB Infinity Cache), outside the event and wall windows. The old 512 MiB `zero_()` **write**-evict left dirty lines whose writeback competed with the timed kernel: on MI355X / GLM-5.2 fused-MoE, decode-weighted speedups read **1.408 / 1.395 with write eviction, 1.121 / 1.129 with read eviction, 1.104 / 1.117 without eviction**. `detail=True` returns the policy as `cache_condition`; its `deployment_calibrated: False` is literal — a sensitivity condition, not a residency proof. A receipt **without** `cache_condition` predates it and was taken under write-evict — never read absence as "no cache preparation" | `cache_policy`, `flush_cache` |
| statistic | **median** of the per-sample device times (and of the reference wall times) | `time_op` |
| host-bound receipt | `primed` is a **three-state** reading, never a bool: `True` = dispatch cheaper than the kernel, score it; `False` = the op dispatches slower than it computes, `ms` is a host-bound latency and a candidate can "win" by collapsing dispatch alone; **absent** = the timer could not tell (no CUDA) — not the same as `False`. `host_ms` is measured on a drained queue over a fixed 30 launches (`_HOST_PROBE_LAUNCHES`; deriving it from one cold read overstated a 512^3 GEMM's dispatch ~2x on MI355X and flipped a primed measurement to False). `primed=True` does not mean overhead-free: a window costs a few µs beyond the kernel — negligible for a 100 µs GEMM, most of the number for a 5 µs op; raise `inner` for sub-µs kernels (the overhead is per window, so it divides away) | `_host_dispatch_ms` |
| graph boundary | `graph=True` times a captured-graph replay (the decode deployment context) with the same event + flush method; falls back to eager event timing if capture is unavailable (see the `timer` field: `cuda_event_graph` / `cuda_event` / `wall`) | `_try_capture`, `_time_graph` |
| legs | acceptance runs **one fresh subprocess per leg per bucket** — a warm interpreter shares JIT/autotune state between legs and reports a pseudo-1.0x. Each rep is an interleaved **B,C pair**; pairs repeat up to `max_reps=3` only while the running median ratio is inside `undecided=(0.95, 1.10)`; `baseline_ms` / `optimized_ms` are the **median over the pairs actually run**; `reps` and `speedup_spread` ride along. Fresh-process variance measured on gfx950: **~0.1–0.5 %** on a prefill bucket, **up to ~14 %** on a small-M decode bucket — a real 1.05x sits inside the noise of a single unpaired pair. Before measuring, `assert_legs_differ` refuses unless the two legs import **different** code and the baseline resolves **outside** the task dir | `measure_legs`, `assert_legs_differ` |
| baseline | the baseline is re-measured **in the same window**, interleaved with the candidate — never a stored number (`## Same-knob baseline`) | `measure_legs` |
| correctness | `correct(out, ref, tol)`; `check_correct_multi` keeps every returned tensor live before comparing (catches a shared `static_out`); `assert_independent_outputs`; `check_random_vs_baseline` validates against the **live** baseline on several random value draws at the same shapes (correctness gates, its speedup is report-only) | `harness_lib` |
| e2e sanity | `amdahl_ceiling` / `amdahl_check`: an observed e2e delta far above the ceiling a kernel at `pct_gpu` can produce is box drift / measurement error, not the kernel | `harness_lib` |

**Commit gate.** GEAK banks a round winner only if its verified geomean beats the cumulative best by
`MIN_IMPROVE` — **2 %** by default (`kernel_workflow/kernel_lane.js`, knob `min_improve`; tested in
`kernel_workflow/scripts/test_candidate_floor.js`). A candidate below that can be **tracked**
(`candidate_floor`) but never **banked**. A noise band measured in this window (a control arm, an
in-window same-arm repeat, `speedup_spread`) **may only make the verdict stricter**: the effective keep
threshold is `max(MIN_IMPROVE, measured band)`. Nothing on this page lowers it.

### Who owns what in GEAK

GEAK's `kernel_workflow` injects this skill into its existing roles; GEAK's phases are the stage machine
(`orchestration.md` maps stages to GEAK roles).

| | in GEAK |
| --- | --- |
| who owns the round loop / acceptance | GEAK: round loop (tech_lead plans, the deep_engineer runs the `deep_explore` direction), `verify_engineer` re-benchmark + Director validation, `MIN_IMPROVE` commit gate |
| harness / baseline | `benchmark_engineer` (COMMANDMENT, baseline timing); `scripts/create_harness.py` emits a `harness_lib`-timed harness when one has to be authored |
| acceptance numbers come from | `verify_engineer` / `measure_legs` (`harness_lib`, fresh process per leg, same-window baseline) — never from `ab_bench.py` |
| `scripts/ab_bench.py`, `plain_autotune.py`, in-process sweeps | **search / screening only** in the deep_engineer's own loop — rank, reject early, measure the window's noise band |
| GPU access | `kernel_workflow/scripts/gpu_lock.sh` (lock dir `/tmp/team_gpu_locks`), one GPU per engineer; optional broker only with `GEAK_GPU_BROKER=1` |

### Search metric vs acceptance metric — where pack techniques sit

Several pack techniques below are excellent **screening** instruments and wrong as **acceptance**
instruments. Each is labelled where it appears; the summary:

| technique | status | why |
| --- | --- | --- |
| `min` over repeats as the kernel floor | **search/screening only (ab_bench)**; report it beside the median as a supplementary column | acceptance statistic is the `harness_lib` median; switching estimator mid-campaign invalidates history (`## Repeatability + measurement order`) |
| batched wall-clock timing (one sync around N calls) | **diagnostic cross-check only** | rejected by the per-sample-sync measurement above; wall-clock is floored by Python dispatch on decode shapes |
| hot-cache protocol (`--cache hot`) | **search/screening only (ab_bench)** for compute-bound kernels with real reuse | acceptance is always read-evict |
| one process timing all variants, interleaved cells | **search/screening only (ab_bench)**, and never for variants that patch the toolchain | acceptance = fresh process per leg; a patched variant always gets its own process |
| write-evict flush | **not used** | inflates speedups through writeback contention (1.40 vs 1.12) |
| fixed iteration count vs time budget (`--budget-ms`) | screening protocol choice | acceptance sample count is the harness's `repeats`, widened by `measure_legs` pairs |
| control arm (`--control`), `--permute`, `fingerprint()`, `--preload` | **kept** — stricter supplementary checks | they can only void or tighten a verdict |
| UUID / PCI anchoring, sustained idle window, sub-ms warmup table, graph-served kernel-only metric | **kept** (pack-only techniques; the warmup table is screening guidance) | GEAK has no counterpart; none conflicts |

**Chapter map (execution order).** Build the harness (`## Toolchain pin` → `## Profiler output hygiene
(do not pollute the repo)`) → before timing, the box and the device (`## Shared-box contention
(headline numbers in a quiet window)` → `## JIT prewarm`) → timing rules (`## The measurement budget:
milliseconds, not iterations` → `## Unseeded inputs, when control flow depends on input values`) →
comparing arms (`## Same-knob baseline` → `## Repeatability + measurement order`) → acceptance
(`## Gates`, `## Acceptance`) → is the number the one you meant (`## Every within-comparison stays green
when both arms run the wrong thing` → `## A result measured in isolation does not transfer to the
kernel`) → `## Red flags`.

---

## Toolchain pin

The run has **one** canonical toolchain pin (`records.md ## 1. Task Contract`): ROCm + Triton tag
(+ LLVM / commit hash for a custom build) + PyTorch + docker image + GPU arch. Two rules keep it honest:

- **Stamp every artifact.** Each results file / sub-report / A/B carries the env it was produced under
  (surface it in `final_report.environment`). A bare aggregate with no env stamp is not trustworthy — a
  kernel can be valid on one build and fail to lower on another (recovered async/shared layouts
  especially).
- **Drift consistency-check on every inherited input.** If any input — a prior result, an inherited
  champion, or a sub-report — was produced on a **different** pin (build / arch / LLVM / Triton tag)
  than the canonical one, **flag it and re-validate on the canonical pin before it counts**. Do not mix
  wins measured on different toolchains into one deliverable, and do not let the version naming of the
  deliverable drift from the pin. A path gated on a version (e.g. a feature available only at/after a
  tag) must be measured on **both** sides of the gate it claims to serve.

## Isolation (mandatory)

Build an isolated, reproducible harness and an immutable command contract **before any timing**.

- Separate baseline and candidate source roots; never let a candidate read a sibling. Always make both
  roots explicit.
- Separate `TRITON_CACHE_DIR` per variant (so IR dumps and JIT artifacts do not collide).
- Identical imports, shape stream, correctness oracle, and timing config across baseline and candidate.
- Keep a **plain-Triton comparator** alongside the Gluon candidates — it is the target line
  (`close.md`, `recover.md`).

Required checks (baseline isolation):

- baseline source root is explicit;
- candidate source root is explicit;
- the harness does not rely on implicit current-working-directory behavior;
- the script location does not silently change import precedence;
- baseline and candidate do not share ambiguous artifact or JIT cache state.

If you cannot state exactly which source tree is imported on both sides, the result is not
trustworthy. Minimum result metadata:

```text
baseline_source_root / candidate_source_root
baseline_module_files / candidate_module_files
baseline_cache_dir / candidate_cache_dir
triton_version / target_arch / timing_method
system_triton_preserved (framework opt-out, e.g. AITER_USE_SYSTEM_TRITON):
package_install_or_editable_state:
```

GEAK enforces part of this mechanically: `harness_lib.assert_legs_differ` refuses to
measure unless the two legs resolve the target callable to **different** code and the baseline leg
resolves **outside** the task dir (the baseline must be the live serving stack, never a copy the
optimizer can edit).

## Import boundary

Protect against accidental import drift:

- avoid harnesses whose script directory becomes the effective source root by accident;
- prefer explicit module loading or explicit source-root injection (the skeleton below);
- record the imported module path in benchmark output;
- keep the same import mechanism for baseline and candidate.

### Practical import skeleton

Load baseline and candidate through explicit roots so a candidate can never read a sibling:

```python
import importlib.util, pathlib, sys

def load_module(name: str, path: str):
    path = pathlib.Path(path).resolve()
    old = sys.path[:]
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(mod); return mod
    finally:
        sys.path[:] = old
```

Record `module.__file__` for each loaded wrapper/kernel and verify sibling imports resolve to the
intended root.

### Import-boundary troubleshooting

Stop-and-check triggers:

- baseline and candidate share a top-level package name;
- benchmark scripts live under one candidate tree;
- explicit file loading is mixed with package-name sibling imports;
- `sys.modules` may preserve stale modules across candidates;
- prebuilt artifacts or compiled extensions may be selected by import path.

Minimum record:

```text
baseline_source_root / candidate_source_root / top_level_package
baseline_module_files / candidate_module_files
baseline_sys_path_prefix / candidate_sys_path_prefix
PYTHONPATH / TRITON_CACHE_DIR / artifact_source
```

Checks: print `module.__file__` for the public wrapper and hot kernel module; verify sibling imports
inside explicitly loaded files; prefer separate short-lived processes when stale `sys.modules` can
survive (acceptance already runs one fresh process per leg); inject the source root explicitly before
package imports; keep separate `TRITON_CACHE_DIR`; run a same-code control after changing import
plumbing.

Common failure shape:

```text
load repo file by explicit path
file imports top_level_package.submodule
submodule resolves to installed package or the OTHER candidate root
benchmark now mixes source trees
```

Treat this as an import-boundary failure until a control run proves otherwise.

## Artifact and cache hygiene

Record and control: JIT cache location; AOT/prebuilt artifact location; generated temp output dir; any
environment variable that changes kernel selection. Do not compare a cold baseline against a warmed
candidate without recording the difference. When baseline and candidate use different
constexpr/config combos, prewarm both paths before timing and record the compiled config keys
(`## JIT prewarm`). Also isolate module import name, kernel function name (the compiler cache may key on
it), temp output dir, the Python process/worker pool, and `TRITON_CACHE_DIR`. If two candidates have the
same visible module/kernel name but different bodies, assume cache pollution until an explicit
cache/process control proves otherwise.

## Boundary declaration (no modes)

Declare the measured boundary explicitly — kernel-only, wrapper+kernel, or full operator — and label
every reported number with it. Do not compare a kernel-only candidate against a full-operator baseline.
There is no `optimization_mode`; the two-tier flow + the layer backbone replace the old mode machinery.
Which boundary is the right one is decided by how production serves the kernel
(`## Measurement basis + benchmark-artifact pitfalls`).

## Generate the harness

Use `scripts/create_harness.py` (defaulted to Gluon imports + a layout-factory skeleton). It generates
a per-case `GEAK_RESULT_LATENCY_MS=<float>` line timed with `harness_lib.time_op` and judged with
`harness_lib.correct()`, so a harness authored here measures the same way GEAK's acceptance does;
`scripts/parse_correctness.py` reads its result (unittest exit codes and per-case JSON). Modes:

```text
--correctness     validate against the reference oracle
--profile         single run for profiling (minimal allocations)
--benchmark       quick benchmark (fewer iters)
--full-benchmark  authoritative benchmark (more iters), emits a latency marker
```

Emit a parseable marker (e.g. `GEAK_RESULT_LATENCY_MS=<float>`) or JSON; document the exact stdout
markers in `COMMANDMENT.md`.

**Keep the PROFILING command separate from the CORRECTNESS command.** The app command you hand
`kernel_workflow/scripts/kernel_tools/capture.sh` after `--` is used for capture/timing AND for the
static ISA dump (`kernel_workflow/scripts/kernel_tools/dump_ir.sh` runs it verbatim to compile the
kernel). It must be a *compile/dispatch* command — do NOT append the correctness `--check` there. A
`--check` (or any flag the kernel's profiling driver does not accept) crashes the IR dump, which empties
the static op-mix and **silently drops the structural T0 directions (defuse-by-parallel-axis /
two-pass / packed-atomic — e.g. splitting a fused attention-bwd into dkdv + dq)** for any
reduction-heavy kernel. The correctness oracle command goes in its own slot: your harness's `--check`
command, run in-process by `ab_bench.py`'s oracle before any timing, which the pipeline runs separately
as the #1 gate. (`dump_ir.sh` also strips a stray `--check` / `--correctness` / `--backends <sel>` as a
safety net, but keep them out of the profiling command.) The same command string is what
`kernel_workflow/scripts/profile_kernel.sh <gpu_id> <cmd> <out>` receives (`profile.md`); it runs under
`gpu_lock.sh`, so never inline `HIP_VISIBLE_DEVICES` into it.

### Two optional stdout markers the config sweeper reads

The batched config sweep (the triton pack's `scripts/plain_autotune.py`) reads two OPTIONAL markers off
the driver's own stdout. Omit both and the sweep still works — it just pays one driver start for the
oracle plus one for the clock at every point, and a winner found by a one-knob probe stays uncertified.
Each is a couple of lines in the driver:

| marker | what printing it buys |
| --- | --- |
| `err=<float>` on the **timing** run, next to the latency marker | the correctness verdict is carried by the same execution that produced the number, so the separate oracle start per point disappears (~2x fewer driver startups). This strengthens the gate rather than relaxing it: the run that was judged IS the run that was timed |
| `PA_EFFECTIVE_CONFIG={...}` — the config as **finally resolved**, after the kernel's own defaults and any `select_config()` | a winner found off the grid (a one-knob probe) can be completed into a full grid point and certified as a local optimum. Without it, that winner is reported honestly as an uncertified order 0 |

Print `PA_EFFECTIVE_CONFIG` once per **distinct** resolved config. A driver that walks several shapes in
one run and resolves a different config per shape must print each: the sweeper reads several differing
reports as *no* report rather than attribute the last shape's config to a latency reduced over all of
them. `scripts/create_harness.py` generates both markers, with a `DEFAULT_CONFIG` dict to fill in with
the kernel's own defaults for every sweepable knob.

## COMMANDMENT.md (immutable evaluation contract)

A single source of truth that **must not be edited to make a candidate look faster**. Minimum:

```markdown
# COMMANDMENT
## Setup          ```bash ... ```
## Correctness    ```bash ... ```   # oracle + tolerance
## Benchmark      ```bash ... ```   # quick
## Full Benchmark ```bash ... ```   # authoritative, per-shape
## Profile        ```bash ... ```   # rocprofv3 / ATT compatible
```

A candidate that changes the boundary, oracle, shapes, or markers is rejected (`close.md`). In GEAK
the task contract (the frozen oracle and `unittest.py`) plays this role; do not author a
second one that disagrees with it.

## Harness stability

The same benchmark must keep identical shape ordering, warmup/repetition counts, benchmark boundary,
correctness oracle, and output schema. If any change, treat the result as a new benchmark, not a new
candidate. If the contracted harness is unstable for sub-ms paths, record both contracted and
isolated-harness results plus the reason for the primary metric rather than silently switching metrics.

## JSON schema (aligned to two-baseline + budget + Round Ledger)

The harness JSON shares fields with `records.md` (Round Ledger), `budget/<round>.json`, the profile
JSON, and `final_report.json`:

```text
benchmark_boundary / target_arch / workload_class
shape_stream_id / warmup / reps / case_count
timing_method / cache_condition / statistic          # harness_lib receipt: timer, read-evict, median
plain_target_metrics : per_case[], geomean_latency_ms      # the target line
gluon_anchor_metrics : per_case[], delta_vs_plain          # the working baseline
cases[]:
  case_name / shape_fields
  correctness_status
  latency: median (acceptance) / mean / min / max
  primed / host_ms                                   # harness_lib host-bound receipt
  vs_plain_target / vs_gluon_anchor
  budget:  { ideal, as_built, gap }
  profile: { mfma_eff, ds_read_interval, achieved_bw, vgpr, spill, occupancy }
  compiler_contract_active: base | +scheduling | +agpr_hint | +authored_pass
  ir_signal
  selected_path / feature_path / selected_config / requested_config
  module_files / cache_dir
aggregate_metrics / best_case / worst_case / repeat_summary
acceptance_decision / dispatch_or_fallback_summary
source_root / artifact_or_cache_info
```

Record every search knob even when the final candidate removes it. Break correctness down by feature
path / dispatch bucket / shape class when a failure may be local.

## Gluon smoke harness

Gluon smoke tests live in a `.py` file, not `python -c` / stdin / `exec`. Record `smoke_file /
triton_version / rocm_version / target_arch / import_path / layout_used / compile_status /
correctness_status / cache_dir`. Smoke code + probe order: `../gluon/gfx950-minimal-examples.md`
(`--arch gfx950` first; gfx942 is the downgrade check).

## Profiler output hygiene (do not pollute the repo)

`rocprofv3` / `rocprof-compute` write large artifacts and can leave **root-owned** dirs. The profiler
entry is `kernel_workflow/scripts/profile_kernel.sh <gpu_id> <cmd> <out>` (wrappers in
`kernel_workflow/scripts/kernel_tools/`: `capture.sh`, `rocprofv3_safe.sh`, `rocprof_compute_probe.sh`),
run under `gpu_lock.sh`:

- **Output dir OUTSIDE the repo.** Point `<out>` / `-d` / the workload dir at a clean scratch path (e.g.
  `/tmp/prof_<run>` or a run dir outside any source tree), never the current repo. With the CWD inside a
  repo, `rocprofv3` drops a **root-owned `<host>/*.db` dir + `gpucore.*`** that a non-root user then
  cannot delete (only removable from inside the container). Keep profiler output off the source root.
- **Force CSV, not SQLite.** `rocprofv3` defaults to a SQLite `.db`; pass `-f csv` /
  `--output-format csv` so the pipeline parsers can read it (and you don't accumulate opaque `.db`
  files). `profile_kernel.sh` already defaults `RPV3_TRACE_ARGS` to `--kernel-trace --stats
  --output-format csv`.
- **Config by env-var path, not argv.** A JSON config passed as an argv token through
  `docker exec bash -lc '…'` can be re-split and fail with rc=4 / `JSONDecodeError`; pass the config via
  an **env-var pointing at a file path** instead.
- **`gpucore.*` core dumps.** A profiled crash can dump `gpucore.*` into the CWD; run from a scratch dir
  and move them aside out of the tree (they are large; scripts GEAK roles may run never `rm`).

---

## Shared-box contention (headline numbers in a quiet window)

On a shared GPU box, concurrent agents/jobs can inflate absolute `us` by ~10%. So: use **same-session
relative A/B deltas** for keep/revert decisions (both sides measured back to back — in acceptance, the
interleaved B,C pairs of `measure_legs`), and **re-measure any headline/absolute number in a quiet
window**. Prefer the clock-insensitive counters (`profile.md ## 3.1 Required evidence — the four dials,
every round`) as the A/B discriminator on a drifting/contended box. Contention noise is **one-sided** —
it only ever adds time — so in **search/screening (ab_bench)** the **min** over repeats of an
interleaved cell is a useful contention-free floor, reported beside the median; the acceptance
statistic stays the `harness_lib` median, and a large min/median gap is itself the signal to re-measure
in a quiet window (`## Repeatability + measurement order`).

### Device identity: a wrong index returns a plausible number, never an error

All of the above assumes you know **which device** produced the number, and on a shared multi-GPU node
that is not a given. It fails silently in both directions — a wrong device selection raises nothing and
returns a number of the right order — which makes it the `## Every within-comparison stays green when
both arms run the wrong thing` failure arriving through the hardware instead of through the source tree.
Six findings, all from one measured instance on an 8-GPU node:

- **Three index namespaces, disagreeing on every device.** A vendor SMI tool's `cardN`, the kernel's
  `/sys/class/drm/cardN` *name*, and the runtime's visible-device ordinal are three different
  namespaces, and in that instance they disagreed on **all 8** indices. The SMI tool enumerated in
  **PCI-bus order**, while the runtime ordinal tracked the **ascending rank** of the drm entries rather
  than their names — and the lowest-numbered drm entry was a **display adapter absent from the GPU list
  entirely**, so the two sequences are not even the same length. **Anchor every per-device claim to a
  UUID or a PCI bus address, and record the anchor in the artifact**, not only on the command line: an
  index stored without its namespace cannot be re-derived afterwards. (GEAK's `scripts/gpu_identity.py`
  resolves the ISA / product identity of the visible agent from `rocminfo`; record its output beside the
  UUID/PCI anchor.)
- **A visible-devices environment variable applied to an already-constrained context is an override,
  not a sub-selection.** It re-indexes against the visible set instead of narrowing it, so selecting the
  first ordinal silently moves the work onto the first *visible* device. A whole round of numbers was
  produced against the wrong physical device this way. Establish whether the context you are entering is
  already constrained **before** adding a constraint to it. This is why device selection belongs to
  `kernel_workflow/scripts/gpu_lock.sh` (it pins the device for the command it runs) and never to an
  inline `HIP_VISIBLE_DEVICES` in a timing or profiler command; `kernel_tools/rocprofv3_safe.sh` unsets
  the visibility variables only when HIP and ROCR are **both** set, because unsetting one alone shifts
  GPU numbering.
- **A `GPU(s)` column beside a PID list may be a count, not an index.** In that instance four processes
  each showed `1`; read as an index that says "all four sit on one device and the rest are free", which
  would have taken a neighbour's devices. The data was right and the column beside it was unreadable.
  **Cross-check any such column against per-device utilisation before acting on it.**
- **A PID→device attribution assembled from two separate tool invocations is a race, not a broken index
  map.** A foreign campaign's PIDs churned completely between two calls 60 s apart, and the failure to
  reconcile them looks exactly like a namespace bug — which sends you debugging the wrong layer.
  **Collect both sides in one pass.**
- **Exclusivity needs two orthogonal guards, because the failure populations are complementary.** Memory
  availability and compute utilisation answer different questions: **compute contention corrupts a
  measurement; memory availability only decides whether you can start.** A memory-side gate passed a
  device sitting at **97%** utilisation held by a foreign job (free/total 0.96); the mirror case — 0%
  utilisation with a large allocation still held — makes a utilisation-only gate refuse a device that is
  actually quiet. Both populations were live in a single census: several devices near-0% compute with
  **~92%** of memory held, the rest at **97–100%** compute with memory nearly free. **A memory-only guard
  passes exactly the devices that will corrupt timing and fails exactly the quiet ones.** Require both,
  and fail closed on either.
- **A utilisation census flags; it never licenses.** A positive read detects a tenant; a **negative read
  does not exclude one.** A job whose entire run took ~3 s was invisible to a census sampled every 3–4 s,
  and that census read the device at **0% while the job was running**. So **state the sampling cadence
  beside any "the device was clean" claim**, and treat cleanliness as **not establishable
  retroactively** — a census taken after the fact cannot speak for the window the measurement occupied.

## Shared / DVFS GPU timing

**Problem.** On a shared or idle gfx950 the clock idles well below peak and light multi-kernel /
torch-eager stages do not hold a ramped clock, so per-stage `do_bench` and coarse e2e timing swing
run-to-run. Absolute TFLOP/s drifts enough that layer-loop A/Bs land in the noise and a single favorable
sample reads as a win. Clock-locking is not always available (`rocm-smi --setperflevel high` can be "Not
supported" on partitioned cards), so you cannot simply pin the clock. Treat absolute time as untrusted
here and lean on clock-insensitive counters (`profile.md`, "Accuracy problems to distrust").

**Recipe (use when absolute time is the only signal):**

0. **Gate on a sustained idle window**: before measuring, require the GPU to be idle for a sustained
   window (poll the device-busy counter; demand several consecutive idle seconds), not a one-shot idle
   check — neighbors on a shared box start mid-run. A single backgrounded long run should self-gate so it
   does not measure under contention. The idle-gate mechanism is **environment-specific** (no helper is
   shipped; `kernel_workflow/scripts/gpu_lock.sh` only serializes access, it does not wait for idle) —
   implement the poll against whatever device-busy signal the box exposes, and apply both guards from
   `### Device identity` (compute utilisation **and** memory).
1. **Ramp + hold the clock**: run a sustained warmup matmul long enough to lift the clock and keep it
   ramped through the timed region (not a single short kernel). In screening, `ab_bench.py --preload SEC`
   burns real compute before the first timed cell: an idle card spends its first window on a decaying
   clock excursion that lands on whichever arm holds the early slot (measured: the same empty kernel at
   **10.7 µs in slot 1 and 6.06 µs in slot 5**; **3 s** of preload removed it **6/6**). Distinct from
   `--warmup`, which warms the *variant* rather than the card.
2. **Device-time anchor**: use a CUDA/HIP-graph device-time measurement (`harness_lib.time_op(...,
   graph=True)`, `do_bench_cudagraph` / graph replay) as the stable number, separate from
   Python-dispatch wall time.
3. **Interleaved median-ratio**: time plain and candidate **interleaved per sample** and report the
   **median of per-sample ratios** — this cancels common-mode clock drift that a separate-runs mean does
   not (the acceptance form is `measure_legs`' interleaved B,C pairs).
4. **Two-method agreement**: require two independent methods (e.g. device-time anchor + interleaved
   median-ratio) to agree before quoting a number; never quote a single coarse sample.

When even this is shaky, fall back to the **clock-insensitive discriminator**: compare cycle /
utilization counters (MFMA efficiency, VALU busy, mem-unit-stall — read the busy counters `VALUBusy` /
`MfmaUtil`, not duty-cycle `VALUUtilization`) instead of absolute time (`profile.md`).

## JIT prewarm

When baseline and candidate use different constexpr/launch combos they may compile different kernels.
Before timing: enumerate every timed config key; run one untimed compile/warm for baseline + candidate;
use separate cache dirs; isolate/clear the candidate cache (move it aside) after kernel-body /
target-sensitive lowering / layout-version changes; record prewarmed configs; keep JIT cost out of timed
reps unless compilation is part of the boundary. Discard a win that only came from a cold baseline or a
surviving stale artifact. (Cold-compile outliers that survive warmup: `## Repeatability + measurement
order`.)

---

## The measurement budget: milliseconds, not iterations

*Screening protocol (ab_bench / plain_autotune / the generated harness's quick mode). Acceptance sample
counts are the harness's `repeats` of single-launch, read-evicted event samples, widened by
`measure_legs` pairs when the verdict is undecided.*

A fixed iteration count spends the same effort on a 16 µs kernel and a 400 µs one, so the short kernel
is timed for about a millisecond and its reading is mostly whatever else the box did. Set the budget in
**time** and derive the count from the kernel's own measured duration — five probe launches, then
`n_repeat = budget_ms / estimate`. This is what `triton.testing.do_bench` does, and it is what
`@triton.autotune` races with.

**Measure your own meter before blaming the box.** Read the *same* config several times under each
protocol and compare the spreads — a fixed count against a time budget. Expect the short-kernel end of
your workload to be where they diverge most, and expect a meaningful part of the "noise" a sweep fights
to be the meter rather than the machine (measured in `ab_bench.py`: **13.9 %** spread at 60 fixed
iterations vs **0.8 %** at a 100 ms budget on a 16 µs kernel). That matters beyond tidiness: the spread
is the floor on what the search can resolve, so a loose meter is paid for twice — once in a wider noise
band, and again in the extra readings the race needs to out-vote it.

```
# one config, N readings, each way -- then compare the spreads
for i in $(seq 7); do BENCH_BUDGET_MS=0  BENCH_ITERS=60 <driver>; done
for i in $(seq 7); do BENCH_BUDGET_MS=100             <driver>; done
```

`ab_bench.py --budget-ms 100` (default), the generated harness's `BENCH_BUDGET_MS` and
`plain_autotune.py --bench-budget-ms` implement the budget; `--budget-ms 0` restores a fixed count when
you need to reproduce an old number exactly. Record which one produced every number: a reading under a
different meter is a different measurement, not a noisier one.

## Sub-ms Measurement Rules (also the gate's launch-floor evidence)

For very short latencies, use stronger repetition and treat small percentage wins as provisional until
repeated. This table is **screening guidance** (warmup / reps for quick and search runs; the noise
column is a prior until this window's band is measured) and doubles as the **escalation gate's
launch-floor evidence**: if all contracted shapes are `<=30-50us` and latency barely scales with
compute, stay plain (`entry.md`, escalation gate).

| Observed latency | Warmup | Reps | Treat deltas inside as noise until repeated |
| --- | --- | --- | --- |
| `< 0.05 ms` | `>= 30` | `>= 200` | `~8%` |
| `0.05` to `< 0.1 ms` | `>= 20` | `>= 100` | `~5%` |
| `0.1` to `< 1 ms` | `>= 10` | `>= 50` | `~3%` |
| `>= 1 ms` | `>= 5` | `>= 20` | `~2%` |

No row lowers the commit gate: the keep threshold is `max(MIN_IMPROVE = 2 %, the row's band, the
measured band)`. Alternate measurement order, rerun near-noise results, run same-code/baseline
controls, isolate noisy buckets, record timing method. Judge by repeat-stable A/B movement, not one best
absolute latency. Below ~a few µs, raise `time_op`'s `inner` (per-window overhead divides away) and
apply the instrument floor (`## The largest additive term is usually your own instrument, and you can
measure it`).

## Hot or cold cache: pick it from the bound class, because it can REVERSE a ranking

A back-to-back timing loop leaves the working set resident. A server does not: a decode kernel's weights
are evicted between steps, so it reads cold from HBM every time. The difference is not a scale factor you
can divide out:

On a memory-bound kernel the two protocols can select **different winning configs**, and the worst case
is not a mis-scaled number — it is a **reversed verdict**: timed hot, a kernel can look already optimal
and the sweep correctly reports "nothing to win", while timed the way it is deployed the same space
contains a real win, sitting in a config the hot protocol ranks *below* the default (measured: hot
prefers the tuned config on a per-token quant kernel, cold prefers the shipped one). The mechanism is
that a hot loop pays nothing for re-reading, so it systematically rewards the configs that re-read the
most — exactly the choice a cold deployment punishes. A hot-cache sweep of a memory-bound kernel does not
merely mis-measure it; it can hide the optimisation entirely.

**Acceptance is always cold (read-evict).** `harness_lib.time_op` read-evicts before every sample and
cannot be switched off, so every acceptance number is a cold number; for a compute-bound kernel with
real reuse this costs little, because the reuse that matters happens *within* a launch and survives the
eviction. The bound-class choice below is a **screening** decision:

**This is falsifiable on your own kernel, and it is worth the two runs** whenever the bound class is
memory or HBM: sweep the space twice, once under each protocol, and compare *which config wins* — not
just the times. If the winners differ, the protocol is a decision your result depends on, and it belongs
in the record next to the winner (`plain_autotune.py` writes `meter{}` for this reason). The rule:

- **memory-bound / HBM-bound (C2's binding class), and anything decode-shaped** → time **cold**
  (`ab_bench.py --cache cold`, harness `BENCH_CACHE=cold`, `plain_autotune.py --cache cold` — or
  `--cache auto` with the measured `--bound`, which picks it from the bound class instead of from
  whoever last exported an env var): **read**-evict a buffer larger than the last-level cache before
  **every** sample (the `harness_lib.flush_cache` method; a write-evict inflates the result through
  writeback contention), and time **one launch per sample** — a burst re-warms the cache on iteration
  two and measures a hot kernel with extra steps. This is also the protocol with the **largest fixed
  per-sample cost** on this page, so measure that cost before reading any absolute off it
  (`## The largest additive term is usually your own instrument, and you can measure it`).
- **compute-bound with real reuse** → **search/screening only (ab_bench)**: time **hot** if you want
  the ranking the resident working set gives; confirm the winner under the cold acceptance protocol.
- Either way, **say which one the number came from**. A cold number and a hot number are two different
  measurements of two different deployments, and comparing across them is the same error as comparing
  across harnesses.

**A protocol is validated at a granularity, and changing the granularity re-opens the question.**
Flushing before every sample is correct for the **whole op**, because a whole op is what the server
re-reads cold. Applied unchanged to per-component timing it is not, and the reason is mechanical: **a
protocol that pays a fixed cost once per sample pays it once per component**, so the cold-miss cost is
charged as many times as you chose to split the op. The parts then sum past the whole; a component that
touches almost nothing is billed for a cold miss it never pays in production; and subtracting the parts
from the total yields a *negative* "fixed overhead". **That sign is a symptom, not a diagnosis** — it is
not the tell it reads as, because a second cause produces the identical sign: components timed in
**isolation** cannot overlap, while consecutive work inside the real graph does, so isolated capture
returns the serialized cost of work that partly ran concurrently and the parts sum past the whole with
no measurement error at all. A negative implied overhead therefore says only that the decomposition and
the total disagree, not why. The discriminator you can actually check is **how the gap scales**: a
once-per-sample fixed cost scales with the **number of splits** — re-cut the same op more finely and the
gap grows with the component count — while overlap loss scales with **how much adjacent work actually
overlaps** and does not move when you merely re-cut the same work at the same boundaries of concurrency.
Neither separates cleanly when both are present, and the three-case reading of the resulting inequality
is in `## Every within-comparison stays green when both arms run the wrong thing`. So **re-argue any
per-sample protocol when you change granularity** — whole op → sub-kernel, whole step → single launch —
paying particular attention to anything with a once-per-sample fixed cost, because that is the term that
multiplies. The free check on the result is the absolute-value invariant in that same section; note that
every ratio in the run stays clean while this is happening.

## ROCm timing

On ROCm/HIP, event timing can be wrong for short or heavily cached paths, and it degrades silently into
host timing when the segment is host-bound (`## Measurement basis + benchmark-artifact pitfalls`). Do
not trust a single event reading when the result is surprising, sub-ms, or contradicts
wall-clock/profiler. The acceptance anchor stays **event device time with per-sample sync**
(`harness_lib.time_op`) — read its `primed` / `host_ms` receipt first: `primed=False` means the number is
a host latency, and the remedy is `inner > 1` or the graph boundary, not a different timer. A
**batched wall-clock** loop is a **diagnostic cross-check only**, never the acceptance anchor (it was
rejected on measurement — see the binding table — and on decode shapes its floor is Python dispatch):

```python
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(batch):
    kernel_call()
torch.cuda.synchronize()
per_call_ms = (time.perf_counter() - t0) * 1000 / batch
```

Pick `batch` so total timed duration is well above timer noise; keep alloc/randomization/validation
outside the timed loop unless part of the boundary; record which timer produced the accepted number; if
event and wall-clock disagree, treat as untrusted until a control rerun explains it.

## Measurement basis + benchmark-artifact pitfalls

- **Match the benchmark boundary to production first.** Before picking a metric, declare how the kernel
  is **served in production** — eager (per-call host launch) or under a captured **CUDA graph** (launch
  overhead amortized) — and measure at that boundary (`harness_lib.time_op(..., graph=True)` for the
  graph-served case). The boundary decides which wins are real: a launch-reducing change (fewer kernels,
  fused dispatch, a per-tile recompute folded into one launch) can win in **eager** yet be **inert or
  negative under a CUDA graph** where the host launch is already amortized away (the **launch-fusion
  law**, `../hardware/roofline-models.md ## Kernel time decomposition`). Never accept an eager-measured
  launch win for a graph-served kernel, or vice versa.
- **Decompose wall time into wrapper / host / kernel** before deciding the target (`## Wrapper overhead
  breakdown`): `full_operator = host_dispatch + wrapper/alloc + kernel`. If host/wrapper dominates and
  production is graph-served, the kernel body is **not** the lever; if production is eager, host/launch
  reduction is a legitimate (eager-only) lever.
- **Event timing degrades into host timing when the segment is host-bound, and it does so silently.** A
  pair of stream events measures the interval between the two events *on the stream*, which includes any
  stretch the stream spent idle waiting for the host to enqueue the next thing. Nothing errors, nothing
  warns, and the number keeps its "device time" label (`harness_lib`'s `primed` receipt exists for exactly
  this). **The free discriminator is that host cost does not scale with problem size:** sweep the shape
  over a decent range and check whether the segment's time tracks it. A segment whose time is flat, or
  nearly flat, against a shape sweep is reporting **issue** cost, not execution cost — and no device-side
  optimization can move it. Run this before decomposing wall time, because it decides whether the
  decomposition's "kernel" column means anything.
- **Then check that the discriminator itself is not broken.** The bullet above teaches the
  discriminator; these three checks say when to distrust it, and they need no model of the kernel at all
  — only the arithmetic of what contains what. Time the same work three ways: **issue-only** (N calls, no
  sync inside, wall/N), **stream events**, and **wall with one trailing sync** (these three are
  diagnostic timings, not acceptance timings). Then require:
  - `issue_only <= wall` **and** `device_events <= wall`, because the wall contains both. Neither can be
    violated by a correct measurement. In one measured instance a published table carried `device 283.8
    µs` against `wall 277.2 µs` and nobody looked; the re-runs were self-consistent, so the method was
    sound and that one run was not.
  - **a host-free variant must never be *slower* than eager.** If it is, you have not removed the host
    cost, you have relabelled it. In one measured instance a graph control captured **one** launch and
    replayed it N times — which removes nothing, since that is still one host call per iteration — read
    slower than eager on every arm and was about to be published as a host-boundness signal that does
    not exist, retracting a table that was correct. **A retraction driven by a broken instrument looks
    like diligence and is not.**
  - **a genuine host cost is constant in problem size.** Check it at two sizes and confirm the *absolute*
    host number barely moves — the discriminator above, applied to the whole decomposition rather than
    to one segment. A useful threshold: when issue-only exceeds roughly 0.8 of the synced wall, you are
    launch-bound and the lever is fewer dispatches.
- **Price a dispatch merge by differencing the two configurations, not by dividing a total by a launch
  count.** A per-launch average is an average over kernels with different signatures, and the survivor
  of a merge is the expensive one: it carries the arguments of both, and argument marshalling is host
  work. In one measured instance the per-launch figure was ~16.5 µs while merging two metadata kernels
  bought **9.1 µs**, reproducibly, at both ends of a 17x shape range. **Savings from merging are
  sublinear in launches removed**, so a `per_launch × launches_removed` estimate is an over-prediction,
  not a bound.
- **Graph-timing method.** If you time graph replays with a **wall clock**, time an **amortized batch of
  replays** (one `synchronize` around many replays), not a `synchronize` per replay — a per-replay sync
  re-introduces a fixed host/sync cost per op that inflates each kernel's time and compresses the
  measured speedup (a measurement artifact, not a kernel cost). The acceptance method
  (`time_op(graph=True)`) avoids that artifact differently: it reads **events** around each replay, so
  the per-sample sync and the read-evict sit outside the timed window; to amortize the per-window event
  overhead on a tiny kernel, capture `inner > 1` launches into the graph rather than batching samples.
  Cross-check with a graph-capable device-time bench (`do_bench_cudagraph`) and require two methods to
  agree (`## Shared / DVFS GPU timing`).
- **Kernel-body acceptance metric = kernel-only time under a CUDA graph** (when that is the production
  boundary). For kernel-internal optimization, time the kernel(s) under a captured CUDA graph (host
  launch overhead amortized) and do NOT chase host/launch overhead as a kernel-opt target. The ONE
  host-side lever in scope: when explicit host-side framework (torch) preprocessing ops can be
  rewritten/fused INTO the kernel **without changing the public function interface** (e.g. index a
  non-contiguous tensor by its stride in-kernel instead of a host `.contiguous()` copy; pull a
  per-element dequant/scale into the kernel), do that fusion — it removes the op from both the host path
  and the timed window. Guard: must not change the signature/return contract.
- **Beware harness-reuse / memoization artifacts.** An optimization that exploits the harness reusing the
  SAME input tensors across timed runs (e.g. memoizing preprocessing keyed on tensor identity) can flatter
  the score while doing no real work, and can be silently STALE when preprocessing makes a
  data-dependent copy. Validate on FRESH inputs and against the correctness oracle, not only the
  reuse-loop score (`harness_lib.check_correct_multi` keeps distinct-input outputs live and catches a
  shared `static_out`; `check_random_vs_baseline` re-draws values against the live baseline). (If a CUDA
  graph is also the deployment vehicle, capture the WHOLE forward — preprocess + kernels — so replay
  re-derives data-dependent copies from live buffers; a launch-only/memo shortcut is stale for
  copy-preprocessing. Valid under the persistent-buffer in-place contract.)

## Wrapper overhead breakdown

For sub-ms or wrapper-heavy operators, split full-operator and kernel-only timing before deciding on
body-level work:

```text
full_operator_us / kernel_only_us
allocation_or_fill_us / wrapper_dispatch_us / wrapper_fraction
```

Same inputs/allocations/stream/warmup/reps for each. If wrapper overhead is a large fraction, the
bottleneck is owned by wrapper/alloc/dispatch -> stay plain (`front-end.md`, `entry.md` escalation
gate), not Gluon body work. Run this before body work when the wrapper does dynamic routing, per-expert
loops, small metadata GPU ops, per-call allocations, or config/import selection. (`harness_lib`'s
`primed=False` / a large wall ≫ device gap is the same signal from the acceptance timer.)

### CUDA/HIP graph diagnostic

For multi-kernel sub-ms operators, graph replay separates Python dispatch from device work:

```text
normal_call_us / graph_replay_us / python_dispatch_fraction
graph_supported_shapes / graph_regression_shapes
```

If graph replay removes a large fraction of latency, body-level tuning is capped by graph replay time —
treat fusion / allocation-caching / boundary change as separate directions. (A graph wrapper never wins
acceptance by itself: device time already excludes the dispatch it collapses.)

## The largest additive term is usually your own instrument, and you can measure it

`### The additive case: the control gets greener as the measurement gets dirtier` (below) treats an
additive term as contamination — something to detect from a magnitude range because you cannot see it
directly. **One additive term is neither contamination nor unknown, and it is normally the biggest one on
the page:** the fixed per-sample cost of the protocol itself — the launch, the graph or replay boundary,
the cache flush that `## Hot or cold cache: pick it from the bound class, because it can REVERSE a
ranking` requires before **every** cold sample. It is measurable in five lines, and almost nobody
measures it, because a timer looks like ground truth.

> **Put an *empty* kernel — one program, immediate return — through the same timer, the same protocol
> and the same boundary you are about to time the real kernel with. That reading is the floor `F`. It is
> a property of your protocol, not of the box and not of the kernel.**

It is not small, and it moves with the protocol rather than with the work. In one measured instance on a
CDNA4 part, `F` was **13.7 µs** under the cold protocol (large flush, one launch per sample) and **2.7 µs**
warm, and the cold figure was identical across a 7x sweep of the batch dimension. Four small bookkeeping
kernels in that campaign read **13.9–14.2 µs** — all four readings were the floor, and the kernel being
credited with ~14 µs actually cost about **0.5 µs**. (The acceptance timer has the same kind of term:
`time_op`'s per-window overhead is "a few µs beyond the kernel", negligible for a 100 µs GEMM and most of
the number for a 5 µs op.) Three consequences, and the second is not a precision issue:

- **An absolute within about `3F` of the floor cannot be quoted as a ratio.** The floor is common to both
  arms and additive, so it compresses the ratio toward 1 exactly as the additive-case section describes:
  every win measured inside that band is a **lower bound**. Quote `(t - F)` ratios, or quote nothing. In
  that instance a reported 1.0380 was 1.0453 once corrected.
- **It changes verdicts, not only digits — and the direction is toward false ceilings.** A
  streaming-bandwidth census in that campaign read **2.0–2.9 TB/s**, was written up as "already at this
  box's cold ceiling", and sent the round to a data-layout rewrite as the only way out. Floor-corrected
  the same data reads **3.39–3.46 TB/s**: 35% of headroom remained, and the binding quantity was the
  launch grid rather than the layout. A purely additive constant turned "at the ceiling" into "a third
  short of it".
- **A graph or replay boundary does not remove it.** The rule that a launch/host-work reduction is inert
  under a captured graph is about the *host* cost, and it is easy to read as "inside the graph the kernel
  time is clean". Under a cold protocol a per-sample additive constant survives inside that boundary,
  and it is large enough to swallow an entire prologue kernel.

**Validate the correction; do not merely announce it.** A correction that is supposed to move absolute
values without moving any conclusion is testable, and the test is free whenever two conditions in the
data should agree: pick two that were measured under different work but matched on the quantity that
ought to make them equal, and check whether subtracting `F` brings them together. In the instance above,
two footprints matched on occupancy disagreed by **33%** raw and **9%** corrected — the correction pulled
together two curves it had no obligation to. **Saying "I subtracted the floor" is an argument; the
convergence is the evidence.**

**And put the `< 3F` check in the path that produces the numbers, not in the prose.** The moment it has
to stop is the moment the table is already computed and the bandwidths are already written down — in
that campaign the person who wrote the rule quoted a whole column from inside their own forbidden band
one round later, and only noticed afterwards. A check that fires after the write-up is a check that has
already lost.

## Unseeded inputs, when control flow depends on input values

Separate from the reuse hazard above, and pointing the other way: for **timing** comparisons, arms must
share inputs, and on some kernels an unseeded harness will swamp everything you are measuring.

**The signature.** The kernel's work depends on input *values*, not only on shapes — routing or gating,
top-k selection, sorting, sparsity, an early exit, a data-dependent trip count. Then two arms handed
independently drawn inputs are not running the same workload: the draw decides how the work is balanced,
and the arms do structurally different amounts of it.

**Observed on a gated kernel whose routing weights decide how work is balanced:** arms compiling to
**byte-identical machine code** scattered over a band wide enough to swamp every effect under test, and
the unseeded configuration sat systematically above the seeded one at the headline shape. Neither spread
is a property of any arm; both are the draw.

**The fix is not "set a seed and rerun".** Four parts, all load-bearing:

1. **Seed before tensor construction, and replicate the harness's construction order exactly** — RNG
   consumption is positional, so building the same tensors in a different order gives different data
   from the same seed.
2. **Share the tensor objects across arms within a comparison cell** — the same device memory, not "the
   same seed, each re-drawn". This is the in-process (screening) form. Across the fresh processes of an
   acceptance leg the equivalent is part 1 done exactly — same seed, same construction order in every
   leg — which is how `harness_lib` keeps legs on identical inputs ("same seed => same inputs", e.g.
   `baseline_random_outputs`).
3. **Interleave arms palindromically within each round** (`A,B,C,…,C,B,A`) so warm-up and neighbour
   drift cancel in the paired difference.
4. **Report the median of per-round paired deltas**, not the difference of two run means.

**Then quantify the floor the same way you quantify any other:** benchmark a byte-identical clone of one
arm against itself through the whole apparatus. Expect the paired floor to land one to two orders of
magnitude below the unseeded band — if it does not, the seeding is not yet complete.

> **The absolute number stays seed-conditional even after all of this.** Only the paired delta is
> portable — so report the delta as the result and the absolute time as context, not the reverse.

---

## Same-knob baseline

When tuning launch/schedule/harness-visible knobs (block/tile size, split factor,
`num_warps`/`num_stages`/`waves_per_eu`/`num_ctas`, chunk size, path-selection env vars, autotune
config), compare against a baseline under the same knob conditions whenever the baseline can legally use
them. Do not compare "candidate with tuned knob" against "baseline with default knob" and call the delta a
win. If a knob is candidate-only, record why the baseline cannot use it. Keep selected knobs in the JSON.
(On the Gluon path `num_stages` is consumed by no pass in Triton 3.8.0 — it is recorded, not tuned; a tile
retune is a `resweep_request` in the deep_engineer's result, which tech_lead hands to GEAK's plain tuning.)

**No stored-baseline cross-run compare.** Never compare a freshly measured candidate against a baseline
*number* recorded in an earlier run/session (a saved JSON, a prior table, a previous box state).
Clock/DVFS/occupancy state drifts between sessions, so the mixed delta is an artifact, not a kernel
effect. Re-measure the baseline in the **same session, interleaved** with the candidate (`measure_legs`
does exactly this; see `## Repeatability + measurement order`). A stored historical number may be shown
for context but must be labelled **indicative only**, never used as the acceptance baseline.

## Tuned config audit

When the repo already has tuned configs / JSON tables / specialized config paths, do not let a candidate
override them blindly. Per case record `is_tuned / selected_config_source / effective_config /
requested_config / selected_bucket / candidate_override_applied / override_scope`. Compare tuned-path
shapes against their tuned baseline, not only a default config. A candidate that wins only by disabling
tuned configs is a dispatch/config-policy change requiring full-stream validation.

## Cross-version ratios: attribute the move before claiming it

When the same candidate is measured against the same baseline across several toolchain versions, a ratio
that improves from version to version is **not** evidence the candidate got better. The denominator
moves too, and it can move more than the numerator. Before reporting any cross-version trend, print
baseline and candidate **absolute** times per version and say which side actually moved. A candidate
that is flat in absolute ms while the ratio climbs means the baseline is degrading — report it that way,
because the kernel-level claim ("this candidate gains with newer versions") would be false.

Rules that make the attribution mechanical:

- **Every version is scored against its own in-version baseline**, re-measured in the same session as the
  candidate. Never carry one version's baseline number across to another.
- **Flag on divergence, not on absolute baseline drift.** Normalise both arms to the version where each is
  fastest; flag the cell only when the baseline's factor exceeds the candidate's by a margin (a few
  percent is enough). A small kernel often has both arms drift together across versions — the ratio is
  then undistorted and needs no flag, whereas flagging on baseline drift alone fires on that noise and
  produces nonsense attribution shares.
- **Carry an un-pipelined third arm** (the same **plain** baseline source with the schedule knob
  neutralised, e.g. `num_stages=1`; on a Gluon arm `num_stages` moves nothing in 3.8.0, so the third arm
  is always the plain source). It separates the two failure modes that look identical in a two-arm
  ratio: `baseline / third_arm` measures whether *this version's* auto-pipeliner is a net win or a net
  loss on this kernel, while the third arm's own cross-version drift measures whether the compiler's
  non-pipelined floor is stable. Only when both are stable can `candidate / baseline` be read as the
  candidate's own result. A shipped config that is merely mistuned (a bad knob, roughly constant across
  versions) is a different finding from a pipeliner whose output regresses version over version, and the
  third arm tells them apart.
- **Baseline degradation is a compiler codegen/expressibility finding, and must not be booked as a
  candidate win.** Say so in the cell and in the prose; a reader who quotes the ratio alone would
  otherwise credit the candidate for the compiler's regression.
- **The reference arm may not exist on every version.** A baseline arm can fail to build on an older
  version (a resource limit such as shared memory is the common case) while building fine on newer ones.
  Then the ratio for that version has no denominator. Record the build failure as the cell's value; do
  not silently substitute a different arm, a different version's number, or drop the row.

## One variant per process when a variant patches the toolchain

A variant that reaches its behaviour through a **process-global monkey-patch** — of a backend lowering
hook, a compiler pass, a `triton.knobs` toggle (e.g. a re-injected pipeliner, `recover.md` "Last
resort") — cannot share a process with another variant that patches the same thing. **The second import
wins, and both variants then silently measure the same pipeline.** Nothing about the result looks wrong:
both arms run, both pass their oracle, and their timings are legitimately identical, which reads as a
clean negative.

This is the same failure class as a shared Triton cache entry and it defeats the same guards:
`fingerprint()` compares compiled artifacts and the two really are identical, so the collision gate
passes. `--permute` cannot see it either, because the numbers do follow the code — there is only one
code. Recorded on an arm that reached parity via exactly such a patch.

The rule is a process boundary, not a flag: run each patched variant in its own process and compare
across processes only through the same-window protocol the rest of this file mandates (in acceptance,
`measure_legs`' fresh process per leg, interleaved pairs). An in-process interleave keyed by a per-variant
cache tag is **not** a substitute and is not used: the patch is process-global whatever the cache key
says.

## Config-sweep isolation + LDS pre-screen

A tile/config sweep must not let one bad config take down the batch or corrupt the rest:

- **One config per subprocess.** Re-keying a build in-process (`importlib.reload` + env vars, or a shared
  `TRITON_CACHE_DIR`) leaves stale module / context / cache / env state and corrupts later results.
  Launch **one fresh subprocess per config** (build + time + print) so each measurement is isolated — and
  so a **fatal LDS-overflow compile (`local memory exceeds limit`) that aborts at the process level**
  kills only that child, not the sweep.
- **Pre-screen LDS before building.** gfx950 (CDNA4): the LDS cap is **160 KiB/CU** (64 banks).
  **gfx942 downgrade:** **64 KiB/CU** (32 banks) — the same tile set overflows far sooner. A cshuffle
  epilogue needs `2·tile_m·tile_n` and a double-buffer needs `NUM_STAGES` whole tiles (on the Gluon path,
  the depth of your authored LDS ring). **Skip any config where `2·tile_m·tile_n` (or the staged tile
  bytes) exceeds the arch `LDS_cap`** (`perf_knowledge/hardware/data/hw_constants.json`
  `lds_per_cu_kib`, per arch) before compiling it — the overflow is a non-catchable process abort, so
  filtering up front is the only safe guard (`../hardware/cdna3-gfx942.md`). The tool that needs the arch
  must be told it (`--arch gfx950`); it does not default one.

## Config sweep harness (Stage-Plain)

Config sweeps need an N-way matrix, not only baseline-vs-candidate (search/screening; the winner is then
accepted under the binding protocol):

```text
config_matrix: knob_values / knob_classification / coupled_groups / shape_subset
per_config:    correctness_first / cache_dir / prewarm / warmup / reps
output:        config_by_shape_latency / per_config_aggregate / winner_per_shape /
               global_winner / split_evidence / rejected_for_correctness /
               configs_attempted / configs_correct / comparator_latency / local_winner_buckets
```

Do not time a broad constexpr sweep before a small correctness-first gate. When a knob controls dependent
data outside the kernel body, derive that data from the candidate config; if inputs were built with fixed
knob-dependent params, mark the sweep invalid rather than calling the candidate incorrect. Sweep
directions live in `front-end.md`.

## Served-range sweep

Do not validate only the seed configs the contract listed — that overfits the tuning to a few points.
Sweep the **full served range** of the size that varies in production (e.g. the decode concurrency /
batch dimension, or the prefill chunk size) **at the production boundary** (eager vs CUDA-graph,
`## Measurement basis + benchmark-artifact pitfalls`):

- cover the range densely enough to see where the **bound class flips** (memory/occupancy-bound small
  sizes -> compute-bound large sizes);
- the **transition ridge** (where one tile stops matching and the next takes over) is the
  **highest-risk point** and must be in the swept set — find it **by sweep, never assume a fixed
  coordinate**;
- add **threshold-adjacent** points (just below / at / just above) every candidate dispatch cutoff so a
  bucket rule is validated, not fit;
- keep the seed configs as the **contract-weighted** acceptance cases, but rank a candidate's robustness
  across the whole swept set.

This sweep feeds the multi-shape no-regression + bucket dispatch (`close.md`, "Multi-shape
no-regression + dispatch").

## Swept-cell correctness proxy

The contracted oracle typically covers only the seed configs, but the served-range sweep covers many more
cells with no golden reference per cell. For those cells:

- **Proxy gate (required for every oracle-less cell):** output-norm match or `allclose` self-consistency
  vs the live production baseline at the SAME shape, on fresh inputs (not the timing reuse buffers,
  `## Measurement basis + benchmark-artifact pitfalls`) — `harness_lib.check_random_vs_baseline` is the
  GEAK form (several value draws per shape, live baseline as the truth source, correctness gates and its
  speedup only reports). A proxy pass is necessary but weaker than the oracle — it catches gross
  divergence, not low-magnitude precision drift.
- **Full check at dispatch-flipping ridges:** where a cutoff changes the winning tier/config, build a full
  oracle / high-precision reference for that cell if one is constructible.
- **Record coverage:** report `full_oracle_cells`, `proxy_cells`, and `proxy_method` (`close.md`,
  `final_report.json` → `correctness.coverage`) so the correctness claim is auditable instead of a
  seed-only "all pass" that hides the expansion.

For changes touching shared-memory ordering this proxy does **not** replace the determinism race-test
(`## Determinism race-test (async / barrier / pipeline / layout changes)`).

## Determinism race-test (async / barrier / pipeline / layout changes)

Signature that demands it: any change touching **shared-memory ordering** — barrier elision,
`wait_group` depth, a relaxed/synced-load attribute, pipeline-stage count (authored LDS ring depth),
buffer indexing, or a recovered layout. For these, rel-error vs a reference **cannot** separate a data
race from ordinary bf16/accumulation noise (both vary run-to-run with random inputs), so a passing
rel-error does **not** prove the change is safe.

Method: fix the seed, run the **same** input tensors `N >= ~40` launches, assert `max |out_i - out_0| ==
0` — **and re-allocate the output buffer between comparisons, or poison it with a sentinel the kernel
must overwrite, or compare against a reference run that allocated its own output.** Without that clause
the assertion is vacuous: launching repeatedly into the *same* destination and comparing it against
itself is bit-identical whether or not the kernel wrote anything, so a kernel that writes nothing on
launches `2..N` — a store mask that excludes every tile, a guard that retires the block, a dispatch that
never reaches the store — passes with a perfect score. In one measured instance a harness reported **4 of
4 repeat runs byte-identical → CLEAN** on a kernel that was not writing what it was being credited with.
**Repeat-run determinism into a reused buffer is evidence about the allocator, not about the kernel** —
`## A control proves exactly the component it cancels, and nothing else` applied to a correctness gate
instead of a timing one: the reused buffer cancels the allocator and the result is then reported as a
property of the kernel. Bit-identical across *independently allocated* runs => deterministic => **no race
fired, which is not the same as the ordering being sufficient** (see below); any nonzero => a race
(revert or gate the change). Run it **alongside** the rel-vs-oracle correctness gate, not instead —
determinism rules out a race that *fires*, rel proves the math is right, and a change can pass one while
failing the other (a layout mis-recovery is deterministic-but-wrong; a barrier elision can be
right-on-average yet racing). This is the gate for the drain-elision knobs
(`../tile-programming/compiler-contract.md ## Scenario B: sanctioned compiler co-design`) and for
transcription equivalence (`transcribe.md`: layout diff + oracle at tolerance + determinism ~40
launches + asm parity).

**Read the green result at its real width: it says no race *fired*, not that the synchronization is
sufficient.** The test samples the interleavings this box actually produced on these launches. A missing
acquire or an elided barrier changes the outcome of no single interleaving — it only makes further
interleavings **possible** — so it is invisible to a gate that inspects values, no matter how many
repeats it accumulates. For the *sufficiency* of an ordering, this gate is not weak evidence, it is
**zero** evidence:

> **A dropped barrier is not a formula computed wrongly. It is a window this machine did not happen to
> open.** Sufficiency is established semantically — which release pairs with which acquire, at which
> scope — or by reading the emitted ISA and confirming the maintenance is there. It cannot be inferred
> from repeats that came out right.

This matters most in the direction people actually travel: **when a speedup comes from removing
synchronization, a green determinism test is not evidence that the removal is safe.** Keep both — the
gate to catch races that do fire, the pairing argument or the ISA read to cover the ones that cannot.

## Control rerun after plumbing changes

If harness, wrapper, import root, cache path, environment routing, or artifact selection changes, run a
control configuration (same source as the previous trusted baseline, exercising the new plumbing) before
interpreting performance. Only after the control rerun is stable should you attribute changes to the
kernel. In screening, `ab_bench.py --control NAME|auto` adds a **byte-identical duplicate** of NAME (or of
the baseline) as a first-class arm that rides the same per-cell rotation as every real arm (a control
pinned to one slot can only falsify a slot-0 bias); the spread across its cells is this window's measured
noise band — feed it to `round_record.py append --noise-band`. That band may only tighten the keep
threshold.

## A control proves exactly the component it cancels, and nothing else

The control above — the same source timed against itself through the new plumbing — is the right check,
and it is routinely over-read. Because both arms are the *same kernel*, every component that acts on the
two arms **by the same factor** divides out of their ratio exactly, by construction. That is precisely
why it is a good check for plumbing changes. State the cancellation that way — a common *factor*, under a
*ratio* — because it is also the statement of the two things the control cannot see: interference that
hits the two arms **unequally** (described under `## Repeatability + measurement order`), and a component
that hits them equally but **additively**, which does not divide out at all (`### The additive case: the
control gets greener as the measurement gets dirtier`):

> **A green null control certifies that a common *multiplicative* component has divided out of the
> ratio. It does not certify that the measurement is clean.** Two things survive it untouched, and the
> control has no channel through which to report either: anything that hits the two arms **unequally**,
> and anything that hits them equally but **additively**.

This is observable rather than theoretical, and it takes both halves to establish: ten consecutive
same-kernel controls came back green, spanning **0.994–1.007**, while the geometric mean being read off
the very same sessions swung by **2.2%**. Neither observation alone says anything — a green control is
unremarkable on its own, and a swing could be attributed to the kernel if nothing else were running.
Together they locate the interference precisely: it is real, and it is not a common *factor*, because the
control would have divided that out. Note what that does **not** rule out — a common *additive* term,
which the control is blind to by construction.

Practical consequences:

- **Do not quote a green null control as evidence that a headline number is trustworthy.** Quote it for
  what it is — plumbing parity, or the cancellation of a common factor — and name the component it
  cancelled.
- **Match the control to the threat.** A control is only sensitive to a disturbance that its two arms
  experience *differently*. To probe a disturbance that scales with an arm's host/device ratio, the
  control's two arms must differ in that ratio; a same-kernel control by definition cannot.
- **A measurement can be simultaneously well-controlled and wrong.** These are independent properties,
  and this page's other controls do not close the gap either.

### The additive case: the control gets greener as the measurement gets dirtier

Of the two blind spots named above, the additive one is worse, because the control does not merely miss
it — it reads *better* for it:

> **An additive common-mode term cancels under subtraction and does not cancel under division.** If
> contamination adds a roughly constant time `c` to both arms, the reported ratio is `(a+c)/(b+c)`, which
> is biased toward 1 — monotonically in `c`, and without bound.

Three consequences, and the first inverts how a green control reads:

- **A null control gets *better* under additive contamination.** Its two arms are the same work, so
  `(a+c)/(a+c) → 1` harder the dirtier the measurement is. In one measured instance the nulls read
  **0.02–0.84%** — pristine — while the absolute times in that same round ran **1.4–3.7x** high; in
  another, a null spanning **0.986–1.036** sat beside a measurement inflated roughly **8x**. So "the null
  was green" is not weak evidence of cleanliness, it is **anti-evidence**, and it may never be quoted as
  reassurance about an absolute.
- **A real A/B under additive contamination is biased toward null, so it understates the effect.** That
  asymmetry is usable in one direction only: a **positive** result measured under suspected additive
  contamination is **conservative** — the clean effect is at least that large. A **negative** result
  under the same conditions is **uninformative**, not a refutation. The second half is the one that gets
  dropped.
- **Therefore the absolute-value cross-check is the primary guard and the null control is a
  supplement**, not the other way round. `## Every within-comparison stays green when both arms run the
  wrong thing` already notes that every ratio is immune to a wrong-but-consistent comparand; that
  immunity is exactly why no ratio in the run can see this. Keep at least one absolute per component and
  check the arithmetic against it.

**The two shapes are discriminable by magnitude range**, at no extra cost when the measurements already
span one. An **additive** adder is roughly constant in absolute terms, so it distorts a cheap
measurement's ratio severely and an expensive one's barely: in one measured instance, across a **12x
range of clean baselines**, the observed *multiple* fell **8.26x → 1.70x** while the *adder* stayed
roughly flat. A **multiplicative** effect does the opposite — stable ratio, absolute cost proportional to
the baseline. Read which of the two is constant across the range you already have.

One additive model explained three anomalies that had been logged separately, and two of them look like
opposite diagnoses, which is why it is worth saying:

- **A larger-workload sample reading *below* a smaller one.** The adder lands on whichever sample it
  hits, independent of that sample's work, so it can invert a monotone series and present as
  instrument-impossible.
- **The suspiciously clean null**, above.
- **An interleaved A/A whose spread *exploded*** — to **112%** in that instance — instead of flattening,
  once the arms were taken seconds apart. Arms far enough apart eat *different* adders, so the ratio
  blows up rather than compressing. **Compression toward 1 and explosion away from 1 are the same
  mechanism at two interleave separations.**

**Report a contamination magnitude additively — "+N µs", not "Nx slower" — until the shape is
established.** Only the additive framing survives a change of baseline: a multiple quoted on one baseline
is meaningless on another, and in the measured instance it was the additive framing that preserved the
clue the model was recovered from. Quote a multiple only once the effect is known to be multiplicative.

## Repeatability + measurement order

- quick results must be followed by at least one full result before a claim;
- small wins must survive at least one repeat run; noisy/sub-ms work needs stronger repetition;
- distinguish a robust search metric (rank candidates cheaply) from the acceptance metric (the
  contracted metric — the `harness_lib` median under read-evict, fresh process per leg);
- for sub-ms / JIT / cache-path-changing candidates, include the order in JSON, run a baseline-only
  control after plumbing changes, alternate baseline/candidate order, and warm both paths before timing
  accepted numbers;
- on shared GPUs, run baseline repeats at the beginning and end, compare baseline drift against the noise
  threshold, and discard/rerun comparisons where drift exceeds the claimed win;
- **interleave baseline and candidate back-to-back per cell** (A,B,A,B...), not as two separate passes.
  In screening that is one process/session (`ab_bench.py` cells; `--permute` re-measures with the arm
  order **reversed** and reports whether each number follows the code or the position — the
  discriminating experiment for a suspected cache collision); in acceptance it is `measure_legs`'
  interleaved B,C pairs of fresh subprocesses. On a shared/drifting box the gap between two separate
  passes is dominated by between-pass drift, not the kernel delta — interleaving cancels the slow drift
  so the per-cell A/B is comparable. **The precondition is that the interference is common-mode**, i.e.
  that it hits both arms by the same factor. Host contention is not: two arms with different host/device
  ratios absorb the same host jitter unequally, so it does not cancel. A fully interleaved A/B run
  exactly as described above still moved by **5.5%**;
- **stamp the host load on every ratio you record.** A ratio without a load stamp cannot be placed in a
  load band after the fact, and the band matters: a **bit-identical** kernel compared against itself — a
  ratio whose true value is exactly 1 — reported a geometric mean of **1.0621** at a system load average
  around 11.6 and **1.0841** at around 6.8. Every digit past the 1 is artifact, and the two readings
  differ by more than many accepted wins. They are not a contradiction and not something to average —
  they are two operating points, and only the stamp says which one a stored ratio came from;
- **which statistic, and the rule against switching it.** Contention/preemption noise is **one-sided** (it
  only ever *adds* time), so over N repeats of an interleaved cell the **min** is the cleanest
  contention-free floor — **search/screening only (ab_bench)**, reported as a supplementary column beside
  the median. The **acceptance** statistic is the **median** (`harness_lib.time_op` over samples,
  `measure_legs` over pairs), and it stays the median, for the reason this bullet exists: **do not
  retrofit an estimator onto a harness already running a centre statistic.** Changing the estimator
  mid-campaign invalidates every earlier comparison, because the old numbers and the new ones then answer
  different questions — and re-deriving the history is usually impossible, since the raw repeats behind a
  stored median are rarely kept. A mixed record with the estimator named per number is honest; a record
  rewritten to a different estimator is not. Use the median/IQR (and `measure_legs`' `speedup_spread`) to
  report the spread; when min and median disagree on the verdict, the stricter reading governs and the
  cell is re-measured in a quiet window;
- **noise-floor gate:** first measure the repeat spread of a *fixed* config (same knob, repeated — the
  `--control` arm), then treat that spread as the noise band. A candidate-vs-baseline delta **smaller
  than the noise band is not a win or a finding** — report it as "within noise", do not act on it. (This
  is what turned an apparent decode "trough" / "lift" into noise once measured properly.) The band only
  ever raises the keep threshold above `MIN_IMPROVE` (2 %), never lowers it. **Measure that band *before*
  spending the comparison, and hold the effect you expect against it.** A fixed config run against
  itself on the same shape gives this run's smallest resolvable effect; if what you are going looking for
  is predicted to be smaller than that, the run cannot produce evidence in either direction and the cost
  buys nothing. This matters because **"no significant difference" and "this instrument could not have
  resolved it" come out of the harness as the same output and are opposite conclusions** — reading the
  second as the first retires a lever that is still alive, and nothing downstream can tell them apart
  afterwards. Give them **different labels in the harness**: where the predicted effect sits under the
  floor, the result string is *predicted unresolvable — not evidence of no effect*, not "no difference".
  That is a code change, not a discipline; a distinction that lives only in the memory of whoever wrote
  the report does not reach the reader. **Gate that label on the *achieved* floor once the run is done,
  not only on the pre-run estimate.** A cheap pre-run estimate — a short A-vs-A on the same shape, taken
  before the real comparison — is an order-of-magnitude **screen, not a bound**, and it errs optimistic as
  readily as conservative: in one measured instance the pre-estimates read **1.37% / 1.25%** against
  achieved floors of **2.12% / 0.62%** on the same two shapes. So a label gated only on the prediction can
  fire at a cell the instrument **did** resolve and discard a real null — in that campaign it fired where
  the predicted floor was **2.25%** and the achieved floor was **0.34%**. Like any quantity derived from a
  prediction about a measurement rather than from the measurement, such a label cannot come out false.
  Record the legs and repeats the estimate was taken with beside it, and re-check the label against the
  achieved floor before the cell is quoted. **And read the floor for what it bounds: it is one
  measurement's resolution, not the sampling variability of the statistic across measurements.** An
  effect can clear its own run's floor several times running, pass a null control every time, and then
  fail to reappear on the next attempts — "could I resolve it this time" and "how does this quantity
  behave over repeated measurements" are different questions, and the first does not answer the second.
  The strengthening move is to **widen the condition set — shapes, sizes, inputs — not to add repeats at
  the same condition.** Repeats of one condition are structurally incapable of exposing an artifact that
  depends on that condition, which makes them, for that purpose, a check that cannot come out false.
  **And say which floor you quoted against, because there are two of them and they are not
  interchangeable.** The **in-window same-arm repeat** — a fixed config re-measured minutes apart, inside
  one session and one machine state — and the **historical reproduction span** — the same config across
  sessions, days, machine states — are different quantities, and they are not close: in one measured
  instance the in-window figure was **0.59%** and the historical span **39.68%**, a factor of ~65. The
  rule is **match the floor to the claim**: a within-session A/B is quoted against the in-window floor; a
  claim that a result **reproduces across sessions** is quoted against the historical span. Neither
  substitutes for the other, and **a result that clears only the in-window floor is a within-session
  result and must be labelled as one.** Quoting against the wrong floor fails no gate — it silently
  misprices the claim, in whichever direction the mismatch happens to point. **And read a `reproduces`
  stamp at its real width: it certifies the *sign*, not the *value*.** A repeat-check that asks "did this
  bucket move the same way again" is answered by the direction of the effect and is insensitive to its
  magnitude, so it keeps returning green while the number being reproduced drifts. In one measured
  instance a bucket carried a clean reproduction stamp across **15** runs while its own ratio wandered
  **0.6905 to 0.8450** — a **15.45-point** span, several times any decision threshold in that campaign —
  and every one of those runs was entitled to say "reproduces". If a downstream claim quotes the *value*,
  the stamp behind it must be a **spread on that value** (min/max or a quantile across the repeats,
  carried with it), not a direction check; if the stamp is only a direction check, the claim it licenses
  is only **"the effect has this sign"**. Quote the span beside the stamp and the two cannot be confused.
- **per-sample null gate — a filter must prove it cleaned, and may only void:** attach an A/A pair to
  **every** measurement triple (the same arm against itself, taken milliseconds apart, *inside* the
  triple) and discard the triple when A disagrees with itself. As a contamination guard this is strong,
  and the reason is that it is **time-side and namespace-independent**: it names no device, consults no
  mapping table, and so cannot be defeated by an identifier that points at the wrong hardware. In one
  measured instance it cut the inter-quartile spread ~10x. **It can also bias what it keeps.** In that
  same instance, at the smallest workload it retained **13%** of samples and reported a **+16.99%**
  effect where the unfiltered data read **+1.62%** — because *steady-high* and *steady-clean* look
  identical to a gate that only asks whether A agrees with A. The gate catches **episodic** contamination
  and is blind to **steady** contamination; where the steady component is additive it does not merely
  miss it, the A/A pair reads *cleaner* for it, and the discriminator that does see it is the
  magnitude-range one in `### The additive case: the control gets greener as the measurement gets
  dirtier`. So require the gate to **certify that it cleaned rather than merely selected**: a
  survival-rate floor, plus filtered spread that comes down to the level of the sibling conditions. Until
  a gate is A/A-calibrated on a known-clean device it **may be used to void a condition, never to accept
  one.** The general form outlives this particular gate: **a filter that selects on the same quantity it
  is later used to measure cannot license the measurement, only kill it.** And a post-hoc rule — one
  added after seeing the data — is **labelled post-hoc**; it is never backfilled as pre-registered.
- **cold-compile outliers survive warmup:** when many configs/shapes are built back-to-back in one
  process, the *first* timed run of a freshly-built config is contaminated by JIT compile (seen:
  implausible TF cells, >1000% ratios). Discard the first run or take **max-or-median of ≥2 passes**, and
  treat a **lone implausible cell as an artifact until reconfirmed**, not as a finding.
- **sweep one config per subprocess:** re-keying a build in-process with `importlib.reload` + env vars
  leaves stale module / context / env state and corrupts the result — launch **one fresh subprocess per
  config** for a tile/config sweep (`## Config-sweep isolation + LDS pre-screen`).

---

## Gates

| Gate | Must include | Use for |
| --- | --- | --- |
| Quick | fixed ordered subset, correctness, enough reps for obvious regressions, per-shape JSON | search + early rejection (`ab_bench.py`, quick harness mode) |
| Full | contracted stream, same boundary, per-shape map, aggregate metric, repeat for small/noisy deltas — timed under the binding protocol (events, per-sample sync, read-evict, median, fresh process per leg, same-window baseline) | final acceptance (`verify_engineer` / `measure_legs`, the `harness_lib`-timed full benchmark) |

Small-gain policy: an aggregate below `MIN_IMPROVE` (**2 %**) is not committed — it may be tracked, never
banked; a gain at or just above the threshold (or inside this window's measured band, whichever is
larger) needs ≥1 independent full repeat; all-shape tasks report best/worst per-shape movement;
aggregate-only wins need dispatch/fallback for repeated losers; disagreeing repeats = noise, not a keep.

## Acceptance

Keep a candidate only when the agreed contract is met under the real benchmark boundary and the binding
protocol — not because quick improved once, one shape improved a lot, kernel-only improved while the true
boundary lost, a screening statistic (min, hot cache, in-process interleave) improved, or it merely
compiles and is correct. The verdict needs ≥ `MIN_IMPROVE` over the same-window baseline, after every
stricter check on this page (control band, `< 3F` floor, `primed`, dispatch identity) has had its say.
Close-out bars and the two-baseline accounting: `close.md`, `recover.md`.

---

## Every within-comparison stays green when both arms run the wrong thing

The sections above on controls and protocols are instances of a wider limit. Interleaving, the null
control, a bit-exactness gate, ratio normalisation against a same-session baseline — all of these are
**same-against-same** constructs, and all of them work by removing what the two arms share. That is
exactly why none of them can see the failure where the two arms are not the entity you meant to measure
at all:

> **A within-comparison cannot establish what it is comparing.** If both arms exercise a fork instead of
> the shipped entry point, an old configuration instead of the current one, or a neighbouring op instead
> of the target, every gate above still passes, every ratio is still tight and reproducible, and the
> result is void.

The reason this deserves an unconditional step rather than a caution is that **it is indistinguishable
from a correct measurement from the inside.** There is no widened spread, no failed control, no
suspicious cell — the dashboard is identical in both worlds. A caution you are supposed to heed when
something looks off never fires, because nothing ever looks off. So run both of these on every campaign,
not when suspicious:

- **Dispatch identity.** Establish that the path under measurement is the path production takes — by
  **testing** it (make the measured artifact observably different and confirm the observation changes),
  not by arguing that the two are equivalent. An equivalence argument is the thing being checked; it
  cannot also be the check. The cost is not hypothetical: in one measured instance a probe that rebuilt
  its own launch arguments left a single `constexpr` at its default, so every timing was an A/B between
  two variants of a **retired** kernel — it predicted **+6.96%** end-to-end where the shipped dispatch
  measured **-0.06%**, with every gate on this page green throughout. **The cheap default is to call the
  production entry point** and read every launch parameter out of the production module rather than
  retyping it, so a default that moves upstream moves in the probe too (in GEAK,
  `assert_legs_differ` checks the weaker half mechanically — the legs import different code and the
  baseline is the live stack — but not that the candidate leg is the dispatch production takes). Where a
  fork is unavoidable, **prove** it is the same code rather than arguing it: compile both at the
  production configuration and diff the *normalised* assembly — symbol names, local labels and comments
  stripped — line by line. "Same instruction stream, 0 differing lines" is a fact; "the defaults match, so
  it should be equivalent" is a prediction from the class rather than a reading of the instance. **And do
  not let a set of counters stand in for that diff.** Counts are order-insensitive, so a reordered or
  differently-operanded stream collides with them: in one measured instance nine ISA counters — matrix
  ops, LDS reads and writes, VALU, loads, register and LDS totals, spills — all agreed between a fork and
  the shipped kernel it was missing four `constexpr` values from. **If your identity check can be passed
  by a different program, it is not an identity check.** **The device is part of the entity**: on a
  shared multi-GPU node the same silent substitution happens in hardware, and the index you think you
  pinned is not re-derivable afterwards (`### Device identity: a wrong index returns a plausible number,
  never an error`).
- **Internal consistency of the absolute values.** Keep at least one absolute number per component and
  check them against each other. One half of the invariant is free and unconditional:

  > **No sub-component may be more expensive than the whole that contains it.** A violation is a broken
  > instrument, not a finding.

  **The parts summing past the whole is the weaker half, and it has two causes that must be separated
  before anything is called broken:**

  - **A once-per-sample fixed cost charged once per component** — a broken measurement, the granularity
    error described in `## Hot or cold cache: pick it from the bound class, because it can REVERSE a
    ranking`. The inflation grows with how finely you split, and the decomposition is void.
  - **Isolated capture losing graph-internal overlap** — components timed separately cannot overlap,
    while consecutive work inside the full graph does, so isolated capture bills serialized time for
    work that partly ran concurrently. This is **expected method bias, not a fault**: it is bounded by
    how much adjacent work really overlaps, and in one measured instance the parts exceeded the whole by
    **15-28% in every bucket** with nothing wrong. What it costs is precision, not validity — each
    component's share becomes an **interval rather than a point**, and must be reported as one.

  So `part > whole` fails the run outright; `sum(parts) > whole` asks which of the two it is, and only the
  first answer discards the decomposition.

  **Every ratio is immune to this check**, because a wrong-but-consistent comparand divides out of a
  ratio and survives only in an absolute — which is the reason to keep absolutes at all once you have
  decided in ratios. That is also why the granularity error above is invisible elsewhere: a per-sample
  protocol re-used per component inflates every part until the parts outweigh the whole, and no ratio
  anywhere in the run moves.

Neither check is expensive, and neither is a substitute for the other: the first fixes *which* entity,
the second catches the case where the entity is right and the instrumentation is attached in the wrong
place. Both still assume the numbers in front of you came out of the run you are attributing them to —
the section below.

## Provenance: every gate acts on a number you already have

The two checks above and the free invariant inside them are answers to "the number is in my hand — is it
the number I meant?". This is the third member of that family and it runs **before** all of them:

> **A bit-exactness gate, a null control, interleaved execution, ratio normalisation, a pre-registered
> criterion — every one of these acts on a number you already have. Not one of them asks whether this run
> produced it.** They inspect the number's *properties*; a result file that was never rewritten has all
> the right properties and passes the lot.

The concrete shape leaves no signature in the data. A run's writes fail silently — the output path is not
writable by this identity, a redirect is refused, the destination is owned elsewhere — and the script's
exit status is taken from a step *after* the one that does the work, so the run reports success. The
reader then picks up an older result, and every gate downstream faithfully evaluates a measurement this
run never made.

Provenance checks, cheapest first and increasing in strength:

- **Make a failed write fail the run.** Take the status from the step that does the work, not from one
  that follows it. A status collected after a print, a summary line, or a cleanup step is a status about
  the print.
- **Write a fresh filename per run; do not overwrite.** A missing file is loud. A stale one is silent.
- **Carry something in the result that only this run could have produced** — a timestamp, an environment
  stamp, the host-load stamp this page already requires, `harness_lib`'s `cache_condition` / `timer`
  receipt — **and check it against the run when you read it back.** This is the one that still fires when
  the first two were skipped, because it is verified at the point of use.

Why it outranks everything else here: every other failure mode on this page at least ran the experiment.
This one's failure mode is that **no measurement happened at all.**

### A number has a lifetime after the run that produced it

Everything above is a property of the measurement *event* — the load stamp, n, the floor, the
interleaving, which run wrote the file. Nothing on this page yet speaks to what happens to a number
**afterwards**, and that omission is enough on its own to put stale figures into two independent
write-ups on the same day:

> **Cite a measured constant by its quantity and its correction state — "3.39–3.46 TB/s, cold streaming,
> floor-corrected" — never by the file that produced it.** A filename names a *producer*, not a value.
> When the producer's value is superseded, every sentence that cited the file silently inherits the dead
> number. If the file must be named, name it as provenance for the **correction**, not for the value.

The incentive is inverted, which is why this survives review: a reader who does the responsible thing and
checks a citation against its source file gets the **stale** value, concludes the live number is wrong,
and "corrects" a current figure into a retired one. **Verifying from source produces the worse answer.**
In one measured instance a cold-streaming ceiling was cited by script name in six places, one of them a
load-bearing refutation; the quoted value was right and the script's own published table — uncorrected
for the instrument floor (`## The largest additive term is usually your own instrument, and you can
measure it`) — read about a third lower and still prints that way.

Three rules, all of the same shape: the enforcement has to live in the **artifact**, because the artifact
is where the mistake is made.

- **A superseded probe retracts in its own output** — a one-line banner the probe itself prints, above
  its motivation. A retraction that exists only in a report does not reach the next person, who re-runs
  the probe and reads its headline.
- **A constant hard-coded into a script carries its conditions inline**: the quantity, the conditions it
  was taken under, whether it has been superseded, and what to do if the table of record is restated.
  `X = <number>` copied out of a run is a measurement whose provenance nothing in the code can see.
- **When two corrections land together, apply both before re-reading the conclusion.** In that instance
  they moved the inputs in **opposite** directions — a higher ceiling weakened the claim, a dispatch-time
  subtraction strengthened it — and only applying both turned a flat count into a range. The conclusion
  survived; the count did not.

## When the gate needs a number you cannot read, bound it — do not guess it

The section above assumes the number exists somewhere and only asks whether this run produced it. This is
its other half: the gate needs an input that **nothing in your run reports** — a value chosen by the other
side of the comparison, a latency, an internal constant of a library you call. Three moves are available
and only two of them are honest:

1. **Call the code that produces it and make it report.** This is the one that gets skipped because it
   looks like the expensive option, and it usually is not: the path that consumes the quantity already
   computes it, so the work is *exposing* it, not deriving it. Try this before concluding the number is
   unreadable.
2. **If it genuinely cannot be read, substitute a justified bound or interval — and state the conclusion
   over the whole interval.** "The guard holds for every value in this range, and here is where the range
   comes from" is a result. It is also falsifiable: a reader can attack the range.
3. **Do not pick a plausible point value.** It turns an unknown into an assumption, and the assumption
   leaves no trace — downstream, a point value is indistinguishable from a measured one. A bound announces
   itself as a bound; a guess announces nothing, and the conclusion inherits a confidence the input never
   had.

Two recurring instances, both of the same form — *which run are you reproducing?*:

- **Reproduce the seed of the leg you are timing, not the leg that checks correctness.** They are usually
  separate draws, and using the correctness leg's seed attributes the timing to a different set of inputs
  than the one that produced it. With a stateful generator this means replaying the draws that precede
  it, not only re-seeding.
- **Take a blocking or sharding size from the identity of the dispatch that actually ran, not from the
  library's default constant.** When the default and the dispatched value differ, every count derived
  from it is off by a whole factor rather than slightly — and a conclusion that rests on the count flips
  with it.

## Narrowing a gate and waiving it are different acts

When a control, a tolerance or a gate comes back out of bounds there are two honest responses, and they
are not interchangeable:

> **A waiver says "this result does not count". A narrowing says "this gate's domain is smaller than I
> first wrote it, and here is the mechanism."** The first leaves no trace and is available again tomorrow
> for the next inconvenient result. The second edits the criterion itself, so it applies to the next run
> as well — and it can be argued with.

One question separates them: **can you state the mechanism by which the gate does not apply to this
subset?** If you can, **narrow** it — put the exclusion in the code with the mechanism written beside it,
so the next run inherits the narrower gate *and* the reason. If you cannot, what you have is a **waiver**,
which is legitimate but has a different obligation: **a waiver belongs in the conclusion, not only in the
process.** A point excluded from a check has to be visible to whoever reads the result, not only to
whoever ran it.

Two consequences worth keeping:

- **An exclusion with no mechanism beside it decays into a habit.** The stated reason is the only thing
  that bounds how often the same move gets made.
- **A narrowing is falsifiable, and that is its value.** A written mechanism can be shown wrong by the
  next reader; "that point was dropped" cannot.

### Audit the summary's quantifiers, not its digits

The rule above puts the obligation on the write-up: an exclusion has to be visible to whoever reads the
result. This is the mechanism by which a visible exclusion quietly stops being visible.

> **When a result is compressed into a summary the drift is not random — it runs toward tidier, and what
> it eats is the failure cases. So audit the universal quantifiers ("all", "every", "each", "none",
> "always"), not the numbers. Numbers are what a reader checks, which is exactly why the rot is not in the
> numbers.**

A digit that disagrees with its source is a visible defect and gets caught. A quantifier that has quietly
widened produces a *cleaner* sentence than the truth, reads as well-organised, and destroys the specific
evidence that a gate has teeth. In one measured instance a summary audited claim-by-claim against its own
source sections scored **25 of 30** numeric claims correct, and the two most damaging errors were not
arithmetic: one had compressed a distinct intermittent artifact into "same as the previous item", merging
two mechanisms that need different detection; the other wrote "**all three** deployments passed the entry
gate" when the first had been **voided on that very gate** — deleting the single strongest piece of
evidence that the gate ever fired, and leaving it looking like a formality. Both errors made the text
shorter and more orderly. **List the quantified claims first and resolve each against the source before
looking at any number; where a universal survives the audit, record the count it ranges over ("3 of 3")
so the next compression cannot widen it silently.**

### An in-place correction is finished when the corpus has been searched for the OLD string

> **Correcting a value in place is not finished when the source site is fixed. Search the whole corpus
> for the **superseded** string — the old value, not the new one. A search for the corrected value shows
> you where you were already right, which is the comfortable search and the useless one.**

- **The edit radius of a correction is about one line.** A retroactive sweep over 25 logged corrections in
  one measured instance found live residuals sitting **one line above** their own correction, one line
  below another, and immediately after a strike-through. The corrector had looked at the context; the
  context is precisely the region they will not re-read.
- **A section-level "retracted" banner does not cover the numbers inside it.** Downstream readers search
  for the number, not the section title. Mark each number in place.
- **Sweep outbound artifacts first.** A wrong *transferable rule* travels further than a wrong number,
  because the recipient has no instance to check it against. In that sweep 3 of the 17 live residuals sat
  in the artifact meant to be handed onward, including a rule that had already been refuted.
- **The raw hit count is not reportable.** In that sweep the automatic "unmarked hit" count over-reported
  by **8.4x** (143 raw → 17 real): a two-digit percentage matches a percentage with a leading zero, a
  register count matches a legitimate unrelated count elsewhere. Triage is the work, not the write-up of
  the work, and a raw count published as a finding carries false authority. **Record zero-hit rows** —
  they are what makes the output a table of evidence rather than a to-do list — **state the scope beside
  the count** (a sweep scoped to one file extension is not making the unqualified claim, and the scope is
  not recoverable from the number), and **do not ask the question with `grep -c`**, which exits non-zero
  on zero matches, so a clean result and a broken command are indistinguishable downstream.

The worst defect this finds is not a stale number: it is **one conclusion existing in two opposite
versions in the same document**. In that instance a corrected section named resource A as the binding
limit and priced relieving resource B at zero, while an un-swept sibling said the reverse from a stale
input. A reader landing on the stale copy gets **no signal**, because the stale copy is perfectly
well-formed — which is also why a sampling audit does not find it: a sampling audit samples the corrected
site.

## A pre-registered criterion fixes the threshold, not the statistic's power

This section and the two after it are one question asked three times: **can this statistic decide**
(here), **can the instrument it is read off be trusted** (`## Check the instrument before you read the
scale off it`), and **are the two statistics you are corroborating with actually two** (`## Two criteria
can collapse into one on the data you actually got`).

Writing the decision rule down before looking at the data removes one failure — choosing the cutoff to
fit the answer — and is worth doing for that alone. It does not remove this one:

> **Pre-registration protects against picking a threshold after the fact. It does not protect against
> picking a statistic that cannot separate the hypotheses you are deciding between.** A criterion fixed
> in advance can still be **identically satisfied** by both explanations in play — at which point it
> reads like evidence and is a restatement.

So the pre-registration needs two entries beside the thresholds: **the alternative you intend to rule
out**, and **what the chosen statistic would read under each alternative**. If both alternatives give the
same sign and roughly the same magnitude, that statistic cannot serve as the criterion, and tightening the
threshold repairs nothing. (The experiment contract that carries these fields: `records.md`, "9.
Experiment / hypothesis-test contract".)

The recurring instance is a statistic that **collapses a curve into one number** — an endpoint ratio, a
total speedup, a geometric mean over a sweep. What it discards is the *shape*, and the shape is usually
the mechanism: a quantity that degrades steadily across the sweep and one that steps once between two
adjacent points and is flat after it can produce the same summary while supporting opposite conclusions
about the cause.

> **Report the per-point data beside the aggregate, and read the per-point data first.** Let the summary
> summarise something you have already looked at; do not let it look on your behalf.

That is not a hypothetical about aggregation. In one measured instance a pre-registered pair — an endpoint
ratio of the per-shape speedups plus an excess-growth term — fired unambiguously (**0.8741** and
**1.1438**, agreeing in 7 of 7 runs) and selected a branch that is a claim about a *slope*. Evaluated on
everything except the two smallest buckets the same two statistics read **0.9914** and **1.0087** — flat.
The whole signal was one step between the first two points, and the pre-registration stamp would have
made the wrong mechanism *harder* to question rather than easier. Two discriminators cost nothing on data
you already have:

- **Report `f(last)/f(second)` beside `f(last)/f(first)`, and evaluate the statistic on a sub-range.** A
  step and a slope agree on the first and disagree on the second. When you pre-register a test for a
  trend, register the **per-point series** as the artifact; an endpoint ratio is a summary of exactly two
  points and discards the shape being established.
- **Exchange the per-bucket rows before two lines claim they found the same mechanism.** In that same
  campaign a second line carried the identical endpoint signature — 0.8741 — over a sub-range that read
  0.8500, i.e. a real slope. Same summary number, two different shapes, **two findings**; reported as
  endpoints they would have been filed as one mechanism independently corroborated.

**And bound each bucket's prize before spending a round on it.** Set one bucket's candidate time equal to
its baseline — a perfect, free, impossible fix — and recompute the headline **through the fold the
harness actually applies**, not the one its printed label names. That is the most any amount of work on
that bucket can ever be worth; it is one line of arithmetic on a log you already have and needs no
device. In one measured instance it retired a refactor outright: the two buckets holding most of the
targeted mechanism were together worth at most **+0.0141** on a 0.8268 headline. Recomputing the same
bound through a plausible-looking but different fold — a call-weighted mean of ratios instead of the
harness's own — misallocated the gain by up to **2.1x** per bucket.

This belongs with the gates rather than with process advice because its failure mode is that **every
process check is green**: sample count, spread, sign consistency, a control arm, an untouched threshold —
all satisfied, and not one of them constrains the functional form.

**The four failures in this family are mutually invisible**, which is why each gets its own check rather
than a shared caution: `## Every within-comparison stays green when both arms run the wrong thing`
catches **comparing the wrong entity**; the granularity invariant it carries, with `## Hot or cold cache:
pick it from the bound class, because it can REVERSE a ranking`, catches **changing the granularity a
protocol was validated at**; `## Provenance: every gate acts on a number you already have` catches
**never having measured**; and this section catches **choosing a quantity that cannot decide the
question**. In all four the number passes every gate on the page.

### Sign agreement across buckets is not a defence against drift

The instance that survives review most often is not a statistic at all, it is a layout: a toggle test
whose two arms were measured in **different runs**, on either side of whatever moved between them. Same
failure shape — a criterion that reads identically under both explanations in play:

> **A toggle test whose arms sit across a drift boundary manufactures the coupling it was built to
> detect.** Slow drift in the baseline is **common-mode across every bucket, shape and size in the sweep**
> — it moves all of them the same way — so **agreement of sign across buckets is not evidence against
> it.** Sign agreement tests whether the effect is *uniform*, not whether it is *real*.

Bucket count buys nothing here: widening the sweep multiplies a quantity the confound has already set to
unanimous. In one measured instance 7 of 7 buckets agreed under the confounded layout, and the effect
**reversed entirely** once the two arms were run adjacent to each other.

The protocol is adjacency, and it is cheap:

- **Run the toggle A/B/A with the arms adjacent in time**, in one session, the way `## Repeatability +
  measurement order` already requires of every other A/B. A toggle is not exempt because its two arms are
  two builds rather than two kernels (two builds that patch the toolchain are two processes, run as
  adjacent legs — `## One variant per process when a variant patches the toolchain`).
- **Publish the same-arm repeat taken inside that window** as the noise floor the result is quoted
  against (`## Repeatability + measurement order`, noise-floor gate). A toggle result with no in-window
  repeat beside it has no floor.
- **Adjacency is the requirement; no interval is a validated threshold.** The gap has to be short enough
  that the baseline does not move between the arms, and that is established by measuring the in-window
  repeat, not by asserting that some number of seconds is short. An interval that sufficed once is one
  observation, not a constant to carry to another kernel, box or session.

### A mechanism proposed to explain a gap must itself differ between the arms

The section above is about the criterion; this is the same failure one step earlier, at
candidate-generation time.

> **A condition that holds equally for both arms is excluded as an explanation of their difference at
> zero cost — no measurement, no threshold, no sample size.** Check the quantifier before the magnitude:
> the question is not "is this happening" but "is this happening **unequally**".

This is the cheapest filter on the page and it gets skipped because the candidate is usually **true**. The
named condition really is present, and often really is limiting both absolute numbers; it simply cannot
produce the gap, because it is subtracted out of it. In one measured instance a shortfall in dispatched
work — real, running between 22% and 97% of the occupancy ceiling across the sweep — was offered as the
cause of a ~19% per-byte efficiency gap against a faster comparator, while the two implementations issued
**bucket-by-bucket identical** work geometry. The correct consequence is a redirection rather than a
deletion: filling the dispatch raises **both** absolutes and leaves the gap exactly where it was, so it
belongs on the absolute-throughput ladder and not on the gap ladder. Two levers, two objectives, and
conflating them costs a round. This is not `## Every within-comparison stays green when both arms run the
wrong thing` — there the comparison itself is invalid; here the comparison is sound and the *explanation*
is what fails.

### An exact boundary is arithmetic, not evidence of causation

> **A proposed mechanism whose boundary lands exactly on the observed transition has told you nothing yet.
> Exactness is a property of the arithmetic you chose, and a quantity derived from the sweep's own
> parameters lands on a sweep boundary by construction. Before spending a measurement, make the candidate
> emit one prediction it could fail — the cheapest is the *sign*.**

A sign prediction is free, is fixed before any data is consulted, and is not tunable after the fact. It is
also the prediction a plausible-but-wrong mechanism most often gets backwards, because its author reasoned
from "this effect exists" to "this effect explains my case" without carrying the direction through. In one
measured instance a quantization candidate reproduced an observed transition **to the sweep point** and was
then excluded by its own sign: the arm that lost in that regime was the arm the mechanism favoured. A
second, model-free refutation followed — two sweep points with **identical** geometry under the
candidate's own accounting differed by **1.03 percentage points** in the effect, so the candidate's
variable was constant across a pair where the effect was not. **A mechanism that predicts the wrong sign is
not weakened, it is excluded.** Record both refutations where both exist; the model-free one survives a
change of model. And when the candidates run out, **"unexplained" is the correct entry** — a replacement
mechanism proposed at the moment of falsification has been selected for compatibility with the surviving
data and has no independent support.

## Check the instrument before you read the scale off it

The second face of that question, and the one about the instrument rather than the statistic. The section
above says a criterion written in advance cannot rescue a statistic with no power. This is its other side:
**when what you write down in advance is a check on the instrument rather than a threshold on the result,
writing it down in advance is exactly what saves you.**

The commonest instrument in this work is not a timer, it is a **fitted reference** — a curve through
measured points (a regression line, a fitted roofline, an intercept read as a floor) that a later number
is compared against or extrapolated from.

> **A fitted reference cannot be extrapolated from while its own residuals are structured and of the same
> order as the effect being measured.** The extrapolated value is not reportable at any margin — including
> the reassurance that it would clear the threshold several times over, because that spread is computed
> from the same untrustworthy line.

So inspect the **residuals**, not the headline error, before a fit is used as a reference. Structure in
them — a sign pattern, a drift with the sweep variable — says the model is wrong in a way an aggregate
error term hides, and a wrong model is worst exactly where extrapolation wants it to be good.

The second half points the other way and is the more useful one:

> **When two curves' residuals move together, the noise belongs to the sample set — the shapes, the
> workloads, the instrument — and not to either curve.** That is why a per-point **ratio** can be steady
> while each curve on its own looks noisy.

Pick the statistic in which the shared component cancels, rather than one that books a property of the
sample set as a property of the thing under test. In practice: if a comparison can be formed per point and
aggregated afterwards, prefer that to comparing two separately fitted aggregates; and when a fit is quoted
at all, quote what its residuals look like beside it.

### A fit cannot adjudicate a question about its own highest-leverage point

> **A fitted parameter that is an extrapolation — an intercept, a floor, a value at zero — takes its value
> almost entirely from whichever measured point lies nearest the extrapolation target. When that is the
> point under investigation, "the two fitted floors agree" and "that point is an outlier" are not two
> findings that corroborate each other. They are one statement, and the fit cannot tell you which reading
> it is.**

Leverage is the name for how much a single observation moves a fitted parameter, and it is never spread
evenly: in a two-parameter fit over a sweep the extreme points carry nearly all of it and the interior
points carry almost none. So before a fit is allowed to settle anything, either compute each point's
leverage, or run the model-free version of the same question — drop each point in turn, refit, and watch
how far the answer travels. A parameter that moves materially when one point is removed was that point's
value wearing a model's clothes, and quoting it as independent support for that same point is circular.

The default follows. **Prefer a per-point, model-free comparison to a fitted one** wherever the data
admits it: tabulate the quantity at every measured point and compare the table, rather than reducing each
arm to two coefficients and comparing those. It answers the question the fit was standing in for, it has
no leverage point to be hostage to, and it shows the curvature and saturation a two-parameter model
necessarily absorbs into a slope — the fit reports a clean trend across a range where the per-point table
shows the effect flattening out entirely, and the flattening is usually the finding.

### Two points can refute a scaling law; they cannot establish independence

The same demand, applied to a model with two points and no residuals to inspect.

> **Refuting `f(x) = kx` with two points tells you the scaling law is wrong. It does not tell you that `f`
> is independent of `x`.** Two equal points are equally consistent with constancy, a step, a saturation
> and a threshold. Before asserting independence, measure the **degenerate point** — the smallest or most
> trivial value of `x` the system admits.

The trap has a specific shape: the same two measurements are used twice, first to kill a model and then to
adopt its complement. The second use is unlicensed — the complement was never tested, it was merely the
other thing that came to mind. In one measured instance a residual allocation was hypothesised
**proportional** to the warp count; measured at 2 and 4 warps it was **identical**, which refuted
proportionality and was immediately over-corrected into "**independent** of warp count", written down as a
property and exported as a transferable rule. The degenerate point falsified it within the hour: at **1
warp the residual was 0** against **128 bytes** at both 2 and 4 — a **step**, consistent with a cross-warp
staging cost that exists only above one warp, which is a different and more useful mechanism than either
model tried. Note what was right both times: the mechanism *class* was correct throughout, and only the
scaling law was wrong, twice. **A correct mechanism class is not a licence to assert a specific scaling
law**, and the degenerate point is usually the cheapest single measurement that discriminates among the
candidates — more ordinary samples would not have helped, because three points clustered in the same
region do not discriminate.

## Two criteria can collapse into one on the data you actually got

The third face of the same question. You designed two criteria to corroborate each other, and as designed
they were independent — nothing on paper makes either follow from the other. Then the run comes back, and
the premise of the first one holds: two quantities that could have differed turn out equal. Substitute
that equality into the second criterion and it reduces to a quantity you already had before the run. It
contributes nothing, and it appears in the write-up as "two independent axes point the same way".

The check is cheap enough that there is no reason to skip it:

> **Substitute the realised result of the first criterion back into the second, and count the degrees of
> freedom left. If none remain, the second is the first restated in different units** — corroboration on
> the page, an identity underneath.

The collapse is a property of the **data**, not of the design, which is why it cannot be caught before the
run and why the substitution belongs at write-up time, beside the claim of agreement. A pair of criteria
that are independent for every other outcome can be one criterion for the outcome you actually got.

Now the part that decides whether this rule helps or harms:

> **What collapsed is the corroboration, not the discrimination.** The test was not wasted. All of the
> resolving power is concentrated in the first criterion, and the alternative the second was meant to
> exclude is still excluded — by the first one.

So the correction is to the *claim*, not to the result: report one axis and say what it resolves, instead
of two axes agreeing. Read as "do not design multiple criteria" this rule does damage — designing several
is how you find out that one of them was redundant, and the redundant one costs almost nothing to have
carried.

That question is about the criteria. The section after next asks the same thing about the **arm** they are
applied to — whether the edit being timed moved one quantity or two (`## Before timing, list every
quantity the edit moves`).

## A compile-only kill step is a falsifier, not a price

The cheap leg of a gate is a compile-time read of what the toolchain emitted: the pattern the edit was
supposed to remove is gone from the emitted code, or it is not. That leg is worth running first and worth
running every time — an edit that never reached the emitted code cannot have moved the time, so it retires
the candidate before a measurement window is spent on it. The failure mode is what happens when it comes
back **positive**.

> **A compile-only check can end a candidate. It cannot price one. It answers "did the transformation
> happen"; the question the round is asking is "what is it worth".**

Hitting the compile-only target completely — every instance of the pattern eliminated on that path, output
bit-exact, occupancy unchanged — establishes exactly that the transformation happened, and it is entirely
compatible with no time change at all. A mechanism that is visible in the emitted code, is real, and is the
only difference you can find against a faster comparator is **still not a measured win**. So the
structural read never substitutes for the timed leg, and a round that reports the kill as its result has
reported its own instrument (`## Check the instrument before you read the scale off it`).

Only one direction holds, and it is the cheap one:

- **Mix unchanged ⇒ stop.** No timing run can separate the arms, and any delta measured against them
  belongs to the harness.
- **Mix changed ⇒ nothing follows about time.** Go to the timed leg, run it against the arm's own control
  (`## A control proves exactly the component it cancels, and nothing else`), and let it decide.

**Evidence to demand:** the structural read *and* the timed comparison, reported as two separate lines. A
write-up carrying only the first has stopped at the falsifier and called it a price. Before the timed leg
can be trusted, the arm also has to move only the quantity you are about to attribute the result to — the
next section.

**One layer below this:** a structural read may not even be measuring the quantity you think it is.
Counting instructions of a given width counts what the *issue* side asked for, which is a different
quantity from what the *access* side moved, and the two come apart in the ordinary case rather than a rare
one (`../tile-programming/memory-path.md ## A load's width is issue-side; a transaction's width is
access-side`). Check that the census names the quantity the bound is stated in before treating a
completed census as a falsifier at all.

**One layer below that again, on a different axis:** the census can name the right quantity and still be
read off an artifact that **cannot represent it**. Compiled ISA is **per-warp code**, so a quantity that
lives *across* warps — the same bytes fetched redundantly by every warp in the workgroup — has no
instruction to count. It is not undercounted or miscounted, it is **absent at any count**, and an ISA diff
proposed to detect cross-warp replication therefore returns a zero delta that is **uninformative, not
negative**.

The trap is that such an instrument can still look alive. In one measured instance the ISA counts *did*
move as the warp count was swept — but what moved was a per-warp tile dimension shrinking, not the
cross-warp effect being probed. **An instrument that moves for the wrong reason is worse than one that
does not move**, because responsiveness reads as validity and nothing else on the dashboard disagrees.

- **Name the quantity, name the artifact's basis, and confirm the artifact can express the quantity** —
  before a structural read is trusted as a falsifier at all. Per-warp code cannot express a per-workgroup
  redundancy. This is the same basis confusion that elsewhere turns a per-warp instruction count into a
  per-block work multiplier (`profile.md`, "Rule: the coverage ratio disqualifies a config, it does not
  price one").
- **When the basis is wrong, look for an artifact whose basis *is* the quantity.** In that instance it was
  readable straight off the compiler's own layout object, where replication appears as an all-zero basis
  vector — `rep = 2 ** (count of zero warp bases)`. That is a direct read of the quantity, not an
  inference from emitted code.
- **Enumerate the surface rather than sampling the shipped points.** Exhaustively evaluating the layout
  constructor's inputs turns a belief into a bounded claim: in that instance two shipped points agreed
  with a rule that failed in **45 of 160** enumerated cases.

**And one layer the other way:** when the timed leg is run on a probe rather than on the kernel, a clean,
well-controlled measurement can still carry the wrong sign (`## A result measured in isolation does not
transfer to the kernel`).

## Before timing, list every quantity the edit moves

An arm is cut to move one quantity. What it actually moves is whatever follows from the source change, and
a second quantity riding along is not visible in the result — it is summed into it.

> **A null is not evidence of "no effect" when the arm moved more than one thing. It is equally consistent
> with a cancellation between two large effects of opposite sign, and the timer cannot tell you which one
> you got.**

The shape to recognise: widening a per-lane access does remove the narrow accesses you aimed at, and,
because each access is wider, it also covers more of the reduction axis per step — so the bytes actually
read go *up* while the narrow accesses go away. Both effects are real, they point opposite ways, and the
measurement is their sum. The same arm read as a single-quantity edit yields "the mechanism is worth
nothing", which is a conclusion the data does not support in either direction.

So the list belongs **before** the timing run, not beside the result:

1. Write down every quantity the edit moves — the one you aimed at, and every one that follows from it
   arithmetically: bytes moved, coverage per step, occupancy, instruction mix, launch geometry, layout
   conversions inserted elsewhere.
2. For each of the others, show it unchanged, or bound how far it moved and carry that bound into the
   conclusion (`## When the gate needs a number you cannot read, bound it — do not guess it`).
3. If a second one moved and you cannot bound it, **the arm cannot answer the question and has to be
   re-cut** until it moves one thing. Running the timer anyway produces a number that is not about the
   mechanism you named.

This is `## Two criteria can collapse into one on the data you actually got` from the other end. There, two
quantities you designed to be independent turned out to be one, so the corroboration was empty; here, one
quantity you believed you were moving turns out to be two, so the attribution is empty. Both are fixed by
writing the relationship down instead of assuming it — and in this case the writing-down is free, because
it is a property of the edit rather than of the data.

A third way the timed leg can be sound and still not answer the question: the thing you timed was not the
kernel (`## A result measured in isolation does not transfer to the kernel`).

### A knob that is off still has to prove the shipped stream did not move

Putting a new path behind a default-off `constexpr` is the standard way to keep an arm cheap (`## A
compile-only kill step is a falsifier, not a price`), and it is free by *semantics*: the default path
computes the same values, so the correctness oracle stays green and nobody lists the shipped configuration
as a quantity the edit moved. It is not free by *schedule*. The compiler sees the new code before it
decides anything, and operand evaluation order, register lifetimes and instruction placement **in the
untouched default path** can come out different.

> **After adding a default-off knob, the shipped configuration's instruction stream must still hash to its
> pre-knob value. Compare against the same source with the branch *deleted*, not with the branch present
> and its predicate set to zero — the second comparison holds the compiler's input constant, so it can
> only come back green and proves nothing.**

Hash a *filtered* stream: strip assembler directives, local labels and comments, join what is left, and
compare digests. Counter summaries are not a substitute, because every count that a register or
opcode-class gate reads is order-insensitive (`## Every within-comparison stays green when both arms run
the wrong thing`). In one measured instance, a spelling that hoisted two operand loads above their uses
added **6 `s_waitcnt`** to the shipped configuration while VGPR, AGPR, SGPR, spill, LDS bytes and every
arithmetic, `ds_*`, `buffer_*` and matrix-op count stayed **identical** and the output stayed bit-exact:
the occupancy gate passed, the per-opcode diff passed, and the timing table on file no longer described
the kernel that shipped. Keep `s_waitcnt` in any per-opcode diff — when only the schedule moved, it is the
line that moves — and prefer spellings that keep each operand's evaluation inline at its use site.

## A result measured in isolation does not transfer to the kernel

An isolating probe — the access pattern alone, on the real tile, in the real loop shape, with nothing else
running — is the cheap way to size a mechanism before committing to it. What it removes along with the
noise is everything the kernel does *concurrently with* that mechanism: the other streams contending for
the same cache and the same address path, the work in flight that hides or exposes its latency, the
register and occupancy pressure the rest of the body imposes, the issue slots it competes for. Those are
not second-order corrections to the probe's number. They are what decides its sign.

> **An isolating probe can size a mechanism. It cannot predict the sign of the change in the kernel.** A
> probe can be faithful in every dimension you are able to name — layout, tile shape, warp count, loop
> structure, working-set size — reproduce a named mechanism closely enough to look like a textbook
> confirmation, and still be inverted by the kernel it was built for.

Two failure modes, with different cures:

- **The probe's structure is not the consumer's.** The probe decides how many times a line is touched
  before it is abandoned; the kernel decides that too, and differently. Ask what the real consumer does
  *between two accesses to the same line*, and whether the probe gives it the chance to do it. Ask it when
  the probe's number is plausible, not only when it is implausibly large — believability is not transfer.
  This one is fixed by re-cutting the probe to the consumer's structure.
- **The probe's environment is not the kernel's.** A consumer-faithful probe still runs without the
  contention, the in-flight depth and the occupancy of the real body. This one is **not** fixed by making
  the probe more faithful. It is fixed by building the edit.

What has to be reproduced in the kernel before a probe's result may be acted on:

1. **A probe result authorises an attempt, never a number.** It may decide whether an edit is worth
   building. It may not appear in a report as the edit's value, and it may not be quoted as a margin.
2. **The reported number comes from the kernel**, under this page's protocol, against the arm's own
   control (`## A control proves exactly the component it cancels, and nothing else`), with the arm cut so
   that it moves one quantity (`## Before timing, list every quantity the edit moves`).
3. **A confirmed mechanism is not a confirmed effect.** A prediction that matches the probe closely raises
   confidence in the *mechanism* and says nothing about what it is worth in the kernel — the same
   asymmetry as `## A compile-only kill step is a falsifier, not a price`, arrived at from the measurement
   side instead of the structural side.

**Evidence to demand:** the probe and the in-kernel measurement as two separate lines, with the probe
labelled as a feasibility result. A write-up that carries only the probe has reported a number from a
program nobody is shipping.

The cost of this is bounded by how the edit is built, not by how careful the probe was: an arm that lands
as a default-off switch and is proved bit-exact before it is timed costs a build if the sign inverts, and
nothing else. That makes a sign inversion a **measured answer** to the question that was asked, which is
worth more than a paragraph of inference agreeing with the probe.

---

## Red flags

- baseline and candidate launched from different script locations with implicit imports;
- the benchmark script lives under one candidate tree and benchmarks another without explicit import
  control;
- cache/artifact/environment selection not recorded;
- same JSON schema but different boundaries;
- baseline drift on a shared GPU larger than the claimed speedup;
- same-named modules/kernels reused across candidates without process/cache isolation;
- an acceptance number taken with a screening protocol (min statistic, hot cache, in-process interleave,
  batched wall-clock, write-evict) or against a stored baseline;
- a keep below `MIN_IMPROVE` (2 %), or a measured noise band used to *lower* the threshold;
- `HIP_VISIBLE_DEVICES` inlined into a timing/profiler command instead of running under `gpu_lock.sh`;
  a per-device claim without a UUID / PCI anchor.

## Output

```text
COMMANDMENT.md                 # GEAK's frozen task contract (benchmark_engineer)
plain_baseline_metrics.json    # written after the plain profile (the target line)
harness.py (or reused project harness; harness_lib-timed)
```

## Sources

Merged into this chapter (old paths, relative to the skill root): `references/benchmark-hygiene.md` (all
sections) and `references/phases/harness.md` (all sections), plus the binding GEAK measurement facts from
`e2e_workflow/scripts/harness_lib.py` (`time_op`, `cache_policy`, `_time_events`, `_host_dispatch_ms`,
`measure_legs`, `assert_legs_differ`, `check_random_vs_baseline`) and the commit gate in
`kernel_workflow/kernel_lane.js` (`MIN_IMPROVE`).

Rewritten / demoted under the benchmark rule (nothing silently dropped):

- **min as the acceptance floor** (`Shared-box contention`, `Repeatability + measurement order`) →
  demoted to *search/screening only (ab_bench)*, reported beside the median; acceptance statistic is the
  `harness_lib` median. The pack's own "do not switch estimator mid-campaign / forward-only" rule is kept
  and now argues for staying on the median.
- **batched wall-clock as the default sub-ms acceptance anchor** (`ROCm timing`) → demoted to a
  diagnostic cross-check; acceptance = events with per-sample sync (batching rejected on measurement in
  `harness_lib`).
- **hot-cache protocol for compute-bound kernels** (`Hot or cold cache`) → *search/screening only
  (ab_bench)*; acceptance is always read-evict. "Flush a buffer" made explicit as **read**-evict;
  write-evict not used (GEAK measurement 1.40 vs 1.12).
- **graph-timing "one sync around many replays"** → kept for wall-clock graph timing; acceptance
  `time_op(graph=True)` uses events per replay with sync/flush outside the window.
- **one process / interleaved cells for all variants** → screening only; acceptance = fresh process per
  leg; patched variants always in their own process; the in-process per-variant cache-tag interleave is
  not used.
- **Small-gain policy "<2% needs a repeat"** → rewritten to the GEAK commit gate (<2 % is never banked; a
  measured band only raises the bar).
- **Sub-ms warmup/reps/noise table** → kept as screening guidance and launch-floor evidence; cannot lower
  the 2 % gate.
- **LDS pre-screen** rewritten gfx950-first (160 KiB / 64 banks) with gfx942 (64 KiB / 32 banks) as the
  downgrade; data path → `perf_knowledge/hardware/data/hw_constants.json`.
- **JSON schema** `latency: mean / min / max` → adds `median (acceptance)` and the `harness_lib` receipt
  fields (`timing_method / cache_condition / statistic`, `primed / host_ms`).
- **Cross-version third arm** clarified: `num_stages=1` on the plain source (dead knob on the Gluon path in
  Triton 3.8.0).
- "clear the candidate cache" / "clean up `gpucore.*`" → "move aside" (no `rm` in GEAK-run scripts).
- The EMBEDDED / STANDALONE two-column table (STANDALONE = the upstream `gluon-direction` agent on the pack's
  `toolctl` spine) collapsed to one GEAK column naming the roles (tech_lead, deep_engineer,
  benchmark_engineer, verify_engineer, Director); the heading `### Two run modes` renamed `### Who owns what in GEAK` (unpinned).

Kept pack-only techniques: control arm (`--control`), `--permute`, `fingerprint()`, `--preload`,
UUID/PCI anchoring, sustained idle window, sub-ms warmup table (screening), graph-served kernel-only
metric, instrument floor `F`, determinism race-test, and every evidence-reasoning section.
Pointer updates: `phases/harness.md` / `../benchmark-hygiene.md` → this file; `evaluate.md` → `close.md`;
`escalation-gate.md` → `entry.md`; `plain-strategies.md` → `front-end.md`; `experiment-records.md` →
`records.md`; `phases/profile.md` → `profile.md`; `phases/transcribe.md` → `transcribe.md`;
`scripts/gpu_lock.sh` → `kernel_workflow/scripts/gpu_lock.sh`; `capture.sh` / `dump_ir.sh` →
`kernel_workflow/scripts/kernel_tools/`; `hardware/hw_constants.json` →
`perf_knowledge/hardware/data/hw_constants.json`.
