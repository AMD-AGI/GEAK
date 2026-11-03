---
name: overlay-ab
description: Install cold-re-verified kernels back into the serving stack through one reversible overlay, prove the overlay is what the serving ranks actually loaded, re-measure end to end, and A/B against the baseline — operator speedup and end-to-end delta reported as two separate numbers.
---

# overlay-ab

The applyback stage, and the **only** place in the pipeline permitted to claim an
end-to-end gain. Read `../README.md` for the slot contract, then do the work below.

Two benchmarks can drive it and they are **one parameter, not two skills**:

- `harness_cmd` absent → GEAK's own `e2e_workflow/scripts/bench_e2e.sh` with the
  `agentx` client.
- `harness_cmd` present → the reference harness that produced the baseline, with the
  overlay handed to it through its own `python_overlay` hook. Use this whenever the
  baseline was not measured by `bench_e2e.sh`: both legs must come from the same command,
  or the A/B is measuring the two harnesses against each other.

Everything else — building the overlay, proving it loaded, the arms, the verdict — is
identical, and the field names of any particular harness are scenario config, not
knowledge that belongs in this directory.

## Why this stage cannot be skipped

The real failure mode here is not "no speedup was found". It is **an operator win with no
system movement**, from this machine's own history:

| Change | Operator / microbenchmark | Whole-machine e2e |
| --- | --- | --- |
| FlyDSL MoE sorting | 1.80× | **+0.2356%** |
| MoE split4 | 1.342× (local replay) | **−0.0872%** |

Both were genuine, both were correct, neither moved the machine. Optimize produces
**candidates**; only this stage produces results.

## 1. Build one overlay

`site-packages` is never modified. GEAK's `e2e_workflow/scripts/overlay_setup.py` builds a
sitecustomize-driven reversible overlay that takes effect solely through
`OVERLAY_PYTHONPATH`; remove that one variable and the stack is original again. Prefer the
`overlay_setup.py` copy shipped inside a task directory over the repository's — it is the
version that was current when the task was extracted.

How each kernel binds back is **not guessed**. The extraction skill already recorded it in
`meta.json["candidate_bind"]` as `{kind, target|module, impl_module, impl_attr, file}` —
the same descriptor it uses for its own candidate leg. Copy it and replace only `file`
with the optimized source.

Before touching an overlay, check each kernel's identity against what optimize recorded:
`task_meta_sha256` against the task's `meta.json`, and `kernel_src_hashes` against the
selected `kernel_src` (or legacy `best_kernel_src`) tree. A row with no recorded hashes
cannot be installed — hashing today's working tree does not retroactively verify anything.

**Chain-seed the overlays; never mount them side by side.** `PYTHONPATH=a:b` imports only
the first `sitecustomize` on `sys.path`, and the second overlay directory silently has no
effect. Pass each added kernel `--from` the previous overlay, so all of them end up in
one.

**`candidate_bind.file` is authoritative.** For a leaf that is HIP C++ it points at a
delivery module rather than at `kernel_src`. Override it with whatever looks editable and
you install the `.py` of a kernel whose edit lives in a `.h`: the overlay shadows the
dispatcher with an identical dispatcher, the image's prebuilt `.so` keeps serving every
token, and **the shadow check below still passes**. The e2e that follows measures the
baseline against itself. This has happened.

A kernel spread over several files is delivered through one module. Record in
`applied.json` what that module is expected to pull in, so a reader can tell at a glance
whether the `.cu`/`.h` actually travelled with the `.py`. Keep each task's complete
`kernel_src` under a stable per-task directory and re-export rebind entries from there, so
`__file__`-relative companion lookups still resolve. Declared
`candidate_bind.requires_env` paths and scenario `binding_options.<task>.source_env` supply
the source-root hooks; install them **before** the overlay imports any patched module, and
keep every value inside the task's own bundle.

Write `$OUT/applied.json`: what was installed, with its target and verified operator
speedup; what was skipped and why; and the hash of every file in the overlay.

Two hygiene notes, both learned from failures:

- Keep the overlay invisible to `importlib.metadata`. A setuptools `_vendor` directory
  reachable from the overlay poisons distribution discovery in the serving process. This
  belongs upstream in `overlay_setup.py`; until it is there, check it.
- Native builders can write build outputs next to `__file__` during an import. Exercise
  local binding checks in a **preserved mirror** of the overlay, never in the accepted
  overlay itself, so the accepted source stays pristine for independent per-arm copies.
  No symlinks: each arm gets a real, independent copy and its own case registration.

## 2. Prove the overlay is what actually ran

Run `overlay_setup.py check` and save its output as `$OUT/shadow_check.json`. Trust the
tool's own verdict line; do not parse the last line of its output as a path. `check`
prints `<module> -> <file>` and then a *separate* verdict line, so taking the last line as
the path judges a correctly installed overlay to be missing — it fails closed and aborts a
run that was in fact fine. `find_spec` alone is not enough either: it can point at a file
that raises on import because its nested computation is missing. Exercise the real import
before spending hours on e2e.

A local callable-identity check is **preflight, not proof.** It says what this process
resolved, not what eight serving ranks loaded.

**For a module loaded lazily, the proof has to come from the serving workers.** Verify the
source path and hash inside each rank *after* a warmup request has exercised that path. A
parent-process import or a pre-warmup check establishes nothing. Where the scenario
configures `serving_source_proof`, prepare and review the one-shot observer it names
before starting any arm — its `producer_file` and reviewed `producer_sha256` are
preconditions, and the builder installs that exact file as a bootstrap in each accepted
source copy. The observer must collect from **every** serving rank after lazy loads, look
at the callable references callers actually used, and **remove itself before measured
traffic**. Launcher hashes, coordinator imports and standalone UT receipts are not serving
proof. Record the restoration; never leave instrumentation in steady-state timing.

Environment variables set around a host command do not automatically reach a container.
Check the runtime boundary before launch: the native launcher must forward the proof
directory and record the actual cache placement it used.

## 3. Measure

Smoke first: `e2e_workflow/scripts/agentx_smoke.sh`. `aiperf` and InferenceX are on
neither `$PATH` nor `/opt` on this machine. When pieces are missing, report `blocked` and
name them. A throughput number produced before smoke passes measured something else.

**`REUSE_SERVER=0`.** The overlay has to be loaded by a freshly started server; reusing a
running one measures an unpatched stack.

Measure a **fresh stock control** as well as the candidate. The historical
`reference_run` receipt reports drift; the fresh control is the A/B denominator. A long
optimization run separates the candidate from its original reference, and an unchanged
control is the only thing that exposes environment drift in between. Use the same cache
placement and freshness policy on both sides — each arm gets independent
Triton/XDG/Inductor/torch-extension directories under the configured `cache_root`.

Before the measured run, exercise representative real serving paths in a **bounded,
unmeasured preflight**: the extracted tasks determine which requests, including dtype,
phase and layout variants. Give stock and candidates identical conditioning, record it
separately from the historical reference, and make observer removal a barrier the
measurement cannot start without. A standalone UT pass does not establish that the serving
caller selected the same branch.

Repeat `repeats` times and take the median. Keep valid slow repeats in the statistic —
dropping them is how a regression becomes a no-change. Report invalid runs with their
receipts and reasons; never delete or reuse a benchmark ID. Run cancellation, request
errors and the harness's end-of-window drain behaviour are three different things:
preserve the native verdict and counts rather than inventing an admission rule after the
fact.

Write `$OUT/e2e/bench_summary.json` with the candidate throughput, its `metric_basis`, the
repeat count, and the per-arm records. Whichever benchmark ran, this one file is what the
verdict reads.

## 4. Do not grade yourself

`../finalize.py` writes `result.json`, after this session, and it is the stage's
conclusion (`stages/CONTRACT.md` 3.1). It checks the shadow proof, the source hashes
against optimize's record, whatever native-validity fields the scenario configured, that
both legs share a metric basis, and the A/B itself. What it requires is written only
there. You can run it early to see what it will say.

If integration fails, preserve the failed arm and send its counterexample to the owning
profile or optimize agent. A changed kernel needs new cold evidence before the overlay is
rebuilt, with fresh arm and cache identities. Reuse a completed arm only when its source
inventory, configuration and native validity still match.

## Outputs

`$OUT/overlay/` (reversible), `$OUT/applied.json`, `$OUT/shadow_check.json`,
`$OUT/e2e/bench_summary.json`, and — written by the slot, not by you — `$OUT/ab.json` and
`$OUT/result.json`.
