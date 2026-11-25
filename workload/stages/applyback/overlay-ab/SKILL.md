---
name: overlay-ab
description: Integrate qualified kernels through a reversible overlay, prove their actual serving use, and hand the runtime to Evaluate when comparison is requested.
---

# overlay-ab

Read [Apply back](../README.md) for integration and effect evidence.
[Evaluate](../../evaluate/README.md) owns serving comparisons. The sequential applyback
stage composes both; a direct integration call need not perform the whole comparison.

Use the harness that produced the reference. `harness_cmd`, when present, selects that
harness and its supported `python_overlay` hook; otherwise use GEAK's bundled
`e2e_workflow/scripts/bench_e2e.sh` with the `agentx` client. Reference and candidate use
the same native protocol. Harness-specific fields belong in scenario inputs; integration
and comparison keep their shared module criteria.

## 1. Build one overlay

`site-packages` is never modified. GEAK's `e2e_workflow/scripts/overlay_setup.py` builds a
sitecustomize overlay through the launcher's supported Python-overlay hook. A fresh launch
without it restores the original stack; changing an environment variable does not undo
patches already loaded in a running process. Prefer the task's shipped `overlay_setup.py`,
which matches its extraction version.

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
resolved, not what the serving workers loaded.

If loading evidence reveals a shared registration collision, declare the qualified
owner's dependency before consumer execution through the normal loader, preserving the
consumer API. Verify which implementation registered; a matching name or schema alone
does not establish candidate selection.

For extension exports, retain the actual loaded module from the declared reference path
and verify that its member is the executed callable. Preserve and recheck that owner
through removal and collection; dynamic loaders can return unregistered modules.

**Lazy-load proof comes from the serving workers.** Verify the selected source identity
and callable references actually used by every serving rank in the declared deployment
after representative requests. Launcher hashes, a coordinator import or a UT pass cannot
establish that claim. Prefer existing native loading/execution evidence. Add instrumentation
only for a material gap, scoped to that gap; diagnostic completeness is not automatically
integration acceptance.

When `serving_source_proof` selects an observer, review its producer file and hash before
launch. Reuse a compatible, qualified collection/removal lifecycle; otherwise resolve it
within the affected call before repeating deployments, without adding a universal admission
stage. Retain all required rank evidence. Remove instrumentation before measured traffic;
record restoration and recover any affected runtime state. A broken observer leaves the
claim unproved: return the gap and cost to Brain for a different evidence method or a separately
scoped observation, without promoting the failed proof to acceptance.

Environment variables set around a host command do not automatically reach a container.
Check the runtime boundary before launch: the native launcher must forward the proof
directory and record the actual cache placement it used.

## 3. Hand off or compose Evaluate

Return the integrated runtime and effect evidence to [Native E2E](../../evaluate/native-e2e/SKILL.md)
with the task's harness, quality and comparison protocol. Use its native readiness/smoke
contract; reuse compatible evidence rather than adding another universal readiness stage.

A changed overlay needs a runtime that actually loads it. An unchanged, qualified deployment
may serve independent repeated clients when the native protocol allows reuse; a new result
identity does not by itself require a new model startup. Preserve the protocol's independent
arm/cache boundaries and equivalent stock/candidate conditioning. Code-correctness evidence
and process-specific loading evidence have different reuse conditions.

For a composed full comparison, supply Evaluate's declared summary and validity artifacts
and preserve its configured repeats, fresh stock control and native quality requirements.
A scoped screen returns its narrower evidence and limitations; it does not admit a replacement.

## 4. Do not grade yourself

For the composed driver delivery, `../finalize.py` checks integration and delegates
comparison to Evaluate's finalizer before writing `result.json`. These owners and the
scenario define acceptance; do not copy their predicates into another validator or expand
them with unrelated diagnostics. Direct calls follow the module contract for their scope.

If integration fails, preserve the failed arm and send its counterexample to the owning
profile or optimize agent. A changed kernel needs new cold evidence before the overlay is
rebuilt, with fresh arm and cache identities. Reuse a completed arm only when its source
inventory, configuration and native validity still match.

## Outputs

Integration returns `$OUT/overlay/`, `$OUT/applied.json`, `$OUT/shadow_check.json` and any
configured actual-loading proof. A composed full comparison also supplies Evaluate's
`e2e/bench_summary.json`; the owning finalizers write `ab.json` and `result.json`.
