---
title: Reviewing kernel changes — advisory findings with deterministic evidence
kind: workflow
gens: [gfx942, gfx950, gfx1250]
status: competitive
updated: 2026-09-21
source_commit: ROCm/aiter@04c7b808
sources:
  - ROCm/aiter@04c7b808:.claude/skills/review-pr/SKILL.md
  - ROCm/aiter@04c7b808:.claude/skills/review-pr/rules.md
---

> **Ingested as a portable workflow, not as the upstream skill.** aiter's `review-pr` skill is an
> executable agent procedure wired to that repo: `fetch.sh`, `triage.py`, a `$WORK` artifact
> directory, six machine gates, and a file-tier table naming aiter paths. None of that runs here and
> none of it is reproduced. What is carried over is the part that holds for any AMD kernel PR: the
> evidence tiers, the defect catalogue, the AI-code structural checks, the refutation discipline, and
> the finding format. Every rule below is stated as a pattern with its measured instance cited, not
> as a checklist id — upstream's `A1`/`B8`/`D9` labels are internal and deliberately not used.

# Reviewing kernel changes

## TL;DR
A kernel-PR review produces **two different kinds of statement and must never blur them**. A
*reviewer judgement* — "this looks like it reads past the allocation" — is **advisory**: stochastic,
useful for directing attention, and never a merge gate. A *reproducible result* — a correctness or
base-vs-head perf run that ships its own reproducer — is **deterministic** and may gate. The single
most common failure is an advisory finding written in the grammar of a deterministic one.

Three rules carry most of the value. **Name the triggering input** — a high-severity finding that
cannot state the concrete shape, dtype, arch or value that makes it fire is unproven, and must be
downgraded to a question. **Tag every finding `[verified]` or `[inferred]`** and never ship an
inferred root cause as fact. **Try to kill each finding before reporting it**, then hand it to
someone who has not seen your reasoning — upstream measured that an independent reader still killed
**28%** of findings that survived self-refutation.

## Preconditions
- A diff resolved against the **merge target**, not your local worktree. A stale branch's CI result
  describes a tree that has moved; anchor line numbers against the base you actually compared.
- The base tree checked out somewhere you can read. Most false findings die by reading one function
  on base that the diff never showed.
- For any perf statement: an idle, clock-locked GPU of the right arch, or an honest `NOT RUN`.
- Knowing which claims you are allowed to make. If nothing was run, **no finding may assert runtime
  behavior — perf, accuracy, or launch failure — as fact.**

---

## 1. Establish what changed, semantically

Answer these before touching any checklist. Upstream found that a review which skipped them was
*textually identical* to one that did them, which is why they are written down rather than implied.

1. **What changed computationally?** Not "improves perf" — which algorithm, formula or data flow.
2. **Hardware scope.** Which arch (gfx942 / gfx950 / gfx1250), which precisions, which phase
   (prefill / decode / both)?
3. **Does this change a public API?** A new exported symbol, a new kwarg, a changed default.
4. **If perf is claimed, what is the mechanism?** Not "faster" — fewer HBM round-trips, fewer
   launches, better tiling.
5. **Does the description explain *why* or only *what*?** "Fuses kernels for speedup" is surface;
   "eliminates the intermediate HBM write between rmsnorm and quant" is understanding. Surface-only
   descriptions correlate with unreviewed AI-generated code — treat as elevated risk.

## 2. Assess blast radius before reading line by line

Rank each changed file by what breaks if the change is wrong, and write the reason down naming
*what this PR changed* — "core file, large blast radius" is equally true of every PR ever opened
against that file and is not an assessment.

| Tier | Test | Failure mode |
|---|---|---|
| **1 — system-critical** | If this file fails to import, does the whole package still import? | Every op fails to load |
| **2 — op-class critical** | Does it hold the dispatch that selects which kernel runs for an op used by more than one model family, or is it the public API for an op? | Wrong result for *all* users of that op |
| **3** | Everything else — an individual kernel or model-specific code | Consumers of that one op |

Two portable refinements upstream had to add after the fact:

- **A C++ header is Tier 2 once ~10 or more translation units include it.** A tier table listing only
  Python files lets a pure-C++ PR reach the end of a review with nothing declared. Count fan-in from
  the tree rather than maintaining a list, so a header that becomes widely included joins on its own.
- **Do not put every op wrapper in Tier 1.** It is technically true that a package doing
  `from .ops.x import *` breaks on any one of them, but with 200+ wrappers a Tier-1 rule written that
  way fires on **71%** of PRs and the assessment stops being performed at all. A wrapper's real blast
  radius is its own op.

## 3. Sweep the defect catalogue

These are the patterns that recur in AMD kernel PRs, each with the instance that produced it. They
are grouped by mechanism because that is how you search for them.

### 3a. Coverage gaps — the sibling still has the bug
The fix changed an address calculation, bounds check, type widening or layout in one kernel; the same
file holds a variant (`_opt`, `_prefill`, `_decode`, `_v2`, `_fast`) that still carries the line this
PR changed. Scope the search to **variant functions in the changed file**: upstream tried the
whole-tree version (a deleted line still present verbatim elsewhere) and dropped it, because
whole-file verbatim identity cannot separate a sibling needing the same fix from two files sharing
boilerplate. Also check that a dispatch condition newly enabling a kernel does not enable it for more
archs or models than were actually validated.

### 3b. Silent bypass — the input reaches the wrong branch
- **Dispatch gate with an unchecked parameter.** For every parameter gated off by a new branch: is it
  *asserted* or *forwarded*? If neither, the result is wrong with no crash and no error. Recurring
  culprits: `dropout_p`, `window_size`, `block_table`, `logits_soft_cap`, `alibi_slopes`, `is_causal`.
- **Unmasked `tl.load` / `tl.store` in Triton.** Triton does not bounds-check; an unmasked tail tile
  reads or writes past the tensor. Before firing, confirm the axis is *not* a compile-time multiple of
  the block and not already clamped — and name the non-aligned dim. A masked load with **no `other=`**
  is a separate, subtler bug: masked lanes default to zero, which is fine for a sum and wrong for a
  max reduction.
- **String dispatch without normalization.** `quant_type == "per_token"` silently misses
  `"fp8_per_token"`, `"per-token"`, `QuantType.per_Token`.
- **A new dispatch value no branch handles.** A new dtype string, arch string or layout flag must be
  handled, fall through to a documented safe default, or assert. Severity depends on whether the wrong
  path is silently incorrect or merely suboptimal.
- **API propagation stopping short.** A new parameter accepted but never used in the body; a removed
  or renamed public symbol whose cross-repo importers were not updated; a new compile-time constant
  missing from the JIT repr key, so a stale binary is served. Before firing on a rename, check for a
  compatibility shim added in the same PR.
- **A buffer descriptor bound that is not the tensor's live extent.** In FlyDSL,
  `make_buffer_tensor(t, max_size=True)` / `create_buffer_resource(t, max_size=True)` declares the
  buffer as large as its *allocation*; when the bound dimension is a runtime extent — M, a token
  count, a per-expert count — the hardware `num_records` field then permits reads past the live data:
  silent garbage, no fault. Full-size tensors (weights, caches, `block_table`) are correct with
  `max_size=True`. See [`../languages/flydsl/authoring_api_migration.md`](../languages/flydsl/authoring_api_migration.md) §2.

### 3c. Hardcoded arch or dtype assumptions
- **Never infer a dtype dialect from the arch string.** `if "gfx942" in arch: treat_as_fnuz()` is
  wrong because one arch can have both fn and fnuz in flight. Gating a *conversion* by arch is fine;
  *inspection* must compare `tensor.dtype`. Full hazard:
  [`../quantization/fnuz_vs_ocp.md`](../quantization/fnuz_vs_ocp.md).
- **Do not hardcode the FP8 saturation bound.** `240.0` is right for e4m3fnuz and wrong for OCP e4m3
  (448). Derive it from the dtype unless the path is already runtime-guarded to one flavor.
- **A new arch string literal in a dispatch condition** should route through the arch registry or a
  named constant — but check the unchanged lines first: if that string already appears in the file,
  this is pre-existing style, not a new violation.

### 3d. Uninitialized and boundary state
- **Atomic reduction into an uninitialized buffer.** `atomic_fmax(*ptr, val)` is
  `*ptr = max(*ptr, val)`; if `*ptr` came from an `empty()` allocation, garbage dominates the max, and
  a corrupted amax silently corrupts every FP8 descale downstream.
- **int32 overflow in index arithmetic.** A multiplication feeding pointer or index arithmetic where
  an operand is a *runtime* parameter and nothing is widened to 64-bit. `token_id * num_heads *
  head_dim` overflows past `token_id > 16M` at H=32/D=128. Upstream's hard-won lesson: **do not
  reduce this to a name list.** A version listing `token_id`/`seq_start`/`batch_offset` missed three
  real defects that used `stride_out_batch`, `block_id` and `physical_block`, and a name-matching
  scanner reported 0 of 4 real lines on one PR while emitting 390 candidates. The trigger is
  structural — compile-time tile constants are excluded because they bound the product.
- **An invariant reversed without a citation.** An old comment says "must X because Y"; the new code
  removes X claiming "X not needed" and cites no spec, asm or test. `zeros() → empty()`, a deleted
  assert, a removed `.contiguous()`.
- **A fake / meta function that disagrees with the real op.** If the real op's return dtype, shape or
  arity changes and the `_fake` does not, torch.compile infers the wrong types — it compiles cleanly
  and asserts or miscomputes at runtime. A new custom op with *no* fake is the same failure.
- **A wrapper that never checks contiguity** before handing a tensor to a C++/HIP kernel. A slice or
  `.T` produces wrong addresses, silently.

### 3e. Cross-repo and resource hazards
- **A change the consumer has not been told about** — a new symbol or kwarg with no linked consumer
  PR, a changed KV layout a bridge module reads directly, or a new parameter whose
  backward-compatible default means the fix never activates until someone downstream passes it.
- **Downstream CI skipped on a change downstream consumes.** When downstream jobs are label-gated and
  off by default, a PR can pass every check with the consuming job skipped, merge green, and break the
  consumer invisibly. Pick the *minimal* label that covers the affected model, not reflexively "all".
- **A new weight variant pinned alongside the original** — `w_preshuffled` stored next to `w` doubles
  HBM for that weight unless the original is freed.
- **Multi-stream use with no synchronization.** A tensor produced on one stream and consumed by a
  kernel on another, with no `wait_stream` / event between them, reads garbage silently.

### 3f. Triton and Gluon specifics
Beyond masking (§3b): an accumulator left in fp16/bf16 over a long K loop loses the tail of the sum,
and `tl.dot` inputs downcast to save registers change results silently — accumulate in fp32 and cast
once at the store. The host launch grid and the in-kernel `program_id` mapping are computed
independently and nothing checks they agree, so a host-side `cdiv` against a kernel assuming exact
division leaves the tail tile unwritten — correct everywhere the tests look. And `num_warps`,
`num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`, `kpack` are per-shape *and* per-arch: a config
tuned on gfx942 is not tuned for gfx950, `num_stages` above what LDS holds silently drops occupancy
rather than failing, and a knob written as a literal in kernel source is invisible to the next tuning
sweep. Knob semantics: [`../languages/triton_amd/knobs.md`](../languages/triton_amd/knobs.md).

### 3g. Hardware limits worth checking by eye
- **Large LDS allocations.** gfx1250 exposes 320 KB, while an ordinary `ds_read`/`ds_write`
  immediate offset covers only 65535 bytes. That is not a 64 KB allocation limit—the VGPR base
  supplies the rest of the address—but crossing the immediate window can add address arithmetic and
  VGPR liveness. Inspect the ISA and resource report before claiming either a spill or a regression.
  See
  [`../optimization/lds_and_bank_conflicts.md`](../optimization/lds_and_bank_conflicts.md).
- **Triton `BLOCK_SIZE` against the LDS budget** — a large block pushes LDS over the limit and shows
  up as a batch of config test failures rather than as one clear error.
- **Named tile constants.** MFMA shape constants scattered as raw `16` / `32` through a FlyDSL or asm
  kernel should be named; this is a real review request, not a style preference, because the numbers
  are arch-dependent.

## 4. Run the AI-code structural checks

The description-level tells — clean round numbers like exactly 2.0×, screenshots instead of values,
tests only at M=1, a template "Test Plan" left unfilled, an AI attribution footer, gated-off
parameters silently ignored, module-level `sys.path` / `os.environ` mutation, unrelated files
committed alongside the change, a new default path with no env-var gate to revert it — are a cheap
pre-filter. Three or more of them, **or any structural check below firing**, warrants an explicit
"elevated AI-code risk — verify dispatch logic and test coverage by hand" note. **A clean,
well-written description is itself something AI produces easily**, so when the diff changes code
these structural checks are mandatory regardless:

1. **Unresolved imports and symbols** — resolve every first-party import against the merge target. A
   miss is usually a *rebase* signal rather than an invented API; upstream's example was valid when
   written and deleted from main 19 hours before merge. New kwargs, attributes and enum members need
   checking by hand.
2. **Twin divergence.** Compare mirrored code field by field — fwd/bwd, v2/v3, prefill/decode,
   gfx942/gfx950. Any asymmetry (one side int64 and the other int32, one masked and the other not, a
   flipped stride order) is a half-finished copy. This is *the* signature AI kernel bug.
3. **Claim ↔ code, and number provenance.** Does the code enforce the invariant the comment asserts?
   Then trace the most impressive number to a script output or log line. A number you cannot trace is
   `[unverified]` — never repeat it as fact. Upstream shipped a fabricated speedup for a PR that had
   never claimed it.
4. **Safety theater.** For each new guard: is it reachable, will it ever fire, does a bare `except`
   swallow a real error?
5. **Tests calibrated to pass rather than to falsify.** Is the reference implementation structurally a
   twin of the kernel, so the same bug lives in both and they always agree? Is the tolerance loosened
   with no justification? Does it assert against the kernel's own output? A test that logs its verdict
   instead of asserting it cannot fail at all. Two more that pass review easily: a test that feeds
   **idealized contiguous inputs where the model feeds a transposed view or a preallocated output
   buffer** is not exercising the call site and will miss exactly the strided-input bugs in §3d; and
   a whole-op arch gate written as a **deny-list** (`if arch == 'gfx1250': skip`) silently runs an
   unbuilt kernel on the next new card, where an allow-list (`if arch not in SUPPORTED: skip`) skips
   cleanly. Dropping *one* candidate on a named arch with a stated reason (a wave64-only path skipped
   on gfx1250) is fine; the tell is a deny-list guarding the whole op.
6. **Magic constants.** A new tile size, threshold or epsilon with no stated derivation or tuning
   basis.

## 5. Judge the performance evidence

Grade what the PR supplies, then notice that grading it produces no number of your own.

- Perf claimed with **no numbers carrying units** — screenshots are not numbers. Numbers for **toy
  shapes only** (M ≤ 256, one token, one model). **No reproduction info** — ROCm version, GPU, TP
  config, checkpoint. **TP=1 head counts only**, when a kernel that passes at H=128 can go
  out-of-bounds at H=32 under TP=4.
- **A timing window drawn to exclude a recurring cost** — a first-call JIT on a path that is not
  cached, or setup running on the live stream every cold start. The false-positive guard matters as
  much as the rule: excluding a genuinely one-time, amortizable cost (weight preshuffle, model load, a
  JIT result cached forever) from steady-state per-call latency is **correct methodology**, and
  charging a one-time shuffle against a single call to manufacture a regression is itself the error.
  Ask whether the cost recurs per call or is paid once per deployment.
- **Nobody measured it.** Every check above grades the PR's own table. Correctness evidence does not
  cover latency: a kernel can compute the right values and be slower. The measurement that counts is
  **base vs head, same box, back to back, from clean trees at the exact compared revisions**—running
  only head against whatever baseline the PR chose reproduces the PR's own comparison, cannot show a
  regression, and inherits any staleness in that baseline. Protocol:
  [`../profiling/benchmarking_methodology.md`](../profiling/benchmarking_methodology.md).

## 6. Refute before reporting

First, a blind-spot pass: ask whether any correctness risk, resource hazard or behavioral edge case
in the diff was caught by none of §1–§5, and write down what you looked for. A bare "no" is not an
answer. Anything found after the findings list is drafted is added as a *late* finding, not silently
merged into the earlier list.

Then attack each finding in this order — it is ranked by what has actually killed findings:

1. **The premise.** Read the code the claim *depends on*, not the code the claim is *about*. Upstream's
   example reported 125 tuned-config rows as unreachable, marked `[verified]`; ten seconds inside the
   lookup function — which tries exact M first — would have ended it.
2. **The tree you read.** A local worktree is not the merge target.
3. **Reachability.** Can the trigger actually occur, or does a caller, guard or arch gate already
   prevent it?
4. **Suspicion versus defect.** "This looks wrong" is not a finding. If you cannot name inputs that
   produce a wrong result, it is a question.
5. **Severity.** A high-severity finding asserts a wrong result or a crash. Without a reachable
   triggering case it is a "worth checking" at most.

Then hand the findings, the diff and the base path to **a reader who has not seen your reasoning**,
with the findings false until defended. Upstream's numbers for why this is a separate step: four
rounds of tooling fixes moved the self-refutation pushbacks from 64 to 33 and moved the independent
reader's **28%** kill rate *not at all*. Self-refutation catches a premise you never tested; it does
not catch a conclusion you are committed to. Record when no such reader existed rather than leaving
it ambiguous — a review that had one and a review that did not are different objects.

Treat that 28% as an **upper bound**, not a measurement: it is the rate at which an adversary
instructed to assume falsity judged a finding dead, and that adversary was never itself audited.

## 7. Report

Each finding needs three parts, and the third is the one that gets dropped:

1. **Problem** — what is wrong, with file and line.
2. **Impact** — what goes wrong at runtime: wrong output, crash, perf regression.
3. **Action** — ending in a verb phrase. "**Author must** cite the spec proving padding is not read."
   "**Reviewer should ask** who passes this flag." No verb phrase means the finding is incomplete.

Tag each `[verified]` (traced through an evidence chain) or `[inferred]` (plausible, unconfirmed —
frame it as a question and do not assert it as the cause). Cap the report at **five** findings,
ranked by severity then blast radius; drop the rest rather than appending them as a tail. This is a
readability limit, not a recall claim.

**State the evidence tier on its own line, always, and never let a finding cap evict it.** Separate
the states rather than collapsing them, because they are different facts:

```
Review (advisory):        NO FINDINGS | NEEDS WORK | HIGH RISK
Validation (deterministic): PASS/NEEDS_WORK/BLOCK/INCONCLUSIVE — target, runtime arch, skipped stages
                          | N/A — no runtime surface changed
                          | NOT RUN — <reason>
Perf (deterministic):     REGRESSION | NO REGRESSION — ratio N on <worst column>, over N rows, threshold T
   ...or when no reproducible perf run exists:
Perf (advisory):          MEASURED — shapes, base vs head with units, delta, sample count
                          | N/A | NOT RUN — <reason>
```

Three distinctions that carry the weight. **`N/A` is not `NOT RUN`** — a docs or tooling PR carrying
an alarming evidence line is what teaches readers to ignore the line. **An environment gap is not a
PR defect** — no idle GPU, wrong arch, missing harness. And **never label a hand-run number
deterministic**, nor soften a reproducible regression into advisory; the label is the reader's only
signal for whether a reproducer exists.

A missing test target is a finding in its own right **only when the changed path executes at run
time**. A tuner input CSV, a tuned-config table or a codegen list is loaded by nothing in the serving
path, so there is nothing a test could have covered — say which case it is on the `NOT RUN` line
rather than reporting the absence as a defect on a data-only diff. A kernel change with no runnable
perf harness, by contrast, is a finding.

Classify CI failures before blaming the PR: read the failed *step*, since timeouts, artifact-count
errors and dependency-resolver noise are infra flakes; compare against main in the same window; and
treat expired logs on an old run as meaningless against today's main — ask for a rebase instead of
quoting them.

---

## Verification gate

A review is complete when every finding on it names a file the PR actually changed, every
high-severity finding names a concrete triggering input, every finding carries `[verified]` or
`[inferred]` and an action verb, both evidence lines state the tier they are in, and anything that
could not be run says why. A finding kept off the report — killed in refutation or cut by the
five-finding cap — is recorded in the reviewer's own notes with the reason, not silently lost.

## Pitfalls
- **Writing an advisory judgement in deterministic grammar.** The most consequential error here.
- **Reading the local worktree instead of the merge target**, then anchoring line numbers to it.
- **Firing on a pattern without the false-positive check.** Nearly every rule above has one, and they
  exist because the unguarded version produced noise: unmasked loads on provably-aligned dims,
  `max_size=True` on full-size weights, hardcoded FP8 bounds on single-dtype paths, arch strings
  already present elsewhere in the file, one-time setup costs excluded correctly.
- **Reducing a structural rule to a name list.** Measured cost: 0 of 4 real overflow lines reported,
  390 candidates emitted.
- **Treating a green checklist as evidence.** A checklist marked off to itself decays silently.
- **Letting an LLM judgement gate a merge.** Only a reproducible blocker that ships its reproducer may
  gate. The rate of *false clearance* — reporting nothing when something is wrong — has never been
  measured for this procedure, so the advisory tier stays advisory.

## Cross-links
- [`../profiling/benchmarking_methodology.md`](../profiling/benchmarking_methodology.md) — the
  base-vs-head protocol and the noise band this workflow's perf line depends on.
- [`../quantization/fnuz_vs_ocp.md`](../quantization/fnuz_vs_ocp.md) — the fn/fnuz dispatch hazard in §3c.
- [`../optimization/lds_and_bank_conflicts.md`](../optimization/lds_and_bank_conflicts.md) — the
  large-allocation DS-addressing check in §3g.
- [`../languages/flydsl/authoring_api_migration.md`](../languages/flydsl/authoring_api_migration.md) —
  buffer-bound and launch hazards in §3b/§3d.
- [`../languages/flydsl/api_stability.md`](../languages/flydsl/api_stability.md) — what a FlyDSL API
  change is allowed to break, for the API-propagation checks in §3b.
- [`../languages/triton_amd/knobs.md`](../languages/triton_amd/knobs.md) · [`../languages/triton_amd/pitfalls.md`](../languages/triton_amd/pitfalls.md) — Triton knob and masking specifics for §3f.
- [`optimize_single_kernel.md`](optimize_single_kernel.md) — the workflow that produces the change this one reviews.

## Sources
- ROCm/aiter@04c7b808:.claude/skills/review-pr/SKILL.md — evidence tiers, the semantic questions,
  blast-radius tiering, the six AI-code structural checks, the report shape and the finding format.
- ROCm/aiter@04c7b808:.claude/skills/review-pr/rules.md — the defect catalogue (§3), the
  false-positive self-checks, the perf-evidence rules, the refutation ladder and its measured
  kill rates, and the promotion bar.
- Deliberately **not** ingested: `fetch.sh`, `triage.py`, `MAPPING.md`, the `$WORK` artifact set, the
  six machine gates, aiter's file-tier table and CI label roster, and the rule ids. They are
  executable machinery bound to one repo's paths, scripts and CI; the patterns they enforce are above.
- Measured claims (rule firing rates, the 28% independent-refutation kill rate, the 0-of-4 scanner
  result, the 71% Tier-1 over-fire) are upstream's, over their ~600-PR open corpus. They are quoted
  as reported and have **not** been reproduced here.
