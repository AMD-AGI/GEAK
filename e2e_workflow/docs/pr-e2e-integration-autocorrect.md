# e2e integration: auto-correct correctness/corruption rejects

## Problem

A head-kernel candidate can PASS its isolated unittest yet CORRUPT the live server, and the workflow
did not auto-correct it — it dropped the head. Concretely (gpt-oss-120b GEMM2 down-proj): the kernel
scored `max_abs_err=0` isolated (1.15×) but produced garbage on the live path (gsm8k 0/200) and posted
an impossible **+166.7%** e2e "speedup". The two retry mechanisms that exist did not help, for three
distinct reasons:

1. **`AB_FINISH_RETRIES` is the wrong kind of retry.** It only re-invokes the integrator when the A/B is
   *incomplete* (`gate:'incomplete'` / `ab_complete:false` — only the ref leg ran, or it hung/degraded).
   A correctness rejection is a *completed, deterministic* A/B (`abDone` is true), so the loop never
   fires; re-measuring the same broken candidate would just reject again.
2. **The corrective re-author path excluded correctness rejects.** `tryCorrectiveReauthor` (which *does*
   re-optimize the kernel) was gated by `FIXABLE_REJECT_RX` — only JIT/no-binary/capture/host-sync
   ("integration-posture") reasons. A parity/corruption reject didn't match, so no fix was attempted.
3. **Even if it had fired, the isolated oracle can't catch the bug.** The immutable unittest replays ONE
   captured snapshot with fixed tensors (stable `data_ptr`, stable contents), so a candidate that caches
   routing-derived indices by `data_ptr()` (correct only when contents never change) is always "correct"
   in isolation. Live serving reuses the same index buffers every step with *new* routing → stale cache
   → corruption. A blind re-run reproduces the bug; the oracle can't validate a fix.

Root cause of this bug class: **the kernel over-fits the single captured snapshot** (assumes input-buffer
identity / contents / routing are stable across calls). The manual fix was to drop the `data_ptr`-keyed
caches (recompute per call) and keep the fused scatter-add — which restored correctness (gsm8k 0.94 ≈
baseline 0.935). (In this particular case the *correct* kernel turned out throughput-neutral, because
dropping the ~1%-GPU `_reduce` is Amdahl-limited — but that is orthogonal to the process gap fixed here.)

## Fix (all generic — no model/kernel/backend hardcode)

**1. A second fix-and-retry class: `correctness`.** New `CORRECTNESS_REJECT_RX` + a unified
`rejectClass(reason)` → `'correctness' | 'integration' | ''`. `tryCorrectiveReauthor` is now eligible for
both classes and emits a **class-specific** corrective brief. The correctness brief tells the kernel layer
the exact anti-pattern (never key a cache on `data_ptr()`/`id()` for a per-call-varying tensor; recompute
or hash-key; don't carry per-call state; zero reused atomic outputs; and *validate adversarially* with a
buffer-reuse / interleaved replay) — describing the failure CLASS, never this kernel.

**2. Implausible-speedup guard (parity-aware).** `amdahlCeilingPct(pct_gpu_time, isolated)` = the
theoretical max e2e speedup for an op that is `pct`% of GPU time at isolated speedup S.
`isImplausibleSpeedup(...)` flags a measured delta far above the ceiling (default: must exceed 2× the
ceiling) as corruption/degenerate work; `integAccepted()` refuses to bank it at every integrate site and
`gateRejectReason()` routes it to the correctness corrective. **Crucially it is applied ONLY to a SOFT
(sampled task-accuracy) accept** — a `byte_exact` parity pass is a hard correctness guarantee, so its
speedup is trusted even above the (imperfect, profile-derived) ceiling. This eliminates false positives
that would otherwise DROP real wins when the profiler under-counts `pct_gpu_time`, when a change has
system-wide effects, or for config (env/flag) wins — while keeping the backstop exactly where byte-parity
is waived (quant / `accuracy_gate=gsm8k`). The integrator now reports `parity_kind`
(`byte_exact|accuracy|none`); fail-open (never flag) when it is absent and the run uses no accuracy gate.

**3. Stronger isolated oracle (closes the gap so auto-correct can converge).** The `kernel_extractor`
role now requires a **cross-call robustness** check in every `unittest.py`: interleaved / buffer-reuse
replay (write case B's contents into case A's buffers, assert B still correct) plus a fresh-buffer repeat.
Any `data_ptr`-keyed cache or stale reuse fails in isolation. Capture records ≥2 snapshots per shape
bucket when the live window provides them (additive; degrades to the fresh-buffer repeat with one
snapshot). The `e2e_integrator` role documents the implausible-speedup guard and the reject-reason
vocabulary the classifier keys on.

## Files
- `e2e_workflow.js` — `CORRECTNESS_REJECT_RX`, `amdahlCeilingPct`/`isImplausibleSpeedup` (parity-aware
  via `parityIsSoft`), `rejectClass`/`integAccepted`/`gateRejectReason`; `tryCorrectiveReauthor`
  class-aware; all 5 integrate sites (default head, fast/parallel head, deep head, milestone, finalize)
  apply the guard + route correctness rejects to corrective; `parity_kind` added to the integrate schema.
- `roles/kernel_extractor.md` — mandatory cross-call robustness oracle + ≥2-snapshot capture.
- `roles/e2e_integrator.md` — report `parity_kind`; soft-gate implausible-speedup guard + reject vocabulary.

## Behavior when off / neutral
With no correctness/implausible reject, `rejectClass` returns `''` and `integAccepted` equals the old
accept condition, so a run with no such failure is behaviorally unchanged. The implausible guard fires
ONLY on a soft (accuracy-gated) accept whose delta exceeds 2× the Amdahl ceiling — a byte-exact accept is
always trusted, so real wins are never dropped (fixes the false-positive review finding). New knobs:
`implausible_speedup_margin` (default 1.0 = must exceed 2× the ceiling); existing `head_corrective_max`
(default 2) bounds the correctness retries too.

## Validation
- Classifier + Amdahl math unit-checked on real reason strings and the observed numbers: the +166.7%
  candidate → flagged & routed to correctness corrective; the correct −0.64% and a legit +3% → not
  flagged; unknown inputs → fail-open.
- Delimiter balance of `e2e_workflow.js` preserved (no `node` available in this env for a full parse;
  reviewer should `node --check e2e_workflow/e2e_workflow.js`).
