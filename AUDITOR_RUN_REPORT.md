# Auditor — live run report (in progress)

**Run:** `exp/e2e_Qwen-Qwen3.5-27B-FP8_20260622_133415_176030_24863` · sglang · ISL/OSL 1024 · conc 64 ·
TP=1 GPU 4 · `use_auditor=true`. Status at ~3h55m: **healthy, in the head-kernel track.** Baseline
1003.7 tok/s.

## What difference the auditor made to THIS run (with vs without)
| Without the auditor (what the run would have done) | With the auditor (what actually happened) |
|---|---|
| Used the **noisy baseline** (spread 3.82%, a 966 tok/s warm-up outlier) as-is — every later win gated against a reference whose own noise (~±4%) is **bigger than a typical config/kernel win** (sub-4% wins would be unprovable / false-positives waiting to happen). | **Auto re-measured** the baseline to spread **0.118%** (1003.7 tok/s) before anything was gated on it. The whole run now stands on a clean reference. |
| **Banked** the +2.9% config accept (`--attention-backend triton --kv-cache-dtype fp8_e4m3`) and carried it forward as the run's config. | **Ejected** it: the comparison was confounded — the aiter baseline effectively ran at mem 0.7225 (smaller KV) vs the triton candidate's 0.85 — so the +2.9% can't be attributed to the lever. The run carries the **clean baseline config** instead. |
| `sweep_results.json`'s claim *"identical invariant … verified in each server.log"* stands **uncorrected** in the record. | The **misreport is surfaced** (raw `server_info` contradicts it) and the raw is trusted. |

**Net so far:** the auditor **changed the measurement foundation** (replaced a noisy baseline with a clean one) and **changed what the run banks** (blocked a confounded config win + caught a false "same invariant" claim). **Honest trade-off:** a *likely-real* ~+2.9% was withheld because it couldn't be **certified** same-conditions — the auditor recommended a matched-mem re-measure rather than banking an unprovable number. Without the auditor, none of these three corrections happen — the run proceeds on a noisy baseline and a confounded, misreported config headline.

## In-run auditor verdicts so far
| Gate | Verdict | What it did |
|---|---|---|
| **baseline** | PASS (after auto-remeasure) | First measure FLAGged (spread **3.82%**, rep-0 warm-up outlier) → orchestrator **auto re-measured** (6 warm reps, rep-0 discarded) → re-audit **PASS** at spread **0.118%**. Now every downstream delta is gated on a clean reference. |
| **profile** | PASS | Top-N attribution sane before routing (no fragmentation/edit-flag/skip issue). |
| **config accept** (`--attention-backend triton --kv-cache-dtype fp8_e4m3`, +2.9%) | **FAIL → eject** | See below. |
| harness / integrate / bundle | pending | head-kernel track in progress. |

## The headline catch — config accept rejected (`FAIL → eject`)
Re-deriving from raw, the auditor found the +2.9% config win was **not same-conditions**:
- All legs launched with `--mem-fraction-static 0.85`, but the **aiter baseline effectively ran at 0.7225**
  (KV pool 947k tok) while the **triton candidate ran at 0.85** (KV pool 1.15M+ tok) — because the *aiter
  attention backend reserves a heavier workspace*, leaving less for KV. So the mem-fraction divergence is a
  **side-effect of the attention-backend lever itself**, handicapping the baseline → invariant mismatch.
- **Misreport caught:** `sweep_results.json` claimed *"MEM_FRACTION=0.85 … identical invariant … verified
  in each server.log."* The raw `server_info` + KV-allocation lines contradict it → flagged, raw trusted.
- **Calibrated, not trigger-happy:** noted a *mitigating* factor (at conc=64 the divergence is **inert** —
  running-requests never exceeded ~65, so the larger KV pool never engaged → the delta is probably real),
  and *did* verify fp8-KV correctness via an 8-prompt greedy probe (`accuracy_gate_pass`). It still
  conservatively FAILed on the raw invariant mismatch and **recommended re-measuring the baseline at a
  matched effective mem-fraction**.
- **Consequence:** the accept was **ejected** (a likely-real +2.9% withheld because it couldn't be certified
  same-conditions); the run carried the clean baseline config into the head-kernel track.

## What difference the auditor made to THIS run (vs. without it)
| Gate | Without the auditor | With the auditor — what actually changed |
|---|---|---|
| **baseline** | The whole run gates every config/kernel delta against a **3.82%-spread** reference (rep-0 warm-up outlier, median ~1001) — wins of 1–3% are smaller than the baseline's own noise, so accept/reject calls are unreliable for the entire run. | The orchestrator **re-measured the baseline** (6 warm reps, warm-up discarded) → **0.118% spread**, reference fixed to **1003.7 tok/s**. **Every downstream decision is now gated on a clean reference.** (reps 3→6, spread 3.82%→0.118%). |
| **profile** | (no change) | Confirmed the routing attribution was sound — no false re-route. |
| **config accept** | The workflow **banks the +2.9% `triton + fp8-KV` config** as a real win and **carries it forward** into the head-kernel track (the sweep even asserted "same invariant verified"); the lossy **fp8 KV-cache** lever rides along silently and the headline counts a confounded +2.9%. | The auditor **ejected it** → the run **carried the clean baseline config forward instead**. The confounded/misreported +2.9% never entered the headline, and the lossy fp8-KV lever was **not silently banked**. |

**Net so far:** the auditor changed the run's trajectory in two concrete ways — (1) it **replaced the noisy
baseline with a clean one** that all later gating uses, and (2) it **stopped a confounded, misreported
config accept from being banked**, so the head-kernel track is building on the verified baseline config
rather than an uncertified `triton + fp8-KV` stack.

**Honest cost + fix:** on the config ejection the auditor itself judged the +2.9% *probably real* (the
mem-fraction divergence is an **inert side-effect** of the attention-backend lever at conc=64, and
correctness passed) — so the `FAIL → eject` was **too rigid**: it should have ALLOWED a logical,
net-positive, correctness-verified step whose only "violation" was a secondary knob shifting as an inert
consequence of the lever itself. We **refined the same-conditions rule** accordingly: a differing invariant
that is (a) a side-effect of the lever under audit, (b) AFFIRMATIVELY shown inert for the measurement, and
(c) correctness-clean ⇒ **PASS (or FLAG to correct a misreport), not eject**; only an independent confound
or an *unproven* difference still FAILs. Under the refined rule this config accept would have been a
**PASS/FLAG** (win kept, misreport flagged). The fix applies to the remaining gates this run and all future
runs; the config phase here had already passed, so its accept stays ejected for this particular run.

## Takeaways (so far)
1. The **baseline auto-remeasure loop works end-to-end** — FLAG → re-measure → re-audit PASS, no human.
2. The **config gate caught a mechanism-level confound + a misreport**, independently, in-run — deeper than
   the hand-analysis we did earlier (it explained *why* the mem-fraction diverged).
3. The auditor is **calibrated** (mitigating factors + a real correctness probe), not a blunt blocker, yet
   stays conservative on an unprovable same-conditions invariant.

## Still pending
Harness (dispatch-parity), kernel/head integrate(s), and the final-bundle sign-off — to be appended when
the head-kernel track and validation complete.
