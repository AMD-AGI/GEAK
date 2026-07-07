# Report ↔ Director reconcile + backend-provenance timeline

## Problem

The headline numbers in `final_report.md` / `architect_report.md` did NOT match the authoritative
`director_e2e_validation.json`. Concretely (gpt-oss-120b): the report quoted throughput `640.4 → 709.0`
while the Director's same-session A/B was `621.365 → 698.373`. The affected fields are all of the
headline metrics: **throughput, speedup, TTFT, TPOT** (and status/parity).

Root cause is a **phase-ordering gap**, not a measurement bug:

1. The orchestrator runs `Report` (System Architect) BEFORE `Validate` (Director) —
   `Finalize → Report → Validate`.
2. `system_architect.md` already mandates that the headline quote the Director's same-session numbers
   (`director_e2e_validation.json` + `validation/{base,final}/bench_summary.json`). But at `report` time
   the Director has not run yet, so those files do not exist — the Architect silently falls back to the
   **Finalize-bundle** bench (`final/bench_final` / `final/bench_baseline_ab`). The two benches differ
   (different session/drift), so the report and the Director disagree.

Separately, the `final_report.md` timeline did not consistently record **which backend each hot kernel
originally ran on**, nor **which backend每 optimization attempt tried** — so a reader could not see the
backend bake-off ladder (what was tried, what won, what was skipped and why).

## Fix (all generic — no model/kernel/backend hardcode)

**1. Director reconciles the report after validation (keeps the existing phase order).** The `validate`
agent now receives the two already-written report paths (`ARCHITECT_REPORT`, `FINAL_REPORT`). New
**step 7** in `director.md`: after writing `director_e2e_validation.json`, the Director reviews both
reports and **overwrites only the headline metrics** — throughput (median+spread), speedup (`×` and `%`),
TTFT, TPOT, validation_status, output_parity — with its authoritative same-session numbers (sources =
`director_e2e_validation.json` + `validation/{base,final}/bench_summary.json`). Every other line/table/tree
is preserved. If validation produced no usable number (server crash / degenerate), it does NOT rewrite —
the Finalize fallback stays and is annotated. A closing self-check asserts the report headline equals
`director_e2e_validation.json`.

**2. Architect writes the headline as provisional.** `system_architect.md` now states that at `report`
time the Director file does not exist yet, so the headline throughput/speedup/TTFT/TPOT are written from
the Finalize bench and tagged `(provisional — pending Director validation)` on clearly-identifiable lines;
the Director drops the tag when it reconciles. This removes the contradiction where the role said "quote
the Director" for a file that was not there.

**3. Backend-provenance in the timeline (headline requirement).** The `final_report.md` Phases tree now
MUST, for every head op, name the op's **original/stock backend + dtype** (from the baseline profile Top-N
+ `meta.json`), and show **one sub-node per backend attempted** (each Tier-A bake-off candidate and each
Tier-C author language; in deep mode one per `(op × backend)` lane) with its best isolated `×` and the e2e
verdict. Backends **considered but not run** appear marked `⊘` with a one-word reason (`⊘ CK — ckProfiler
absent`, `⊘ flydsl — seam mismatch`, …) — never silently dropped. The head-kernel deep-dive is aligned: an
explicit **Backend ladder** line + a **backend column** in the Directions table.

**4. Clearer validation vocabulary + honest emoji (report readability).** `validation_status = accepted`
was misleading — on a no-win run (empty overlay, `0.9997×`) "accepted" read like a success. The Director
verdict is now **three self-explaining values**: `validated_win` (real e2e-gated win over the noise band),
`validated_no_win` (measurement trustworthy, final ≈ baseline — no regression, no win), `flagged`
(regression / parity fail / claim mismatch / crash). The word `accepted` is no longer used for this field
(no code branches on the literal, so this is safe). Separately, the timeline emoji must reflect the actual
gate: a **rejected/regressed** integrate node (e.g. `A/B 1768.5 → 1536.7 = ✘−13.1%`) must use `❌`, never
`⭐` — a slowdown never gets a star; `parity ✓` marks numeric parity only and never upgrades a `✘` delta;
`⭐` is reserved for a candidate actually banked into the final stack.

## Files
- `e2e_workflow.js` — `Validate` agent gets `ARCHITECT_REPORT` + `FINAL_REPORT`; task string asks it to
  reconcile the report. No phase reorder (order stays `Finalize → Report → Validate`).
- `roles/director.md` — `validate` **step 7** (reconcile the report headline to the Director numbers) +
  the two report paths added to Inputs.
- `roles/system_architect.md` — headline written provisionally from the Finalize bench (tagged), reconciled
  by the Director; timeline now requires stock-backend per op + one node per attempted/skipped backend; the
  deep-dive gains a Backend-ladder line and a backend column in the Directions table; emoji rule tightened
  (rejected/regressed → `❌`, `⭐` only for a banked win) and the Validate node prints the status word.
- `roles/director.md` — validate arbitration emits `validated_win|validated_no_win|flagged` (no more
  `accepted`); step 7 reconciles the report status/conclusion to match honestly.
- Add `e2e_workflow/docs/pr-report-director-reconcile.md`.

## Behavior when off / neutral
No new runtime knobs. When the Director produces a usable number (the normal case) the report headline is
made identical to `director_e2e_validation.json`; when it does not, the report is left on the Finalize
fallback and annotated (previous behavior, now explicit). The backend-provenance changes are report-content
only — they do not alter measurement, gating, or the overlay contract.

## Validation
- Root cause reproduced on a real run dir: report `640.4 → 709.0` vs `director_e2e_validation.json`
  `621.365 → 698.373` (finalize-vs-director divergence).
- `node --check` on `e2e_workflow.js` passes when wrapped (the file has a top-level `return` for the
  single-kernel pass-through, so it must be checked inside a function wrapper; no standalone `node` parse).
- The only JS change is two added keys in the `validate` agent's input object; delimiter balance preserved.
