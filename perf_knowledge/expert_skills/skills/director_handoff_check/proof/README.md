# Director handoff check — proof harness

Frozen, evidence-grounded test set that demonstrates the (prompt-only) Director handoff check catches the
failure modes in `GEAK_v4_FINDINGS.md` / `PerfSkills_GAP_FINDINGS.md`, **and** that its verdicts are
reliable. See the proof methodology (§ Methods below). The headline metric is generalization, NOT "passed
the N known cases" — a prompt tuned to a fixed set proves nothing.

## Layout
- `fixtures.yaml` — the FROZEN answer key. Each fixture: source artifact (or injection recipe) + the
  expected verdict/action/labels + the gap it targets + the key evidence numbers. Freeze this BEFORE
  tuning gate prose; do not edit labels to match the check.
- (later) `run_backtest.*` — invokes the check skill per fixture and diffs its JSON verdict vs the
  manifest; emits precision/recall + a per-gate pass table.
- (later) `injected/` — the perturbed fixture dirs produced from the recipes in `fixtures.yaml`.

## Fixture classes
- **live** — replayed against unmodified real run artifacts on disk (`/root/GEAK/exp/...`). Proves the
  gates work on real data.
- **injected** — a controlled mutation of the real PASS case (P1) with a known-correct verdict. This is the
  generalization signal: faults the prompt was not written against, verbatim.

## Methods (run in this order)
1. Backtest the `live` fixtures → verdict must equal the frozen label.
2. Apply each `injected` recipe to P1, run the check → verdict must equal the frozen label.
3. Reliability: run every fixture N times across >=2 models; record verdict stability, separately for the
   arithmetic gates (expect ~deterministic) and the two interpretive judgments (lever-class, coherence) —
   the latter is where a prompt-only gate is weakest.
4. (deferred, needs the e2e_workflow.js hook) live A/B: same slate +/- the check; false-accept-rate -> ~0.

## What this proves / does not
- Proves: the gate LOGIC is correct and reliable WHEN INVOKED.
- Does NOT prove (until wiring): enforcement — that it is always invoked and its FAIL always honored.
- Framing: the historical negatives are "would-have-caught / regression-guard," not "v4 ships these today"
  (v4 already prompt-mitigated Gaps 1/2/4 — the handoff check's value is independent, unfakeable enforcement).
