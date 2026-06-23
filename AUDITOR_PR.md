# PR: Independent Patch Auditor for the GEAK e2e_workflow

## Summary
Adds an **independent, prompt-driven Patch Auditor** that signs off **every place the workflow banks a
decision** — the baseline, the profile/routing, each config/kernel accept, the kernel harness, and the
final bundle. It re-derives every number **from the raw `bench_runs.jsonl`** (never the producing role's
reported numbers), so a win can't be unreal, unfairly measured, misattributed, or unsafe. The producing
agents are told to treat its verdict as authoritative and fix until it would pass. Everything is **additive
and gated behind `use_auditor` (default OFF)** — with the flag off the workflow is byte-identical to before.

## The problem (grounded in our run corpus)
Every producer persona either optimizes or **grades its own work**; the Integrator both *builds* the overlay
and *gates* it, and the gates were prompt-asserted prose an LLM can skip. Across real multi-model runs that
let through, repeatedly:
- an orthogonal **spec-decode** lever (+34.5%) reported as a kernel win (Gap 1);
- a **benchmark-gamed** "verified 1.5×/3.0×" isolated speedup whose deployed kernel was actually ~0.985×
  (slower), measured through a graph-replay path the live server never uses (Gap 5);
- an accept recorded with **null evidence fields** (medians/non-overlap unprovable from the artifact) (Gap 4);
- a **+8–9% banked on a gibberish-emitting model** where byte-parity is meaningless and the accuracy probe
  was perpetually deferred (Gap 7/8);
- a **mem-fraction-handicapped / misreported baseline** inflating a headline (found live, see below).

The meta-finding: v4's fixes for these were **prompt-level, not enforced** — non-deterministic and
re-overfittable. This PR makes the verification **independent and re-derived from raw**, at every gate.

## What this PR adds
- **Skill** `perf_knowledge/expert_skills/skills/patch_auditor/skill.md` — the auditor persona + 5 audit
  scopes: `patch`, `bundle`, `baseline`, `profile`, `harness`. Verdict is `PASS | FLAG | FAIL` with a
  per-finding `action`. (+ `proof/` backtest fixtures and `index.yaml` entry.)
- **Orchestrator wiring** in `e2e_workflow.js` (+161/-13), additive + `use_auditor`-gated:
  - **baseline** sign-off after setup, with **auto re-measure** if the baseline is noisy/contended/wrong-invariant;
  - **profile/route** sign-off before strategize (closes mis-attribution at the source);
  - **harness** sign-off after the op bake-off, with an op-benchmarker **redo loop** on FAIL;
  - **accept** sign-off at the config gate + all three kernel/head integrate gates;
  - **final bundle** sign-off after validation.
- **Prompt-level enforcement** in the producing roles (`e2e_integrator`, `config_tuner`, `op_benchmarker`,
  `kernel_extractor`): the auditor is authoritative; read its verdict, fix every reason, resubmit, and do
  not consider the work done until it would PASS.

## How it works
- **Re-derive, never trust:** every gate recomputes medians/min-max/non-overlap/TTFT/TPOT and diffs the two
  legs' launch config from the raw `server_info`; it cross-checks the reported summary against the raw.
- **Fix-until-pass, prompt-level:** an explicit `FAIL` is fed back to the producing agent (with reasons) to
  fix and resubmit; the auditor re-audits from raw, so a superficial "fix" can't pass (independence).
- **Fail-OPEN on tooling failure:** a degraded/no-verdict auditor never hard-blocks a real win (a parser
  glitch must not kill a result); only an explicit `FAIL` withholds a bank. Loops are bounded.
- **Separation of duties:** the auditor optimizes/integrates nothing; the **kernel changer never consults
  it** — it trusts a harness the auditor has already validated.

## Evidence it works (blind tests on real recorded runs + first live run)
The auditor was run **blind** (skill + raw artifacts only; findings/answer-key withheld) against recorded
runs in `exp/`. It reasoned from primary evidence in every case — it did **not** pattern-match known cases:

| Case (real artifact) | What the auditor did | Verdict |
|---|---|---|
| FlyDSL GEMM "+66.9% clean win" (Qwen3.5-27B) | Re-derived raw legs: **baseline ran at mem 0.7225 vs final 0.85** (handicapped) **and `director_e2e_validation.json` misreported mem as 0.85 for both** → confounded bundle, not a clean kernel win | **FAIL / eject** — *and corrected a flaw the human answer key had missed* |
| Authored GEMM integrate claiming `isolated 3.0306×` | Independently opened the kernel-layer `opbench_result.json`, found a **self-comparing baseline** (baseline backend == winner) and `rel_err 45.4` on the only faster alt → the isolated number is fabricated; the e2e win is real | **FLAG → replace_headline_with_e2e** |
| MoE attention CK→triton "+9%" | Inspected the parity dumps: reference decode is **gibberish** (`--dataset random`), **0/12 byte-exact** → byte-parity uninformative, cannot certify | **FAIL / eject** (`unverifiable_noncoherent_reference`) |
| Crashed authored attention kernel (iso 6.6×) | Saw the live server **crashed** (HSA fault) → do-no-harm violation | **FAIL / eject** (agreed with the justified reject) |

**First live (auditor-enabled) run — in-loop baseline sign-off fired automatically:**
re-derived the baseline reps `[966.36, 1004.05, 1004.71]`, flagged **spread 3.82% (>7× the 0.5% noise band)**
driven by a **rep-0 warm-up outlier**, and recommended re-measure — *exactly* the "too noisy to gate a
sub-4% win" blind spot, caught in-run with actionable advice. With this PR that FLAG now **auto-triggers a
baseline re-measure** before anything is gated on it.

## Gaps closed
| Gap | How the auditor closes it |
|---|---|
| 1 — orthogonal lever as kernel win | lever A/B classification at every accept; B never folded into a kernel headline |
| 2 — profiler mis-attribution / mis-routing | `profile` gate re-aggregates GPU-time **by op**, checks fragmentation + edit-flag + skip, before routing |
| 4 — accept without evidence | re-derives from raw; reported nulls are ignored; missing raw ⇒ FAIL |
| 5 — benchmark-gaming | `harness` gate enforces **dispatch/launch parity** + fair A/B + served shapes; e2e gate backstops isolated≠e2e |
| 6 (storm) | same-conditions uncontended-legs check (detection) |
| 6 (latency) | headline-integrity flags a TTFT/TPOT regression hidden by a throughput headline |
| 7/8 — correctness=determinism | coherence check; non-coherent reference ⇒ unverifiable ⇒ FAIL |
| baseline quality | `baseline` gate flags a noisy/handicapped reference and **auto re-measures** |

## Honest limitations (by design)
- It catches **false accepts**, not **missed wins / wasted runs** (that's the future Researcher/PI persona).
- It **detects** contamination; it does not **prevent** the process storm (the reaper does).
- The interpretive judgments (lever-class, coherence) are LLM calls anchored on the objective gates; their
  reliability at scale is not yet proven (the N×/multi-model sweep is future work).
- The harness redo-loop is wired on the **serial** head path (the default); the fast-mode parallel path and
  the kernel-track nested workflow rely on the maker prompts (loop wiring there is a follow-up).

## Risk / safety
- Purely **additive + `use_auditor`-gated (default OFF)** → byte-identical to the current workflow when off
  (verified). All loops are **bounded**; a degraded auditor **fails open** (never blocks a real win or hangs).
- Live e2e run (`exp/e2e_Qwen-Qwen3.5-27B-FP8_20260622_133415_176030_24863`) is exercising it end-to-end
  with `use_auditor=true`.

## Test plan
- [x] `use_auditor=false` ⇒ workflow byte-identical (syntax-checked; gates inert).
- [x] Blind backtests on recorded runs reproduce the catches above.
- [x] First in-run baseline sign-off fires and flags the noisy baseline.
- [ ] Full auditor-enabled e2e run to completion (in progress) — attach baseline/profile/harness/accept/bundle verdicts.
- [ ] Reliability sweep (N× × ≥2 models) on the interpretive judgments.
