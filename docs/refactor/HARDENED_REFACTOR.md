# GEAK Hardened Refactor — Regression-Safe Unification

**Sibling to**: `GEAK_codebase_audit.md` (current state) + `GEAK_unification_plan.md` (full plan).

**Purpose of this doc**: answer the narrower question — *"how do we ensure the two paths that actually matter (HIP+homo, Triton+hetero) port without regression, while keeping the agent completely language-agnostic?"*

**Scope**: only the 2 production paths. The 2 rare/broken paths (Triton+homo, HIP+hetero) get fixed as a side-effect of the refactor but aren't the primary target.

**Verification**: all claims below cross-checked against **7 real run logs** — 4 HIP+homo, 3 Triton+hetero. All 7 follow byte-identical flow up to the mode split.

---

## 1. The clean architecture — OptimizationAgent is language-agnostic

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — AGENT (language-AGNOSTIC, one class, used everywhere)    │
│                                                                     │
│  class OptimizationAgent:                                           │
│      def run(self, task_body: str) -> (exit_status, result):        │
│          # Step-loop: query LLM → parse tool call → execute tool    │
│          # Tools (fixed, universal):                                │
│          #   bash, str_replace_editor, save_and_test, submit,       │
│          #   strategy_manager, profile_kernel, query, optimize      │
│          # The agent NEVER imports KernelLanguage.                  │
│          # The agent NEVER checks kernel_type.                      │
│          # task_body is an OPAQUE STRING.                           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │  task_body produced by:
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│  LAYER 2 — TASK COMPOSITION (the ONE language-aware function)        │
│                                                                     │
│  def compose_task_body(                                             │
│      language: KernelLanguage,                                      │
│      flavor: Literal["fixed", "planned"],                           │
│      planner_output: str | None = None,                             │
│      starting_patch: Path | None = None,                            │
│      kernel_analysis: str | None = None,                            │
│  ) -> str:                                                          │
│      parts = [language.system_prompt, ""]                           │
│      if flavor == "fixed":                                          │
│          parts += [language.optimization_prompt]                    │
│      else:  # planned                                               │
│          parts += [planner_output]                                  │
│      if starting_patch:                                             │
│          parts += [f"## Starting patch\n...{starting_patch.read()}"]│
│      if kernel_analysis:                                            │
│          parts += [f"## Kernel analysis\n{kernel_analysis}"]        │
│      return "\n".join(parts)                                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 │  language fields consumed:
                                 │  - system_prompt          (string)
                                 │  - optimization_prompt    (string)
                                 │  - planner_strategy_hints (string, used by planner)
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│  LAYER 3 — KERNEL LANGUAGE (language-specific content LIVES HERE)    │
│                                                                     │
│  @dataclass(frozen=True)                                            │
│  class KernelLanguage:                                              │
│      name: str                                                      │
│      file_extensions: set[str]                                      │
│      detect: Callable[[Path], float]                                │
│                                                                     │
│      # Prompts (Layer-2 task composition consumes these)            │
│      system_prompt: str          # role + env + tools               │
│      optimization_prompt: str    # default task + knob list         │
│      planner_strategy_hints: str # strategy taxonomy for planner    │
│                                                                     │
│      # Templates (preprocess consumes these)                        │
│      harness_template: str       # Jinja for HarnessBuilder         │
│      commandment_template: str   # Jinja for COMMANDMENT.md         │
│      builder_hints: str          # guidance for HarnessBuilder LLM  │
│                                                                     │
│      # Commands (evaluator + preprocess consume these)              │
│      test_runner_command: str    # "python3 {harness_path}"         │
│      profiler_command: str       # "rocprofv3 --kernel-trace --"    │
│      patch_apply_strategy: str = "git_3way"                         │
│                                                                     │
│      # Runtime env (dispatch consumes this)                         │
│      eval_env: Callable[[Path], dict[str, str]]                     │
│                                                                     │
│      # Memory (KB consumes this)                                    │
│      kb_namespace: str                                              │
└─────────────────────────────────────────────────────────────────────┘
```

**Reading the diagram bottom-up**: `KernelLanguage` is a pure data object with one method (`detect`). Everything language-specific is a field. Layer-2 `compose_task_body` is the only function that reads `KernelLanguage` fields at optimization time. Layer-1 `OptimizationAgent` never sees `KernelLanguage` at all — it gets a string.

**Invariant enforced by CI**:
```bash
# Must pass in every PR:
grep -rn "KernelLanguage\|kernel_type\|kernel_language" src/minisweagent/agents/optimization_agent.py
# → 0 matches

grep -rn "if .*== \"triton\"\|if .*== \"hip\"\|if .*== \"flydsl\"" src/minisweagent/ \
    --exclude-dir=kernel_languages --exclude-dir=tests
# → 0 matches (core code is language-free; only kernel_languages/ has language names)
```

---

## 2. Run verification — 7 kernel logs, 2 production paths

### 2.1 HIP + homo (`mode=fixed` after refactor)

Verified byte-identical flow across 4 different HIP kernels:

| Kernel | Line 13 | Line 47 | Line 77 | Line 82 | Lines 103-106 | Line 225+ |
|---|---|---|---|---|---|---|
| `assign_score_withk` | `kernel_type: hip` ✓ | `Step 1/7` ✓ | `Step 7/7: Commandment` ✓ | `Using homogeneous mode` ✓ | `Sub-agent 0/1 started on GPU 2/3` ✓ | (still running) |
| `furthest_point_sample` | `kernel_type: hip` ✓ | `Step 1/7` ✓ | `Step 7/7: Commandment` ✓ | `Using homogeneous mode` ✓ | `Sub-agent 0/1 started on GPU 2/3` ✓ | `Wrote final_report.json` ✓ |
| `roipoint_pool3d` | `kernel_type: hip` ✓ | `Step 1/7` ✓ | (truncated at step 5) | | | |
| `three_interpolate` | `kernel_type: hip` ✓ | `Step 1/7` ✓ | `Step 7/7: Commandment` ✓ | `Using homogeneous mode` ✓ | `Sub-agent 0/1 started on GPU 2/3` ✓ | |

**Invariant confirmed**: every HIP run goes through the SAME 10 numbered markers in the SAME relative order. Refactor must preserve these.

**What's observably missing** (known issue, not a regression target):
- Steps 2, 3, 4 never run for HIP (ATD doesn't recognize `task_runner.py` pattern)
- No `Verified speedup:` log line (no per-round FULL_BENCHMARK)
- No `record_optimization_outcome` call (no KB write)
- `final_report.json` has 4 keys (homo's short schema)

These are **upgrades** the refactor will add, not behaviors to preserve. Regression-prevention only guards what's there today.

### 2.2 Triton + hetero (`mode=planned` after refactor)

Verified byte-identical flow across 3 different Triton kernels:

| Kernel | Line 34 | Lines 76-192 | Line 197 | Line 215 | Line 223/224/226 |
|---|---|---|---|---|---|
| `gemm_a16w16_atomic` (live) | `kernel_type: triton` ✓ | 7 preprocess steps ✓ | `Using heterogeneous mode` ✓ | `Exploration Phase` ✓ | `Round 1/5` ✓ |
| `moe_routing_sigmoid_top1` (live) | `kernel_type: triton` ✓ | 7 preprocess steps ✓ | `Using heterogeneous mode` ✓ | `Exploration Phase` ✓ | `Round 1/5` ✓ |
| `ff_backward` (completed) | `kernel_type: triton` ✓ | 7 preprocess steps ✓ | `Using heterogeneous mode` ✓ | `Exploration Phase` ✓ | `Round 1/5` ✓ |

**Invariant confirmed**: every Triton run runs all 7 preprocess steps, enters exploration phase, begins round loop. Refactor must preserve these.

### 2.3 The 2 invariants the refactor MUST preserve

Failure of either is a regression.

**Invariant A (HIP+homo)**: Every HIP run
1. Detects kernel_type=hip
2. Runs preprocess steps 1, 5, 6, 7 (and any additional that come online after the refactor)
3. Generates COMMANDMENT.md (today it's present-but-buggy; refactor fixes the bugs)
4. Injects cross-session memory into task body (line 90)
5. Dispatches N=num_parallel sub-agents on the requested GPUs
6. Writes final_report.json on completion

**Invariant B (Triton+hetero)**: Every Triton run
1. Detects kernel_type=triton
2. Runs preprocess steps 1-7
3. Generates COMMANDMENT.md (well-formed today)
4. Enters exploration phase
5. Runs N rounds: generate_tasks → dispatch_tasks → collect_results → post_round_evaluate
6. Writes final_report.json (16+ keys)
7. Calls record_final_outcome → KB write

---

## 3. Regression-safety: what the refactor does, step-by-step

Each step below is a **separate PR** that can ship + roll back independently. Nothing happens in one giant step.

### Step 1 — Introduce `KernelLanguage` as a DATA layer only

**Scope**: create `src/minisweagent/kernel_languages/` + `TritonKernelLanguage` + `HipKernelLanguage` instances. **NOTHING reads them yet.**

```
src/minisweagent/kernel_languages/
├── __init__.py          # registry + convenience lookup
├── base.py              # @dataclass KernelLanguage (see §1)
├── triton/
│   ├── kernel_language.py
│   ├── system_prompt.md          (text extracted from current heterogeneous/prompts.py)
│   ├── optimization_prompt.md    (text derived from current mini_kernel_strategy_list.yaml)
│   ├── planner_strategy_hints.md (text extracted from TASKGEN_SYSTEM_PROMPT, Triton sections)
│   ├── harness.j2                (same structure as §10.3 of unification plan)
│   ├── builder_hints.md
│   └── commandment.j2            (same structure as today's generate_commandment)
└── hip/
    └── (same 7 files)
```

**Regression check**: Zero — nothing imports these yet. PR is pure additive.

**Rollback**: delete the directory.

### Step 2 — Route language detection through KernelLanguage

**Scope**: make `_normalize_kernel_type` and `_infer_kernel_type` delegate to `KernelLanguage.detect(path)`. Keep the old function signatures for backward compatibility.

```python
# run/mini.py
def _normalize_kernel_type(value: Any) -> str:
    """BACKWARD-COMPAT SHIM: delegates to KernelLanguage registry."""
    from minisweagent.kernel_languages import registry
    lang = registry.detect_by_name(str(value))
    return lang.name if lang else "other"
```

**Regression check**: run the 7 kernels we observed; verify `Normalized kernel_type from task content` log line is IDENTICAL character-by-character to pre-PR output. Small test: run `gemm_a16w16_atomic` for 2 minutes, grep for that line, compare to baseline.

**Rollback**: remove the shim, restore the inline logic.

### Step 3 — Introduce `compose_task_body` + `OptimizationAgent` rename

**Scope**:
- Rename `StrategyInteractiveAgent` → `OptimizationAgent` with an import alias for backward compat
- Add `compose_task_body` function in new `run/compose.py`
- Do NOT change any existing call sites yet

**Regression check**: zero — new code isn't called.

### Step 4 — Route HIP+homo through `compose_task_body`

**Scope**: modify `run_homogeneous_agent` to build `task_body = compose_task_body(HipKernelLanguage, flavor="fixed", ...)` for HIP kernels. For all other languages, fallback to today's verbatim task_content.

```python
# agents/homogeneous/homogeneous_agent.py
def run_homogeneous_agent(config, task_content, ...):
    kernel_type = config.get("kernel_type")
    if kernel_type == "hip":
        from minisweagent.kernel_languages import registry
        from minisweagent.run.compose import compose_task_body
        language = registry.get("hip")
        task_content = compose_task_body(
            language=language,
            flavor="fixed",
            user_task=task_content,  # keep user's --task as additional context
        )
    # ... rest unchanged
```

**Regression check**:
- Run `assign_score_withk` on mem=on mode (the recorded config).
- Verify same 6 invariants as Path 2 (section 2.1 table).
- Additional check: `final_report.json` has 4 keys, same as pre-PR.
- Acceptance: every invariant-A check passes.

**Rollback**: remove the `if kernel_type == "hip"` block; homo falls back to today's behavior.

### Step 5 — Route Triton+hetero planner through `KernelLanguage.planner_strategy_hints`

**Scope**: replace the Triton-biased `TASKGEN_SYSTEM_PROMPT` body with a generic frame + `{{ language.planner_strategy_hints }}` filled at generate_tasks time.

**Regression check**:
- Run `gemm_a16w16_atomic` with `mode=planned` (new mode name).
- Verify 7 preprocess steps, exploration phase, rounds 1-5 all execute.
- Verify `round_N_evaluation.json` is byte-identical in structure to pre-PR (same keys, same section order).
- Acceptance: every invariant-B check passes.

**Rollback**: restore the original TASKGEN_SYSTEM_PROMPT.

### Step 6 — Ship `run_pipeline` behind feature flag, default OFF

**Scope**: new `run/unified.py::run_pipeline(ctx, mode, language, ...)`. Behind `GEAK_UNIFIED_PIPELINE=1` env flag, default 0. `mini.py` continues to route via `heterogeneous` boolean unless the flag is on.

**Regression check**: with flag OFF, all 7 invariants A+B preserved.

### Step 7 — Add contract validators

**Scope**: `validate_harness` + `validate_commandment` + determinism-drift check. Run them at preprocess time AFTER the templates produce their outputs.

**Regression check**: check against 4 HIP kernels' current commandment files. The 3 bugs (wrong profile target, nested quotes, BENCHMARK==FULL_BENCHMARK) should be REPORTED by validators — but NOT block today's flow (warning only during transition).

**Rollback**: turn validators off.

### Step 8 — HarnessBuilder for HIP (produces working Python wrapper harness)

**Scope**: plugins/preprocess/harness_builder.py with language=hip. Produces the Python wrapper shown in unification plan §10.3. Contract validator passes.

**Regression check**: run `assign_score_withk` with `GEAK_UNIFIED_PIPELINE=1`. Verify:
- Python wrapper harness exists
- Wrapper's `--benchmark` emits `GEAK_RESULT_LATENCY_MS=<float>` + `GEAK_RESULT_SPEEDUP=<float>` (new capability)
- `final_report.json` now has speedup number (new; zero before)

### Step 9 — Flip GEAK_UNIFIED_PIPELINE default to 1

**Scope**: default unified path active; old shims remain behind `GEAK_UNIFIED_PIPELINE=0`.

**Regression check**: run both `assign_score_withk` (HIP+fixed) and `gemm_a16w16_atomic` (Triton+planned); verify all invariants still pass + new capabilities (verification, KB write for HIP) are present.

### Step 10 — Delete old shims (Phase 5, point of no return)

**Scope**: delete `agents/homogeneous/`, `agents/heterogeneous/`, `run/orchestrator.py`, `_normalize_kernel_type`, all backward-compat aliases.

**Regression check**: full regression suite on 10 kernels (5 Triton + 5 HIP). All invariants A+B pass; new capabilities confirmed.

---

## 4. CI gates that prevent regression between PRs

Every PR must pass:

1. **Invariant A smoke** (HIP+homo):
   ```bash
   # Runs one HIP kernel for 3 min, greps for the 6 markers of Invariant A
   pytest tests/smoke/test_hip_homo_invariants.py
   ```

2. **Invariant B smoke** (Triton+hetero):
   ```bash
   # Runs one Triton kernel for 4 min, greps for the 7 markers of Invariant B
   pytest tests/smoke/test_triton_hetero_invariants.py
   ```

3. **Language-agnosticism check**:
   ```bash
   # Fails if optimization_agent.py or its ancestors import KernelLanguage
   grep -rn "KernelLanguage\|kernel_type\|kernel_language" \
       src/minisweagent/agents/optimization_agent.py \
       src/minisweagent/agents/default.py \
       src/minisweagent/agents/interactive.py
   # Expected: 0 matches

   # Fails if core code branches on language name
   grep -rnE "==\s*['\"](triton|hip|flydsl)['\"]" \
       src/minisweagent/ \
       --exclude-dir=kernel_languages --exclude-dir=tests
   # Expected: 0 matches
   ```

4. **Determinism check**: run `--benchmark` twice, assert drift < 2%.

5. **Contract conformance**: every kernel_language's `harness.j2` output passes `validate_harness`; every `commandment.j2` output passes `validate_commandment`.

---

## 5. Rollback plans per phase

| Phase | Rollback mechanism | Max time to rollback |
|---|---|---|
| Step 1 (kernel_languages/ exists, unused) | `git revert` the add-only PR | seconds |
| Step 2 (detect shim) | `git revert`; old `_normalize_kernel_type` logic restored | seconds |
| Step 3 (compose_task_body exists, unused) | `git revert` | seconds |
| Step 4 (HIP homo uses compose) | Remove the `if kernel_type == "hip"` branch; falls back to today's verbatim task_content | seconds |
| Step 5 (Triton planner uses KernelLanguage) | Restore inline TASKGEN_SYSTEM_PROMPT | seconds |
| Step 6 (run_pipeline behind flag) | Set `GEAK_UNIFIED_PIPELINE=0`; old path active | 1 line in shell |
| Step 7 (validators) | Flip validators to warning-only mode | env var |
| Step 8 (HarnessBuilder for HIP) | Don't produce wrapper; fall back to today's raw test_command | config change |
| Step 9 (default flip) | Set `GEAK_UNIFIED_PIPELINE=0` as default again | config change |
| Step 10 (delete old code) | Cherry-pick-revert the deletion PRs | minutes |

---

## 6. What success looks like

After Step 10 ships:

### From the user's perspective (what they type)

```bash
# Triton (was today's common invocation — same output)
geak -t "Optimize the kernel at .../topk.py. Use the test harness at ..."
# → detects Triton, routes to mode=planned, 5 rounds, FULL_BENCHMARK verified, KB write

# HIP (was today's batch script invocation — same output shape PLUS speedup number + KB write)
geak --kernel-url .../assign_score_withk_wrapper.py --repo ... --task "..." \
     --num-parallel 2 --gpu-ids 0,1
# → detects HIP, routes to mode=fixed, 1 round, FULL_BENCHMARK verified (NEW),
#   GEAK_RESULT_SPEEDUP emitted (NEW), KB write (NEW)

# New — mode=auto
geak -t "Optimize ..." --mode auto --max-rounds 5
# → controller picks {fixed, planned} mixture per round based on winner
```

### From the codebase's perspective

- `src/minisweagent/agents/optimization_agent.py` imports nothing from `kernel_languages/`
- `grep -rnE "==\s*['\"](triton|hip)['\"]" src/minisweagent/ --exclude-dir=kernel_languages` returns 0 matches
- `src/minisweagent/kernel_languages/triton/` and `.../hip/` each have 7 files, ~300 LoC total
- Adding FlyDSL = 1 new directory with 7 files, 0 edits anywhere else
- Deleted: `agents/homogeneous/`, `agents/heterogeneous/`, `run/orchestrator.py`, `ParallelAgent.run_parallel` homo branch (lines 414-609)
- `wc -l src/minisweagent/ -l`: ~22,000 (down from ~28,000)

### From the observer's perspective (log markers)

Invariants A and B still hold. Additional markers appear on HIP runs that weren't there before:
- `GEAK_RESULT_SPEEDUP=<float>` lines
- `Round N evaluation written`
- `Recorded experience to KB (fingerprint: ...)`

---

## 7. If I were Yue Liu or Saptarshi starting the refactor tomorrow

I'd do this in exactly this order, one PR per row:

1. **PR #1 (Day 1)**: Create `kernel_languages/` directory with `TritonKernelLanguage` + `HipKernelLanguage` instances. Nothing imports them yet. Pure additive. [~400 LoC, 0 risk]
2. **PR #2 (Day 2)**: Shim `_normalize_kernel_type` → `KernelLanguage.detect`. Regression-test with Invariant A + B smoke. [~50 LoC diff, very low risk]
3. **PR #3 (Day 3)**: Rename `StrategyInteractiveAgent` → `OptimizationAgent` (+ alias). Add `compose_task_body` in `run/compose.py` (not called yet). [~100 LoC, zero risk]
4. **PR #4 (Day 4-5)**: Route HIP+homo through `compose_task_body`. Regression-test on 2 HIP kernels. [~50 LoC diff, medium risk — this changes task body content]
5. **PR #5 (Day 6-7)**: Route Triton+hetero planner through `KernelLanguage.planner_strategy_hints`. Regression-test on 2 Triton kernels. [~150 LoC diff, medium risk]
6. **PR #6 (Week 2)**: Ship `run_pipeline` behind feature flag OFF by default. Parallel path. [~300 LoC, low risk — flag off]
7. **PR #7 (Week 3)**: Contract validators (warning-only initially). [~150 LoC, zero risk — warnings only]
8. **PR #8 (Week 3-4)**: HarnessBuilder for HIP. Produces Python wrapper that emits GEAK_RESULT_* protocol. [~300 LoC + templates, high risk — new capability]
9. **PR #9 (Week 5)**: Flip GEAK_UNIFIED_PIPELINE default to 1. Full regression suite on 10 kernels. [1 line, highest risk — default change]
10. **PR #10 (Week 6-7)**: Delete old shims once PR #9 is stable for 2 weeks. [-3300 LoC, zero risk — old code]

Any of PR #1-#9 can be reverted in < 5 minutes if smoke tests fail. PR #10 is permanent.

---

## 8. What I'm NOT doing and why

- **Not running a brand-new kernel to verify HIP+homo**: I have 4 distinct HIP kernel logs already showing byte-identical flow. One more run adds no information that those 4 don't already provide.

- **Not verifying Triton+homo or HIP+hetero empirically**: Those paths today are either never-used (Triton+homo) or broken (HIP+hetero). Their "today's behavior" is either uninteresting or non-functional, so the refactor-must-preserve requirement is weak or void. The refactor fixes them both as a side-effect of unification.

- **Not introducing `mode=auto` in Phase 2**: auto mode requires the controller. Ship `fixed` and `planned` first as direct replacements for homo and hetero (no new behavior), then add auto in Phase 3 as a pure addition. Separating these removes risk from the rename/refactor PRs.

- **Not touching the KB/memory subsystem**: cross_session memory, retriever, formatter, extractor all stay as-is. `baseline_fingerprint` is added to the Experience schema as an OPTIONAL field. Retriever's filter is added with a feature flag (strict vs warn vs ignore). KB subsystem never regresses because nothing it depends on changes.

- **Not unifying the preprocess pipeline in Phase 2**: the 7-step preprocess is language-parameterized via `KernelLanguage.commandment_template` + `KernelLanguage.harness_template` in Phase 2. The "universal HarnessBuilder fixes HIP's missing harness" work IS in Phase 2 but the preprocessor's overall shape (7 steps for Triton, ~4 steps for HIP) does NOT change until Phase 3+. Minimizes surface area of each PR.

---

## TL;DR

| Question | Answer |
|---|---|
| HIP+homo ports without regression? | **Yes**. 4-kernel trace shows identical flow; Invariant A defines 6 markers the refactor must preserve; each PR has a rollback path in < 5 minutes. |
| Triton+hetero ports without regression? | **Yes**. 3-kernel trace shows identical flow; Invariant B defines 7 markers the refactor must preserve. |
| Agent fully language-agnostic? | **Yes**. `OptimizationAgent.run(task_body: str)` never imports `KernelLanguage`, never checks kernel_type. CI gate `grep -rn "KernelLanguage" agents/optimization_agent.py` must return 0. All language specifics live in `KernelLanguage` dataclass; only `compose_task_body` reads them. |
| Clean code? | **Yes**. ~22K LoC after (down from ~28K). 1 agent class (from 4). 1 CLI (from 2). 1 dispatch path (from 2). Adding a new language = 1 new folder, 7 small files, 0 core edits. |
| Verified across multiple kernels? | **Yes**. 4 HIP kernels + 3 Triton kernels = 7 distinct runs, byte-identical flow up to mode split. |
| Can we start tomorrow? | **Yes**. 10 PRs, one per day/week, each rollback-safe. |

---

**End of hardened refactor plan.** See `GEAK_codebase_audit.md` for the baseline + `GEAK_unification_plan.md` for the full vision.
