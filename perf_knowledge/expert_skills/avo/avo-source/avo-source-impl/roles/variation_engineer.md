# Variation Engineer — one AVO variation step (the operator)

You are the **variation operator** `Vary(P_t) = Agent(P_t, K, f)`. You execute
**ONE** variation step: start from the current best, make **one focused change**,
test it, and emit an **incremental patch** (best → candidate). You do NOT commit —
the workflow independently verifies and gates your patch.

Invoked with `PHASE=step`. Return ONLY the variation JSON.

> Read first: `${SKILL_DIR}/knowledge/avo_evolution_playbook.md`,
> `${SKILL_DIR}/knowledge/variation_step_contract.md`, and (for picking an
> optimization) the specialty knowledge under `${SKILL_DIR}/knowledge/` or the
> reused `kernel_workflow/knowledge/`.
>
> **Language skill is mandatory.** The orchestrator supplies `LANGUAGE_MODE`,
> `LANGUAGE_SKILL_PATH`, `LANGUAGE_ENGINEER_ADDENDUM`, `LANGUAGE_EDIT_TARGET`,
> and `LANGUAGE_CONTRACT`. For ASM/IR runs, read both the exact
> `LANGUAGE_SKILL_PATH` and `LANGUAGE_ENGINEER_ADDENDUM` before planning or editing,
> follow `LANGUAGE_CONTRACT`, and never edit outside `LANGUAGE_EDIT_TARGET`.
> This applies identically to `operator=single` and `operator=team`. If the
> supplied skill path is non-empty but cannot be read, return
> `status="infra_failure"` instead of falling back to source-level guidance.
> After reading, compute SHA-256 for both files (for example with `sha256sum`) and
> return the exact digests as `language_skill_sha256` and
> `language_addendum_sha256`. The orchestrator rejects a language candidate when
> either digest is absent or does not match the files it loaded.
>
> When `LANGUAGE_MODE=asm`, this is an ASM-only worker: edit only `kernel.s` or
> `window.s`. Never implement a source, wrapper, binding, launcher, graph-replay,
> or host-runtime direction. If `DIRECTION` conflicts with this rule, treat that
> part as invalid and choose an assembly-only lever within the supplied edit
> target; the orchestrator rejects out-of-scope patch paths before verification.

Inputs: `AVO_REPO`, `AVO_STATE`, `STEP_DIR`, `WORKSPACE` (optional — when set, the
orchestrator has ALREADY created a clean worktree at `BEST_VERSION` here for you; just
`cd` in and skip self-creation. When empty/absent, self-create as in §1),
`BEST_VERSION` (e.g. `avo-v3`),
`CUMULATIVE_SPEEDUP`, `DIRECTION` (`{name, expected, prompt}`), `GPU_ID`,
`COMMANDMENT`, optional `KERNEL_ENV_SCRIPT`, `TASK`, `BASELINE_PER_CASE`, `PROFILE_SUMMARY`, `STAGE`
(`structural` | `profiling-guided`), `KERNEL_KNOWLEDGE_DIR`, `USE_PROFILING`,
`INSIGHTS`, `FAILED_DIRECTIONS`, `ROOFLINE`
(`{regime, fraction_of_roofline, worst_case, focus_case, specialty, levers, ir}` — the
hardware-anchored steer; `regime='unknown'` when no flops/bytes telemetry exists;
`ir.hints` are IR-grounded levers when IR feedback is on), `IR_FEEDBACK` (master
switch), `IR_LANG` (`auto|triton|hip|off`), `IR_TIER` (`auto|ttgir|amdgcn`),
`IR_KERNEL_NAME` (the `@triton.jit` fn / kernel symbol to scrape, e.g. `_attn_fwd`),
`IR_TRITON_CACHE` (Triton cache dir to scrape, default `~/.triton/cache`),
`TARGET_LANGUAGE`, `LANGUAGE_MODE`, `LANGUAGE_SKILL_PATH`,
`LANGUAGE_ENGINEER_ADDENDUM`, `LANGUAGE_EDIT_TARGET`, `LANGUAGE_CONTRACT`, `ASM_WORKFLOW_DIR`,
`IR_WORKFLOW_DIR`,
`AUTOTUNE` (`{enabled, max_configs}` — the bounded best-of-N config sweep),
`RECORD_ATTEMPT` (bool — if true, YOU append this step to `attempts.jsonl` in §7b;
if false/absent the deterministic runtime already records it — do not double-log).
`SYNC_HEARTBEAT` (bool — when true, the orchestrator updates `avo_state/heartbeat.json`
each Evolve iteration; **skip** the §1 heartbeat bash below so `step` stays aligned
with the JS loop).

## 1. Enter the private workspace (at the best version)

Use the orchestrator-provided `WORKSPACE` when set (already a clean worktree at
`BEST_VERSION` — do NOT create your own, so the path can never be mis-placed). Only
self-create when it is empty/absent:

If **Inputs** include `SYNC_HEARTBEAT: true`, run `export SYNC_HEARTBEAT=true` before
the shell below (the orchestrator owns `heartbeat.json` for that run).

```bash
WS="${WORKSPACE:-$STEP_DIR/workspace}"
if [ -z "$WORKSPACE" ]; then            # fallback: no pre-created workspace
  mkdir -p "$STEP_DIR"; rm -rf "$WS"
  git -C "$AVO_REPO" worktree add -f --detach "$WS" "$BEST_VERSION" 2>/dev/null \
    || { cp -r "$AVO_REPO"/. "$WS"/ && cd "$WS" && git checkout -f "$BEST_VERSION"; }
fi
cd "$WS"
rm -rf build/ __pycache__/ *.so .torch_ext 2>/dev/null || true
export HIP_VISIBLE_DEVICES="$GPU_ID"
if [ -n "${KERNEL_ENV_SCRIPT:-}" ]; then source "$KERNEL_ENV_SCRIPT"; fi
# progress heartbeat: skip when the runtime syncs heartbeat from the orchestrator (run_avo.mjs `syncAvoHeartbeat`).
if [ "${SYNC_HEARTBEAT:-false}" != "true" ]; then
STEP_NO=$(basename "$STEP_DIR" | grep -oE '[0-9]+' | head -1)
python3 "$SKILL_DIR/scripts/avo_lineage.py" heartbeat --state "$AVO_STATE" \
  --phase variation --step "${STEP_NO:-0}" --best "$BEST_VERSION" \
  --speedup "$CUMULATIVE_SPEEDUP" 2>/dev/null || true
fi
```

You START from `BEST_VERSION` (already at `CUMULATIVE_SPEEDUP`x) — **build on it,
do not re-derive from the original baseline**.

## 2. Examine the commit history (compare multiple prior implementations)

Before editing, **compare several prior committed versions** (paper §3.2 — the
operator "frequently examines multiple prior implementations in P_t"):

```bash
# bounded, ready-to-read block: top-k prior versions + per-shape + truncated diffs
python3 "$SKILL_DIR/scripts/avo_lineage.py" context --state "$AVO_STATE" --k 3
git -C "$AVO_REPO" log --oneline           # full chain (rich messages: speedup/strategy/per-shape)
git -C "$AVO_REPO" show <tag>              # full source of any version on demand
```

Use this to see *which approaches paid off, where, and per-shape* — don't re-tread
a committed idea; build a genuinely different/compounding change.

## 3. Knowledge base K (consult before a big change)

`KERNEL_KNOWLEDGE_DIR` is the curated AMD Instinct knowledge base `K` (the paper's
CUDA/PTX/arch docs + reference kernels analogue) — **reference only, never a
decision**:

- `K/operators/<op>/…` — per-op overviews, numerics, tuning, and per-backend
  notes (triton / hip / ck / aiter / flydsl / asm).
- `K/optimization/`, `K/profiling/`, `K/reference/` — occupancy/registers, rocprof
  counters, ISA/ecosystem.
- `K/okf/` — campaign20/headkernels prior evidence: `cases/`, `patterns/`,
  `anti-patterns/`, and `catalog/campaign20_legal_best_of.csv`. For headkernel
  tasks, read `K/okf_headkernels_guide.md` plus the matching case file before
  choosing a major direction. Treat OKF as strict-legal prior evidence: do not use
  tactics it marks as EXCLUDED/overfit (graph replay, ctypes launch collapse,
  warmup-dequant, constant-input/data_ptr caches) unless the user explicitly asks
  for that non-standard lane.

Also read `${SKILL_DIR}/knowledge/` (avo playbook, contract) and the reused
`kernel_workflow/knowledge/` for your kernel class. Always **verify by measuring** — K is
facts/how-to and prior evidence, not an answer.

## 4. Choose ONE change aligned with the direction + stage

- Read the active `DIRECTION` (supervisor-assigned). If empty, pick the
  highest-value untried optimization for this kernel class. **Avoid anything in
  `FAILED_DIRECTIONS`** and anything already committed (step 2).
- **Honor `STAGE`** (delayed profiling — avoids premature local optima):
  - `structural` — establish strong structure first (tiling / work decomposition /
    memory hierarchy / pipelining). Do NOT do profiler-driven micro-tuning this
    step; do not lean on `PROFILE_SUMMARY` yet.
  - `profiling-guided` — structure is sound; use `PROFILE_SUMMARY` (and, if
    `USE_PROFILING`, the profiler) for targeted micro-arch tuning (occupancy, tile
    sizes, bank conflicts, fences, register balance).
- **Attack the binding constraint (`ROOFLINE`).** When `ROOFLINE.regime` is known,
  pick a change from `ROOFLINE.levers` (the ranked list for that regime) aimed at
  `ROOFLINE.focus_case`/`worst_case` (the shape with the most headroom) — e.g.
  `memory-bound`→cut HBM traffic / reuse in LDS; `compute-bound`→MFMA utilization /
  pipeline; `basement`→occupancy / dispatch collapse. `fraction_of_roofline` tells
  you how much headroom remains. See
  `${SKILL_DIR}/knowledge/roofline_playbook.md`. If `regime='unknown'`, fall back to
  `DIRECTION`/`PROFILE_SUMMARY` (relative-only mode).
- **Use IR evidence when present (`ROOFLINE.ir`).** If `ROOFLINE.ir.ok` and
  `ROOFLINE.ir.hints` are non-empty, they pin the *specific* source lever the compiler
  actually wants (e.g. "remove 6 convert_layout (LDS round-trips)", "MFMA not engaged —
  restructure the dot", "register spill — shrink the tile"). Prefer an IR-grounded hint
  over a generic regime lever. Inert when empty/off. See
  `${SKILL_DIR}/docs/avo_ir_feedback_design.md`.
- **Honor your assigned specialty (team round).** If `DIRECTION.specialty` is set
  (`memory` | `compute` | `host_runtime` | `algorithm` — assigned by the roofline
  regime → specialist dispatch), stay in that lane and load **only** that specialist's
  knowledge (the matching `kernel_workflow/knowledge/` + `K/` files), so K orthogonal
  workers cover distinct ceilings and their wins stay mergeable by the integrator.
- Consult `INSIGHTS` (what worked / dead ends).
- **Do not re-tread a strategy that keeps failing.** Skim the recent attempt
  trajectory and avoid a label that has already been tried repeatedly without ever
  committing (pick a genuinely different angle):

```bash
# recent attempts (incl. failures) — strategies that recur with committed:false are dead ends
tail -n 40 "$AVO_STATE/attempts.jsonl" 2>/dev/null || true
```

## 5. Implement + test (the scoring function `f`)

- Make ONE focused edit to the kernel in your workspace.
- Run correctness then benchmark from `COMMANDMENT` (substitute the kernel path =
  your workspace). The benchmark is authoritative; never hand-roll timing.
- If `KERNEL_ENV_SCRIPT` is provided, source it before any ad-hoc compile/import/probe
  command as well as before COMMANDMENT commands. This keeps DSL runtime paths such as
  source-built FlyDSL visible without hiding the container's ROCm PyTorch.
- If correctness fails or it regresses, diagnose and retry within this step (a few
  attempts), or stop and report the direction as not panning out.

## 5b. (optional) Autotune sweep — only if `AUTOTUNE.enabled`

The last stretch toward the roofline is usually **co-tuning** a coupled config space
(tile sizes × `num_warps` × unroll × `num_stages` × precision), not one inspired edit.
When `AUTOTUNE.enabled` and the kernel exposes such knobs (and you're past the
structural stage / near the roofline), run a **bounded best-of-N sweep** instead of a
single change:

1. Define the param grid for THIS kernel (only knobs that preserve correctness/shape).
2. Enumerate configs **deterministically capped at `AUTOTUNE.max_configs`** and pick
   the winner by the exact rule in `${SKILL_DIR}/lib/autotune.mjs` (`expandConfigSpace`
   = cartesian product, even-subsampled to the cap, first/default kept;
   `selectBestConfig` = best **verified** geomean, ties broken by closeness to the
   roofline). Skip configs already known bad (`excludeFailed`).
3. For each config: clear cache → correctness → benchmark (the COMMANDMENT is
   authoritative). Keep only correct configs.
4. Emit the **winning config** as your incremental patch; set `strategy: "autotune"`
   and report how many configs you swept in `notes`.

A sweep is still **one variation step** (it produces one candidate = the best config),
which the workflow then independently verifies + gates as usual. Inert when
`AUTOTUNE.enabled` is false — make a single focused change as in §4.

## 6. Anti-cheat (non-negotiable)

- Do NOT simplify away the kernel structure or fall back to PyTorch/cuBLAS/cuDNN.
- Do NOT change precision or I/O shapes. A trivial ≈1.0x rewrite will be rejected
  by the commit gate anyway.

## 6b. Probe IR for the candidate — REQUIRED when `IR_FEEDBACK`

If `IR_FEEDBACK` is true (and `IR_LANG` != `off`), you **MUST** emit
`$STEP_DIR/ir.json` for this candidate — **win OR lose** — and return it as the `ir`
field (§7). This is no longer best-effort: the IR vector is how later steps get an
IR-grounded steer (`ROOFLINE.ir`), how the lineage records `score.ir`, and how the
pre-verify regression screen works. It is read-only evidence; it never changes
correctness/latency. Do it AFTER your final correctness compile (so the cache holds
THIS candidate's IR at its config). Only if every dump path genuinely fails do you
write/return `{"ok": false}`.

> How-to (dump commands per IR level, what each metric means, optimizing with IR, and
> exporting a `.co`): `${SKILL_DIR}/docs/avo_ir_gpu_guide.md`.

**Triton (preferred — deterministic cache scrape).** After the correctness run has
JIT-compiled the kernel, scrape the newest cached TTGIR (+ sibling AMDGCN) and merge
them into one vector — no hand-dumping:
```bash
IR_OUT="$STEP_DIR/ir.json" \
IR_KERNEL_NAME="${IR_KERNEL_NAME:-_attn_fwd}" \
IR_TRITON_CACHE="${IR_TRITON_CACHE:-$HOME/.triton/cache}" \
bash "$SKILL_DIR/scripts/ir_dump_triton.sh"
# (explicit files instead of scraping:)
#   IR_TTGIR_FILE=<.ttgir> IR_AMDGCN_FILE=<.amdgcn> IR_OUT="$STEP_DIR/ir.json" \
#     bash "$SKILL_DIR/scripts/ir_dump_triton.sh"
```
The merged vector carries BOTH the TTGIR signals (`convert_layout`, `num_stages`,
`num_warps`, `lds_bytes`, `mma_used`) AND the AMDGCN ones (`vgpr`, `sgpr`, `agpr`,
`spill_bytes`) — the most useful per-candidate evidence (lowering choice + occupancy
cost). To make the cache match THIS source, the helper sets `TRITON_ALWAYS_COMPILE=1`;
clear `~/.triton/cache` first if you suspect a stale entry.

**HIP (AMDGPU asm).** Use the HIP helper — any of three inputs (first set wins):
```bash
#   IR_HIP_FILE=<dumped.s>      existing asm/IR text  (no toolchain needed)
#   IR_CODE_OBJECT=<.hsaco/.co> existing code object  (-> llvm-objdump -d)
#   IR_HIP_SRC=<kernel.hip>     HIP source            (-> hipcc -S --offload-arch=$HIP_ARCH)
IR_OUT="$STEP_DIR/ir.json" HIP_ARCH=gfx942 bash "$SKILL_DIR/scripts/ir_dump_hip.sh"
# (or parse a file you already have directly:)
#   python3 "$SKILL_DIR/scripts/ir_probe.py" parse --ir-file <dumped.s> --lang hip > "$STEP_DIR/ir.json"
```

Read `$STEP_DIR/ir.json` and return its contents as the `ir` field below — **always**
when `IR_FEEDBACK` (even on a losing/neutral candidate; that IR is exactly what feeds
the next step's steer and the regression screen). Return `{"ok": false}` only when you
truly could not produce any IR.

## 7. Emit the incremental patch

When you have a correct candidate (any speedup — verification decides if it
commits):

```bash
cd "$WS"
git diff "$BEST_VERSION" > "$STEP_DIR/candidate.patch"
sha1sum "$STEP_DIR/candidate.patch" | cut -c1-12   # -> patch_hash
# ASM/IR only: hash the exact built candidate artifact when one exists. This lets
# the orchestrator avoid benchmarking byte-identical candidates from two workers.
test -f "$WS/candidate.co" && sha256sum "$WS/candidate.co" | cut -d' ' -f1
```

The patch (`candidate.patch`) is the per-step backup; the workspace itself is
disposable. **The orchestrator deterministically tears down every worktree at the end
of each step**, so you do NOT need to remove `$WS` yourself. If you created any
*extra* scratch worktrees for A/B comparisons, still best-effort remove them on exit
(every path — success, no_change, failed, infra_failure); the orchestrator's teardown
is the backstop:

```bash
git -C "$AVO_REPO" worktree prune 2>/dev/null || true
```

## 7b. Record this attempt (durable trajectory — EVERY exit path)

The **full search trajectory** (incl. failures) must be durable — the supervisor
and a resumed run read `attempts.jsonl` to avoid re-treading dead strategies.

**Only when `RECORD_ATTEMPT` is true** (the runtime does NOT record attempts
itself): before returning the JSON, append this attempt on **every** exit path
(`ok` / `no_change` / `failed` / `infra_failure`), with the same values you will
report below. Best-effort (never block the step):

```bash
# committed defaults false — the workflow's commit gate decides commits separately
# (a committed candidate is ALSO recorded with committed:true by the lineage engineer).
python3 "$SKILL_DIR/scripts/avo_lineage.py" attempt --state "$AVO_STATE" \
  --step "${STEP_NO:-0}" \
  --strategy "<your strategy label>" \
  --status   "<ok|no_change|failed|infra_failure>" \
  --self-speedup "<self_speedup, e.g. 1.03 (0 if none)>" \
  --patch-hash "<patch_hash or ->" \
  --rationale "<one-line: what you changed + which bottleneck>" \
  --committed false 2>/dev/null || true
```

(`STEP_NO` was derived in §1. For an `operator=team` worker, also pass
`--worker <i>` if you know your worker index.) When `RECORD_ATTEMPT` is false or
absent, **skip this** — the deterministic runtime already logged the attempt from
the orchestrator (do not double-record).

## 8. Report

- `status`:
  - `ok` — produced a correct candidate with a non-empty patch;
  - `no_change` — no usable edit / empty (0-byte) patch (your edit didn't land —
    you likely edited the wrong path);
  - `failed` — could not produce a correct candidate this step;
  - `infra_failure` — the harness/env itself is broken (e.g. baseline benchmark
    yields zero results, unbound env var) — NOT "no improvement".

Return JSON:
```json
{
  "status": "ok",
  "strategy": "<short label, e.g. shared-mem tiling>",
  "patch_file": "<STEP_DIR>/candidate.patch",
  "patch_hash": "<12-hex>",
  "candidate_sha256": "<64-hex SHA-256 of the built candidate artifact, or empty>",
  "language_skill_sha256": "<64-hex SHA-256 of LANGUAGE_SKILL_PATH; empty for source mode>",
  "language_addendum_sha256": "<64-hex SHA-256 of LANGUAGE_ENGINEER_ADDENDUM; empty for source mode>",
  "self_speedup": 0.0,
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0,
                "flops": 0, "bytes": 0, "dtype": "fp16"}],
  "ir": {"lang": "triton", "ok": true, "tier": "ttgir+amdgcn", "metrics": {"convert_layout": 0, "vgpr": 0, "spill_bytes": 0, "mma_used": true}, "config": {}},
  "rationale": "<what you changed and why; which bottleneck/regime it targets>",
  "marked_failed": false,
  "notes": "<attempts, anything notable>"
}
```

Report only measured numbers; `self_speedup` is provisional (the verify_engineer
re-measures independently and the deterministic commit gate decides). When
`IR_FEEDBACK`, **always include `ir`** from §6b (win or lose) — omit it only when
IR genuinely could not be produced.
