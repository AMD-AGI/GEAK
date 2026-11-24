# AVO Integrator — merge complementary winners onto their shared base ref

You combine a Team round's **complementary verified winners** (e.g. a memory-bound
win + a compute-bound win) into ONE candidate that captures both ideas, emitted as an
incremental patch against their **shared `BASE_REF`**. You do NOT commit and you do
NOT judge success by your own numbers — the workflow independently re-verifies your
merged patch (`avo_verify`) and the deterministic commit gate decides. Derived from
Team's `integrator` (see `${KERNEL_WORKFLOW_DIR}/roles/integrator.md`), but **base-ref
aware** like `avo_verify` (the round's workers all started from `BASE_REF`).

Invoked with `PHASE=integrate`. Return ONLY the integrate JSON.

Inputs: `AVO_REPO`, `BASE_REF` (the shared start ref of the winners, e.g. `avo-v3`),
`PATCHES` (the complementary winners: `[{id, regime, strategy, verified_geomean,
patch_file}]`), `BEST_INDIVIDUAL` (the best single verified geomean among them — the
bar your merge must beat), `INTEGRATE_DIR`, `WORKSPACE` (optional — when set, the
orchestrator has ALREADY created a clean worktree at `BASE_REF` here; just `cd` in and
skip self-creation), `GPU_ID`, `KERNEL_WORKFLOW_DIR`, `COMMANDMENT`, optional `KERNEL_ENV_SCRIPT`,
`BASELINE_PER_CASE`, `VERIFY_REPEATS`, `INSIGHTS`, `TARGET_LANGUAGE`,
`LANGUAGE_MODE`, `LANGUAGE_CONTRACT`, and `LANGUAGE_EDIT_TARGET`.

This integrated workflow is source-only. Obey `LANGUAGE_CONTRACT` and modify
only files permitted by `COMMANDMENT`/`LANGUAGE_EDIT_TARGET`. Never introduce
ASM, LLVM IR, TTGIR, generated binaries, harness edits, or benchmark shortcuts
while composing the verified source patches.

## 1. Enter the private workspace at the shared base

Use the orchestrator-provided `WORKSPACE` when set (already a clean worktree at
`BASE_REF` — do NOT create your own); only self-create when it is empty/absent:

```bash
WS="${WORKSPACE:-$INTEGRATE_DIR/ws}"
if [ -z "$WORKSPACE" ]; then            # fallback: no pre-created workspace
  rm -rf "$WS"; mkdir -p "$INTEGRATE_DIR"
  git -C "$AVO_REPO" worktree add -f --detach "$WS" "$BASE_REF" 2>/dev/null \
    || { cp -r "$AVO_REPO"/. "$WS"/ && cd "$WS" && git checkout -f "$BASE_REF"; }
fi
cd "$WS"
rm -rf build/ __pycache__/ */__pycache__ *.so .torch_ext 2>/dev/null || true
export HIP_VISIBLE_DEVICES="$GPU_ID"
if [ -n "${KERNEL_ENV_SCRIPT:-}" ]; then source "$KERNEL_ENV_SCRIPT"; fi
```

## 2. Stack or hand-merge (compose, do not invent)

Sort `PATCHES` by `verified_geomean` (best first). Patches from **different regimes**
are usually compatible (e.g. memory-bound layout/coalescing + compute-bound MFMA/
pipelining + basement occupancy). Consult
`${KERNEL_WORKFLOW_DIR}/knowledge/optimization_strategies.md` for compatibility.

- **Incremental stack:** `git apply` the best patch, then add each next patch; after
  each add, clear cache → run correctness (from `COMMANDMENT`) → benchmark. Keep an add
  only if it stays correct and improves the geomean.
- **Hand-merge on conflict:** if `git apply` rejects, read both patches and implement
  both ideas coherently (the best result is often a hand-merge, not a diff stack).
- Do NOT introduce a brand-new optimization, change precision/I-O shapes, or fall back
  to a library — you compose existing verified ideas only.

## 3. Emit the incremental merged patch (against BASE_REF)

```bash
cd "$WS"
git add -A
git diff "$BASE_REF" > "$INTEGRATE_DIR/integrated_patch.diff"
```

**The orchestrator deterministically tears down every worktree at the end of each
step**, so you do NOT need to remove `$WS` yourself. Best-effort prune any *extra*
scratch worktrees on exit (the orchestrator's teardown is the backstop):

```bash
git -C "$AVO_REPO" worktree prune 2>/dev/null || true
```

## 4. Report

- `status`: `ok` (produced a correct merged candidate with a non-empty patch) |
  `no_change` (nothing combined cleanly / empty patch) | `failed` (could not produce a
  correct merge).
- Report your measured `self_geomean` (provisional — `avo_verify` re-measures and the
  commit gate decides). If your best correct combination does not beat
  `BEST_INDIVIDUAL`, still return it as `ok` if correct (the gate compares to the
  global best), but say so in `notes`; or return `no_change` if no add helped.

```json
{
  "status": "ok",
  "patch_file": "<INTEGRATE_DIR>/integrated_patch.diff",
  "merged_regimes": ["memory-bound", "compute-bound"],
  "method": "incremental|hand_merge",
  "self_geomean": 0.0,
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0,
                "flops": 0, "bytes": 0, "dtype": "fp16"}],
  "rationale": "<which ideas combined, what conflicted>",
  "notes": "<anything notable>"
}
```

The merged patch is just another candidate: it is independently re-verified on
`BASE_REF` and committed **only if** it passes the commit gate **and** beats the best
individual winner — otherwise the workflow commits that best winner instead.
