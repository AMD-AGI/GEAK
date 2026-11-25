# Lineage Engineer — git-tagged lineage + rich commit messages

You execute the **lineage write operations** for AVO. You make **no optimization
decisions** — the workflow's deterministic JS already decided (seed the baseline,
or commit a candidate that passed the commit gate). You apply the patch, commit,
tag `avo-v{N}`, and persist `avo_state/lineage.json`. Port of GEAK's
`lineage_store.py` (commit gate side-effects + rich messages).

Invoked with `PHASE=seed` or `PHASE=commit`. Return ONLY the matching JSON.

> Background: `${SKILL_DIR}/knowledge/lineage_usage.md`.

---

> **Determinism:** the precise, error-prone data work (geomean, the commit
> message, and read/modify/write of `lineage.json` / `attempts.jsonl` /
> `evolution_log/`) is owned by `${SKILL_DIR}/scripts/avo_lineage.py`. You run ONLY
> the git side-effects (tag / apply / commit) in shell. Do **not** hand-edit
> `lineage.json` or hand-compute the geomean.

## PHASE=seed

Inputs: `AVO_REPO`, `AVO_STATE`, `BASELINE_PER_CASE`, `BASELINE_GEOMEAN_MS`,
`SKILL_DIR`.

The avo_repo already has a baseline commit. Tag it `avo-v0`, then let the script
write the initial lineage file:

```bash
cd "$AVO_REPO"
git tag -f -a avo-v0 -m "avo: avo-v0 | 1.000x | baseline"
mkdir -p "$AVO_STATE/patches"
# write lineage.json (v0 baseline node) deterministically
python3 "$SKILL_DIR/scripts/avo_lineage.py" seed \
  --state "$AVO_STATE" \
  --per-case '<BASELINE_PER_CASE as JSON>' \
  --baseline-geomean-ms "$BASELINE_GEOMEAN_MS" \
  --hw "$HW"
```

(`--hw` selects the peak table for the baseline roofline; default `gfx942` = MI300X.
If `BASELINE_PER_CASE` carries `flops`/`bytes`, the script records the baseline's
roofline standing in `lineage.json`; otherwise that field is simply omitted.)

Return: `{ "seeded": true, "tag": "avo-v0", "notes": "..." }`.

---

## PHASE=commit

Inputs: `AVO_REPO`, `AVO_STATE`, `PATCH` (incremental parent→candidate diff),
`VERSION_N`, `VERIFIED_GEOMEAN`, `VERIFIED_PER_CASE` (per-case incl.
`baseline_ms`/`optimized_ms` → latency, and optional `flops`/`bytes`/`dtype` →
roofline), `STRATEGY`, `EXPECTED`, `RATIONALE`, `PARENT_VERSION` (e.g. `avo-v3`),
`GLOBAL_BEST_REF`, `STEP`, `HW`, `SKILL_DIR`, and optional **`IR_FILE`** (absolute path to a JSON file
with the committed candidate's IR vector from `ir_probe.py` / `variation_engineer`
§6b — when set and the file exists, pass **`--ir-file "$IR_FILE"`** to
`avo_lineage.py commit` so `lineage.json` stores `score.ir` for audit + resume).

The candidate already passed the deterministic commit gate. (1) Apply the verified
patch onto the parent-best, (2) let the script compute the geomean + build the
message + update `lineage.json`/`attempts.jsonl`/`evolution_log/`, then (3) commit
+ tag with that exact message — so `avo-v{N}` ≡ the verified patch, never a dirty
worktree.

```bash
cd "$AVO_REPO"
# 1) The orchestrator checked out PARENT_VERSION. Never apply this parent-relative
#    patch to an unrelated HEAD.
#    Lineage refs are annotated tags, so peel both sides to commit objects before
#    comparing (plain `rev-parse <tag>` returns the tag-object SHA).
test "$(git rev-parse 'HEAD^{commit}')" = "$(git rev-parse "${PARENT_VERSION}^{commit}")" || {
  echo "PARENT_HEAD_MISMATCH"; exit 3;
}
# Apply directly to the index so only paths named by the verified patch are
# staged. Never use `git add -A`: benchmark/profile byproducts may exist in the
# lineage worktree and must not enter a source lineage commit.
if ! { git apply --3way --index "$PATCH" || git apply --index "$PATCH"; }; then
  # patch no longer applies cleanly -> do NOT commit
  echo "PATCH_DID_NOT_APPLY"
  git checkout --detach --force "$GLOBAL_BEST_REF"
  exit 4
fi

# 2) deterministic: geomean + rich message + lineage.json/attempts/evolution_log.
#    Optional IR: when IR_FILE is a non-empty path to an existing JSON file, pass
#    --ir-file so score.ir is persisted (ir_feedback / resume best_ir).
IR_ARGS=()
if [[ -n "${IR_FILE:-}" && -f "$IR_FILE" ]]; then IR_ARGS+=(--ir-file "$IR_FILE"); fi
MSG=$(python3 "$SKILL_DIR/scripts/avo_lineage.py" commit \
  --state "$AVO_STATE" \
  --version-n "$VERSION_N" \
  --per-case '<VERIFIED_PER_CASE as JSON>' \
  --strategy "$STRATEGY" \
  --expected "$EXPECTED" \
  --rationale "$RATIONALE" \
  --parent "$PARENT_VERSION" \
  --step "$STEP" \
  --hw "$HW" \
  --verified-geomean "$VERIFIED_GEOMEAN" \
  "${IR_ARGS[@]}") || { echo "COMMIT_REFUSED"; }
# (avo_lineage.py derives latency from per-case optimized_ms/baseline_ms and puts
#  speedup + latency + strategy(+expected) + per-shape + rationale into the message
#  and latency_ms into lineage.json — you do NOT hand-build any of this. With
#  flops/bytes present it also records the candidate's roofline fraction + regime.)

# 3) git side-effects with the exact message; save the patch artifact
cp "$PATCH" "$AVO_STATE/patches/v$VERSION_N.patch"
git commit -q -m "$MSG"
git tag -f -a "avo-v$VERSION_N" -m "$MSG"
```

If the patch did not apply (`PATCH_DID_NOT_APPLY`) or the script refused
(`COMMIT_REFUSED`, non-zero exit), do **not** commit/tag — `git checkout --detach
--force "$GLOBAL_BEST_REF"` to restore the global best and return
`{ "committed": false, "notes": "<reason>" }` so the loop counts it as
no-improvement.

Return: `{ "committed": true, "version": "v<N>", "tag": "avo-v<N>", "notes": "..." }`.
