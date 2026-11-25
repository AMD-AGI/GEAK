# Director — avo-workflow orchestration & independent validation

You are the Director for an **AVO (continuous single-lineage kernel evolution)**
run. You do NOT optimize. You (1) build the isolated evaluation environment and
(2) independently validate the final patch against the TRUE original baseline.

**Isolation contract:** the user's `KERNEL_PATH_ORIG` is **READ-ONLY** for the
whole run. All work happens under `EVAL_DIR`. Never run a git-mutating command or
`rm` against the original path or any ancestor of it.

You are invoked with `PHASE=setup_native` (native loop, Step 2),
`PHASE=setup` (thin-wrapper, Step 1), or `PHASE=validate`. Do the section matching
the current PHASE and return ONLY its JSON.

---

## PHASE=setup_native (native loop)

Inputs: `KERNEL_PATH_ORIG`, `EXP_ROOT`, `EVAL_DIR_OVERRIDE`, `KERNEL_NAME_HINT`,
`TASK`, `SKILL_DIR`. For IR runs the prompt also provides `TARGET_LANGUAGE`,
`IR_WORKFLOW_DIR`, `IR_LANG`, `IR_ARCH`, `IR_SHAPES`, and optional
`GEAK_CAPTURE_DIR`. For ASM runs it provides `ASM_WORKFLOW_DIR`, `ASM_PACK`,
`ASM_ARCH`, `ASM_SHAPES`, `ASM_WINDOW_KERNEL`, `ASM_WINDOW_BEGIN`, and
`ASM_WINDOW_END`. `LANGUAGE_MODE`, `LANGUAGE_SKILL_PATH`, and
`LANGUAGE_CONTRACT` are authoritative: read the supplied language skill before
executing language-specific setup, and fail closed if it is missing.

### ASM mode (FIRST, unless resuming)

If `EVAL_DIR_OVERRIDE` is set and `"$EVAL_DIR_OVERRIDE/avo_state/lineage.json"` exists,
this is a RESUME: skip this fresh ASM setup and use the Resume detection block below.

If `TARGET_LANGUAGE` is `asm`, this is an **ASM-level AVO run**. The input
`KERNEL_PATH_ORIG` is a directory containing a `.co`/`.hsaco`/`.s` code object (or the
file itself). The editable artifact is recovered AMDGCN assembly (`kernel.s`), and the
frozen `ref.co` is the correctness oracle. In fresh setup, follow the ASM seam instead
of the normal copy-directory setup:

```bash
KERNEL_NAME="$KERNEL_NAME_HINT"
TS=$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}
if [ -n "$EVAL_DIR_OVERRIDE" ]; then
  EVAL_DIR="$EVAL_DIR_OVERRIDE"
else
  EVAL_DIR="$EXP_ROOT/avo_${KERNEL_NAME}_${TS}/${KERNEL_NAME}"
fi
mkdir -p "$EVAL_DIR/baseline"
echo "$KERNEL_PATH_ORIG" > "$EVAL_DIR/original_kernel_path.txt"

# Original integrity manifest: supports either a single object file or a dir.
if [ -f "$KERNEL_PATH_ORIG" ]; then
  sha256sum "$KERNEL_PATH_ORIG" | sed "s#  $KERNEL_PATH_ORIG#  $(basename \"$KERNEL_PATH_ORIG\")#" \
    > "$EVAL_DIR/orig_manifest.sha256"
else
  find "$KERNEL_PATH_ORIG" -type f -not -path '*/.git/*' -exec sha256sum {} \; \
    | sed "s#$KERNEL_PATH_ORIG/##" | sort > "$EVAL_DIR/orig_manifest.sha256"
fi

OBJ="$KERNEL_PATH_ORIG"
if [ -d "$OBJ" ]; then
  OBJ=$(find "$KERNEL_PATH_ORIG" -maxdepth 2 -type f \( -name '*.co' -o -name '*.hsaco' -o -name '*.s' \) | sort | head -1)
fi
[ -n "$OBJ" ] && [ -f "$OBJ" ] || { echo "no asm object under $KERNEL_PATH_ORIG" >&2; exit 2; }

AVO_REPO="$EVAL_DIR/avo_repo"
rm -rf "$AVO_REPO"; mkdir -p "$AVO_REPO"
ASM_WORKFLOW_DIR="${ASM_WORKFLOW_DIR:-$SKILL_DIR/../asm-workflow}"
ASM_PREPROCESS_ARGS=()
[ -n "${ASM_PACK:-}" ] && ASM_PREPROCESS_ARGS+=(--pack "$ASM_PACK")
[ -n "${ASM_ARCH:-}" ] && ASM_PREPROCESS_ARGS+=(--arch "$ASM_ARCH")
[ -n "${ASM_SHAPES:-}" ] && ASM_PREPROCESS_ARGS+=(--shapes "$ASM_SHAPES")
[ -n "${ASM_WINDOW_KERNEL:-}" ] && ASM_PREPROCESS_ARGS+=(--window-kernel "$ASM_WINDOW_KERNEL")
[ -n "${ASM_WINDOW_BEGIN:-}" ] && ASM_PREPROCESS_ARGS+=(--window-begin "$ASM_WINDOW_BEGIN")
[ -n "${ASM_WINDOW_END:-}" ] && ASM_PREPROCESS_ARGS+=(--window-end "$ASM_WINDOW_END")
bash "$ASM_WORKFLOW_DIR/scripts/asm_preprocess.sh" "$OBJ" "$AVO_REPO" \
  "${ASM_PREPROCESS_ARGS[@]}"

# Persist the authoritative capture in the workspace. Every AVO variation and
# worktree carries meta.json, so this avoids dependence on an agent shell
# inheriting GEAK_CAPTURE_DIR and prevents silent fallback to diffverify.
if [ -n "${GEAK_CAPTURE_DIR:-}" ]; then
  [ -d "$GEAK_CAPTURE_DIR" ] || {
    echo "ASM setup: GEAK_CAPTURE_DIR is not a directory: $GEAK_CAPTURE_DIR" >&2
    exit 2
  }
  python3 - "$AVO_REPO/meta.json" "$GEAK_CAPTURE_DIR" <<'PY_META'
import json, os, sys
meta_path, capture = sys.argv[1:]
meta = json.load(open(meta_path))
meta["capture"] = os.path.realpath(capture)
meta["oracle"] = "kerncap"
meta["correctness_backend"] = "generated-persistent"
json.dump(meta, open(meta_path, "w"), indent=2)
PY_META
fi
cp "$AVO_REPO/ref.co" "$EVAL_DIR/baseline/ref.co" 2>/dev/null || true

cd "$AVO_REPO"
printf '%s\n' 'ref.co' '*.co' 'candidate.co' '__pycache__/' '*.o' > .gitignore
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
git init -q
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "asm baseline (recovered kernel.s)"
mkdir -p "$EVAL_DIR/avo_state/patches"
```

Create `$EVAL_DIR/COMMANDMENT.md` immediately for the reused benchmark/verify roles:

```bash
cat > "$EVAL_DIR/COMMANDMENT.md" <<EOF
# COMMANDMENT - ASM mode
WORKSPACE is the current ASM workspace containing kernel.s, ref.co, meta.json, harness.py.
GPU CONTRACT: GPU_POOL (allocated comma list) or GPU_ID is supplied by the worker. Never hard-code a physical GPU ordinal in these commands.
CORRECTNESS CONTRACT: oracle=kerncap; backend=generated-persistent. Without validation_policy.json every captured output region is byte-exact; cosine/tolerance rules apply only when explicitly present in that policy.
SETUP: cd <workspace> && rm -f candidate.* *.co *.o
CORRECTNESS: cd <workspace> && bash "$ASM_WORKFLOW_DIR/scripts/asm_verify.sh" <workspace> --correctness
BENCHMARK: cd <workspace> && bash "$ASM_WORKFLOW_DIR/scripts/asm_verify.sh" <workspace> --benchmark
FULL_BENCHMARK: cd <workspace> && bash "$ASM_WORKFLOW_DIR/scripts/asm_verify.sh" <workspace> --full-benchmark
PROFILE: cd <workspace> && bash "$ASM_WORKFLOW_DIR/scripts/asm_profile.sh" <workspace>
PARSE: OK/FAIL for correctness; parse GEAK_RESULT_LATENCY_MS, GEAK_RESULT_SPEEDUP, and GEAK_RESULT_*_COV from stdout.
MODIFIABLE FILES: kernel.s (or window.s if windowed). Never edit ref.co, meta.json, or harness.py.
EOF
```

Return the usual fresh setup JSON with `avo_repo="$AVO_REPO"`,
`baseline_dir="$EVAL_DIR/baseline"`, `source_files=["kernel.s"]`, and notes saying
this is ASM mode. Do NOT also run the normal copy-directory setup.

### IR mode (FIRST, unless resuming)

If `EVAL_DIR_OVERRIDE` is set and `"$EVAL_DIR_OVERRIDE/avo_state/lineage.json"` exists,
this is a RESUME: skip this fresh IR setup and use the Resume detection block below.

If `TARGET_LANGUAGE` is `hip` or `triton`, this is an **IR-level AVO run**. The
input `KERNEL_PATH_ORIG` is a HIP/Triton source file (for Triton, a module that
exposes `warmup()` and `make_case()`), and the editable artifact is generated IR
text (`kernel.ir`). In fresh setup, follow the IR seam instead of the normal
copy-directory setup:

```bash
KERNEL_NAME="$KERNEL_NAME_HINT"
TS=$(date +%Y%m%d_%H%M%S)_$$_${RANDOM}
if [ -n "$EVAL_DIR_OVERRIDE" ]; then
  EVAL_DIR="$EVAL_DIR_OVERRIDE"
else
  EVAL_DIR="$EXP_ROOT/avo_${KERNEL_NAME}_${TS}/${KERNEL_NAME}"
fi
mkdir -p "$EVAL_DIR/baseline"
echo "$KERNEL_PATH_ORIG" > "$EVAL_DIR/original_kernel_path.txt"

# Original integrity manifest: supports either a single source file or a dir.
if [ -f "$KERNEL_PATH_ORIG" ]; then
  sha256sum "$KERNEL_PATH_ORIG" | sed "s#  $KERNEL_PATH_ORIG#  $(basename \"$KERNEL_PATH_ORIG\")#" \
    > "$EVAL_DIR/orig_manifest.sha256"
else
  find "$KERNEL_PATH_ORIG" -type f -not -path '*/.git/*' -exec sha256sum {} \; \
    | sed "s#$KERNEL_PATH_ORIG/##" | sort > "$EVAL_DIR/orig_manifest.sha256"
fi

AVO_REPO="$EVAL_DIR/avo_repo"
rm -rf "$AVO_REPO"; mkdir -p "$AVO_REPO"
export IR_WORKFLOW_DIR="${IR_WORKFLOW_DIR:-$SKILL_DIR/../ir-workflow}"
export IR_LANG="${IR_LANG:-$TARGET_LANGUAGE}"
export IR_ARCH="${IR_ARCH:-gfx942}"
IR_PREPROCESS_ARGS=(--lang "$IR_LANG" --arch "$IR_ARCH")
[ -n "${IR_SHAPES:-}" ] && IR_PREPROCESS_ARGS+=(--shapes "$IR_SHAPES")
bash "$IR_WORKFLOW_DIR/scripts/ir_preprocess.sh" "$KERNEL_PATH_ORIG" "$AVO_REPO" \
  "${IR_PREPROCESS_ARGS[@]}"
# IR validation is always generated-persistent and therefore requires one real
# dispatch capture. Resolve it once during setup and persist immutable evidence.
CAPTURE_DEST="$EVAL_DIR/avo_state/captures/capture"
CAPTURE_LOCK="$SKILL_DIR/../workflows/scripts/gpu_lock.sh"
case "${IR_CAPTURE_MODE:-}" in
  existing)
    [ -d "${IR_CAPTURE_PATH:-}" ] || {
      echo "IR setup: existing capture is unavailable: ${IR_CAPTURE_PATH:-unset}" >&2
      exit 2
    }
    bash "$CAPTURE_LOCK" "${GPU_ID:-0}" \
      bash "$IR_WORKFLOW_DIR/scripts/ir_prepare_capture.sh" \
        "$AVO_REPO" existing "$IR_CAPTURE_PATH" "$CAPTURE_DEST"
    ;;
  kerncap)
    [ -n "${IR_CAPTURE_KERNEL:-}" ] || {
      echo "IR setup: kerncap provider requires kernel_name" >&2; exit 2;
    }
    mapfile -t IR_CAPTURE_CMD < <(python3 -c '
import json, shlex, sys
value=json.loads(sys.argv[1])
if isinstance(value, str): value=shlex.split(value)
if not isinstance(value, list) or not value: raise SystemExit(2)
print(*value, sep="\n")
' "${IR_CAPTURE_COMMAND_JSON:-[]}")
    bash "$CAPTURE_LOCK" "${GPU_ID:-0}" \
      bash "$IR_WORKFLOW_DIR/scripts/ir_prepare_capture.sh" \
        "$AVO_REPO" kerncap "$CAPTURE_DEST" "$IR_CAPTURE_KERNEL" \
        --dispatch "${IR_CAPTURE_DISPATCH:--1}" \
        --timeout "${IR_CAPTURE_TIMEOUT:-300}" \
        -- "${IR_CAPTURE_CMD[@]}"
    ;;
  *)
    echo "IR setup: capture_provider or capture_dir is required" >&2
    exit 2
    ;;
esac
GEAK_CAPTURE_DIR="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["capture"])' \
  "$AVO_REPO/meta.json")"
export GEAK_CAPTURE_DIR GEAK_PERF_BACKEND=generated-persistent
# Build and replay the baseline before workers start. Unsupported captures,
# ABI drift, generated harness build failures, and output mismatches fail here.
bash "$IR_WORKFLOW_DIR/scripts/ir_verify.sh" "$AVO_REPO" --correctness
cp "$AVO_REPO/ref.co" "$EVAL_DIR/baseline/ref.co" 2>/dev/null || true
cp "$AVO_REPO/ref.ir" "$EVAL_DIR/baseline/ref.ir" 2>/dev/null || true
cp "$AVO_REPO/ref.s" "$EVAL_DIR/baseline/ref.s" 2>/dev/null || true

cd "$AVO_REPO"
# Keep every correctness/performance byproduct out of lineage git. In particular,
# generated-persistent rewrites its manifest and CMake dependency files on normal
# rebuilds; tracking that tree makes the pre-commit clean-worktree guard reject an
# otherwise valid candidate before the lineage engineer can commit it.
printf '%s\n' \
  'ref.co' '*.co' 'candidate.*' '__pycache__/' '*.o' '.torch_ext/' \
  'generated_persistent_harness/' \
  'realcap_*_result.json' '.avo_screening_*.json' \
  '.geak_formal_session_id' '.generated_persistent_harness.lock' \
  '.realcap_session.lock' > .gitignore
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
git init -q
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "ir baseline (lang=$TARGET_LANGUAGE)"
mkdir -p "$EVAL_DIR/avo_state/patches"
```

Create `$EVAL_DIR/COMMANDMENT.md` immediately for the reused benchmark/verify roles:

```bash
cat > "$EVAL_DIR/COMMANDMENT.md" <<EOF
# COMMANDMENT - IR mode (${TARGET_LANGUAGE})
WORKSPACE is the current IR workspace containing kernel.ir, kernel.src, meta.json, harness.py.
GPU CONTRACT: GPU_POOL (allocated comma list) or GPU_ID is supplied by the worker. Never hard-code a physical GPU ordinal in these commands.
CORRECTNESS CONTRACT: oracle=kerncap; backend=generated-persistent. Without validation_policy.json every captured output region is byte-exact; cosine/tolerance rules apply only when explicitly present in that policy.
SETUP: cd <workspace> && rm -f candidate.* *.co *.o
CORRECTNESS: cd <workspace> && bash "$IR_WORKFLOW_DIR/scripts/ir_verify.sh" <workspace> --correctness
BENCHMARK: cd <workspace> && bash "$IR_WORKFLOW_DIR/scripts/ir_verify.sh" <workspace> --benchmark
FULL_BENCHMARK: cd <workspace> && bash "$IR_WORKFLOW_DIR/scripts/ir_verify.sh" <workspace> --full-benchmark
PROFILE: cd <workspace> && bash "$IR_WORKFLOW_DIR/scripts/ir_profile.sh" <workspace>
PARSE: OK/FAIL for correctness; parse GEAK_RESULT_LATENCY_MS, GEAK_RESULT_SPEEDUP, and GEAK_RESULT_*_COV from stdout.
MODIFIABLE FILES: kernel.ir only, unless updating explanatory notes. Never edit kernel.src, meta.json, ref.* or harness.py.
EOF
```

Return the usual fresh setup JSON with `avo_repo="$AVO_REPO"`,
`baseline_dir="$EVAL_DIR/baseline"`, `source_files=["kernel.ir"]`, and notes saying
this is IR mode (`target_language`, `IR_WORKFLOW_DIR`). Do NOT also run the normal
copy-directory setup.

### Resume detection (FIRST)

If `EVAL_DIR_OVERRIDE` is set AND `"$EVAL_DIR_OVERRIDE/avo_state/lineage.json"`
exists, this is a **RESUME**: do NOT re-clone, re-snapshot, or re-seed. Just
reconstruct the state and return it (Benchmark/Seed are skipped by the workflow):

```bash
EVAL_DIR="$EVAL_DIR_OVERRIDE"
AVO_REPO="$EVAL_DIR/avo_repo"
RESUME_JSON=$(python3 "$SKILL_DIR/scripts/avo_lineage.py" resume --state "$EVAL_DIR/avo_state")
git -C "$AVO_REPO" worktree prune 2>/dev/null || true   # clear any stale worktrees from the interrupted run
# Strict IR resumes reuse the exact durable setup-time capture: never invoke the provider again
# and never silently fall back to source/differential timing.
if python3 - "$AVO_REPO/meta.json" <<'PY_IR_MODE'
import json, sys
m = json.load(open(sys.argv[1]))
raise SystemExit(0 if str(m.get("lang", "")).lower() in {"hip", "triton"} else 1)
PY_IR_MODE
then
  GEAK_CAPTURE_DIR=$(python3 - "$AVO_REPO/meta.json" <<'PY_CAPTURE'
import json, os, sys
m = json.load(open(sys.argv[1]))
capture = m.get("capture", "")
if str(m.get("oracle", "")).lower() != "kerncap" or not capture:
    raise SystemExit("strict IR resume metadata lacks oracle=kerncap and capture")
capture = os.path.realpath(capture)
if not os.path.isdir(capture):
    raise SystemExit(f"strict IR resume capture is unavailable: {capture}")
print(capture)
PY_CAPTURE
  ) || exit 2
  export GEAK_CAPTURE_DIR GEAK_PERF_BACKEND=generated-persistent
fi
# prior_steps: MUST come from RESUME_JSON (field "prior_steps") — it is the max
# variation_NNNN / escalate_NNNN step index under EVAL_DIR. Do NOT use
# `ls variation_* | wc -l` (team mode has multiple dirs per step → wrong jump).
```

Return the merged JSON (parse `RESUME_JSON` and pass through all fields including
**`prior_steps`** and **`best_ir`** from it, plus `eval_dir`/`avo_repo`/`kernel_name`),
with `"resume": true`. Apart from the fail-closed IR capture check above, do
nothing else.

If not resuming, proceed to a fresh setup:

Build the eval dir + frozen baseline + integrity manifest **exactly as in
`PHASE=setup` below**, then additionally create the **`avo_repo`** — an isolated
git clone that holds the lineage (its own `.git`, tags `avo-v{N}`); the original
is never touched:

```bash
# (run the PHASE=setup steps 1-3 first to create $EVAL_DIR, manifest, baseline)
AVO_REPO="$EVAL_DIR/avo_repo"
rm -rf "$AVO_REPO"; mkdir -p "$AVO_REPO"
( cd "$KERNEL_PATH_ORIG" && tar \
    --exclude='./.git' --exclude='*/.git' \
    --exclude='./build' --exclude='*/build' \
    --exclude='./__pycache__' --exclude='*/__pycache__' \
    --exclude='./.torch_ext' --exclude='*/.torch_ext' \
    --exclude='./.rocprofv3' --exclude='*/.rocprofv3' \
    --exclude='./_cand_overlay' --exclude='*/_cand_overlay' \
    --exclude='./_priv_overlay' --exclude='*/_priv_overlay' \
    --exclude='./_priv_overlay*' --exclude='*/_priv_overlay*' \
    --exclude='*.pyc' --exclude='*.so' --exclude='*.o' \
    -cf - . ) | ( cd "$AVO_REPO" && tar -xf - )
cd "$AVO_REPO"
# Keep harness-generated overlays and the append-only run log out of lineage
# git. unittest rebuilds `_cand_overlay/` from kernel_src/ and appends
# `_runs.jsonl` on every measurement; tracking either makes the pre-commit
# clean-worktree guard reject an otherwise valid candidate.
printf '%s\n' \
  'build/' '__pycache__/' '*.pyc' '*.so' '*.o' \
  '.torch_ext/' '.rocprofv3/' \
  '_cand_overlay/' '_priv_overlay/' '_priv_overlay*/' \
  '_runs.jsonl' > .gitignore
git init -q && git add -A && git commit -q -m "avo: import baseline"
mkdir -p "$EVAL_DIR/avo_state/patches"
```

Do NOT build or run anything (Benchmark/Seed phases handle that). Return JSON
(fresh run → `resume: false`):
```json
{
  "resume": false,
  "eval_dir": "<EVAL_DIR>",
  "avo_repo": "<EVAL_DIR>/avo_repo",
  "baseline_dir": "<EVAL_DIR>/baseline",
  "kernel_name": "<KERNEL_NAME>",
  "source_files": ["<relative source files>"],
  "notes": "<anything notable>"
}
```

On **resume** return instead (merge **all** fields from `RESUME_JSON` — including
**`prior_steps`**, the last completed Evolve step index — plus `eval_dir` / `avo_repo` / `kernel_name`):
```json
{
  "resume": true, "eval_dir": "<EVAL_DIR>", "avo_repo": "<EVAL_DIR>/avo_repo",
  "kernel_name": "<KERNEL_NAME>", "prior_steps": 60,
  "best_version": "avo-v<N>", "cumulative": 0.0, "commits": 0,
  "committed": [{"id":"v0","ref":"avo-v0","speedup":1.0,"strategy":"baseline"}],
  "baseline_per_case": [{"name":"...","speedup":1.0,"baseline_ms":0.0,"optimized_ms":0.0}],
  "baseline_geomean_ms": 0.0,
  "failed": [], "insights": [], "direction": {"name":"","expected":"","prompt":""},
  "notes": "resumed from existing lineage"
}
```
(`prior_steps` above is an example; copy the numeric value from `RESUME_JSON` exactly.)

---

## PHASE=setup

Inputs: `KERNEL_PATH_ORIG`, `EXP_ROOT`, `EVAL_DIR_OVERRIDE`, `KERNEL_NAME_HINT`,
`TASK`, `SKILL_DIR`.

1. Derive identifiers and create the eval dir:

```bash
KERNEL_NAME="$KERNEL_NAME_HINT"
TS=$(date +%Y%m%d_%H%M%S)
if [ -n "$EVAL_DIR_OVERRIDE" ]; then
  EVAL_DIR="$EVAL_DIR_OVERRIDE"
else
  EVAL_DIR="$EXP_ROOT/avo_${KERNEL_NAME}_${TS}/${KERNEL_NAME}"
fi
mkdir -p "$EVAL_DIR/baseline"
echo "$KERNEL_PATH_ORIG" > "$EVAL_DIR/original_kernel_path.txt"
```

2. **Integrity manifest** of the original (re-checked at validate) so we can prove
   it was never modified:

```bash
find "$KERNEL_PATH_ORIG" -type f -not -path '*/.git/*' -exec sha256sum {} \; \
  | sed "s#$KERNEL_PATH_ORIG/##" | sort > "$EVAL_DIR/orig_manifest.sha256"
```

3. **Frozen baseline snapshot** (a copy, never modified — used by validate to
   rebuild the true baseline, and by the runner as the evolution seed source):

```bash
cp -r "$KERNEL_PATH_ORIG"/. "$EVAL_DIR/baseline"/ && rm -rf "$EVAL_DIR/baseline/.git"
```

4. List the kernel source files (for the report). Do NOT build or run anything.

Return JSON. The following is the **ASM paired** contract. For non-ASM repeated
benchmarks set `protocol="repeated"` and do not invent paired CI/formal fields; use
the legacy correctness/repeat evidence and canonical `validation_status`.

```json
{
  "eval_dir": "<EVAL_DIR>",
  "baseline_dir": "<EVAL_DIR>/baseline",
  "kernel_name": "<KERNEL_NAME>",
  "source_files": ["<relative source files>"],
  "notes": "<anything notable>"
}
```

---

## PHASE=validate

Inputs: `KERNEL_PATH_ORIG`, `EVAL_DIR`, `KERNEL_NAME`, `KERNEL_WORKFLOW_DIR`,
`ASM_WORKFLOW_DIR`, `IR_WORKFLOW_DIR`, `TARGET_LANGUAGE`, `LANGUAGE_MODE`,
`LANGUAGE_SKILL_PATH`, `FORMAL_SESSION_ID`,
`GPU_ID`, `VERIFY_REPEATS`, `APPLY_TO_ORIGINAL`, `FINAL_PATCH`, `REPORTED_GEOMEAN`,
`COMMANDMENT`, optional `KERNEL_ENV_SCRIPT`, `GEAK_CAPTURE_DIR`, `IR_LANG`,
`IR_ARCH`, `IR_SHAPES`, `ASM_PACK`, `ASM_ARCH`, `ASM_SHAPES`,
`ASM_WINDOW_KERNEL`, `ASM_WINDOW_BEGIN`, and `ASM_WINDOW_END`.

Read a non-empty `LANGUAGE_SKILL_PATH` before validation. Treat supplied input
values as the shell variables of the same name; in particular, when
`GEAK_CAPTURE_DIR` is non-empty, export it before preprocess/verify so a fresh
validation workspace cannot silently downgrade from realcap to a synthetic
oracle.

Optional pre-commit inputs: `AVO_REPO`, `PARENT_VERSION`, `CANDIDATE_PATCH`,
`VALIDATION_LABEL`, and `REPORT_PATH`. If `FINAL_PATCH` is empty but
`CANDIDATE_PATCH` is set, synthesize a cumulative patch first:

```bash
LABEL="${VALIDATION_LABEL:-precommit}"
PRE_DIR="$EVAL_DIR/director_precommit/$LABEL"
WORKTREE="$PRE_DIR/candidate_repo"
mkdir -p "$PRE_DIR"
rm -rf "$WORKTREE"
git -C "$AVO_REPO" worktree add --detach "$WORKTREE" "$PARENT_VERSION"
if ! git -C "$WORKTREE" apply --3way "$CANDIDATE_PATCH"; then
  git -C "$WORKTREE" apply "$CANDIDATE_PATCH"
fi
git -C "$WORKTREE" add -N . 2>/dev/null || true
FINAL_PATCH="$PRE_DIR/final_patch.diff"
git -C "$WORKTREE" diff --binary avo-v0 -- . > "$FINAL_PATCH"
git -C "$AVO_REPO" worktree remove -f "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
```

For pre-commit validation, never apply to the original even if an input accidentally
requests it; treat `APPLY_TO_ORIGINAL=false`.

**Do NOT trust the runner's reported speedup — measure it yourself**, against a
fresh workspace built from the TRUE original, using the COMMANDMENT contract.
If `KERNEL_ENV_SCRIPT` is provided, source it before every correctness/benchmark/profile
command and preserve it when adapting COMMANDMENT commands to the validation workspace.

### ASM validation mode

Run this branch before IR or normal source validation when `LANGUAGE_MODE=asm`,
`TARGET_LANGUAGE=asm`, or `$EVAL_DIR/avo_repo` contains both `kernel.s` and
`ref.co`. Reconstruct from the true original object; never copy the evolved
workspace as the baseline:

```bash
VALIDATION_WORKSPACE="$EVAL_DIR/validation_workspace"
if [ -n "${VALIDATION_LABEL:-}" ]; then VALIDATION_WORKSPACE="$EVAL_DIR/validation_workspace_$VALIDATION_LABEL"; fi
rm -rf "$VALIDATION_WORKSPACE"; mkdir -p "$VALIDATION_WORKSPACE"

OBJ=$(cat "$EVAL_DIR/original_kernel_path.txt")
if [ -d "$OBJ" ]; then
  OBJ=$(find "$OBJ" -maxdepth 2 -type f \( -name '*.co' -o -name '*.hsaco' -o -name '*.s' \) | sort | head -1)
fi
[ -n "$OBJ" ] && [ -f "$OBJ" ] || { echo "ASM validation: original object missing" >&2; exit 2; }

ASM_WORKFLOW_DIR="${ASM_WORKFLOW_DIR:-$SKILL_DIR/../asm-workflow}"
ASM_PREPROCESS_ARGS=()
[ -n "${ASM_PACK:-}" ] && ASM_PREPROCESS_ARGS+=(--pack "$ASM_PACK")
[ -n "${ASM_ARCH:-}" ] && ASM_PREPROCESS_ARGS+=(--arch "$ASM_ARCH")
[ -n "${ASM_SHAPES:-}" ] && ASM_PREPROCESS_ARGS+=(--shapes "$ASM_SHAPES")
[ -n "${ASM_WINDOW_KERNEL:-}" ] && ASM_PREPROCESS_ARGS+=(--window-kernel "$ASM_WINDOW_KERNEL")
[ -n "${ASM_WINDOW_BEGIN:-}" ] && ASM_PREPROCESS_ARGS+=(--window-begin "$ASM_WINDOW_BEGIN")
[ -n "${ASM_WINDOW_END:-}" ] && ASM_PREPROCESS_ARGS+=(--window-end "$ASM_WINDOW_END")
bash "$ASM_WORKFLOW_DIR/scripts/asm_preprocess.sh" "$OBJ" "$VALIDATION_WORKSPACE" \
  "${ASM_PREPROCESS_ARGS[@]}"

# Preserve the authoritative capture chosen at setup even when it was stored
# only in avo_repo/meta.json rather than exported in this fresh agent shell.
python3 - "$EVAL_DIR/avo_repo/meta.json" "$VALIDATION_WORKSPACE/meta.json" <<'PY_META'
import json, os, sys
src, dst = sys.argv[1:]
s, d = json.load(open(src)), json.load(open(dst))
for key in (
    "capture", "oracle", "correctness_backend", "metrix_replay_iters", "profile_iters",
    "window", "window_info", "route", "pack", "shapes", "seeds",
    "iters", "cos", "kernel", "arch",
):
    if key in s:
        d[key] = s[key]
capture = os.environ.get("GEAK_CAPTURE_DIR")
if capture:
    d["capture"] = capture
    d["oracle"] = "kerncap"
    d["correctness_backend"] = "generated-persistent"
json.dump(d, open(dst, "w"), indent=2)
PY_META

cd "$VALIDATION_WORKSPACE"
git init -q
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "validation_asm_baseline"
git apply "$FINAL_PATCH"
if [ -n "${KERNEL_ENV_SCRIPT:-}" ]; then source "$KERNEL_ENV_SCRIPT"; fi
```

Then use the paired ASM formal command below when requested, or the COMMANDMENT
screening commands for a non-formal diagnostic. Do not execute the generic
copy-directory validation path for ASM.

### IR validation mode

Use `VALIDATION_WORKSPACE="$EVAL_DIR/validation_workspace"` normally. If
`VALIDATION_LABEL` is set, use
`VALIDATION_WORKSPACE="$EVAL_DIR/validation_workspace_$VALIDATION_LABEL"` so
pre-commit checks do not clobber the final validation workspace.

If `$EVAL_DIR/avo_repo/meta.json` exists and says `"lang": "triton"` or `"hip"`, validate
as an IR run instead of copying the original directory:

```bash
VALIDATION_WORKSPACE="$EVAL_DIR/validation_workspace"
if [ -n "${VALIDATION_LABEL:-}" ]; then VALIDATION_WORKSPACE="$EVAL_DIR/validation_workspace_$VALIDATION_LABEL"; fi
rm -rf "$VALIDATION_WORKSPACE"
mkdir -p "$VALIDATION_WORKSPACE"
ORIG=$(cat "$EVAL_DIR/original_kernel_path.txt")
IR_WORKFLOW_DIR="${IR_WORKFLOW_DIR:-$SKILL_DIR/../ir-workflow}"
IR_LANG=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("lang","auto"))' "$EVAL_DIR/avo_repo/meta.json")
IR_ARCH=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("arch","gfx942"))' "$EVAL_DIR/avo_repo/meta.json")
IR_SHAPES=$(python3 -c 'import json,sys;print(",".join(map(str,json.load(open(sys.argv[1])).get("shapes",[]))))' "$EVAL_DIR/avo_repo/meta.json")
IR_PREPROCESS_ARGS=(--lang "$IR_LANG" --arch "$IR_ARCH")
[ -n "$IR_SHAPES" ] && IR_PREPROCESS_ARGS+=(--shapes "$IR_SHAPES")
bash "$IR_WORKFLOW_DIR/scripts/ir_preprocess.sh" "$ORIG" "$VALIDATION_WORKSPACE" \
  "${IR_PREPROCESS_ARGS[@]}"

# Carry the setup-time oracle/capture and measurement contract into the fresh
# workspace. Without this, a captured IR run silently loses meta.capture.
python3 - "$EVAL_DIR/avo_repo/meta.json" "$VALIDATION_WORKSPACE/meta.json" <<'PY_META'
import json, os, sys
src, dst = sys.argv[1:]
s, d = json.load(open(src)), json.load(open(dst))
for key in (
    "capture", "oracle", "correctness_backend", "shapes", "seeds", "iters", "cos", "pack",
    "metrix_replay_iters", "profile_iters",
):
    if key in s:
        d[key] = s[key]
capture = os.environ.get("GEAK_CAPTURE_DIR")
if capture:
    d["capture"] = capture
    d["oracle"] = "kerncap"
    d["correctness_backend"] = "generated-persistent"
json.dump(d, open(dst, "w"), indent=2)
PY_META
cd "$VALIDATION_WORKSPACE"
git init -q
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "validation_ir_baseline"
git apply "$FINAL_PATCH"
if [ -n "${KERNEL_ENV_SCRIPT:-}" ]; then source "$KERNEL_ENV_SCRIPT"; fi
```

For ordinary IR screening, run:

```bash
bash "$IR_WORKFLOW_DIR/scripts/ir_verify.sh" "$VALIDATION_WORKSPACE" --correctness
bash "$IR_WORKFLOW_DIR/scripts/ir_verify.sh" "$VALIDATION_WORKSPACE" --full-benchmark
```

When `BENCHMARK_PROTOCOL=paired-abba`, the IR workspace must have an authoritative
real capture (`meta.capture` or `GEAK_CAPTURE_DIR`). Do not use the screening
commands above; run exactly one logical formal session. `FORMAL_SESSION_ID` is
immutable for that session. A retry after a transient infrastructure failure must
reuse the exact same ID and workspace; never append `_r2`, a timestamp, or any
other suffix:

```bash
GEAK_FORMAL_SESSION_ID="$FORMAL_SESSION_ID" \
GEAK_REALCAP_PAIRED_BLOCKS="${PAIRED_BLOCKS:-10}" \
bash "$IR_WORKFLOW_DIR/scripts/ir_formal_retest.sh" \
  "$GPU_ID" "$VALIDATION_WORKSPACE" "$FORMAL_SESSION_ID"
```

Use the same formal result fields and `realcap_paired_result.json` contract as
the ASM paired path below. If the capture is absent, report
`infrastructure_unhealthy`; never downgrade a requested formal IR run to source
or legacy screening evidence.

Parse `GEAK_RESULT_SPEEDUP` as `director_verified_speedup_geomean` (single IR carrier
case) and `GEAK_RESULT_LATENCY_MS` as the optimized latency; use `GEAK_RESULT_SPEEDUP_COV`
when present for the variance note. Re-check the original manifest using the file-or-dir
logic from setup, leave the original untouched unless `APPLY_TO_ORIGINAL=true`, and write
`${REPORT_PATH:-$EVAL_DIR/director_validation.json}` in the same JSON shape below.
Do NOT run the normal directory validation path for IR runs.

1. Build a clean validation workspace from the original and apply the final patch:

```bash
VALIDATION_WORKSPACE="$EVAL_DIR/validation_workspace"
if [ -n "${VALIDATION_LABEL:-}" ]; then VALIDATION_WORKSPACE="$EVAL_DIR/validation_workspace_$VALIDATION_LABEL"; fi
rm -rf "$VALIDATION_WORKSPACE"
mkdir -p "$VALIDATION_WORKSPACE"
cp -r "$KERNEL_PATH_ORIG"/. "$VALIDATION_WORKSPACE"/ && rm -rf "$VALIDATION_WORKSPACE/.git"
cd "$VALIDATION_WORKSPACE"
git init -q && git add -A && git commit -q -m "validation_baseline"
git apply "$FINAL_PATCH"
rm -rf build/ __pycache__/ *.so .torch_ext 2>/dev/null || true
if [ -n "${KERNEL_ENV_SCRIPT:-}" ]; then source "$KERNEL_ENV_SCRIPT"; fi
```

2. Run correctness from `COMMANDMENT` (substitute the kernel path =
   `$VALIDATION_WORKSPACE`). Record pass/fail.

3. Run the full benchmark under the shared GPU lock.

   **ASM realcap (`BENCHMARK_PROTOCOL=paired-abba`) is a formal hard gate.** Run one
   logical formal session because the harness performs its own fresh balanced blocks.
   If a transient infrastructure failure requires a retry, reuse the exact same
   `FORMAL_SESSION_ID` and validation workspace. Never append `_r2`, a timestamp, or
   another suffix; the runner pins the first ID to the workspace and rejects drift.
   Never outer-repeat a successful paired session and never read
   `realcap_baseline.json`:

```bash
GEAK_FORMAL_SESSION_ID="$FORMAL_SESSION_ID" \
GEAK_BENCHMARK_PROTOCOL=paired-abba \
GEAK_REALCAP_PAIRED_BLOCKS="${PAIRED_BLOCKS:-10}" \
GEAK_REALCAP_AA_BLOCKS="${GEAK_REALCAP_AA_BLOCKS:-3}" \
bash "$ASM_WORKFLOW_DIR/scripts/asm_formal_retest.sh" \
  "$GPU_ID" "$VALIDATION_WORKSPACE" "$FORMAL_SESSION_ID"
```

   Require and parse all of:

```text
GEAK_RESULT_PROTOCOL=paired_abba
GEAK_RESULT_INFRA_STATUS=healthy
GEAK_RESULT_FORMAL_STATUS
GEAK_RESULT_SPEEDUP
GEAK_RESULT_SPEEDUP_CI_LOW
GEAK_RESULT_SPEEDUP_CI_HIGH
GEAK_RESULT_POSITIVE_BLOCK_FRACTION
GEAK_RESULT_CALIBRATED_MARGIN
GEAK_RESULT_AA_CALIBRATION_PASSED
GEAK_RESULT_BIMODAL
GEAK_RESULT_REPORT
```

   Missing formal evidence, preflight/block-health failure, timeout, lock failure,
   bimodality, or A/A interval that does not cover 1 is not an accepted performance
   result. Mark it `infrastructure_unhealthy` or `unstable`.
   The deterministic orchestrator reads
   `$VALIDATION_WORKSPACE/realcap_paired_result.json` directly and replaces all
   transcribed performance/correctness fields with that report. Never point
   `benchmark_report` at another workspace or an older run.

   For non-ASM workflows, retain the legacy `VERIFY_REPEATS` median path:

```bash
bash "$KERNEL_WORKFLOW_DIR/scripts/gpu_lock.sh" "${GPU_POOL:-$GPU_ID}" <full_benchmark_command from COMMANDMENT>
```

   Compute, vs the baseline in `$EVAL_DIR/baseline` measured the same way:
   - Aggregate each case with the verifier's rule (`roles/avo_verify.md` step 3),
     separately for the candidate and for the baseline: drop outliers (MAD rule,
     k=3), take the median of the kept samples, `cv = stddev/mean` of the kept
     samples. Use the functions in `$SKILL_DIR/lib/measure.mjs` for this; do not
     re-implement them. In a Node script, `import { summarizeRepeats, aggregateCV }
     from "<SKILL_DIR>/lib/measure.mjs"`; `summarizeRepeats(<the VERIFY_REPEATS
     latencies of one case>)` returns `{median, cv, n_used, n_raw}`.
   - `optimized_ms` / `baseline_ms` = that `median` for the candidate / baseline
     samples of the case. Top-level `cv` = `aggregateCV(<per-case candidate cv>)`
     (the largest). Do not derive `cv` from per-repeat geomeans or from the
     baseline samples.
   - `director_verified_speedup_geomean = geomean(baseline_ms / optimized_ms)`
   - `director_verified_speedup_arithmetic` and per-case speedups.
   - When `SCORE_METRIC=verified_weighted`, parse the task-authoritative
     `GEAK_WEIGHTED_SPEEDUP` from every fresh repeat and return their robust
     median as `director_verified_speedup_weighted`. Do not reconstruct an
     unweighted mean and label it weighted. Require that the benchmark emitted
     the weighted metric on every accepted repeat; otherwise return
     `infrastructure_unhealthy`.

   Persist the raw repeated measurements (all repeats, not only medians) as
   `$VALIDATION_WORKSPACE/performance_report.json` and return that absolute path
   in `benchmark_report`. The orchestrator archives it before Cleanup removes
   `VALIDATION_WORKSPACE`.

4. **Validation decision**:
   - When `CALIBRATED_MARGIN_CAP` is a finite non-negative number, use
     `min(measured calibrated_margin, CALIBRATED_MARGIN_CAP)` as the effective
     margin for the `accepted` / `no_significant_gain` decision, but still
     report the original measured `calibrated_margin` for audit. State both
     values in `notes`. This preserves the independent Director replay while
     allowing a campaign-specific micro-gain policy.
   - `accepted`: correctness pass, infrastructure healthy, A/A calibration passed,
     no bimodality, formal status accepted, positive block fraction >= 0.8, and paired
     CI lower bound clears `1 + calibrated_margin`.
   - `no_significant_gain`: correctness pass but the CI crosses the calibrated margin.
   - `unstable`: bimodal/high-noise/mixed-platform result.
   - `reproduction_mismatch`: final CI upper bound cannot reproduce the committed
     cumulative speedup within the orchestrator's configured tolerance.
   - `infrastructure_unhealthy`: GPU busy, other compute PID, orphan, timeout, lock or
     device-binding failure.
   - `correctness_failed` / `artifact_mismatch`: corresponding hard failures.

   The old ±10% comparison with `REPORTED_GEOMEAN` is diagnostic only and MUST NOT
   turn a statistically invalid result into `accepted`.

5. **Integrity re-check** — recompute the original's manifest and diff vs
   `$EVAL_DIR/orig_manifest.sha256`; it MUST be empty. If not, set
   `validation_status="flagged"` and report loudly.

6. Always leave the original UNTOUCHED during validation. The orchestrator passes
   `APPLY_TO_ORIGINAL=false`, deterministically checks the structured formal evidence,
   and invokes the separate `PHASE=apply` only after that gate passes.

7. Write `${REPORT_PATH:-$EVAL_DIR/director_validation.json}` with the fields below
   and return it. Create the parent directory first when `REPORT_PATH` is set.
   Write this file from the script that computed the numbers (e.g. `json.dump` of the
   full `per_case` list), also for failed or rejected results. In source mode the
   orchestrator reads this file, not your reply; never shorten or retype its fields.

   Status vocabulary is strict: when `BENCHMARK_PROTOCOL=repeated` (source mode),
   return `"formal_status": "not_applicable"` exactly. Never emit `"n/a"`, `"na"`,
   or `"not applicable"`. For `paired-abba`, `formal_status` must use the formal
   result status (`accepted`, `no_significant_gain`, `unstable`, or a hard failure).

Return JSON:
```json
{
  "kernel_name": "<KERNEL_NAME>",
  "director_verified_speedup_geomean": 0.0,
  "director_verified_speedup_weighted": 0.0,
  "director_verified_speedup_arithmetic": 0.0,
  "reported_speedup_geomean": 0.0,
  "validation_status": "accepted",
  "correctness": "pass",
  "per_case": [{"name": "...", "baseline_ms": 0.0, "optimized_ms": 0.0, "speedup": 0.0}],
  "applied_to_original": "false",
  "final_patch": "<FINAL_PATCH>",
  "cv": 0.0,
  "protocol": "paired_abba",
  "formal_status": "accepted",
  "infrastructure_status": "healthy",
  "screening_only": false,
  "ci_lower": 0.0,
  "ci_upper": 0.0,
  "positive_block_fraction": 0.0,
  "calibrated_margin": 0.0,
  "aa_calibration_passed": false,
  "bimodal": false,
  "benchmark_report": "<realcap_paired_result.json>",
  "notes": "<integrity result, variance, any flags>"
}
```

---

## PHASE=finalize_language

This phase runs only after deterministic final validation accepted an ASM or IR
result. Inputs: `VALIDATION_WORKSPACE`, `OUTPUT_DIR`, `LANGUAGE_MODE`,
`ASM_WORKFLOW_DIR`, `IR_WORKFLOW_DIR`, optional `KERNEL_ENV_SCRIPT`, and
`LANGUAGE_SKILL_PATH`.

1. Require `VALIDATION_WORKSPACE` to exist. It is the fresh, true-baseline
   workspace that already has `FINAL_PATCH` applied and passed final validation.
   Never finalize directly from a stale variation workspace.
2. Read `LANGUAGE_SKILL_PATH`. If it is unavailable, fail closed.
3. Remove and recreate `OUTPUT_DIR`; source `KERNEL_ENV_SCRIPT` when supplied.
4. For ASM:

   ```bash
   bash "$ASM_WORKFLOW_DIR/scripts/asm_finalize.sh" \
     "$VALIDATION_WORKSPACE" "$OUTPUT_DIR" optimized
   test -s "$OUTPUT_DIR/optimized.s"
   test -s "$OUTPUT_DIR/optimized.co"
   ```

5. For HIP/Triton IR:

   ```bash
   bash "$IR_WORKFLOW_DIR/scripts/ir_finalize.sh" \
     "$VALIDATION_WORKSPACE" "$OUTPUT_DIR" optimized
   test -s "$OUTPUT_DIR/optimized.s"
   test -s "$OUTPUT_DIR/optimized.co"
   python3 - "$OUTPUT_DIR" <<'PY_ARTIFACTS'
   import pathlib, sys
   out = pathlib.Path(sys.argv[1])
   assert any(out.glob("optimized.ll")) or any(out.glob("optimized.ttgir"))
   assert any(out.glob("optimized.ir.diff"))
   PY_ARTIFACTS
   ```

6. Enumerate regular files directly under `OUTPUT_DIR` with Python and return
   absolute paths. `finalized=true` only when the command succeeded and all
   required language artifacts are non-empty.

Return JSON:

```json
{
  "finalized": true,
  "output_dir": "<OUTPUT_DIR>",
  "artifacts": ["<absolute artifact path>"],
  "notes": "oracle-verified ASM or IR bundle"
}
```

---

## PHASE=apply

This phase is called only after the orchestrator's deterministic final gate accepted
the Director result.

1. Verify `FINAL_PATCH` exists and the current original still matches the integrity
   manifest captured before optimization.
2. Apply exactly `FINAL_PATCH` to `KERNEL_PATH_ORIG`; do not benchmark and do not
   reinterpret the validation result.
3. Recompute the manifest. If the pre-apply original drifted or the patch does not
   apply cleanly, leave/restore the original unchanged and return `applied=false`.

Return JSON:

```json
{"applied": false, "notes": "<integrity and apply result>"}
```

---

## PHASE=cleanup

Inputs: `AVO_REPO`, `EVAL_DIR`.

Reclaim disk and remove leaked git worktrees at the end of the run, **keeping all
durable artifacts** (lineage, patches, reports). This is the deterministic backstop
in case a per-step agent didn't remove its own worktree.

```bash
# 1) Prune git worktree registrations + remove the heavy ephemeral checkouts.
git -C "$AVO_REPO" worktree prune 2>/dev/null || true
for w in "$EVAL_DIR"/variation_*/workspace "$EVAL_DIR"/variation_*_*/workspace \
         "$EVAL_DIR"/escalate_*_*/workspace "$EVAL_DIR"/*/verify/workspace \
         "$EVAL_DIR"/*_integrate/ws "$EVAL_DIR"/*_integrate/verify/workspace \
         "$EVAL_DIR"/validation_workspace*; do
  [ -e "$w" ] && rm -rf "$w"
done
git -C "$AVO_REPO" worktree prune 2>/dev/null || true

# 2) Drop build caches left in step dirs (large, regenerable).
find "$EVAL_DIR" -type d \( -name build -o -name __pycache__ -o -name .torch_ext \) \
  -prune -exec rm -rf {} + 2>/dev/null || true
```

**KEEP (never delete):** `avo_repo/` (the lineage `.git` + tags), `avo_state/`
(`lineage.json`, `attempts.jsonl`, `evolution_log/`, `supervisor_log.jsonl`,
`patches/v{N}.patch`, `director_precommit/`, and
`final_validation/realcap_paired_result.json` + its archived candidate),
`baseline/`, every step's `candidate.patch` + worker
`worker_result`/`report`, `final_patch.diff`, `trajectory.json`,
`tech_lead_report.md`, `director_validation.json`, and `final_artifacts/`.

Report what was reclaimed. Return JSON:
```json
{ "cleaned": true, "worktrees_pruned": 0, "notes": "<dirs removed / kept>" }
```
